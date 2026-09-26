//! Installed native embedding launch ABI and its passive selected evidence.
use super::*;
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
};
use ferrum_interfaces::vnext::{ElementType, HadamardApplication, HadamardSigns};
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(in crate::backend::cuda::vnext_ops) const F16_ENTRY: &str = "vnext_gguf_embedding_f16";
pub(in crate::backend::cuda::vnext_ops) const F32_ENTRY: &str = "vnext_gguf_embedding_f32";

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct LookupPlan {
    pub entry: &'static str,
    pub config: LaunchConfig,
    pub parameters: [u32; 6],
    pub elements: u32,
}

pub(in crate::backend::cuda::vnext_ops) fn lookup_plan(
    part: &weights::MatrixPart,
    count: u32,
    activation: ElementType,
) -> Result<LookupPlan, CudaDeviceRuntimeError> {
    let elements = embedding_elements(part, count)?;
    let entry = match activation {
        ElementType::F16 => F16_ENTRY,
        ElementType::F32 => F32_ENTRY,
        _ => {
            return Err(CudaDeviceRuntimeError::contract(
                "unsupported embedding dtype",
            ))
        }
    };
    let [format, values, bytes] = part.format.parameters();
    Ok(LookupPlan {
        entry,
        config: LaunchConfig::for_num_elems(elements),
        parameters: [count, part.columns, part.rows, format, values, bytes],
        elements,
    })
}

pub(in crate::backend::cuda::vnext_ops) fn selected(
    part: &weights::MatrixPart,
    counts: impl IntoIterator<Item = u64>,
    tokens: u64,
    activation: ElementType,
    transform_scratch_bytes: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    embedding_elements(part, 1).ok()?;
    let limit = u64::from(u32::MAX / part.columns).min(super::super::MAXIMUM_TOKENS_PER_LAUNCH);
    let mut seen = 0_u64;
    for count in counts {
        if count == 0 {
            return None;
        }
        seen = seen.checked_add(count)?;
        if seen > tokens {
            return None;
        }
        let mut remaining = count;
        while remaining > 0 {
            let count = remaining.min(limit) as u32;
            let transformed = part.transform.is_some();
            let plan = lookup_plan(
                part,
                count,
                if transformed {
                    ElementType::F32
                } else {
                    activation
                },
            )
            .ok()?;
            let scratch = if transformed {
                let required = u64::from(plan.elements).checked_mul(4)?;
                if transform_scratch_bytes < required {
                    return None;
                }
                transform_scratch_bytes
            } else {
                0
            };
            let fixed = plan.parameters.map(u64::from);
            push(
                &mut builder,
                plan.entry,
                part.format.parameters(),
                plan.config,
                u64::from(plan.elements),
                1,
                scratch,
                &fixed,
            )?;
            if let Some(spec) = &part.transform {
                if !matches!(spec.application, HadamardApplication::AfterEmbeddingLookup)
                    || matches!(spec.signs, HadamardSigns::Explicit(_))
                        != part.signs_region.is_some()
                {
                    return None;
                }
                let transform = hadamard::launch_descriptor(
                    count,
                    part.columns,
                    ElementType::F32,
                    activation,
                    spec,
                )
                .ok()?;
                let fixed = transform.parameters.map(u64::from);
                push(
                    &mut builder,
                    transform.entry,
                    [
                        spec.block_size.get(),
                        u32::from(part.signs_region.is_some()),
                        1,
                    ],
                    transform.config,
                    u64::from(plan.elements),
                    u64::from(spec.block_size.get().ilog2()) + 1,
                    scratch,
                    &fixed,
                )?;
            } else if part.signs_region.is_some() {
                return None;
            }
            remaining -= u64::from(count);
        }
    }
    if seen != tokens {
        return None;
    }
    builder.finish().ok()
}

#[allow(clippy::too_many_arguments)]
fn push(
    builder: &mut SelectedCommandCostBuilderV1,
    entry: &'static str,
    layout: [u32; 3],
    config: LaunchConfig,
    logical: u64,
    inner: u64,
    scratch: u64,
    fixed: &[u64],
) -> Option<()> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let mut identity = Sha256::new();
    identity.update(b"cuda.native_embedding.layout.v1");
    for value in layout {
        identity.update(value.to_le_bytes());
    }
    let algorithm = SelectedAlgorithmClassV1::new(
        entry,
        1,
        *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into()),
        identity.finalize().into(),
    )
    .ok()?;
    let padded = if entry == F16_ENTRY || entry == F32_ENTRY {
        u64::from(config.grid_dim.0).checked_mul(u64::from(config.block_dim.0))?
    } else {
        logical
    };
    builder
        .kernel_with_replay_geometry(
            algorithm,
            KernelNumericWorkV1 {
                logical_units: logical,
                padded_units: padded,
                inner_units_per_logical_unit: inner,
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
                dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
                fixed_parameters: fixed,
            },
        )
        .ok()
}

#[cfg(test)]
mod tests;
