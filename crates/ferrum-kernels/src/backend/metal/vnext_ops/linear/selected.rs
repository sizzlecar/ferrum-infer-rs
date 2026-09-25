//! Passive evidence for the real immutable PSO selector. No shader, threshold,
//! dispatch order, workspace authority or numeric policy is selected here.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1,
    SelectedCommandCostEvidenceV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Numeric description of the actual dispatch_grid ABI.
/// Native/transformed paths outside this first producer return None.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Grid {
    pub groups: [u32; 3],
    pub threads: [u32; 3],
    pub threadgroup_memory: Option<u64>,
    pub padded_outputs: u64,
}

pub(super) fn grid(params: LinearParams, kind: LinearDispatchKind) -> Option<Grid> {
    let m = u64::from(params.rows);
    let n = u64::from(params.out_features);
    let (groups, threads, memory, padded) = match kind {
        LinearDispatchKind::CooperativeGemv => (
            [n.div_ceil(4), m, 1],
            [32, 2, 1],
            None,
            n.div_ceil(4).checked_mul(4)?.checked_mul(m)?,
        ),
        LinearDispatchKind::SharedWeightGemv => (
            [n.div_ceil(4), 1, 1],
            [32, 2, 1],
            None,
            n.div_ceil(4).checked_mul(4)?.checked_mul(m)?,
        ),
        LinearDispatchKind::NarrowDenseGemv => (
            [n, m, 1],
            [NARROW_DENSE_THREADS as u32, 1, 1],
            None,
            m.checked_mul(n)?,
        ),
        LinearDispatchKind::TiledGemm | LinearDispatchKind::TiledGemmM8 => {
            let tile = if kind == LinearDispatchKind::TiledGemmM8 {
                8
            } else {
                32
            };
            (
                [m.div_ceil(tile), n.div_ceil(64), 1],
                [128, 1, 1],
                Some(8192),
                m.div_ceil(tile)
                    .checked_mul(tile)?
                    .checked_mul(n.div_ceil(64).checked_mul(64)?)?,
            )
        }
        _ => return None,
    };
    Some(Grid {
        groups: [
            u32::try_from(groups[0]).ok()?,
            u32::try_from(groups[1]).ok()?,
            u32::try_from(groups[2]).ok()?,
        ],
        threads,
        threadgroup_memory: memory,
        padded_outputs: padded,
    })
}

fn class(
    entry: &'static str,
    launch: LinearLaunch,
    geometry: Grid,
    staged: bool,
) -> Option<SelectedAlgorithmClassV1> {
    // Hash source once, never hash the shader text per planner candidate.
    static NUMERICS: OnceLock<[u8; 32]> = OnceLock::new();
    let numerical = *NUMERICS.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"ferrum.metal.linear.actual-source.default-compile-options.v1");
        hash.update(FINGERPRINT_SOURCE.as_bytes());
        hash.finalize().into()
    });
    let mut layout = Sha256::new();
    layout.update(b"ferrum.metal.linear.physical-layout.v1");
    // Extents M/N/K live in numeric work. Stride/column layout is the exact
    // immutable ABI, not an owner ID or an allocation address.
    layout.update(launch.params.output_stride.to_le_bytes());
    layout.update(launch.params.output_column_offset.to_le_bytes());
    layout.update(launch.activation_type.size_bytes().to_le_bytes());
    layout.update([u8::from(staged)]);
    for value in geometry.threads {
        layout.update(value.to_le_bytes());
    }
    layout.update(geometry.threadgroup_memory.unwrap_or(0).to_le_bytes());
    SelectedAlgorithmClassV1::new(entry, 1, numerical, layout.finalize().into()).ok()
}

/// Names are bound to actual PSO references, not inferred from M/N/K. The
/// references only identify this process's immutable catalog; pointer values
/// never enter the stable signature. Optional/unmapped PSOs stay unavailable.
fn entry(pipelines: &MetalLinearPipelines, actual: &ComputePipelineState) -> Option<&'static str> {
    macro_rules! named {
        ($field:expr, $name:expr) => {
            if std::ptr::eq(actual, $field) {
                return Some($name);
            }
        };
    }
    named!(&pipelines.dense, LINEAR_DENSE_KERNEL);
    named!(&pipelines.dense_f32, LINEAR_DENSE_F32_KERNEL);
    named!(&pipelines.q8_0, LINEAR_Q8_0_KERNEL);
    named!(&pipelines.q8_0_f32, LINEAR_Q8_0_F32_KERNEL);
    named!(
        &pipelines.q4_k_gemv,
        crate::backend::metal::q4_k_gemv_v2::F16_BATCHED_KERNEL_NAME
    );
    named!(
        &pipelines.q4_k_gemv_f32,
        crate::backend::metal::q4_k_gemv_v2::F32_BATCHED_KERNEL_NAME
    );
    named!(
        &pipelines.q5_k_gemv,
        crate::backend::metal::q5_k_gemv::F16_BATCHED_KERNEL_NAME
    );
    named!(
        &pipelines.q6_k_gemv,
        crate::backend::metal::q6_k_gemv::F16_BATCHED_KERNEL_NAME
    );
    named!(
        &pipelines.q6_k_gemv_f32,
        crate::backend::metal::q6_k_gemv::F32_BATCHED_KERNEL_NAME
    );
    named!(&pipelines.k_quant_gemm.q4_k, "gemm_f16a_q4kw_tiled");
    named!(&pipelines.k_quant_gemm.q5_k, "gemm_f16a_q5kw_tiled");
    named!(&pipelines.k_quant_gemm.q6_k, "gemm_f16a_q6kw_tiled");
    named!(&pipelines.k_quant_gemm.q8_0, "gemm_f16a_q8_0w_tiled");
    named!(&pipelines.k_quant_gemm.q4_k_m8, "gemm_f16a_q4kw_m8");
    named!(&pipelines.k_quant_gemm.q5_k_m8, "gemm_f16a_q5kw_m8");
    named!(&pipelines.k_quant_gemm.q6_k_m8, "gemm_f16a_q6kw_m8");
    if let Some(pso) = &pipelines.dense_narrow {
        named!(pso, LINEAR_DENSE_NARROW_KERNEL);
    }
    pipelines.small_batch.selected_entry(actual)
}

fn plain(
    builder: &mut SelectedCommandCostBuilderV1,
    pipelines: &MetalLinearPipelines,
    launch: LinearLaunch,
    scratch: u64,
) -> Option<()> {
    if launch.transform.is_some() {
        return None;
    }
    let (pipeline, kind) =
        pipelines.plain_linear_dispatch(launch.format, launch.activation_type, launch.params);
    let geometry = grid(launch.params, kind)?;
    let algorithm = class(entry(pipelines, pipeline)?, launch, geometry, false)?;
    builder
        .kernel(
            algorithm,
            KernelNumericWorkV1 {
                logical_units: u64::from(launch.params.rows)
                    .checked_mul(u64::from(launch.params.out_features))?,
                padded_units: geometry.padded_outputs,
                inner_units_per_logical_unit: u64::from(launch.params.in_features),
                grid: geometry.groups,
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
        )
        .ok()
}

pub(super) fn projection(
    builder: &mut SelectedCommandCostBuilderV1,
    pipelines: &MetalLinearPipelines,
    launch: LinearLaunch,
    policy: Option<staged_prefill::StagingPolicy>,
    scratch: u64,
) -> Option<()> {
    if launch.transform.is_some() {
        return None;
    }
    if policy.is_some_and(|policy| staged_prefill::selected_for(launch, policy)) {
        let elements = u64::from(launch.params.in_features)
            .checked_mul(u64::from(launch.params.out_features))?;
        let blocks = elements.checked_div(256)?;
        let stage_name = match launch.format {
            LinearPhysicalFormat::Q4K => "stage_q4k_f16",
            LinearPhysicalFormat::Q5K => "stage_q5k_f16",
            LinearPhysicalFormat::Q6K => "stage_q6k_f16",
            _ => return None,
        };
        let stage = Grid {
            groups: [u32::try_from(blocks.div_ceil(8)).ok()?, 1, 1],
            threads: [128, 1, 1],
            threadgroup_memory: Some(0),
            padded_outputs: blocks.div_ceil(8).checked_mul(8)?.checked_mul(256)?,
        };
        builder
            .kernel(
                class(stage_name, launch, stage, true)?,
                KernelNumericWorkV1 {
                    logical_units: elements,
                    padded_units: stage.padded_outputs,
                    inner_units_per_logical_unit: 1,
                    grid: stage.groups,
                    scratch_bytes: scratch,
                    staged_weight_bytes: elements.checked_mul(2)?,
                },
            )
            .ok()?;
        let geometry = grid(launch.params, LinearDispatchKind::TiledGemm)?;
        builder
            .kernel(
                class("gemm_f16a_f16w_tiled", launch, geometry, true)?,
                KernelNumericWorkV1 {
                    logical_units: u64::from(launch.params.rows)
                        .checked_mul(u64::from(launch.params.out_features))?,
                    padded_units: geometry.padded_outputs,
                    inner_units_per_logical_unit: u64::from(launch.params.in_features),
                    grid: geometry.groups,
                    scratch_bytes: scratch,
                    staged_weight_bytes: 0,
                },
            )
            .ok()?;
    } else if let Some(parts) = launch.plain_plan.grouped_parts(launch) {
        for part in parts {
            plain(builder, pipelines, part, scratch)?;
        }
    } else if let Some(parts) = launch.plain_plan.parts(launch) {
        for part in parts {
            plain(builder, pipelines, part, scratch)?;
        }
    } else {
        plain(builder, pipelines, launch, scratch)?;
    }
    Some(())
}

pub(super) fn dense(
    pipelines: &MetalLinearPipelines,
    launches: &[LinearLaunch],
    tokens: u64,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = crate::backend::metal::vnext_runtime::selected_cost_builder(
        pipelines.structured_capture(),
        tokens,
    );
    for &launch in launches {
        projection(&mut builder, pipelines, launch, None, 0)?;
    }
    builder.finish().ok()
}

pub(super) fn swiglu(
    pipelines: &MetalLinearPipelines,
    gate: &[LinearLaunch],
    down: LinearLaunch,
    activation: SwiGluLaunch,
    policy: Option<staged_prefill::StagingPolicy>,
    tokens: u64,
    scratch: u64,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = crate::backend::metal::vnext_runtime::selected_cost_builder(
        pipelines.structured_capture(),
        tokens,
    );
    // Transformed routes are unavailable, so there is no hidden reuse decision
    // to reconstruct. Plain gate/up order is exactly Sequence::encode's order.
    for &launch in gate {
        projection(&mut builder, pipelines, launch, policy, scratch)?;
    }
    let units = u64::from(activation.params.rows)
        .checked_mul(u64::from(activation.params.intermediate_size))?;
    let geometry = Grid {
        groups: [u32::try_from(units.div_ceil(THREADS_PER_GROUP)).ok()?, 1, 1],
        threads: [u32::try_from(THREADS_PER_GROUP).ok()?, 1, 1],
        threadgroup_memory: None,
        padded_outputs: units
            .div_ceil(THREADS_PER_GROUP)
            .checked_mul(THREADS_PER_GROUP)?,
    };
    // Activation layout is bound to the actual gate/up stride, not down's.
    let mut activation_abi = down;
    activation_abi.params.output_stride = activation.params.gate_up_stride;
    activation_abi.params.output_column_offset = 0;
    builder
        .kernel(
            class(SWIGLU_KERNEL, activation_abi, geometry, false)?,
            KernelNumericWorkV1 {
                logical_units: units,
                padded_units: geometry.padded_outputs,
                inner_units_per_logical_unit: 1,
                grid: geometry.groups,
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
        )
        .ok()?;
    projection(&mut builder, pipelines, down, policy, scratch)?;
    builder.finish().ok()
}

#[cfg(test)]
pub(super) mod tests;
