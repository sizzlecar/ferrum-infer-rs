//! Ordered, complete GDN command evidence. Native projections and the explicit
//! RN-F16 cuBLAS API contract are separate; other opaque backends stay Unknown.
//! The current invocation supplies both numeric work and replay
//! geometry; no capture-wave totals are copied into a later token.
use super::super::cublas_api::{CublasHandleApiIdentity, GemmF16ApiPlan};
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::{
    q8_f32scale, q8_pair, selected as native_selected,
};
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1, StatisticalTransferKindV1,
};
use ferrum_types::SloStructuredCostCapture;
use std::sync::OnceLock;

pub(super) fn attach(
    command: ferrum_interfaces::vnext::OperationCostCommand,
    evidence: Option<SelectedCommandCostEvidenceV1>,
) -> ferrum_interfaces::vnext::OperationCostCommand {
    match evidence {
        Some(evidence) => command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command),
        None => command,
    }
}

fn class(entry: &str, layout: &[u8]) -> Option<SelectedAlgorithmClassV1> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let numerical = *NUMERICAL.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"cuda.gdn.installed-native-arithmetic.v1");
        hash.update(include_bytes!("../attention.rs"));
        hash.update(include_bytes!("precision.rs"));
        hash.update(include_bytes!("native_projection.rs"));
        hash.update(include_bytes!("../../native_blocks/q8_pair.rs"));
        for ptx in [
            crate::ptx::VNEXT_GGUF,
            crate::ptx::RMS_NORM,
            crate::ptx::LINEAR_ATTENTION,
            crate::ptx::GATED_DELTA_RULE,
            crate::ptx::SANDWICH_NORM,
            crate::ptx::RESIDUAL_ADD,
        ] {
            hash.update(ptx.as_bytes());
        }
        hash.finalize().into()
    });
    SelectedAlgorithmClassV1::new(entry, 1, numerical, Sha256::digest(layout).into()).ok()
}

fn transfer(
    builder: &mut SelectedCommandCostBuilderV1,
    kind: StatisticalTransferKindV1,
    bytes: u64,
) -> Option<()> {
    let entry = match kind {
        StatisticalTransferKindV1::HostToDevice => "cuda.cuMemcpyHtoDAsync.gdn-control",
        StatisticalTransferKindV1::Fill => "cuda.cuMemsetD8Async.gdn-token-sequence",
        _ => return None,
    };
    builder
        .transfer(
            class(entry, b"gdn.retained-contiguous-control.v1")?,
            kind,
            bytes,
        )
        .ok()
}

#[allow(clippy::too_many_arguments)]
fn kernel(
    builder: &mut SelectedCommandCostBuilderV1,
    entry: &str,
    config: LaunchConfig,
    logical: u64,
    padded: u64,
    inner: u64,
    scratch: u64,
    fixed: &[u64],
) -> Option<()> {
    let mut layout = Vec::with_capacity(40);
    layout.extend_from_slice(b"gdn.native-kernel-geometry.v1");
    for dimension in [config.block_dim.0, config.block_dim.1, config.block_dim.2] {
        layout.extend_from_slice(&dimension.to_le_bytes());
    }
    builder
        .kernel_with_replay_geometry(
            class(entry, &layout)?,
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

pub(super) fn bindings(
    participants: usize,
    tokens: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    for _ in 0..participants {
        transfer(
            &mut builder,
            StatisticalTransferKindV1::HostToDevice,
            STATE_BINDING_SLOT_BYTES,
        )?;
    }
    builder.finish().ok()
}

/// Native kernel selection and library API selection have different identities.
/// Dense F16 is admitted only for the explicit RN contract, never inferred from
/// an empty native matrix list or from a vendor-private kernel name.
#[derive(Clone, Copy)]
pub(super) enum ProjectionEvidence<'a> {
    Native {
        input: &'a [weights::MatrixPart],
        output: &'a [weights::MatrixPart],
    },
    Library(CublasHandleApiIdentity),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute(
    shape: AttentionShape,
    precision: AttentionPrecision,
    projection: AttentionProjection,
    evidence: ProjectionEvidence<'_>,
    leaves: impl IntoIterator<Item = (u64, u32, bool)>,
    tokens: u64,
    participants: usize,
    pair_enabled: bool,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    if matches!(precision, AttentionPrecision::F32MasterGgufF16Projections)
        != matches!(evidence, ProjectionEvidence::Library(_))
    {
        return None;
    }
    let q8 = match (projection, evidence) {
        (AttentionProjection::NativeQ8 { .. }, ProjectionEvidence::Native { .. }) => {
            Some(q8_f32scale::Q8SumPolicy::Quantized)
        }
        (AttentionProjection::Native { .. }, ProjectionEvidence::Native { .. }) => None,
        (AttentionProjection::F16, ProjectionEvidence::Library(_))
            if matches!(precision, AttentionPrecision::F32MasterGgufF16Projections) =>
        {
            None
        }
        _ => return None,
    };
    if q8.is_some() != precision.quantizes_projections() {
        return None;
    }
    let layout = ScratchLayout::new(shape, tokens, participants, projection).ok()?;
    let cuda = shape.cuda_shape().ok()?;
    let scratch = layout.required_bytes;
    let transform_bytes = projection
        .staging_bytes_per_token(shape.projection_staging_features())
        .ok()?
        .checked_mul(tokens)?;
    let mut seen_tokens = 0_u64;
    let mut seen_participants = 0_usize;
    for (count, batch, packed_indices) in leaves {
        if count == 0 || batch == 0 || (!packed_indices && batch != 1) {
            return None;
        }
        seen_tokens = seen_tokens.checked_add(count)?;
        seen_participants = seen_participants.checked_add(batch as usize)?;
        if seen_tokens > tokens || seen_participants > participants {
            return None;
        }
        let batch_i32 = i32::try_from(batch).ok()?;
        let count_i32 = i32::try_from(count).ok()?;
        shape.validate_launch_extents(count).ok()?;
        transfer(
            &mut builder,
            StatisticalTransferKindV1::HostToDevice,
            (u64::from(batch) + 1).checked_mul(4)?,
        )?;
        transfer(
            &mut builder,
            if packed_indices {
                StatisticalTransferKindV1::HostToDevice
            } else {
                StatisticalTransferKindV1::Fill
            },
            count.checked_mul(4)?,
        )?;
        kernel(
            &mut builder,
            precision.norm(),
            launch_geometry::rms(count, cuda.hidden_size).ok()?,
            count,
            count,
            shape.hidden_size,
            scratch,
            &[shape.hidden_size, u64::from(shape.epsilon.to_bits())],
        )?;
        let pack_bytes = match projection {
            AttentionProjection::NativeQ8 {
                pack_bytes_per_token,
            } => pack_bytes_per_token.checked_mul(count)?,
            _ => 0,
        };
        match evidence {
            ProjectionEvidence::Native { input, .. } => projection_work(
                &mut builder,
                input,
                count,
                u32::try_from(shape.qkvzba_features).ok()?,
                q8,
                pair_enabled,
                transform_bytes,
                pack_bytes,
            )?,
            ProjectionEvidence::Library(identity) => library_projection(
                &mut builder,
                identity,
                count,
                shape.qkvzba_features,
                shape.hidden_size,
            )?,
        }
        let prepare_work = count
            .checked_mul(shape.qkv_features)?
            .max(count.checked_mul(shape.value_heads)?)
            .max(
                shape
                    .conv_state_elements()
                    .ok()?
                    .checked_mul(u64::from(batch))?,
            );
        let prepare_config = launch_geometry::prepare(shape, count, batch_i32).ok()?;
        let prepare_entry = match shape.decay_parameterization {
            GatedDeltaDecayParameterization::LogRate => PREPARE_FUNCTION,
            GatedDeltaDecayParameterization::NegativeRate => PREPARE_NEGATIVE_RATE_FUNCTION,
        };
        let dimensions = [
            u64::from(batch),
            count,
            shape.key_heads,
            shape.value_heads,
            shape.key_head_dim,
            shape.value_head_dim,
            shape.conv_kernel,
        ];
        kernel(
            &mut builder,
            prepare_entry,
            prepare_config,
            prepare_work,
            u64::from(prepare_config.grid_dim.0).checked_mul(u64::from(THREADS_PER_BLOCK))?,
            shape.conv_kernel.max(1),
            scratch,
            &dimensions,
        )?;
        let conv_elements = shape.conv_state_elements().ok()?;
        let conv_config =
            launch_geometry::conv(batch_i32, i32::try_from(conv_elements).ok()?).ok()?;
        kernel(
            &mut builder,
            CONV_STATE_COMMIT_FUNCTION,
            conv_config,
            conv_elements.checked_mul(u64::from(batch))?,
            u64::from(conv_config.grid_dim.0).checked_mul(u64::from(THREADS_PER_BLOCK))?,
            1,
            scratch,
            &[u64::from(batch), conv_elements],
        )?;
        let qk_rows = count.checked_mul(shape.key_heads)?;
        kernel(
            &mut builder,
            QK_NORM_FUNCTION,
            launch_geometry::qk(count, cuda).ok()?,
            qk_rows,
            qk_rows,
            shape.key_head_dim.checked_mul(2)?,
            scratch,
            &[
                count,
                shape.key_heads,
                shape.key_head_dim,
                u64::from(1.0e-6_f32.to_bits()),
            ],
        )?;
        let delta_logical = u64::from(batch).checked_mul(shape.value_features)?;
        let delta_config = launch_geometry::delta(batch_i32, cuda).ok()?;
        let delta_padded = if cuda.tiled_delta {
            u64::from(batch)
                .checked_mul(shape.value_heads)?
                .checked_mul(shape.value_head_dim.div_ceil(16))?
                .checked_mul(16)?
        } else {
            delta_logical
        };
        kernel(
            &mut builder,
            launch_geometry::delta_entry(cuda),
            delta_config,
            delta_logical,
            delta_padded,
            count.checked_mul(shape.key_head_dim)?,
            scratch,
            &[
                u64::from(batch),
                count_i32 as u64,
                shape.key_heads,
                shape.value_heads,
                shape.key_head_dim,
                shape.value_head_dim,
                0,
                u64::from(cuda.scale.to_bits()),
            ],
        )?;
        let gated_rows = count.checked_mul(shape.value_heads)?;
        kernel(
            &mut builder,
            GATED_NORM_FUNCTION,
            launch_geometry::gated(count, cuda).ok()?,
            gated_rows,
            gated_rows,
            shape.value_head_dim,
            scratch,
            &[
                gated_rows,
                shape.value_head_dim,
                u64::from(shape.epsilon.to_bits()),
            ],
        )?;
        let cast_elements = count.checked_mul(shape.value_features)?;
        let cast_config = launch_geometry::cast(cast_elements).ok()?;
        kernel(
            &mut builder,
            F32_TO_F16_FUNCTION,
            cast_config,
            cast_elements,
            u64::from(cast_config.grid_dim.0).checked_mul(u64::from(THREADS_PER_BLOCK))?,
            1,
            scratch,
            &[cast_elements],
        )?;
        match evidence {
            ProjectionEvidence::Native { output, .. } => projection_work(
                &mut builder,
                output,
                count,
                u32::try_from(shape.hidden_size).ok()?,
                q8,
                pair_enabled,
                transform_bytes,
                pack_bytes,
            )?,
            ProjectionEvidence::Library(identity) => library_projection(
                &mut builder,
                identity,
                count,
                shape.hidden_size,
                shape.value_features,
            )?,
        }
        let residual_elements = count.checked_mul(shape.hidden_size)?;
        let residual_config = launch_geometry::residual(residual_elements).ok()?;
        kernel(
            &mut builder,
            precision.residual(),
            residual_config,
            residual_elements,
            u64::from(residual_config.grid_dim.0).checked_mul(u64::from(THREADS_PER_BLOCK))?,
            1,
            scratch,
            &[residual_elements],
        )?;
    }
    if seen_tokens != tokens || seen_participants != participants {
        return None;
    }
    builder.finish().ok()
}

fn library_projection(
    builder: &mut SelectedCommandCostBuilderV1,
    identity: CublasHandleApiIdentity,
    rows: u64,
    columns: u64,
    reduction: u64,
) -> Option<()> {
    GemmF16ApiPlan::new(
        i32::try_from(rows).ok()?,
        i32::try_from(columns).ok()?,
        i32::try_from(reduction).ok()?,
    )
    .ok()?
    .append_selected(builder, identity)
    .ok()
}

fn projection_work(
    builder: &mut SelectedCommandCostBuilderV1,
    parts: &[weights::MatrixPart],
    rows: u64,
    stride: u32,
    q8: Option<q8_f32scale::Q8SumPolicy>,
    pair_enabled: bool,
    transform_bytes: u64,
    pack_bytes: u64,
) -> Option<()> {
    let columns = parts.first()?.columns;
    if parts.iter().try_fold(0_u32, |end, part| {
        (part.columns == columns && part.output_offset == end && part.rows != 0).then_some(())?;
        end.checked_add(part.rows)
    }) != Some(stride)
    {
        return None;
    }
    let mut remaining = rows;
    while remaining != 0 {
        let count = remaining.min(super::super::native_matrix::MAX_ROWS) as u32;
        if let Some(policy) = q8 {
            q8_f32scale::selected::append_projection(
                builder, parts, rows, count, stride, policy, pack_bytes,
            )?;
        } else {
            let mut index = 0;
            while let Some(part) = parts.get(index) {
                let pair = parts
                    .get(index + 1)
                    .and_then(|next| {
                        q8_pair::PairPlan::select(
                            part,
                            next,
                            count,
                            columns,
                            stride,
                            ElementType::F16,
                            ElementType::F16,
                        )
                    })
                    .filter(|_| pair_enabled);
                if let Some(pair) = pair {
                    let config = pair.launch_config();
                    let fixed = pair.parameters().map(u64::from);
                    kernel(
                        builder,
                        "vnext_gguf_linear_q8_pair_tiled_f16",
                        config,
                        u64::from(count).checked_mul(64)?,
                        u64::from(config.grid_dim.0)
                            .checked_mul(4)?
                            .checked_mul(u64::from(config.grid_dim.1))?
                            .checked_mul(8)?
                            .checked_mul(2)?,
                        u64::from(columns),
                        0,
                        &fixed,
                    )?;
                    index += 2;
                } else {
                    native_selected::append_transformed_linear(
                        builder,
                        part,
                        count,
                        stride,
                        ElementType::F16,
                        transform_bytes,
                    )?;
                    index += 1;
                }
            }
        }
        remaining -= u64::from(count);
    }
    Some(())
}

pub(super) fn from_prepared(
    prepared: &prepared::PreparedAttention,
    precision: AttentionPrecision,
    pair_enabled: bool,
    capture: SloStructuredCostCapture,
    identity: Option<CublasHandleApiIdentity>,
) -> Option<SelectedCommandCostEvidenceV1> {
    if capture == SloStructuredCostCapture::Disabled {
        return None;
    }
    let evidence = match (&prepared.shared.qkvzba, &prepared.shared.output) {
        (
            SharedProjectionWeight::Native { parts: input, .. },
            SharedProjectionWeight::Native { parts: output, .. },
        ) => ProjectionEvidence::Native { input, output },
        (SharedProjectionWeight::F16 { .. }, SharedProjectionWeight::F16 { .. })
            if matches!(precision, AttentionPrecision::F32MasterGgufF16Projections) =>
        {
            ProjectionEvidence::Library(identity?)
        }
        _ => return None,
    };
    compute(
        prepared.shape,
        precision,
        prepared.projection,
        evidence,
        prepared.launches.iter().map(|launch| {
            (
                launch.tokens,
                launch.batch_i32 as u32,
                launch.host_token_seq_indices.is_some(),
            )
        }),
        prepared.total_tokens,
        prepared.participant_token_counts.len(),
        pair_enabled,
        capture,
    )
}

#[cfg(test)]
mod tests;
