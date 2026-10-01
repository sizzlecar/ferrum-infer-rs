//! Actual primitive PSO selection and its passive statistical declaration.
//! No buffer, permission, coefficient, or dispatch order is changed here.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1,
    SelectedCommandCostEvidenceV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
pub(super) mod observation;
pub(super) use observation::{
    argmax as argmax_observation, embedding as embedding_observation,
    residual as residual_observation, rms as rms_observation,
};

#[derive(Clone, Copy)]
struct SelectionStamp {
    entry: &'static str,
    specialization: u64,
}

pub(super) struct Selection<'a> {
    pub pipeline: &'a ComputePipelineState,
    entry: &'static str,
    specialization: u64,
}
impl Selection<'_> {
    fn stamp(&self) -> SelectionStamp {
        SelectionStamp {
            entry: self.entry,
            specialization: self.specialization,
        }
    }
}
fn pick<'a>(
    pipeline: &'a ComputePipelineState,
    entry: &'static str,
    specialization: u64,
) -> Selection<'a> {
    Selection {
        pipeline,
        entry,
        specialization,
    }
}
pub(super) fn embedding(
    p: &MetalPrimitivePipelines,
    f: EmbeddingPhysicalFormat,
    out: ElementType,
) -> Option<Selection<'_>> {
    let (pipeline, entry) = match (f, out) {
        (EmbeddingPhysicalFormat::DenseF16, ElementType::F16) => {
            (&p.embedding_dense, EMBEDDING_DENSE_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q4K, ElementType::F16) => {
            (&p.embedding_q4_k, EMBEDDING_Q4_K_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q6K, ElementType::F16) => {
            (&p.embedding_q6_k, EMBEDDING_Q6_K_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q8_0, ElementType::F16) => {
            (&p.embedding_q8_0, EMBEDDING_Q8_0_KERNEL)
        }
        (EmbeddingPhysicalFormat::Pq2_0, ElementType::F16) => {
            (&p.embedding_pq2_0, EMBEDDING_PQ2_0_KERNEL)
        }
        (EmbeddingPhysicalFormat::DenseF16, ElementType::F32) => {
            (&p.embedding_dense_f32, EMBEDDING_DENSE_F32_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q4K, ElementType::F32) => {
            (&p.embedding_q4_k_f32, EMBEDDING_Q4_K_F32_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q6K, ElementType::F32) => {
            (&p.embedding_q6_k_f32, EMBEDDING_Q6_K_F32_KERNEL)
        }
        (EmbeddingPhysicalFormat::Q8_0, ElementType::F32) => {
            (&p.embedding_q8_0_f32, EMBEDDING_Q8_0_F32_KERNEL)
        }
        (EmbeddingPhysicalFormat::Pq2_0, ElementType::F32) => {
            (&p.embedding_pq2_0_f32, EMBEDDING_PQ2_0_F32_KERNEL)
        }
        _ => return None,
    };
    Some(pick(pipeline, entry, 0))
}
pub(super) fn rms(
    p: &MetalPrimitivePipelines,
    input: ElementType,
    out: ElementType,
) -> Option<Selection<'_>> {
    let (pipeline, entry) = match (input, out) {
        (ElementType::F16, ElementType::F16) => (&p.rms_norm, RMS_NORM_KERNEL),
        (ElementType::F32, ElementType::F16) => {
            (&p.rms_norm_f32_to_f16, RMS_NORM_F32_TO_F16_KERNEL)
        }
        (ElementType::F32, ElementType::F32) => (&p.rms_norm_f32, RMS_NORM_F32_KERNEL),
        _ => return None,
    };
    Some(pick(pipeline, entry, 0))
}
pub(super) fn residual(
    p: &MetalPrimitivePipelines,
    left: ElementType,
    right: ElementType,
    out: ElementType,
) -> Option<Selection<'_>> {
    let (pipeline, entry) = match (left, right, out) {
        (ElementType::F16, ElementType::F16, ElementType::F16) => {
            (&p.residual_add, RESIDUAL_ADD_KERNEL)
        }
        (ElementType::F32, ElementType::F16, ElementType::F32) => {
            (&p.residual_add_f32_f16, RESIDUAL_ADD_F32_F16_KERNEL)
        }
        _ => return None,
    };
    Some(pick(pipeline, entry, 0))
}
pub(super) fn argmax(
    p: &MetalPrimitivePipelines,
    logits: ElementType,
    parallel: bool,
) -> Option<Selection<'_>> {
    let (pipeline, entry) = match (logits, parallel) {
        (ElementType::F16, false) => (&p.last_token_masked_argmax, LAST_TOKEN_MASKED_ARGMAX_KERNEL),
        (ElementType::F32, false) => (
            &p.last_token_masked_argmax_f32,
            LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL,
        ),
        (ElementType::F16, true) => (&p.parallel_masked_argmax, LAST_TOKEN_MASKED_ARGMAX_KERNEL),
        (ElementType::F32, true) => (
            &p.parallel_masked_argmax_f32,
            LAST_TOKEN_MASKED_ARGMAX_F32_KERNEL,
        ),
        _ => return None,
    };
    Some(pick(pipeline, entry, u64::from(parallel)))
}

#[allow(clippy::too_many_arguments)]
fn push(
    builder: &mut SelectedCommandCostBuilderV1,
    selected: Selection<'_>,
    logical: u64,
    padded: u64,
    inner: u64,
    grid: [u64; 3],
    threads: u64,
    scratch: u64,
    extra: u64,
) -> Option<()> {
    push_stamp(
        builder,
        selected.stamp(),
        logical,
        padded,
        inner,
        grid,
        threads,
        scratch,
        extra,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_stamp(
    builder: &mut SelectedCommandCostBuilderV1,
    selected: SelectionStamp,
    logical: u64,
    padded: u64,
    inner: u64,
    grid: [u64; 3],
    threads: u64,
    scratch: u64,
    extra: u64,
) -> Option<()> {
    static NUMERIC: OnceLock<[u8; 32]> = OnceLock::new();
    let numeric = *NUMERIC.get_or_init(|| {
        let mut h = Sha256::new();
        h.update(b"metal.primitives.compile-options-default.v1");
        h.update(SHADER_SOURCE.as_bytes());
        h.update(include_str!("selected.rs").as_bytes());
        h.update(include_str!("selected/observation.rs").as_bytes());
        h.finalize().into()
    });
    let mut layout = Sha256::new();
    layout.update(b"contiguous.primitive.params.v1");
    for n in [selected.specialization, threads, extra] {
        layout.update(n.to_le_bytes());
    }
    let class =
        SelectedAlgorithmClassV1::new(selected.entry, 1, numeric, layout.finalize().into()).ok()?;
    builder
        .kernel(
            class,
            KernelNumericWorkV1 {
                logical_units: logical,
                padded_units: padded,
                inner_units_per_logical_unit: inner,
                grid: [
                    u32::try_from(grid[0]).ok()?,
                    u32::try_from(grid[1]).ok()?,
                    u32::try_from(grid[2]).ok()?,
                ],
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
        )
        .ok()
}
pub(super) fn push_embedding(
    builder: &mut SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    format: EmbeddingPhysicalFormat,
    out: ElementType,
    params: EmbeddingParams,
) -> Option<()> {
    observation::append_embedding(builder, embedding(p, format, out)?.stamp(), params)
}
pub(super) fn embedding_evidence(
    p: &MetalPrimitivePipelines,
    launches: &[EmbeddingLaunch],
    out: ElementType,
    tokens: u64,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b =
        crate::backend::metal::vnext_runtime::selected_cost_builder(p.structured_capture(), tokens);
    for launch in launches {
        if launch.transform.is_some() {
            return None;
        }
        push_embedding(&mut b, p, launch.format, out, launch.params)?;
    }
    b.finish().ok()
}
pub(super) fn rms_evidence(
    p: &MetalPrimitivePipelines,
    params: RmsNormParams,
    input: ElementType,
    out: ElementType,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b = crate::backend::metal::vnext_runtime::selected_cost_builder(
        p.structured_capture(),
        u64::from(params.rows),
    );
    push_rms(&mut b, p, params, input, out, 0)?;
    b.finish().ok()
}
pub(super) fn push_rms(
    builder: &mut SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    params: RmsNormParams,
    input: ElementType,
    out: ElementType,
    scratch: u64,
) -> Option<()> {
    observation::append_rms(builder, rms(p, input, out)?.stamp(), params, scratch)
}
pub(super) fn residual_evidence(
    p: &MetalPrimitivePipelines,
    params: ResidualAddParams,
    left: ElementType,
    right: ElementType,
    out: ElementType,
    tokens: u64,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b =
        crate::backend::metal::vnext_runtime::selected_cost_builder(p.structured_capture(), tokens);
    push_residual(&mut b, p, params, left, right, out, 0)?;
    b.finish().ok()
}
pub(super) fn push_residual(
    builder: &mut SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    params: ResidualAddParams,
    left: ElementType,
    right: ElementType,
    out: ElementType,
    scratch: u64,
) -> Option<()> {
    observation::append_residual(
        builder,
        residual(p, left, right, out)?.stamp(),
        params,
        scratch,
    )
}
pub(super) fn push_argmax(
    b: &mut SelectedCommandCostBuilderV1,
    p: &MetalPrimitivePipelines,
    params: LastTokenMaskedArgmaxParams,
    logits: ElementType,
    scratch: u64,
) -> Option<()> {
    observation::Argmax::freeze(p, params, logits)?.append(b, scratch)
}
pub(super) fn argmax_evidence(
    p: &MetalPrimitivePipelines,
    launches: &[LastTokenMaskedArgmaxLaunch],
    logits: ElementType,
    scratch: u64,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b = crate::backend::metal::vnext_runtime::selected_cost_builder(
        p.structured_capture(),
        launches.len() as u64,
    );
    for launch in launches {
        push_argmax(&mut b, p, launch.params, logits, scratch)?;
    }
    b.finish().ok()
}

#[cfg(test)]
pub(super) mod tests;

pub(super) use observation::payload_upper as observation_payload_upper;
