//! Selector stamps and scalar parameters only. The worker never consults PSOs.
use super::*;
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};

/// Frozen actual selector identity and its scalar ABI, reusable by compound
/// commands. No pipeline or execution resource crosses into the worker.
#[derive(Clone, Copy)]
pub(in crate::backend::metal::vnext_ops) enum FrozenPrimitive {
    Rms(SelectionStamp, RmsNormParams),
    Residual(SelectionStamp, ResidualAddParams),
}
impl FrozenPrimitive {
    pub(in crate::backend::metal::vnext_ops) fn rms(
        p: &MetalPrimitivePipelines,
        input: ElementType,
        output: ElementType,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
    ) -> Option<Self> {
        Some(Self::Rms(
            super::rms(p, input, output)?.stamp(),
            RmsNormParams {
                rows,
                hidden_size,
                epsilon,
            },
        ))
    }
    pub(in crate::backend::metal::vnext_ops) fn residual(
        p: &MetalPrimitivePipelines,
        hidden: ElementType,
        elements: u32,
    ) -> Option<Self> {
        Some(Self::Residual(
            super::residual(p, hidden, ElementType::F16, hidden)?.stamp(),
            ResidualAddParams { elements },
        ))
    }
    pub(in crate::backend::metal::vnext_ops) fn append(
        &self,
        b: &mut SelectedCommandCostBuilderV1,
        scratch: u64,
    ) -> Option<()> {
        match self {
            Self::Rms(s, p) => append_rms(b, *s, *p, scratch),
            Self::Residual(s, p) => append_residual(b, *s, *p, scratch),
        }
    }
}

enum Primitive {
    Embedding(Vec<(SelectionStamp, EmbeddingParams)>),
    Rms(SelectionStamp, RmsNormParams),
    Residual(SelectionStamp, ResidualAddParams),
    Argmax { rows: Vec<Argmax>, scratch: u64 },
}
struct Template {
    primitive: Primitive,
    capture: ferrum_types::SloStructuredCostCapture,
    tokens: u64,
    occurrences: u64,
}
pub(in crate::backend::metal::vnext_ops) fn payload_upper(rows: usize) -> Option<usize> {
    let entry =
        std::mem::size_of::<Argmax>().max(std::mem::size_of::<(SelectionStamp, EmbeddingParams)>());
    std::mem::size_of::<Template>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(
            crate::backend::metal::vnext_runtime::observation_capacity_upper(rows)?
                .checked_mul(entry)?,
        )
}
impl DeviceObservationTemplate for Template {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        let variable = match &self.primitive {
            Primitive::Embedding(rows) => rows
                .capacity()
                .checked_mul(std::mem::size_of::<(SelectionStamp, EmbeddingParams)>())?,
            Primitive::Argmax { rows, .. } => {
                rows.capacity().checked_mul(std::mem::size_of::<Argmax>())?
            }
            _ => 0,
        };
        std::mem::size_of::<Self>()
            .checked_add(variable)?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(self.occurrences)?
            .checked_add(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
    }
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        if input.tokens() != self.tokens
            || !input.participant_ranges().is_empty()
            || !input.source_ranges().is_empty()
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let mut builder =
            crate::backend::metal::vnext_runtime::selected_cost_builder(self.capture, self.tokens);
        let projected = match &self.primitive {
            Primitive::Embedding(rows) => rows
                .iter()
                .try_for_each(|(stamp, params)| append_embedding(&mut builder, *stamp, *params)),
            Primitive::Rms(stamp, params) => append_rms(&mut builder, *stamp, *params, 0),
            Primitive::Residual(stamp, params) => append_residual(&mut builder, *stamp, *params, 0),
            Primitive::Argmax { rows, scratch } => rows
                .iter()
                .try_for_each(|row| row.append(&mut builder, *scratch)),
        };
        projected.ok_or(StatisticalEvidenceUnknown::Overflow)?;
        Ok(vec![Some(builder.finish()?)])
    }
}

fn template(
    p: &MetalPrimitivePipelines,
    primitive: Primitive,
    tokens: u64,
    occurrences: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if occurrences == 0 || occurrences > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS as u64
    {
        return None;
    }
    Some(Arc::new(Template {
        primitive,
        capture: p.structured_capture(),
        tokens,
        occurrences,
    }))
}

pub(in crate::backend::metal::vnext_ops) fn embedding(
    p: &MetalPrimitivePipelines,
    launches: &[EmbeddingLaunch],
    out: ElementType,
    tokens: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if launches.is_empty() || launches.len() > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS
    {
        return None;
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(launches.len()).ok()?;
    for launch in launches {
        if launch.transform.is_some() {
            return None;
        }
        rows.push((
            super::embedding(p, launch.format, out)?.stamp(),
            launch.params,
        ));
    }
    template(p, Primitive::Embedding(rows), tokens, launches.len() as u64)
}
pub(in crate::backend::metal::vnext_ops) fn rms(
    p: &MetalPrimitivePipelines,
    params: RmsNormParams,
    input: ElementType,
    out: ElementType,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    template(
        p,
        Primitive::Rms(super::rms(p, input, out)?.stamp(), params),
        u64::from(params.rows),
        1,
    )
}
pub(in crate::backend::metal::vnext_ops) fn residual(
    p: &MetalPrimitivePipelines,
    params: ResidualAddParams,
    left: ElementType,
    right: ElementType,
    out: ElementType,
    tokens: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    template(
        p,
        Primitive::Residual(super::residual(p, left, right, out)?.stamp(), params),
        tokens,
        1,
    )
}
pub(in crate::backend::metal::vnext_ops) fn argmax(
    p: &MetalPrimitivePipelines,
    launches: &[LastTokenMaskedArgmaxLaunch],
    logits: ElementType,
    scratch: u64,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if launches.is_empty() || launches.len() > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS
    {
        return None;
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(launches.len()).ok()?;
    let mut occurrences = 0u64;
    for launch in launches {
        let row = Argmax::freeze(p, launch.params, logits)?;
        occurrences = occurrences.checked_add(1 + u64::from(row.finalize.is_some()))?;
        rows.push(row);
    }
    template(
        p,
        Primitive::Argmax { rows, scratch },
        launches.len() as u64,
        occurrences,
    )
}

#[derive(Clone, Copy)]
pub(super) struct Argmax {
    selected: SelectionStamp,
    finalize: Option<SelectionStamp>,
    params: LastTokenMaskedArgmaxParams,
}
impl Argmax {
    pub(super) fn freeze(
        p: &MetalPrimitivePipelines,
        params: LastTokenMaskedArgmaxParams,
        logits: ElementType,
    ) -> Option<Self> {
        let parallel = masked_argmax_dispatch_count(params.vocabulary_size) == 2;
        Some(Self {
            selected: super::argmax(p, logits, parallel)?.stamp(),
            finalize: parallel.then(|| {
                pick(&p.masked_argmax_finalize, "vnext_masked_argmax_finalize", 0).stamp()
            }),
            params,
        })
    }
    pub(super) fn append(
        &self,
        builder: &mut SelectedCommandCostBuilderV1,
        scratch: u64,
    ) -> Option<()> {
        let groups = if self.finalize.is_some() {
            MASKED_ARGMAX_PARTITIONS
        } else {
            1
        };
        let n = u64::from(self.params.vocabulary_size);
        let width = groups.checked_mul(THREADS_PER_GROUP)?;
        push_stamp(
            builder,
            self.selected,
            n,
            n.div_ceil(width).checked_mul(width)?,
            1,
            [groups, 1, 1],
            THREADS_PER_GROUP,
            scratch,
            u64::from(self.params.repetition_capacity),
        )?;
        if let Some(finalize) = self.finalize {
            push_stamp(
                builder,
                finalize,
                MASKED_ARGMAX_PARTITIONS,
                32,
                1,
                [1, 1, 1],
                32,
                scratch,
                0,
            )?;
        }
        Some(())
    }
}
pub(super) fn append_embedding(
    builder: &mut SelectedCommandCostBuilderV1,
    stamp: SelectionStamp,
    params: EmbeddingParams,
) -> Option<()> {
    let groups = u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP);
    let rows = u64::from(params.token_count);
    push_stamp(
        builder,
        stamp,
        rows.checked_mul(u64::from(params.hidden_size))?,
        rows.checked_mul(groups)?.checked_mul(THREADS_PER_GROUP)?,
        1,
        [groups, rows, 1],
        THREADS_PER_GROUP,
        0,
        0,
    )
}
pub(super) fn append_rms(
    builder: &mut SelectedCommandCostBuilderV1,
    stamp: SelectionStamp,
    params: RmsNormParams,
    scratch: u64,
) -> Option<()> {
    let rows = u64::from(params.rows);
    push_stamp(
        builder,
        stamp,
        rows,
        rows,
        u64::from(params.hidden_size),
        [rows, 1, 1],
        THREADS_PER_GROUP,
        scratch,
        u64::from(params.epsilon.to_bits()),
    )
}
pub(super) fn append_residual(
    builder: &mut SelectedCommandCostBuilderV1,
    stamp: SelectionStamp,
    params: ResidualAddParams,
    scratch: u64,
) -> Option<()> {
    let n = u64::from(params.elements);
    let groups = n.div_ceil(THREADS_PER_GROUP);
    push_stamp(
        builder,
        stamp,
        n,
        groups.checked_mul(THREADS_PER_GROUP)?,
        1,
        [groups, 1, 1],
        THREADS_PER_GROUP,
        scratch,
        0,
    )
}
