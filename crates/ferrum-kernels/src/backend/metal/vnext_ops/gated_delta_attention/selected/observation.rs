//! Scalar records of the actual recurrent selector. No pipeline reaches worker.
use super::super::super::linear::FrozenLinear;
use super::super::super::primitives::FrozenPrimitive;
use super::*;
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};
struct Io {
    params: GatedDeltaParams,
    rms: FrozenPrimitive,
    input: FrozenLinear,
    output: FrozenLinear,
    residual: FrozenPrimitive,
}
impl Io {
    fn freeze(
        l: &MetalLinearPipelines,
        p: &MetalPrimitivePipelines,
        hidden: ElementType,
        v: Projection<'_>,
    ) -> Option<Self> {
        let rms = FrozenPrimitive::rms(
            p,
            hidden,
            ElementType::F16,
            v.params.tokens,
            v.params.hidden_size,
            v.params.epsilon,
        )?;
        let residual = FrozenPrimitive::residual(
            p,
            hidden,
            v.params.tokens.checked_mul(v.params.hidden_size)?,
        )?;
        let mut input = FrozenLinear::empty();
        for &launch in v.input {
            input.projection(
                l,
                launch,
                v.staged
                    .then_some(staged_prefill::StagingPolicy::GatedDelta),
            )?;
        }
        let mut output = FrozenLinear::empty();
        output.projection(l, v.output, None)?;
        Some(Self {
            params: v.params,
            rms,
            input,
            output,
            residual,
        })
    }
    fn input(&self, b: &mut CommandBuilder<'_>, scratch: u64) -> Option<()> {
        self.rms.append(&mut b.selected, scratch)?;
        self.input.append(&mut b.selected, scratch)
    }
    fn output(&self, b: &mut CommandBuilder<'_>, scratch: u64) -> Option<()> {
        norm(b, &self.params, true, scratch)?;
        self.output.append(&mut b.selected, scratch)?;
        self.residual.append(&mut b.selected, scratch)
    }
    fn variable_bytes(&self) -> Option<usize> {
        self.input
            .variable_bytes()?
            .checked_add(self.output.variable_bytes()?)
    }
    fn occurrences(&self) -> Option<u64> {
        self.input
            .occurrences()?
            .checked_add(self.output.occurrences()?)?
            .checked_add(3)
    }
}
#[derive(Clone, Copy)]
struct Recurrent {
    entry: &'static str,
    tile: u64,
    threads: u64,
}
impl Recurrent {
    fn append(&self, b: &mut CommandBuilder<'_>, p: &GatedDeltaParams, scratch: u64) -> Option<()> {
        let groups = u64::from(p.value_dim).div_ceil(self.tile);
        let heads = u64::from(p.value_heads);
        kernel(
            b,
            self.entry,
            p,
            heads.checked_mul(u64::from(p.value_dim))?,
            heads.checked_mul(groups)?.checked_mul(self.tile)?,
            u64::from(p.tokens).checked_mul(u64::from(p.key_dim))?,
            [groups, heads, 1],
            self.threads,
            scratch,
        )
    }
}
struct FrozenRow {
    params: GatedDeltaParams,
    recurrent: Recurrent,
    io: Option<Io>,
}
struct Template {
    packed: Option<Io>,
    rows: Vec<FrozenRow>,
    tokens: u64,
    scratch: u64,
    capture: ferrum_types::SloStructuredCostCapture,
    occurrences: u64,
}
pub(in crate::backend::metal::vnext_ops) fn payload_upper(
    dispatches: u64,
    rows: usize,
) -> Option<usize> {
    std::mem::size_of::<Template>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(
            crate::backend::metal::vnext_runtime::observation_capacity_upper(rows)?
                .checked_mul(std::mem::size_of::<FrozenRow>())?,
        )?
        .checked_add(FrozenLinear::variable_upper(
            dispatches,
            rows.checked_add(1)?.checked_mul(2)?,
        )?)
}
impl DeviceObservationTemplate for Template {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        let mut n = std::mem::size_of::<Self>()
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add(
                self.rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<FrozenRow>())?,
            )?;
        if let Some(io) = &self.packed {
            n = n.checked_add(io.variable_bytes()?)?;
        }
        for row in &self.rows {
            if let Some(io) = &row.io {
                n = n.checked_add(io.variable_bytes()?)?;
            }
        }
        Some(n)
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
        let mut b = CommandBuilder::new(self.capture, self.tokens, None);
        let projected = (|| {
            if let Some(io) = &self.packed {
                io.input(&mut b, self.scratch)?;
                gates(&mut b, &io.params, self.scratch)?;
                for row in &self.rows {
                    conv(&mut b, &row.params, self.scratch)?;
                }
                norm(&mut b, &io.params, false, self.scratch)?;
                for row in &self.rows {
                    row.recurrent.append(&mut b, &row.params, self.scratch)?;
                }
                io.output(&mut b, self.scratch)?;
            } else {
                for row in &self.rows {
                    let io = row.io.as_ref()?;
                    io.input(&mut b, self.scratch)?;
                    conv(&mut b, &row.params, self.scratch)?;
                    gates(&mut b, &row.params, self.scratch)?;
                    norm(&mut b, &row.params, false, self.scratch)?;
                    row.recurrent.append(&mut b, &row.params, self.scratch)?;
                    io.output(&mut b, self.scratch)?;
                }
            }
            Some(())
        })();
        projected.ok_or(StatisticalEvidenceUnknown::Overflow)?;
        Ok(vec![Some(b.selected.finish()?)])
    }
}
#[allow(clippy::too_many_arguments)]
pub(in crate::backend::metal::vnext_ops) fn freeze<'a>(
    a: &MetalGatedDeltaPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection<'a>>,
    rows: impl Iterator<Item = Row<'a>> + Clone,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    let mut count = 0usize;
    let mut total = 0u64;
    for row in rows.clone() {
        if !matches!(row.form, GatedDeltaExecutionForm::RecurrentScan) {
            return None;
        }
        count = count.checked_add(1)?;
        total = total.checked_add(u64::from(row.projection.params.tokens))?;
    }
    if count == 0 || count > ferrum_interfaces::execution_cost::MAX_COST_ROWS || total != tokens {
        return None;
    }
    let packed = match packed {
        Some(v) => {
            if u64::from(v.params.tokens) != tokens {
                return None;
            }
            Some(Io::freeze(l, p, hidden, v)?)
        }
        None => None,
    };
    let mut frozen = Vec::new();
    frozen.try_reserve_exact(count).ok()?;
    let mut occurrences = match &packed {
        Some(io) => io.occurrences()?.checked_add(2)?,
        None => 0,
    };
    for row in rows {
        let v = row.projection;
        let (_, entry, tile, threads) = recurrent(a, &v.params);
        let io = if packed.is_none() {
            Some(Io::freeze(l, p, hidden, v)?)
        } else {
            None
        };
        occurrences = occurrences
            .checked_add(if packed.is_some() { 4 } else { 6 })?
            .checked_add(io.as_ref().map_or(Some(0), Io::occurrences)?)?;
        frozen.push(FrozenRow {
            params: v.params,
            recurrent: Recurrent {
                entry,
                tile,
                threads,
            },
            io,
        });
    }
    Some(Arc::new(Template {
        packed,
        rows: frozen,
        tokens,
        scratch,
        capture: l.structured_capture(),
        occurrences,
    }))
}
