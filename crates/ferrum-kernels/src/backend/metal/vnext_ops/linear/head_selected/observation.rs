use super::*;
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
use ferrum_interfaces::vnext::{DeviceObservationTemplate, FrozenObservationInput};
enum Projection {
    Strict(FrozenLinear),
    Half(Vec<half_head::FrozenHalfHead>),
}
struct Template {
    projection: Projection,
    packed: Option<(LastTokenPackedScratchLayout, u32, bool)>,
    tokens: u64,
    capture: ferrum_types::SloStructuredCostCapture,
    occurrences: u64,
}
pub(in crate::backend::metal::vnext_ops) fn payload_upper(
    dispatches: u64,
    rows: usize,
) -> Option<usize> {
    let half = crate::backend::metal::vnext_runtime::observation_capacity_upper(rows)?
        .checked_mul(std::mem::size_of::<half_head::FrozenHalfHead>())?;
    let strict = FrozenLinear::variable_upper(dispatches, 1)?;
    std::mem::size_of::<Template>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(half.max(strict))
}
impl DeviceObservationTemplate for Template {
    fn command_count(&self) -> usize {
        1
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        let variable = match &self.projection {
            Projection::Strict(p) => p.variable_bytes()?,
            Projection::Half(rows) => rows
                .capacity()
                .checked_mul(std::mem::size_of::<half_head::FrozenHalfHead>())?,
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
        let mut b =
            crate::backend::metal::vnext_runtime::selected_cost_builder(self.capture, self.tokens);
        let scratch = self.packed.map_or(0, |(l, _, _)| l.required_bytes);
        let projected = (|| {
            if let Some((l, count, false)) = self.packed {
                for _ in 0..count {
                    blit(&mut b, l.input_row_bytes, true)?;
                }
            }
            match &self.projection {
                Projection::Strict(p) => p.append(&mut b, scratch)?,
                Projection::Half(rows) => {
                    for row in rows {
                        row.append(&mut b, scratch)?;
                    }
                }
            }
            if let Some((l, count, _)) = self.packed {
                for _ in 0..count {
                    blit(&mut b, l.output_row_bytes, false)?;
                }
            }
            Some(())
        })();
        projected.ok_or(StatisticalEvidenceUnknown::Overflow)?;
        Ok(vec![Some(b.finish()?)])
    }
}
pub(in crate::backend::metal::vnext_ops) fn freeze(
    projection: &LastTokenProjection,
    launches: &[LinearLaunch],
    tokens: u64,
    packed: Option<(LastTokenPackedScratchLayout, u32, bool)>,
) -> Option<Arc<dyn DeviceObservationTemplate>> {
    if launches.is_empty()
        || launches.len() > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS
        || launches.iter().any(|l| l.transform.is_some())
    {
        return None;
    }
    if packed.is_none() && launches.iter().any(|l| l.params.rows != 1) {
        return None;
    }
    if let Some((_, n, _)) = packed {
        if launches.len() != 1 || n == 0 || launches[0].params.rows != n {
            return None;
        }
    }
    let (frozen, mut occurrences) = match projection {
        LastTokenProjection::Strict(p) => {
            let mut f = FrozenLinear::empty();
            for &l in launches {
                f.projection(p, l, None)?;
            }
            let n = f.occurrences()?;
            (Projection::Strict(f), n)
        }
        LastTokenProjection::Half(p) => {
            let mut rows = Vec::new();
            rows.try_reserve_exact(launches.len()).ok()?;
            let mut n = 0u64;
            for &l in launches {
                let f = p.freeze_observation(l)?;
                n = n.checked_add(f.occurrences())?;
                rows.push(f);
            }
            (Projection::Half(rows), n)
        }
    };
    if let Some((_, n, shared)) = packed {
        occurrences =
            occurrences.checked_add(u64::from(n).checked_mul(if shared { 1 } else { 2 })?)?;
    }
    Some(Arc::new(Template {
        projection: frozen,
        packed,
        tokens,
        capture: projection.structured_capture(),
        occurrences,
    }))
}
