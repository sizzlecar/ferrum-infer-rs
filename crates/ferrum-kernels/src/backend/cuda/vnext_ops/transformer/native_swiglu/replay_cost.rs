//! Numeric FFN plan retained from actual preparation for resident replay.
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_types::SloStructuredCostCapture;
use std::ops::Range;

enum Leaves {
    Packed(u32),
    Participants(Box<[(u32, u64)]>),
}

pub(in crate::backend::cuda::vnext_ops::transformer) struct Recipe {
    gate: Box<[weights::MatrixPart]>,
    down: Box<[weights::MatrixPart]>,
    leaves: Leaves,
    hidden: u32,
    intermediate: u32,
    transform_bytes: u64,
    q8_bytes: u64,
    q8: Option<Q8SumPolicy>,
    mmq: selected::MmqLayouts,
}

impl Recipe {
    pub(super) fn from_prepared(
        prepared: &prepared::Prepared,
        q8: Option<Q8SumPolicy>,
        mmq: Option<&StreamMmq>,
    ) -> Option<Self> {
        // A missing actual table cannot become Known through a recipe. Check
        // this before cloning even the purely numeric matrix descriptors.
        prepared.selection.command.statistical_evidence()?;
        let leaves = if let Some(rows) = prepared.selection.packed_rows {
            if prepared.launches.len() != 1
                || prepared.launches[0].2 != rows
                || prepared.launches[0].3 != 0
            {
                return None;
            }
            Leaves::Packed(rows)
        } else {
            Leaves::Participants(prepared.launches.iter().map(|row| (row.2, row.3)).collect())
        };
        Some(Self {
            gate: prepared.gate_up.clone().into_boxed_slice(),
            down: prepared.down.clone().into_boxed_slice(),
            leaves,
            hidden: prepared.hidden,
            intermediate: prepared.intermediate,
            transform_bytes: prepared.transform_bytes,
            q8_bytes: prepared.q8_bytes,
            q8,
            mmq: selected::MmqLayouts::from_selected(
                mmq,
                prepared.selection.mmq_hit,
                &prepared.down,
                prepared.hidden,
                prepared.intermediate,
            )?,
        })
    }

    pub(in crate::backend::cuda::vnext_ops::transformer) fn project(
        &self,
        tokens: u64,
        participant_ranges: &[Range<u64>],
    ) -> Option<SelectedCommandCostEvidenceV1> {
        // Preserve actual prepare's per-owner launch bounds even when the
        // kernel consumes packed total M. Fresh core work is a complete cover.
        let mut end = 0;
        for range in participant_ranges {
            if range.start != end
                || range.end <= range.start
                || range.end - range.start > u64::from(u16::MAX)
            {
                return None;
            }
            end = range.end;
        }
        if participant_ranges.is_empty() || end != tokens {
            return None;
        }
        match &self.leaves {
            Leaves::Packed(rows) if u64::from(*rows) == tokens => {
                self.table(std::iter::once(*rows), tokens)
            }
            Leaves::Participants(leaves)
                if leaves.len() == participant_ranges.len()
                    && leaves
                        .iter()
                        .zip(participant_ranges)
                        .all(|(&(rows, start), range)| {
                            u64::from(rows) == range.end - range.start && start == range.start
                        }) =>
            {
                self.table(leaves.iter().map(|leaf| leaf.0), tokens)
            }
            _ => None,
        }
    }

    fn table(
        &self,
        rows: impl IntoIterator<Item = u32>,
        tokens: u64,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        selected::swiglu_planned(
            &self.gate,
            &self.down,
            rows,
            tokens,
            self.hidden,
            self.intermediate,
            self.transform_bytes,
            self.q8_bytes,
            self.q8,
            self.mmq,
            SloStructuredCostCapture::HostSettledV1,
        )
    }
}

#[cfg(test)]
mod tests;
