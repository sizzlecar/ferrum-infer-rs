//! Bounded aggregation of an already inspected typed report. This is descriptive
//! evidence only; its extrema do not assert joint support or branch qualification.
use super::*;
use ferrum_interfaces::execution_cost::HostTerminalExpectationV1;
use std::collections::{BTreeMap, BTreeSet};

const MAX_DOMAINS: usize = 128;
const MAX_COORDINATES: usize = 65_536;
const MAX_JOINT_PAIRS: usize = 16_384;
const MAX_UNKNOWN_REASONS: usize = 32;

#[derive(Serialize)]
pub(in crate::commands::calibrate_slo) struct DiscoverySummaryV2 {
    model_revision: &'static str,
    scope: &'static str,
    pub(in crate::commands::calibrate_slo) collection_completed: bool,
    wave_reports_seen: u64,
    known_input_waves: u64,
    unknown_input_reports: u64,
    unknown_reasons: BTreeMap<String, u64>,
    inventory_truncated: bool,
    omitted_known_waves: u64,
    domains: Vec<DomainSummary>,
    #[serde(skip)]
    coordinates: usize,
    #[serde(skip)]
    joint_pairs: usize,
}
#[derive(Serialize)]
struct DomainSummary {
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
    waves: u64,
    pending_counts: BTreeSet<u32>,
    length_counts: BTreeSet<u32>,
    pending_positions: BTreeSet<u32>,
    length_positions: BTreeSet<u32>,
    joint_counts: BTreeSet<(u32, u32)>,
    /// Same ordered numeric axes as each original raw V2 discovery report.
    /// Separate extrema must never be interpreted as observed joint corners.
    basis_ranges: Vec<Range<f64>>,
    support_ranges: Vec<Range<u64>>,
}
#[derive(Serialize)]
struct Range<T> {
    minimum: T,
    maximum: T,
}
impl<T: Copy + PartialOrd> Range<T> {
    fn new(value: T) -> Self {
        Self {
            minimum: value,
            maximum: value,
        }
    }
    fn observe(&mut self, value: T) {
        if value < self.minimum {
            self.minimum = value;
        }
        if value > self.maximum {
            self.maximum = value;
        }
    }
}
impl DiscoverySummaryV2 {
    pub(in crate::commands::calibrate_slo) fn new() -> Self {
        Self {
            model_revision: MODEL_REVISION_V2,
            scope: "independent actual waves only; observed Length expectations and UTF8 pending are not qualification, retry counts, or joint numeric support",
            collection_completed: false,
            wave_reports_seen: 0,
            known_input_waves: 0,
            unknown_input_reports: 0,
            unknown_reasons: BTreeMap::new(),
            inventory_truncated: false,
            omitted_known_waves: 0,
            domains: Vec::new(),
            coordinates: 0,
            joint_pairs: 0,
        }
    }
    pub(in crate::commands::calibrate_slo) fn observe(
        &mut self,
        phase: crate::commands::calibrate_slo::report::Phase,
        report: &DiscoveryReportV2,
    ) {
        if !matches!(
            phase,
            crate::commands::calibrate_slo::report::Phase::Discovery
        ) {
            return;
        }
        self.wave_reports_seen += 1;
        let DiscoveryInputV2::Known {
            owner,
            domain_signature,
            basis,
            support,
            physical_host_rows,
        } = &report.input
        else {
            self.unknown_input_reports += 1;
            if let DiscoveryInputV2::Unknown { reason } = &report.input {
                let reason = format!("{reason:?}");
                if self.unknown_reasons.contains_key(&reason)
                    || self.unknown_reasons.len() < MAX_UNKNOWN_REASONS
                {
                    *self.unknown_reasons.entry(reason).or_default() += 1;
                } else {
                    self.inventory_truncated = true;
                }
            }
            return;
        };
        self.known_input_waves += 1;
        let index = match self
            .domains
            .iter()
            .position(|d| d.domain_signature == *domain_signature)
        {
            Some(index) => index,
            None => {
                let Some(next) = self
                    .coordinates
                    .checked_add(basis.len())
                    .and_then(|n| n.checked_add(support.len()))
                else {
                    return self.omit();
                };
                if self.domains.len() == MAX_DOMAINS || next > MAX_COORDINATES {
                    return self.omit();
                }
                self.coordinates = next;
                self.domains.push(DomainSummary {
                    owner: owner.clone(),
                    domain_signature: *domain_signature,
                    waves: 0,
                    pending_counts: BTreeSet::new(),
                    length_counts: BTreeSet::new(),
                    pending_positions: BTreeSet::new(),
                    length_positions: BTreeSet::new(),
                    joint_counts: BTreeSet::new(),
                    basis_ranges: basis.iter().copied().map(Range::new).collect(),
                    support_ranges: support.iter().copied().map(Range::new).collect(),
                });
                self.domains.len() - 1
            }
        };
        let d = &mut self.domains[index];
        if d.owner != *owner
            || d.basis_ranges.len() != basis.len()
            || d.support_ranges.len() != support.len()
        {
            return self.omit();
        }
        let pending_count = physical_host_rows
            .iter()
            .filter(|r| r.pending_decoded_utf8)
            .count() as u32;
        let length_count = physical_host_rows
            .iter()
            .filter(|r| r.terminal_expectation == HostTerminalExpectationV1::LengthBoundary)
            .count() as u32;
        let joint = (pending_count, length_count);
        if !d.joint_counts.contains(&joint) {
            if self.joint_pairs == MAX_JOINT_PAIRS {
                return self.omit();
            }
            self.joint_pairs += 1;
            d.joint_counts.insert(joint);
        }
        d.waves += 1;
        d.pending_counts.insert(pending_count);
        d.length_counts.insert(length_count);
        d.pending_positions.extend(
            physical_host_rows
                .iter()
                .filter(|r| r.pending_decoded_utf8)
                .map(|r| r.physical_position),
        );
        d.length_positions.extend(
            physical_host_rows
                .iter()
                .filter(|r| r.terminal_expectation == HostTerminalExpectationV1::LengthBoundary)
                .map(|r| r.physical_position),
        );
        for (range, value) in d.basis_ranges.iter_mut().zip(basis) {
            range.observe(*value);
        }
        for (range, value) in d.support_ranges.iter_mut().zip(support) {
            range.observe(*value);
        }
    }
    fn omit(&mut self) {
        self.inventory_truncated = true;
        self.omitted_known_waves += 1;
    }
}

#[cfg(test)]
mod tests;
