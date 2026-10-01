//! Input-only checks before any measured probe. Presence in this inventory is
//! an opportunity, not an observed owner, numerical qualification or permission.
use super::*;
use ferrum_interfaces::vnext::{DecodeContextBoundary, ExecutorDecodeContextCoverage};

#[derive(Debug, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ProbeInputCoverage {
    /// Configuration/model/domain upper bound, before the private probe cap.
    /// Conditional live policies and resources may admit fewer rows.
    pub configured_maximum_rows: usize,
    pub planned_rows: Vec<usize>,
    pub unprobed_rows: Vec<usize>,
    pub effective_context_tokens: usize,
    pub minimum_planned_decode_sequence_tokens: Option<u64>,
    pub maximum_planned_decode_sequence_tokens: Option<u64>,
    pub provider_nodes: usize,
    pub undeclared_provider_nodes: usize,
    pub provider_domains_shorter_than_configured: usize,
    /// Axis checks across the planned population. They do not claim every
    /// width/policy/route combination occurs, or that ordinary output reaches it.
    pub decode_boundaries: Vec<PlannedDecodeBoundary>,
}

#[derive(Debug, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct PlannedDecodeBoundary {
    pub boundary: DecodeContextBoundary,
    pub lower_sequence_planned: bool,
    pub upper_sequence_planned: bool,
}

impl ProbeInputCoverage {
    pub fn has_known_gaps(&self) -> bool {
        !self.unprobed_rows.is_empty()
            || self.provider_domains_shorter_than_configured != 0
            || self
                .decode_boundaries
                .iter()
                .any(|b| !b.lower_sequence_planned || !b.upper_sequence_planned)
    }

    pub fn has_undeclared_routes(&self) -> bool {
        self.provider_nodes == 0 || self.undeclared_provider_nodes != 0
    }

    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        self.planned_rows
            .capacity()
            .checked_add(self.unprobed_rows.capacity())?
            .checked_mul(std::mem::size_of::<usize>())?
            .checked_add(
                self.decode_boundaries
                    .capacity()
                    .checked_mul(std::mem::size_of::<PlannedDecodeBoundary>())?,
            )
    }
}

pub(super) fn retained_upper_bound(
    maximum_rows: usize,
    planned_cohorts: usize,
    declarations: &ExecutorDecodeContextCoverage,
) -> Result<usize> {
    // Reserve each array once, before inspecting the inventory. Include the
    // temporary frontier array while it coexists with the retained report.
    let boundaries = declarations.nodes.iter().try_fold(0usize, |n, node| {
        n.checked_add(node.coverage.known_boundaries().len())
    });
    maximum_rows
        .checked_mul(2 * std::mem::size_of::<usize>())
        .and_then(|n| {
            n.checked_add(boundaries?.checked_mul(std::mem::size_of::<PlannedDecodeBoundary>())?)
        })
        .and_then(|n| {
            n.checked_add(planned_cohorts.checked_mul(std::mem::size_of::<(u64, u64)>())?)
        })
        .ok_or_else(|| error("probe coverage retained capacity overflow"))
}

pub(super) fn inspect(
    cohorts: &[PreparedProbeCohort],
    prompts: &[usize],
    maximum_rows: usize,
    context: usize,
    declarations: &ExecutorDecodeContextCoverage,
) -> Result<ProbeInputCoverage> {
    let mut widths = Vec::with_capacity(maximum_rows);
    let mut frontiers = Vec::with_capacity(cohorts.len());
    for cohort in cohorts {
        if cohort.width == 0 || cohort.width > maximum_rows {
            return Err(error("probe coverage width exceeds declared capacity"));
        }
        if !widths.contains(&cohort.width) {
            widths.push(cohort.width);
        }
        let prompt = *prompts
            .get(cohort.template)
            .ok_or_else(|| error("probe coverage template is missing"))?;
        // Prefill emits token one. A decode with `generated` output tokens
        // already present uses sequence frontier prompt + generated, including
        // its current input token. Prepared prefixes enter at their release.
        let generated = match cohort.prefix {
            PrefixKind::Ordinary => 1,
            _ => cohort
                .maximum_output
                .get()
                .checked_sub(cohort.suffix_tokens)
                .ok_or_else(|| error("probe coverage prefix/output differs"))?,
        };
        if generated < cohort.maximum_output.get() {
            let start = prompt
                .checked_add(generated)
                .ok_or_else(|| error("probe coverage frontier overflow"))?;
            let end = prompt
                .checked_add(cohort.maximum_output.get() - 1)
                .ok_or_else(|| error("probe coverage frontier overflow"))?;
            if start == 0 || start > end || end >= context {
                return Err(error("probe coverage frontier exceeds declared context"));
            }
            let interval = (start as u64, end as u64);
            if !frontiers.contains(&interval) {
                frontiers.push(interval);
            }
        }
    }
    widths.sort_unstable();
    let mut unprobed_rows = Vec::with_capacity(maximum_rows);
    unprobed_rows.extend((1..=maximum_rows).filter(|n| widths.binary_search(n).is_err()));
    let boundary_capacity = declarations
        .nodes
        .iter()
        .try_fold(0usize, |n, node| {
            n.checked_add(node.coverage.known_boundaries().len())
        })
        .ok_or_else(|| error("probe coverage boundary capacity overflow"))?;
    let mut boundaries: Vec<PlannedDecodeBoundary> = Vec::with_capacity(boundary_capacity);
    let contains = |point| frontiers.iter().any(|&(lo, hi)| lo <= point && point <= hi);
    let mut undeclared = 0;
    let mut shorter = 0;
    for node in &declarations.nodes {
        undeclared += usize::from(!node.coverage.is_complete());
        shorter += usize::from(
            node.coverage
                .maximum_sequence_tokens()
                .is_some_and(|n| n.get() < context.saturating_sub(1) as u64),
        );
        for boundary in node.coverage.known_boundaries() {
            let first = boundary.first_sequence_tokens.get();
            // A complete private probe retains room for its final output token.
            if first >= context as u64
                || first <= 1
                || boundaries.iter().any(|b| b.boundary == *boundary)
            {
                continue;
            }
            boundaries.push(PlannedDecodeBoundary {
                boundary: *boundary,
                lower_sequence_planned: contains(first - 1),
                upper_sequence_planned: contains(first),
            });
        }
    }
    boundaries.sort_unstable_by_key(|b| b.boundary);
    Ok(ProbeInputCoverage {
        configured_maximum_rows: maximum_rows,
        planned_rows: widths,
        unprobed_rows,
        effective_context_tokens: context,
        minimum_planned_decode_sequence_tokens: frontiers.iter().map(|p| p.0).min(),
        maximum_planned_decode_sequence_tokens: frontiers.iter().map(|p| p.1).max(),
        provider_nodes: declarations.nodes.len(),
        undeclared_provider_nodes: undeclared,
        provider_domains_shorter_than_configured: shorter,
        decode_boundaries: boundaries,
    })
}

#[cfg(test)]
mod tests;
