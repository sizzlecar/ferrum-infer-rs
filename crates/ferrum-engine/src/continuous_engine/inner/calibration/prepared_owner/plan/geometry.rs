//! Input geometry required before choosing numerical populations. This is not
//! a declaration that these routes exist or that any model can predict them.
use super::*;
use ferrum_interfaces::vnext::ExecutorDecodeContextCoverage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(super) struct DecodeInterval {
    pub first_sequence_tokens: u32,
    pub last_sequence_tokens: u32,
}

impl DecodeInterval {
    /// Every post-release decode in this interval must fit in the original
    /// request. Prefill emits output one; the final output still needs room.
    /// The caller renders a real product prompt inside this token window.
    pub fn prompt_window(
        self,
        release_generated: NonZeroUsize,
        maximum_output: NonZeroUsize,
    ) -> Option<(NonZeroUsize, NonZeroUsize)> {
        let release = release_generated.get();
        let output = maximum_output.get();
        if release >= output {
            return None;
        }
        let minimum = (self.last_sequence_tokens as usize)
            .saturating_sub(output - 1)
            .max(1);
        let maximum = (self.first_sequence_tokens as usize).checked_sub(release)?;
        (minimum <= maximum).then_some((NonZeroUsize::new(minimum)?, NonZeroUsize::new(maximum)?))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct ProbeGeometryRequirements {
    pub configured_maximum_rows: usize,
    pub probe_maximum_rows: usize,
    pub intervals: Vec<DecodeInterval>,
    pub sequence_tokens: Vec<u32>,
    pub provider_nodes: usize,
    pub undeclared_provider_nodes: usize,
    pub provider_domains_shorter_than_configured: usize,
}

impl ProbeGeometryRequirements {
    pub fn new(
        configured_rows: usize,
        probe_rows: usize,
        context: usize,
        ordinary_sequence_tokens: usize,
        declarations: &ExecutorDecodeContextCoverage,
        maximum_retained_bytes: usize,
    ) -> Result<Self> {
        if probe_rows == 0 || probe_rows > configured_rows || ordinary_sequence_tokens < 2 {
            return Err(error("invalid automatic probe geometry domain"));
        }
        let last = context
            .checked_sub(1)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n >= 2)
            .ok_or_else(|| error("probe geometry context cannot fit a complete request"))?;
        let ordinary = u32::try_from(ordinary_sequence_tokens)
            .ok()
            .filter(|n| *n <= last)
            .ok_or_else(|| error("probe geometry ordinary prompt exceeds actual context"))?;
        // Bound storage before allocating, including duplicate provider
        // declarations. A duplicate never creates more probe work.
        let capacity = declarations
            .nodes
            .iter()
            .try_fold(2usize, |n, node| {
                n.checked_add(node.coverage.known_boundaries().len())
            })
            .ok_or_else(|| error("probe geometry boundary capacity overflow"))?;
        let bytes = capacity
            .checked_mul(std::mem::size_of::<DecodeInterval>() + 2 * std::mem::size_of::<u32>())
            .ok_or_else(|| error("probe geometry retained capacity overflow"))?;
        if bytes > maximum_retained_bytes {
            return Err(error("probe geometry exceeds shared retained capacity"));
        }
        let mut intervals = Vec::with_capacity(capacity);
        intervals.push(DecodeInterval {
            first_sequence_tokens: ordinary,
            last_sequence_tokens: ordinary,
        });
        intervals.push(DecodeInterval {
            first_sequence_tokens: last,
            last_sequence_tokens: last,
        });
        let mut undeclared = 0;
        let mut shorter = 0;
        for node in &declarations.nodes {
            undeclared += usize::from(!node.coverage.is_complete());
            shorter += usize::from(
                node.coverage
                    .maximum_sequence_tokens()
                    .is_some_and(|n| n.get() < u64::from(last)),
            );
            for boundary in node.coverage.known_boundaries() {
                let first = boundary.first_sequence_tokens.get();
                if first < 2 || first > u64::from(last) {
                    // Sequence one cannot decode a nonempty prompt. A switch
                    // beyond last would leave no room for the final output.
                    continue;
                }
                intervals.push(DecodeInterval {
                    first_sequence_tokens: (first as u32 - 1).max(2),
                    last_sequence_tokens: first as u32,
                });
            }
        }
        intervals.sort_unstable_by_key(|i| (i.first_sequence_tokens, i.last_sequence_tokens));
        intervals.dedup();
        let mut sequence_tokens = Vec::with_capacity(capacity * 2);
        for interval in &intervals {
            sequence_tokens.extend([
                interval.first_sequence_tokens,
                interval.last_sequence_tokens,
            ]);
        }
        sequence_tokens.sort_unstable();
        sequence_tokens.dedup();
        Ok(Self {
            configured_maximum_rows: configured_rows,
            probe_maximum_rows: probe_rows,
            intervals,
            sequence_tokens,
            provider_nodes: declarations.nodes.len(),
            undeclared_provider_nodes: undeclared,
            provider_domains_shorter_than_configured: shorter,
        })
    }

    /// Integers are deliberate: a kernel selector may change at B3 or B5 even
    /// when its neighbouring powers of two share a checked numerical family.
    pub fn points(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        (1..=self.probe_maximum_rows).flat_map(|rows| {
            self.sequence_tokens
                .iter()
                .copied()
                .map(move |tokens| (rows, tokens))
        })
    }

    pub fn point_count(&self) -> Result<usize> {
        self.probe_maximum_rows
            .checked_mul(self.sequence_tokens.len())
            .ok_or_else(|| error("probe geometry point count overflow"))
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.intervals
            .capacity()
            .checked_mul(std::mem::size_of::<DecodeInterval>())?
            .checked_add(
                self.sequence_tokens
                    .capacity()
                    .checked_mul(std::mem::size_of::<u32>())?,
            )
    }
}

#[cfg(test)]
mod tests;
