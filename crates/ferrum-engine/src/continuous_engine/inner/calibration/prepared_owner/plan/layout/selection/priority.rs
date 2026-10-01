//! Input-only coverage order, fixed before geometry, reservations and sealing.
//! Priority is an opportunity to collect a complete source, never qualification.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CoveragePriority {
    pub policy: u8,
    template: usize,
    route: u8,
}

impl CoveragePriority {
    pub(super) fn same_policy_route(self, other: Self) -> bool {
        self.policy == other.policy && self.route == other.route
    }

    pub fn from_cases(indices: &[usize], cases: &[Case]) -> Result<Self> {
        indices
            .iter()
            .map(|&index| {
                let case = &cases[index];
                let configured = case.preset == SloAutomaticCostProbeSamplingPresetV1::Configured;
                let actual = case.route == CalibrationDecodeRoute::Actual;
                Self {
                    // Configured retains the original EOS, stop, sampler and
                    // output policy. FullLogits is an auxiliary execution
                    // route even when it retains those host policy fields.
                    // Native prefix preparation is an input acquisition
                    // contract, not a less important installed product policy.
                    policy: if configured && actual {
                        0
                    } else if configured {
                        1
                    } else {
                        2
                    },
                    template: case.template,
                    route: u8::from(!actual),
                }
            })
            .min()
            .ok_or_else(|| error("coverage priority has no checked original case"))
    }
}

pub(super) fn order_populations(
    candidates: &mut [PopulationCandidate],
    inputs: &[Vec<CheckedInputFacts>],
) {
    candidates.sort_unstable_by_key(|candidate| {
        (
            candidate.coverage.policy,
            candidate.decode_width_tier,
            candidate.declared_work,
            candidate.input_priority,
            candidate.population_index,
        )
    });
    for index in 0..candidates.len() {
        candidates[index].coverage_round = checked_coverage_round(
            &candidates[index].candidates,
            candidates[..index]
                .iter()
                .filter(|earlier| earlier.coverage.policy == candidates[index].coverage.policy)
                .map(|earlier| earlier.candidates.as_slice()),
            inputs,
        );
    }
    candidates.sort_unstable_by_key(|candidate| {
        (
            candidate.coverage.policy,
            candidate.coverage_round,
            candidate.decode_width_tier,
            candidate.declared_work,
            candidate.input_priority,
            candidate.population_index,
        )
    });
}

pub(super) fn order_batches(candidates: &mut [BatchCandidate], inputs: &[Vec<CheckedInputFacts>]) {
    // Visit complete sources by their declared width tier and work before assigning the
    // checked execution-role rounds. This same rule precedes geometry above
    // and both retention reservations and original source execution below.
    // No measured duration, EOS result or fitted coefficient enters this key.
    candidates.sort_unstable_by_key(BatchCandidate::order_key);
    for index in 0..candidates.len() {
        candidates[index].coverage_round = checked_coverage_round(
            &candidates[index].batch.representative_case_indices,
            candidates[..index]
                .iter()
                .filter(|earlier| earlier.coverage.policy == candidates[index].coverage.policy)
                .map(|earlier| earlier.batch.representative_case_indices.as_slice()),
            inputs,
        );
    }
    candidates.sort_unstable_by_key(BatchCandidate::order_key);
}

// Coverage is scheduling opportunity only. Keep the full original population
// key, numerical endpoints and phase membership elsewhere; these comparisons
// neither merge model identities nor grant prediction/execution authority.
fn same_role(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    left.owner.role == right.owner.role
        && left.owner.product == right.owner.product
        && left.owner.readback == right.owner.readback
}

fn same_channel(left: &CheckedInputFacts, right: &CheckedInputFacts) -> bool {
    let same_policy = match (left.homogeneous_host_policy, right.homogeneous_host_policy) {
        (Some(left), Some(right)) => left == right,
        // Heterogeneous policy sequences retain their exact ordered identity.
        (None, None) => left.owner.installed_policy == right.owner.installed_policy,
        _ => false,
    };
    same_role(left, right)
        && same_policy
        && left.owner.installed_policy == right.owner.installed_policy
        && left.owner.algorithm_domain == right.owner.algorithm_domain
        && left.owner.provider_template == right.owner.provider_template
}

/// The lightest complete source for each checked execution role gets the
/// first round. Another installed host policy or algorithm for that role is
/// a later round: several API projections of Decode must not consume the
/// complete Prefill source's opportunity. Every exact policy/channel identity
/// remains separate for qualification and prediction. Repeat
/// coverage of the same complete channels follows their first coverage.
///
/// Borrow the already charged input facts: no additional domain catalogue,
/// hash, payload allocation, desired-product label or observed output is used.
fn checked_coverage_round<'a>(
    cases: &[usize],
    earlier: impl Iterator<Item = &'a [usize]> + Clone,
    inputs: &[Vec<CheckedInputFacts>],
) -> usize {
    let facts = || cases.iter().flat_map(|&index| &inputs[index]);
    let first_role = facts().any(|fact| {
        earlier.clone().all(|previous| {
            !previous
                .iter()
                .flat_map(|&index| &inputs[index])
                .any(|old| same_role(old, fact))
        })
    });
    if first_role {
        return 0;
    }
    1 + earlier
        .filter(|previous| {
            facts().all(|fact| {
                previous
                    .iter()
                    .flat_map(|&index| &inputs[index])
                    .any(|old| same_channel(old, fact))
            })
        })
        .count()
}
