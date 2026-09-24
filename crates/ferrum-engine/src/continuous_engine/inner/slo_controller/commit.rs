//! Host commit fences retain exact ownership even without a cost signature.
use super::*;
use ferrum_interfaces::execution_cost::{
    host_history_cost_signature, ExpectedWaveInput, ExpectedWaveParticipant, ExpectedWorkSelection,
};

#[derive(Clone, Copy)]
pub(in crate::continuous_engine::inner) enum ControllerCommitFence<'a> {
    Cost(&'a ExpectedWaveParticipant),
    Completion(&'a ExpectedWorkSelection),
}

impl ControllerCommitFence<'_> {
    pub fn request_id(&self) -> &RequestId {
        match self {
            Self::Cost(row) => &row.request_id,
            Self::Completion(row) => &row.request_id,
        }
    }
    pub fn input(&self) -> &ExpectedWaveInput {
        match self {
            Self::Cost(row) => &row.input,
            Self::Completion(row) => &row.input,
        }
    }
    pub fn matches(&self, sequence: &SequenceState) -> bool {
        let (owner, generation) = match self {
            Self::Cost(row) => {
                if row.host.request_id != sequence.request_id
                    || sequence.cost_policy_signature.map(|policy| {
                        host_history_cost_signature(policy, sequence.generated_tokens.len() as u64)
                    }) != row.host.output_policy_signature
                {
                    return false;
                }
                (row.host.owner_incarnation, row.host.work_generation)
            }
            Self::Completion(row) => (row.owner_incarnation.get(), row.work_generation.get()),
        };
        self.request_id() == &sequence.request_id
            && sequence.cost_frontier.is_some_and(|frontier| {
                frontier.owner_incarnation.get() == owner
                    && frontier.work_generation.get() == generation
            })
    }
}
