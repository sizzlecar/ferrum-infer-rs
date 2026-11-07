//! Selected-provider promises for same-plan accepted-prefix settlement.
use crate::{DynamicStorageProfile, ResolvedValueRole};
use serde::{Deserialize, Serialize};

/// This is an implementation promise, not inferred from a stateless signature.
/// Prefix-causal nodes promise that each output/state prefix is independent of
/// later token rows. Terminal reductions instead require a compiler dependency
/// proof. Non-token inputs that affect state are immutable for the sequence;
/// declared output-only inputs cannot affect state. Unsupported preserves old behavior.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderCandidateStateEffects {
    #[default]
    Unsupported,
    PrefixCausal {
        ports: Vec<ProviderCandidateStatePort>,
    },
    /// May reduce across token rows, but has no persistent state effects or
    /// hidden persistent workspace. The compiler must prove that its outputs
    /// (including aliased writes) cannot reach any persistent state operation.
    /// This is not a promise that a reduced output is prefix-causal.
    TerminalOnly,
}
impl ProviderCandidateStateEffects {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderCandidateStatePort {
    /// Original state is read-only. Output is exactly [2] + state shape:
    /// states after N-1 and N rows, including original state for slot0 at N=1.
    TailTwo {
        input_ordinal: u32,
        output_ordinal: u32,
        storage_profile: DynamicStorageProfile,
    },
    /// Writes only the current work span [P,P+N); prefix [0,P) is unchanged.
    /// Each written row is independent of future rows in the current span.
    PrefixAppendOnly {
        role: ResolvedValueRole,
        ordinal: u32,
        storage_profile: DynamicStorageProfile,
    },
}
