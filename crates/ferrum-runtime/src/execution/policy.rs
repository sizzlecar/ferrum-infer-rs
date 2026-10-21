use super::{
    canonical_fingerprint, invalid_plan, CapabilityCatalog, CompletionRetentionSpec,
    ContractVersion, DynamicStorageProfile, ExecutionWeightPlan, PlanNodeResolution,
    PreparedModelFamily, ReusableExecutionPolicy, Serialize, TrustedExecutionWeightPlan,
    VNextError,
};
use crate::{CheckpointCapacityPolicy, ExecutionDeterminismRequirement};
use ferrum_types::AttentionExecutionPolicy;

/// Work dimensions are part of the compiled capacity contract. Tensor jobs
/// have no token ceiling; their exact extents come from compiled port bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeWorkLimits {
    Tokens { maximum_scheduled_tokens: u64 },
    FixedTensor,
}

impl<'de> serde::Deserialize<'de> for RuntimeWorkLimits {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A struct variant, unlike a Serde internally tagged unit variant,
        // rejects fields from another work domain as well as unknown fields.
        #[derive(serde::Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Tokens { maximum_scheduled_tokens: u64 },
            FixedTensor {},
        }
        let limits = match Wire::deserialize(deserializer)? {
            Wire::Tokens {
                maximum_scheduled_tokens,
            } => Self::Tokens {
                maximum_scheduled_tokens,
            },
            Wire::FixedTensor {} => Self::FixedTensor,
        };
        limits.validate().map_err(serde::de::Error::custom)?;
        Ok(limits)
    }
}

impl RuntimeWorkLimits {
    pub fn maximum_scheduled_tokens(self) -> Result<u64, VNextError> {
        match self {
            Self::Tokens {
                maximum_scheduled_tokens,
            } => {
                super::validate_scheduled_token_ceiling(maximum_scheduled_tokens)?;
                Ok(maximum_scheduled_tokens)
            }
            Self::FixedTensor => Err(invalid_plan("tensor work has no token dimension")),
        }
    }

    pub fn validate(self) -> Result<(), VNextError> {
        match self {
            Self::Tokens { .. } => self.maximum_scheduled_tokens().map(|_| ()),
            Self::FixedTensor => Ok(()),
        }
    }

    pub const fn is_tensor(self) -> bool {
        matches!(self, Self::FixedTensor)
    }
}

/// Typed policy selected before planning. Memory capacity is part of the
/// public policy contract so a plan cannot depend on an undocumented env var.
pub trait RuntimePolicy:
    Clone + Send + Sync + Serialize + serde::de::DeserializeOwned + std::fmt::Debug + 'static
{
    fn version(&self) -> ContractVersion;

    /// Policy memory ceiling before the explicit reserve is subtracted. It may
    /// be lower than, but never exceed, the raw device capacity.
    fn memory_capacity_bytes(&self) -> u64;

    fn memory_reserve_bytes(&self) -> u64;

    /// A non-zero protocol ceiling only. Planning must not reserve, claim,
    /// iterate, or materialize resources up to this value.
    fn maximum_active_sequences(&self) -> u32;

    /// The compiled work domain and its logical limits. A token ceiling is not
    /// installed capacity; tensor extents come from compiled ports. Admission
    /// evaluates the exact step against live backing before provider encoding.
    fn work_limits(&self) -> RuntimeWorkLimits;

    fn maximum_scheduled_tokens(&self) -> Result<u64, VNextError> {
        self.work_limits().maximum_scheduled_tokens()
    }

    /// Provider-family policy sealed into the plan fingerprint. Physical
    /// kernel variants remain an invocation-time provider decision.
    fn attention_execution_policy(&self) -> AttentionExecutionPolicy;

    /// Minimum repeatability and reusable-execution equivalence that every
    /// selected operation provider must prove.
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement;

    /// Ordered, non-empty allowlist used by planning after intersecting every
    /// selected provider requirement with the concrete runtime offers.
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile];

    /// Optional reusable-execution capacity policy selected before planning.
    /// Core treats class identities as opaque and owns their memory derivation.
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy>;

    /// Optional physical State-pool growth permission. Runtime aggregate
    /// charging and foreground protection remain separate requirements.
    fn checkpoint_capacity_policy(&self) -> Option<&CheckpointCapacityPolicy> {
        None
    }

    fn validate(&self) -> Result<(), VNextError>;
}

pub fn canonical_runtime_policy_fingerprint<P: RuntimePolicy>(
    policy: &P,
) -> Result<String, VNextError> {
    policy.validate()?;
    canonical_fingerprint(policy, "fingerprint validated runtime policy")
}

pub struct PlanBuildRequest<'a, P: RuntimePolicy> {
    pub(super) family: &'a PreparedModelFamily,
    pub(super) capabilities: &'a CapabilityCatalog,
    pub(super) policy: &'a P,
    pub(super) node_resolutions: Vec<PlanNodeResolution>,
    pub(super) completion_retention: CompletionRetentionSpec,
    pub(super) execution_weights: TrustedExecutionWeightPlan,
}

impl<'a, P: RuntimePolicy> PlanBuildRequest<'a, P> {
    pub fn new(
        family: &'a PreparedModelFamily,
        capabilities: &'a CapabilityCatalog,
        policy: &'a P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<Self, VNextError> {
        if node_resolutions.is_empty() {
            return Err(invalid_plan("plan build request has no node resolutions"));
        }
        policy.validate()?;
        Ok(Self {
            family,
            capabilities,
            policy,
            node_resolutions,
            completion_retention: CompletionRetentionSpec::default(),
            execution_weights: TrustedExecutionWeightPlan::identity(family)?,
        })
    }

    pub fn with_completion_retention(
        mut self,
        completion_retention: CompletionRetentionSpec,
    ) -> Result<Self, VNextError> {
        let produced_values = self
            .family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| block.nodes.iter())
            .flat_map(|node| node.outputs.iter())
            .collect::<std::collections::BTreeSet<_>>();
        if completion_retention
            .values()
            .iter()
            .any(|value_id| !produced_values.contains(value_id))
        {
            return Err(invalid_plan(
                "completion retention must reference a semantic node output",
            ));
        }
        self.completion_retention = completion_retention;
        Ok(self)
    }

    pub fn with_execution_weights(
        mut self,
        execution_weights: TrustedExecutionWeightPlan,
    ) -> Result<Self, VNextError> {
        execution_weights.validate_against_catalog(self.family, self.capabilities)?;
        self.execution_weights = execution_weights;
        Ok(self)
    }

    pub fn family(&self) -> &PreparedModelFamily {
        self.family
    }

    pub fn capabilities(&self) -> &CapabilityCatalog {
        self.capabilities
    }

    pub fn policy(&self) -> &P {
        self.policy
    }

    pub fn execution_weights(&self) -> &ExecutionWeightPlan {
        self.execution_weights.plan()
    }
}
