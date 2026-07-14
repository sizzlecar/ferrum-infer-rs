use super::{
    canonical_fingerprint, invalid_plan, CapabilityCatalog, ContractVersion, DynamicStorageProfile,
    PlanNodeResolution, PreparedModelFamily, Serialize, VNextError,
};

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

    /// A non-zero ceiling for tokens selected into one scheduler step. This is
    /// not installed capacity: admission still evaluates the exact step shape
    /// against then-live backing before any provider work is encoded.
    fn maximum_scheduled_tokens(&self) -> u64;

    /// Ordered, non-empty allowlist used by planning after intersecting every
    /// selected provider requirement with the concrete runtime offers.
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile];

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
        })
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
}
