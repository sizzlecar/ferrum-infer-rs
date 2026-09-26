use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StructuredWaveRoleV2 {
    OrdinaryDecode,
    DecodeWithNoGeneratedHistory,
    Prefill,
    Mixed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StructuredProductV2 {
    GreedyToken,
    FullLogits,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StructuredTemplateV2 {
    Ordered([u8; 32]),
    ProviderGrouped([u8; 32]),
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerKeyV2 {
    pub rows: u32,
    pub role: StructuredWaveRoleV2,
    pub product: StructuredProductV2,
    pub readback: CoreReadbackRoute,
    pub provider_template: StructuredTemplateV2,
    /// Sorted (real algorithm signature, kind) keys, excluding numeric work.
    pub algorithm_domain: [u8; 32],
    /// Installed policy identities and declared decoder bounds, excluding state.
    pub installed_policy: [u8; 32],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredCoverageV2 {
    /// Positions which a future unresolved query may vary. Sorted and unique.
    pub pending_eligible_positions: Vec<u32>,
    pub authorized_pending_constraints: Vec<HostPendingConstraintV2>,
    /// These are required observed challenges, not per-bucket model populations.
    pub pending_counts: Vec<u32>,
    pub length_counts: Vec<u32>,
    pub pending_positions: Vec<u32>,
    pub length_positions: Vec<u32>,
    /// (pending count, Length count), observed jointly in one actual wave.
    pub joint_counts: Vec<(u32, u32)>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredScopeV2 {
    pub owner: StructuredOwnerKeyV2,
    pub coverage: StructuredCoverageV2,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredPhaseV2 {
    Fit,
    Residual,
    Qualification,
}
#[derive(Debug, Clone)]
pub struct StructuredSourceContractV2 {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub membership_rule: [u8; 32],
    pub cohort_manifest: [u8; 32],
    /// Counts refer to pre-execution reserved members, never all offered waves.
    pub phase_members: [usize; 3],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredMemberBindingV2 {
    pub rule_signature: [u8; 32],
    pub offered_ordinal: u64,
    pub member_ordinal: u64,
    pub phase: StructuredPhaseV2,
}
#[derive(Debug, Clone)]
pub struct StructuredObservationV2<I> {
    pub source: [u8; 32],
    pub protocol: [u8; 32],
    /// Original global accepted FIFO ordinal, not the member coordinate.
    pub ordinal: u64,
    pub membership: StructuredMemberBindingV2,
    pub call_id: u64,
    pub fingerprint: ExecutionFingerprint,
    pub input: I,
    pub boundary: CostBoundary,
    pub outcome: WaveObservationOutcome,
    pub observed_at_ns: u64,
    pub wall_ns: u64,
}
/// Frozen empirical uncertainty. None of these components is a hard future bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StructuredUncertaintyV2 {
    /// Largest positive error on the complete fit population at fit freeze.
    pub fit_error_floor_ns: u64,
    /// Nearest-rank q99 positive error from the independent residual population.
    pub residual_ns: u64,
    /// max(fit_error_floor_ns, residual_ns), before the declared static margin.
    pub effective_residual_ns: u64,
    pub static_margin_ns: u64,
}
#[derive(Debug, Clone, Copy)]
pub struct StructuredPredictionV2 {
    pub fitted_lower_ns: u64,
    pub fitted_upper_ns: u64,
    /// Independent residual q99; does not include the fit floor or static margin.
    pub residual_ns: u64,
    /// Largest observed positive fit error, not a statistical confidence bound.
    pub fit_error_floor_ns: u64,
    /// max(fit_error_floor_ns, residual_ns), before the declared static margin.
    pub effective_residual_ns: u64,
    pub planning_ns: u64,
    /// Original model epoch. Engine must subtract imported.model_now_ns(local_now).
    pub valid_until_ns: u64,
    pub fit_samples: usize,
    pub residual_samples: usize,
    pub identified_rank: usize,
}
