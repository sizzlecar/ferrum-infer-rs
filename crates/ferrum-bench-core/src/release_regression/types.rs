use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeArea {
    Download,
    Template,
    Termination,
    Structured,
    Tools,
    Scheduler,
    Kv,
    /// Proven submission/completion lifecycle changes without operator changes.
    BackendSubmission,
    /// Checkpoint decoding, physical layout construction and source validation
    /// before model execution; does not imply device operator changes.
    WeightMaterialization,
    Kernel,
    Architecture,
    Build,
    Observability,
    /// Proven sink, metadata-adapter or process-accounting changes that do not
    /// alter model-side event acquisition. Unknown instrumentation keeps the
    /// broader Observability area and its real-model obligations.
    ObservabilityContract,
    Validation,
}
impl ChangeArea {
    pub const ALL: [Self; 15] = [
        Self::Download,
        Self::Template,
        Self::Termination,
        Self::Structured,
        Self::Tools,
        Self::Scheduler,
        Self::Kv,
        Self::BackendSubmission,
        Self::WeightMaterialization,
        Self::Kernel,
        Self::Architecture,
        Self::Build,
        Self::Observability,
        Self::ObservabilityContract,
        Self::Validation,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathImpact {
    pub path: String,
    pub areas: Vec<ChangeArea>,
    pub reason: String,
    /// A content proof may narrow this path to independent execution routes.
    /// Absent evidence retains every route; an empty list is invalid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_paths: Option<Vec<String>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Impact {
    pub areas: Vec<ChangeArea>,
    pub paths: Vec<PathImpact>,
    pub unknown_paths: Vec<String>,
    pub product_contract_changed: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    PullRequest,
    Release,
    Nightly,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Entrypoint {
    Run,
    ServeSync,
    ServeStream,
}

/// Independent dimensions of declared execution coverage. Stored weight precision
/// does not assert a universal compute or KV dtype. The execution path can distinguish
/// accelerator implementations without using a particular machine's identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionTarget {
    pub architecture: String,
    pub protocol: ModelOutputProtocol,
    pub precision: String,
    pub backend: Backend,
    pub execution_path: String,
}

/// An estimate for one selected profile's planned checks, not a measured pass.
/// Fields include paid setup separately from model download/loading and execution.
/// Missing prices and missing estimates must remain visible as unknown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostEstimate {
    pub download_ms: u64,
    pub load_ms: u64,
    pub check_ms: u64,
    pub setup_ms: u64,
    pub cleanup_ms: u64,
    /// Estimated lease time on the priced resource, including any charged idle time.
    /// Downloads prepared before acquiring that resource must not be billed at its rate.
    pub billable_ms: u64,
    pub hourly_rate_microunits: Option<u64>,
    pub fixed_cost_microunits: Option<u64>,
    pub currency: Option<String>,
}
impl CostEstimate {
    pub fn total_ms(&self) -> Option<u64> {
        [
            self.download_ms,
            self.load_ms,
            self.check_ms,
            self.setup_ms,
            self.cleanup_ms,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
    }
    pub fn cost_microunits(&self) -> Option<u64> {
        if self
            .currency
            .as_deref()
            .is_none_or(|currency| currency.trim().is_empty())
        {
            return None;
        }
        let total = u128::from(self.billable_ms);
        let rate = u128::from(self.hourly_rate_microunits?);
        let numerator = total.checked_mul(rate)?;
        let time_cost = numerator.checked_add(3_599_999)? / 3_600_000;
        u64::try_from(time_cost.checked_add(u128::from(self.fixed_cost_microunits?))?).ok()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    /// Catalog declaration checked against the loaded model's ready/health observation.
    /// Old catalogs remain readable, but unknown cannot satisfy reasoning obligations.
    #[serde(default)]
    pub reasoning_protocol: ModelReasoningProtocol,
    pub id: String,
    pub model: String,
    /// Exact GGUF artifact and independently pinned metadata expected from the product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf: Option<GgufSourceProfile>,
    pub target: ExecutionTarget,
    pub available: bool,
    pub estimate: Option<CostEstimate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgufSourceProfile {
    pub filename: String,
    /// owner/repository@40-hex-commit; checked against observed resolution.
    pub semantic_source: String,
    /// Defaults to semantic_source when both metadata roles share a snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Behavior {
    SourceClosure,
    SourceRevision,
    DownloadRecovery,
    CacheCompleteness,
    TemplateHistory,
    ProtocolFraming,
    ReasoningBoundaries,
    ReasoningAbsence,
    UserStop,
    NaturalEnd,
    LengthLimit,
    StructuredSampling,
    StructuredValidity,
    ToolSelection,
    ToolHandoff,
    ToolContinuation,
    SchedulingProgress,
    Cancellation,
    CapacityAdmission,
    KvIsolation,
    KvRelease,
    /// Resume supported preemption by recomputing preserved prompt/generated
    /// history after physical KV release; does not imply KV swapping support.
    KvResume,
    SubmissionCompletion,
    /// Host-side numerical decoding, source closure and physical weight layout
    /// boundaries. Model execution and device arithmetic require other evidence.
    WeightMaterialization,
    KernelNumerics,
    KernelBoundaries,
    ModelLoad,
    ModelForward,
    ArchitectureState,
    Installation,
    QuickStart,
    Performance,
    /// Request metadata, instrumentation delivery and sink completion/error handling.
    /// Runtime sampling does not certify every operator or numerical layout.
    Observability,
    WorkspaceChecks,
}
/// Evidence categories are not an ordering: compilation does not subsume protocol,
/// model runtime does not subsume reference numerics or measured performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLayer {
    Contract,
    BackendNumerics,
    ModelRuntime,
    Installation,
    Performance,
    Compilation,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObligationScope {
    Global,
    Backend {
        backend: Backend,
    },
    /// Backend command submission is shared across model weight precisions and
    /// architectures, but independent execution routes cannot represent it.
    ExecutionPath {
        backend: Backend,
        execution_path: String,
    },
    Architecture {
        architecture: String,
        protocol: ModelOutputProtocol,
        /// Different production executors cannot represent one another.
        execution_path: String,
    },
    Protocol {
        protocol: ModelOutputProtocol,
        backend: Backend,
        execution_path: String,
    },
    Reasoning {
        protocol: ModelOutputProtocol,
        backend: Backend,
        execution_path: String,
        reasoning_protocol: ModelReasoningProtocol,
    },
    Target {
        target: ExecutionTarget,
    },
    Profile {
        profile_id: String,
        target: ExecutionTarget,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Obligation {
    pub behavior: Behavior,
    pub layer: EvidenceLayer,
    pub entrypoints: Vec<Entrypoint>,
    pub scope: ObligationScope,
    pub reason: String,
    /// Available implementation assignments, not completed executions.
    pub checkers: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckDescriptor {
    pub id: String,
    pub behavior: Behavior,
    pub layer: EvidenceLayer,
    pub entrypoints: Vec<Entrypoint>,
    /// None declares a generic implementation; executing it must still bind the
    /// actual inputs/target and validate observations before evidence is accepted.
    pub target: Option<ExecutionTarget>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanInput {
    pub stage: Stage,
    /// Explicit release-only CUDA model sampling; device numerical targets remain intact.
    #[serde(default)]
    pub release_cuda: Option<ReleaseCudaPolicy>,
    /// Resource-bounded release Metal sampling; larger model coverage stays not-run.
    #[serde(default)]
    pub release_metal: Option<ReleaseMetalPolicy>,
    /// Release scheduling only; correctness layers cannot be deferred by this policy.
    #[serde(default)]
    pub release_performance: ReleasePerformancePolicy,
    pub impact: Impact,
    pub profiles: Vec<ModelProfile>,
    pub quick_start_profile_ids: Vec<String>,
    /// Explicit model commitments for this release, in addition to README examples.
    #[serde(default)]
    pub release_profile_ids: Vec<String>,
    /// Advertised execution inventory, not an unconditional full-model matrix.
    pub required_targets: Vec<ExecutionTarget>,
    #[serde(default)]
    pub checks: Vec<CheckDescriptor>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedProfile {
    pub profile: ModelProfile,
    /// Assigned model obligations, each owned by exactly one selected profile.
    /// Other compatible profiles do not implicitly repeat these checks.
    /// References into Plan.obligations are traceability, not pass ratios.
    pub obligations: Vec<usize>,
    pub reasons: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmittedProfile {
    pub profile_id: String,
    pub reason: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Gap {
    UnmappedChange {
        path: String,
    },
    ProductContractReview,
    UnknownReasoningCapability {
        profile_id: String,
    },
    EmptyInventory,
    MissingQuickStart {
        profile_id: String,
    },
    MissingReleaseProfile {
        profile_id: String,
    },
    InvalidTarget {
        target: ExecutionTarget,
        reason: String,
    },
    MissingRepresentative {
        obligation: usize,
    },
    UnassignedCheck {
        obligation: usize,
    },
    InvalidEstimate {
        profile_id: String,
        reason: String,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrencyEstimate {
    pub currency: String,
    pub microunits: u64,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanCost {
    /// Sum of profile durations; not parallel wall time or a latency percentile.
    pub known_total_ms: u64,
    pub known_billable_ms: u64,
    pub estimated_costs: Vec<CurrencyEstimate>,
    pub unknown_duration_profiles: Vec<String>,
    pub unknown_price_profiles: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub stage: Stage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_cuda: Option<ReleaseCudaPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_metal: Option<ReleaseMetalPolicy>,
    /// Explicit model coverage outside enabled release lanes. Never a pass.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extended_not_run: Vec<Obligation>,
    /// Model-dependent semantics excluded by documented capability limitations.
    /// These are uncovered obligations, not successful execution evidence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_limitations_not_run: Vec<Obligation>,
    pub impact: Impact,
    pub obligations: Vec<Obligation>,
    /// Explicitly postponed measurements, never passing execution evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_performance: Option<DeferredPerformance>,
    pub selected: Vec<SelectedProfile>,
    pub omitted: Vec<OmittedProfile>,
    pub gaps: Vec<Gap>,
    pub cost: PlanCost,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReleasePerformancePolicy {
    #[default]
    Required,
    Deferred {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeferredPerformance {
    pub reason: String,
    pub obligations: Vec<Obligation>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudCudaMode {
    #[default]
    Disabled,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaModelLane {
    Local,
    Cloud,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseCudaPolicy {
    #[serde(default)]
    pub cloud: CloudCudaMode,
    /// Ordered representatives: earlier local profiles own shared matching behavior.
    pub mandatory_local_profile_ids: Vec<String>,
    pub extended_cloud_profile_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_limitations: Vec<ModelCapabilityLimitation>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityLimitation {
    pub profile_id: String,
    /// Exact pinned source reviewed by the independent reference experiment.
    pub model_source: String,
    pub behaviors: Vec<Behavior>,
    pub reason: String,
    pub evidence: String,
}

impl ModelCapabilityLimitation {
    fn validate(&self) -> Result<(), String> {
        let allowed = |behavior: &Behavior| {
            matches!(
                behavior,
                Behavior::ArchitectureState
                    | Behavior::ToolSelection
                    | Behavior::ToolHandoff
                    | Behavior::ToolContinuation
            )
        };
        if [
            &self.profile_id,
            &self.model_source,
            &self.reason,
            &self.evidence,
        ]
        .iter()
        .any(|value| value.trim().is_empty() || value.trim() != value.as_str())
            || self.behaviors.is_empty()
            || self.behaviors.iter().any(|behavior| !allowed(behavior))
            || self
                .behaviors
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.behaviors.len()
        {
            return Err("model limitation requires unique semantic behaviors, a reason and reference evidence".into());
        }
        if super::model_sources::pinned_hf_source(&self.model_source)?.is_none() {
            return Err("model limitation requires an immutable model source".into());
        }
        let tools = [
            Behavior::ToolSelection,
            Behavior::ToolHandoff,
            Behavior::ToolContinuation,
        ];
        if tools
            .iter()
            .any(|behavior| self.behaviors.contains(behavior))
            && !tools
                .iter()
                .all(|behavior| self.behaviors.contains(behavior))
        {
            return Err(
                "the combined tool probe requires declaring all three tool behaviors".into(),
            );
        }
        Ok(())
    }
}

impl ReleaseCudaPolicy {
    pub fn limits(&self, profile_id: &str, behavior: Behavior) -> bool {
        self.model_limitations.iter().any(|limitation| {
            limitation.profile_id == profile_id && limitation.behaviors.contains(&behavior)
        })
    }

    pub fn lane(&self, profile_id: &str) -> Option<CudaModelLane> {
        if self
            .mandatory_local_profile_ids
            .iter()
            .any(|id| id == profile_id)
        {
            Some(CudaModelLane::Local)
        } else if self
            .extended_cloud_profile_ids
            .iter()
            .any(|id| id == profile_id)
        {
            Some(CudaModelLane::Cloud)
        } else {
            None
        }
    }

    pub fn validate(&self, profiles: &[ModelProfile]) -> Result<(), String> {
        let mut limited_profiles = std::collections::BTreeSet::new();
        for limitation in &self.model_limitations {
            limitation.validate()?;
            if self.lane(&limitation.profile_id) != Some(CudaModelLane::Local)
                || !limited_profiles.insert(&limitation.profile_id)
                || !profiles.iter().any(|profile| {
                    profile.id == limitation.profile_id && profile.model == limitation.model_source
                })
            {
                return Err("model limitations must name unique mandatory local profiles".into());
            }
        }
        if self.reason.trim().is_empty()
            || self.reason.trim() != self.reason
            || self.mandatory_local_profile_ids.is_empty()
            || self.extended_cloud_profile_ids.is_empty()
        {
            return Err(
                "CUDA release policy requires local/extended profiles and an explicit reason"
                    .into(),
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        for id in self
            .mandatory_local_profile_ids
            .iter()
            .chain(&self.extended_cloud_profile_ids)
        {
            if id.trim().is_empty() || id.trim() != id || !seen.insert(id) {
                return Err("CUDA lane profile IDs must be nonblank, unique and disjoint".into());
            }
            let profile = profiles
                .iter()
                .find(|profile| &profile.id == id)
                .ok_or_else(|| format!("CUDA lane profile {id} is absent from the catalog"))?;
            if profile.target.backend != Backend::Cuda {
                return Err(format!("CUDA lane profile {id} declares another backend"));
            }
            if self.lane(id) == Some(CudaModelLane::Local) {
                super::model_sources::validate_profile_sources(profile)?;
                if super::model_sources::pinned_hf_source(&profile.model)?.is_none() {
                    return Err(format!(
                        "local CUDA profile {id} requires an immutable HF revision"
                    ));
                }
            }
        }
        if profiles
            .iter()
            .any(|profile| profile.target.backend == Backend::Cuda && !seen.contains(&profile.id))
        {
            return Err("every CUDA model profile requires an explicit local or cloud lane".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalModelLane {
    Local,
    Extended,
}

/// A bounded Metal release samples its mandatory representatives only. Extended
/// profiles remain declared but are not enabled by this resource policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseMetalPolicy {
    pub mandatory_local_profile_ids: Vec<String>,
    pub extended_profile_ids: Vec<String>,
    pub reason: String,
}

impl ReleaseMetalPolicy {
    pub fn lane(&self, profile_id: &str) -> Option<MetalModelLane> {
        if self
            .mandatory_local_profile_ids
            .iter()
            .any(|id| id == profile_id)
        {
            Some(MetalModelLane::Local)
        } else if self.extended_profile_ids.iter().any(|id| id == profile_id) {
            Some(MetalModelLane::Extended)
        } else {
            None
        }
    }

    fn validate_inventory(&self) -> Result<(), String> {
        let ids: Vec<_> = self
            .mandatory_local_profile_ids
            .iter()
            .chain(&self.extended_profile_ids)
            .collect();
        if self.reason.trim().is_empty()
            || self.reason.trim() != self.reason
            || self.mandatory_local_profile_ids.is_empty()
            || self.extended_profile_ids.is_empty()
            || ids
                .iter()
                .any(|id| id.trim().is_empty() || id.trim() != id.as_str())
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != ids.len()
        {
            return Err("Metal release policy requires unique, disjoint local/extended profiles and a reason".into());
        }
        Ok(())
    }

    pub fn validate(&self, profiles: &[ModelProfile]) -> Result<(), String> {
        self.validate_inventory()?;
        for id in self
            .mandatory_local_profile_ids
            .iter()
            .chain(&self.extended_profile_ids)
        {
            let profile = profiles
                .iter()
                .find(|profile| &profile.id == id)
                .ok_or_else(|| format!("Metal lane profile {id} is absent from the catalog"))?;
            if profile.target.backend != Backend::Metal {
                return Err(format!("Metal lane profile {id} declares another backend"));
            }
            if self.lane(id) == Some(MetalModelLane::Local) {
                super::model_sources::validate_profile_sources(profile)?;
                if super::model_sources::pinned_hf_source(&profile.model)?.is_none() {
                    return Err(format!(
                        "local Metal profile {id} requires an immutable HF revision"
                    ));
                }
            }
        }
        if profiles.iter().any(|profile| {
            profile.target.backend == Backend::Metal && self.lane(&profile.id).is_none()
        }) {
            return Err(
                "every Metal model profile requires an explicit local or extended lane".into(),
            );
        }
        Ok(())
    }
}

impl Plan {
    /// Compatibility entrypoint: consumers of the original CUDA policy must also
    /// reject malformed Metal policy and never mistake its not-run coverage for CUDA.
    pub fn validate_cuda_policy(&self) -> Result<(), String> {
        self.validate_release_policies()
    }

    pub fn validate_release_policies(&self) -> Result<(), String> {
        self.validate_cuda_lane_policy()?;
        for obligation in &self.model_limitations_not_run {
            let permitted = obligation.layer == EvidenceLayer::ModelRuntime
                && self.release_cuda.as_ref().is_some_and(|policy| {
                    self.selected.iter().any(|selected| {
                        policy.limits(&selected.profile.id, obligation.behavior)
                            && super::selection::profile_covers(&selected.profile, obligation)
                    })
                });
            if !permitted {
                return Err(
                    "model limitation cannot exempt undeclared or non-model obligations".into(),
                );
            }
        }
        if let Some(policy) = &self.release_metal {
            policy.validate_inventory()?;
            if self.stage != Stage::Release {
                return Err("Metal model sampling requires a release plan".into());
            }
            for selected in &self.selected {
                if selected.profile.target.backend == Backend::Metal
                    && policy.lane(&selected.profile.id) != Some(MetalModelLane::Local)
                {
                    return Err("selected Metal profile is outside its mandatory local lane".into());
                }
            }
            for id in &policy.mandatory_local_profile_ids {
                if !self.selected.iter().any(|selected| {
                    selected.profile.id == *id && selected.profile.target.backend == Backend::Metal
                }) {
                    return Err(format!(
                        "required Metal profile {id} has no selected Metal execution"
                    ));
                }
            }
        }
        let cuda_disabled = self
            .release_cuda
            .as_ref()
            .is_some_and(|policy| policy.cloud == CloudCudaMode::Disabled);
        let metal_disabled = self.release_metal.is_some();
        for obligation in &self.extended_not_run {
            let permitted_backend = match &obligation.scope {
                ObligationScope::Backend { backend }
                | ObligationScope::ExecutionPath { backend, .. }
                | ObligationScope::Protocol { backend, .. }
                | ObligationScope::Reasoning { backend, .. } => Some(*backend),
                ObligationScope::Target { target } | ObligationScope::Profile { target, .. } => {
                    Some(target.backend)
                }
                ObligationScope::Global | ObligationScope::Architecture { .. } => None,
            };
            let enabled_extension = match permitted_backend {
                Some(Backend::Cuda) => cuda_disabled,
                Some(Backend::Metal) => metal_disabled,
                Some(_) => false,
                None => cuda_disabled || metal_disabled,
            };
            if obligation.layer != EvidenceLayer::ModelRuntime || !enabled_extension {
                return Err("extended not-run coverage requires an explicit disabled model lane for its backend".into());
            }
        }
        Ok(())
    }

    fn validate_cuda_lane_policy(&self) -> Result<(), String> {
        let Some(policy) = &self.release_cuda else {
            return Ok(());
        };
        let mut limited_profiles = std::collections::BTreeSet::new();
        for limitation in &policy.model_limitations {
            limitation.validate()?;
            if policy.lane(&limitation.profile_id) != Some(CudaModelLane::Local)
                || !limited_profiles.insert(&limitation.profile_id)
                || !self.selected.iter().any(|selected| {
                    selected.profile.id == limitation.profile_id
                        && selected.profile.model == limitation.model_source
                })
            {
                return Err("invalid frozen model capability limitation".into());
            }
        }
        let ids: Vec<_> = policy
            .mandatory_local_profile_ids
            .iter()
            .chain(&policy.extended_cloud_profile_ids)
            .collect();
        if self.stage != Stage::Release
            || policy.reason.trim().is_empty()
            || policy.reason.trim() != policy.reason
            || policy.mandatory_local_profile_ids.is_empty()
            || policy.extended_cloud_profile_ids.is_empty()
            || ids
                .iter()
                .any(|id| id.trim().is_empty() || id.trim() != id.as_str())
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != ids.len()
        {
            return Err("invalid frozen CUDA release policy or extended not-run evidence".into());
        }
        for selected in &self.selected {
            if selected.profile.target.backend == Backend::Cuda {
                match policy.lane(&selected.profile.id) {
                    Some(CudaModelLane::Local) => {}
                    Some(CudaModelLane::Cloud) if policy.cloud == CloudCudaMode::Required => {}
                    _ => return Err("selected CUDA profile is outside its enabled lane".into()),
                }
            }
        }
        for id in policy.mandatory_local_profile_ids.iter().chain(
            policy
                .extended_cloud_profile_ids
                .iter()
                .filter(|_| policy.cloud == CloudCudaMode::Required),
        ) {
            if !self.selected.iter().any(|selected| {
                selected.profile.id == *id && selected.profile.target.backend == Backend::Cuda
            }) {
                return Err(format!(
                    "required CUDA profile {id} has no selected CUDA execution"
                ));
            }
        }
        Ok(())
    }

    pub fn validate_performance_deferral(&self) -> Result<(), String> {
        if let Some(deferred) = &self.deferred_performance {
            if self.stage != Stage::Release
                || deferred.reason.trim().is_empty()
                || deferred.reason.trim() != deferred.reason
                || deferred.obligations.is_empty()
                || deferred
                    .obligations
                    .iter()
                    .any(|o| o.layer != EvidenceLayer::Performance)
                || self
                    .obligations
                    .iter()
                    .any(|o| o.layer == EvidenceLayer::Performance)
            {
                return Err("performance deferral must retain only performance obligations with an explicit release reason".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate() -> CostEstimate {
        CostEstimate {
            download_ms: 3_600_000,
            load_ms: 300_000,
            check_ms: 1_200_000,
            setup_ms: 100_000,
            cleanup_ms: 200_000,
            billable_ms: 1_800_000,
            hourly_rate_microunits: Some(6_000_000),
            fixed_cost_microunits: Some(100_000),
            currency: Some("CNY".into()),
        }
    }

    #[test]
    fn prepared_download_time_is_not_charged_at_the_gpu_rate() {
        let estimate = estimate();
        assert_eq!(estimate.total_ms(), Some(5_400_000));
        assert_eq!(estimate.cost_microunits(), Some(3_100_000));
    }

    #[test]
    fn unknown_prices_are_not_zero_and_invalid_sums_are_not_wrapped() {
        let mut value = estimate();
        value.currency = None;
        assert_eq!(value.cost_microunits(), None);
        value = estimate();
        value.hourly_rate_microunits = None;
        assert_eq!(value.cost_microunits(), None);
        value = estimate();
        value.fixed_cost_microunits = None;
        assert_eq!(value.cost_microunits(), None);
        value = estimate();
        value.download_ms = u64::MAX;
        assert_eq!(value.total_ms(), None);
        value.billable_ms = u64::MAX;
        value.hourly_rate_microunits = Some(u64::MAX);
        assert_eq!(value.cost_microunits(), None);
    }

    #[test]
    fn small_costs_round_up_and_an_explicit_free_resource_is_representable() {
        let mut value = estimate();
        value.billable_ms = 1;
        value.hourly_rate_microunits = Some(1);
        value.fixed_cost_microunits = Some(0);
        assert_eq!(value.cost_microunits(), Some(1));
        value.hourly_rate_microunits = Some(0);
        assert_eq!(value.cost_microunits(), Some(0));
    }

    #[test]
    fn price_and_duration_input_rejects_negative_or_fractional_integer_units() {
        let original = serde_json::to_value(estimate()).unwrap();
        for bad in [
            serde_json::json!(-1),
            serde_json::json!(0.5),
            serde_json::json!("NaN"),
        ] {
            let mut value = original.clone();
            value["billable_ms"] = bad;
            assert!(serde_json::from_value::<CostEstimate>(value).is_err());
        }
    }

    #[test]
    fn older_catalog_reasoning_is_unknown_not_an_absence_declaration() {
        let profile: ModelProfile = serde_json::from_value(serde_json::json!({
            "id": "unresolved", "model": "fixture/model", "available": true, "estimate": null,
            "target": {"architecture": "dense", "protocol": "text", "precision": "bf16",
                "backend": "cpu", "execution_path": "production-plan-runtime"}
        }))
        .unwrap();
        assert_eq!(profile.reasoning_protocol, ModelReasoningProtocol::Unknown);
        for (text, expected) in [
            ("none", ModelReasoningProtocol::None),
            ("prompt_opened", ModelReasoningProtocol::PromptOpened),
            ("model_generated", ModelReasoningProtocol::ModelGenerated),
            ("unknown", ModelReasoningProtocol::Unknown),
        ] {
            assert_eq!(
                serde_json::from_value::<ModelReasoningProtocol>(serde_json::json!(text)).unwrap(),
                expected
            );
        }
        assert!(serde_json::from_value::<ModelReasoningProtocol>(serde_json::Value::Null).is_err());
    }
}
