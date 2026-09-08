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
    pub target: ExecutionTarget,
    pub available: bool,
    pub estimate: Option<CostEstimate>,
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
    pub impact: Impact,
    pub profiles: Vec<ModelProfile>,
    pub quick_start_profile_ids: Vec<String>,
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
    pub impact: Impact,
    pub obligations: Vec<Obligation>,
    pub selected: Vec<SelectedProfile>,
    pub omitted: Vec<OmittedProfile>,
    pub gaps: Vec<Gap>,
    pub cost: PlanCost,
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
