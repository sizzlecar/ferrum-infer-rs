use super::*;
use crate::implementations::continuous::cost_model::structured_v2::StructuredPopulationPolicyV1;

pub type StructuredServiceClockV7 = StructuredServiceClockV6;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceDeclarationV7 {
    pub schedule: OwnerBlockScheduleV1,
    pub route_population: ferrum_types::SloCalibrationRoutePopulationV1,
    pub domain_policy: StructuredServiceDomainPolicyV1,
    pub nonnegative_envelope: Option<NonNegativeEnvelopeContractV1>,
    pub settings: StructuredSettingsV2,
    pub maximum_window_ns: u64,
    pub maximum_owners: usize,
    pub maximum_retained_numeric_bytes: usize,
    pub maximum_discovery_bytes: usize,
}
impl StructuredServiceDeclarationV7 {
    pub fn population_policy(&self) -> StructuredPopulationPolicyV1 {
        self.nonnegative_envelope
            .as_ref()
            .map_or(StructuredPopulationPolicyV1::ExactOwnerV1, |v| {
                v.population_policy
            })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceHeaderV7 {
    artifact_type: String,
    schema_version: u32,
    model_revision: String,
    pub capture_identity: [u8; 32],
    pub generation: u64,
    pub protocol: [u8; 32],
    pub fingerprint: ProfileFingerprint,
    pub producer: serde_json::Value,
    pub opening: StructuredServiceClockV7,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monotonic_domain: Option<ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    pub declaration: StructuredServiceDeclarationV7,
    pub declaration_sha256: [u8; 32],
    pub maximum_file_bytes: u64,
}
impl StructuredServiceHeaderV7 {
    pub fn new(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV7,
        declaration: StructuredServiceDeclarationV7,
        maximum_file_bytes: u64,
    ) -> Result<Self, CostProfileError> {
        Self::new_for_population_validation(
            capture_identity,
            generation,
            fingerprint,
            producer,
            opening,
            declaration,
            maximum_file_bytes,
        )
    }

    /// Shared cold validation only. Source8 has its own policy-bound original
    /// header and must never serialize this temporary value as source7.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_for_population_validation(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV7,
        declaration: StructuredServiceDeclarationV7,
        maximum_file_bytes: u64,
    ) -> Result<Self, CostProfileError> {
        let declaration_sha256 = Sha256::digest(serde_json::to_vec(&declaration)?).into();
        let mut out = Self {
            artifact_type: "ferrum.structured-owner-block-source".into(),
            schema_version: 7,
            model_revision: MODEL_REVISION_V2.into(),
            capture_identity,
            generation,
            protocol: [0; 32],
            fingerprint,
            producer,
            opening,
            monotonic_domain: None,
            declaration,
            declaration_sha256,
            maximum_file_bytes,
        };
        out.protocol = out.signature()?;
        out.validate_population()?;
        Ok(out)
    }
    /// Bind the original producer's boot-relative clock before the first record.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_monotonic_domain(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV7,
        declaration: StructuredServiceDeclarationV7,
        maximum_file_bytes: u64,
        monotonic_domain: ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    ) -> Result<Self, CostProfileError> {
        let header = Self::new(
            capture_identity,
            generation,
            fingerprint,
            producer,
            opening,
            declaration,
            maximum_file_bytes,
        )?;
        header.with_monotonic_domain(monotonic_domain)
    }
    /// Consuming builder, used only before the original header is emitted.
    pub fn with_monotonic_domain(
        mut self,
        monotonic_domain: ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    ) -> Result<Self, CostProfileError> {
        monotonic_domain
            .validate()
            .map_err(|_| CostProfileError::Clock("invalid original monotonic domain"))?;
        self.monotonic_domain = Some(monotonic_domain);
        self.protocol = self.signature()?;
        self.validate()?;
        Ok(self)
    }
    fn signature(&self) -> Result<[u8; 32], CostProfileError> {
        let mut h = Sha256::new();
        h.update(SERVICE_SOURCE_PROTOCOL_V7.as_bytes());
        h.update(MODEL_REVISION_V2.as_bytes());
        h.update(self.declaration_sha256);
        h.update(serde_json::to_vec(&self.fingerprint)?);
        h.update(self.maximum_file_bytes.to_le_bytes());
        if let Some(domain) = &self.monotonic_domain {
            domain
                .validate()
                .map_err(|_| CostProfileError::Clock("invalid original monotonic domain"))?;
            h.update(b"\0ferrum.original-monotonic-domain.v1\0");
            h.update(domain.sha256());
        }
        Ok(h.finalize().into())
    }
    pub(super) fn validate(&self) -> Result<(), CostProfileError> {
        // Source7 assigns original waves at complete block boundaries. The
        // signed population policy changes their grouping, not their phase
        // intervals. Source8 additionally enforces its declared cohort policy.
        self.validate_population()
    }
    fn validate_population(&self) -> Result<(), CostProfileError> {
        let d = &self.declaration;
        if self.artifact_type != "ferrum.structured-owner-block-source"
            || self.schema_version != 7
            || self.model_revision != MODEL_REVISION_V2
            || self.capture_identity == [0; 32]
            || self.generation == 0
            || self.opening.wall_unix_ns == 0
            || self.maximum_file_bytes == 0
            || self.maximum_file_bytes > 8 * 1024 * 1024 * 1024
            || d.maximum_owners == 0
            || d.maximum_owners > 128
            || d.maximum_window_ns == 0
            || d.maximum_window_ns > d.settings.max_sample_age_ns
            || d.maximum_discovery_bytes == 0
            || d.maximum_discovery_bytes > 512 * 1024 * 1024
            || d.maximum_retained_numeric_bytes == 0
            || d.maximum_retained_numeric_bytes > 512 * 1024 * 1024
            || self.declaration_sha256 != <[u8; 32]>::from(Sha256::digest(serde_json::to_vec(d)?))
            || self.protocol != self.signature()?
            || self
                .producer
                .get("executable_path")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(invalid("source7 declaration, identity or capacity differs"));
        }
        d.settings.validate().map_err(numeric_error)?;
        d.schedule.validate(&d.settings).map_err(numeric_error)?;
        if d.schedule.phase_support.is_some()
            && d.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
        {
            return Err(invalid(
                "phase support requires the original physical envelope",
            ));
        }
        if d.schedule.input_readiness.is_some()
            && d.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
        {
            return Err(invalid(
                "input readiness requires the original physical envelope",
            ));
        }
        if d.schedule.algorithm_universe.is_some()
            && (d.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
                || d.nonnegative_envelope.as_ref().is_none_or(|c|
                    c.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
                    || c.template_policy != crate::implementations::continuous::cost_model::structured_v2::StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
                    || (c.algorithm_universe.is_some() != matches!(d.schedule.algorithm_universe,
                        Some(crate::implementations::continuous::cost_model::structured_v2::OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)))))
        {
            return Err(invalid("source7 discovered universe requires an unfrozen homogeneous physical envelope"));
        }
        match (&d.domain_policy, &d.nonnegative_envelope) {
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, Some(c)) => {
                c.validate().map_err(numeric_error)?;
                if matches!(d.schedule.algorithm_universe,
                    Some(crate::implementations::continuous::cost_model::structured_v2::OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)) {
                    c.algorithm_universe.as_ref().ok_or_else(|| invalid("source7 checked algorithm seed absent"))?
                        .validate_budget(d.settings.max_axes, d.maximum_discovery_bytes).map_err(numeric_error)?;
                }
                let id = ferrum_interfaces::execution_cost::ExecutorCostIdentity {
                    schema_version:
                        ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA,
                    model_weights: self.fingerprint.model_weights,
                    numerical_policy: self.fingerprint.numerical_policy,
                    device_runtime: self.fingerprint.device_runtime,
                    execution_config: self.fingerprint.execution_config,
                };
                if !c.workload_domain.matches_execution_identity(&id) {
                    return Err(invalid(
                        "source7 physical domain execution identity differs",
                    ));
                }
            }
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, None)
            | (_, Some(_)) => return Err(invalid("source7 numerical policy payload differs")),
            (_, None) => {}
        }
        Ok(())
    }
}

/// Original physical DTO only; deserialization never creates a live capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceWaveV7 {
    pub ticket: u64,
    pub issued_at_ns: u64,
    pub fifo: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepared_route: Option<route_population::StructuredServicePreparedRouteV6>,
    pub(super) host_stages: Stages,
    pub(super) independent: Option<IndependentAttentionWaveEvidenceWireV2>,
}
impl StructuredServiceWaveV7 {
    pub fn from_diagnostic(
        ticket: u64,
        issued_at_ns: u64,
        fifo: u64,
        host_stages: serde_json::Value,
        independent: Option<IndependentAttentionWaveEvidenceWireV2>,
    ) -> Result<Self, CostProfileError> {
        Ok(Self {
            ticket,
            issued_at_ns,
            fifo,
            prepared_route: None,
            host_stages: serde_json::from_value(host_stages)?,
            independent,
        })
    }
    pub fn with_prepared_route(
        mut self,
        diagnostic: serde_json::Value,
    ) -> Result<Self, CostProfileError> {
        self.prepared_route = Some(serde_json::from_value(diagnostic)?);
        Ok(self)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceOutsideRouteV7 {
    pub ticket: u64,
    pub issued_at_ns: u64,
    pub fifo: u64,
    pub(super) evidence: route_population::OutsideEvidence,
}
impl StructuredServiceOutsideRouteV7 {
    pub fn from_diagnostic(
        ticket: u64,
        issued_at_ns: u64,
        fifo: u64,
        evidence: serde_json::Value,
    ) -> Result<Self, CostProfileError> {
        Ok(Self {
            ticket,
            issued_at_ns,
            fifo,
            evidence: serde_json::from_value(evidence)?,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerAssignmentV7 {
    pub owner_attempt_id: u64,
    pub phase: StructuredPhaseV2,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerDiscoveryV7 {
    pub owner_attempt_id: u64,
    pub scope: StructuredScopeV2,
    pub contract: StructuredOwnerPhaseContractV1,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerFreezeV7 {
    pub owner_attempt_id: u64,
    pub close: StructuredOwnerPhaseCloseV1,
    pub domain: StructuredServiceDomainFreezeV1,
    pub parameters_sha256: Option<[u8; 32]>,
    pub failure: Option<String>,
    pub nonnegative_fit_certificate: Option<NonNegativeFitCertificateV1>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuredServiceRecordV7 {
    BlockOpen {
        block: u64,
        opened_at_ns: u64,
        fifo_cutoff: u64,
        assignments: Vec<StructuredOwnerAssignmentV7>,
    },
    Completed {
        wave: StructuredServiceWaveV7,
    },
    OutsideDeclaredRoute {
        wave: StructuredServiceOutsideRouteV7,
    },
    NotSubmitted {
        attempt: StructuredServiceNoSubmissionV7,
    },
    BlockClose {
        block: u64,
        closing: StructuredServiceClockV7,
        offered: u64,
        accepted_fifo_cutoff: u64,
        source_prefix_bytes: u64,
        source_prefix_sha256: [u8; 32],
        route_population: StructuredServiceRouteCountsV1,
        discoveries: Vec<StructuredOwnerDiscoveryV7>,
        freezes: Vec<StructuredOwnerFreezeV7>,
    },
    Checkpoint {
        block: u64,
        closing: StructuredServiceClockV7,
        offered: u64,
        accepted_fifo_cutoff: u64,
        source_prefix_bytes: u64,
        source_prefix_sha256: [u8; 32],
        qualified_attempts: Vec<u64>,
        pending_attempts: Vec<u64>,
    },
    Failed {
        ticket: u64,
        fifo: u64,
        at_ns: u64,
        reason: String,
        source_prefix_bytes: u64,
        source_prefix_sha256: [u8; 32],
    },
    Footer {
        closing: StructuredServiceClockV7,
        offered: u64,
        accepted_fifo_cutoff: u64,
        incomplete_block: bool,
        source_prefix_bytes: u64,
        source_prefix_sha256: [u8; 32],
    },
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredOwnerAuditV7 {
    pub owner_attempt_id: u64,
    pub owner: StructuredOwnerKeyV2,
    pub phase: Option<StructuredPhaseV2>,
    pub global_offers: u64,
    pub owner_offered: usize,
    pub eligible: usize,
    pub qualified: bool,
    pub failure: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredServiceAuditV7 {
    pub block: u64,
    pub offered: u64,
    pub block_offered: usize,
    pub last_fifo: u64,
    pub source_bytes: u64,
    /// Runtime audit only: never serialized into source records or receipts.
    pub readiness_scalar_visits: u64,
    /// Attempt count, not a measurement of total numerical CPU time.
    pub phase_transition_attempts: u64,
    pub retained_numeric_bytes: usize,
    pub poisoned: bool,
    pub closed: bool,
    pub owners: Vec<StructuredOwnerAuditV7>,
}
