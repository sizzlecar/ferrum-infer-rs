use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceClockV6 {
    pub wall_unix_ns: u64,
    pub monotonic_ns: u64,
}
impl From<StructuredServiceClockV6> for PairedClock {
    fn from(v: StructuredServiceClockV6) -> Self {
        Self {
            wall_unix_ns: v.wall_unix_ns,
            monotonic_ns: v.monotonic_ns,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceDeclarationV6 {
    #[serde(
        default,
        skip_serializing_if = "ferrum_types::SloCalibrationRoutePopulationV1::is_all_attempts"
    )]
    pub route_population: ferrum_types::SloCalibrationRoutePopulationV1,
    #[serde(
        default,
        skip_serializing_if = "StructuredServiceDomainPolicyV1::is_all_offered"
    )]
    pub domain_policy: StructuredServiceDomainPolicyV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonnegative_envelope: Option<NonNegativeEnvelopeContractV1>,
    pub phase_offered_waves: [usize; 3],
    pub maximum_window_ns: u64,
    pub settings: StructuredSettingsV2,
    #[serde(deserialize_with = "bounded_rows")]
    pub scopes: Vec<StructuredScopeV2>,
    pub maximum_retained_numeric_bytes: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceHeaderV6 {
    artifact_type: String,
    schema_version: u32,
    model_revision: String,
    pub capture_identity: [u8; 32],
    pub generation: u64,
    pub protocol: [u8; 32],
    pub fingerprint: ProfileFingerprint,
    pub producer: serde_json::Value,
    pub opening: StructuredServiceClockV6,
    pub declaration: StructuredServiceDeclarationV6,
    pub declaration_sha256: [u8; 32],
    pub maximum_file_bytes: u64,
}
impl StructuredServiceHeaderV6 {
    pub fn new(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV6,
        declaration: StructuredServiceDeclarationV6,
        maximum_file_bytes: u64,
    ) -> Result<Self, CostProfileError> {
        let declaration_sha256 = Sha256::digest(serde_json::to_vec(&declaration)?).into();
        let mut h = Self {
            artifact_type: "ferrum.structured-service-live-source".into(),
            schema_version: 6,
            model_revision: MODEL_REVISION_V2.into(),
            capture_identity,
            generation,
            protocol: [0; 32],
            fingerprint,
            producer,
            opening,
            declaration,
            declaration_sha256,
            maximum_file_bytes,
        };
        h.protocol = h.protocol_signature()?;
        h.validate()?;
        Ok(h)
    }
    fn protocol_signature(&self) -> Result<[u8; 32], CostProfileError> {
        let mut digest = Sha256::new();
        digest.update(SERVICE_SOURCE_PROTOCOL_V6.as_bytes());
        digest.update(MODEL_REVISION_V2.as_bytes());
        digest.update(self.declaration_sha256);
        digest.update(serde_json::to_vec(&self.fingerprint)?);
        digest.update(self.maximum_file_bytes.to_le_bytes());
        Ok(digest.finalize().into())
    }
    pub(super) fn validate(&self) -> Result<(), CostProfileError> {
        let d = &self.declaration;
        if self.artifact_type != "ferrum.structured-service-live-source"
            || self.schema_version != 6
            || self.model_revision != MODEL_REVISION_V2
            || self.capture_identity == [0; 32]
            || self.generation == 0
            || self.opening.wall_unix_ns == 0
            || self.maximum_file_bytes == 0
            || self.maximum_file_bytes > 8 * 1024 * 1024 * 1024
            || d.scopes.is_empty()
            || d.scopes.len() > 128
            || d.phase_offered_waves.iter().any(|n| *n == 0 || *n > 65_536)
            || d.maximum_window_ns == 0
            || d.maximum_window_ns > d.settings.max_sample_age_ns
            || d.maximum_retained_numeric_bytes == 0
            || d.maximum_retained_numeric_bytes > 512 * 1024 * 1024
            || self.declaration_sha256 != <[u8; 32]>::from(Sha256::digest(serde_json::to_vec(d)?))
            || self.protocol != self.protocol_signature()?
            || self
                .producer
                .get("executable_path")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(invalid("source6 declaration, identity or capacity differs"));
        }
        // Source6 selects each declared scope's samples by exact owner. Its
        // numerical population must not authorize a wider family afterward.
        if d.nonnegative_envelope
            .as_ref()
            .is_some_and(|contract| !contract.population_policy.is_exact_owner())
            || d.scopes
                .iter()
                .any(|scope| scope.numerical_family.is_some())
        {
            return Err(invalid("source6 population policy requires exact owner"));
        }
        d.settings.validate().map_err(numeric_error)?;
        match (&d.domain_policy, &d.nonnegative_envelope) {
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, Some(contract)) => {
                contract.validate().map_err(numeric_error)?;
                let identity = ferrum_interfaces::execution_cost::ExecutorCostIdentity {
                    schema_version:
                        ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA,
                    model_weights: self.fingerprint.model_weights,
                    numerical_policy: self.fingerprint.numerical_policy,
                    device_runtime: self.fingerprint.device_runtime,
                    execution_config: self.fingerprint.execution_config,
                };
                if !contract
                    .workload_domain
                    .matches_execution_identity(&identity)
                {
                    return Err(invalid(
                        "source6 physical domain execution identity differs",
                    ));
                }
            }
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, None)
            | (_, Some(_)) => return Err(invalid("source6 numerical policy payload differs")),
            (_, None) => {}
        }

        for (i, s) in d.scopes.iter().enumerate() {
            s.validate().map_err(numeric_error)?;
            let template_policy = d
                .nonnegative_envelope
                .as_ref()
                .map(|c| c.template_policy)
                .unwrap_or_default();
            if s.owner.cost_template_policy() != template_policy {
                return Err(invalid("source6 owner numerical template policy differs"));
            }
            if d.scopes[..i].iter().any(|p| p.owner == s.owner) {
                return Err(invalid("source6 duplicate owner"));
            }
        }
        Ok(())
    }
    pub(super) fn child_contract(
        &self,
        child: usize,
    ) -> Result<StructuredServiceWindowContractV2, CostProfileError> {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.service-window-owner-predicate.v1\0");
        digest.update(self.declaration_sha256);
        digest.update(serde_json::to_vec(&self.declaration.scopes[child])?);
        Ok(StructuredServiceWindowContractV2 {
            capture_identity: self.capture_identity,
            protocol: self.protocol,
            membership_rule: digest.finalize().into(),
            window_declaration: self.declaration_sha256,
            phase_offered: self.declaration.phase_offered_waves,
            domain_policy: self.declaration.domain_policy,
            nonnegative_envelope: self.declaration.nonnegative_envelope.clone(),
        })
    }
}

/// A replay DTO built from an original private diagnostic view. No live
/// receipt, Prepared projection or executable authority can be deserialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceWaveV6 {
    pub ticket: u64,
    pub phase: StructuredPhaseV2,
    pub issued_at_ns: u64,
    pub fifo: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) prepared_route: Option<route_population::StructuredServicePreparedRouteV6>,
    pub(super) host_stages: Stages,
    pub(super) independent: Option<IndependentAttentionWaveEvidenceWireV2>,
}
impl StructuredServiceWaveV6 {
    pub fn from_diagnostic(
        ticket: u64,
        phase: StructuredPhaseV2,
        issued_at_ns: u64,
        fifo: u64,
        host_stages: serde_json::Value,
        independent: Option<IndependentAttentionWaveEvidenceWireV2>,
    ) -> Result<Self, CostProfileError> {
        Ok(Self {
            ticket,
            phase,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceChildFreezeV6 {
    pub child: usize,
    pub members: usize,
    pub parameters_sha256: Option<[u8; 32]>,
    pub failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<StructuredServiceDomainFreezeV1>,
    /// Successful new-policy Fit only. Replay validates this integer witness
    /// against every original Fit member instead of rerunning a floating solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonnegative_fit_certificate: Option<NonNegativeFitCertificateV1>,
}

/// Complete original owner population, distinct from eligible numerical rows.
/// The source prefix retains every original record, including excluded rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceDomainFreezeV1 {
    pub owner_offered: usize,
    pub eligible: usize,
    pub outside_fit_support: usize,
    pub outside_residual_support: usize,
    pub unclassified_failed_owner: usize,
    /// Fit identity before Residual, calibrated identity before Qualification.
    pub frozen_domain_parameters_sha256: Option<[u8; 32]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuredServiceRecordV6 {
    PhaseOpen {
        phase: StructuredPhaseV2,
        opened_at_ns: u64,
        fifo_cutoff: u64,
    },
    Completed {
        wave: StructuredServiceWaveV6,
    },
    OutsideDeclaredRoute {
        wave: StructuredServiceOutsideRouteV6,
    },
    /// Explicit V2-only retirement; never a numerical wave or a refill.
    NotSubmitted {
        attempt: StructuredServiceNoSubmissionV6,
    },
    /// Every failed private ticket is retained, including an unqueued last one.
    TicketFailed {
        phase: StructuredPhaseV2,
        ticket: u64,
        issued_at_ns: u64,
        call_id: u64,
        reason: String,
    },
    PhaseFreeze {
        phase: StructuredPhaseV2,
        frozen_at_ns: u64,
        source_prefix_bytes: u64,
        source_prefix_sha256: [u8; 32],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        route_population: Option<route_population::StructuredServiceRouteCountsV1>,
        #[serde(deserialize_with = "bounded_rows")]
        children: Vec<StructuredServiceChildFreezeV6>,
    },
    Footer {
        offered: u64,
        accepted_fifo_cutoff: u64,
        closing: StructuredServiceClockV6,
        failure: Option<String>,
    },
}
