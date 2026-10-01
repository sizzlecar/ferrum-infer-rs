use super::*;
use crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixPlanV5;
use crate::implementations::continuous::cost_model::structured_v2::StructuredPopulationPolicyV1;

pub const PREPARED_OWNER_BLOCK_SOURCE_PROTOCOL_V8: &str =
    "ferrum.structured-prepared-owner-blocks.v1";

/// Input-only membership is frozen by the first eligible original owner phase
/// of each complete declared cohort. Later phases require a fresh cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredPreparedCohortPhasePolicyV8 {
    FirstEligibleOwnerPhaseV1,
    FirstEligibleNumericalFamilyPhaseV1,
}
impl StructuredPreparedCohortPhasePolicyV8 {
    fn for_population(policy: StructuredPopulationPolicyV1) -> Self {
        match policy {
            StructuredPopulationPolicyV1::ExactOwnerV1 => Self::FirstEligibleOwnerPhaseV1,
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1 => {
                Self::FirstEligibleNumericalFamilyPhaseV1
            }
        }
    }
}

/// Completed declared cohorts may end between original full block barriers.
/// Such a tail is retained as audit evidence and never freezes a numeric phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredPreparedTailPolicyV8 {
    CompleteCohortsAuditOnlyV1,
}

/// The original driver passes and complete request cohorts are independent of
/// each numerical owner's phase. Source replay checks both ledgers, rather than
/// changing a request's declared pass when an owner advances or fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPreparedOwnerBlockDeclarationV8 {
    pub population: StructuredServiceDeclarationV7,
    pub cohort_plan: CohortPlanV2,
    pub prefix_plan: StructuredPrefixPlanV5,
    /// Explicit native acquisition is fixed before collecting the source.
    /// Absence preserves the original cold preparation trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_prefix_acquisition: Option<StructuredNativePrefixAcquisitionPlanV1>,
    pub cohort_manifest_payload: Box<serde_json::value::RawValue>,
    /// Includes preparation and every original formal/nonmember attempt.
    pub maximum_offered_waves: usize,
}
impl StructuredPreparedOwnerBlockDeclarationV8 {
    pub fn validate(&self) -> Result<(), CostProfileError> {
        self.cohort_plan.validate().map_err(numeric_error)?;
        self.prefix_plan
            .validate(&self.cohort_plan)
            .map_err(numeric_error)?;
        if let Some(plan) = &self.native_prefix_acquisition {
            plan.validate(&self.cohort_plan, &self.prefix_plan)?;
        }
        self.cohort_manifest_sha256()?;
        let p = &self.population;
        if p.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
            || p.nonnegative_envelope.as_ref().is_none_or(|v| {
                v.template_policy != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
            })
            || self.maximum_offered_waves == 0
            || self.maximum_offered_waves > 65_536
            || self.maximum_offered_waves < p.schedule.block_offered
        {
            return Err(invalid(
                "source8 requires a bounded declared physical-envelope population",
            ));
        }
        // Charge actual retained payload, not the encoded file size. Allocator
        // metadata is outside this payload accounting, as for imported models.
        if self
            .retained_payload_bytes()
            .is_none_or(|bytes| bytes > p.maximum_retained_numeric_bytes)
        {
            return Err(CostProfileError::Limit(
                "source8 declaration retained capacity",
            ));
        }
        Ok(())
    }
    pub fn cohort_manifest_sha256(&self) -> Result<[u8; 32], CostProfileError> {
        let payload = serde_json::from_str(self.cohort_manifest_payload.get())?;
        self.cohort_plan.signature(&payload).map_err(numeric_error)
    }
    pub fn prefix_plan_sha256(&self) -> Result<[u8; 32], CostProfileError> {
        self.prefix_plan
            .signature(&self.cohort_plan)
            .map_err(numeric_error)
    }
    pub fn signature(&self) -> Result<[u8; 32], CostProfileError> {
        self.validate()?;
        let mut h = Sha256::new();
        h.update(b"ferrum.structured-prepared-owner-block-declaration.v1\0");
        h.update(record_bytes_v7(self)?);
        Ok(h.finalize().into())
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        let mut n =
            std::mem::size_of::<Self>().checked_add(self.cohort_manifest_payload.get().len())?;
        n = n.checked_add(
            self.population
                .nonnegative_envelope
                .as_ref()
                .and_then(|c| c.algorithm_universe.as_ref())
                .map_or(Some(0), |u| u.retained_payload_bytes())?,
        )?;
        if let Some(plan) = &self.native_prefix_acquisition {
            n = n.checked_add(plan.retained_heap_bytes()?)?;
        }
        for phase in &self.cohort_plan.phases {
            n = n.checked_add(phase.capacity().checked_mul(std::mem::size_of::<
                crate::implementations::continuous::cost_model::structured_v2::windows::CohortV2,
            >())?)?;
            for cohort in phase {
                n = n.checked_add(cohort.requests.capacity().checked_mul(std::mem::size_of::<crate::implementations::continuous::cost_model::structured_v2::windows::CohortRequestV2>())?)?;
            }
        }
        for phase in &self.prefix_plan.phases {
            n = n.checked_add(phase.capacity().checked_mul(std::mem::size_of::<Option<crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixCohortV5>>())?)?;
            for cohort in phase.iter().flatten() {
                n = n.checked_add(cohort.slots.capacity().checked_mul(std::mem::size_of::<crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixSlotV5>())?)?;
                for slot in &cohort.slots {
                    n = n.checked_add(
                        slot.token_ids
                            .capacity()
                            .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())?,
                    )?;
                    n = n.checked_add(
                        slot.token_bytes
                            .capacity()
                            .checked_mul(std::mem::size_of::<Vec<u8>>())?,
                    )?;
                    for bytes in &slot.token_bytes {
                        n = n.checked_add(bytes.capacity())?;
                    }
                }
            }
        }
        Some(n)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPreparedOwnerBlockHeaderV8 {
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
    pub declaration: StructuredPreparedOwnerBlockDeclarationV8,
    pub declaration_sha256: [u8; 32],
    pub maximum_file_bytes: u64,
    cohort_phase_policy: StructuredPreparedCohortPhasePolicyV8,
    tail_policy: StructuredPreparedTailPolicyV8,
}
impl StructuredPreparedOwnerBlockHeaderV8 {
    pub fn cohort_phase_policy(&self) -> StructuredPreparedCohortPhasePolicyV8 {
        self.cohort_phase_policy
    }
    pub fn tail_policy(&self) -> StructuredPreparedTailPolicyV8 {
        self.tail_policy
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV7,
        declaration: StructuredPreparedOwnerBlockDeclarationV8,
        maximum_file_bytes: u64,
    ) -> Result<Self, CostProfileError> {
        let declaration_sha256 = declaration.signature()?;
        let cohort_phase_policy = StructuredPreparedCohortPhasePolicyV8::for_population(
            declaration.population.population_policy(),
        );
        let mut h = Self {
            artifact_type: "ferrum.structured-prepared-owner-block-source".into(),
            schema_version: 8,
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
            cohort_phase_policy,
            tail_policy: StructuredPreparedTailPolicyV8::CompleteCohortsAuditOnlyV1,
        };
        h.protocol = h.signature()?;
        h.validate()?;
        Ok(h)
    }
    /// Bind the original producer's boot-relative clock before the first record.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_monotonic_domain(
        capture_identity: [u8; 32],
        generation: u64,
        fingerprint: ProfileFingerprint,
        producer: serde_json::Value,
        opening: StructuredServiceClockV7,
        declaration: StructuredPreparedOwnerBlockDeclarationV8,
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
        h.update(PREPARED_OWNER_BLOCK_SOURCE_PROTOCOL_V8.as_bytes());
        h.update([0]);
        h.update(MODEL_REVISION_V2.as_bytes());
        h.update(self.declaration_sha256);
        h.update(record_bytes_v7(&self.fingerprint)?);
        h.update(self.maximum_file_bytes.to_le_bytes());
        h.update(record_bytes_v7(&self.cohort_phase_policy)?);
        h.update(record_bytes_v7(&self.tail_policy)?);
        if let Some(domain) = &self.monotonic_domain {
            domain
                .validate()
                .map_err(|_| CostProfileError::Clock("invalid original monotonic domain"))?;
            h.update(b"\0ferrum.original-monotonic-domain.v1\0");
            h.update(domain.sha256());
        }
        Ok(h.finalize().into())
    }
    pub fn validate(&self) -> Result<(), CostProfileError> {
        if self.artifact_type != "ferrum.structured-prepared-owner-block-source"
            || self.schema_version != 8
            || self
                .declaration
                .population
                .schedule
                .input_readiness
                .is_some()
            || self.declaration.population.schedule.phase_support.is_some()
            || self
                .declaration
                .population
                .schedule
                .opening_frontier
                .is_some()
            || self
                .declaration
                .population
                .schedule
                .algorithm_universe
                .is_some()
            || self.model_revision != MODEL_REVISION_V2
            || self.declaration_sha256 != self.declaration.signature()?
            || self.protocol != self.signature()?
            || self.cohort_phase_policy
                != StructuredPreparedCohortPhasePolicyV8::for_population(
                    self.declaration.population.population_policy(),
                )
        {
            return Err(invalid(
                "source8 identity or preparation declaration differs",
            ));
        }
        // Reuse the existing population/domain/capacity checks only. This
        // temporary declaration is never serialized, journaled or returned as
        // a source7 receipt; source8 has its own original header/hash chain.
        StructuredServiceHeaderV7::new_for_population_validation(
            self.capture_identity,
            self.generation,
            self.fingerprint.clone(),
            self.producer.clone(),
            self.opening,
            self.declaration.population.clone(),
            self.maximum_file_bytes,
        )?;
        let bytes = record_bytes_v7(self)?;
        if bytes.len() as u64 > self.maximum_file_bytes {
            return Err(CostProfileError::Limit("source8 header byte capacity"));
        }
        Ok(())
    }
}
