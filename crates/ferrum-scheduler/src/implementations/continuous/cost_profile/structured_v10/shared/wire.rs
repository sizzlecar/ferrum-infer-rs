//! Source4 stores physical records once. Child declarations retain their
//! independent numerical protocol; they do not describe physical source3 files.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HeaderV4 {
    pub artifact_type: String,
    pub schema_version: u32,
    pub model_revision: String,
    pub maximum_file_bytes: u64,
    pub maximum_children: usize,
    pub maximum_retained_numeric_bytes: usize,
    pub maximum_retained_coordinates: usize,
    pub capture_protocol: [u8; 32],
    pub common: CommonDeclarationV4,
    #[serde(deserialize_with = "bounded_entries")]
    pub children: Vec<ChildDeclarationV4>,
}
impl HeaderV4 {
    pub fn signature(&self) -> Result<[u8; 32], CostProfileError> {
        let mut h = Sha256::new();
        h.update(b"ferrum.structured-shared-live-source.v4\0");
        h.update(serde_json::to_vec(&(
            &self.artifact_type,
            self.schema_version,
            &self.model_revision,
            self.maximum_file_bytes,
            self.maximum_children,
            self.maximum_retained_numeric_bytes,
            self.maximum_retained_coordinates,
            &self.common,
            &self.children,
        ))?);
        Ok(h.finalize().into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CommonDeclarationV4 {
    pub fingerprint: ProfileFingerprint,
    pub producer: serde_json::Value,
    pub opening: PairedClock,
    pub initial_fifo_cutoff: u64,
    pub cohort_plan: CohortPlanV2,
    pub cohort_manifest_payload: serde_json::Value,
    pub cohort_manifest_sha256: [u8; 32],
    pub maximum_offered_waves: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChildDeclarationV4 {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub declared_protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub opened_at_ns: u64,
    pub scope: StructuredScopeV2,
    pub membership_rule: MembershipRuleV2,
    pub phase_members: [usize; 3],
    pub settings: Settings,
}

/// Shared immutable common storage is never expanded into N legacy headers.
pub(super) struct ChildHeaderV4 {
    pub declaration: ChildDeclarationV4,
    pub common: Arc<CommonDeclarationV4>,
    pub maximum_file_bytes: u64,
}
impl std::ops::Deref for ChildHeaderV4 {
    type Target = ChildDeclarationV4;
    fn deref(&self) -> &Self::Target {
        &self.declaration
    }
}
impl ChildHeaderV4 {
    pub fn view(&self) -> super::super::replay::HeaderRef<'_> {
        self.declaration.view(&self.common, self.maximum_file_bytes)
    }
}
impl ChildDeclarationV4 {
    pub fn view<'a>(
        &'a self,
        common: &'a CommonDeclarationV4,
        maximum_file_bytes: u64,
    ) -> super::super::replay::HeaderRef<'a> {
        super::super::replay::HeaderRef {
            artifact_type: "ferrum.structured-live-source",
            schema_version: 3,
            model_revision: MODEL_REVISION_V2,
            capture_identity: self.capture_identity,
            protocol: self.protocol,
            declared_protocol: self.declared_protocol,
            rule_signature: self.rule_signature,
            fingerprint: &common.fingerprint,
            producer: &common.producer,
            opening: common.opening,
            opened_at_ns: self.opened_at_ns,
            initial_fifo_cutoff: common.initial_fifo_cutoff,
            scope: &self.scope,
            membership_rule: &self.membership_rule,
            cohort_plan: &common.cohort_plan,
            cohort_manifest_payload: &common.cohort_manifest_payload,
            cohort_manifest_sha256: common.cohort_manifest_sha256,
            phase_members: self.phase_members,
            maximum_offered_waves: common.maximum_offered_waves,
            maximum_file_bytes: maximum_file_bytes,
            settings: &self.settings,
        }
    }
}
impl CommonDeclarationV4 {
    pub fn from_header(h: Header) -> (Self, ChildDeclarationV4) {
        (
            Self {
                fingerprint: h.fingerprint,
                producer: h.producer,
                opening: h.opening,
                initial_fifo_cutoff: h.initial_fifo_cutoff,
                cohort_plan: h.cohort_plan,
                cohort_manifest_payload: h.cohort_manifest_payload,
                cohort_manifest_sha256: h.cohort_manifest_sha256,
                maximum_offered_waves: h.maximum_offered_waves,
            },
            ChildDeclarationV4 {
                capture_identity: h.capture_identity,
                protocol: h.protocol,
                declared_protocol: h.declared_protocol,
                rule_signature: h.rule_signature,
                opened_at_ns: h.opened_at_ns,
                scope: h.scope,
                membership_rule: h.membership_rule,
                phase_members: h.phase_members,
                settings: h.settings,
            },
        )
    }
    pub fn matches(&self, h: &Header) -> bool {
        self.fingerprint == h.fingerprint
            && self.producer == h.producer
            && self.opening.monotonic_ns == h.opening.monotonic_ns
            && self.opening.wall_unix_ns == h.opening.wall_unix_ns
            && self.initial_fifo_cutoff == h.initial_fifo_cutoff
            && self.cohort_plan == h.cohort_plan
            && self.cohort_manifest_payload == h.cohort_manifest_payload
            && self.cohort_manifest_sha256 == h.cohort_manifest_sha256
            && self.maximum_offered_waves == h.maximum_offered_waves
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MembershipV4 {
    pub member: Option<u64>,
    pub window: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum RecordV4 {
    /// Only the common cohort/admission/completion/offered/unavailable records
    /// are accepted here. All population-dependent records have typed variants.
    Common { record: Record },
    Reserved {
        offered: u64,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        boundary: String,
        prepared: Prepared,
        #[serde(deserialize_with = "bounded_entries")]
        memberships: Vec<MembershipV4>,
    },
    Completed {
        offered: u64,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        #[serde(deserialize_with = "bounded_entries")]
        members: Vec<Option<u64>>,
        queue: Option<Queue>,
        reconciled: bool,
        host_stages: Option<Stages>,
        outside_settlement: Option<OutsideSettlement>,
        selected_structured_capture: Option<std::result::Result<Recipe, serde_json::Value>>,
        selected_independent_attention_v2: Option<IndependentAttentionWaveEvidenceWireV2>,
        numeric: Option<Numeric>,
        conversion_error: Option<String>,
    },
    Unsubmitted {
        offered: u64,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        #[serde(deserialize_with = "bounded_entries")]
        members: Vec<Option<u64>>,
        reason: String,
    },
    Coverage {
        phase: StructuredProfilePhaseV10,
        #[serde(deserialize_with = "bounded_entries")]
        reports: Vec<serde_json::Value>,
    },
    PhaseFreeze {
        #[serde(deserialize_with = "bounded_entries")]
        receipts: Vec<Freeze>,
    },
    PhaseFailed {
        reason: String,
        #[serde(deserialize_with = "bounded_entries")]
        child_failures: Vec<ChildFailureV4>,
        #[serde(deserialize_with = "bounded_entries")]
        completed_freezes: Vec<Freeze>,
    },
    Footer {
        phase: String,
        failure: Option<String>,
        offered: u64,
        #[serde(deserialize_with = "bounded_entries")]
        members: Vec<u64>,
        #[serde(deserialize_with = "bounded_entries")]
        failed_members: Vec<u64>,
        accepted_fifo_cutoff: u64,
        last_captured_fifo: u64,
        fifo_audit_complete: bool,
        closing: Option<PairedClock>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChildFailureV4 {
    pub child: usize,
    pub capture_identity: [u8; 32],
    pub reason: String,
}

/// Reject excess child entries before retaining an unbounded vector.
pub(super) fn bounded_entries<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct Entries<T>(std::marker::PhantomData<T>);
    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for Entries<T> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("at most 128 declared child entries")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = seq.next_element()? {
                if values.len() == 128 {
                    return Err(serde::de::Error::custom("shared child capacity"));
                }
                values.push(value);
            }
            Ok(values)
        }
    }
    d.deserialize_seq(Entries(std::marker::PhantomData))
}
