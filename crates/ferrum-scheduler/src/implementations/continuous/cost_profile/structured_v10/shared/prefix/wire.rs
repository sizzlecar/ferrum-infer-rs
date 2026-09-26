//! Untrusted source DTOs. None construct an engine capability or live receipt.
use super::*;
use ferrum_interfaces::execution_cost::HostCostPolicyV2;
use ferrum_types::TokenId;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HeaderV5 {
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
    pub prefix_plan: StructuredPrefixPlanV5,
    pub prefix_plan_sha256: [u8; 32],
}
impl HeaderV5 {
    pub fn new(
        h: HeaderV4,
        prefix_plan: StructuredPrefixPlanV5,
        prefix_plan_sha256: [u8; 32],
    ) -> Self {
        Self {
            artifact_type: "ferrum.structured-prefix-live-source".into(),
            schema_version: 5,
            model_revision: h.model_revision,
            maximum_file_bytes: h.maximum_file_bytes,
            maximum_children: h.maximum_children,
            maximum_retained_numeric_bytes: h.maximum_retained_numeric_bytes,
            maximum_retained_coordinates: h.maximum_retained_coordinates,
            capture_protocol: [0; 32],
            common: h.common,
            children: h.children,
            prefix_plan,
            prefix_plan_sha256,
        }
    }
    pub fn signature(&self) -> Result<[u8; 32], CostProfileError> {
        let mut h = Sha256::new();
        h.update(b"ferrum.structured-prefix-live-source.v5\0");
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
            &self.prefix_plan,
            self.prefix_plan_sha256,
        ))?);
        Ok(h.finalize().into())
    }
    /// Move common declarations into the internal shared engine. This does not
    /// serialize or replay any source4 stream, and retains source5 identity.
    pub fn into_parts(self) -> (HeaderV4, StructuredPrefixPlanV5) {
        (
            HeaderV4 {
                artifact_type: self.artifact_type,
                schema_version: self.schema_version,
                model_revision: self.model_revision,
                maximum_file_bytes: self.maximum_file_bytes,
                maximum_children: self.maximum_children,
                maximum_retained_numeric_bytes: self.maximum_retained_numeric_bytes,
                maximum_retained_coordinates: self.maximum_retained_coordinates,
                capture_protocol: self.capture_protocol,
                common: self.common,
                children: self.children,
            },
            self.prefix_plan,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Frontier {
    pub request_id: String,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub generated_tokens: u64,
    pub kv_tokens: u64,
    pub model_cache_id: Option<String>,
    pub pending_utf8: Vec<u8>,
    pub output_accepted_ordinal: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Route {
    FullLogitsSampler,
    ModelGreedyArgmax,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Commit {
    pub request_id: String,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub generated_before: u64,
    pub generated_after: u64,
    pub original_candidate: TokenId,
    pub committed_token: TokenId,
    pub route: Route,
    pub pending_before: Vec<u8>,
    pub pending_after: Vec<u8>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Offered {
    pub before: Frontier,
    pub work: PreparedWorkV2,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Completed {
    pub before: Frontier,
    pub after: Option<Frontier>,
    pub preparation_commit: Option<Commit>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Released {
    pub frontier: Frontier,
    pub original_policy_signature: [u8; 32],
    pub original_numeric_policy: HostCostPolicyV2,
    pub generated_prefix_sha256: [u8; 32],
    pub through_call_id: u64,
    pub through_fifo_ordinal: u64,
    pub actor_applied_output_ordinal: u64,
}
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum PreparationRecord {
    PreparationOffered {
        offered: u64,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        #[serde(deserialize_with = "bounded_entries")]
        rows: Vec<Offered>,
    },
    PreparationCompleted {
        offered: u64,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        reconciled: bool,
        queue: Option<Queue>,
        host_stages: Option<Stages>,
        #[serde(deserialize_with = "bounded_entries")]
        rows: Vec<Completed>,
        failure: Option<String>,
    },
    PreparationReleased {
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        slot: usize,
        receipt: Released,
    },
}
