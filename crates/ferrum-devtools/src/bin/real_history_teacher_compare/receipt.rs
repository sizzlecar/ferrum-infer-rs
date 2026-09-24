//! Typed projections preserve the producer's field order for hash validation.
use super::*;
use ferrum_interfaces::vnext::{
    ExecutionIdentityEnvelope, ProviderExecutionSemantics, UnvalidatedExecutionIdentityParts,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Layout {
    pub element_type: String,
    pub element_count: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ReadbackRequest {
    pub node_id: String,
    pub participant_index: u32,
    pub resource_id: String,
    pub expected_usage: String,
    pub logical_offset_bytes: u64,
    pub output_layout: Layout,
}

/// One fixed plan has one product output range; only the participant index
/// varies across owners and physical waves.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct OutputBinding {
    node_id: String,
    resource_id: String,
    expected_usage: String,
    logical_offset_bytes: u64,
    output_layout: Layout,
}

impl OutputBinding {
    pub(super) fn resource_id(&self) -> &str {
        &self.resource_id
    }

    /// Only the explicitly pinned numerical-profile transition may use this
    /// projection. Each arm's complete binding is still validated per wave.
    pub(super) fn same_product_range(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.expected_usage == other.expected_usage
            && self.logical_offset_bytes == other.logical_offset_bytes
            && self.output_layout == other.output_layout
    }
}

impl From<&ReadbackRequest> for OutputBinding {
    fn from(request: &ReadbackRequest) -> Self {
        Self {
            node_id: request.node_id.clone(),
            resource_id: request.resource_id.clone(),
            expected_usage: request.expected_usage.clone(),
            logical_offset_bytes: request.logical_offset_bytes,
            output_layout: request.output_layout.clone(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Authority {
    pub sparse_id: u32,
    pub generation: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct NodeKey {
    pub sequence_authority: Authority,
    pub request_authority: Authority,
    pub frame_id: Value,
    pub node_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Participant {
    pub participant_index: usize,
    pub node_key: NodeKey,
    pub identity: UnvalidatedExecutionIdentityParts,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Node {
    pub node_index: usize,
    pub node_id: String,
    pub operation_id: String,
    pub provider_id: String,
    pub provider_implementation_fingerprint: String,
    pub provider_execution_semantics: ProviderExecutionSemantics,
    pub work_shape_fingerprint: String,
    pub participants: Vec<Participant>,
    pub fingerprint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Batch {
    pub batch_step_id: Value,
    pub batch_invocation_id: Value,
    pub plan_id: String,
    pub plan_hash: String,
    pub device_id: String,
    pub runtime_implementation_fingerprint: String,
    pub lane_id: Value,
    pub claimed_backing_fingerprint: String,
    pub nodes: Vec<Node>,
    pub participants: Vec<Participant>,
    pub fingerprint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SubmissionParticipant {
    pub slot_id: Value,
    pub participant_index: usize,
    pub identity: UnvalidatedExecutionIdentityParts,
    pub batch_submission_fingerprint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Submission {
    pub slot_id: Value,
    pub batch_identity: Batch,
    pub participants: Vec<SubmissionParticipant>,
    pub fingerprint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CompletionParticipant {
    pub submission: SubmissionParticipant,
    pub disposition: Value,
    pub batch_completion_fingerprint: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Completion {
    pub submission: Submission,
    pub disposition: Value,
    pub fence_timing: Value,
    pub submission_timing: Value,
    pub participants: Vec<CompletionParticipant>,
    pub fingerprint: String,
}

fn digest(value: &impl Serialize) -> Result<String> {
    Ok(sha256(&serde_json::to_vec(value)?))
}

pub(super) fn node_fingerprint(node: &Node) -> Result<String> {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        node_index: usize,
        node_id: &'a str,
        operation_id: &'a str,
        provider_id: &'a str,
        provider_implementation_fingerprint: &'a str,
        provider_execution_semantics: ProviderExecutionSemantics,
        work_shape_fingerprint: &'a str,
        participants: &'a [Participant],
    }
    digest(&Input {
        domain: "ferrum.runtime-vnext.batch-operation-node-identity.v2",
        node_index: node.node_index,
        node_id: &node.node_id,
        operation_id: &node.operation_id,
        provider_id: &node.provider_id,
        provider_implementation_fingerprint: &node.provider_implementation_fingerprint,
        provider_execution_semantics: node.provider_execution_semantics,
        work_shape_fingerprint: &node.work_shape_fingerprint,
        participants: &node.participants,
    })
}

pub(super) fn submission_fingerprint(submission: &Submission) -> Result<String> {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        slot_id: &'a Value,
        batch_identity_fingerprint: &'a str,
    }
    digest(&Input {
        domain: "ferrum.runtime-vnext.batch-operation-submission.v1",
        slot_id: &submission.slot_id,
        batch_identity_fingerprint: &submission.batch_identity.fingerprint,
    })
}

pub(super) fn completion_fingerprint(completion: &Completion) -> Result<String> {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        submission_fingerprint: &'a str,
        disposition: &'a Value,
    }
    digest(&Input {
        domain: "ferrum.runtime-vnext.batch-operation-completion.v1",
        submission_fingerprint: &completion.submission.fingerprint,
        disposition: &completion.disposition,
    })
}

pub(super) fn readback_fingerprint(
    completion: &str,
    readbacks: &[VNextTeacherReadbackEvidence],
) -> Result<String> {
    #[derive(Serialize)]
    struct Detail {
        request: ReadbackRequest,
        output_sha256: String,
    }
    #[derive(Serialize)]
    struct Disposition {
        status: &'static str,
        detail: Detail,
    }
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        completion_fingerprint: &'a str,
        dispositions: Vec<Disposition>,
    }
    let dispositions = readbacks
        .iter()
        .map(|readback| {
            Ok(Disposition {
                status: "succeeded",
                detail: Detail {
                    request: serde_json::from_value(readback.request.clone())?,
                    output_sha256: readback.sha256.clone(),
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    digest(&Input {
        domain: "ferrum.runtime-vnext.completion-readback-batch.v2",
        completion_fingerprint: completion,
        dispositions,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct NodeSignature {
    pub node_id: String,
    pub operation_id: String,
    pub provider_id: String,
    pub implementation_fingerprint: String,
    pub execution_semantics: ProviderExecutionSemantics,
}

pub(super) struct CheckedReceipt {
    pub node_signature: Vec<NodeSignature>,
    pub owners: BTreeMap<String, (String, Authority, Authority)>,
    pub plan_id: String,
    pub plan_hash: String,
    pub runtime: String,
    pub run_id: String,
    pub output_binding: OutputBinding,
    pub batch_step_id: u64,
    pub batch_invocation_id: u64,
}

pub(super) fn validate(
    directory: &Path,
    wave: &VNextTeacherWaveEvidence,
) -> Result<CheckedReceipt> {
    let artifact = wave
        .completion_receipt
        .as_ref()
        .context("actual completion receipt sidecar missing")?;
    let bytes = files::artifact(directory, &artifact.file, artifact.bytes, &artifact.sha256)?;
    let completion: Completion =
        serde_json::from_slice(&bytes).context("parse actual completion receipt")?;
    ensure!(
        completion.disposition == json!({"status":"succeeded"}),
        "physical wave did not complete successfully"
    );
    ensure!(
        completion.fingerprint == wave.completion_fingerprint
            && completion_fingerprint(&completion)? == completion.fingerprint,
        "completion fingerprint differs"
    );
    let submitted = &completion.submission;
    ensure!(
        submission_fingerprint(submitted)? == submitted.fingerprint,
        "submission fingerprint differs"
    );
    ensure!(
        readback_fingerprint(&completion.fingerprint, &wave.readbacks)? == wave.receipt_fingerprint,
        "readback receipt fingerprint differs"
    );
    let batch = &submitted.batch_identity;
    let batch_step_id = batch
        .batch_step_id
        .as_u64()
        .filter(|id| *id > 0)
        .context("physical batch step ID is not a nonzero integer")?;
    let batch_invocation_id = batch
        .batch_invocation_id
        .as_u64()
        .filter(|id| *id > 0)
        .context("physical batch invocation ID is not a nonzero integer")?;
    ensure!(
        !batch.nodes.is_empty()
            && files::canonical_sha(&batch.fingerprint)
            && files::canonical_sha(&batch.plan_hash)
            && files::canonical_sha(&batch.runtime_implementation_fingerprint)
            && files::canonical_sha(&batch.claimed_backing_fingerprint),
        "incomplete physical batch root identity"
    );
    let mut flattened = Vec::new();
    let mut owners = BTreeMap::new();
    let mut run = None;
    let mut node_ids = BTreeSet::new();
    for (node_index, node) in batch.nodes.iter().enumerate() {
        ensure!(
            node.node_index == node_index && node_ids.insert(node.node_id.clone()),
            "node inventory is reordered or duplicated"
        );
        ensure!(
            node.participants.len() == wave.participant_count,
            "physical node width differs from the logical owner batch"
        );
        ensure!(
            node_fingerprint(node)? == node.fingerprint,
            "node {} fingerprint differs",
            node.node_id
        );
        for (logical, part) in node.participants.iter().enumerate() {
            ensure!(
                part.participant_index == flattened.len(),
                "node×owner flattened index is not contiguous"
            );
            let owner = wave
                .participants
                .iter()
                .find(|owner| owner.participant_index == logical)
                .context("physical node has no logical owner")?;
            let identity = &part.identity;
            ExecutionIdentityEnvelope::new(identity.clone().into()).map_err(anyhow::Error::msg)?;
            ensure!(
                identity.request_id.to_string()
                    == format!("request.diagnostic.{}", owner.request_id),
                "node participant points at another request owner"
            );
            ensure!(
                identity
                    .plan_id
                    .as_ref()
                    .map(ToString::to_string)
                    .as_deref()
                    == Some(batch.plan_id.as_str())
                    && identity
                        .plan_hash
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref()
                        == Some(batch.plan_hash.as_str())
                    && identity
                        .node_id
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref()
                        == Some(node.node_id.as_str())
                    && identity
                        .operation_id
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref()
                        == Some(node.operation_id.as_str())
                    && identity
                        .provider_id
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref()
                        == Some(node.provider_id.as_str())
                    && identity
                        .device_id
                        .as_ref()
                        .map(ToString::to_string)
                        .as_deref()
                        == Some(batch.device_id.as_str())
                    && identity.runtime_implementation_fingerprint.as_deref()
                        == Some(batch.runtime_implementation_fingerprint.as_str()),
                "node participant projection differs from batch/node identity"
            );
            ensure!(
                part.node_key.node_id == node.node_id
                    && serde_json::to_value(identity.frame_id)? == part.node_key.frame_id,
                "node key differs from execution frame"
            );
            let binding = identity
                .active_sequence_fingerprint
                .as_ref()
                .filter(|sha| files::canonical_sha(sha))
                .context("node lacks actual active sequence binding")?;
            let state = (
                binding.clone(),
                part.node_key.sequence_authority.clone(),
                part.node_key.request_authority.clone(),
            );
            if let Some(prior) = owners.insert(owner.owner_id.clone(), state.clone()) {
                ensure!(
                    prior == state,
                    "owner changes active authority between nodes"
                );
            }
            let run_id = identity.run_id.to_string();
            if let Some(prior) = &run {
                ensure!(prior == &run_id, "wave contains multiple run identities");
            } else {
                run = Some(run_id);
            }
            flattened.push(part.clone());
        }
    }
    ensure!(
        flattened == batch.participants,
        "batch flattened participants differ from real nodes"
    );
    ensure!(
        submitted.participants.len() == flattened.len()
            && completion.participants.len() == flattened.len(),
        "submission/completion participant inventory is incomplete"
    );
    for (index, ((part, submitted_part), completed_part)) in flattened
        .iter()
        .zip(&submitted.participants)
        .zip(&completion.participants)
        .enumerate()
    {
        ensure!(
            submitted_part.participant_index == index
                && submitted_part.slot_id == submitted.slot_id
                && submitted_part.batch_submission_fingerprint == submitted.fingerprint
                && submitted_part.identity == part.identity,
            "submission participant projection differs"
        );
        ensure!(
            completed_part.submission == *submitted_part
                && completed_part.disposition == json!({"status":"succeeded"})
                && completed_part.batch_completion_fingerprint == completion.fingerprint,
            "completion participant projection differs"
        );
    }
    let mut outputs = BTreeSet::new();
    let mut output_binding = None;
    for readback in &wave.readbacks {
        let request: ReadbackRequest = serde_json::from_value(readback.request.clone())?;
        ensure!(
            request.participant_index as usize == readback.participant_index
                && outputs.insert(readback.participant_index)
                && readback.participant_index < wave.participant_count,
            "readback duplicates or changes a logical participant"
        );
        ensure!(
            request.expected_usage == "activations"
                && node_ids.contains(&request.node_id)
                && !request.resource_id.is_empty(),
            "readback is not an actual activation output node"
        );
        let binding = OutputBinding::from(&request);
        if let Some(prior) = &output_binding {
            ensure!(
                prior == &binding,
                "physical owners have different product output bindings"
            );
        } else {
            output_binding = Some(binding);
        }
    }
    ensure!(
        outputs.len() == wave.participant_count,
        "full-logits readback omits an owner"
    );
    Ok(CheckedReceipt {
        node_signature: batch
            .nodes
            .iter()
            .map(|node| NodeSignature {
                node_id: node.node_id.clone(),
                operation_id: node.operation_id.clone(),
                provider_id: node.provider_id.clone(),
                implementation_fingerprint: node.provider_implementation_fingerprint.clone(),
                execution_semantics: node.provider_execution_semantics,
            })
            .collect(),
        owners,
        plan_id: batch.plan_id.clone(),
        plan_hash: batch.plan_hash.clone(),
        runtime: batch.runtime_implementation_fingerprint.clone(),
        run_id: run.context("missing actual run")?,
        output_binding: output_binding.context("missing product output binding")?,
        batch_step_id,
        batch_invocation_id,
    })
}
