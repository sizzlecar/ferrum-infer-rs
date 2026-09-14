//! Plan-derived checkpoint closure. A layout authorizes byte planning only;
//! completed-frame evidence and session transfer arbitration are still required
//! before any state is captured or imported.

use serde::{Deserialize, Serialize};

use super::{
    canonical_fingerprint, invalid_plan, AllocationLifetime, DynamicResourceDescriptor,
    DynamicStorageContract, ExecutionPlan, NodeId, PlanHash, ProgramValueId, ResolvedTensorSpec,
    ResourceId, StateId, StateInitialization, VNextError,
};
use crate::vnext::{
    CheckpointCompletedInputCapture, CheckpointInputDependency, ContractVersion,
    ProgramCheckpointInputs, ProviderCheckpointContract, ProviderCheckpointStateLayout, ProviderId,
    StateCheckpointContract,
};

mod derive;
mod output_only;
pub(super) use derive::derive_sequence_checkpoint;
mod ranges;
pub use ranges::{
    SequenceCheckpointBytePlan, SequenceCheckpointCopyRange, SequenceCheckpointResourceRanges,
};
mod inputs;
pub use inputs::{CheckpointCanonicalInput, SequenceCheckpointInputIdentity};

pub const SEQUENCE_CHECKPOINT_LAYOUT_VERSION: ContractVersion = ContractVersion::new(1, 0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceCheckpointUnsupportedReason {
    InputsUndeclared,
    InputCoverage,
    OutputOnlyInputAffectsState {
        value_id: ProgramValueId,
        node_id: NodeId,
        state_id: StateId,
    },
    MissingInput {
        value_id: ProgramValueId,
    },
    InvalidInput {
        value_id: ProgramValueId,
    },
    NoSequenceState,
    StateUndeclared {
        state_id: StateId,
    },
    StateLifetime {
        state_id: StateId,
        lifetime: AllocationLifetime,
    },
    StateWithoutWriter {
        state_id: StateId,
    },
    ProviderUndeclared {
        node_id: NodeId,
        provider_id: ProviderId,
    },
    ProviderPersistentWorkspace {
        node_id: NodeId,
    },
    BoundaryIntersection,
    StateOracleCoverage {
        node_id: NodeId,
    },
    StatePortUndeclared {
        node_id: NodeId,
        state_id: StateId,
    },
    InvalidStatePort {
        node_id: NodeId,
    },
    StateLayout {
        state_id: StateId,
        reason: String,
    },
    UncoveredResource {
        resource_id: ResourceId,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum SequenceCheckpointCapability<'a> {
    Enabled(&'a SequenceCheckpointLayout),
    Unsupported(&'a [SequenceCheckpointUnsupportedReason]),
}

/// Every actual selected operation participates, including stateless operations
/// that can affect prefix results or legality. An unselected provider cannot
/// establish this contract for the plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceCheckpointProvider {
    node_id: NodeId,
    provider_id: ProviderId,
    operation_fingerprint: String,
    implementation_fingerprint: String,
    contract: ProviderCheckpointContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceCheckpointState {
    state_id: StateId,
    value_id: ProgramValueId,
    writers: Vec<NodeId>,
    semantics: StateCheckpointContract,
    tensor: ResolvedTensorSpec,
    resource_id: ResourceId,
    offset_bytes: u64,
    layout: ProviderCheckpointStateLayout,
    storage: DynamicStorageContract,
    initialization: StateInitialization,
    descriptor: DynamicResourceDescriptor,
}

impl SequenceCheckpointState {
    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn writers(&self) -> &[NodeId] {
        &self.writers
    }
    pub fn semantics(&self) -> StateCheckpointContract {
        self.semantics
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }
    pub fn storage(&self) -> &DynamicStorageContract {
        &self.storage
    }
    /// The Sequence initialization cell is identified by this base resource;
    /// every projection of that cell must be imported before it is committed.
    pub fn initialization(&self) -> StateInitialization {
        self.initialization
    }
    pub fn layout(&self) -> ProviderCheckpointStateLayout {
        self.layout
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SequenceCheckpointLayoutData {
    contract_version: ContractVersion,
    inputs: ProgramCheckpointInputs,
    input_dependency: CheckpointInputDependency,
    boundaries: crate::vnext::CheckpointBoundaryConstraint,
    #[serde(
        default,
        skip_serializing_if = "CheckpointCompletedInputCapture::is_unsupported"
    )]
    completed_input_capture: CheckpointCompletedInputCapture,
    providers: Vec<SequenceCheckpointProvider>,
    states: Vec<SequenceCheckpointState>,
}

impl SequenceCheckpointLayoutData {
    pub(super) fn validate_version(&self) -> Result<(), VNextError> {
        if self.contract_version != SEQUENCE_CHECKPOINT_LAYOUT_VERSION {
            return Err(invalid_plan(
                "unsupported sequence checkpoint layout version",
            ));
        }
        Ok(())
    }
}

/// Constructed only from a trusted plan build. Public deserialization cannot
/// create this authority; plan wire data must survive a full semantic rebuild.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SequenceCheckpointLayout {
    data: SequenceCheckpointLayoutData,
}

impl SequenceCheckpointLayout {
    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(&self.data, "fingerprint sequence checkpoint layout")
    }
    pub fn inputs(&self) -> &ProgramCheckpointInputs {
        &self.data.inputs
    }
    pub fn input_dependency(&self) -> CheckpointInputDependency {
        self.data.input_dependency
    }
    /// Aggregate constraint on the actual span ending at a capture boundary.
    pub fn capture_span_constraint(
        &self,
    ) -> super::super::operation::CheckpointTokenSpanConstraint {
        self.data.boundaries.prefix()
    }
    /// Greatest shared boundary reachable from the source's actual retired
    /// offset, with a legal nonempty suffix for every participant.
    pub fn shared_prefix_boundary(
        &self,
        processed: u64,
        source_prompt: u64,
        common_prefix: u64,
        follower_prompts: &[u64],
    ) -> Option<u64> {
        if self.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
            || follower_prompts.is_empty()
        {
            return None;
        }
        self.latest_reusable_boundary(processed, source_prompt, common_prefix, follower_prompts)
    }

    /// Nearest boundary before the end of this input that can serve an exact
    /// repeat while leaving a legal nonempty suffix to execute for logits.
    /// This is pure planning, not a capture or capacity reservation.
    pub fn prompt_tail_boundary(&self, processed: u64, prompt: u64) -> Option<u64> {
        if self.input_dependency() != CheckpointInputDependency::ExactTokenPrefix {
            return None;
        }
        self.latest_reusable_boundary(processed, prompt, prompt, &[])
    }

    fn latest_reusable_boundary(
        &self,
        processed: u64,
        source_prompt: u64,
        common_prefix: u64,
        follower_prompts: &[u64],
    ) -> Option<u64> {
        let prefix = self.data.boundaries.prefix();
        let suffix = self.data.boundaries.suffix();
        let prefix_alignment = prefix.alignment().get();
        let suffix_alignment = suffix.alignment().get();
        let suffix_residue = source_prompt % suffix_alignment;
        let mut upper =
            common_prefix.min(source_prompt.checked_sub(suffix.minimum_tokens().get())?);
        for &prompt in follower_prompts {
            // All suffixes share one alignment. Incompatible residues cannot
            // acquire a common boundary, regardless of common-prefix length.
            if prompt % suffix_alignment != suffix_residue {
                return None;
            }
            upper = upper.min(prompt.checked_sub(suffix.minimum_tokens().get())?);
        }
        let lower = processed.checked_add(prefix.minimum_tokens().get())?;
        if lower > upper {
            return None;
        }
        let (mut divisor, mut remainder) = (prefix_alignment, suffix_alignment);
        while remainder != 0 {
            (divisor, remainder) = (remainder, divisor % remainder);
        }
        if processed % divisor != source_prompt % divisor {
            return None;
        }
        // Search only the sparser of the two alignment lattices. This skips
        // invalid token positions without multiplying alignments or overflowing
        // an LCM. The aggregate contract remains the final authority.
        let (step, residue) = if prefix_alignment >= suffix_alignment {
            (prefix_alignment, processed % prefix_alignment)
        } else {
            (suffix_alignment, suffix_residue)
        };
        let remainder = upper % step;
        let adjustment = if remainder >= residue {
            remainder - residue
        } else {
            step - (residue - remainder)
        };
        let mut boundary = upper.checked_sub(adjustment)?;
        while boundary >= lower {
            if self.permits_capture_from(processed, boundary, source_prompt) {
                return Some(boundary);
            }
            boundary = boundary.checked_sub(step)?;
        }
        None
    }
    pub fn states(&self) -> &[SequenceCheckpointState] {
        &self.data.states
    }
    pub fn providers(&self) -> &[SequenceCheckpointProvider] {
        &self.data.providers
    }
    pub fn permits_capture_from(&self, processed: u64, boundary: u64, prompt: u64) -> bool {
        if boundary == prompt {
            return self.data.completed_input_capture == CheckpointCompletedInputCapture::Supported
                && boundary
                    .checked_sub(processed)
                    .is_some_and(|span| self.data.boundaries.prefix().permits(span));
        }
        self.data
            .boundaries
            .permits_from(processed, boundary, prompt)
    }
    pub fn completed_input_capture(&self) -> CheckpointCompletedInputCapture {
        self.data.completed_input_capture
    }
    pub fn permits_suffix(&self, boundary: u64, prompt: u64) -> bool {
        boundary > 0
            && prompt
                .checked_sub(boundary)
                .is_some_and(|suffix| self.data.boundaries.suffix().permits(suffix))
    }
}

impl SequenceCheckpointProvider {
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }
    pub fn contract(&self) -> &ProviderCheckpointContract {
        &self.contract
    }
}

impl ExecutionPlan {
    pub fn sequence_checkpoint_capability(&self) -> SequenceCheckpointCapability<'_> {
        match &self.payload.sequence_checkpoint_layout {
            Some(layout) => SequenceCheckpointCapability::Enabled(layout),
            None => SequenceCheckpointCapability::Unsupported(&self.checkpoint_unsupported_reasons),
        }
    }

    /// Pure, checked allocation/copy-range planning. N is not a proof that a
    /// source has actually completed that boundary; transfer requires its own
    /// completed-frame authority and checks the source's actual backing.
    pub fn checkpoint_byte_plan(
        &self,
        boundary: u64,
    ) -> Result<SequenceCheckpointBytePlan, VNextError> {
        self.payload
            .sequence_checkpoint_layout
            .as_ref()
            .ok_or_else(|| invalid_plan("plan does not support sequence checkpoints"))?
            .byte_plan(self.plan_hash.clone(), boundary)
    }
}
