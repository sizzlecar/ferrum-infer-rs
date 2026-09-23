//! Exact work identity is mandatory even when elapsed cost cannot be predicted.
//! These immutable observations grant no admission, backing or submit authority.
use super::{
    ActualRowWork, ActualWaveKind, ExpectedExecutionCostWave, ExpectedWaveInput, MAX_COST_ROWS,
};
use crate::model_executor::LogitsReturnPolicy;
use crate::vnext::{
    ExecutionLaneId, LogicalAdmissionCoordinatorId, PlanHash, ResourcePlanningParticipant,
    ResourcePlanningView,
};
use ferrum_types::{FerrumError, RequestId, Result};
use std::num::NonZeroU64;

/// Proposed rows in the intended physical order. The constructor below checks
/// their phase/span and binds each row to a captured real resource participant.
#[derive(Debug, Clone)]
pub struct ExpectedWorkSelection {
    pub participant_index: usize,
    pub request_id: RequestId,
    pub owner_incarnation: NonZeroU64,
    pub work_generation: NonZeroU64,
    pub input: ExpectedWaveInput,
    pub work: ActualRowWork,
    /// Required for decode, absent for prefill. Clones retain immutable host
    /// mask/history allocations so the native gate can check them in O(1).
    pub decode_policy: Option<LogitsReturnPolicy>,
}

#[derive(Debug, Clone)]
pub struct ExpectedWorkParticipant {
    selection: ExpectedWorkSelection,
    resource: ResourcePlanningParticipant,
}

impl ExpectedWorkParticipant {
    pub fn selection(&self) -> &ExpectedWorkSelection {
        &self.selection
    }

    /// Captured identity, never a permit or an assertion that capacity fits.
    /// Preparation may legitimately extend backing; the final native gate must
    /// validate the actual claim instead of comparing old free byte counts.
    pub fn resource(&self) -> &ResourcePlanningParticipant {
        &self.resource
    }

    pub fn produces_token(&self) -> bool {
        match self.selection.input {
            ExpectedWaveInput::Decode { .. } => true,
            ExpectedWaveInput::Prefill { chunk } => {
                chunk.tokens_processed() + chunk.tokens_to_process() == chunk.total_prompt_tokens()
            }
        }
    }
}

/// A checked, cost-independent description of exactly one wave. It retains no
/// sequence, lane, physical allocation or executable plan. Unknown projections
/// are allowed: the actual single Step/Invocation attempt still decides fit.
#[derive(Debug, Clone)]
pub struct ExpectedWaveWork {
    plan_hash: PlanHash,
    coordinator_id: LogicalAdmissionCoordinatorId,
    lane_id: ExecutionLaneId,
    kind: ActualWaveKind,
    query_tokens: u64,
    participants: Vec<ExpectedWorkParticipant>,
}

impl ExpectedWaveWork {
    pub fn new(
        resources: &ResourcePlanningView,
        kind: ActualWaveKind,
        selections: Vec<ExpectedWorkSelection>,
    ) -> Result<Self> {
        let invalid = || FerrumError::invalid_request("invalid exact guarded wave work");
        let lane_id = resources.lane_id().ok_or_else(invalid)?;
        if selections.is_empty()
            || selections.len() > MAX_COST_ROWS
            || selections.len() > resources.limits().maximum_participants
            || !matches!(
                kind,
                ActualWaveKind::Prefill | ActualWaveKind::Decode | ActualWaveKind::Mixed
            )
        {
            return Err(invalid());
        }
        let mut participants: Vec<ExpectedWorkParticipant> = Vec::with_capacity(selections.len());
        let mut query_tokens = 0_u64;
        let mut has_prefill = false;
        let mut has_decode = false;
        for selection in selections {
            let resource = resources
                .participants()
                .get(selection.participant_index)
                .ok_or_else(invalid)?;
            let (end, count, maximum_needed) = match (&selection.input, selection.work) {
                (
                    ExpectedWaveInput::Prefill { chunk },
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                ) if selection.decode_policy.is_none()
                    && usize::try_from(offset).ok() == Some(chunk.tokens_processed())
                    && usize::try_from(count).ok() == Some(chunk.tokens_to_process())
                    && usize::try_from(total_prompt_tokens).ok()
                        == Some(chunk.total_prompt_tokens()) =>
                {
                    has_prefill = true;
                    let end = offset
                        .checked_add(count)
                        .filter(|end| count > 0 && *end <= total_prompt_tokens)
                        .ok_or_else(invalid)?;
                    (
                        u64::from(end),
                        u64::from(count),
                        u64::from(total_prompt_tokens),
                    )
                }
                (ExpectedWaveInput::Decode { cache_id }, ActualRowWork::Decode { kv_tokens })
                    if !cache_id.is_empty() && selection.decode_policy.is_some() =>
                {
                    has_decode = true;
                    let end = kv_tokens.checked_add(1).ok_or_else(invalid)?;
                    (u64::from(end), 1, u64::from(end))
                }
                _ => return Err(invalid()),
            };
            if end > resource.maximum_tokens()
                || maximum_needed > resource.maximum_tokens()
                || participants.iter().any(|previous| {
                    previous.selection.participant_index == selection.participant_index
                        || previous.selection.request_id == selection.request_id
                        || previous.resource.authority() == resource.authority()
                        || matches!((&previous.selection.input, &selection.input),
                            (ExpectedWaveInput::Decode { cache_id: a }, ExpectedWaveInput::Decode { cache_id: b }) if a == b)
                })
            {
                return Err(invalid());
            }
            query_tokens = query_tokens.checked_add(count).ok_or_else(invalid)?;
            participants.push(ExpectedWorkParticipant {
                selection,
                resource: resource.clone(),
            });
        }
        if !matches!(
            (kind, has_prefill, has_decode),
            (ActualWaveKind::Prefill, true, false)
                | (ActualWaveKind::Decode, false, true)
                | (ActualWaveKind::Mixed, true, true)
        ) {
            return Err(invalid());
        }
        Ok(Self {
            plan_hash: resources.plan_hash().clone(),
            coordinator_id: resources.coordinator_id(),
            lane_id,
            kind,
            query_tokens,
            participants,
        })
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }
    pub fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }
    pub fn kind(&self) -> ActualWaveKind {
        self.kind
    }
    pub fn query_tokens(&self) -> u64 {
        self.query_tokens
    }
    pub fn participants(&self) -> &[ExpectedWorkParticipant] {
        &self.participants
    }
}

/// Explicitly recorded reasons for choosing safe completion without claiming a
/// time witness. This does not authorize abandoning physical or output limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionOnlyReason {
    CostUnavailable,
    WitnessExpired,
    SearchInconclusive,
    ExistingSloMiss,
}

#[derive(Debug, Clone)]
pub enum WaveCommitment {
    CostWitness(ExpectedExecutionCostWave),
    CompleteRequests(CompletionOnlyReason),
}

/// Both commitments carry the same mandatory checked physical work. A caller
/// cannot attach an unrelated witness to a separately constructed work list.
#[derive(Debug, Clone)]
pub struct ExpectedExecutionWave {
    work: ExpectedWaveWork,
    commitment: WaveCommitment,
}

impl ExpectedExecutionWave {
    pub fn from_cost_witness<'p>(
        witness: ExpectedExecutionCostWave,
        mut decode_policy: impl FnMut(&RequestId) -> Option<&'p LogitsReturnPolicy>,
    ) -> Result<Self> {
        let selections = witness
            .participants()
            .iter()
            .zip(&witness.canonical().rows)
            .map(|(row, work)| {
                Ok(ExpectedWorkSelection {
                    participant_index: row.participant_index,
                    request_id: row.request_id.clone(),
                    owner_incarnation: NonZeroU64::new(row.host.owner_incarnation).ok_or_else(
                        || FerrumError::invalid_request("missing guarded owner incarnation"),
                    )?,
                    work_generation: NonZeroU64::new(row.host.work_generation).ok_or_else(
                        || FerrumError::invalid_request("missing guarded work generation"),
                    )?,
                    input: row.input.clone(),
                    work: *work,
                    decode_policy: match row.input {
                        ExpectedWaveInput::Decode { .. } => decode_policy(&row.request_id).cloned(),
                        ExpectedWaveInput::Prefill { .. } => None,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let work = ExpectedWaveWork::new(
            witness.route_view().resource_view(),
            witness.canonical().kind,
            selections,
        )?;
        Ok(Self {
            work,
            commitment: WaveCommitment::CostWitness(witness),
        })
    }

    pub fn complete_requests(work: ExpectedWaveWork, reason: CompletionOnlyReason) -> Self {
        Self {
            work,
            commitment: WaveCommitment::CompleteRequests(reason),
        }
    }

    pub fn work(&self) -> &ExpectedWaveWork {
        &self.work
    }

    pub fn commitment(&self) -> &WaveCommitment {
        &self.commitment
    }
}
