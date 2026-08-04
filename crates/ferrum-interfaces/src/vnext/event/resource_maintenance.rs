use serde::Serialize;
use std::collections::BTreeSet;

use crate::vnext::{
    DynamicPoolGrowthBatchReceipt, RequestIdentity, RunId, SequenceAuthorityId,
    TrustedActiveSequenceBinding, TrustedPlanRuntimeEvidence, VNextError,
};

use super::{canonical_fingerprint, invalid_event};

pub const EXECUTION_RESOURCE_MAINTENANCE_EVENT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionResourceMaintenanceStage {
    SequenceExtension,
    StepAdmission,
    SubmissionWave,
}

impl ExecutionResourceMaintenanceStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SequenceExtension => "sequence_extension",
            Self::StepAdmission => "step_admission",
            Self::SubmissionWave => "submission_wave",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ExecutionResourceMaintenanceParticipant {
    run_id: RunId,
    request_id: RequestIdentity,
    sequence_authority: SequenceAuthorityId,
    active_sequence_fingerprint: String,
}

impl ExecutionResourceMaintenanceParticipant {
    fn from_active(active: &TrustedActiveSequenceBinding) -> Self {
        Self {
            run_id: active.run_id().clone(),
            request_id: active.request_id().clone(),
            sequence_authority: active.sequence_authority(),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
        }
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub fn active_sequence_fingerprint(&self) -> &str {
        &self.active_sequence_fingerprint
    }
}

/// Allocator-issued proof for one successful post-admission backing mutation.
///
/// This event is plan/batch scoped instead of belonging to one request
/// lifecycle cursor. A multi-request wave emits one event with a canonical
/// participant set and one allocator receipt, preventing physical growth or
/// reclaim from being counted once per participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundExecutionResourceMaintenance {
    schema_version: u32,
    plan: TrustedPlanRuntimeEvidence,
    stage: ExecutionResourceMaintenanceStage,
    participants: Box<[ExecutionResourceMaintenanceParticipant]>,
    receipt: DynamicPoolGrowthBatchReceipt,
    #[serde(skip)]
    fingerprint: String,
}

impl BoundExecutionResourceMaintenance {
    pub fn bind<'a>(
        stage: ExecutionResourceMaintenanceStage,
        participants: impl IntoIterator<Item = &'a TrustedActiveSequenceBinding>,
        receipt: DynamicPoolGrowthBatchReceipt,
    ) -> Result<Self, VNextError> {
        if receipt.growths().is_empty() || receipt.capacity_epoch() == 0 {
            return Err(invalid_event(
                "execution resource maintenance requires installed backing growth",
            ));
        }
        let mut growth_pool_ids = BTreeSet::new();
        let mut chunk_ids = BTreeSet::new();
        if receipt.growths().iter().any(|growth| {
            growth.chunk_bytes() == 0
                || growth.published_capacity_bytes() < growth.chunk_bytes()
                || growth.capacity_epoch() != receipt.capacity_epoch()
                || growth.chunk().pool_id() != growth.pool_id()
                || !growth_pool_ids.insert(growth.pool_id().clone())
                || !chunk_ids.insert(growth.chunk().clone())
        }) {
            return Err(invalid_event(
                "execution resource maintenance contains invalid or duplicate growth evidence",
            ));
        }
        if let Some(rebalance) = receipt.rebalance() {
            let mut reclaimed_pool_ids = BTreeSet::new();
            let mut reclaimed_chunk_ids = BTreeSet::new();
            let detailed_chunks = rebalance.pools().iter().try_fold(0_usize, |total, pool| {
                total.checked_add(pool.chunks().len())
            });
            let detailed_bytes = rebalance.pools().iter().try_fold(0_u64, |total, pool| {
                total.checked_add(pool.reclaimed_bytes())
            });
            if rebalance.pools().is_empty()
                || rebalance.reclaimed_chunks() == 0
                || rebalance.reclaimed_bytes() == 0
                || rebalance.logical_capacity_epoch() == 0
                || rebalance.plan_device_capacity_epoch() == 0
                || rebalance.process_device_capacity_epoch() == 0
                || detailed_chunks != Some(rebalance.reclaimed_chunks())
                || detailed_bytes != Some(rebalance.reclaimed_bytes())
                || rebalance.pools().iter().any(|pool| {
                    pool.chunks().is_empty()
                        || pool.reclaimed_bytes() == 0
                        || !reclaimed_pool_ids.insert(pool.pool_id().clone())
                        || pool.chunks().iter().any(|chunk| {
                            chunk.pool_id() != pool.pool_id()
                                || !reclaimed_chunk_ids.insert(chunk.clone())
                        })
                })
            {
                return Err(invalid_event(
                    "execution resource maintenance contains invalid rebalance evidence",
                ));
            }
            let boundary = receipt.maintenance_boundary().ok_or_else(|| {
                invalid_event(
                    "execution resource rebalance requires its pre-mutation boundary receipt",
                )
            })?;
            let selected = boundary
                .selected_chunks()
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            if !boundary.reclaim_sufficient()
                || boundary.selected_bytes() != rebalance.reclaimed_bytes()
                || selected.len() != boundary.selected_chunks().len()
                || selected != reclaimed_chunk_ids
            {
                return Err(invalid_event(
                    "execution resource rebalance differs from its maintenance boundary",
                ));
            }
        } else if receipt.maintenance_boundary().is_some() {
            return Err(invalid_event(
                "successful execution maintenance boundary requires a rebalance receipt",
            ));
        }

        let mut plan = None;
        let mut bound = Vec::new();
        for active in participants {
            active.ensure_open_for_emission()?;
            match &plan {
                Some(expected) if expected != active.plan() => {
                    return Err(invalid_event(
                        "execution resource maintenance participants differ in plan authority",
                    ));
                }
                None => plan = Some(active.plan().clone()),
                Some(_) => {}
            }
            bound.push(ExecutionResourceMaintenanceParticipant::from_active(active));
        }
        let plan = plan.ok_or_else(|| {
            invalid_event("execution resource maintenance requires at least one participant")
        })?;
        if receipt.coordinator_id() != plan.coordinator_id() {
            return Err(invalid_event(
                "execution resource maintenance receipt belongs to a different coordinator",
            ));
        }
        let mut participant_authorities = BTreeSet::new();
        if bound.iter().any(|participant| {
            !participant_authorities.insert((
                participant.run_id.clone(),
                participant.request_id.clone(),
                participant.sequence_authority,
            ))
        }) {
            return Err(invalid_event(
                "execution resource maintenance contains duplicate participants",
            ));
        }
        bound.sort();

        let mut event = Self {
            schema_version: EXECUTION_RESOURCE_MAINTENANCE_EVENT_SCHEMA_VERSION,
            plan,
            stage,
            participants: bound.into_boxed_slice(),
            receipt,
            fingerprint: String::new(),
        };
        event.fingerprint = canonical_fingerprint(&event);
        Ok(event)
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub const fn stage(&self) -> ExecutionResourceMaintenanceStage {
        self.stage
    }

    pub fn participants(&self) -> &[ExecutionResourceMaintenanceParticipant] {
        &self.participants
    }

    pub fn receipt(&self) -> &DynamicPoolGrowthBatchReceipt {
        &self.receipt
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}
