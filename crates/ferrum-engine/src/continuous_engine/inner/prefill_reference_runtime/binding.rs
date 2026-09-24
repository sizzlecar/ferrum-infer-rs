use super::*;
use ferrum_interfaces::model_executor::PrefillChunk;
use ferrum_scheduler::implementations::continuous::{
    prefill_reference::{ReferenceIdentity, RequestPrefillReferenceBinding},
    slo_planner::{PlanningTimeError, PlanningTimeOrigin, PrefillProgressView},
};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) enum ReferenceBindingUnknown {
    UntrustedTiming,
    IncarnationExhausted,
    LengthOverflow,
    InvalidReceipt,
    StaleReceipt,
    Reference(ReferenceUnknown),
    Time(PlanningTimeError),
}
impl From<ReferenceUnknown> for ReferenceBindingUnknown {
    fn from(value: ReferenceUnknown) -> Self {
        Self::Reference(value)
    }
}
impl From<PlanningTimeError> for ReferenceBindingUnknown {
    fn from(value: PlanningTimeError) -> Self {
        Self::Time(value)
    }
}

#[derive(Debug)]
pub(in crate::continuous_engine) enum SequencePrefillReference {
    Known(SequenceReferenceBinding),
    Unknown(ReferenceBindingUnknown),
}
impl SequencePrefillReference {
    pub fn known(&self) -> std::result::Result<&SequenceReferenceBinding, ReferenceBindingUnknown> {
        match self {
            Self::Known(binding) => {
                if let Some(reason) = binding.invalid {
                    Err(reason)
                } else {
                    Ok(binding)
                }
            }
            Self::Unknown(reason) => Err(*reason),
        }
    }
}

#[derive(Debug)]
pub(in crate::continuous_engine) struct SequenceReferenceBinding {
    calibration: Arc<LoadedPrefillReference>,
    binding: RequestPrefillReferenceBinding,
    request_id: RequestId,
    owner: Arc<()>,
    incarnation: NonZeroU64,
    generation: NonZeroU64,
    total: NonZeroU32,
    ingress: Instant,
    admitted_at: Instant,
    invalid: Option<ReferenceBindingUnknown>,
}

impl SequenceReferenceBinding {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        calibration: Arc<LoadedPrefillReference>,
        request_id: RequestId,
        owner: Arc<()>,
        incarnation: NonZeroU64,
        total: NonZeroU32,
        ingress: Instant,
        admitted_at: Instant,
        first_deadline: Instant,
        verified_prefix: u32,
    ) -> std::result::Result<Self, ReferenceBindingUnknown> {
        // A per-request epoch accepts trusted ingress preceding engine startup.
        // All later projections keep these original Instants and work units.
        let local = PlanningTimeOrigin::from_origin(ingress, admitted_at)?;
        let binding = calibration.bind(
            request_id.clone(),
            incarnation,
            total,
            local.at_ns(admitted_at)?,
            local.at_ns(first_deadline)?,
            verified_prefix,
        )?;
        Ok(Self {
            calibration,
            binding,
            request_id,
            owner,
            incarnation,
            generation: NonZeroU64::new(1).unwrap(),
            total,
            ingress,
            admitted_at,
            invalid: None,
        })
    }

    pub fn binding(&self) -> &RequestPrefillReferenceBinding {
        &self.binding
    }
    pub fn identity(&self) -> ReferenceIdentity {
        self.calibration.identity()
    }
    pub fn tau_ref_ns(&self) -> NonZeroU64 {
        self.calibration.tau_ref_ns()
    }
    pub fn total_prompt_tokens(&self) -> NonZeroU32 {
        self.total
    }

    /// Checkpoints are in this snapshot's clock, which may change when another
    /// earlier request leaves. Convert through original Instants, not a rebind.
    pub fn project_progress(
        &self,
        snapshot: &PlanningTimeOrigin,
        offset: u32,
        executable_until: u32,
        checkpoints: &[u64],
    ) -> std::result::Result<PrefillProgressView, ReferenceBindingUnknown> {
        self.project_progress_with_granule(snapshot, offset, executable_until, checkpoints, None)
    }
    pub fn project_progress_with_granule(
        &self,
        snapshot: &PlanningTimeOrigin,
        offset: u32,
        executable_until: u32,
        checkpoints: &[u64],
        legal_granule: Option<NonZeroU32>,
    ) -> std::result::Result<PrefillProgressView, ReferenceBindingUnknown> {
        if let Some(reason) = self.invalid {
            return Err(reason);
        }
        if checkpoints.len() > 1024 {
            return Err(ReferenceUnknown::PointBudget.into());
        }
        let local = PlanningTimeOrigin::from_origin(self.ingress, snapshot.observed_at())?;
        let local_checkpoints = checkpoints
            .iter()
            .map(|at| snapshot.instant_at_ns(*at).and_then(|at| local.at_ns(at)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut progress = self.binding.progress_with_granule(
            offset,
            executable_until,
            &local_checkpoints,
            legal_granule,
        )?;
        progress.admitted_at_ns = snapshot.at_ns(self.admitted_at)?;
        let mut milestones = progress.milestones.to_vec();
        for milestone in &mut milestones {
            let at = self
                .ingress
                .checked_add(Duration::from_nanos(milestone.at_ns))
                .ok_or(PlanningTimeError::TimeOverflow)?;
            milestone.at_ns = snapshot.at_ns(at)?;
        }
        progress.milestones = milestones.into();
        Ok(progress)
    }
}

/// A local receipt captured after validating the executor's actual chunk, and
/// before changing sequence state. Only successful scheduler publication may
/// consume it. Retaining the owner Arc fences RequestId reuse without retaining
/// any model lease or cache.
#[derive(Debug)]
pub(in crate::continuous_engine) struct ReferencePrefillCommit {
    request_id: RequestId,
    owner: Arc<()>,
    incarnation: NonZeroU64,
    generation: NonZeroU64,
    chunk: PrefillChunk,
    generated_before: usize,
}

impl SequenceState {
    pub(in crate::continuous_engine) fn prepare_prefill_reference_commit(
        &mut self,
        chunk: PrefillChunk,
    ) -> Option<ReferencePrefillCommit> {
        let SequencePrefillReference::Known(binding) = self.prefill_reference.as_mut()? else {
            return None;
        };
        if binding.invalid.is_some() {
            return None;
        }
        if self.request_id != binding.request_id
            || !Arc::ptr_eq(&self.stream_projection_identity, &binding.owner)
            || chunk.tokens_processed() != self.prefill_tokens_processed
            || chunk.total_prompt_tokens() != binding.total.get() as usize
            || self
                .input_tokens
                .len()
                .checked_add(self.generated_tokens.len())
                != Some(chunk.total_prompt_tokens())
        {
            binding.invalid = Some(ReferenceBindingUnknown::InvalidReceipt);
            return None;
        }
        Some(ReferencePrefillCommit {
            request_id: self.request_id.clone(),
            owner: self.stream_projection_identity.clone(),
            incarnation: binding.incarnation,
            generation: binding.generation,
            chunk,
            generated_before: self.generated_tokens.len(),
        })
    }

    pub(in crate::continuous_engine) fn publish_prefill_reference_commit(
        &mut self,
        receipt: Option<ReferencePrefillCommit>,
    ) {
        if let Some(receipt) = receipt {
            if let Err(reason) = self.apply_prefill_reference_commit(receipt) {
                tracing::warn!(request_id = %self.request_id, ?reason,
                    "prefill reference receipt unavailable; completion continues");
            }
        }
    }

    fn apply_prefill_reference_commit(
        &mut self,
        receipt: ReferencePrefillCommit,
    ) -> std::result::Result<u64, ReferenceBindingUnknown> {
        // A stale receipt must not poison or advance a replacement owner.
        if self.request_id != receipt.request_id
            || !Arc::ptr_eq(&self.stream_projection_identity, &receipt.owner)
        {
            return Err(ReferenceBindingUnknown::StaleReceipt);
        }
        let Some(SequencePrefillReference::Known(binding)) = &mut self.prefill_reference else {
            return Err(ReferenceBindingUnknown::StaleReceipt);
        };
        if binding.incarnation != receipt.incarnation || binding.generation != receipt.generation {
            return Err(ReferenceBindingUnknown::StaleReceipt);
        }
        let result = (|| {
            if let Some(reason) = binding.invalid {
                return Err(reason);
            }
            if self.prefill_tokens_processed != receipt.chunk.end()
                || self.prefill_complete != receipt.chunk.is_final()
                || receipt
                    .generated_before
                    .checked_add(usize::from(receipt.chunk.is_final()))
                    != Some(self.generated_tokens.len())
            {
                return Err(ReferenceBindingUnknown::InvalidReceipt);
            }
            let next = binding
                .generation
                .get()
                .checked_add(1)
                .and_then(NonZeroU64::new)
                .ok_or(ReferenceBindingUnknown::IncarnationExhausted)?;
            let credit = binding.binding.record_committed(
                &self.request_id,
                binding.incarnation,
                receipt
                    .chunk
                    .tokens_processed()
                    .try_into()
                    .map_err(|_| ReferenceBindingUnknown::LengthOverflow)?,
                receipt
                    .chunk
                    .end()
                    .try_into()
                    .map_err(|_| ReferenceBindingUnknown::LengthOverflow)?,
                receipt.chunk.is_final(),
            )?;
            binding.generation = next;
            Ok(credit)
        })();
        if let Err(reason) = result {
            binding.invalid = Some(reason);
        }
        result
    }
}
