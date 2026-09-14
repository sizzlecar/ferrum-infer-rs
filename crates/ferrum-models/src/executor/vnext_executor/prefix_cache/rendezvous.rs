use super::*;
use ferrum_interfaces::model_executor::{
    PrefixCaptureBoundary, PrefixCaptureLease, PrefixCapturePlan, PrefixCaptureRequest,
    PrefixCaptureStatus,
};

pub(in super::super) struct NativePrefixCapture<R: DeviceRuntime> {
    source: std::sync::Weak<VNextSequence<R>>,
    owner: Arc<()>,
    boundary: usize,
    expires_at: Instant,
    unavailable: AtomicBool,
    checkpoint: Mutex<Option<SequenceCheckpoint<R>>>,
}

impl<R: DeviceRuntime> std::fmt::Debug for NativePrefixCapture<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePrefixCapture")
            .field("boundary", &self.boundary)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> PrefixCaptureLease for NativePrefixCapture<R> {
    fn boundary(&self) -> usize {
        self.boundary
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn status(&self) -> PrefixCaptureStatus {
        if Instant::now() >= self.expires_at || self.unavailable.load(Ordering::Acquire) {
            return PrefixCaptureStatus::Unavailable;
        }
        if self.checkpoint.lock().is_some() {
            return PrefixCaptureStatus::Ready;
        }
        match self.source.upgrade() {
            Some(source)
                if source.active.load(Ordering::Acquire)
                    && source.prefill_tokens_processed.load(Ordering::Acquire) <= self.boundary =>
            {
                PrefixCaptureStatus::Pending
            }
            _ => PrefixCaptureStatus::Unavailable,
        }
    }
}

pub(super) fn select_boundary(
    layout: &SequenceCheckpointLayout,
    input: PrefixCaptureBoundary<'_>,
) -> Option<PrefixCapturePlan> {
    if layout.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
        || input.follower_prompt_tokens.is_empty()
    {
        return None;
    }
    let followers = input
        .follower_prompt_tokens
        .iter()
        .map(|&length| length as u64)
        .collect::<Vec<_>>();
    let boundary = layout.shared_prefix_boundary(
        input.processed_tokens as u64,
        input.source_prompt_tokens as u64,
        input.common_prefix_tokens as u64,
        &followers,
    )?;
    Some(PrefixCapturePlan {
        boundary: usize::try_from(boundary).ok()?,
        span: layout.capture_span_constraint(),
    })
}

pub(super) fn select_prompt_tail_boundary(
    layout: &SequenceCheckpointLayout,
    chunk: PrefillChunk,
) -> Option<PrefixCapturePlan> {
    let boundary = layout.prompt_tail_boundary(
        chunk.tokens_processed() as u64,
        chunk.total_prompt_tokens() as u64,
    )?;
    if boundary > chunk.end() as u64 {
        return None;
    }
    Some(PrefixCapturePlan {
        boundary: usize::try_from(boundary).ok()?,
        span: layout.capture_span_constraint(),
    })
}

/// An already completed immutable source retained across target capacity
/// reprobes. It does not keep the producer or an index entry alive.
struct RetainedRestoreCheckpoint<R: DeviceRuntime> {
    owner: Arc<()>,
    checkpoint: SequenceCheckpoint<R>,
}

impl<R: DeviceRuntime> std::fmt::Debug for RetainedRestoreCheckpoint<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedRestoreCheckpoint")
            .field("boundary", &self.checkpoint.completed_tokens())
            .finish()
    }
}

impl<R: DeviceRuntime> PrefixCaptureLease for RetainedRestoreCheckpoint<R> {
    fn boundary(&self) -> usize {
        self.checkpoint.completed_tokens()
    }
    fn status(&self) -> PrefixCaptureStatus {
        PrefixCaptureStatus::Ready
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(in super::super) fn prompt_tail_boundary(
        &self,
        chunk: PrefillChunk,
    ) -> Option<PrefixCapturePlan> {
        select_prompt_tail_boundary(usable_layout(self.resolved_plan.execution_plan())?, chunk)
    }

    pub(super) fn retain_restore_checkpoint(
        &self,
        checkpoint: SequenceCheckpoint<R>,
    ) -> Arc<dyn PrefixCaptureLease> {
        Arc::new(RetainedRestoreCheckpoint {
            owner: Arc::clone(&self.prefix_capture_identity),
            checkpoint,
        })
    }

    pub(in super::super) fn rendezvous_boundary(
        &self,
        input: PrefixCaptureBoundary<'_>,
    ) -> Option<PrefixCapturePlan> {
        select_boundary(usable_layout(self.resolved_plan.execution_plan())?, input)
    }

    pub(in super::super) fn arm_prefix_capture(
        &self,
        input: PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn PrefixCaptureLease>>> {
        let Some(layout) = usable_layout(self.resolved_plan.execution_plan()) else {
            return Ok(None);
        };
        if layout.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
            || Instant::now() >= input.expires_at
        {
            return Ok(None);
        }
        let sequence = {
            let registry = self.sequences.lock();
            let Some(slot) = registry.prefills.get(input.source_request_id) else {
                return Ok(None);
            };
            if slot.cancelled.load(Ordering::Acquire) {
                return Ok(None);
            }
            let state = slot.state.lock();
            match &*state {
                VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
                _ => return Ok(None),
            }
        };
        let processed = sequence.prefill_tokens_processed.load(Ordering::Acquire);
        if sequence.request_origin != ExecutorRequestOrigin::Product
            || !sequence.active.load(Ordering::Acquire)
            || sequence.maximum_tokens != input.maximum_sequence_tokens
            || sequence
                .tokens
                .lock()
                .iter()
                .copied()
                .ne(input.source_tokens.iter().map(|token| token.get()))
            || !layout.permits_capture_from(
                processed as u64,
                input.boundary as u64,
                input.source_tokens.len() as u64,
            )
        {
            return Ok(None);
        }
        let lease = Arc::new(NativePrefixCapture {
            source: Arc::downgrade(&sequence),
            owner: Arc::clone(&self.prefix_capture_identity),
            boundary: input.boundary,
            expires_at: input.expires_at,
            unavailable: AtomicBool::new(false),
            checkpoint: Mutex::new(None),
        });
        let mut interests = sequence.prefix_capture_interests.lock();
        interests.retain(|interest| interest.strong_count() > 0);
        interests.push(Arc::downgrade(&lease));
        Ok(Some(lease))
    }

    pub(super) fn retained_rendezvous_checkpoint(
        &self,
        lease: &dyn PrefixCaptureLease,
    ) -> Option<SequenceCheckpoint<R>> {
        if let Some(retained) = lease
            .as_any()
            .downcast_ref::<RetainedRestoreCheckpoint<R>>()
        {
            return Arc::ptr_eq(&retained.owner, &self.prefix_capture_identity)
                .then(|| retained.checkpoint.clone());
        }
        let native = lease.as_any().downcast_ref::<NativePrefixCapture<R>>()?;
        if !Arc::ptr_eq(&native.owner, &self.prefix_capture_identity)
            || native.status() != PrefixCaptureStatus::Ready
        {
            return None;
        }
        native.checkpoint.lock().clone()
    }
}

pub(super) fn interests_at<R: DeviceRuntime>(
    sequence: &VNextSequence<R>,
    boundary: usize,
) -> Vec<Arc<NativePrefixCapture<R>>> {
    let mut interests = sequence.prefix_capture_interests.lock();
    interests.retain(|interest| interest.strong_count() > 0);
    interests
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter(|interest| {
            interest.boundary == boundary && interest.status() == PrefixCaptureStatus::Pending
        })
        .collect()
}

pub(super) fn publish<R: DeviceRuntime>(
    sequence: &VNextSequence<R>,
    checkpoint: &SequenceCheckpoint<R>,
) {
    for interest in interests_at(sequence, checkpoint.completed_tokens()) {
        *interest.checkpoint.lock() = Some(checkpoint.clone());
    }
}

pub(super) fn finish<R: DeviceRuntime>(interests: &[Arc<NativePrefixCapture<R>>]) {
    for interest in interests {
        if interest.checkpoint.lock().is_none() {
            interest.unavailable.store(true, Ordering::Release);
        }
    }
}
