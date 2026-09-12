//! Optional cross-request state retention. The index owns immutable native
//! checkpoints; admission, physical allocation and in-flight pins remain in the
//! plan's existing resource ledger.

use super::*;
use std::collections::VecDeque;

/// Request counters are shared with the consuming publication callback. Index
/// occupancy is read from owners instead of estimated from token/text lengths.
#[derive(Default)]
pub(super) struct PrefixCacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    saved_prefill_tokens: AtomicU64,
}

impl PrefixCacheMetrics {
    fn acknowledge_restore(
        &self,
        restored_tokens: usize,
        acknowledge: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        // This callback includes exact-target/cancellation checks and consumes
        // the native publication. Lookup, copy completion and a dropped output
        // do not establish a successfully published restore.
        acknowledge()?;
        self.saved_prefill_tokens
            .fetch_add(restored_tokens as u64, Ordering::Relaxed);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn reset(&self) {
        for counter in [
            &self.hits,
            &self.misses,
            &self.evictions,
            &self.saved_prefill_tokens,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

struct Entry<C> {
    prefix: Arc<[u32]>,
    input: Arc<[u32]>,
    checkpoint: C,
}

/// Oldest at the front. This is an index, not a second byte or slot budget.
pub(super) struct PrefixIndex<C> {
    entries: VecDeque<Entry<C>>,
}

impl<C> Default for PrefixIndex<C> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }
}

impl<C> PrefixIndex<C> {
    fn snapshot(
        &self,
        plan: &ExecutionPlan,
        requested: bool,
        metrics: &PrefixCacheMetrics,
        retained_bytes: impl Fn(&C) -> u64,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut snapshot = serde_json::json!({
            "source": "vnext-native-sequence-checkpoint-cache",
            "position": "model-executor",
            "requested": requested,
            "enabled": usable_layout(plan).is_some(),
            "entries": self.entries.len(),
            "bytes": self.entries.iter().map(|entry| retained_bytes(&entry.checkpoint)).sum::<u64>(),
            "bytes_scope": "index-owned-allocator-aligned-checkpoint-extents",
            "excludes_evicted_inflight_pins": true,
            "hits": metrics.hits.load(Ordering::Relaxed),
            "hits_scope": "successfully-acknowledged-native-restores",
            "misses": metrics.misses.load(Ordering::Relaxed),
            "misses_scope": "enabled-index-lookups-without-reusable-entry",
            "evictions": metrics.evictions.load(Ordering::Relaxed),
            "evictions_scope": "pressure-removed-index-entries",
            "saved_prefill_tokens": metrics.saved_prefill_tokens.load(Ordering::Relaxed),
        });
        let fields = snapshot
            .as_object_mut()
            .expect("native prefix snapshot is an object");
        match plan.sequence_checkpoint_capability() {
            SequenceCheckpointCapability::Unsupported(reasons) => {
                fields.insert("unsupported_reasons".into(), serde_json::json!(reasons));
            }
            SequenceCheckpointCapability::Enabled(layout) => {
                // These are executor evidence gates, separate from plan-derived
                // unsupported reasons. Do not serialize the enabled layout.
                let mut missing_evidence = Vec::new();
                if !layout.inputs().conditioning_inputs().is_empty() {
                    missing_evidence.push("conditioning_inputs");
                }
                if layout.providers().iter().any(|provider| {
                    provider.contract().partition_numerics()
                        == CheckpointPartitionNumerics::SamePartitionOnly
                }) {
                    missing_evidence.push("same_partition_execution_trace");
                }
                if !missing_evidence.is_empty() {
                    fields.insert(
                        "missing_executor_evidence".into(),
                        serde_json::json!(missing_evidence),
                    );
                }
            }
        }
        std::mem::take(fields)
    }

    fn remove_replaced(&mut self, input: &[u32], completed_tokens: usize) -> Vec<C> {
        let mut removed = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            let entry = &self.entries[index];
            // A checkpoint at the input end needs a longer future request.
            // Preserve the partial checkpoint that can serve an exact repeat
            // of this input. Replacements within either use retain one owner.
            if entry.input.as_ref() == input
                && (entry.prefix.len() == input.len()) == (completed_tokens == input.len())
            {
                removed.push(self.entries.remove(index).expect("known entry").checkpoint);
            } else {
                index += 1;
            }
        }
        removed
    }

    fn insert(&mut self, prefix: Arc<[u32]>, input: Arc<[u32]>, checkpoint: C) -> Vec<C> {
        let replaced = self.remove_replaced(&input, prefix.len());
        self.entries.push_back(Entry {
            prefix,
            input,
            checkpoint,
        });
        replaced
    }

    fn evict(&mut self) -> Option<C> {
        self.entries.pop_front().map(|entry| entry.checkpoint)
    }
}

impl<C: Clone> PrefixIndex<C> {
    fn longest(
        &mut self,
        input: &[u32],
        entire_input: bool,
        permits_suffix: impl Fn(usize) -> bool,
    ) -> Option<C> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                !entry.prefix.is_empty()
                    && entry.prefix.len() < input.len()
                    && input.starts_with(&entry.prefix)
                    && (!entire_input || entry.input.as_ref() == input)
                    && permits_suffix(entry.prefix.len())
            })
            .max_by_key(|(_, entry)| entry.prefix.len())
            .map(|(index, _)| index)?;
        let entry = self.entries.remove(index).expect("selected entry");
        let checkpoint = entry.checkpoint.clone();
        self.entries.push_back(entry);
        Some(checkpoint)
    }
}

fn usable_layout(plan: &ExecutionPlan) -> Option<&SequenceCheckpointLayout> {
    plan.payload().memory().checkpoint_capacity()?;
    let SequenceCheckpointCapability::Enabled(layout) = plan.sequence_checkpoint_capability()
    else {
        return None;
    };
    // These evidence classes are not yet supplied by the product executor.
    // The native façade independently enforces the same restrictions.
    if !layout.inputs().conditioning_inputs().is_empty()
        || layout.providers().iter().any(|provider| {
            provider.contract().partition_numerics()
                == CheckpointPartitionNumerics::SamePartitionOnly
        })
    {
        return None;
    }
    Some(layout)
}

fn capture_candidate(chunk: PrefillChunk) -> bool {
    !chunk.is_final()
        && chunk.end() > 0
        && chunk.total_prompt_tokens() - chunk.end() <= chunk.tokens_to_process()
}

enum CaptureAttempt<M, C> {
    Finished(Option<C>),
    NeedsMaintenance(M),
    RetentionLimited(CheckpointRetentionSkipReason),
}

enum CaptureMaintenanceDecision {
    RetryCapture,
    CapacityLimited,
    Skip,
}

fn retention_release_can_help(reason: &CheckpointRetentionSkipReason) -> bool {
    matches!(
        reason,
        CheckpointRetentionSkipReason::Capacity {
            requested_bytes,
            retained_bytes,
            maximum_bytes,
        } if requested_bytes <= maximum_bytes && *retained_bytes > 0
    )
}

fn capture_maintenance_decision(
    result: std::result::Result<CheckpointCapacityMaintenanceOutcome, VNextError>,
) -> CaptureMaintenanceDecision {
    match result {
        Ok(CheckpointCapacityMaintenanceOutcome::Ready(_)) => {
            CaptureMaintenanceDecision::RetryCapture
        }
        Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
            CheckpointCapacityMaintenanceSkipReason::DeviceCapacity(_)
            | CheckpointCapacityMaintenanceSkipReason::PoolResident(_),
        )) => CaptureMaintenanceDecision::CapacityLimited,
        Ok(CheckpointCapacityMaintenanceOutcome::Skipped(
            CheckpointCapacityMaintenanceSkipReason::Retention(reason),
        )) if retention_release_can_help(&reason) => CaptureMaintenanceDecision::CapacityLimited,
        // Disabled retention, an oversized checkpoint, and arbitrary backend or
        // contract errors cannot justify discarding a valid cached checkpoint.
        Ok(CheckpointCapacityMaintenanceOutcome::Skipped(_)) | Err(_) => {
            CaptureMaintenanceDecision::Skip
        }
    }
}

/// The callbacks keep native allocation and completion on the existing worker.
/// Each successful index removal permits one new maintenance attempt; concurrent
/// insertions and still-pinned owners cannot increase this call's finite budget.
async fn capture_with_capacity<M, C, A, F>(
    entries: usize,
    mut capture: impl FnMut() -> A,
    mut maintain: impl FnMut(M) -> F,
    mut evict: impl FnMut() -> bool,
) -> Result<Option<C>>
where
    A: std::future::Future<Output = Result<CaptureAttempt<M, C>>>,
    F: std::future::Future<Output = Result<CaptureMaintenanceDecision>>,
{
    let mut remaining_evictions = entries;
    let mut maintenance_attempted = false;
    loop {
        match capture().await? {
            CaptureAttempt::Finished(result) => return Ok(result),
            CaptureAttempt::NeedsMaintenance(owner) => {
                if maintenance_attempted {
                    // Even successful growth grants no claim. If a fresh claim
                    // still fails, skip optional capture instead of evicting on
                    // an old shortage or chasing concurrent allocations.
                    return Ok(None);
                }
                maintenance_attempted = true;
                match maintain(owner).await? {
                    CaptureMaintenanceDecision::RetryCapture => continue,
                    CaptureMaintenanceDecision::CapacityLimited => {}
                    CaptureMaintenanceDecision::Skip => return Ok(None),
                }
            }
            CaptureAttempt::RetentionLimited(reason) => {
                if !retention_release_can_help(&reason) {
                    return Ok(None);
                }
            }
        }
        if remaining_evictions == 0 || !evict() {
            return Ok(None);
        }
        remaining_evictions -= 1;
        maintenance_attempted = false;
        // No previous capture guard or maintenance owner is retained. Re-probe
        // the complete native capture after release, including source validity.
    }
}

fn checkpoint_release_can_help(
    wait: &CapacityWaitCondition,
    checkpoint_domains: &BTreeSet<CapacityDomainId>,
) -> bool {
    if wait
        .observed()
        .iter()
        .any(|epoch| epoch.source() == CapacityAvailabilitySource::ActiveSequenceSlots)
    {
        // Checkpoints consume no execution slot. Wait for that necessary
        // prerequisite instead of throwing out entries on each queue probe.
        return false;
    }
    wait.observed().iter().any(|epoch| match epoch.source() {
        CapacityAvailabilitySource::Domain(domain) => checkpoint_domains.contains(&domain),
        CapacityAvailabilitySource::PlanDeviceBudget
        | CapacityAvailabilitySource::ProcessDeviceCapacity => !checkpoint_domains.is_empty(),
        CapacityAvailabilitySource::ActiveSequenceSlots => false,
    })
}

/// One optional growth attempt for a foreground admission or extension. A
/// successful maintenance result must be re-probed before any entry is evicted.
pub(super) struct PrefixPressureMaintenance {
    attempted: bool,
    capacity_limited: bool,
    remaining_evictions: usize,
    additional_attempts: usize,
}

pub(super) enum PrefixPressureRecovery<T> {
    Maintained(T),
    Evicted,
    Unchanged,
}

impl PrefixPressureMaintenance {
    fn new(entries: usize) -> Self {
        Self {
            attempted: false,
            capacity_limited: false,
            remaining_evictions: entries,
            additional_attempts: 0,
        }
    }

    pub(super) fn allows_backing_attempt(&self, attempts: u32) -> bool {
        (attempts as usize).saturating_sub(self.additional_attempts)
            < MAX_BACKING_MAINTENANCE_ATTEMPTS as usize
    }

    pub(super) fn evict_after_pressure(&mut self, evict: impl FnOnce() -> bool) -> bool {
        if self.remaining_evictions == 0 || !evict() {
            return false;
        }
        self.remaining_evictions -= 1;
        self.additional_attempts += 1;
        true
    }

    pub(super) fn eviction_after_wait<R: DeviceRuntime>(
        &mut self,
        executor: &VNextModelExecutor<R>,
        wait: &CapacityWaitCondition,
    ) -> bool {
        self.evict_after_pressure(|| executor.evict_prefix_for_wait(wait))
    }

    fn recover<T>(
        &mut self,
        relevant: bool,
        maintain: impl FnOnce() -> std::result::Result<Option<T>, VNextError>,
        evict: impl FnOnce() -> bool,
    ) -> std::result::Result<PrefixPressureRecovery<T>, VNextError> {
        if !relevant {
            return Ok(PrefixPressureRecovery::Unchanged);
        }
        if !self.attempted {
            self.attempted = true;
            match maintain() {
                Ok(Some(receipt)) => return Ok(PrefixPressureRecovery::Maintained(receipt)),
                Ok(None) => return Ok(PrefixPressureRecovery::Unchanged),
                Err(VNextError::DeviceCapacityUnavailable(_))
                | Err(VNextError::DynamicPoolResidentUnavailable(_)) => {
                    self.capacity_limited = true;
                }
                Err(error) => return Err(error),
            }
        }
        if self.capacity_limited && self.evict_after_pressure(evict) {
            Ok(PrefixPressureRecovery::Evicted)
        } else {
            Ok(PrefixPressureRecovery::Unchanged)
        }
    }
}

pub(super) fn retain_token_evidence(
    plan: &ExecutionPlan,
    span: TokenSpanWork,
    tokens: &[u32],
) -> Result<TokenSpanWork> {
    if usable_layout(plan).is_none() {
        return Ok(span);
    }
    // Keep the ordinary range, fit ceiling, wire and fingerprint. No complete
    // input copy or second hash occurs for disabled/unsupported execution.
    span.with_checkpoint_tokens(Arc::from(tokens))
        .map_err(|error| FerrumError::backend(error.to_string()))
}

/// Runs only on the existing completion worker. An unknown write is never a
/// cache miss: failed drain leaves the owner in the native reaper quarantine.
fn finish_transfer<R: DeviceRuntime>(
    reaper: &CompletionReaper<R>,
    start: NativeCheckpointStart<R>,
) -> Result<Option<NativeCheckpointResult<R>>> {
    let (mut transfer, contract_error) = match start {
        NativeCheckpointStart::Skipped(_) => return Ok(None),
        NativeCheckpointStart::CapacityMaintenance { .. } => {
            return Err(FerrumError::internal(
                "checkpoint maintenance must be handled before submission",
            ));
        }
        NativeCheckpointStart::NotSubmitted(error) => {
            return Err(FerrumError::backend(format!(
                "checkpoint was not submitted: {error}"
            )));
        }
        NativeCheckpointStart::Submitted(transfer)
        | NativeCheckpointStart::Indeterminate(transfer) => (transfer, None),
        NativeCheckpointStart::ContractAfterSubmission { error, transfer } => {
            (transfer, Some(error))
        }
    };
    let result = (|| {
        let mut observation = transfer.poll();
        if !matches!(observation, Ok(NativeCheckpointObservation::Ready)) {
            observation = transfer.wait_for_recovery();
        }
        if !matches!(observation, Ok(NativeCheckpointObservation::Ready)) {
            observation = transfer.recover_by_draining_lane();
        }
        match observation {
            Ok(NativeCheckpointObservation::Ready) => {}
            other => {
                return Err(FerrumError::backend(format!(
                    "checkpoint completion remains owned by recovery: {other:?}"
                )))
            }
        }
        let result = transfer
            .take_result()
            .map_err(|error| FerrumError::backend(error.to_string()))?
            .ok_or_else(|| FerrumError::internal("ready checkpoint has no result"))?;
        if let Some(error) = contract_error {
            drop(result);
            return Err(FerrumError::backend(format!(
                "checkpoint submission contract: {error}"
            )));
        }
        Ok(Some(result))
    })();
    drop(transfer);
    // Drop marks a still-owned outbox abandoned. Sweep that same registry; no
    // receiver, request future, or additional worker is required for cleanup.
    let _ = reaper.recover_abandoned_checkpoints(MAX_COMPLETION_SWEEP_SLOTS);
    result
}

/// Owned by the output callback, including while the engine commits scheduler
/// progress. No request-id lookup can substitute another admitted incarnation.
struct RestoreAcknowledgement<R: DeviceRuntime> {
    publication: Option<CheckpointRestorePublication<R>>,
    slot: Arc<VNextPrefillSlot<R>>,
    sequence: Arc<VNextSequence<R>>,
    acknowledged: bool,
}

impl<R: DeviceRuntime> RestoreAcknowledgement<R> {
    fn acknowledge(mut self) -> Result<()> {
        let state = self.slot.state.lock();
        if self.slot.cancelled.load(Ordering::Acquire)
            || !self.sequence.active.load(Ordering::Acquire)
            || self
                .sequence
                .prefill_tokens_processed
                .load(Ordering::Acquire)
                != 0
            || !matches!(&*state, VNextPrefillSlotState::Ready(sequence)
                if Arc::ptr_eq(sequence, &self.sequence))
        {
            return Err(FerrumError::cancelled(
                "prefix restore lost its exact Ready target",
            ));
        }
        let publication = self
            .publication
            .take()
            .ok_or_else(|| FerrumError::internal("prefix restore was already acknowledged"))?;
        if !publication.matches_target(&self.sequence.session)
            || !self
                .sequence
                .tokens
                .lock()
                .starts_with(publication.token_prefix())
        {
            return Err(FerrumError::backend(
                "prefix restore publication names another target",
            ));
        }
        self.sequence
            .prefill_tokens_processed
            .store(publication.completed_tokens(), Ordering::Release);
        publication
            .acknowledge()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        self.acknowledged = true;
        Ok(())
    }
}

impl<R: DeviceRuntime> Drop for RestoreAcknowledgement<R> {
    fn drop(&mut self) {
        if !self.acknowledged {
            self.slot.cancelled.store(true, Ordering::Release);
            // The native publication cancels before reopening its gate. Abort
            // the exact model sequence even if the outer request was replaced.
            drop(self.publication.take());
            self.sequence.abort();
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn prefix_pressure_maintenance(&self) -> PrefixPressureMaintenance {
        // New captures from other callers cannot grow this call's retry budget.
        PrefixPressureMaintenance::new(self.prefix_cache.lock().entries.len())
    }

    pub(super) fn prefix_cache_metrics_snapshot(
        &self,
    ) -> serde_json::Map<String, serde_json::Value> {
        self.prefix_cache.lock().snapshot(
            self.resolved_plan.execution_plan(),
            self.policy.memory().checkpoint_capacity.is_some(),
            &self.metrics.prefix_cache,
            SequenceCheckpoint::retained_bytes,
        )
    }

    pub(super) fn prefix_restore_enabled(&self) -> bool {
        usable_layout(self.resolved_plan.execution_plan()).is_some()
    }

    pub(super) fn evict_prefix_checkpoint(&self) -> bool {
        let started = Instant::now();
        let removed = self.prefix_cache.lock().evict();
        let evicted = removed.is_some();
        if evicted {
            self.metrics
                .prefix_cache
                .evictions
                .fetch_add(1, Ordering::Relaxed);
        }
        // Physical/logical release takes place outside the index lock. A clone
        // pinned by a transfer continues to count against the native ledger.
        drop(removed);
        self.reaper.record_checkpoint_cache_timing(
            CheckpointCacheTimingPhase::EvictionDrop,
            started.elapsed(),
        );
        evicted
    }

    pub(super) fn prefix_checkpoint_may_block(&self, wait: &CapacityWaitCondition) -> bool {
        if self.prefix_cache.lock().entries.is_empty() {
            return false;
        }
        let Some(layout) = usable_layout(self.resolved_plan.execution_plan()) else {
            return false;
        };
        let Ok(status) = self.plan_resources.dynamic_pool_status() else {
            return false;
        };
        let resources = layout
            .states()
            .iter()
            .map(|state| state.resource_id())
            .collect::<BTreeSet<_>>();
        let checkpoint_domains = status
            .pools()
            .iter()
            .filter(|pool| {
                pool.contract()
                    .resources()
                    .iter()
                    .any(|resource| resources.contains(resource.resource_id()))
            })
            .map(|pool| pool.domain_id())
            .collect::<BTreeSet<_>>();
        checkpoint_release_can_help(wait, &checkpoint_domains)
    }

    pub(super) fn evict_prefix_for_wait(&self, wait: &CapacityWaitCondition) -> bool {
        self.prefix_checkpoint_may_block(wait) && self.evict_prefix_checkpoint()
    }

    pub(super) fn recover_prefix_pressure(
        &self,
        attempt: &mut PrefixPressureMaintenance,
        deferred: &AdmissionDeferred,
    ) -> Result<PrefixPressureRecovery<DynamicPoolGrowthBatchReceipt>> {
        attempt
            .recover(
                self.prefix_checkpoint_may_block(deferred.wait_condition()),
                || {
                    self.plan_resources
                        .try_maintain_for_capacity_pressure(deferred)
                },
                || self.evict_prefix_for_wait(deferred.wait_condition()),
            )
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    pub(super) async fn retain_prefill_boundary(
        &self,
        sequence: &Arc<VNextSequence<R>>,
        tokens: &[u32],
        chunk: PrefillChunk,
    ) -> Result<()> {
        let Some(layout) = usable_layout(self.resolved_plan.execution_plan()) else {
            return Ok(());
        };
        if sequence.request_origin != ExecutorRequestOrigin::Product
            || !capture_candidate(chunk)
            || !layout.permits_capture_from(
                chunk.tokens_processed() as u64,
                chunk.end() as u64,
                tokens.len() as u64,
            )
        {
            return Ok(());
        }
        self.retain_sequence_boundary(sequence, tokens, chunk.end())
            .await
    }

    pub(super) async fn retain_completed_sequence_boundary(
        &self,
        sequence: &Arc<VNextSequence<R>>,
    ) -> Result<()> {
        let Some(layout) = usable_layout(self.resolved_plan.execution_plan()) else {
            return Ok(());
        };
        if sequence.request_origin != ExecutorRequestOrigin::Product
            || layout.completed_input_capture() != CheckpointCompletedInputCapture::Supported
            // A completed whole-input-dependent state cannot both match that
            // same input and leave the nonempty suffix required for restore.
            || layout.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
        {
            return Ok(());
        }
        // The caller holds the operation lock while the original session is
        // still Open. These are only tokens actually consumed by FullPlan
        // execution; the final sampled token is deliberately absent.
        let tokens = sequence.tokens.lock().clone();
        if tokens.is_empty() {
            return Ok(());
        }
        // The native facade checks the last retired span and every selected
        // provider's permission to capture at the end of the known input.
        self.retain_sequence_boundary(sequence, &tokens, tokens.len())
            .await
    }

    async fn retain_sequence_boundary(
        &self,
        sequence: &Arc<VNextSequence<R>>,
        tokens: &[u32],
        completed_tokens: usize,
    ) -> Result<()> {
        // Replacing a candidate must not require holding both copies at once.
        let replacement_started = Instant::now();
        let replaced = self
            .prefix_cache
            .lock()
            .remove_replaced(tokens, completed_tokens);
        drop(replaced);
        self.reaper.record_checkpoint_cache_timing(
            CheckpointCacheTimingPhase::ReplacementDrop,
            replacement_started.elapsed(),
        );
        let entries = self.prefix_cache.lock().entries.len();
        let result = capture_with_capacity(entries, || async {
            let reaper = Arc::clone(&self.reaper);
            let plan = self.resolved_plan.execution_plan().clone();
            let resources = Arc::clone(&self.plan_resources);
            let source = Arc::clone(&sequence.session);
            let lane = Arc::clone(&self.lane);
            let timing_mode = self.device_timing_mode();
            self
                .completion_worker
                .execute(VNextCompletionTaskKind::CheckpointTransfer, move || -> Result<_> {
                    let recovery_started = Instant::now();
                    let _ = reaper.recover_abandoned_checkpoints(MAX_COMPLETION_SWEEP_SLOTS);
                    reaper.record_checkpoint_cache_timing(
                        CheckpointCacheTimingPhase::AbandonedRecovery,
                        recovery_started.elapsed(),
                    );
                    let binding = resources
                        .trusted_runtime_binding()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    let start = match reaper.try_capture_sequence_checkpoint_with_timing(&plan, &binding, Arc::clone(&source), lane, timing_mode) {
                        Ok(NativeCheckpointStart::NotSubmitted(error)) | Err(error) => {
                            // A cancelled/poisoned source is not a cache miss.
                            // This existing projection rechecks the exact Open
                            // session without allocating or submitting work.
                            source.write_release_capacity_sources(&mut Vec::new())
                                .map_err(|source_error| FerrumError::backend(format!("capture source became unavailable after {error}: {source_error}")))?;
                            tracing::warn!(reason = %error, "prefix capture skipped before submission");
                            return Ok(CaptureAttempt::Finished(None));
                        }
                        Ok(start) => start,
                    };
                    let start = match start {
                        NativeCheckpointStart::CapacityMaintenance { reason, maintenance } => {
                            tracing::debug!(?reason, "prefix capture needs optional pool maintenance");
                            return Ok(CaptureAttempt::NeedsMaintenance(maintenance));
                        }
                        NativeCheckpointStart::Skipped(CheckpointAccessSkipReason::Retention(reason)) => {
                            tracing::debug!(?reason, "prefix capture skipped before submission");
                            return Ok(CaptureAttempt::RetentionLimited(reason));
                        }
                        NativeCheckpointStart::Skipped(reason) => {
                            tracing::debug!(?reason, "prefix capture skipped before submission");
                            NativeCheckpointStart::Skipped(reason)
                        }
                        start => start,
                    };
                    let result = finish_transfer(&reaper, start)?;
                    if let Some(NativeCheckpointResult::Failed(reason @ (NativeCheckpointFailure::FailedButQuiescent(_) | NativeCheckpointFailure::AbandonedAfterDrain))) = &result {
                        source.write_release_capacity_sources(&mut Vec::new())
                            .map_err(|error| FerrumError::backend(format!("capture failed and its source is no longer Open: {reason:?}: {error}")))?;
                        tracing::warn!(?reason, "prefix capture skipped after quiescent failure; source remains Open");
                        return Ok(CaptureAttempt::Finished(None));
                    }
                    Ok(CaptureAttempt::Finished(result))
                })
                .await
                .map_err(|error| FerrumError::backend(error.to_string()))?
        }, |maintenance: CheckpointCapacityMaintenance<R>| async move {
                    let source = Arc::clone(&sequence.session);
                    let reaper = Arc::clone(&self.reaper);
                    self
                        .completion_worker
                        .execute(VNextCompletionTaskKind::CheckpointTransfer, move || -> Result<_> {
                            // No capture guard survives in this owner. Budget
                            // and packing are rechecked by the resource layer,
                            // without waiting for or reclaiming foreground work.
                            source.write_release_capacity_sources(&mut Vec::new())
                                .map_err(|error| FerrumError::backend(format!("capture source became unavailable before maintenance: {error}")))?;
                            let maintenance_started = Instant::now();
                            let result = maintenance.try_maintain();
                            reaper.record_checkpoint_cache_timing(
                                CheckpointCacheTimingPhase::Maintenance,
                                maintenance_started.elapsed(),
                            );
                            match &result {
                                Ok(CheckpointCapacityMaintenanceOutcome::Ready(_)) => {}
                                Ok(CheckpointCapacityMaintenanceOutcome::Skipped(reason)) => {
                                    source.write_release_capacity_sources(&mut Vec::new())
                                        .map_err(|error| FerrumError::backend(format!("capture maintenance skipped and its source is no longer Open: {reason:?}: {error}")))?;
                                    tracing::debug!(?reason, "prefix capture skipped after optional pool maintenance");
                                }
                                Err(error) => {
                                    source.write_release_capacity_sources(&mut Vec::new())
                                        .map_err(|source_error| FerrumError::backend(format!("capture source became unavailable after maintenance error {error}: {source_error}")))?;
                                    tracing::warn!(reason = %error, "prefix capture maintenance failed before submission");
                                }
                            }
                            Ok(capture_maintenance_decision(result))
                        })
                        .await
                        .map_err(|error| FerrumError::backend(error.to_string()))?
        }, || self.evict_prefix_checkpoint()).await?;
        match result {
            Some(NativeCheckpointResult::Captured(checkpoint)) => {
                if checkpoint.completed_tokens() != completed_tokens
                    || tokens.get(..completed_tokens) != Some(checkpoint.token_prefix())
                    || checkpoint.full_input() != tokens
                {
                    return Err(FerrumError::backend(
                        "capture differs from the retired sequence boundary",
                    ));
                }
                let publication_started = Instant::now();
                let removed = self.prefix_cache.lock().insert(
                    Arc::from(checkpoint.token_prefix()),
                    Arc::from(checkpoint.full_input()),
                    checkpoint,
                );
                self.reaper.record_checkpoint_cache_timing(
                    CheckpointCacheTimingPhase::IndexPublication,
                    publication_started.elapsed(),
                );
                let replacement_started = Instant::now();
                drop(removed);
                self.reaper.record_checkpoint_cache_timing(
                    CheckpointCacheTimingPhase::ReplacementDrop,
                    replacement_started.elapsed(),
                );
                Ok(())
            }
            None => Ok(()),
            Some(NativeCheckpointResult::Failed(reason)) => Err(FerrumError::backend(format!(
                "capture completion contract failed: {reason:?}"
            ))),
            _ => Err(FerrumError::internal(
                "capture returned a non-capture result",
            )),
        }
    }

    pub(super) async fn restore_prefix(
        &self,
        input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<Option<PlanRuntimePrefixRestoreOutput>> {
        let Some(layout) = usable_layout(self.resolved_plan.execution_plan()) else {
            return Ok(None);
        };
        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let checkpoint = self.prefix_cache.lock().longest(
            &tokens,
            layout.input_dependency() == CheckpointInputDependency::EntireTokenInput,
            |boundary| layout.permits_suffix(boundary as u64, tokens.len() as u64),
        );
        let Some(checkpoint) = checkpoint else {
            self.metrics
                .prefix_cache
                .misses
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        let (slot, sequence) = self
            .sequences
            .lock()
            .begin_prefill_execution(input.request_id)?;
        let mut execution = VNextPrefillExecutionGuard::new(
            &self.sequences,
            Arc::clone(&slot),
            Arc::clone(&sequence),
        );
        let _operation = sequence.operation.lock().await;
        if sequence.maximum_tokens != input.maximum_sequence_tokens
            || *sequence.tokens.lock() != tokens
            || sequence.prefill_tokens_processed.load(Ordering::Acquire) != 0
        {
            return Err(FerrumError::request_validation(
                "prefix restore differs from fresh admitted input",
            ));
        }
        let restored_tokens = checkpoint.completed_tokens();
        let span = TokenSpanWork::from_token_ids_with_fit(
            &tokens,
            0..restored_tokens,
            input.maximum_sequence_tokens,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let work = ResourceWorkShape::single(span)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        if !matches!(
            self.extend_sequence_with_capacity(&sequence, work)?,
            VNextExecutionCapacityDecision::Ready(())
        ) {
            execution.restore_ready()?;
            return Ok(None);
        }
        if slot.cancelled.load(Ordering::Acquire) || !sequence.active.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled(
                "prefix restore target was cancelled before submission",
            ));
        }
        let reaper = Arc::clone(&self.reaper);
        let plan = self.resolved_plan.execution_plan().clone();
        let target = Arc::clone(&sequence.session);
        let lane = Arc::clone(&self.lane);
        let timing_mode = self.device_timing_mode();
        let full_input: Arc<[u32]> = Arc::from(tokens.as_slice());
        let result = self
            .completion_worker
            .execute(VNextCompletionTaskKind::CheckpointTransfer, move || {
                let _ = reaper.recover_abandoned_checkpoints(MAX_COMPLETION_SWEEP_SLOTS);
                let start = reaper
                    .try_restore_sequence_checkpoint_with_timing(
                        &plan,
                        target,
                        &checkpoint,
                        full_input,
                        lane,
                        timing_mode,
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                finish_transfer(&reaper, start)
            })
            .await
            .map_err(|error| FerrumError::backend(error.to_string()))??;
        let publication = match result {
            None => {
                execution.restore_ready()?;
                return Ok(None);
            }
            Some(NativeCheckpointResult::Restored(publication)) => publication,
            Some(NativeCheckpointResult::Failed(reason)) => {
                return Err(FerrumError::backend(format!(
                    "prefix restore failed: {reason:?}"
                )));
            }
            _ => {
                return Err(FerrumError::internal(
                    "restore returned a non-restore result",
                ));
            }
        };
        if publication.completed_tokens() != restored_tokens
            || publication.token_prefix() != &tokens[..restored_tokens]
            || !publication.matches_target(&sequence.session)
        {
            return Err(FerrumError::backend(
                "restore result differs from the selected exact prefix",
            ));
        }
        execution.restore_ready()?;
        let callback = RestoreAcknowledgement {
            publication: Some(publication),
            slot,
            sequence: Arc::clone(&sequence),
            acknowledged: false,
        };
        let metrics = Arc::clone(&self.metrics.prefix_cache);
        PlanRuntimePrefixRestoreOutput::new(
            input.request_id.clone(),
            restored_tokens,
            tokens.len(),
            self.cache_handle(&sequence, restored_tokens),
            move || metrics.acknowledge_restore(restored_tokens, || callback.acknowledge()),
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests;
