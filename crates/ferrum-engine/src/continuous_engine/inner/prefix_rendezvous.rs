//! Optional first-request prefix handoff. Scheduler holds are logical; native
//! checkpoint leases alone own retained state through the existing ledger.
use super::*;
use ferrum_interfaces::model_executor::{
    PrefixCaptureBoundary, PrefixCaptureLease, PrefixCaptureRequest, PrefixCaptureStatus,
};
use ferrum_scheduler::implementations::continuous::{PrefixRendezvousHold, PrefixRequestKey};

pub(in super::super) struct PrefixRendezvous {
    hold: PrefixRendezvousHold,
    followers: Vec<PrefixRequestKey>,
    source_tokens: Vec<TokenId>,
    maximum_sequence_tokens: usize,
    expires_at: Instant,
    capture: Option<Arc<dyn PrefixCaptureLease>>,
}

impl EngineInner {
    pub(super) fn prepare_prefix_rendezvous(&self) -> Result<()> {
        let Some(max_wait) = self.config.scheduler.prefix_rendezvous_max_wait_ms else {
            return Ok(());
        };
        self.refresh_prefix_rendezvous()?;
        if !self.model_executor.supports_plan_runtime_prefix_restore() {
            return Ok(());
        }
        let Some(expires_at) = Instant::now().checked_add(Duration::from_millis(max_wait.get()))
        else {
            return Ok(());
        };
        let candidates = self.scheduler.prefix_rendezvous_candidates();
        let sequences = self.sequences.read();
        let mut used = HashSet::new();
        for producer in &candidates {
            if used.contains(producer.key.request_id()) {
                continue;
            }
            let Some(source) = sequences.get(producer.key.request_id()) else {
                continue;
            };
            if !source.generated_tokens.is_empty()
                || source.preemption_count != 0
                || source.prefill_tokens_processed != producer.processed_tokens
            {
                continue;
            }
            let source_tokens = source.prefill_context_tokens();
            let mut common = source_tokens.len();
            let mut followers = Vec::new();
            let mut prompt_lengths = Vec::new();
            for follower in &candidates {
                if !follower.waiting
                    || follower.processed_tokens != 0
                    || follower.key.request_id() == producer.key.request_id()
                    || used.contains(follower.key.request_id())
                    || follower.priority > producer.priority
                {
                    continue;
                }
                let Some(target) = sequences.get(follower.key.request_id()) else {
                    continue;
                };
                if target.preemption_count != 0 || !target.generated_tokens.is_empty() {
                    continue;
                }
                let tokens = target.prefill_context_tokens();
                let shared = source_tokens
                    .iter()
                    .zip(&tokens)
                    .take_while(|(a, b)| a == b)
                    .count()
                    .min(common);
                let mut lengths = prompt_lengths.clone();
                lengths.push(tokens.len());
                if self
                    .model_executor
                    .plan_prefix_capture_boundary(PrefixCaptureBoundary {
                        processed_tokens: producer.processed_tokens,
                        source_prompt_tokens: source_tokens.len(),
                        common_prefix_tokens: shared,
                        follower_prompt_tokens: &lengths,
                    })
                    .is_some()
                {
                    common = shared;
                    prompt_lengths = lengths;
                    followers.push(follower.key.clone());
                }
            }
            if followers.is_empty() {
                continue;
            }
            let Some(plan) =
                self.model_executor
                    .plan_prefix_capture_boundary(PrefixCaptureBoundary {
                        processed_tokens: producer.processed_tokens,
                        source_prompt_tokens: source_tokens.len(),
                        common_prefix_tokens: common,
                        follower_prompt_tokens: &prompt_lengths,
                    })
            else {
                continue;
            };
            let boundary = plan.boundary;
            if boundary <= producer.processed_tokens
                || boundary > common
                || boundary >= source_tokens.len()
                || prompt_lengths.iter().any(|&length| boundary >= length)
            {
                continue;
            }
            let Some(hold) = self
                .scheduler
                .hold_prefix_followers(&producer.key, &followers, plan)
            else {
                continue;
            };
            used.insert(producer.key.request_id().clone());
            used.extend(followers.iter().map(|key| key.request_id().clone()));
            self.prefix_rendezvous.lock().push(PrefixRendezvous {
                hold,
                followers,
                source_tokens,
                maximum_sequence_tokens: source.model_maximum_sequence_tokens(),
                expires_at,
                capture: None,
            });
        }
        Ok(())
    }

    /// Never wait here for a future device wave. This runs only at iteration
    /// boundaries; cancellation/ingress already wake that loop, and its existing
    /// idle/capacity waits also observe the explicit monotonic deadline.
    pub(super) fn refresh_prefix_rendezvous(&self) -> Result<()> {
        if self
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .is_none()
        {
            return Ok(());
        }
        self.refresh_prefix_rendezvous_at(Instant::now())
    }

    pub(in super::super) fn refresh_prefix_rendezvous_at(&self, now: Instant) -> Result<()> {
        let cohorts = std::mem::take(&mut *self.prefix_rendezvous.lock());
        let mut retained = Vec::with_capacity(cohorts.len());
        for mut cohort in cohorts {
            cohort.followers.retain(|key| {
                self.scheduler
                    .prefix_request_progress(key)
                    .is_some_and(|(_, offset)| offset == 0)
            });
            if cohort.followers.is_empty() || now >= cohort.expires_at {
                continue;
            }
            if let Some(capture) = &cohort.capture {
                match capture.status() {
                    PrefixCaptureStatus::Ready => {
                        cohort.hold.release();
                        retained.push(cohort);
                        continue;
                    }
                    PrefixCaptureStatus::Unavailable => continue,
                    PrefixCaptureStatus::Pending => {}
                }
            }
            if !cohort.hold.is_pending() {
                continue;
            }
            let Some((waiting, offset)) =
                self.scheduler.prefix_request_progress(cohort.hold.source())
            else {
                continue;
            };
            if offset >= cohort.hold.boundary() {
                continue;
            }
            if cohort.capture.is_none() && !waiting {
                let capture =
                    self.model_executor
                        .retain_prefix_capture_interest(PrefixCaptureRequest {
                            source_request_id: cohort.hold.source().request_id(),
                            source_tokens: &cohort.source_tokens,
                            maximum_sequence_tokens: cohort.maximum_sequence_tokens,
                            boundary: cohort.hold.boundary(),
                            expires_at: cohort.expires_at,
                        });
                match capture {
                    Ok(Some(capture)) if capture.boundary() == cohort.hold.boundary() => {
                        cohort.capture = Some(capture)
                    }
                    Ok(_) => continue,
                    Err(error) => {
                        warn!(%error, "Optional prefix interest unavailable before submission");
                        continue;
                    }
                }
            }
            retained.push(cohort);
        }
        *self.prefix_rendezvous.lock() = retained;
        Ok(())
    }

    /// Real capacity pressure relinquishes optional pins, including when an
    /// unrelated request needs the same model ledger. Ordinary cold backing
    /// materialization preserves sharing. Call only outside scheduler/capacity
    /// locks; releasing a pin grants no admission or physical allocation.
    pub(super) fn release_prefix_rendezvous_for_capacity_pressure(&self) {
        if self
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .is_none()
        {
            return;
        }
        let cohorts = std::mem::take(&mut *self.prefix_rendezvous.lock());
        drop(cohorts);
    }

    /// Called only after the scheduler has prepared this exact admitted target.
    /// Consume its one attempt even if physical restoration safely returns None.
    pub(super) fn take_rendezvous_checkpoint(
        &self,
        request_id: &RequestId,
    ) -> Option<Arc<dyn PrefixCaptureLease>> {
        if self
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .is_none()
        {
            return None;
        }
        let mut cohorts = self.prefix_rendezvous.lock();
        for cohort in cohorts.iter_mut() {
            let Some(index) = cohort.followers.iter().position(|key| {
                key.request_id() == request_id
                    && self.scheduler.prefix_request_progress(key) == Some((false, 0))
            }) else {
                continue;
            };
            cohort.followers.remove(index);
            if Instant::now() < cohort.expires_at {
                return cohort
                    .capture
                    .as_ref()
                    .filter(|capture| capture.status() == PrefixCaptureStatus::Ready)
                    .cloned();
            }
            return None;
        }
        None
    }

    pub(in super::super) async fn wait_for_prefix_deadline(&self) {
        let deadline = self
            .prefix_rendezvous
            .lock()
            .iter()
            .map(|cohort| cohort.expires_at)
            .min();
        match deadline {
            Some(deadline) => {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await
            }
            None => std::future::pending::<()>().await,
        }
    }
}
