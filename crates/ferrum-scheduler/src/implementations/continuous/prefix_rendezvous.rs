//! Logical prefix dependencies before admission. No resource authority lives here.
use super::*;
use ferrum_interfaces::model_executor::PrefixCapturePlan;
use std::sync::Weak;

#[derive(Clone, Debug)]
pub struct PrefixRequestKey {
    request_id: RequestId,
    ticket: WaitingAdmissionTicket,
    incarnation: Arc<()>,
}

impl PrefixRequestKey {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }
    pub fn ordinal(&self) -> u64 {
        self.ticket.get()
    }
    fn matches(&self, request: &ContinuousBatchRequest) -> bool {
        request.inner.request.id == self.request_id
            && request.waiting_admission_ticket == Some(self.ticket)
            && Arc::ptr_eq(&self.incarnation, &request.prefix_rendezvous.incarnation)
    }
}

#[derive(Clone, Debug)]
pub struct PrefixRendezvousCandidate {
    pub key: PrefixRequestKey,
    pub processed_tokens: usize,
    pub waiting: bool,
    pub priority: Priority,
}

#[derive(Debug)]
struct Dependency {
    boundary: usize,
    span: ferrum_interfaces::vnext::CheckpointTokenSpanConstraint,
    pending: AtomicBool,
}

#[derive(Debug)]
pub struct PrefixRendezvousHold {
    dependency: Arc<Dependency>,
    source: PrefixRequestKey,
    followers: Vec<PrefixRequestKey>,
}

impl PrefixRendezvousHold {
    pub fn source(&self) -> &PrefixRequestKey {
        &self.source
    }
    pub fn followers(&self) -> &[PrefixRequestKey] {
        &self.followers
    }
    pub fn boundary(&self) -> usize {
        self.dependency.boundary
    }
    pub fn is_pending(&self) -> bool {
        self.dependency.pending.load(Ordering::Acquire)
    }
    /// This only permits an ordinary admission probe. It grants no capacity.
    pub fn release(&self) {
        self.dependency.pending.store(false, Ordering::Release);
    }
}

impl Drop for PrefixRendezvousHold {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Clone, Debug)]
pub(super) struct PrefixRendezvousRequestState {
    incarnation: Arc<()>,
    attempted: bool,
    follower: Weak<Dependency>,
    source: Weak<Dependency>,
}

impl Default for PrefixRendezvousRequestState {
    fn default() -> Self {
        Self {
            incarnation: Arc::new(()),
            attempted: false,
            follower: Weak::new(),
            source: Weak::new(),
        }
    }
}

impl PrefixRendezvousRequestState {
    pub(super) fn held(&self) -> bool {
        self.follower
            .upgrade()
            .is_some_and(|dependency| dependency.pending.load(Ordering::Acquire))
    }
    pub(super) fn cap(&self, offset: usize, tokens: usize) -> usize {
        let Some(dependency) = self
            .source
            .upgrade()
            .filter(|dependency| dependency.pending.load(Ordering::Acquire))
        else {
            return tokens;
        };
        let remaining = dependency.boundary.saturating_sub(offset);
        let selected = (|| {
            let alignment = usize::try_from(dependency.span.alignment().get()).ok()?;
            let minimum = usize::try_from(dependency.span.minimum_tokens().get()).ok()?;
            let minimum = minimum
                .checked_add(alignment - 1)?
                .checked_div(alignment)?
                .checked_mul(alignment)?;
            if remaining < minimum || !remaining.is_multiple_of(alignment) {
                return None;
            }
            if remaining <= tokens {
                return Some(remaining);
            }
            let maximum = tokens.min(remaining.checked_sub(minimum)?);
            let chunk = maximum / alignment * alignment;
            (chunk >= minimum).then_some(chunk)
        })();
        selected.unwrap_or_else(|| {
            // Never steal mixed-decode budget or spin on an impossible span.
            dependency.pending.store(false, Ordering::Release);
            tokens
        })
    }
}

fn candidate(request: &ContinuousBatchRequest) -> Option<PrefixRendezvousCandidate> {
    if request.prefix_rendezvous.attempted
        || request.capacity_deferred_from_decode
        || !matches!(
            request.phase,
            RequestPhase::Waiting | RequestPhase::Prefilling
        )
    {
        return None;
    }
    if request.phase == RequestPhase::Prefilling
        && !request
            .logical_work_frontier
            .can_restore_at(request.prefill_chunk_offset)
    {
        return None;
    }
    Some(PrefixRendezvousCandidate {
        key: PrefixRequestKey {
            request_id: request.inner.request.id.clone(),
            ticket: request.waiting_admission_ticket?,
            incarnation: Arc::clone(&request.prefix_rendezvous.incarnation),
        },
        processed_tokens: request.prefill_chunk_offset,
        waiting: request.phase == RequestPhase::Waiting,
        priority: request.inner.request.priority,
    })
}

impl ContinuousBatchScheduler {
    pub fn prefix_rendezvous_candidates(&self) -> Vec<PrefixRendezvousCandidate> {
        let waiting = self.waiting_queue.read();
        let prefill = self.prefill_queue.read();
        let mut result = prefill
            .iter()
            .chain(waiting.iter())
            .filter_map(candidate)
            .collect::<Vec<_>>();
        result.sort_by_key(|item| (item.waiting, item.key.ordinal()));
        result
    }

    /// Exact current identity, including after waiting -> admitted promotion.
    pub fn prefix_request_progress(&self, key: &PrefixRequestKey) -> Option<(bool, usize)> {
        let waiting = self.waiting_queue.read();
        let prefill = self.prefill_queue.read();
        let progress = prefill
            .iter()
            .chain(waiting.iter())
            .find(|request| key.matches(request))
            .map(|request| {
                (
                    request.phase == RequestPhase::Waiting,
                    request.prefill_chunk_offset,
                )
            });
        progress
    }

    /// All-or-nothing install, before resource probes. Existing active work is
    /// never turned into a follower or preempted to obtain sharing.
    pub fn hold_prefix_followers(
        &self,
        source: &PrefixRequestKey,
        followers: &[PrefixRequestKey],
        plan: PrefixCapturePlan,
    ) -> Option<PrefixRendezvousHold> {
        let boundary = plan.boundary;
        if followers.is_empty() || boundary == 0 {
            return None;
        }
        let mut waiting = self.waiting_queue.write();
        let mut prefill = self.prefill_queue.write();
        let producer = prefill
            .iter()
            .chain(waiting.iter())
            .find(|request| source.matches(request))?;
        let producer_candidate = candidate(producer)?;
        if boundary <= producer_candidate.processed_tokens
            || !plan
                .span
                .permits((boundary - producer_candidate.processed_tokens) as u64)
        {
            return None;
        }
        let mut ids = HashSet::new();
        ids.insert(source.request_id.clone());
        for follower in followers {
            if !ids.insert(follower.request_id.clone()) {
                return None;
            }
            let request = waiting.iter().find(|request| follower.matches(request))?;
            let follower_candidate = candidate(request)?;
            if !follower_candidate.waiting
                || follower_candidate.processed_tokens != 0
                || follower_candidate.priority > producer_candidate.priority
            {
                return None;
            }
        }
        let dependency = Arc::new(Dependency {
            boundary,
            span: plan.span,
            pending: AtomicBool::new(true),
        });
        for request in prefill.iter_mut().chain(waiting.iter_mut()) {
            if source.matches(request) {
                request.prefix_rendezvous.attempted = true;
                request.prefix_rendezvous.source = Arc::downgrade(&dependency);
            } else if followers.iter().any(|key| key.matches(request)) {
                request.prefix_rendezvous.attempted = true;
                request.prefix_rendezvous.follower = Arc::downgrade(&dependency);
            }
        }
        Some(PrefixRendezvousHold {
            dependency,
            source: source.clone(),
            followers: followers.to_vec(),
        })
    }

    pub fn prefix_held_waiting_count(&self) -> usize {
        self.waiting_queue
            .read()
            .iter()
            .filter(|request| request.prefix_rendezvous.held())
            .count()
    }
}

#[cfg(test)]
mod tests;
