use crate::{
    queue::{Envelope, Shared},
    snapshot::{self, Entry, Member, Snapshot, Source},
};
use ferrum_execution::{execution_attempt::*, token_execution::ExecutedTokenWork};
use ferrum_scheduler::cost_prediction::CostWork;
use ferrum_types::ReferenceEstimatorConfig;
use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};
use tokio::sync::watch;

#[derive(Default)]
pub(super) struct Hooks {
    #[cfg(test)]
    pub gate: Option<Arc<crate::tests::Gate>>,
    #[cfg(test)]
    pub panic_next: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl Hooks {
    fn before_apply(&self) {
        #[cfg(test)]
        {
            if let Some(gate) = &self.gate {
                gate.enter();
            }
            if self
                .panic_next
                .as_ref()
                .is_some_and(|panic| panic.swap(false, Ordering::AcqRel))
            {
                panic!("injected estimator training panic");
            }
        }
    }
}

struct Domain {
    members: Vec<Member>,
    window: VecDeque<Duration>,
    observed: Instant,
}

struct Table {
    domains: HashMap<u64, Vec<Domain>>,
    count: usize,
    revision: u64,
}

pub(super) enum Excluded {
    Ineligible,
    Invalid,
}

pub(super) fn sample_work(member: &ExecutionAttemptParticipant) -> CostWork {
    CostWork {
        kind: member.kind,
        requested: member.requested,
        context_tokens: member.context_tokens,
        prefix_restore: member.prefix_restore,
        internally_reused_tokens: member.internally_reused_tokens,
    }
}

pub(super) fn validate_sample(
    sample: &ExecutionAttemptSummary,
    now: Instant,
) -> Result<Duration, Excluded> {
    if sample.eligibility != ExecutionAttemptEligibility::Completed {
        return Err(Excluded::Ineligible);
    }
    let wall = sample
        .executor_wall()
        .filter(|wall| !wall.is_zero())
        .ok_or(Excluded::Invalid)?;
    if sample.completed_at > now || sample.participants.is_empty() {
        return Err(Excluded::Invalid);
    }
    for (index, member) in sample.participants.iter().enumerate() {
        let AttemptParticipantOutcome::Completed {
            work,
            frontier_committed: true,
        } = member.outcome
        else {
            return Err(Excluded::Invalid);
        };
        let kind_matches = matches!(
            (member.kind, work),
            (AttemptWorkKind::Prefill, ExecutedTokenWork::Prefill { .. })
                | (AttemptWorkKind::Decode, ExecutedTokenWork::Decode { .. })
        );
        // A partial successful prefix is not silently assigned the requested
        // full-domain cost. Product sampling failure does not erase real work.
        if !kind_matches
            || work.range().ok() != Some(member.requested)
            || Member::from_work(sample_work(member)).is_none()
            || sample.participants[..index]
                .iter()
                .any(|prior| prior.request_id == member.request_id)
        {
            return Err(Excluded::Invalid);
        }
    }
    Ok(wall)
}

impl Table {
    fn apply(
        &mut self,
        config: &ReferenceEstimatorConfig,
        shared: &Shared,
        sample: &ExecutionAttemptSummary,
    ) -> bool {
        let wall = match validate_sample(sample, Instant::now()) {
            Ok(wall) => wall,
            Err(Excluded::Ineligible) => {
                shared
                    .counters
                    .excluded_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return false;
            }
            Err(Excluded::Invalid) => {
                shared
                    .counters
                    .invalid_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };
        let fingerprint = snapshot::fingerprint(
            sample
                .participants
                .iter()
                .map(|member| Member::from_work(sample_work(member)).expect("validated member")),
        );
        let existing = self.domains.get(&fingerprint).and_then(|bucket| {
            bucket.iter().position(|domain| {
                domain.members.len() == sample.participants.len()
                    && domain
                        .members
                        .iter()
                        .zip(&sample.participants)
                        .all(|(key, member)| {
                            Member::from_work(sample_work(member)).as_ref() == Some(key)
                        })
            })
        });
        if existing.is_none() && self.count == config.max_domains {
            shared.counters.domain_limit.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let bucket = self.domains.entry(fingerprint).or_default();
        let index = if let Some(index) = existing {
            index
        } else {
            // Domain capacity was checked before allocating its key/window.
            bucket.push(Domain {
                members: sample
                    .participants
                    .iter()
                    .map(|member| Member::from_work(sample_work(member)).expect("validated member"))
                    .collect(),
                window: VecDeque::new(),
                observed: sample.completed_at,
            });
            self.count += 1;
            bucket.len() - 1
        };
        let domain = &mut bucket[index];
        if domain.window.len() == config.window_samples {
            domain.window.pop_front();
        }
        domain.window.push_back(wall);
        domain.observed = domain.observed.max(sample.completed_at);
        shared
            .counters
            .trained_samples
            .fetch_add(1, Ordering::Relaxed);
        // Accumulating a cold domain does not change a published prediction.
        // Avoid empty revisions and scheduler wakeups until it is eligible.
        domain.window.len() >= config.min_samples
    }

    fn snapshot(&mut self, config: &ReferenceEstimatorConfig) -> Option<Snapshot> {
        let revision = self.revision.checked_add(1)?;
        let ttl = Duration::from_millis(config.snapshot_ttl_ms);
        let mut entries = HashMap::new();
        for (fingerprint, bucket) in &self.domains {
            let mut eligible = Vec::new();
            for domain in bucket {
                if domain.window.len() < config.min_samples {
                    continue;
                }
                let maximum = domain.window.iter().copied().max()?;
                let upper = Duration::try_from_secs_f64(
                    maximum.as_secs_f64() * (1.0 + config.safety_margin),
                )
                .ok()?
                .max(maximum);
                let expires = domain.observed.checked_add(ttl)?;
                eligible.push(Entry {
                    members: domain.members.clone(),
                    central: Duration::try_from_secs_f64(
                        domain.window.iter().map(Duration::as_secs_f64).sum::<f64>()
                            / domain.window.len() as f64,
                    )
                    .ok()?,
                    upper,
                    samples: domain.window.len(),
                    observed: domain.observed,
                    expires,
                });
            }
            if !eligible.is_empty() {
                entries.insert(*fingerprint, eligible);
            }
        }
        self.revision = revision;
        Some(Snapshot {
            revision,
            maximum_participants: config.max_participants,
            entries,
        })
    }
}

struct Liveness<'a> {
    shared: Arc<Shared>,
    receiver: &'a crossbeam_channel::Receiver<Envelope>,
    wake: &'a crossbeam_channel::Receiver<()>,
}

impl Drop for Liveness<'_> {
    fn drop(&mut self) {
        self.shared.stop();
        // A surviving observer retains the bounded channel's allocation, even
        // after its last receiver drops. Retire queued payload explicitly on
        // unwind. The publication guard covers producers that passed the open
        // check before stop, so they cannot enqueue after this drain completes.
        // Normal shutdown already trained/drained everything before this point.
        while !self.shared.drained() {
            crossbeam_channel::select! {
                recv(self.receiver) -> message => {
                    if let Ok(envelope) = message {
                        drop(envelope);
                        self.shared.counters.discarded_pending.fetch_add(1, Ordering::Relaxed);
                    }
                }
                recv(self.wake) -> _ => {}
            }
        }
    }
}

pub(super) fn run(
    config: ReferenceEstimatorConfig,
    shared: Arc<Shared>,
    source: Arc<Source>,
    revisions: watch::Sender<u64>,
    receiver: crossbeam_channel::Receiver<Envelope>,
    wake: crossbeam_channel::Receiver<()>,
    hooks: Hooks,
) {
    let mut table = Table {
        domains: HashMap::new(),
        count: 0,
        revision: 0,
    };
    receive(Arc::clone(&shared), receiver, wake, hooks, |sample| {
        if table.apply(&config, &shared, sample) {
            if let Some(snapshot) = table.snapshot(&config) {
                let revision = snapshot.revision;
                source.install(snapshot);
                revisions.send_replace(revision);
                shared
                    .counters
                    .published_snapshots
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                shared
                    .counters
                    .training_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    });
}

pub(super) fn receive(
    shared: Arc<Shared>,
    receiver: crossbeam_channel::Receiver<Envelope>,
    wake: crossbeam_channel::Receiver<()>,
    hooks: Hooks,
    mut apply: impl FnMut(&ExecutionAttemptSummary),
) {
    let _liveness = Liveness {
        shared: Arc::clone(&shared),
        receiver: &receiver,
        wake: &wake,
    };
    loop {
        if shared.drained() {
            break;
        }
        crossbeam_channel::select! {
            recv(receiver) -> message => {
                let Ok(envelope) = message else { break; };
                hooks.before_apply();
                apply(&envelope.summary);
            }
            recv(wake) -> _ => {}
        }
    }
}
