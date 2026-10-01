//! A passive, nonblocking join of scheduler membership and original ingress.
use super::*;
use ferrum_scheduler::implementations::continuous::ContinuousSchedulerAdmissionCounts;
use ferrum_types::ExecutorQueueObservation;

pub(super) struct Clock {
    instance: String,
    origin: Instant,
}
impl Clock {
    pub fn new() -> Self {
        Self {
            instance: uuid::Uuid::new_v4().to_string(),
            origin: Instant::now(),
        }
    }
}

pub(super) fn capture(
    inner: &EngineInner,
) -> std::result::Result<(ContinuousSchedulerAdmissionCounts, ExecutorQueueObservation), String> {
    // Never wait behind execution or hold a read lock across HTTP/IO. Retain
    // sequence ownership while reading the index so a reused RequestId cannot
    // join an earlier incarnation's ingress. Both acquisitions are try-only.
    let sequences = inner
        .sequences
        .try_read()
        .ok_or("sequence observation busy")?;
    inner
        .scheduler
        .try_observe_admission_phases(|phases, now| {
            observe(&inner.admission_observer, phases, now, |id| {
                sequences
                    .get(id)
                    .and_then(|s| s.admission_observation_ingress)
            })
        })
        .ok_or_else(|| "scheduler observation busy".to_owned())?
}

fn observe(
    clock: &Clock,
    phases: &HashMap<RequestId, RequestPhase>,
    now: Instant,
    mut ingress: impl FnMut(&RequestId) -> Option<Instant>,
) -> std::result::Result<(ContinuousSchedulerAdmissionCounts, ExecutorQueueObservation), String> {
    let mut queue = ExecutorQueueObservation {
        schema_version: 1,
        engine_instance: clock.instance.clone(),
        observed_at_ns: nanos(
            now.checked_duration_since(clock.origin)
                .ok_or("queue clock precedes engine origin")?,
        )?,
        waiting_requests: 0,
        active_prefill_sequences: 0,
        active_decode_sequences: 0,
        preempted_requests: 0,
        oldest_waiting_ingress_age_ns: None,
        oldest_unfinished_ingress_age_ns: None,
    };
    for (id, phase) in phases {
        let counter = match phase {
            RequestPhase::Waiting => &mut queue.waiting_requests,
            RequestPhase::Prefilling => &mut queue.active_prefill_sequences,
            RequestPhase::Decoding => &mut queue.active_decode_sequences,
            RequestPhase::Preempted => &mut queue.preempted_requests,
            RequestPhase::Completed | RequestPhase::Cancelled | RequestPhase::AdmissionFailed => {
                continue
            }
        };
        *counter = counter.checked_add(1).ok_or("queue count exceeds u32")?;
        let entered = ingress(id).ok_or("scheduler member lacks original trusted ingress")?;
        let age = nanos(
            now.checked_duration_since(entered)
                .ok_or("original ingress is in the future")?,
        )?;
        queue.oldest_unfinished_ingress_age_ns =
            Some(queue.oldest_unfinished_ingress_age_ns.unwrap_or(0).max(age));
        if *phase == RequestPhase::Waiting {
            queue.oldest_waiting_ingress_age_ns =
                Some(queue.oldest_waiting_ingress_age_ns.unwrap_or(0).max(age));
        }
    }
    queue.validate()?;
    let counts = ContinuousSchedulerAdmissionCounts {
        waiting_requests: queue.waiting_requests as usize,
        active_prefill_sequences: queue.active_prefill_sequences as usize,
        active_decode_sequences: queue.active_decode_sequences as usize,
    };
    Ok((counts, queue))
}
fn nanos(duration: Duration) -> std::result::Result<u64, String> {
    duration
        .as_nanos()
        .try_into()
        .map_err(|_| "queue clock duration exceeds u64 nanoseconds".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queue_observation_retains_original_ingress_across_requeue_and_separates_phases() {
        let clock = Clock::new();
        let now = clock.origin + Duration::from_secs(10);
        let id = RequestId::new();
        let active = RequestId::new();
        let mut phases = HashMap::from([
            (id.clone(), RequestPhase::Waiting),
            (active.clone(), RequestPhase::Decoding),
        ]);
        let ingress = |key: &RequestId| {
            Some(clock.origin + Duration::from_secs(if key == &id { 1 } else { 2 }))
        };
        let (_, first) = observe(&clock, &phases, now, ingress).unwrap();
        assert_eq!(first.oldest_waiting_ingress_age_ns, Some(9_000_000_000));
        phases.insert(id.clone(), RequestPhase::Prefilling);
        let (_, running) = observe(&clock, &phases, now, ingress).unwrap();
        assert_eq!(running.oldest_waiting_ingress_age_ns, None);
        assert_eq!(
            running.oldest_unfinished_ingress_age_ns,
            first.oldest_unfinished_ingress_age_ns
        );
        phases.insert(id.clone(), RequestPhase::Waiting);
        let (_, returned) =
            observe(&clock, &phases, now + Duration::from_secs(1), ingress).unwrap();
        assert_eq!(returned.oldest_waiting_ingress_age_ns, Some(10_000_000_000));
        assert_eq!(returned.active_decode_sequences, 1);
    }
    #[test]
    fn queue_observation_missing_or_future_ingress_is_unknown_and_empty_is_known() {
        let clock = Clock::new();
        let now = clock.origin + Duration::from_secs(1);
        let phases = HashMap::from([(RequestId::new(), RequestPhase::Waiting)]);
        assert!(observe(&clock, &phases, now, |_| None).is_err());
        assert!(observe(&clock, &phases, now, |_| Some(now + Duration::from_secs(1))).is_err());
        let (_, empty) = observe(&clock, &HashMap::new(), now, |_| None).unwrap();
        assert_eq!(empty.oldest_unfinished_ingress_age_ns, None);
        assert_ne!(clock.instance, Clock::new().instance);
    }
}
