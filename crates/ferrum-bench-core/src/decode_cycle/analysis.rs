use super::*;

const PAIRS: &[(&str, &str, &str)] = &[
    (
        "iteration_lock_start",
        "iteration_lock_end",
        "iteration_lock",
    ),
    ("admission_start", "admission_end", "admission"),
    ("schedule_start", "schedule_end", "schedule"),
    ("call_started", "call_returned", "call"),
    ("encode_start", "encode_end", "encode"),
    (
        "command_enqueue_start",
        "command_enqueue_end",
        "command_enqueue",
    ),
    (
        "compute_graph_launch_start",
        "compute_graph_launch_end",
        "host_compute_graph_launch",
    ),
    (
        "completion_queued",
        "completion_worker_started",
        "completion_queue",
    ),
    (
        "completion_worker_started",
        "completion_result_ready",
        "completion_worker",
    ),
    (
        "terminal_readback_start",
        "terminal_readback_end",
        "terminal_readback",
    ),
    (
        "completion_result_ready",
        "completion_await_resumed",
        "completion_await_resume",
    ),
    ("retirement_start", "retirement_end", "retirement"),
    ("request_lock_start", "request_lock_end", "request_lock"),
    ("sample_start", "sample_end", "sample"),
    ("stop_check_start", "stop_check_end", "stop_check"),
    ("text_decode_start", "text_decode_end", "text_decode"),
    ("text_lock_start", "text_lock_end", "text_lock"),
    ("stream_update_start", "stream_update_end", "stream_update"),
    ("stream_send_start", "stream_send_end", "stream_send"),
    ("stream_yield_start", "stream_yield_resumed", "stream_yield"),
    ("loop_yield_start", "loop_yield_resumed", "loop_yield"),
    (
        "request_complete_start",
        "request_complete_end",
        "request_complete",
    ),
];

pub fn analyze_capture(
    capture: &DecodeCycleCapture,
    physical_width: Option<usize>,
) -> DecodeCycleAnalysis {
    let mut issues = BTreeMap::new();
    let stats = &capture.stats;
    for (name, value) in [
        ("collector_busy", stats.busy),
        ("collector_full", stats.full),
        ("collector_closed", stats.closed),
        ("collector_poisoned", stats.poisoned),
        ("collector_invalid_metadata", stats.invalid_metadata),
        (
            "collector_outstanding_producers",
            stats.outstanding_producers,
        ),
        (
            "collector_active_producers_at_close",
            capture.active_producers_at_close,
        ),
    ] {
        if value != 0 {
            issue(&mut issues, name);
        }
    }
    if stats.accepted != capture.events.len() as u64 || stats.attempted < stats.accepted {
        issue(&mut issues, "collector_counter_inconsistent");
    }
    if capture.capacity == 0 || capture.events.len() > capture.capacity {
        issue(&mut issues, "collector_capacity_inconsistent");
    }
    if stats.attempted != stats.accepted && issues.is_empty() {
        issue(&mut issues, "collector_unexplained_attempts");
    }
    let collector_complete = issues.is_empty();
    if capture.clock.id == 0 || capture.clock.wall_anchor_unix_nanos <= 0 {
        issue(&mut issues, "invalid_clock");
    }
    if physical_width == Some(0) {
        issue(&mut issues, "invalid_physical_width_filter");
    }
    if capture.events.is_empty() {
        issue(&mut issues, "empty_capture");
    }
    let before_identity = issues.values().sum::<usize>();
    let mut events = capture.events.iter().collect::<Vec<_>>();
    events.sort_by_key(|event| event.at.nanos);
    let mut stage_counts = BTreeMap::new();
    let mut outcome_counts = BTreeMap::new();
    for event in &events {
        *stage_counts.entry(event.stage.clone()).or_default() += 1;
        if let Some(outcome) = &event.outcome {
            *outcome_counts.entry(outcome.clone()).or_default() += 1;
        }
    }
    let mut waves = BTreeMap::<CycleWaveKey, Wave<'_>>::new();
    for event in &events {
        if event.at.clock_id != capture.clock.id
            || event
                .scope
                .clock_id()
                .is_some_and(|id| id != capture.clock.id)
        {
            issue(&mut issues, "foreign_clock");
        }
        if event.stage != "wave_prepared" {
            if event.wave_witness.is_some() {
                issue(&mut issues, "unexpected_wave_witness");
            }
            continue;
        }
        let Some(witness) = &event.wave_witness else {
            issue(&mut issues, "missing_wave_witness");
            continue;
        };
        let key = witness_key(witness);
        if event.scope.wave_key() != Some(key)
            || !matches!(event.scope, CycleScope::Wave { .. })
            || event.scope.prepared_fingerprint()
                != Some(witness.prepared_wave_fingerprint.as_str())
            || witness.clock_id != capture.clock.id
            || witness.batch_step_id == 0
            || witness.batch_invocation_id == 0
            || witness.lane_id == 0
            || witness.device_id.is_empty()
            || witness.runtime_implementation_fingerprint.is_empty()
            || witness.prepared_wave_fingerprint.is_empty()
            || !matches!(witness.kind.as_str(), "prefill" | "decode" | "mixed")
            || witness.owners.is_empty()
        {
            issue(&mut issues, "invalid_wave_identity");
        }
        validate_owners(witness, &mut issues);
        if waves
            .insert(
                key,
                Wave {
                    witness,
                    stages: BTreeMap::new(),
                    submission: None,
                },
            )
            .is_some()
        {
            issue(&mut issues, "duplicate_wave_prepared");
        }
    }
    let mut submissions = BTreeMap::<SubmissionKey, (CycleWaveKey, &str)>::new();
    for event in &events {
        if event.stage != "submission_bound" {
            continue;
        }
        let CycleScope::Submission {
            clock_id,
            slot_id,
            submission_fingerprint,
            ..
        } = &event.scope
        else {
            issue(&mut issues, "invalid_submission_bound_scope");
            continue;
        };
        let Some(key) = event.scope.wave_key() else {
            continue;
        };
        let Some(wave) = waves.get_mut(&key) else {
            issue(&mut issues, "submission_without_prepared_wave");
            continue;
        };
        if *slot_id == 0
            || submission_fingerprint.is_empty()
            || event.scope.prepared_fingerprint()
                != Some(wave.witness.prepared_wave_fingerprint.as_str())
        {
            issue(&mut issues, "invalid_submission_identity");
        }
        let slot = (*clock_id, *slot_id, submission_fingerprint.clone());
        if wave
            .submission
            .replace((slot.clone(), submission_fingerprint))
            .is_some()
        {
            issue(&mut issues, "duplicate_submission_bound");
        }
        if submissions
            .insert(slot, (key, submission_fingerprint))
            .is_some()
        {
            issue(&mut issues, "reused_submission_slot");
        }
    }
    let mut calls = BTreeMap::<CallKey, BTreeMap<&str, Vec<u64>>>::new();
    let mut iterations = BTreeMap::<u64, BTreeMap<&str, Vec<u64>>>::new();
    for event in &events {
        if let Some(key) = event.scope.wave_key() {
            match waves.get_mut(&key) {
                Some(wave) => {
                    if event.scope.prepared_fingerprint()
                        != Some(wave.witness.prepared_wave_fingerprint.as_str())
                    {
                        issue(&mut issues, "wave_fingerprint_mismatch");
                    }
                    wave.stages
                        .entry(&event.stage)
                        .or_default()
                        .push(event.at.nanos);
                }
                None => issue(&mut issues, "scope_without_prepared_wave"),
            }
        }
        match &event.scope {
            CycleScope::Iteration { iteration } => {
                iterations
                    .entry(*iteration)
                    .or_default()
                    .entry(&event.stage)
                    .or_default()
                    .push(event.at.nanos);
            }
            CycleScope::Call { context } => {
                calls
                    .entry((context.clock_id, context.iteration, context.call))
                    .or_default()
                    .entry(&event.stage)
                    .or_default()
                    .push(event.at.nanos);
            }
            CycleScope::Submission {
                clock_id,
                slot_id,
                submission_fingerprint,
                ..
            } => {
                if submissions
                    .get(&(*clock_id, *slot_id, submission_fingerprint.clone()))
                    .is_none_or(|(key, fingerprint)| {
                        Some(*key) != event.scope.wave_key()
                            || *fingerprint != submission_fingerprint
                    })
                {
                    issue(&mut issues, "submission_scope_mismatch");
                }
            }
            CycleScope::Participant {
                clock_id,
                slot_id,
                submission_fingerprint,
                owner,
            } => {
                let bound = submissions.get(&(*clock_id, *slot_id, submission_fingerprint.clone()));
                let valid = bound.is_some_and(|(key, fingerprint)| {
                    *fingerprint == submission_fingerprint
                        && waves.get(key).and_then(|wave| {
                            wave.witness.owners.get(owner.canonical_ordinal as usize)
                        }) == Some(owner)
                });
                if !valid {
                    issue(&mut issues, "participant_owner_mismatch");
                }
            }
            CycleScope::Wave { .. } => {}
        }
    }
    let identity_valid = issues.values().sum::<usize>() == before_identity;
    let (phase_metrics, unmatched_boundaries, union) = phases(&events, capture.clock.id);
    let (cohorts, chain_metrics) = super::cohorts::collect(
        &events,
        &waves,
        &calls,
        &iterations,
        physical_width,
        collector_complete && identity_valid && issues.is_empty(),
    );
    DecodeCycleAnalysis {
        clock_id: capture.clock.id,
        events: events.len(),
        waves: waves.len(),
        physical_width_filter: physical_width,
        collector_complete,
        identity_valid,
        usable: collector_complete && identity_valid && issues.is_empty(),
        issues,
        stage_counts,
        outcome_counts,
        phase_metrics,
        unmatched_boundaries,
        observed_interval_union_ns: union,
        chain_metrics,
        cohorts,
        semantics: vec![
            "all timestamps use one host monotonic clock; records are sorted by timestamp",
            "host_compute_graph_launch is cuGraphLaunch host API, not first kernel or GPU start",
            "terminal_readback includes fence waiting, copies and terminal validation",
            "phase union avoids summing overlapping participant prefixes or nested phases",
            "missing edges are absent evidence, never zero duration",
            "coarse observed intervals do not establish idle or removable CPU time",
            "chain metrics include only complete same-width Decode-to-Decode windows",
            "optional next-wave binding/scatter subphase metrics require paired successful observations; absent probes have no samples",
            "scatter allocation/readiness coverage is regular binding coalesce only; compiled binding preparation is unchanged and unobserved",
            "nested subphase unions overlap their parent and must not be added to parent or whole-chain unions",
            "qualified before-coalesce, coalesce and after-coalesce intervals partition next encode-to-enqueue without overlap",
            "complete source windows require every owner's lock, sample, commit and stop boundaries",
            "stream/text/yield branches are conditional; absent branches do not imply zero duration",
            "whole-capture diagnostic: client warmup/formal/startup phase IDs have not been joined",
        ],
    }
}

fn validate_owners(witness: &CycleWaveWitness, issues: &mut BTreeMap<String, usize>) {
    let mut requests = BTreeSet::new();
    let mut identities = BTreeSet::new();
    let mut authorities = BTreeSet::new();
    let mut callers = BTreeSet::new();
    for (canonical, owner) in witness.owners.iter().enumerate() {
        let domain = owner
            .context
            .map(|context| (context.iteration, context.call));
        if owner.canonical_ordinal as usize != canonical
            || owner.frame_id == 0
            || owner.request_id.is_empty()
            || owner.request_identity.is_empty()
            || !requests.insert(&owner.request_id)
            || !identities.insert(&owner.request_identity)
            || !authorities.insert((
                owner.sequence_authority,
                owner.request_authority,
                owner.frame_id,
            ))
            || !callers.insert((domain, owner.caller_ordinal))
            || owner.context.is_some_and(|context| {
                context.clock_id != witness.clock_id
                    || context.caller_ordinal != owner.caller_ordinal
            })
        {
            issue(issues, "invalid_canonical_owner_map");
        }
    }
}

fn phases(
    events: &[&CycleEvent],
    clock_id: u64,
) -> (
    BTreeMap<String, IntervalMetric>,
    BTreeMap<String, usize>,
    u64,
) {
    let mut result = BTreeMap::new();
    let mut missing = BTreeMap::new();
    let mut all = Vec::new();
    for &(start, end, name) in PAIRS {
        let mut pending = BTreeMap::<CycleScope, Vec<u64>>::new();
        let mut intervals = Vec::new();
        let mut phase_events = events
            .iter()
            .copied()
            .filter(|event| event.stage == start || event.stage == end)
            .collect::<Vec<_>>();
        // Equal timestamps carry a legitimate zero-duration interval even if
        // recorder insertion order differs. Start precedes End only for this
        // exact phase/scope pairing; no absent boundary is synthesized.
        phase_events.sort_by_key(|event| (event.at.nanos, event.stage != start));
        for event in phase_events {
            if event.at.clock_id != clock_id
                || event.scope.clock_id().is_some_and(|id| id != clock_id)
            {
                continue;
            }
            if event.stage == start {
                pending
                    .entry(event.scope.clone())
                    .or_default()
                    .push(event.at.nanos);
            } else if event.stage == end {
                match pending.get_mut(&event.scope).and_then(Vec::pop) {
                    Some(start) => intervals.push((start, event.at.nanos)),
                    None => {
                        *missing.entry(name.to_owned()).or_default() += 1;
                    }
                }
            }
        }
        let remaining = pending.values().map(Vec::len).sum::<usize>();
        if remaining != 0 {
            *missing.entry(name.to_owned()).or_default() += remaining;
        }
        all.extend_from_slice(&intervals);
        result.insert(name.to_owned(), metric(&intervals));
    }
    (result, missing, union_ns(&all))
}

pub(super) fn metric(intervals: &[Interval]) -> IntervalMetric {
    let values = intervals
        .iter()
        .map(|(start, end)| (end - start) as f64)
        .collect::<Vec<_>>();
    IntervalMetric {
        count: intervals.len(),
        median_ns: (!values.is_empty()).then(|| percentile(&values, 0.5)),
        p99_ns: (!values.is_empty()).then(|| percentile(&values, 0.99)),
        union_ns: union_ns(intervals),
    }
}

fn union_ns(intervals: &[Interval]) -> u64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_unstable();
    let mut total = 0;
    let mut current: Option<Interval> = None;
    for (start, end) in intervals {
        match current {
            Some((left, right)) if start <= right => current = Some((left, right.max(end))),
            Some((left, right)) => {
                total += right - left;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        total += end - start;
    }
    total
}
