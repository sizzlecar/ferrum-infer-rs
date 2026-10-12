use super::*;

type Stages<'a> = BTreeMap<&'a str, Vec<u64>>;
type SubphaseEvents<'a> = BTreeMap<CycleWaveKey, BTreeMap<&'a str, Vec<&'a CycleEvent>>>;

const OWN: &[&str] = &[
    "terminal_readback_end",
    "completion_result_ready",
    "completion_await_resumed",
    "retirement_start",
    "retirement_end",
    "product_return",
];
const NEXT: &[&str] = &[
    "loop_resumed",
    "iteration_lock_start",
    "iteration_lock_end",
    "admission_start",
    "admission_end",
    "schedule_start",
    "schedule_end",
    "prepare_start",
    "prepare_end",
    "encode_start",
    "command_enqueue_start",
    "compute_graph_launch_start",
];
const CHAIN: &[&str] = &[
    "terminal_readback_end",
    "completion_result_ready",
    "completion_await_resumed",
    "retirement_start",
    "retirement_end",
    "product_return",
    "call_returned",
    "iteration_end",
    "next_loop_resumed",
    "next_iteration_lock_start",
    "next_iteration_lock_end",
    "next_admission_start",
    "next_admission_end",
    "next_schedule_start",
    "next_schedule_end",
    "next_prepare_start",
    "next_prepare_end",
    "next_encode_start",
    "next_command_enqueue_start",
    "next_compute_graph_launch_start",
];
const NEXT_SUBPHASES: &[(&str, &str, &str)] = &[
    (
        "program_binding_coalesce_start",
        "program_binding_coalesce_end",
        "next_program_binding_coalesce",
    ),
    (
        "scatter_allocation_start",
        "scatter_allocation_end",
        "next_scatter_allocation",
    ),
    (
        "scatter_readiness_wait_start",
        "scatter_readiness_wait_end",
        "next_scatter_readiness_wait",
    ),
];
const NEXT_COALESCE_RESIDUALS: &[&str] = &[
    "next_encode_before_binding_coalesce",
    "next_post_binding_coalesce_to_enqueue",
];

fn unique(stages: Option<&Stages<'_>>, name: &str) -> Option<u64> {
    let values = stages?.get(name)?;
    match values.as_slice() {
        [at] => Some(*at),
        _ => None,
    }
}

fn call_key(wave: &Wave<'_>) -> Option<CallKey> {
    let mut calls = wave.witness.owners.iter().map(|owner| {
        owner
            .context
            .map(|context| (context.clock_id, context.iteration, context.call))
    });
    let first = calls.next()??;
    calls.all(|call| call == Some(first)).then_some(first)
}

fn next_subphases(
    events: &SubphaseEvents<'_>,
    next: Option<CycleWaveKey>,
    boundaries: &mut BTreeMap<String, Option<u64>>,
    missing: &mut Vec<String>,
) -> Vec<(&'static str, Interval)> {
    let Some(stages) = next.and_then(|next| events.get(&next)) else {
        return Vec::new();
    };
    let mut intervals = Vec::new();
    for &(start, end, name) in NEXT_SUBPHASES {
        let starts = stages.get(start).map(Vec::as_slice).unwrap_or(&[]);
        let ends = stages.get(end).map(Vec::as_slice).unwrap_or(&[]);
        // These probes are optional. An old capture or a backend without the
        // probe has absent samples, not zero-duration observations.
        if starts.is_empty() && ends.is_empty() {
            continue;
        }
        boundaries.insert(
            format!("next_{start}"),
            match starts {
                [event] => Some(event.at.nanos),
                _ => None,
            },
        );
        boundaries.insert(
            format!("next_{end}"),
            match ends {
                [event] => Some(event.at.nanos),
                _ => None,
            },
        );
        let ([start], [end]) = (starts, ends) else {
            missing.push(format!("{name}.unique_pair"));
            continue;
        };
        if start.scope != end.scope {
            missing.push(format!("{name}.scope_mismatch"));
            continue;
        }
        if end.outcome.as_deref() != Some("success") {
            missing.push(format!("{name}.successful_end"));
            continue;
        }
        if end.at.nanos < start.at.nanos
            || boundaries["next_encode_start"].is_none_or(|at| start.at.nanos < at)
            || boundaries["next_command_enqueue_start"].is_none_or(|at| end.at.nanos > at)
        {
            missing.push(format!("{name}.outside_encode_to_enqueue"));
            continue;
        }
        intervals.push((name, (start.at.nanos, end.at.nanos)));
    }
    let observed = |name| {
        intervals
            .iter()
            .find_map(|&(candidate, interval)| (candidate == name).then_some(interval))
    };
    let coalesce = observed("next_program_binding_coalesce");
    let allocation = observed("next_scatter_allocation");
    let readiness = observed("next_scatter_readiness_wait");
    // In the regular compact-scatter path a successful allocation is always
    // followed by its readiness wait, inside the same coalescing call. No
    // allocation attempt (for example a budget fallback) remains legitimate.
    if allocation.is_some() || readiness.is_some() {
        match (coalesce, allocation, readiness) {
            (Some(coalesce), Some(allocation), Some(readiness))
                if coalesce.0 <= allocation.0
                    && allocation.1 <= readiness.0
                    && readiness.1 <= coalesce.1 => {}
            _ => missing.push("next_scatter.allocation_readiness_nesting".to_owned()),
        }
    }
    if let Some((start, end)) = coalesce {
        if let (Some(encode), Some(enqueue)) = (
            boundaries["next_encode_start"],
            boundaries["next_command_enqueue_start"],
        ) {
            intervals.push((NEXT_COALESCE_RESIDUALS[0], (encode, start)));
            intervals.push((NEXT_COALESCE_RESIDUALS[1], (end, enqueue)));
        }
    }
    intervals
}

pub(super) fn collect(
    events: &[&CycleEvent],
    waves: &BTreeMap<CycleWaveKey, Wave<'_>>,
    calls: &BTreeMap<CallKey, Stages<'_>>,
    iterations: &BTreeMap<u64, Stages<'_>>,
    physical_width: Option<usize>,
    globally_usable: bool,
) -> (Vec<DecodeCycleCohort>, BTreeMap<String, IntervalMetric>) {
    let mut cohorts = Vec::new();
    let mut intervals = BTreeMap::<String, Vec<Interval>>::new();
    let mut subphase_events = SubphaseEvents::new();
    for &event in events {
        if NEXT_SUBPHASES
            .iter()
            .any(|&(start, end, _)| event.stage == start || event.stage == end)
        {
            if let Some(key) = event.scope.wave_key() {
                subphase_events
                    .entry(key)
                    .or_default()
                    .entry(&event.stage)
                    .or_default()
                    .push(event);
            }
        }
    }
    let graph_starts = events
        .iter()
        .filter_map(|event| {
            if event.stage != "compute_graph_launch_start" {
                return None;
            }
            let key = event.scope.wave_key()?;
            waves.contains_key(&key).then_some((event.at, key))
        })
        .collect::<Vec<_>>();
    let coverage = super::coverage::CoverageIndex::new(events, waves);
    let mut selected = waves
        .iter()
        .filter(|(_, wave)| {
            wave.witness.kind == "decode"
                && physical_width.is_none_or(|width| width == wave.witness.owners.len())
        })
        .collect::<Vec<_>>();
    selected.sort_by_key(|(_, wave)| unique(Some(&wave.stages), "wave_prepared"));
    for (&key, wave) in selected {
        let call = call_key(wave);
        let mut boundaries = BTreeMap::new();
        for &stage in OWN {
            boundaries.insert(stage.to_owned(), unique(Some(&wave.stages), stage));
        }
        boundaries.insert(
            "call_returned".to_owned(),
            call.and_then(|key| unique(calls.get(&key), "call_returned")),
        );
        boundaries.insert(
            "iteration_end".to_owned(),
            call.and_then(|(_, iteration, _)| unique(iterations.get(&iteration), "iteration_end")),
        );
        // Select the first observed next physical graph, regardless of kind or
        // width. Filtering first would silently discard intervening mixed work.
        let anchor = boundaries["product_return"].or(boundaries["terminal_readback_end"]);
        let next = anchor.and_then(|anchor| {
            let start = graph_starts.partition_point(|(at, _)| at.nanos < anchor);
            graph_starts[start..].iter().find_map(|(at, candidate)| {
                (at.clock_id == key.clock_id && *candidate != key).then_some(*candidate)
            })
        });
        let next_wave = next.and_then(|key| waves.get(&key));
        let next_call = next_wave.and_then(call_key);
        for &stage in NEXT {
            let value = match stage {
                "prepare_start" => next_call.and_then(|key| unique(calls.get(&key), stage)),
                "prepare_end"
                | "encode_start"
                | "command_enqueue_start"
                | "compute_graph_launch_start" => {
                    next_wave.and_then(|wave| unique(Some(&wave.stages), stage))
                }
                _ => next_call
                    .and_then(|(_, iteration, _)| unique(iterations.get(&iteration), stage)),
            };
            boundaries.insert(format!("next_{stage}"), value);
        }
        let mut missing = CHAIN
            .iter()
            .filter(|&&name| boundaries[name].is_none())
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        if wave.submission.is_none() {
            missing.push("submission_bound".to_owned());
        }
        if next_wave.is_some_and(|wave| wave.submission.is_none()) {
            missing.push("next_submission_bound".to_owned());
        }
        if call.is_none() {
            missing.push("decode_call_context".to_owned());
        }
        // The launch's call identifies its iteration. A different earlier loop
        // must remain visible; do not turn intervening idle/empty iterations
        // into the immediate next-cycle chain.
        if let (Some((_, previous, _)), Some((_, next, _))) = (call, next_call) {
            if next != previous.checked_add(1).unwrap_or(previous) {
                missing.push("next_iteration_is_not_adjacent".to_owned());
            }
        }
        for pair in CHAIN.windows(2) {
            if let (Some(start), Some(end)) = (boundaries[pair[0]], boundaries[pair[1]]) {
                if end < start {
                    missing.push(format!("reversed:{}->{}", pair[0], pair[1]));
                }
            }
        }
        let call_returned = boundaries["call_returned"];
        let iteration_end = boundaries["iteration_end"];
        let owner_coverage = coverage.assess(
            key,
            wave,
            call_returned,
            iteration_end,
            &mut boundaries,
            &mut missing,
        );
        missing.extend(coverage.unknown_outcomes(key).cloned());
        if let Some(next) = next {
            missing.extend(
                coverage
                    .unknown_outcomes(next)
                    .map(|label| format!("next.{label}")),
            );
        }
        let failed = coverage.failed(key) || next.is_some_and(|key| coverage.failed(key));
        let subphases = next_subphases(&subphase_events, next, &mut boundaries, &mut missing);
        let classification = if failed {
            "failed_window"
        } else if next.is_none() && anchor.is_some() {
            "incomplete_tail"
        } else if next_wave.is_some_and(|next| next.witness.kind != "decode") {
            "mixed_or_prefill_next"
        } else if next_wave
            .is_some_and(|next| next.witness.owners.len() != wave.witness.owners.len())
        {
            "different_width_next"
        } else if !globally_usable {
            "unqualified_capture"
        } else if missing.is_empty() {
            "complete_decode_to_decode"
        } else {
            "incomplete_edges"
        };
        if classification == "complete_decode_to_decode" {
            for (name, interval) in subphases {
                intervals.entry(name.to_owned()).or_default().push(interval);
            }
            for pair in CHAIN.windows(2) {
                if let (Some(start), Some(end)) = (boundaries[pair[0]], boundaries[pair[1]]) {
                    intervals
                        .entry(format!("{}->{}", pair[0], pair[1]))
                        .or_default()
                        .push((start, end));
                }
            }
            if let (Some(start), Some(end)) = (
                boundaries["terminal_readback_end"],
                boundaries["next_compute_graph_launch_start"],
            ) {
                intervals
                    .entry("terminal_readback_end->next_host_graph_launch".to_owned())
                    .or_default()
                    .push((start, end));
            }
        }
        cohorts.push(DecodeCycleCohort {
            wave: key,
            width: wave.witness.owners.len(),
            iteration: call.map(|(_, iteration, _)| iteration),
            call: call.map(|(_, _, call)| call),
            next_wave: next,
            next_wave_kind: next_wave.map(|wave| wave.witness.kind.clone()),
            next_width: next_wave.map(|wave| wave.witness.owners.len()),
            classification: classification.to_owned(),
            boundaries,
            missing_edges: missing,
            owner_coverage,
        });
    }
    // An unsupported chain has zero samples with absent quantiles, not a
    // fabricated zero-latency sample.
    let mut metrics = BTreeMap::new();
    for pair in CHAIN.windows(2) {
        let name = format!("{}->{}", pair[0], pair[1]);
        metrics.insert(
            name.clone(),
            super::analysis::metric(intervals.get(&name).map(Vec::as_slice).unwrap_or(&[])),
        );
    }
    let name = "terminal_readback_end->next_host_graph_launch";
    metrics.insert(
        name.to_owned(),
        super::analysis::metric(intervals.get(name).map(Vec::as_slice).unwrap_or(&[])),
    );
    for &(_, _, name) in NEXT_SUBPHASES {
        metrics.insert(
            name.to_owned(),
            super::analysis::metric(intervals.get(name).map(Vec::as_slice).unwrap_or(&[])),
        );
    }
    for &name in NEXT_COALESCE_RESIDUALS {
        metrics.insert(
            name.to_owned(),
            super::analysis::metric(intervals.get(name).map(Vec::as_slice).unwrap_or(&[])),
        );
    }
    (cohorts, metrics)
}
