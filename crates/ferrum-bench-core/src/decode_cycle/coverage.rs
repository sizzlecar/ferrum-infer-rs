//! Bind postprocessing and failures to the actual participant/receipt owner map.
use super::*;

// The current coarse producer always supplies an outcome for these stages.
// Other End/Ready/Retirement/Stop/Update/Yield records deliberately use None.
const CURRENT_OUTCOME_STAGES: &[&str] = &[
    "call_returned",
    "sample_end",
    "text_decode_end",
    "stream_send_end",
    "request_complete_end",
    "compute_graph_launch_end",
    "program_binding_coalesce_end",
    "scatter_allocation_end",
    "scatter_readiness_wait_end",
    "failure_observed",
];

const REQUIRED: &[&str] = &[
    "request_lock_start",
    "request_lock_end",
    "sample_start",
    "sample_end",
    "token_commit",
    "stop_check_start",
    "stop_check_end",
];
const OPTIONAL: &[(&str, &str, &str)] = &[
    ("stream_update_start", "stream_update_end", "stream_update"),
    ("text_lock_start", "text_lock_end", "text_lock"),
    ("text_decode_start", "text_decode_end", "text_decode"),
    ("stream_send_start", "stream_send_end", "stream_send"),
    ("stream_yield_start", "stream_yield_resumed", "stream_yield"),
    (
        "request_complete_start",
        "request_complete_end",
        "request_complete",
    ),
];
type Stages<'a> = BTreeMap<&'a str, Vec<u64>>;

pub(super) struct CoverageIndex<'a> {
    owners: BTreeMap<(CycleWaveKey, u32), Stages<'a>>,
    failed: BTreeSet<CycleWaveKey>,
    unknown_outcomes: BTreeMap<CycleWaveKey, BTreeSet<String>>,
    unknown_owners: BTreeSet<(CycleWaveKey, u32)>,
    successful_sends: BTreeSet<(CycleWaveKey, u32)>,
}

impl<'a> CoverageIndex<'a> {
    pub(super) fn new(events: &[&'a CycleEvent], waves: &BTreeMap<CycleWaveKey, Wave<'a>>) -> Self {
        let mut submissions = BTreeMap::new();
        let mut calls = BTreeMap::<CallKey, BTreeSet<CycleWaveKey>>::new();
        let mut iterations = BTreeMap::<(u64, u64), BTreeSet<CycleWaveKey>>::new();
        for (&key, wave) in waves {
            if let Some((slot, _)) = &wave.submission {
                submissions.insert(slot.clone(), key);
            }
            for owner in &wave.witness.owners {
                if let Some(context) = owner.context {
                    calls
                        .entry((context.clock_id, context.iteration, context.call))
                        .or_default()
                        .insert(key);
                    iterations
                        .entry((context.clock_id, context.iteration))
                        .or_default()
                        .insert(key);
                }
            }
        }
        let mut result = Self {
            owners: BTreeMap::new(),
            failed: BTreeSet::new(),
            unknown_outcomes: BTreeMap::new(),
            unknown_owners: BTreeSet::new(),
            successful_sends: BTreeSet::new(),
        };
        for event in events {
            let failed = event.stage == "failure_observed"
                || event
                    .outcome
                    .as_deref()
                    .is_some_and(|outcome| outcome != "success");
            let mut affected = BTreeSet::new();
            match &event.scope {
                CycleScope::Participant {
                    clock_id,
                    slot_id,
                    submission_fingerprint,
                    owner,
                } => {
                    let slot = (*clock_id, *slot_id, submission_fingerprint.clone());
                    if let Some(&key) = submissions.get(&slot) {
                        if waves.get(&key).and_then(|wave| {
                            wave.witness.owners.get(owner.canonical_ordinal as usize)
                        }) != Some(owner)
                        {
                            continue;
                        }
                        result
                            .owners
                            .entry((key, owner.canonical_ordinal))
                            .or_default()
                            .entry(&event.stage)
                            .or_default()
                            .push(event.at.nanos);
                        affected.insert(key);
                        if event.stage == "stream_send_end"
                            && event.outcome.as_deref() == Some("success")
                        {
                            result
                                .successful_sends
                                .insert((key, owner.canonical_ordinal));
                        }
                    }
                }
                CycleScope::Call { context } => {
                    if let Some(keys) =
                        calls.get(&(context.clock_id, context.iteration, context.call))
                    {
                        affected.extend(keys.iter().copied());
                    }
                }
                CycleScope::Iteration { iteration } => {
                    if let Some(keys) = iterations.get(&(event.at.clock_id, *iteration)) {
                        affected.extend(keys.iter().copied());
                    }
                }
                _ => {
                    if let Some(key) = event.scope.wave_key() {
                        affected.insert(key);
                    }
                }
            }
            if failed {
                result.failed.extend(affected.iter().copied());
            }
            if event.outcome.is_none() && CURRENT_OUTCOME_STAGES.contains(&event.stage.as_str()) {
                for key in affected {
                    let label = match &event.scope {
                        CycleScope::Participant { owner, .. } => {
                            result.unknown_owners.insert((key, owner.canonical_ordinal));
                            format!("owner.{}.{}_outcome", owner.canonical_ordinal, event.stage)
                        }
                        _ => format!("{}_outcome", event.stage),
                    };
                    result
                        .unknown_outcomes
                        .entry(key)
                        .or_default()
                        .insert(label);
                }
            }
        }
        result
    }

    pub(super) fn failed(&self, key: CycleWaveKey) -> bool {
        self.failed.contains(&key)
    }
    pub(super) fn unknown_outcomes(&self, key: CycleWaveKey) -> impl Iterator<Item = &String> + '_ {
        self.unknown_outcomes
            .get(&key)
            .into_iter()
            .flat_map(|labels| labels.iter())
    }

    pub(super) fn assess(
        &self,
        key: CycleWaveKey,
        wave: &Wave<'_>,
        call_returned: Option<u64>,
        iteration_end: Option<u64>,
        boundaries: &mut BTreeMap<String, Option<u64>>,
        missing: &mut Vec<String>,
    ) -> CycleOwnerCoverage {
        let mut coverage = CycleOwnerCoverage {
            expected: wave.witness.owners.len(),
            optional_branches: OPTIONAL
                .iter()
                .map(|(_, _, name)| ((*name).to_owned(), 0))
                .collect(),
            ..CycleOwnerCoverage::default()
        };
        for owner in &wave.witness.owners {
            let stages = self.owners.get(&(key, owner.canonical_ordinal));
            let before = missing.len();
            let mut previous = call_returned;
            for &stage in REQUIRED {
                let value = unique(stages, stage);
                let label = format!("owner.{}.{stage}", owner.canonical_ordinal);
                if value.is_none() {
                    *coverage
                        .required_missing
                        .entry(stage.to_owned())
                        .or_default() += 1;
                    boundaries.insert(label.clone(), None);
                    missing.push(label);
                } else if let Some(value) = value {
                    if previous.is_some_and(|previous| value < previous)
                        || iteration_end.is_some_and(|end| value > end)
                    {
                        missing.push(format!("reversed:{label}"));
                        coverage.invalid_order += 1;
                    }
                }
                previous = value.or(previous);
            }
            for &(start, end, name) in OPTIONAL {
                let present = stages
                    .is_some_and(|stages| stages.contains_key(start) || stages.contains_key(end));
                if !present {
                    continue;
                }
                *coverage
                    .optional_branches
                    .entry(name.to_owned())
                    .or_default() += 1;
                self.optional_pair(
                    stages,
                    owner.canonical_ordinal,
                    start,
                    end,
                    unique(stages, "stop_check_end"),
                    iteration_end,
                    boundaries,
                    missing,
                    &mut coverage,
                );
            }
            let has_update = stages.is_some_and(|stages| {
                stages.contains_key("stream_update_start")
                    || stages.contains_key("stream_update_end")
            });
            let has_child = OPTIONAL
                .iter()
                .filter(|(_, _, name)| !matches!(*name, "stream_update" | "request_complete"))
                .any(|(start, end, _)| {
                    stages.is_some_and(|stages| {
                        stages.contains_key(start) || stages.contains_key(end)
                    })
                });
            if has_child && !has_update {
                for stage in ["stream_update_start", "stream_update_end"] {
                    let label = format!("owner.{}.{stage}", owner.canonical_ordinal);
                    boundaries.insert(label.clone(), None);
                    missing.push(label);
                }
            }
            if self
                .successful_sends
                .contains(&(key, owner.canonical_ordinal))
            {
                for stage in ["stream_yield_start", "stream_yield_resumed"] {
                    if unique(stages, stage).is_none() {
                        let label = format!("owner.{}.{stage}", owner.canonical_ordinal);
                        boundaries.insert(label.clone(), None);
                        if !missing.contains(&label) {
                            missing.push(label);
                        }
                    }
                }
            }
            let has_yield = stages.is_some_and(|stages| {
                stages.contains_key("stream_yield_start")
                    || stages.contains_key("stream_yield_resumed")
            });
            if has_yield {
                for stage in ["stream_send_start", "stream_send_end"] {
                    if unique(stages, stage).is_none() {
                        let label = format!("owner.{}.{stage}", owner.canonical_ordinal);
                        boundaries.insert(label.clone(), None);
                        if !missing.contains(&label) {
                            missing.push(label);
                        }
                    }
                }
                if !self
                    .successful_sends
                    .contains(&(key, owner.canonical_ordinal))
                {
                    missing.push(format!(
                        "owner.{}.stream_yield_requires_successful_send",
                        owner.canonical_ordinal
                    ));
                }
            }
            if let (Some(update_end), Some(complete_start)) = (
                unique(stages, "stream_update_end"),
                unique(stages, "request_complete_start"),
            ) {
                if complete_start < update_end {
                    missing.push(format!(
                        "reversed:owner.{}.stream_update_end->request_complete_start",
                        owner.canonical_ordinal
                    ));
                    coverage.invalid_order += 1;
                }
            }
            // The actual stream-update helper begins with text-lock/decode even
            // when UTF8 or an empty delta later suppresses the send/yield.
            if has_update {
                for (start, end) in [
                    ("text_lock_start", "text_lock_end"),
                    ("text_decode_start", "text_decode_end"),
                ] {
                    if !stages.is_some_and(|stages| {
                        stages.contains_key(start) || stages.contains_key(end)
                    }) {
                        for stage in [start, end] {
                            let label = format!("owner.{}.{stage}", owner.canonical_ordinal);
                            boundaries.insert(label.clone(), None);
                            missing.push(label);
                        }
                    }
                }
                // Optional nested branches are observed only inside the actual
                // stream-update bracket. They are never filled in when absent.
                for &(start, end, name) in OPTIONAL {
                    if matches!(name, "stream_update" | "request_complete") {
                        continue;
                    }
                    if let (Some(start_at), Some(end_at), Some(outer_start), Some(outer_end)) = (
                        unique(stages, start),
                        unique(stages, end),
                        unique(stages, "stream_update_start"),
                        unique(stages, "stream_update_end"),
                    ) {
                        if start_at < outer_start || end_at > outer_end {
                            missing.push(format!(
                                "outside_stream_update:owner.{}.{name}",
                                owner.canonical_ordinal
                            ));
                            coverage.invalid_order += 1;
                        }
                    }
                }
            }
            for (left, right) in [
                ("text_lock_end", "text_decode_start"),
                ("text_decode_end", "stream_send_start"),
                ("stream_send_end", "stream_yield_start"),
            ] {
                if let (Some(left_at), Some(right_at)) =
                    (unique(stages, left), unique(stages, right))
                {
                    if right_at < left_at {
                        missing.push(format!(
                            "reversed:owner.{}.{left}->{right}",
                            owner.canonical_ordinal
                        ));
                        coverage.invalid_order += 1;
                    }
                }
            }
            if missing.len() == before
                && !self
                    .unknown_owners
                    .contains(&(key, owner.canonical_ordinal))
            {
                coverage.complete += 1;
            }
        }
        coverage
    }

    #[allow(clippy::too_many_arguments)]
    fn optional_pair(
        &self,
        stages: Option<&Stages<'_>>,
        ordinal: u32,
        start: &str,
        end: &str,
        lower: Option<u64>,
        upper: Option<u64>,
        boundaries: &mut BTreeMap<String, Option<u64>>,
        missing: &mut Vec<String>,
        coverage: &mut CycleOwnerCoverage,
    ) {
        let start_at = unique(stages, start);
        let end_at = unique(stages, end);
        for (stage, value) in [(start, start_at), (end, end_at)] {
            if value.is_none() {
                let label = format!("owner.{ordinal}.{stage}");
                boundaries.insert(label.clone(), None);
                missing.push(label);
            }
        }
        if let (Some(start_at), Some(end_at)) = (start_at, end_at) {
            if end_at < start_at
                || lower.is_some_and(|lower| start_at < lower)
                || upper.is_some_and(|upper| end_at > upper)
            {
                missing.push(format!("reversed:owner.{ordinal}.{start}->{end}"));
                coverage.invalid_order += 1;
            }
        }
    }
}

fn unique(stages: Option<&Stages<'_>>, stage: &str) -> Option<u64> {
    match stages?.get(stage)?.as_slice() {
        [at] => Some(*at),
        _ => None,
    }
}
