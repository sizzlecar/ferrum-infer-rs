//! Bounded paired-run analysis with explicit workload and metric contracts.
//! A caller must freeze its inferential method before acceptance measurements.
//! Primary clusters are complete paired server runs of the identical ordered
//! workload. Episode resampling is explicitly sensitivity-only.
mod types;
pub use types::*;
#[cfg(test)]
mod tests;

fn add(a: usize, b: usize) -> Result<usize, Issue> {
    a.checked_add(b).ok_or(Issue::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, Issue> {
    a.checked_mul(b).ok_or(Issue::Overflow)
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}
fn valid_fraction(q: Fraction) -> bool {
    q.denominator > 0 && q.numerator <= q.denominator
}

/// Same nearest-rank definition as the external Rust client, including ties.
/// Callers must supply sorted finite values. Empty is never filled with zero.
fn rank(sorted: &[f64], q: Fraction) -> Result<f64, Issue> {
    if sorted.is_empty() {
        return Err(Issue::UndefinedMetric);
    }
    if !valid_fraction(q) {
        return Err(Issue::InvalidSpec);
    }
    let n = mul(sorted.len(), q.numerator as usize)?;
    let d = q.denominator as usize;
    let count = add(n / d, usize::from(n % d != 0))?;
    Ok(sorted[count.saturating_sub(1).min(sorted.len() - 1)])
}

/// A failure-only proof for partial observations. At rank k of at most N
/// eligible samples, more than N-k observed strict exceedances proves failure.
/// N may be an upper bound (for example, missing outcomes might be successful).
/// This does not produce a quantile or impute the unobserved values.
fn partial_quantile_fails(
    observed: &[f64],
    population_upper_bound: Option<usize>,
    q: Fraction,
    limit: f64,
) -> Result<bool, Issue> {
    let Some(population) = population_upper_bound else {
        return Ok(false);
    };
    if population == 0 || observed.len() > population {
        return Ok(false);
    }
    if !valid_fraction(q) || q.numerator == 0 {
        return Err(Issue::InvalidSpec);
    }
    let scaled = mul(population, q.numerator as usize)?;
    let denominator = q.denominator as usize;
    let k = add(scaled / denominator, usize::from(scaled % denominator != 0))?;
    Ok(observed.iter().filter(|&&value| value > limit).count() > population - k)
}
fn percentiles(sorted: &[f64]) -> Option<Percentiles> {
    Some(Percentiles {
        p50: rank(
            sorted,
            Fraction {
                numerator: 50,
                denominator: 100,
            },
        )
        .ok()?,
        p95: rank(
            sorted,
            Fraction {
                numerator: 95,
                denominator: 100,
            },
        )
        .ok()?,
        p99: rank(
            sorted,
            Fraction {
                numerator: 99,
                denominator: 100,
            },
        )
        .ok()?,
    })
}
fn first_issue(slot: &mut Option<Issue>, issue: Issue) {
    if slot.is_none() {
        *slot = Some(issue);
    }
}
fn combine_gates(values: impl IntoIterator<Item = Gate>) -> Gate {
    let mut unresolved = None;
    for value in values {
        match value {
            Gate::Fail => return Gate::Fail,
            Gate::Unresolved(issue) => first_issue(&mut unresolved, issue),
            Gate::Pass => {}
        }
    }
    unresolved.map_or(Gate::Pass, Gate::Unresolved)
}
fn episode_ids(workload: &Workload) -> Vec<u32> {
    let mut ids = Vec::with_capacity(workload.requests.len());
    for request in &workload.requests {
        if ids.last() != Some(&request.episode) {
            ids.push(request.episode);
        }
    }
    ids
}

fn validate(workload: &Workload, pairs: &[RunPair], spec: Spec) -> Result<usize, Issue> {
    let a = spec.absolute;
    let t = spec.tolerances;
    let b = spec.bootstrap;
    if !valid_fraction(a.quantile)
        || a.quantile.numerator == 0
        || ![a.ttft_ms, a.tpot_ms, a.visible_itl_ms]
            .into_iter()
            .all(positive)
        || a.request_deadline.is_zero()
        || !valid_fraction(b.confidence)
        || b.confidence.numerator == 0
        || b.confidence.numerator == b.confidence.denominator
        || b.replicates == 0
        || b.max_rng_draws == 0
        || ![
            t.feasible_min_throughput_ratio,
            t.feasible_max_latency_ratio,
            t.rescue_min_throughput_ratio,
        ]
        .into_iter()
        .all(positive)
        || t.feasible_min_throughput_ratio > 1.0
        || t.rescue_min_throughput_ratio > t.feasible_min_throughput_ratio
        || t.feasible_max_latency_ratio < 1.0
    {
        return Err(Issue::InvalidSpec);
    }
    if pairs.is_empty() {
        return Err(Issue::InsufficientRunClusters);
    }
    if pairs.len() != workload.repeat_ids.len() || workload.requests.is_empty() {
        return Err(Issue::PairMismatch);
    }
    if pairs.len() > spec.budget.max_runs
        || workload.requests.len() > spec.budget.max_requests
        || b.replicates > spec.budget.max_replicates
    {
        return Err(Issue::Budget);
    }
    let validation_scratch = add(
        mul(
            pairs.len(),
            std::mem::size_of::<u64>() + 2 * std::mem::size_of::<Digest>(),
        )?,
        mul(workload.requests.len(), std::mem::size_of::<u32>())?,
    )?;
    if validation_scratch > spec.budget.max_scratch_bytes {
        return Err(Issue::Budget);
    }
    let mut seen_repeats = Vec::with_capacity(pairs.len());
    let mut seen_servers = Vec::with_capacity(mul(pairs.len(), 2)?);
    let mut requests = 0;
    let mut events = 0;
    let mut max_run_points = 0;
    let ids = episode_ids(workload);
    // Episodes must be contiguous blocks in the declared order, not regrouped
    // after seeing timings. Their sizes need not be equal or a fixed matrix.
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            return Err(Issue::WorkloadMismatch);
        }
    }
    for (index, key) in workload.requests.iter().enumerate() {
        if key.index != index
            || key.output_cap == 0
            || (index > 0 && key.scheduled_ms < workload.requests[index - 1].scheduled_ms)
        {
            return Err(Issue::WorkloadMismatch);
        }
    }
    for (index, pair) in pairs.iter().enumerate() {
        let repeat = workload.repeat_ids[index];
        if seen_repeats.contains(&repeat) {
            return Err(Issue::PairMismatch);
        }
        seen_repeats.push(repeat);
        for run in [&pair.baseline, &pair.candidate] {
            if run.repeat_id != repeat || run.cell != workload.cell || run.inputs != workload.inputs
            {
                return Err(Issue::PairMismatch);
            }
            if seen_servers.contains(&run.server_incarnation) {
                return Err(Issue::DuplicateServerIncarnation);
            }
            seen_servers.push(run.server_incarnation);
            if run.requests.len() != workload.requests.len()
                || run
                    .requests
                    .iter()
                    .zip(&workload.requests)
                    .any(|(r, key)| &r.key != key)
            {
                return Err(Issue::WorkloadMismatch);
            }
            requests = add(requests, run.requests.len())?;
            let mut run_events = 0;
            for request in &run.requests {
                run_events = add(run_events, request.visible_gaps_ms.len())?;
            }
            events = add(events, run_events)?;
            max_run_points = max_run_points.max(add(mul(run.requests.len(), 2)?, run_events)?);
        }
    }
    let budget = spec.budget;
    if pairs.len() > budget.max_runs
        || requests > budget.max_requests
        || events > budget.max_events
        || b.replicates > budget.max_replicates
    {
        return Err(Issue::Budget);
    }
    let episode_factor = if b.episode_sensitivity { ids.len() } else { 1 };
    let point_bound = mul(max_run_points, episode_factor)?;
    let primary_visits = mul(mul(b.replicates, pairs.len())?, 4)?;
    let sensitivity_visits = if b.episode_sensitivity {
        mul(mul(mul(b.replicates, pairs.len())?, point_bound)?, 2)?
    } else {
        0
    };
    if add(primary_visits, sensitivity_visits)? > budget.max_sample_visits {
        return Err(Issue::Budget);
    }
    // Samples + one sorted CI column, three latency buffers, paired log effects,
    // episode draws, and validation identity vectors. Returned report excluded.
    let scratch = add(
        add(
            mul(b.replicates, 5 * std::mem::size_of::<f64>())?,
            mul(point_bound, 3 * std::mem::size_of::<f64>())?,
        )?,
        add(
            mul(
                pairs.len(),
                4 * std::mem::size_of::<f64>()
                    + 2 * std::mem::size_of::<Digest>()
                    + std::mem::size_of::<u64>(),
            )?,
            mul(
                add(workload.requests.len(), ids.len())?,
                std::mem::size_of::<u32>(),
            )?,
        )?,
    )?;
    if scratch > budget.max_scratch_bytes {
        return Err(Issue::Budget);
    }
    Ok(point_bound)
}

fn valid_usage(request: &Request) -> bool {
    request.usage.is_some_and(|u| {
        u.completion <= request.key.output_cap
            && u.prompt.checked_add(u.completion) == Some(u.total)
    })
}
fn append_timing(request: &Request, values: &mut [Vec<f64>; 3]) -> Result<(), Issue> {
    let first = request
        .ttft_from_scheduled_ms
        .filter(|x| nonnegative(*x))
        .ok_or(Issue::InvalidTiming)?;
    if request.visible_text_events == Some(0)
        || request.terminal_ms.is_some_and(|terminal| {
            !nonnegative(terminal) || terminal < request.key.scheduled_ms as f64 + first
        })
    {
        return Err(Issue::InvalidTiming);
    }
    values[0].push(first);
    // Usage availability does not determine whether TTFT or SSE gaps exist.
    let usage = request.usage.ok_or(Issue::InvalidUsage)?;
    if !valid_usage(request) {
        return Err(Issue::InvalidUsage);
    }
    if usage.completion < 2 {
        return Ok(());
    }
    let span = request
        .visible_span_ms
        .filter(|x| nonnegative(*x))
        .ok_or(Issue::InvalidTiming)?;
    if request
        .terminal_ms
        .is_some_and(|terminal| terminal < request.key.scheduled_ms as f64 + first + span)
    {
        return Err(Issue::InvalidTiming);
    }
    values[1].push(span / (usage.completion - 1) as f64);
    Ok(())
}

struct GapEvidence {
    /// Count is usable independently of missing gap values, but a contradictory
    /// count or missing stream outcome cannot establish the full denominator.
    population: Option<usize>,
    issue: Option<Issue>,
}

/// Independent of usage, finish, TTFT, and terminal availability. Retain each
/// independently verified finite gap even when another field is missing/bad.
/// Qualification of the whole stream remains separate from descriptive data.
fn append_visible_gaps(request: &Request, gaps: &mut Vec<f64>) -> GapEvidence {
    let expected = request.visible_text_events.map(|n| n.saturating_sub(1));
    let population = expected
        .filter(|&n| request.outcome != Outcome::Missing && request.visible_gaps_ms.len() <= n);
    let mut issue =
        (expected != Some(request.visible_gaps_ms.len())).then_some(Issue::InvalidTiming);
    if request.outcome == Outcome::Missing {
        first_issue(&mut issue, Issue::MissingTerminal);
    }
    let mut sum = 0.0_f64;
    for &gap in &request.visible_gaps_ms {
        if nonnegative(gap) {
            gaps.push(gap);
            sum += gap;
            if !sum.is_finite() {
                first_issue(&mut issue, Issue::Numerical);
            }
        } else {
            first_issue(&mut issue, Issue::InvalidTiming);
        }
    }
    if let Some(span) = request.visible_span_ms {
        if !nonnegative(span) {
            first_issue(&mut issue, Issue::InvalidTiming);
        } else if issue.is_none() {
            // Only floating-point summation error; not a workload tolerance.
            let rounding = f64::EPSILON
                * (request.visible_gaps_ms.len() as f64 + 2.0)
                * span.abs().max(sum.abs()).max(1.0)
                * 4.0;
            if (sum - span).abs() > rounding {
                first_issue(&mut issue, Issue::InvalidTiming);
            }
        }
    }
    GapEvidence { population, issue }
}

fn summarize(run: &Run, spec: Spec, point_bound: usize) -> Result<RunSummary, Issue> {
    let mut result = RunSummary {
        repeat_id: run.repeat_id,
        client_exit: run.client_exit,
        arrivals: run.requests.len(),
        successes: 0,
        failures: 0,
        missing: 0,
        protocol_violations: 0,
        deadline_violations: 0,
        tpot_undefined: 0,
        tpot_unknown: 0,
        visible_intervals: 0,
        itl_unknown_requests: 0,
        successful_usage_tokens: Some(0),
        finite_drain_seconds: None,
        throughput_tps: None,
        ttft: None,
        tpot: None,
        visible_itl: None,
        target_quantiles: [None; 3],
        latency_gates: [Gate::Unresolved(Issue::UndefinedMetric); 3],
        absolute: Gate::Pass,
        inference_issue: None,
    };
    let mut values: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(point_bound));
    let mut known_failure = run.client_exit.is_some_and(|code| code != 0);
    if known_failure {
        first_issue(&mut result.inference_issue, Issue::FailedClient);
    }
    if run.client_exit.is_none() {
        first_issue(&mut result.inference_issue, Issue::MissingClientExit);
    }
    let mut timing_issue = None;
    let mut terminal_complete = true;
    let mut itl_population = Some(0_usize);
    let mut last = 0.0_f64;
    let first = run.requests[0].key.scheduled_ms as f64;
    for request in &run.requests {
        let gaps = append_visible_gaps(request, &mut values[2]);
        itl_population = match (itl_population, gaps.population) {
            (Some(total), Some(count)) => Some(add(total, count)?),
            _ => None,
        };
        if let Some(issue) = gaps.issue {
            result.itl_unknown_requests += 1;
            first_issue(&mut result.inference_issue, issue);
        }
        match request.terminal_ms {
            Some(t) if nonnegative(t) && t >= request.key.scheduled_ms as f64 => {
                last = last.max(t);
                if t - request.key.scheduled_ms as f64
                    > spec.absolute.request_deadline.as_secs_f64() * 1000.0
                {
                    known_failure = true;
                    result.deadline_violations += 1;
                    first_issue(&mut result.inference_issue, Issue::FailedRequest);
                }
            }
            _ => {
                terminal_complete = false;
                first_issue(&mut result.inference_issue, Issue::MissingTerminal);
            }
        }
        match request.outcome {
            Outcome::Missing => {
                result.missing += 1;
                first_issue(&mut result.inference_issue, Issue::MissingTerminal);
            }
            Outcome::Failed(_) => {
                result.failures += 1;
                known_failure = true;
                first_issue(&mut result.inference_issue, Issue::FailedRequest);
            }
            Outcome::Success => {
                result.successes += 1;
                if !request.done || request.finish.is_none() || !valid_usage(request) {
                    known_failure = true;
                    result.protocol_violations += 1;
                    first_issue(&mut result.inference_issue, Issue::InvalidUsage);
                }
                match request.usage.filter(|_| valid_usage(request)) {
                    Some(u) => {
                        result.successful_usage_tokens = result
                            .successful_usage_tokens
                            .and_then(|sum| sum.checked_add(u.completion));
                        if u.completion < 2 {
                            result.tpot_undefined += 1;
                        }
                    }
                    None => {
                        result.successful_usage_tokens = None;
                    }
                }
                let tpot_before = values[1].len();
                if let Err(issue) = append_timing(request, &mut values) {
                    first_issue(&mut timing_issue, issue);
                }
                // Classify each successful request once: a malformed usage
                // record must not also add a second unknown for timing failure.
                if !valid_usage(request)
                    || (request.usage.is_some_and(|u| u.completion >= 2)
                        && values[1].len() == tpot_before)
                {
                    result.tpot_unknown += 1;
                }
            }
        }
    }
    if terminal_complete && last > first {
        let seconds = (last - first) / 1000.0;
        if positive(seconds) {
            result.finite_drain_seconds = Some(seconds);
            result.throughput_tps = result.successful_usage_tokens.map(|t| t as f64 / seconds);
        }
    }
    if result.throughput_tps.is_none() {
        first_issue(&mut result.inference_issue, Issue::NonpositiveRatio);
    }
    if let Some(issue) = timing_issue {
        first_issue(&mut result.inference_issue, issue);
    }
    for v in &mut values {
        // Equal scalar identity order has no meaning; avoid stable-sort heap
        // scratch outside the checked explicit Vec-capacity bound.
        v.sort_unstable_by(f64::total_cmp);
    }
    result.ttft = percentiles(&values[0]);
    result.tpot = percentiles(&values[1]);
    if result.missing == 0 && values[0].len() == result.successes {
        result.target_quantiles[0] = rank(&values[0], spec.absolute.quantile).ok();
    }
    if result.missing == 0
        && result.tpot_unknown == 0
        && values[1].len() == result.successes - result.tpot_undefined
    {
        result.target_quantiles[1] = rank(&values[1], spec.absolute.quantile).ok();
    }
    result.visible_itl = percentiles(&values[2]);
    result.visible_intervals = values[2].len();
    if result.itl_unknown_requests == 0 {
        result.target_quantiles[2] = rank(&values[2], spec.absolute.quantile).ok();
    }
    let limits = [
        spec.absolute.ttft_ms,
        spec.absolute.tpot_ms,
        spec.absolute.visible_itl_ms,
    ];
    let population_bounds = [
        Some(add(result.successes, result.missing)?),
        Some(add(
            result.successes - result.tpot_undefined,
            result.missing,
        )?),
        itl_population,
    ];
    for metric in 0..3 {
        result.latency_gates[metric] = match result.target_quantiles[metric] {
            Some(value) if value > limits[metric] => Gate::Fail,
            Some(_) => Gate::Pass,
            None if partial_quantile_fails(
                &values[metric],
                population_bounds[metric],
                spec.absolute.quantile,
                limits[metric],
            )? =>
            {
                Gate::Fail
            }
            None => Gate::Unresolved(Issue::UndefinedMetric),
        };
    }
    result.absolute = if known_failure {
        Gate::Fail
    } else {
        combine_gates(
            result
                .latency_gates
                .into_iter()
                .chain(result.inference_issue.map(Gate::Unresolved)),
        )
    };
    Ok(result)
}

struct SplitMix64 {
    state: u64,
    remaining: usize,
}
impl SplitMix64 {
    fn new(config: Bootstrap) -> Self {
        match config.prng {
            Prng::SplitMix64V1 => Self {
                state: config.seed,
                remaining: config.max_rng_draws,
            },
        }
    }
    fn next(&mut self) -> Result<u64, Issue> {
        self.remaining = self.remaining.checked_sub(1).ok_or(Issue::Budget)?;
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        Ok(z ^ (z >> 31))
    }
    fn index(&mut self, length: usize) -> Result<usize, Issue> {
        let bound = length as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next()?;
            if value >= threshold {
                return Ok((value % bound) as usize);
            }
        }
    }
}
fn log_ratio(candidate: f64, baseline: f64) -> Result<f64, Issue> {
    if !positive(candidate) || !positive(baseline) {
        return Err(Issue::NonpositiveRatio);
    }
    let value = candidate.ln() - baseline.ln();
    value.is_finite().then_some(value).ok_or(Issue::Numerical)
}
fn paired_logs(pair: &PairSummary) -> Result<[f64; 4], Issue> {
    if let Some(issue) = pair
        .baseline
        .inference_issue
        .or(pair.candidate.inference_issue)
    {
        return Err(issue);
    }
    let scalar = |r: &RunSummary| -> Result<[f64; 4], Issue> {
        Ok([
            r.throughput_tps.ok_or(Issue::UndefinedMetric)?,
            r.target_quantiles[0].ok_or(Issue::UndefinedMetric)?,
            r.target_quantiles[1].ok_or(Issue::UndefinedMetric)?,
            r.target_quantiles[2].ok_or(Issue::UndefinedMetric)?,
        ])
    };
    let baseline = scalar(&pair.baseline)?;
    let candidate = scalar(&pair.candidate)?;
    let mut result = [0.0; 4];
    for i in 0..4 {
        result[i] = log_ratio(candidate[i], baseline[i])?;
    }
    Ok(result)
}
fn interval(
    log_samples: &mut [f64],
    log_estimate: f64,
    confidence: Fraction,
) -> Result<Interval, Issue> {
    if !log_samples.iter().all(|x| x.is_finite()) {
        return Err(Issue::Numerical);
    }
    log_samples.sort_unstable_by(f64::total_cmp);
    let denominator = confidence
        .denominator
        .checked_mul(2)
        .ok_or(Issue::Overflow)?;
    let low = Fraction {
        numerator: confidence.denominator - confidence.numerator,
        denominator,
    };
    let high = Fraction {
        numerator: confidence
            .denominator
            .checked_add(confidence.numerator)
            .ok_or(Issue::Overflow)?,
        denominator,
    };
    let result = Interval {
        estimate: log_estimate.exp(),
        lower: rank(log_samples, low)?.exp(),
        upper: rank(log_samples, high)?.exp(),
    };
    [result.estimate, result.lower, result.upper]
        .into_iter()
        .all(positive)
        .then_some(result)
        .ok_or(Issue::Numerical)
}
fn bootstrap_primary(logs: &[[f64; 4]], config: Bootstrap) -> Result<Ratios, Issue> {
    let mut rng = SplitMix64::new(config);
    let mut samples = Vec::with_capacity(config.replicates);
    for _ in 0..config.replicates {
        let mut sample = [0.0; 4];
        for _ in 0..logs.len() {
            // One shared run-pair draw supplies every arm and metric.
            let draw = &logs[rng.index(logs.len())?];
            for metric in 0..4 {
                sample[metric] += draw[metric] / logs.len() as f64;
            }
        }
        samples.push(sample);
    }
    let mut result = [Interval {
        estimate: 0.0,
        lower: 0.0,
        upper: 0.0,
    }; 4];
    for metric in 0..4 {
        let estimate = logs.iter().map(|x| x[metric] / logs.len() as f64).sum();
        let mut column: Vec<_> = samples.iter().map(|x| x[metric]).collect();
        result[metric] = interval(&mut column, estimate, config.confidence)?;
    }
    Ok(Ratios {
        throughput: result[0],
        ttft: result[1],
        tpot: result[2],
        visible_itl: result[3],
    })
}
fn directional(interval: Interval, limit: f64, lower_bound: bool) -> Gate {
    if lower_bound {
        if interval.lower >= limit {
            Gate::Pass
        } else if interval.upper < limit {
            Gate::Fail
        } else {
            Gate::Unresolved(Issue::IntervalCrossesTolerance)
        }
    } else if interval.upper <= limit {
        Gate::Pass
    } else if interval.lower > limit {
        Gate::Fail
    } else {
        Gate::Unresolved(Issue::IntervalCrossesTolerance)
    }
}
fn primary(pairs: &[PairSummary], spec: Spec) -> Result<Primary, Issue> {
    if pairs.len() < 2 {
        return Err(Issue::InsufficientRunClusters);
    }
    let logs: Vec<_> = pairs.iter().map(paired_logs).collect::<Result<_, _>>()?;
    let feasible = pairs
        .iter()
        .filter(|p| p.baseline.absolute == Gate::Pass)
        .count();
    let infeasible = pairs
        .iter()
        .filter(|p| p.baseline.absolute == Gate::Fail)
        .count();
    let static_class = if feasible == pairs.len() {
        StaticClass::AllFeasible
    } else if infeasible == pairs.len() {
        StaticClass::AllInfeasible
    } else if feasible + infeasible == pairs.len() {
        return Err(Issue::MixedStaticFeasibility);
    } else {
        return Err(Issue::UncertainStaticFeasibility);
    };
    let ratios = bootstrap_primary(&logs, spec.bootstrap)?;
    let feasible_rules =
        static_class == StaticClass::AllFeasible || spec.kind == CandidateKind::Observe;
    let throughput_limit = if feasible_rules {
        spec.tolerances.feasible_min_throughput_ratio
    } else {
        spec.tolerances.rescue_min_throughput_ratio
    };
    let mut gates = vec![directional(ratios.throughput, throughput_limit, true)];
    if feasible_rules {
        gates.extend(
            [ratios.ttft, ratios.tpot, ratios.visible_itl]
                .into_iter()
                .map(|i| directional(i, spec.tolerances.feasible_max_latency_ratio, false)),
        );
    }
    Ok(Primary {
        static_class,
        ratios,
        tolerance_gate: combine_gates(gates),
    })
}

fn episode_quantiles(
    run: &Run,
    draws: &[u32],
    spec: Spec,
    point_bound: usize,
) -> Result<[f64; 3], Issue> {
    let mut values: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(point_bound));
    for episode in draws {
        for request in run.requests.iter().filter(|r| r.key.episode == *episode) {
            // Failures remain in the block; they are not removed to make a CI.
            if request.outcome != Outcome::Success {
                return Err(Issue::FailedRequest);
            }
            append_timing(request, &mut values)?;
            if let Some(issue) = append_visible_gaps(request, &mut values[2]).issue {
                return Err(issue);
            }
        }
    }
    let mut result = [0.0; 3];
    for i in 0..3 {
        values[i].sort_unstable_by(f64::total_cmp);
        result[i] = rank(&values[i], spec.absolute.quantile)?;
    }
    Ok(result)
}
fn sensitivity(
    workload: &Workload,
    raw: &[RunPair],
    summaries: &[PairSummary],
    spec: Spec,
    point_bound: usize,
) -> Result<Sensitivity, Issue> {
    if raw.len() < 2 {
        return Err(Issue::InsufficientRunClusters);
    }
    let logs: Vec<_> = summaries
        .iter()
        .map(paired_logs)
        .collect::<Result<_, _>>()?;
    let ids = episode_ids(workload);
    let mut rng = SplitMix64::new(spec.bootstrap);
    let mut samples = Vec::with_capacity(spec.bootstrap.replicates);
    let mut draws = Vec::with_capacity(ids.len());
    for _ in 0..spec.bootstrap.replicates {
        let mut sample = [0.0; 3];
        for _ in 0..raw.len() {
            let pair = &raw[rng.index(raw.len())?];
            draws.clear();
            for _ in 0..ids.len() {
                draws.push(ids[rng.index(ids.len())?]);
            }
            // The same full episode draw enters both arms. Every request's
            // entire gap vector stays together; event counts may differ.
            let baseline = episode_quantiles(&pair.baseline, &draws, spec, point_bound)?;
            let candidate = episode_quantiles(&pair.candidate, &draws, spec, point_bound)?;
            for metric in 0..3 {
                sample[metric] +=
                    log_ratio(candidate[metric], baseline[metric])? / raw.len() as f64;
            }
        }
        samples.push(sample);
    }
    let mut intervals = [Interval {
        estimate: 0.0,
        lower: 0.0,
        upper: 0.0,
    }; 3];
    for metric in 0..3 {
        let estimate = logs.iter().map(|x| x[metric + 1] / logs.len() as f64).sum();
        let mut column: Vec<_> = samples.iter().map(|x| x[metric]).collect();
        intervals[metric] = interval(&mut column, estimate, spec.bootstrap.confidence)?;
    }
    Ok(Sensitivity {
        latency_ratios: intervals,
        interpretation: "Sensitivity only: episode exchangeability is an additional unproved assumption; never replaces the ordered-run primary or absolute per-run gates.",
    })
}

/// No IO. An evidence reader must map every declared missing arrival to
/// Outcome::Missing and bind original evidence hashes; it must not drop runs.
/// Structural invalidity returns Err. Validly shaped observed failures remain
/// in Report with absolute Fail even when paired inference is unavailable.
pub fn analyze(workload: &Workload, pairs: &[RunPair], spec: Spec) -> Result<Report, Issue> {
    let point_bound = validate(workload, pairs, spec)?;
    let summaries = pairs
        .iter()
        .map(|pair| {
            Ok(PairSummary {
                baseline: summarize(&pair.baseline, spec, point_bound)?,
                candidate: summarize(&pair.candidate, spec, point_bound)?,
            })
        })
        .collect::<Result<Vec<_>, Issue>>()?;
    let candidate_absolute = combine_gates(summaries.iter().map(|pair| pair.candidate.absolute));
    let primary = primary(&summaries, spec);
    let development_gate = combine_gates([
        candidate_absolute,
        primary.as_ref().map_or_else(
            |issue| Gate::Unresolved(*issue),
            |value| value.tolerance_gate,
        ),
    ]);
    let episode_sensitivity = spec
        .bootstrap
        .episode_sensitivity
        .then(|| sensitivity(workload, pairs, &summaries, spec, point_bound));
    Ok(Report {
        pairs: summaries, candidate_absolute, primary, episode_sensitivity,
        development_gate,
        method_status: "DRAFT_METHOD_NOT_FORMAL_ACCEPTANCE",
        interpretation: "Conditional on the complete fixed ordered workload and sampled server runs. Four runs remain four clusters; bootstrap repetitions do not add information. Marginal intervals, not familywise coverage; no population completion probability=1 claim.",
        bootstrap: spec.bootstrap,
    })
}
