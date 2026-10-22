//! One fixed external-input admission stress case, not a historical replay.
//! Actual rendered lengths come from successful protocol records; queue state,
//! cost distributions, cap endpoints and resource ownership are conditional.

use super::*;
use ferrum_execution::{
    model_executor::{
        PersistentCapacityMember, PersistentCapacityQuery, PersistentCapacityState,
        PersistentCohortCapacitySnapshot,
    },
    token_execution::ExecutorSequenceIssuer,
};
use ferrum_interfaces::request_observation::*;
use ferrum_types::{RequestIngressAnchor, RequestIngressBoundary};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs::File, io::Read, path::PathBuf, result::Result};

const MEMBERS: usize = 48;
const SLOTS: usize = 16;
const CAP: usize = 64;
const FILE_BYTES: usize = 8 * 1024 * 1024;
const CLIENT_ENV: &str = "FERRUM_DIAGNOSTIC_BURST_CLIENT";
const RECORDS_ENV: &str = "FERRUM_DIAGNOSTIC_SUCCESS_RECORDS";

#[derive(Deserialize)]
struct Client {
    requests: Vec<ClientRequest>,
}

#[derive(Deserialize)]
struct ClientRequest {
    prompt: String,
    scenario: String,
    source: Value,
    offset_ms: u64,
    max_tokens: usize,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct Record {
    index: usize,
    scenario: String,
    source: Value,
    scheduled_ms: f64,
    requested_max_tokens: usize,
    success: bool,
    timed_out: bool,
    error: Option<Value>,
    done: bool,
    prompt_tokens: usize,
    usage: Usage,
}

fn read_external(variable: &str) -> Result<(PathBuf, Vec<u8>), String> {
    let path = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or_else(|| format!("set {variable} to its original external input"))?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|e| e.to_string())?
        .take((FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > FILE_BYTES {
        return Err(format!("{variable} exceeds {FILE_BYTES} bytes"));
    }
    Ok((path, bytes))
}

fn inputs() -> Result<(Client, Vec<usize>, Value), String> {
    let (client_path, client_bytes) = read_external(CLIENT_ENV)?;
    let (records_path, records_bytes) = read_external(RECORDS_ENV)?;
    let client: Client = serde_json::from_slice(&client_bytes).map_err(|e| e.to_string())?;
    if client.requests.len() != MEMBERS {
        return Err("this diagnostic declares exactly 48 original burst requests".into());
    }
    let text = std::str::from_utf8(&records_bytes).map_err(|e| e.to_string())?;
    let mut prompts = vec![None; MEMBERS];
    for line in text.lines() {
        let record: Record = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let Some(request) = client.requests.get(record.index) else {
            return Err("successful record has an out-of-range index".into());
        };
        if prompts[record.index].is_some()
            || record.scenario != request.scenario
            || record.source != request.source
            || record.scheduled_ms != request.offset_ms as f64
            || record.requested_max_tokens != request.max_tokens
            || request.max_tokens != CAP
            || !record.success
            || record.timed_out
            || record.error.is_some()
            || !record.done
            || record.prompt_tokens == 0
            || record.prompt_tokens != record.usage.prompt_tokens
            || record.usage.completion_tokens > CAP
            || record
                .usage
                .prompt_tokens
                .checked_add(record.usage.completion_tokens)
                != Some(record.usage.total_tokens)
            || record
                .prompt_tokens
                .checked_add(CAP)
                .is_none_or(|end| end > 2048)
        {
            return Err(format!(
                "invalid or mismatched original record {}",
                record.index
            ));
        }
        prompts[record.index] = Some(record.prompt_tokens);
    }
    if client
        .requests
        .iter()
        .enumerate()
        .any(|(index, request)| request.offset_ms != index as u64 * 40 || request.prompt.is_empty())
    {
        return Err("original 40ms burst arrival policy is not present".into());
    }
    let prompts = prompts
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or("missing original successful record")?;
    let evidence = json!({
        "client_path": client_path, "client_bytes": client_bytes.len(),
        "successful_records_path": records_path, "successful_records_bytes": records_bytes.len(),
        "binding": "all 48 indices, full source metadata, scenario, scheduled offset and cap matched; rendered lengths cross-checked against successful usage, not raw selection-tokenizer lengths",
        "geometry": client.requests.iter().zip(&prompts).enumerate().map(|(index, (r, prompt))| json!({
            "index": index, "source": r.source, "offset_ms": r.offset_ms,
            "rendered_prompt_tokens": prompt, "output_cap": r.max_tokens,
        })).collect::<Vec<_>>(),
    });
    Ok((client, prompts, evidence))
}

struct ConditionalCosts {
    costs: CheapKnownCosts,
    stopping: AtomicUsize,
    completion: AtomicUsize,
}

impl CostSnapshot for ConditionalCosts {
    fn revision(&self) -> u64 {
        self.costs.revision()
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        self.costs.query_limits()
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.costs.predict(query, now)
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.costs.cost_scenarios(query, now, count)
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.stopping.fetch_add(1, Ordering::Relaxed);
        StoppingForecast {
            stops: vec![StoppingMass {
                generated_tokens: query.cap,
                mass: 1.0,
            }],
            known_survival_mass: 0.0,
            unresolved_mass: 0.0,
            tpot_undefined_mass: f64::from(query.cap < 2),
        }
    }
    fn completion_forecast(&self, _: &CompletionQuery<'_>) -> CompletionForecast {
        self.completion.fetch_add(1, Ordering::Relaxed);
        CompletionForecast {
            succeeded_mass: 1.0,
            failed_mass: 0.0,
            unresolved_mass: 0.0,
            samples: 1,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }
}

#[derive(Clone, Copy)]
enum CacheAccounting {
    Disabled,
    Enabled,
}

impl CacheAccounting {
    fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }

    fn scope(self) -> &'static str {
        match self {
            Self::Disabled => "Cache-accounting capture disabled: no TLS capture scope or populated accounting counters for this gate. This remains a cfg(test) diagnostic with existing test-only cache hit counters, not a production timing or GPU measurement. A null report means not collected, not zero events.",
            Self::Enabled => "Test-only TLS opt-in for this one gate; fixed two reports plus explicit overflow. The cache-teardown report is the last retained-allocation ledger (trace may already be freed by local drop order); first/last byte-rejection snapshots are current ledger values. Neither is allocator RSS or a simultaneous peak. Counts add overhead, so this instrumented call is mechanism evidence, not a timing baseline. No extra prediction, comparison, request or cache traversal.",
        }
    }
}

fn run(accounting: CacheAccounting) -> Result<Value, String> {
    let (client, prompts, evidence) = inputs()?;
    let mut slo: SchedulerSloConfig = "ttft:15684.62,tpot:110.56,itl:196.55".parse().unwrap();
    slo.strategy = SloStrategy::Joint;
    slo.planning.max_elapsed_us = 10_000;
    slo.planning.max_queries = 4096;
    slo.planning.max_scenarios = 32;
    slo.validate().map_err(|e| e.to_string())?;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: SLOTS,
        active_decode_prefill_chunk: Some(256),
        prefill_step_chunk: Some(128),
        slo: Some(slo),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let mut ids = Vec::with_capacity(MEMBERS);
    for (request, prompt) in client.requests.iter().zip(&prompts) {
        let mut input = InferenceRequest::new(&request.prompt, "conditional-planning-load")
            .with_ingress(ingress)
            .with_metadata(PROMPT_TOKENS_METADATA_KEY, json!(prompt));
        input.sampling_params.max_tokens = CAP;
        input.stream = true;
        ids.push(scheduler.submit_request(input).map_err(|e| e.to_string())?);
    }
    // Declare the first 16 owners resident without performing device work.
    // Remaining 32 remain in the actual waiting queue and original order.
    let mut scan = scheduler
        .begin_token_admission(SLOTS, None)
        .map_err(|e| e.to_string())?;
    let mut admitted = 0;
    while let Some(candidate) = scan.next(&mut |_| {}, None).map_err(|e| e.to_string())? {
        let id = candidate.request().id.clone();
        if ids.get(admitted) != Some(&id) {
            return Err("initial admission reordered requests".into());
        }
        match scan.apply(
            candidate,
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id: id }),
        ) {
            TokenAdmissionApply::Applied { accepted: Some(_) } => admitted += 1,
            _ => return Err("synthetic owner admission failed".into()),
        }
    }
    scan.finish().map_err(|e| e.to_string())?;
    if admitted != SLOTS
        || scheduler.active_count() != SLOTS
        || scheduler.waiting_count() != MEMBERS - SLOTS
    {
        return Err("declared 16 resident / 32 waiting cohort was not constructed".into());
    }
    let captured = Instant::now();
    let source = Arc::new(ConditionalCosts {
        costs: CheapKnownCosts {
            observed_at: captured,
            calls: AtomicUsize::new(0),
        },
        stopping: AtomicUsize::new(0),
        completion: AtomicUsize::new(0),
    });
    let rules = Arc::new(
        EffectiveStoppingRules::new(
            b"conditional-cap-endpoint".to_vec(),
            StoppingMechanism::CapOnly,
            CAP,
        )
        .ok_or("invalid conditional stopping rules")?,
    );
    let cut = client.requests.last().ok_or("empty client")?.offset_ms;
    let mut round = TimePlanningRound::new(
        source.clone(),
        slo,
        PrefillShapePolicy::Identity,
        CostExecution::SequentialPrefillThenDecode,
    );
    if !round.set_request_facts(ids.iter().enumerate().map(|(index, id)| {
        (
            id.clone(),
            RequestStoppingObservation {
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: index as u64 + 1,
                },
                admitted_at: captured
                    .checked_sub(Duration::from_millis(
                        cut - client.requests[index].offset_ms,
                    ))
                    .unwrap(),
                observed_at: captured,
                revision: 1,
                rules: rules.clone(),
                cap: CAP,
                generated_tokens: 0,
                verified_survival_through: 0,
                terminal: None,
                completion: None,
            },
        )
    })) {
        return Err("conditional request facts rejected".into());
    }
    if !round.set_request_clocks(ids.iter().enumerate().map(|(index, id)| {
        (
            id.clone(),
            RequestClockFacts {
                captured_at: captured,
                age: Duration::from_millis(cut - client.requests[index].offset_ms),
                first_text: None,
                last_text: None,
                max_text_interval: None,
                nonempty_text_events: 0,
                violating_text_intervals: 0,
                itl_threshold: Some(Duration::from_secs_f64(slo.itl_ms / 1000.0)),
                generated_tokens: 0,
                streaming: true,
            },
        )
    })) {
        return Err("conditional clocks rejected".into());
    }
    let issuer = ExecutorSequenceIssuer::new();
    let members = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            Ok(PersistentCapacityMember {
                request_id: id.clone(),
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: index as u64 + 1,
                },
                state: if index < SLOTS {
                    PersistentCapacityState::Resident {
                        sequence: issuer.issue_sequence().map_err(|e| e.to_string())?,
                    }
                } else {
                    PersistentCapacityState::Waiting
                },
                through_tokens: prompts[index] + CAP - 1,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let query = PersistentCapacityQuery {
        members: &members,
        max_live_sequences: SLOTS,
        max_wave_sequences: SLOTS,
        inspection_budget: Default::default(),
    };
    let proof = PersistentCohortCapacitySnapshot::from_verified_query(&query)
        .map_err(|e| format!("{e:?}"))?;
    round.set_persistent_capacity(&query, Some(proof));
    if round.persistent_capacity().is_none() {
        return Err("complete conditional capacity binding rejected".into());
    }
    let hint = BatchHint {
        max_batch_size: SLOTS,
        max_tokens: 512,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    };
    let mut observed = scheduler.observe_planning_round(Some(round));
    let started = Instant::now();
    let cache_accounting = match accounting {
        CacheAccounting::Disabled => {
            scheduler.prepare_joint_admission_gate(&hint, observed.as_mut().unwrap(), started);
            None
        }
        CacheAccounting::Enabled => {
            let ((), capture) = SloPolicy::capture_cache_accounting(|| {
                scheduler.prepare_joint_admission_gate(&hint, observed.as_mut().unwrap(), started);
            });
            Some(capture)
        }
    };
    let gate_elapsed = started.elapsed();
    let round = observed.finish().ok_or("lost planning round")?;
    let with_health_merge = started.elapsed();
    let calls = source.costs.calls.load(Ordering::Relaxed);
    if calls as u64 != round.statistics.joint_evaluation.admission.provider_calls {
        return Err("independent provider count differs from evaluator ledger".into());
    }
    Ok(json!({
        "scope": "one actual admission-gate planning call, original 48 input shapes, conditional all-prefill cohort and cheap Known costs; not historical replay, learned support, GPU capacity or performance",
        "inputs": evidence, "policy": slo,
        "premises": {"resident_prefillers": SLOTS, "waiting": MEMBERS-SLOTS, "initial_prefill_progress":0, "initial_generated":0,
            "synthetic_cut_ms":cut, "ages":"last original offset minus each original offset; no sleep, no actual service trace replay",
            "endpoints":"conditional cap64, natural EOS not predicted", "completion":"conditional [1,1], synthetic metadata",
            "costs":"same CheapKnownCosts as existing 16-member CPU profile; 1ms central, 1000+rank microseconds, shared residual ranks",
            "capacity":"synthetic original-owner complete 48-member envelope, only 16 concurrent slots; no physical allocator or device work",
            "entrypoint":"public submit, original token-admission receipts for first16, public prepare_joint_admission_gate; current automatic actions and same-budget fresh recheck retained"},
        "cache_accounting_enabled": accounting.enabled(),
        "cache_accounting": cache_accounting,
        "cache_accounting_scope": accounting.scope(),
        "gate_elapsed":gate_elapsed, "gate_and_health_merge_elapsed":with_health_merge,
        "original_provider_calls":calls,
        "conditional_stopping_calls":source.stopping.load(Ordering::Relaxed),
        "conditional_completion_calls":source.completion.load(Ordering::Relaxed),
        "charged_queries":round.query_count, "charged_elapsed":round.prediction_elapsed,
        "scheduler":scheduler.slo_snapshot(),
        "actual_waiting_after":scheduler.waiting_count(), "actual_residents_after":scheduler.active_count(),
        "runtime_execution_performed":false,
        "interpretation":"All traversal failures and decision classes are retained. Incomplete evaluation proves neither feasible nor infeasible. No completion count, choice or runtime is a pass gate. Admission-only observation does not dispatch or execute any cap commitment."
    }))
}

fn print_report(accounting: CacheAccounting) {
    match run(accounting) {
        Ok(report) => println!(
            "ADMISSION_LOAD_PROFILE {}",
            serde_json::to_string(&report).unwrap()
        ),
        Err(error) => panic!("admission load diagnostic setup failed: {error}"),
    }
}

#[test]
#[ignore = "external-input fixed 48-member CPU diagnostic without cache-accounting capture; set both diagnostic input paths"]
fn profile_admission_original_burst_with_production_budget() {
    print_report(CacheAccounting::Disabled);
}

#[test]
#[ignore = "external-input fixed 48-member CPU cache-accounting diagnostic; set both diagnostic input paths"]
fn profile_admission_original_burst_with_production_budget_cache_accounting() {
    print_report(CacheAccounting::Enabled);
}
