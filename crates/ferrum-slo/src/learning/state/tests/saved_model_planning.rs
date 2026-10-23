//! Opt-in CPU diagnostic: saved real costs, synthetic resident cohort.
//! No historical decision replay, cost qualification, GPU capacity, learned
//! stopping, or service performance is established by this experiment.

use crate::learning::state::{byte_limit, decode, Clock, SavedState};
use crate::learning::{rejection::Recorder, snapshot::Snapshot, worker::Learner};
use crate::learning::{LearningConfig, StateLoadStatus, TrainingHistoryPolicy};
use crate::model::{ExecutionIdentity, ModelFamily};
use ferrum_execution::{
    execution_attempt::{AttemptPrefixRestore, AttemptWorkKind},
    model_executor::{
        ExecutorPrefillAdmissionReceipt, PrefillShapePolicy, ResidentCohortCapacitySnapshot,
    },
    token_execution::ExecutionRange,
};
use ferrum_interfaces::request_observation::*;
use ferrum_scheduler::{
    cost_prediction::*, implementations::continuous::TokenAdmissionApply,
    vnext::AdmissionProbeOutcome, BatchHint, ContinuousBatchScheduler, SchedulerConfig,
};
use ferrum_types::{
    InferenceRequest, RequestIngressAnchor, RequestIngressBoundary, SchedulerSloConfig,
    SloStrategy, PROMPT_TOKENS_METADATA_KEY,
};
use serde_json::{json, Value};
use std::{
    fs::File,
    io::Read,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, UNIX_EPOCH},
};

const RESIDENTS: usize = 16;
const PREFILLERS: usize = 8;
const PROMPT: usize = 256;
const CAP: usize = 64;
const INPUT_VARIABLE: &str = "FERRUM_DIAGNOSTIC_LEARNING_STATE";

/// Counts original calls independently of the scheduler's simulation/cache
/// ledger. No per-query timer, trace allocation, or second prediction is used.
struct ConditionalSavedCosts {
    costs: Snapshot,
    scalar_calls: AtomicUsize,
    scenario_calls: AtomicUsize,
    known: AtomicUsize,
    // NoCoverage, Stale, Inexact, unsupported (all details), QueryLimit,
    // PlanningBudget. Details remain in normal scheduler diagnostics.
    unknown: [AtomicUsize; 6],
    stopping_calls: AtomicUsize,
    completion_calls: AtomicUsize,
    // One at-most-16-member input sample; not a complete query transcript.
    first_query: OnceLock<(CostExecution, Vec<CostWork>)>,
}

impl ConditionalSavedCosts {
    fn capture_first(&self, query: &CostQuery<'_>) {
        if query.work.len() <= RESIDENTS {
            self.first_query
                .get_or_init(|| (query.execution, query.work.to_vec()));
        }
    }
    fn record(&self, reason: Option<CostUnknownReason>) {
        let Some(reason) = reason else {
            self.known.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let index = match reason {
            CostUnknownReason::NoCoverage => 0,
            CostUnknownReason::Stale => 1,
            CostUnknownReason::InexactWork => 2,
            CostUnknownReason::UnsupportedDomain | CostUnknownReason::UnsupportedWithDetail(_) => 3,
            CostUnknownReason::QueryLimit => 4,
            CostUnknownReason::PlanningBudget => 5,
        };
        self.unknown[index].fetch_add(1, Ordering::Relaxed);
    }

    fn counters(&self) -> Value {
        json!({
            "scalar_calls": self.scalar_calls.load(Ordering::Relaxed),
            "scenario_calls": self.scenario_calls.load(Ordering::Relaxed),
            "known": self.known.load(Ordering::Relaxed),
            "unknown": {
                "no_coverage": self.unknown[0].load(Ordering::Relaxed),
                "stale": self.unknown[1].load(Ordering::Relaxed),
                "inexact_work": self.unknown[2].load(Ordering::Relaxed),
                "unsupported": self.unknown[3].load(Ordering::Relaxed),
                "query_limit": self.unknown[4].load(Ordering::Relaxed),
                "planning_budget": self.unknown[5].load(Ordering::Relaxed),
            },
            "conditional_stopping_calls": self.stopping_calls.load(Ordering::Relaxed),
            "conditional_completion_calls": self.completion_calls.load(Ordering::Relaxed),
        })
    }
}

impl CostSnapshot for ConditionalSavedCosts {
    fn revision(&self) -> u64 {
        self.costs.revision()
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        self.costs.query_limits()
    }

    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.scalar_calls.fetch_add(1, Ordering::Relaxed);
        self.capture_first(query);
        let prediction = self.costs.predict(query, now);
        self.record(match &prediction {
            CostPrediction::Known { .. } => None,
            CostPrediction::Unknown(reason) => Some(*reason),
        });
        prediction
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.scenario_calls.fetch_add(1, Ordering::Relaxed);
        self.capture_first(query);
        let prediction = self.costs.cost_scenarios(query, now, count);
        self.record(match &prediction {
            CostScenarioPrediction::Known { .. } => None,
            CostScenarioPrediction::Unknown(reason) => Some(*reason),
        });
        prediction
    }

    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.stopping_calls.fetch_add(1, Ordering::Relaxed);
        // A declared counterfactual endpoint, not recovered stopping evidence.
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
        self.completion_calls.fetch_add(1, Ordering::Relaxed);
        CompletionForecast {
            succeeded_mass: 1.0,
            failed_mass: 0.0,
            unresolved_mass: 0.0,
            // Required metadata is a declared synthetic premise, not a
            // recovered or fitted empirical completion cohort.
            samples: 1,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }
}

fn config() -> LearningConfig {
    LearningConfig {
        query: policy().query,
        training_history: TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 2,
        },
        ..Default::default()
    }
}

fn policy() -> SchedulerSloConfig {
    let mut value: SchedulerSloConfig = "ttft:15684.62,tpot:110.56,itl:196.55".parse().unwrap();
    value.strategy = SloStrategy::Joint;
    value.planning.max_elapsed_us = 10_000;
    value.planning.max_queries = 4096;
    value.planning.max_scenarios = 32;
    value.validate().unwrap();
    value
}

fn boundary_query(snapshot: &Snapshot, tokens: usize, now: Instant) -> Value {
    let work = [CostWork {
        kind: AttemptWorkKind::Prefill,
        requested: ExecutionRange::new(PROMPT - tokens, tokens).unwrap(),
        context_tokens: PROMPT,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }];
    let result = snapshot.cost_scenarios(
        &CostQuery {
            work: &work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        now,
        32,
    );
    match result {
        CostScenarioPrediction::Unknown(reason) => {
            json!({"requested_tokens": tokens, "unknown": reason})
        }
        CostScenarioPrediction::Known {
            central_service_time,
            scenarios,
            samples,
            observed_at,
            expires_at,
            ..
        } => json!({
            "requested_tokens": tokens,
            "known": true,
            "central_service_time": central_service_time,
            "scenario_count": scenarios.len(),
            "samples": samples,
            "evidence_age": now.saturating_duration_since(observed_at),
            "valid_for": expires_at.saturating_duration_since(now),
        }),
    }
}

fn run() -> Result<Value, String> {
    let path = std::env::var_os(INPUT_VARIABLE).ok_or_else(|| {
        format!("set {INPUT_VARIABLE} to an external retained learned-state.json")
    })?;
    let learning = config();
    learning.validate().map_err(|error| error.to_string())?;
    let limit = byte_limit(&learning).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| error.to_string())?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > limit {
        return Err("learning state exceeds existing byte limit".into());
    }
    let saved: SavedState = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let first = saved
        .domains
        .first()
        .ok_or("saved state contains no domains")?;
    let model: Value = serde_json::from_str(&first.model).map_err(|error| error.to_string())?;
    let identity_text = model
        .get("identity")
        .and_then(Value::as_str)
        .ok_or("saved model lacks its original execution identity")?;
    let identity =
        ExecutionIdentity::new(identity_text.to_owned()).map_err(|error| error.to_string())?;
    let identity_bytes = identity_text.len();
    let saved_domains: Vec<_> = saved
        .domains
        .iter()
        .map(|domain| {
            json!({
                "model_revision": domain.model_revision,
                "model_remaining_ns": domain.model_remaining_ns,
            })
        })
        .collect();
    // Finish queue assembly before choosing the one monotonic restore origin.
    let slo = policy();
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: RESIDENTS,
        active_decode_prefill_chunk: Some(256),
        prefill_step_chunk: Some(128),
        slo: Some(slo),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let mut members = Vec::with_capacity(RESIDENTS);
    for index in 0..RESIDENTS {
        let mut request = InferenceRequest::new(
            "synthetic cost-only diagnostic",
            "conditional-synthetic-cohort",
        )
        .with_ingress(ingress)
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, json!(PROMPT));
        request.sampling_params.max_tokens = CAP;
        request.stream = true;
        let id = scheduler
            .submit_request(request)
            .map_err(|error| error.to_string())?;
        members.push((id, usize::from(index >= PREFILLERS)));
    }
    let mut admission = scheduler
        .begin_token_admission(RESIDENTS, None)
        .map_err(|error| error.to_string())?;
    let mut admitted = 0;
    while let Some(candidate) = admission
        .next(&mut |_| {}, None)
        .map_err(|error| error.to_string())?
    {
        let request_id = candidate.request().id.clone();
        // Explicit synthetic owner premise; no actual device resources exist.
        match admission.apply(
            candidate,
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id }),
        ) {
            TokenAdmissionApply::Applied { accepted: Some(_) } => admitted += 1,
            _ => return Err("synthetic admission receipt was not applied".into()),
        }
    }
    admission.finish().map_err(|error| error.to_string())?;
    if admitted != RESIDENTS {
        return Err(format!("fixture admitted {admitted}/{RESIDENTS}"));
    }
    for (id, generated) in &members {
        if *generated == 1 {
            scheduler.mark_prefill_complete(id, PROMPT);
            scheduler.update_decode_progress(id, 1);
        }
    }

    let origin = Instant::now();
    let clock = Clock {
        monotonic: origin,
        wall: UNIX_EPOCH
            .checked_add(Duration::from_nanos(saved.saved_unix_ns))
            .ok_or("saved wall clock overflow")?,
    };
    // This tests the state at save time. It does not make the file fresh now.
    // Existing decoder enforces exact semantics, identity, domain, and ages.
    let restored = decode(&bytes, &learning, &identity, ModelFamily::Physical, clock)
        .map_err(|error| error.to_string())?;
    let load_report = restored.report;
    if load_report.status != StateLoadStatus::Loaded {
        return Ok(
            json!({"classification":"state_unqualified_at_save", "load_report":load_report}),
        );
    }
    let learner = Learner::restore(learning.clone(), identity, ModelFamily::Physical, restored);
    let mut snapshot = learner.current_snapshot(origin);
    // Separate pre-comparison diagnostics; no warmed ScenarioCache is shared.
    let boundary_at = Instant::now();
    let boundaries = [
        boundary_query(&snapshot, 1, boundary_at),
        boundary_query(&snapshot, 2, boundary_at),
    ];
    let rejection = Arc::new(Recorder::new(learning.query.max_participants, origin));
    snapshot.rejections = Some(rejection.clone());
    let source = Arc::new(ConditionalSavedCosts {
        costs: snapshot,
        scalar_calls: AtomicUsize::new(0),
        scenario_calls: AtomicUsize::new(0),
        known: AtomicUsize::new(0),
        unknown: std::array::from_fn(|_| AtomicUsize::new(0)),
        stopping_calls: AtomicUsize::new(0),
        completion_calls: AtomicUsize::new(0),
        first_query: OnceLock::new(),
    });
    let facts_at = Instant::now();
    let rules = Arc::new(
        EffectiveStoppingRules::new(
            b"synthetic-conditional-cap-endpoint".to_vec(),
            StoppingMechanism::CapOnly,
            64,
        )
        .ok_or("invalid synthetic rules")?,
    );
    let mut round = TimePlanningRound::new(
        source.clone(),
        slo,
        PrefillShapePolicy::Identity,
        CostExecution::SequentialPrefillThenDecode,
    );
    if !round.set_request_facts(members.iter().enumerate().map(|(index, (id, generated))| {
        (
            id.clone(),
            RequestStoppingObservation {
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: index as u64 + 1,
                },
                admitted_at: ingress.instant(),
                observed_at: facts_at,
                revision: 1,
                rules: rules.clone(),
                cap: CAP,
                generated_tokens: *generated,
                verified_survival_through: *generated,
                terminal: None,
                completion: None,
            },
        )
    })) {
        return Err("synthetic request facts rejected".into());
    }
    if !round.set_request_clocks(members.iter().map(|(id, generated)| {
        (
            id.clone(),
            RequestClockFacts {
                captured_at: facts_at,
                age: facts_at.duration_since(ingress.instant()),
                first_text: (*generated > 0).then_some(Duration::ZERO),
                last_text: (*generated > 0).then_some(Duration::ZERO),
                max_text_interval: None,
                nonempty_text_events: *generated as u64,
                violating_text_intervals: 0,
                itl_threshold: Some(Duration::from_secs_f64(slo.itl_ms / 1000.0)),
                generated_tokens: *generated,
                streaming: true,
            },
        )
    })) {
        return Err("synthetic request clocks rejected".into());
    }
    // Typed all-resident capacity is an explicit premise of this CPU example,
    // not a proof obtained from a GPU allocator or permission to admit waiters.
    round.set_resident_capacity(Some(ResidentCohortCapacitySnapshot {
        participants: RESIDENTS,
        max_batch_sequences: RESIDENTS,
    }));
    let hint = BatchHint {
        max_batch_size: RESIDENTS,
        max_tokens: 512,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    };
    let mut observed = scheduler.observe_planning_round(Some(round));
    let started = Instant::now();
    let selection = scheduler.next_batch_with_token_admissions(
        scheduler.prepare_token_batch_selection(hint),
        None,
        &mut |_| {},
        observed.as_mut(),
    );
    drop(observed.finish());
    let elapsed = started.elapsed();
    let (batch, selection_error) = match selection {
        Ok(selection) => (selection.batch, None),
        Err(error) => (None, Some(error.to_string())),
    };
    let selected: Vec<_> = batch
        .as_ref()
        .map(|batch| {
            batch
                .requests
                .iter()
                .map(|request| {
                    json!({
                        "member": members.iter().position(|(id, _)| *id == request.request.id),
                        "tokens_processed": request.tokens_processed,
                        "tokens_to_process": request.tokens_to_process,
                        "initially_decoder": members.iter().find(|(id, _)| *id == request.request.id).map(|(_, generated)| *generated == 1),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(json!({
        "scope": "one saved-cost at-save counterfactual on a synthetic all-resident public scheduler path",
        "input_path": std::path::PathBuf::from(path), "input_bytes": bytes.len(),
        "saved_revision": saved.revision, "saved_unix_ns": saved.saved_unix_ns,
        "original_identity_bytes": identity_bytes, "identity_rewritten": false,
        "saved_domains": saved_domains, "load_report": load_report,
        "semantics": saved.semantics, "learning_config": learning,
        "premises": {
            "residents": RESIDENTS, "prefillers": PREFILLERS, "prompt_tokens": PROMPT,
            "output_cap": CAP, "decoder_output_already_generated": 1,
            "endpoint": "cap, conditional and not learned", "product_success": "[1,1], conditional and not learned; samples=1/EmpiricalMaturedCohort are synthetic contract premises, not recovered records",
            "capacity": "declared synthetic resident-only capacity, no GPU owner",
            "clocks": "fresh synthetic ingress, one visible event per generated token, no prior violations",
            "execution_identity": "retained saved-model identity; current planner is a counterfactual consumer",
        },
        "policy": slo, "boundary_probes_outside_comparison": boundaries,
        "comparison_elapsed": elapsed, "restore_to_comparison_end": origin.elapsed(),
        "original_provider_calls": source.counters(),
        "first_original_query": source.first_query.get().map(|(execution, work)| json!({
            "execution": format!("{execution:?}"),
            "work": work.iter().map(|item| json!({
                "kind": format!("{:?}", item.kind),
                "start": item.requested.start(),
                "tokens": item.requested.as_range().len(),
                "context_tokens": item.context_tokens,
                "prefix_restore": format!("{:?}", item.prefix_restore),
                "internally_reused_tokens": item.internally_reused_tokens,
            })).collect::<Vec<_>>(),
        })),
        "owner_support_rejections": rejection.stats(),
        "scheduler": scheduler.slo_snapshot(),
        "selected_batch_tokens": batch.as_ref().map(|batch| batch.total_tokens()),
        "selected": selected, "selection_error": selection_error,
        "runtime_execution_performed": false,
        "interpretation": "Only completed conditional decisions classify model feasibility; budget failures leave it unknown. No outcome is required to select an alternative.",
    }))
}

#[test]
#[ignore = "external saved-state CPU diagnostic; set FERRUM_DIAGNOSTIC_LEARNING_STATE explicitly"]
fn saved_model_public_scheduler_diagnostic() {
    match run() {
        Ok(report) => println!("{}", serde_json::to_string(&report).unwrap()),
        Err(error) => {
            println!(
                "{}",
                json!({"setup_error": error, "runtime_execution_performed": false})
            );
            panic!("saved-model diagnostic setup failed: {error}");
        }
    }
}
