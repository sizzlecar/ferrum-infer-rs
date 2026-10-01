use super::{
    metrics::{count, value},
    wire::*,
    ReplayPair,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct Context {
    phase: String,
    rows: Option<usize>,
    model: Option<u64>,
}
impl Context {
    fn key(&self) -> String {
        format!(
            "phase={};work_rows={:?};model_version={:?}",
            self.phase, self.rows, self.model
        )
    }
}
struct Attempt {
    context: Context,
    constructed: BTreeSet<usize>,
    queried: BTreeSet<usize>,
}
struct Scope {
    checkpoint: Checkpoint,
    context: Context,
}
struct Projection {
    start: Checkpoint,
    previous: Checkpoint,
    context: Context,
}
#[derive(Default)]
struct Replay {
    ended: bool,
    selected: bool,
}
pub(super) struct Transaction {
    pub id: u64,
    pub maximum_ordinal: u64,
    pub event_count: usize,
    pub counters: BTreeMap<String, u64>,
    pub metrics: BTreeMap<String, Vec<u64>>,
    pub issues: Vec<String>,
    pub end: Option<End>,
    snapshot: Option<(u64, usize)>,
    attempts: BTreeMap<u64, Attempt>,
    attempt_ids: BTreeSet<u64>,
    current_attempt: Option<u64>,
    replays: BTreeMap<u64, Replay>,
    active_replay: Option<u64>,
    scopes: BTreeMap<String, Vec<Scope>>,
    projection: Option<Projection>,
    search_projection_ns: u64,
    replay_projection_ns: u64,
    first_replay: Option<(usize, u64, Option<u64>)>,
    previous_elapsed: Option<u64>,
    hard_budget: Option<u64>,
}
impl Transaction {
    pub fn new(id: u64, ordinal: u64) -> Self {
        Self {
            id,
            maximum_ordinal: ordinal,
            event_count: 1,
            counters: BTreeMap::new(),
            metrics: BTreeMap::new(),
            issues: vec![],
            end: None,
            snapshot: None,
            attempts: BTreeMap::new(),
            attempt_ids: BTreeSet::new(),
            current_attempt: None,
            replays: BTreeMap::new(),
            active_replay: None,
            scopes: BTreeMap::new(),
            projection: None,
            search_projection_ns: 0,
            replay_projection_ns: 0,
            first_replay: None,
            previous_elapsed: None,
            hard_budget: None,
        }
    }
    fn issue(&mut self, why: impl Into<String>) {
        if self.issues.len() < 32 {
            self.issues.push(why.into());
        }
    }
    fn context(&self) -> Context {
        self.current_attempt
            .and_then(|id| self.attempts.get(&id))
            .map(|a| a.context.clone())
            .unwrap_or(Context {
                phase: "controller".into(),
                rows: None,
                model: self.snapshot.map(|s| s.0),
            })
    }
    pub fn event(&mut self, event: Event) {
        match event {
            Event::Snapshot {
                cost_model_version,
                requests,
            } => {
                self.snapshot = Some((cost_model_version, requests.len()));
                count(
                    &mut self.counters,
                    format!(
                        "snapshot/model_version={cost_model_version}/queue_rows={}",
                        requests.len()
                    ),
                    1,
                );
            }
            Event::ControllerCheckpoint(checkpoint) => self.checkpoint(checkpoint),
            Event::AttemptBegin {
                attempt,
                phase,
                depth,
                requests,
                work,
            } => {
                if attempt == 0
                    || !self.attempt_ids.insert(attempt)
                    || self.current_attempt.is_some()
                {
                    self.issue("duplicate/overlapping attempt");
                }
                let group = if self.active_replay.is_some() {
                    "independent_replay"
                } else if phase == "Search" {
                    "search"
                } else {
                    "unbound_phase"
                };
                if group == "unbound_phase" {
                    self.issue("attempt phase lacks its replay boundary");
                }
                let context = Context {
                    phase: group.into(),
                    rows: Some(work.len()),
                    model: self.snapshot.map(|s| s.0),
                };
                count(
                    &mut self.counters,
                    format!(
                        "attempts/{}/depth={depth}/request_rows={}",
                        context.key(),
                        requests.len()
                    ),
                    1,
                );
                count(&mut self.counters, format!("attempt_phase_wire/{phase}"), 1);
                self.attempts.insert(
                    attempt,
                    Attempt {
                        context,
                        constructed: BTreeSet::new(),
                        queried: BTreeSet::new(),
                    },
                );
                self.current_attempt = Some(attempt);
            }
            Event::QueryConstructed {
                attempt,
                alternative,
                input_unknown,
                demand_error,
            } => {
                let Some(a) = self.attempts.get_mut(&attempt) else {
                    self.issue("constructed query without open attempt");
                    return;
                };
                if !a.constructed.insert(alternative) {
                    self.issue("duplicate constructed alternative");
                    return;
                }
                let context = a.context.key();
                count(&mut self.counters, format!("constructed/{context}"), 1);
                if let Some(reason) = input_unknown {
                    count(
                        &mut self.counters,
                        format!("input_unknown/{context}/reason={reason}"),
                        1,
                    );
                }
                if let Some(reason) = demand_error {
                    count(
                        &mut self.counters,
                        format!("demand_error/{context}/reason={reason}"),
                        1,
                    );
                }
            }
            Event::QueryLookup {
                attempt,
                alternative,
                outcome,
            } => {
                let Some(a) = self.attempts.get_mut(&attempt) else {
                    self.issue("lookup without open attempt");
                    return;
                };
                if !a.constructed.contains(&alternative) || !a.queried.insert(alternative) {
                    self.issue("lookup missing construction or duplicate");
                    return;
                }
                let context = a.context.key();
                let outcome_key = match outcome {
                    LookupOutcome::Known { cost } => {
                        value(
                            &mut self.metrics,
                            format!("predicted_typical/{context}"),
                            cost.typical_ns,
                        );
                        value(
                            &mut self.metrics,
                            format!("predicted_planning/{context}"),
                            cost.planning_ns,
                        );
                        value(
                            &mut self.metrics,
                            format!("prediction_valid_for/{context}"),
                            cost.valid_for_ns,
                        );
                        format!("known/model_version={}", cost.model_version)
                    }
                    LookupOutcome::StructuredUnknown { reason } => {
                        format!("structured_unknown/{reason}")
                    }
                    LookupOutcome::ModelUnavailable => "model_unavailable".into(),
                    LookupOutcome::ClockMappingFailed => "clock_mapping_failed".into(),
                };
                count(
                    &mut self.counters,
                    format!("lookup/{context}/{outcome_key}"),
                    1,
                );
            }
            Event::AttemptEnd {
                attempt,
                constructed,
                queried,
                not_queried_start,
                not_queried_end,
                reason,
            } => {
                let Some(a) = self.attempts.remove(&attempt) else {
                    self.issue("attempt_end without begin");
                    return;
                };
                if self.current_attempt != Some(attempt)
                    || a.constructed.len() != constructed
                    || a.queried.len() != queried
                    || not_queried_start != queried
                    || not_queried_end != constructed
                    || queried > constructed
                    || a.constructed.iter().copied().ne(0..constructed)
                    || a.queried.iter().copied().ne(0..queried)
                {
                    self.issue("attempt alternatives/counts/nonqueried suffix mismatch");
                }
                self.current_attempt = None;
                count(
                    &mut self.counters,
                    format!("attempt_end/{}/reason={reason}", a.context.key()),
                    1,
                );
                count(
                    &mut self.counters,
                    format!("not_queried/{0}", a.context.key()),
                    (constructed.saturating_sub(queried)) as u64,
                );
            }
            Event::ReplayBegin { replay, waves } => {
                if replay == 0
                    || self.active_replay.is_some()
                    || self.replays.insert(replay, Replay::default()).is_some()
                {
                    self.issue("duplicate/overlapping replay");
                }
                self.active_replay = Some(replay);
                count(&mut self.counters, format!("replay_begin/waves={waves}"), 1);
            }
            Event::ReplayEnd { replay, reason } => {
                if self.active_replay != Some(replay) || self.current_attempt.is_some() {
                    self.issue("replay end order mismatch");
                }
                self.active_replay = None;
                match self.replays.get_mut(&replay) {
                    Some(r) if !r.ended => r.ended = true,
                    _ => self.issue("replay_end without unique begin"),
                }
                count(&mut self.counters, format!("replay_end/{reason}"), 1);
            }
            Event::SelectedReplay { replay } => {
                match self.replays.get_mut(&replay) {
                    Some(r) if r.ended && !r.selected => r.selected = true,
                    _ => self.issue("selected replay not uniquely completed"),
                }
                count(&mut self.counters, "selected_replay", 1);
            }
            Event::TransactionEnd(end) => {
                if self.end.is_some() {
                    self.issue("duplicate transaction end");
                }
                if !self.attempts.is_empty()
                    || self.active_replay.is_some()
                    || self.scopes.values().any(|s| !s.is_empty())
                {
                    self.issue("transaction ended with open attempt/replay/stage");
                }
                if !end.audit_available {
                    self.issue("transaction audit unavailable");
                }
                if self
                    .hard_budget
                    .is_some_and(|budget| budget != end.budget_ns)
                {
                    self.issue("end budget differs from original checkpoint budget");
                }
                count(
                    &mut self.counters,
                    format!(
                        "transaction_end/outcome={}/decision={}/reason={}",
                        end.outcome, end.decision, end.reason
                    ),
                    1,
                );
                count(
                    &mut self.counters,
                    if self.snapshot.is_some() {
                        "transactions_with_snapshot"
                    } else {
                        "transactions_without_snapshot"
                    },
                    1,
                );
                if let Some((version, rows)) = self.snapshot {
                    count(
                        &mut self.counters,
                        format!(
                            "transaction_final_snapshot/model_version={version}/queue_rows={rows}"
                        ),
                        1,
                    );
                }
                let context = format!(
                    "model_version={:?}/reason={}",
                    self.snapshot.map(|s| s.0),
                    end.reason
                );
                count(
                    &mut self.counters,
                    format!("transaction_population/{context}"),
                    1,
                );
                for (field, n) in [
                    ("enumeration_attempts", end.search.enumeration_attempts),
                    ("expanded_candidates", end.search.expanded_candidates),
                    ("generated_candidates", end.search.generated_candidates),
                    ("candidate_truncations", end.search.candidate_truncations),
                    (
                        "resource_unknown_candidates",
                        end.search.resource_unknown_candidates,
                    ),
                    (
                        "cost_unknown_candidates",
                        end.search.cost_unknown_candidates,
                    ),
                    (
                        "shape_unknown_candidates",
                        end.search.shape_unknown_candidates,
                    ),
                ] {
                    count(
                        &mut self.counters,
                        format!("search_population/{context}/{field}"),
                        n,
                    );
                }
                value(
                    &mut self.metrics,
                    format!("transaction_planning_wall/{context}"),
                    end.planning_wall_ns,
                );
                value(
                    &mut self.metrics,
                    format!("transaction_budget/{context}"),
                    end.budget_ns,
                );
                count(
                    &mut self.counters,
                    "planner_exhausted_transactions",
                    u64::from(end.planner_exhausted),
                );
                count(
                    &mut self.counters,
                    "hard_exhausted_transactions",
                    u64::from(end.hard_exhausted),
                );
                count(
                    &mut self.counters,
                    "resource_unknown_candidates",
                    end.search.resource_unknown_candidates,
                );
                count(
                    &mut self.counters,
                    "cost_unknown_candidates",
                    end.search.cost_unknown_candidates,
                );
                count(
                    &mut self.counters,
                    "shape_unknown_candidates",
                    end.search.shape_unknown_candidates,
                );
                count(
                    &mut self.counters,
                    "search_soft_stops",
                    end.search.search_soft_stops,
                );
                count(
                    &mut self.counters,
                    "replay_reserve_stops",
                    end.search.replay_reserve_stops,
                );
                value(
                    &mut self.metrics,
                    "measured_replay_work",
                    end.search.measured_replay_work_ns,
                );
                value(
                    &mut self.metrics,
                    "replay_reserve",
                    end.search.replay_reserve_ns,
                );
                self.end = Some(end);
            }
            Event::TransactionBegin | Event::RunIdentity(_) => {
                self.issue("unexpected begin/identity inside transaction")
            }
        }
    }
    fn checkpoint(&mut self, c: Checkpoint) {
        let context = self.context();
        if self.hard_budget.is_some_and(|b| b != c.hard_budget_ns) {
            self.issue("hard budget changed within transaction");
        }
        self.hard_budget = Some(c.hard_budget_ns);
        count(
            &mut self.counters,
            format!("checkpoint/{}/edge={}", c.stage, c.edge),
            1,
        );
        if let Some(elapsed) = c.elapsed_ns {
            if self.previous_elapsed.is_some_and(|p| p > elapsed) {
                self.issue("controller clock moved backwards");
            }
            self.previous_elapsed = Some(elapsed);
            value(
                &mut self.metrics,
                format!("hard_remaining_at/{}/{}/{}", c.stage, c.edge, context.key()),
                c.hard_budget_ns.saturating_sub(elapsed),
            );
            if let Some(deadline) = c.optional_deadline_elapsed_ns {
                value(
                    &mut self.metrics,
                    format!(
                        "optional_remaining_at/{}/{}/{}",
                        c.stage,
                        c.edge,
                        context.key()
                    ),
                    deadline.saturating_sub(elapsed),
                );
            }
        } else {
            count(&mut self.counters, "checkpoint_missing_clock", 1);
        }
        if let Some(n) = c.completion_preparation_ns {
            value(
                &mut self.metrics,
                format!("completion_preparation_at/{}", c.stage),
                n,
            );
        }
        if let Some(n) = c.publication_reserve_ns {
            value(
                &mut self.metrics,
                format!("publication_reserve_at/{}", c.stage),
                n,
            );
        }
        // Each projection's adjacent original points partition its wall time.
        // Nested inclusive scope quantiles are reported separately, never added.
        if let Some(projection) = self.projection.as_mut() {
            if let Some(delta) = c
                .elapsed_ns
                .zip(projection.previous.elapsed_ns)
                .and_then(|(b, a)| b.checked_sub(a))
            {
                value(
                    &mut self.metrics,
                    format!(
                        "projection_segment/{}:{}->{}:{}/{}",
                        projection.previous.stage,
                        projection.previous.edge,
                        c.stage,
                        c.edge,
                        projection.context.key()
                    ),
                    delta,
                );
            }
            projection.previous = c.clone();
        }
        match c.edge.as_str() {
            "begin" => {
                self.scopes.entry(c.stage.clone()).or_default().push(Scope {
                    checkpoint: c.clone(),
                    context: context.clone(),
                });
                if c.stage == "candidate_projection" {
                    if self.projection.is_some() {
                        self.issue("overlapping candidate projection");
                    }
                    self.projection = Some(Projection {
                        start: c.clone(),
                        previous: c.clone(),
                        context,
                    });
                }
            }
            "end" => {
                let scope = self.scopes.get_mut(&c.stage).and_then(Vec::pop);
                if let Some(scope) = scope {
                    if let Some(delta) = c
                        .elapsed_ns
                        .zip(scope.checkpoint.elapsed_ns)
                        .and_then(|(b, a)| b.checked_sub(a))
                    {
                        value(
                            &mut self.metrics,
                            format!("scope_wall/{}/{}", c.stage, scope.context.key()),
                            delta,
                        );
                    }
                } else {
                    self.issue(format!("stage end without begin: {}", c.stage));
                }
                if c.stage == "candidate_projection" {
                    if let Some(p) = self.projection.take() {
                        if let Some((end, start)) = c.elapsed_ns.zip(p.start.elapsed_ns) {
                            if let Some(wall) = end.checked_sub(start) {
                                if p.context.phase == "search" {
                                    self.search_projection_ns =
                                        self.search_projection_ns.saturating_add(wall);
                                }
                                if p.context.phase == "independent_replay" {
                                    self.replay_projection_ns =
                                        self.replay_projection_ns.saturating_add(wall);
                                    self.first_replay.get_or_insert((
                                        p.context.rows.unwrap_or(0),
                                        p.start.hard_budget_ns.saturating_sub(start),
                                        p.start
                                            .optional_deadline_elapsed_ns
                                            .map(|d| d.saturating_sub(start)),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            "point" => {}
            _ => self.issue("unknown checkpoint edge"),
        }
    }
    pub fn pair(&self) -> Option<ReplayPair> {
        let (work_rows, hard_remaining_ns, optional_remaining_ns) = self.first_replay?;
        Some(ReplayPair {
            transaction: self.id,
            model_version: self.snapshot.map(|s| s.0),
            work_rows,
            search_projection_sum_ns: self.search_projection_ns,
            first_replay_hard_remaining_ns: hard_remaining_ns,
            first_replay_optional_remaining_ns: optional_remaining_ns,
            replay_projection_sum_ns: self.replay_projection_ns,
            measured_replay_work_ns: self.end.as_ref()?.search.measured_replay_work_ns,
            replay_reserve_ns: self.end.as_ref()?.search.replay_reserve_ns,
        })
    }
}
