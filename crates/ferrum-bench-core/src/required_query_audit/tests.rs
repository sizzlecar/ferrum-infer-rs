use super::*;
use serde_json::{json, Value};
use std::io::Cursor;

struct Trace {
    records: Vec<Value>,
    ordinal: u64,
    transactions: u64,
}
impl Trace {
    fn new() -> Self {
        let mut trace = Self {
            records: vec![json!({
                "schema": SCHEMA, "event": "run_header", "run_id": "fixture",
                "authority": "diagnostic_only_not_profile_source_or_witness"
            })],
            ordinal: 0,
            transactions: 0,
        };
        trace.event(
            0,
            "run_identity",
            json!({"config":{"planner":{"max_planning_us":2000}}}),
        );
        trace
    }
    fn event(&mut self, transaction: u64, event: &str, data: Value) {
        self.ordinal += 1;
        self.records.push(json!({
            "schema": SCHEMA, "ordinal": self.ordinal, "retained_ordinal": self.ordinal,
            "transaction": transaction, "event": event, "data": data
        }));
    }
    fn begin(&mut self, transaction: u64) {
        self.transactions += 1;
        self.event(transaction, "transaction_begin", Value::Null);
    }
    fn snapshot(&mut self, transaction: u64, version: u64, rows: usize) {
        self.event(
            transaction,
            "snapshot",
            json!({
                "cost_model_version": version, "requests": vec![Value::Null; rows]
            }),
        );
    }
    fn checkpoint(&mut self, tx: u64, stage: &str, edge: &str, elapsed: u64) {
        self.event(
            tx,
            "controller_checkpoint",
            json!({
                "stage":stage, "edge":edge, "elapsed_ns":elapsed,
                "hard_budget_ns":2_000_000, "optional_deadline_elapsed_ns":1_600_000,
                "completion_preparation_ns":200_000, "publication_reserve_ns":400_000
            }),
        );
    }
    fn attempt(&mut self, tx: u64, id: u64, phase: &str, rows: usize) {
        self.event(
            tx,
            "attempt_begin",
            json!({
                "attempt":id, "phase":phase, "depth":0, "requests":vec![Value::Null;8],
                "work":vec![Value::Null;rows]
            }),
        );
    }
    fn query(&mut self, tx: u64, id: u64, outcome: Value) {
        self.event(
            tx,
            "query_constructed",
            json!({
                "attempt":id,"alternative":0,"input_unknown":null,"demand_error":null
            }),
        );
        self.event(
            tx,
            "query_lookup",
            json!({"attempt":id,"alternative":0,"outcome":outcome}),
        );
    }
    fn end_attempt(&mut self, tx: u64, id: u64, constructed: usize, queried: usize) {
        self.event(
            tx,
            "attempt_end",
            json!({
                "attempt":id,"constructed":constructed,"queried":queried,
                "not_queried_start":queried,"not_queried_end":constructed,"reason":"Completed"
            }),
        );
    }
    fn end(&mut self, tx: u64, reason: &str) {
        self.event(
            tx,
            "transaction_end",
            json!({
                "audit_available":true,"audit_times_exclude_end_recording":true,
                "outcome":"submitted","decision":"unknown","reason":reason,
                "planning_wall_ns":1_000_000,"budget_ns":2_000_000,
                "planner_exhausted":false,"hard_exhausted":false,
            "search":{
                "enumeration_attempts":0,"expanded_candidates":0,
                "generated_candidates":0,"candidate_truncations":0,
                "resource_unknown_candidates":0,"cost_unknown_candidates":0,
                    "shape_unknown_candidates":0,"measured_replay_work_ns":50_000,
                    "replay_reserve_ns":100_000,"replay_reserve_stops":0,"search_soft_stops":0
                }
            }),
        );
    }
    fn footer(&mut self) {
        self.records.push(json!({
            "schema":SCHEMA,"event":"run_footer","recording_complete":true,
            "requires_successful_close":true,
            "statistics":{
                "offered":self.ordinal,"accepted":self.ordinal,"written":self.ordinal,
                "lost":{"contention":0,"capacity":0,"transaction_limit":0,"event_limit":0,
                    "closed":0,"writer_failed":0,"allocation":0,"encoding":0},
                "first_lost_ordinal":0,"last_lost_ordinal":0,
                "transactions":self.transactions,"active_transactions":0,"abandoned_transactions":0,
                "counter_exhausted":false,"writer_failed":false,"first_error":null,
                "closed":false,"footer_written":false,"flushed_and_synced":false
            }
        }));
    }
    fn bytes(&self) -> Vec<u8> {
        self.records
            .iter()
            .flat_map(|record| {
                let mut bytes = serde_json::to_vec(record).unwrap();
                bytes.push(b'\n');
                bytes
            })
            .collect()
    }
    fn audit(&self) -> RequiredQueryAudit {
        audit_required_queries(Cursor::new(self.bytes()), AuditOptions::default()).unwrap()
    }
}

#[test]
fn complete_recording_never_attests_successful_close_or_slo() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.end(1, "cost_unavailable");
    trace.footer();
    let report = trace.audit();
    assert!(
        report.integrity.trace_content_complete,
        "{:?}",
        report.integrity.issues
    );
    assert!(!report.integrity.successful_close_attested);
    assert!(report.integrity.footer_requires_external_successful_close);
    assert_eq!(report.integrity.transactions_ended, 1);
    assert_eq!(report.configured_planning_budget_ns, Some(2_000_000));
    assert_eq!(report.input_bytes, trace.bytes().len() as u64);
    assert_eq!(
        report.input_jsonl_sha256,
        format!("{:x}", Sha256::digest(trace.bytes()))
    );
}

#[test]
fn snapshot_absence_and_model_unavailable_have_different_denominators() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.checkpoint(1, "resource_read_unavailable", "point", 10);
    trace.end(1, "resources_unavailable");
    trace.begin(2);
    trace.snapshot(2, 0, 8);
    trace.attempt(2, 1, "Search", 3);
    trace.query(2, 1, json!({"kind":"model_unavailable"}));
    trace.end_attempt(2, 1, 1, 1);
    trace.end(2, "cost_unavailable");
    trace.footer();
    let report = trace.audit();
    assert!(
        report.integrity.trace_content_complete,
        "{:?}",
        report.integrity.issues
    );
    assert_eq!(report.counters["transactions_without_snapshot"], 1);
    assert_eq!(report.counters["transactions_with_snapshot"], 1);
    assert_eq!(
        report.counters
            ["lookup/phase=search;work_rows=Some(3);model_version=Some(0)/model_unavailable"],
        1
    );
    assert!(!report
        .counters
        .keys()
        .any(|key| key.contains("lookup/phase=controller")));
}

fn paired_transaction(trace: &mut Trace, tx: u64, search_wall: u64, replay_wall: u64) {
    trace.begin(tx);
    trace.snapshot(tx, 7, 8);
    trace.attempt(tx, 1, "Search", 3);
    trace.checkpoint(tx, "candidate_projection", "begin", 100_000);
    trace.checkpoint(
        tx,
        "route_providers_begin",
        "point",
        100_000 + search_wall / 2,
    );
    trace.checkpoint(tx, "candidate_projection", "end", 100_000 + search_wall);
    trace.end_attempt(tx, 1, 0, 0);
    trace.event(tx, "replay_begin", json!({"replay":1,"waves":1}));
    trace.attempt(tx, 2, "IndependentReplay { replay: 1 }", 3);
    trace.checkpoint(tx, "candidate_projection", "begin", 900_000);
    trace.checkpoint(tx, "candidate_projection", "end", 900_000 + replay_wall);
    trace.end_attempt(tx, 2, 0, 0);
    trace.event(tx, "replay_end", json!({"replay":1,"reason":"Completed"}));
    trace.event(tx, "selected_replay", json!({"replay":1}));
    trace.end(tx, "selected");
}
#[test]
fn timings_use_same_transaction_pairs_and_original_work_rows() {
    let mut trace = Trace::new();
    paired_transaction(&mut trace, 1, 100_000, 20_000);
    paired_transaction(&mut trace, 2, 600_000, 10_000);
    trace.footer();
    let report = trace.audit();
    assert!(
        report.integrity.trace_content_complete,
        "{:?}",
        report.integrity.issues
    );
    assert_eq!(report.replay_pair_examples.len(), 2);
    let first = &report.replay_pair_examples[0];
    assert_eq!(first.work_rows, 3);
    assert_eq!(first.search_projection_sum_ns, 100_000);
    assert_eq!(first.replay_projection_sum_ns, 20_000);
    assert_eq!(first.first_replay_hard_remaining_ns, 1_100_000);
    assert_eq!(first.first_replay_optional_remaining_ns, Some(700_000));
    let key =
        "scope_wall/candidate_projection/phase=search;work_rows=Some(3);model_version=Some(7)";
    let distribution = &report.nanoseconds[key];
    assert_eq!(
        (
            distribution.samples,
            distribution.p50_ns,
            distribution.p99_ns
        ),
        (2, 100_000, 600_000)
    );
    assert_eq!(distribution.total_ns, 700_000);
}

#[test]
fn externally_supplied_cut_filters_whole_transactions_without_hiding_global_loss() {
    let mut trace = Trace::new();
    paired_transaction(&mut trace, 1, 100_000, 20_000);
    let cut = trace.ordinal + 1;
    paired_transaction(&mut trace, 2, 600_000, 10_000);
    trace.footer();
    let options = AuditOptions {
        before_ordinal: Some(cut),
        ..AuditOptions::default()
    };
    let report = audit_required_queries(Cursor::new(trace.bytes()), options.clone()).unwrap();
    assert!(report.integrity.trace_content_complete);
    assert_eq!(report.integrity.included_transactions, 1);
    assert_eq!(report.integrity.transactions_ended, 2);
    assert_eq!(report.replay_pair_examples[0].transaction, 1);
    trace.records.last_mut().unwrap()["statistics"]["lost"]["contention"] = json!(1);
    let report = audit_required_queries(Cursor::new(trace.bytes()), options).unwrap();
    assert!(!report.integrity.trace_content_complete);
    assert_eq!(report.integrity.included_transactions, 1);
}

#[test]
fn missing_footer_ordinal_duplicate_and_open_transaction_are_detected() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.end(1, "cost_unavailable");
    assert!(!trace.audit().integrity.trace_content_complete);
    trace.footer();
    trace.records[3]["ordinal"] = json!(2);
    let report = trace.audit();
    assert!(report
        .integrity
        .issues
        .iter()
        .any(|s| s.contains("duplicate offered")));
    let mut trace = Trace::new();
    trace.begin(1);
    trace.footer();
    let report = trace.audit();
    assert!(report
        .integrity
        .issues
        .iter()
        .any(|s| s.contains("missing end")));
}

#[test]
fn unmatched_stages_and_attempt_query_holes_cannot_produce_complete_evidence() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.snapshot(1, 1, 1);
    trace.checkpoint(1, "snapshot", "end", 100);
    trace.attempt(1, 1, "Search", 1);
    trace.event(1, "query_constructed", json!({"attempt":1,"alternative":1}));
    trace.end_attempt(1, 1, 1, 0);
    trace.end(1, "cost_unavailable");
    trace.footer();
    let report = trace.audit();
    assert!(!report.integrity.trace_content_complete);
    assert!(report
        .integrity
        .issues
        .iter()
        .any(|s| s.contains("stage end without begin")));
    assert!(report
        .integrity
        .issues
        .iter()
        .any(|s| s.contains("alternatives/counts")));
}

#[test]
fn event_and_line_limits_fail_without_unbounded_reading() {
    let trace = Trace::new();
    let options = AuditOptions {
        maximum_line_bytes: 8,
        ..AuditOptions::default()
    };
    assert_eq!(
        audit_required_queries(Cursor::new(trace.bytes()), options)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    let mut trace = Trace::new();
    trace.begin(1);
    trace.end(1, "cost_unavailable");
    trace.footer();
    let options = AuditOptions {
        maximum_events: 2,
        ..AuditOptions::default()
    };
    assert_eq!(
        audit_required_queries(Cursor::new(trace.bytes()), options)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn offered_order_can_differ_from_retained_order_but_gaps_cannot() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.end(1, "cost_unavailable");
    trace.footer();
    trace.records[2]["ordinal"] = json!(3);
    trace.records[3]["ordinal"] = json!(2);
    assert!(trace.audit().integrity.trace_content_complete);
    trace.records[3]["retained_ordinal"] = json!(4);
    assert!(!trace.audit().integrity.trace_content_complete);
}
