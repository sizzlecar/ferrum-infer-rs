//! Bounded streaming audit of original planner diagnostic JSONL. This cannot
//! create a cost model, scheduler witness, successful close receipt or SLO claim.
mod metrics;
mod transaction;
mod wire;
pub use metrics::Distribution;
use metrics::{count, value};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, BufRead},
};
use transaction::Transaction;
use wire::*;

#[derive(Clone, Debug, Serialize)]
pub struct AuditOptions {
    pub maximum_line_bytes: usize,
    pub maximum_events: u64,
    pub maximum_active_transactions: usize,
    pub maximum_transaction_events: usize,
    /// External boundary only; the trace does not contain observation QueueLoss.
    pub through_transaction: Option<u64>,
    /// Include only complete transactions whose every ordinal precedes this cut.
    pub before_ordinal: Option<u64>,
}
impl Default for AuditOptions {
    fn default() -> Self {
        Self {
            maximum_line_bytes: 2 * 1024 * 1024,
            maximum_events: 4_000_000,
            maximum_active_transactions: 64,
            maximum_transaction_events: 8192,
            through_transaction: None,
            before_ordinal: None,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Integrity {
    pub trace_content_complete: bool,
    pub successful_close_attested: bool,
    pub footer_requires_external_successful_close: bool,
    pub event_records: u64,
    pub unique_offered_ordinals: u64,
    pub maximum_offered_ordinal: u64,
    pub transactions_begun: u64,
    pub transactions_ended: u64,
    pub included_transactions: u64,
    pub issue_count: u64,
    pub issues: Vec<String>,
}
impl Integrity {
    fn issue(&mut self, message: impl Into<String>) {
        self.issue_count += 1;
        if self.issues.len() < 128 {
            self.issues.push(message.into());
        }
    }
}
#[derive(Debug, Serialize)]
pub struct ReplayPair {
    pub transaction: u64,
    pub model_version: Option<u64>,
    pub work_rows: usize,
    pub search_projection_sum_ns: u64,
    pub first_replay_hard_remaining_ns: u64,
    pub first_replay_optional_remaining_ns: Option<u64>,
    pub replay_projection_sum_ns: u64,
    pub measured_replay_work_ns: u64,
    pub replay_reserve_ns: u64,
}
#[derive(Debug, Serialize)]
pub struct RequiredQueryAudit {
    pub schema: &'static str,
    pub purpose: &'static str,
    pub input_jsonl_sha256: String,
    pub input_bytes: u64,
    pub run_id: Option<String>,
    pub configured_planning_budget_ns: Option<u64>,
    pub scope: AuditOptions,
    pub integrity: Integrity,
    pub counters: BTreeMap<String, u64>,
    pub nanoseconds: BTreeMap<String, Distribution>,
    pub replay_pair_examples: Vec<ReplayPair>,
    pub timing_interpretation: &'static str,
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Reads each bounded line once. All distributions use nearest-rank quantiles
/// over original paired values; snapshots absent from a failed transaction are
/// never relabeled as model-unavailable query results.
pub fn audit_required_queries(
    mut reader: impl BufRead,
    options: AuditOptions,
) -> io::Result<RequiredQueryAudit> {
    if options.maximum_line_bytes == 0
        || options.maximum_line_bytes > 16 * 1024 * 1024
        || options.maximum_events == 0
        || options.maximum_events > 16_000_000
        || options.maximum_active_transactions == 0
        || options.maximum_active_transactions > 1024
        || options.maximum_transaction_events == 0
        || options.maximum_transaction_events > 65536
    {
        return Err(invalid("invalid diagnostic audit resource limits"));
    }
    let mut integrity = Integrity {
        trace_content_complete: false,
        successful_close_attested: false,
        footer_requires_external_successful_close: false,
        event_records: 0,
        unique_offered_ordinals: 0,
        maximum_offered_ordinal: 0,
        transactions_begun: 0,
        transactions_ended: 0,
        included_transactions: 0,
        issue_count: 0,
        issues: vec![],
    };
    let mut hash = Sha256::new();
    let mut input_bytes = 0u64;
    let mut line_number = 0u64;
    let mut header = None;
    let mut footer = None;
    let mut identity_seen = false;
    let mut configured_budget = None;
    let mut active = BTreeMap::<u64, Transaction>::new();
    let mut transaction_ids = BTreeSet::new();
    let mut offered = vec![0u64; ((options.maximum_events + 63) / 64) as usize];
    let mut previous_retained = 0u64;
    let mut counters = BTreeMap::new();
    let mut metrics = BTreeMap::new();
    let mut pair_examples = Vec::new();
    loop {
        let mut line = Vec::new();
        // read_until alone has no bound. Consume chunks only up to a declared
        // maximum, including a newline, so a malformed line cannot grow memory.
        loop {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                break;
            }
            let take = available
                .iter()
                .position(|b| *b == b'\n')
                .map_or(available.len(), |i| i + 1);
            if line
                .len()
                .checked_add(take)
                .is_none_or(|n| n > options.maximum_line_bytes)
            {
                return Err(invalid("trace line exceeds audit byte bound"));
            }
            let newline = available[take - 1] == b'\n';
            line.extend_from_slice(&available[..take]);
            reader.consume(take);
            if newline {
                break;
            }
        }
        if line.is_empty() {
            break;
        }
        line_number += 1;
        input_bytes = input_bytes
            .checked_add(line.len() as u64)
            .ok_or_else(|| invalid("input length overflow"))?;
        hash.update(&line);
        if line.last() != Some(&b'\n') {
            integrity.issue(format!(
                "line {line_number}: incomplete newline-delimited record"
            ));
        }
        let v: serde_json::Value = match serde_json::from_slice(&line) {
            Ok(v) => v,
            Err(e) => {
                integrity.issue(format!("line {line_number}: JSON error {e}"));
                continue;
            }
        };
        if footer.is_some() {
            integrity.issue(format!("line {line_number}: data after footer"));
            continue;
        }
        match v.get("event").and_then(|v| v.as_str()) {
            Some("run_header") => {
                if line_number != 1 || header.is_some() {
                    integrity.issue("header not unique/first");
                    continue;
                }
                match serde_json::from_value::<Header>(v) {
                    Ok(h) => {
                        if h.schema != SCHEMA
                            || h.event != "run_header"
                            || h.authority != "diagnostic_only_not_profile_source_or_witness"
                        {
                            integrity.issue("unsupported header schema/authority");
                        }
                        header = Some(h);
                    }
                    Err(e) => integrity.issue(format!("invalid header: {e}")),
                }
            }
            Some("run_footer") => match serde_json::from_value::<Footer>(v) {
                Ok(f) => footer = Some(f),
                Err(e) => integrity.issue(format!("invalid footer: {e}")),
            },
            _ => {
                if header.is_none() {
                    integrity.issue("record before valid header");
                }
                let e: Envelope = match serde_json::from_value(v) {
                    Ok(e) => e,
                    Err(error) => {
                        integrity.issue(format!("line {line_number}: unsupported event: {error}"));
                        continue;
                    }
                };
                integrity.event_records += 1;
                if integrity.event_records > options.maximum_events {
                    return Err(invalid("trace exceeds audit event bound"));
                }
                if e.schema != SCHEMA {
                    integrity.issue("event schema mismatch");
                }
                if e.ordinal == 0 || e.ordinal > options.maximum_events {
                    return Err(invalid("offered ordinal exceeds audit bound"));
                }
                let bit = e.ordinal - 1;
                let word = &mut offered[(bit / 64) as usize];
                let mask = 1u64 << (bit % 64);
                if *word & mask != 0 {
                    integrity.issue(format!("duplicate offered ordinal {}", e.ordinal));
                } else {
                    *word |= mask;
                    integrity.unique_offered_ordinals += 1;
                }
                integrity.maximum_offered_ordinal =
                    integrity.maximum_offered_ordinal.max(e.ordinal);
                if previous_retained.checked_add(1) != Some(e.retained_ordinal) {
                    integrity.issue(format!(
                        "retained ordinal gap/order at {}",
                        e.retained_ordinal
                    ));
                }
                previous_retained = e.retained_ordinal;
                match e.event {
                    Event::RunIdentity(data) => {
                        if identity_seen || e.transaction != 0 || integrity.event_records != 1 {
                            integrity.issue("identity not unique/first or transaction nonzero");
                        }
                        identity_seen = true;
                        configured_budget = data
                            .pointer("/config/planner/max_planning_us")
                            .and_then(|v| v.as_u64())
                            .and_then(|v| v.checked_mul(1000));
                        if configured_budget.is_none() {
                            integrity.issue("run identity lacks planner budget");
                        }
                    }
                    Event::TransactionBegin => {
                        integrity.transactions_begun += 1;
                        if e.transaction == 0 || !transaction_ids.insert(e.transaction) {
                            integrity.issue("duplicate/zero transaction begin");
                            continue;
                        }
                        if active.len() >= options.maximum_active_transactions {
                            return Err(invalid("too many open transactions"));
                        }
                        active.insert(e.transaction, Transaction::new(e.transaction, e.ordinal));
                    }
                    event => {
                        let Some(tx) = active.get_mut(&e.transaction) else {
                            integrity
                                .issue(format!("event has no open transaction {}", e.transaction));
                            continue;
                        };
                        tx.maximum_ordinal = tx.maximum_ordinal.max(e.ordinal);
                        tx.event_count += 1;
                        if tx.event_count > options.maximum_transaction_events {
                            return Err(invalid("transaction exceeds audit event bound"));
                        }
                        let ended = matches!(event, Event::TransactionEnd(_));
                        tx.event(event);
                        if ended {
                            let tx = active.remove(&e.transaction).expect("active checked");
                            integrity.transactions_ended += 1;
                            for issue in &tx.issues {
                                integrity.issue(format!("transaction {}: {issue}", tx.id));
                            }
                            let selected =
                                options.through_transaction.is_none_or(|cut| tx.id <= cut)
                                    && options
                                        .before_ordinal
                                        .is_none_or(|cut| tx.maximum_ordinal < cut);
                            if !selected {
                                continue;
                            }
                            integrity.included_transactions += 1;
                            if let Some(pair) = tx.pair() {
                                let group = format!(
                                    "model_version={:?}/work_rows={}",
                                    pair.model_version, pair.work_rows
                                );
                                value(
                                    &mut metrics,
                                    format!("paired_search_projection_sum/{group}"),
                                    pair.search_projection_sum_ns,
                                );
                                value(
                                    &mut metrics,
                                    format!("paired_replay_projection_sum/{group}"),
                                    pair.replay_projection_sum_ns,
                                );
                                value(
                                    &mut metrics,
                                    format!("paired_first_replay_hard_remaining/{group}"),
                                    pair.first_replay_hard_remaining_ns,
                                );
                                if let Some(n) = pair.first_replay_optional_remaining_ns {
                                    value(
                                        &mut metrics,
                                        format!("paired_first_replay_optional_remaining/{group}"),
                                        n,
                                    );
                                }
                                count(
                                    &mut counters,
                                    "transactions_with_independent_replay_projection",
                                    1,
                                );
                                if pair_examples.len() < 16 {
                                    pair_examples.push(pair);
                                }
                            }
                            for (key, n) in tx.counters {
                                let aggregate = counters.entry(key).or_insert(0u64);
                                *aggregate = aggregate
                                    .checked_add(n)
                                    .ok_or_else(|| invalid("diagnostic counter overflow"))?;
                            }
                            for (key, values) in tx.metrics {
                                metrics.entry(key).or_insert_with(Vec::new).extend(values);
                            }
                            if counters.len() > 32768 || metrics.len() > 32768 {
                                return Err(invalid("diagnostic group cardinality exceeds bound"));
                            }
                        }
                    }
                }
            }
        }
    }
    if header.is_none() {
        integrity.issue("missing header");
    }
    if !identity_seen {
        integrity.issue("missing run identity");
    }
    for id in active.keys() {
        integrity.issue(format!("transaction {id}: missing end"));
    }
    if let Some(f) = footer {
        integrity.footer_requires_external_successful_close = f.requires_successful_close;
        if f.schema != SCHEMA || f.event != "run_footer" || !f.recording_complete {
            integrity.issue("footer does not attest complete recording");
        }
        let s = f.statistics;
        for field in [
            "contention",
            "capacity",
            "transaction_limit",
            "event_limit",
            "closed",
            "writer_failed",
            "allocation",
            "encoding",
        ] {
            if !s.lost.contains_key(field) {
                integrity.issue(format!("footer missing required loss counter {field}"));
            }
        }
        let total_loss = s.lost.values().try_fold(0u64, |a, b| a.checked_add(*b));
        if total_loss != Some(0) || s.first_lost_ordinal != 0 || s.last_lost_ordinal != 0 {
            integrity.issue(format!(
                "footer records lost offered ordinals: {:?}",
                s.lost
            ));
        }
        if s.counter_exhausted
            || s.writer_failed
            || s.first_error.is_some()
            || s.active_transactions != 0
            || s.abandoned_transactions != 0
        {
            integrity.issue("footer records writer/counter/active/abandoned failure");
        }
        if s.offered != s.accepted
            || s.accepted != s.written
            || s.written != integrity.event_records
            || s.offered != integrity.unique_offered_ordinals
            || s.offered != integrity.maximum_offered_ordinal
            || s.transactions != integrity.transactions_begun
            || s.transactions != integrity.transactions_ended
        {
            integrity.issue("footer totals do not match complete unique events/transactions");
        }
        if !f.requires_successful_close {
            integrity.issue("footer close requirement changed");
        }
    } else {
        integrity.issue("missing footer");
    }
    integrity.trace_content_complete = integrity.issue_count == 0;
    Ok(RequiredQueryAudit{schema:"ferrum.required-query-diagnostic-summary.v1",purpose:"diagnostic_only_not_slo_compliance_or_execution_authority",input_jsonl_sha256:format!("{:x}",hash.finalize()),input_bytes,run_id:header.map(|h|h.run_id),configured_planning_budget_ns:configured_budget,scope:options,integrity,counters,nanoseconds:metrics::distributions(metrics),replay_pair_examples:pair_examples,timing_interpretation:"Nearest-rank quantiles of original same-transaction pairs. Projection segments partition their own projection; scope wall values are inclusive and may nest. Never sum marginal quantiles. Remaining time is saturating original deadline minus original elapsed. No QueueLoss boundary is inferred; filters are externally supplied. Footer is written before flush/sync and cannot itself prove successful close."})
}

#[cfg(test)]
mod tests;
