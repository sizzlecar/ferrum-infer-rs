//! Offline, bounded reader for one original external SSE-client invocation per
//! run. This draft does not implement a CLI, network client, or formal gate.
use crate::paired_cluster as core;
use serde_json::Value;

mod io;
mod projection;
mod types;
mod wire;
use io::{digest, hex_digest, parse, semantic_eq, Reader};
pub use types::*;
use ReaderIssueKind as Kind;

const SCHEMA: &str = "ferrum_external_sse_comparison_v2";

pub fn read_pairs(manifest: &Manifest, spec: core::Spec, limits: ReaderLimits) -> Audit {
    let mut audit = Audit {
        runs: Vec::new(),
        identity_issues: Vec::new(),
        report: None,
        candidate_observed_fail: false,
        method_status: "DRAFT_READER_NOT_FORMAL_ACCEPTANCE",
    };
    if manifest
        .pairs
        .len()
        .checked_mul(2)
        .is_none_or(|n| n > limits.max_runs)
        || manifest.requests.len() > limits.max_requests_per_run
        || manifest.requests.is_empty()
        || limits.max_line_bytes > limits.max_artifact_bytes
        || limits.max_issues == 0
        || limits.max_raw_failures == 0
    {
        audit.identity_issues.push(Kind::Budget);
        return audit;
    }
    let mut reader = Reader::new(limits);
    for binding in [&manifest.cell, &manifest.inputs] {
        if let Err(e) = verify_digest(&mut reader, binding) {
            audit.identity_issues.push(e);
        }
    }
    if manifest.cell.expected != manifest.workload.cell
        || manifest.inputs.expected != manifest.workload.inputs
        || manifest.workload.requests.len() != manifest.requests.len()
        || manifest
            .workload
            .requests
            .iter()
            .zip(&manifest.requests)
            .enumerate()
            .any(|(i, (a, b))| a != &b.key || a.index != i)
        || manifest.pairs.len() != manifest.workload.repeat_ids.len()
    {
        audit.identity_issues.push(Kind::Input);
    }
    for pair in &manifest.pairs {
        let reference = &manifest.pairs[0].baseline.policy;
        // Output destinations/provenance are pinned separately, while the
        // actual request policy must agree across arms and repetitions.
        if [&pair.baseline.policy, &pair.candidate.policy]
            .iter()
            .any(|p| {
                p.model != reference.model
                    || p.protocol != reference.protocol
                    || p.deadline_ms != reference.deadline_ms
                    || p.temperature != reference.temperature
                    || !semantic_eq(&p.request_policy, &reference.request_policy)
            })
        {
            audit.identity_issues.push(Kind::Input);
        }
        if spec.absolute.request_deadline != std::time::Duration::from_millis(reference.deadline_ms)
        {
            audit.identity_issues.push(Kind::Input);
        }
        let baseline = read_run_inner(&mut reader, manifest, &pair.baseline);
        let candidate = read_run_inner(&mut reader, manifest, &pair.candidate);
        audit.candidate_observed_fail |= candidate.matched_failures > 0
            || candidate
                .run
                .as_ref()
                .and_then(|r| r.client_exit)
                .is_some_and(|x| x != 0)
            || candidate
                .raw_failures
                .iter()
                .any(|f| f.kind == RawFailureKind::ClientExit);
        audit.runs.push(baseline);
        audit.runs.push(candidate);
    }
    if audit.identity_issues.is_empty() && audit.runs.iter().all(|r| r.run.is_some()) {
        let pairs: Vec<_> = audit
            .runs
            .chunks_exact(2)
            .map(|r| core::RunPair {
                baseline: r[0].run.as_ref().unwrap().clone(),
                candidate: r[1].run.as_ref().unwrap().clone(),
            })
            .collect();
        audit.report = Some(core::analyze(&manifest.workload, &pairs, spec));
    }
    audit
}

/// Standalone inspection still verifies identity receipts. The result's `run`
/// is unavailable on structural/source mismatches, but raw failures survive.
pub fn read_run(manifest: &Manifest, run: &RunManifest, limits: ReaderLimits) -> RunAudit {
    let mut reader = Reader::new(limits);
    if manifest.requests.len() > limits.max_requests_per_run
        || manifest.requests.is_empty()
        || limits.max_line_bytes > limits.max_artifact_bytes
        || limits.max_issues == 0
        || limits.max_raw_failures == 0
    {
        let mut log = Log::new(limits);
        log.issue(Kind::Budget, None, true);
        return log.finish(None, None);
    }
    let mut audit = read_run_inner(&mut reader, manifest, run);
    if manifest.cell.expected != manifest.workload.cell
        || manifest.inputs.expected != manifest.workload.inputs
        || manifest.workload.requests.len() != manifest.requests.len()
        || manifest
            .workload
            .requests
            .iter()
            .zip(&manifest.requests)
            .enumerate()
            .any(|(i, (a, b))| a != &b.key || a.index != i)
    {
        audit.run = None;
        if audit.issues.len() < limits.max_issues {
            audit.issues.push(ReaderIssue {
                kind: Kind::Input,
                index: None,
                blocks_pairing: true,
            });
        } else {
            audit.issues_omitted += 1;
        }
    }
    for binding in [&manifest.cell, &manifest.inputs] {
        if let Err(kind) = verify_digest(&mut reader, binding) {
            audit.run = None;
            if audit.issues.len() < limits.max_issues {
                audit.issues.push(ReaderIssue {
                    kind,
                    index: None,
                    blocks_pairing: true,
                });
            } else {
                audit.issues_omitted += 1;
            }
        }
    }
    audit
}

struct Log {
    audit: RunAudit,
    blocked: bool,
    limits: ReaderLimits,
}
impl Log {
    fn new(limits: ReaderLimits) -> Self {
        Self {
            audit: RunAudit {
                run: None,
                observations: Vec::new(),
                issues: Vec::new(),
                issues_omitted: 0,
                raw_failures: Vec::new(),
                raw_failures_omitted: 0,
                observed_failures: 0,
                matched_failures: 0,
                observed_timeouts: 0,
                source_summary: None,
            },
            blocked: false,
            limits,
        }
    }
    fn issue(&mut self, kind: Kind, index: Option<usize>, blocks_pairing: bool) {
        self.blocked |= blocks_pairing;
        if self.audit.issues.len() < self.limits.max_issues {
            self.audit.issues.push(ReaderIssue {
                kind,
                index,
                blocks_pairing,
            });
        } else {
            self.audit.issues_omitted += 1;
        }
    }
    fn failure(&mut self, index: Option<usize>, kind: RawFailureKind) {
        self.audit.observed_failures += 1;
        self.audit.observed_timeouts += usize::from(kind == RawFailureKind::Timeout);
        if self.audit.raw_failures.len() < self.limits.max_raw_failures {
            self.audit.raw_failures.push(RawFailure {
                index,
                kind,
                matched_input: false,
            });
        } else {
            self.audit.raw_failures_omitted += 1;
        }
    }
    fn take<T>(&mut self, value: Result<T, Kind>) -> Option<T> {
        match value {
            Ok(x) => Some(x),
            Err(e) => {
                self.issue(e, None, true);
                None
            }
        }
    }
    fn finish(mut self, run: Option<core::Run>, summary: Option<Value>) -> RunAudit {
        self.audit.run = if self.blocked { None } else { run };
        self.audit.source_summary = summary;
        self.audit
    }
}

fn verify_digest(reader: &mut Reader, b: &DigestEvidence) -> Result<(), Kind> {
    let v: Value = reader.json(&b.artifact)?;
    if v.pointer(&b.pointer).and_then(hex_digest) != Some(b.expected) {
        return Err(Kind::Identity);
    }
    Ok(())
}
fn verify_build(reader: &mut Reader, b: &BuildEvidence) -> Result<(), Kind> {
    let v: Value = reader.json(&b.artifact)?;
    if b.source_identity.is_empty()
        || v.pointer(&b.source_pointer).and_then(Value::as_str) != Some(b.source_identity.as_str())
        || v.pointer(&b.binary_pointer).and_then(hex_digest) != Some(b.binary_sha256)
    {
        return Err(Kind::Identity);
    }
    Ok(())
}

fn server_identity(reader: &mut Reader, run: &RunManifest) -> Result<core::Digest, Kind> {
    let bytes = reader.artifact(&run.monitor)?;
    let line = bytes.split(|b| *b == b'\n').next().ok_or(Kind::Identity)?;
    if line.len() > reader.limits.max_line_bytes {
        return Err(Kind::Budget);
    }
    let v: Value = parse(line)?;
    let (pid, canonical) = match &run.server {
        ServerIdentity::Linux {
            boot_id,
            pid,
            start_ticks,
        } => {
            if boot_id.len() != 36
                || !boot_id.bytes().enumerate().all(|(i, b)| {
                    if [8, 13, 18, 23].contains(&i) {
                        b == b'-'
                    } else {
                        b.is_ascii_hexdigit() && !b.is_ascii_uppercase()
                    }
                })
                || *pid == 0
                || *start_ticks == 0
            {
                return Err(Kind::Identity);
            }
            (
                *pid,
                format!("linux:boot={boot_id}:pid={pid}:start_ticks={start_ticks}"),
            )
        }
        ServerIdentity::MacOs {
            pid,
            start_abstime,
            uuid,
        } => {
            if *pid == 0 || *start_abstime == 0 {
                return Err(Kind::Identity);
            }
            (
                *pid,
                format!("macos:pid={pid}:start_abstime={start_abstime}:uuid={uuid:02x?}"),
            )
        }
    };
    if v["event"] != "monitor_start"
        || v["pid"].as_u64() != Some(u64::from(pid))
        || v["incarnation"].as_str() != Some(&canonical)
        || v["initial_host"]["pid"].as_u64() != Some(u64::from(pid))
        || v["initial_host"]["incarnation"].as_str() != Some(&canonical)
    {
        return Err(Kind::Identity);
    }
    // New, explicitly versioned semantic incarnation identity. This is NOT an
    // existing receipt/file digest and does not alter frozen workload hashes.
    let mut hash_input = b"ferrum-paired-reader/server-incarnation/v1\0".to_vec();
    hash_input.extend_from_slice(canonical.as_bytes());
    Ok(digest(&hash_input))
}

fn read_run_inner(reader: &mut Reader, manifest: &Manifest, run: &RunManifest) -> RunAudit {
    let mut log = Log::new(reader.limits);
    let n = manifest.requests.len();
    // Independent reads: a broken summary/build does not hide raw failures.
    let config: Option<wire::Config> = log.take(reader.json(&run.config));
    let summary: Option<wire::Summary> = log.take(reader.json(&run.summary));
    let identity = log.take(server_identity(reader, run));
    log.take(verify_build(reader, &run.server_build));
    log.take(verify_build(reader, &run.client_build));
    let exit =
        log.take(reader.artifact(&run.client_exit)).and_then(|b| {
            match std::str::from_utf8(&b)
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
            {
                Some(n) => Some(n),
                None => {
                    log.issue(Kind::Exit, None, true);
                    None
                }
            }
        });
    if exit.is_some_and(|n| n != 0) {
        log.failure(None, RawFailureKind::ClientExit);
    }
    if let Some(config) = &config {
        validate_config(config, run, manifest, &mut log);
    }
    if let Some(summary) = &summary {
        if summary.schema != SCHEMA
            || summary.model != run.policy.model
            || summary.protocol != run.policy.protocol
            || summary.request_deadline_ms != run.policy.deadline_ms
            || !semantic_eq(&summary.request_policy, &run.policy.request_policy)
            || !semantic_eq(&summary.provenance, &run.policy.provenance)
        {
            log.issue(Kind::Summary, None, true);
        }
        if !run.epoch_unix_ms.is_finite()
            || run.epoch_unix_ms <= 0.0
            || summary.epoch_unix_ms != run.epoch_unix_ms
        {
            log.issue(Kind::Epoch, None, true);
        }
    }
    let mut rows: Vec<Option<core::Request>> = (0..n).map(|_| None).collect();
    let mut occurrences = vec![0usize; n];
    let mut failed = vec![false; n];
    let mut events_budget = reader.limits.max_events_per_run;
    let mut raw_successes = 0usize;
    let mut raw_errors = 0usize;
    let mut raw_timeouts = 0usize;
    let mut lines = 0usize;
    if let Some(bytes) = log.take(reader.artifact(&run.records)) {
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            if lines == reader.limits.max_requests_per_run
                || line.len() > reader.limits.max_line_bytes
            {
                log.issue(Kind::Budget, None, true);
                break;
            }
            lines += 1;
            // This independent projection retains explicit failures even when
            // another field makes the full row unparsable/unpairable.
            let header = serde_json::from_slice::<projection::Projection>(line).ok();
            // Even a row with a malformed timing/source field occupies its
            // stated index. A later valid row must not resolve that ambiguity.
            if let Some(i) = header.as_ref().and_then(|h| h.index()).filter(|i| *i < n) {
                occurrences[i] += 1;
                if occurrences[i] > 1 {
                    rows[i] = None;
                    log.issue(Kind::DuplicateIndex, Some(i), true);
                }
            }
            let failure = header.as_ref().and_then(|h| h.failure());
            let explicit_failure = failure.is_some();
            if let Some(kind) = failure {
                let h = header.as_ref().unwrap();
                log.failure(h.index(), kind);
                if let Some(i) = h
                    .index()
                    .filter(|i| *i < n && occurrences[*i] == 1 && h.matches(&manifest.requests[*i]))
                {
                    failed[i] = true;
                    // Identity is independent of timing. A clear, matched
                    // failure survives full-row parsing failure with unknown
                    // terminal/usage/events; nothing is synthesized.
                    let mut r = missing(&manifest.requests[i].key);
                    r.outcome = core::Outcome::Failed(if kind == RawFailureKind::Timeout {
                        core::FailureKind::Timeout
                    } else {
                        core::FailureKind::Transport
                    });
                    rows[i] = Some(r);
                }
            }
            raw_successes += usize::from(header.as_ref().and_then(|h| h.success()) == Some(true));
            raw_errors += usize::from(header.as_ref().and_then(|h| h.success()) == Some(false));
            raw_timeouts += usize::from(header.as_ref().and_then(|h| h.timed_out()) == Some(true));
            let record: wire::Record = match parse(line) {
                Ok(r) => r,
                Err(e) => {
                    log.issue(e, header.and_then(|h| h.index()), true);
                    continue;
                }
            };
            let i = record.index;
            if i >= n {
                log.issue(Kind::UnexpectedIndex, Some(i), true);
                continue;
            }
            if occurrences[i] > 1 {
                rows[i] = None;
                continue;
            }
            let expected = &manifest.requests[i];
            if record.scenario != expected.scenario
                || record.scheduled_ms != expected.key.scheduled_ms
                || record.requested_max_tokens != expected.key.output_cap
                || !semantic_eq(&record.source, &expected.source)
            {
                log.issue(Kind::Source, Some(i), true);
                continue;
            }
            let evidence_size = [
                &record.visible_event_ms,
                &record.visible_event_utf8_bytes,
                &record.visible_event_transport_chunk,
                &record.itl_sse_text_event_ms,
            ]
            .iter()
            .try_fold(0usize, |a, v| a.checked_add(v.as_ref().map_or(0, Vec::len)));
            match evidence_size {
                Some(count) if count <= events_budget => events_budget -= count,
                _ => {
                    log.issue(Kind::Budget, Some(i), true);
                    continue;
                }
            }
            let (mapped, invalid_protocol) = map_record(&record, &expected.key, &mut log);
            let past_deadline = mapped.terminal_ms.is_some_and(|t| {
                t - expected.key.scheduled_ms as f64 > run.policy.deadline_ms as f64
            });
            failed[i] = explicit_failure || invalid_protocol || past_deadline;
            if invalid_protocol && !explicit_failure {
                log.failure(Some(i), RawFailureKind::InvalidProtocol);
            } else if past_deadline && !explicit_failure {
                log.failure(Some(i), RawFailureKind::DeadlineExceeded);
            }
            rows[i] = Some(mapped);
        }
    }
    log.audit.matched_failures = failed
        .iter()
        .enumerate()
        .filter(|(i, f)| **f && occurrences[*i] == 1)
        .count();
    for failure in &mut log.audit.raw_failures {
        failure.matched_input = failure
            .index
            .is_some_and(|i| i < n && failed[i] && occurrences[i] == 1);
    }
    let requests: Vec<_> = rows
        .into_iter()
        .enumerate()
        .map(|(i, r)| r.unwrap_or_else(|| missing(&manifest.requests[i].key)))
        .collect();
    if let Some(summary) = &summary {
        let o = &summary.overall;
        if o["requests"].as_u64() != Some(lines as u64)
            || o["successes"].as_u64() != Some(raw_successes as u64)
            || o["errors"].as_u64() != Some(raw_errors as u64)
            || o["timeouts"].as_u64() != Some(raw_timeouts as u64)
        {
            log.issue(Kind::Summary, None, true);
        }
        // Missing expected rows are retained, never silently accepted because
        // the original summary only counted the rows it happened to receive.
        if requests.iter().all(|r| r.outcome != core::Outcome::Missing) && !log.blocked {
            reconcile_summary(o, &requests, &mut log);
        }
    }
    let core_run = identity.map(|server_incarnation| core::Run {
        repeat_id: run.repeat_id,
        cell: manifest.workload.cell,
        inputs: manifest.workload.inputs,
        server_incarnation,
        client_exit: exit,
        requests: requests.clone(),
    });
    log.audit.observations = requests;
    log.finish(core_run, summary.map(|s| s.overall))
}

fn validate_config(config: &wire::Config, run: &RunManifest, manifest: &Manifest, log: &mut Log) {
    let p = &run.policy;
    if config.model != p.model
        || config.protocol != p.protocol
        || config.deadline_ms != p.deadline_ms
        || !p.temperature.is_finite()
        || config.temperature != p.temperature
        || !semantic_eq(&config.request_policy, &p.request_policy)
        || !semantic_eq(&config.provenance, &p.provenance)
        || config.requests.len() != manifest.requests.len()
        || !matches!(p.protocol.as_str(), "chat_completions" | "completions")
    {
        log.issue(Kind::Input, None, true);
    }
    for (i, (r, e)) in config.requests.iter().zip(&manifest.requests).enumerate() {
        if e.key.index != i
            || e.key.output_cap == 0
            || e.key.output_cap > u64::from(u32::MAX)
            || e.key.scheduled_ms.checked_add(p.deadline_ms).is_none()
            || r.scenario != e.scenario
            || r.max_tokens != e.key.output_cap
            || r.offset_ms != e.key.scheduled_ms
            || digest(r.prompt.as_bytes()) != e.key.prompt
            || !semantic_eq(&r.source, &e.source)
            || e.source
                .pointer(&e.source_digest_pointer)
                .and_then(hex_digest)
                != Some(e.key.source)
        {
            log.issue(Kind::Input, Some(i), true);
        }
    }
}

fn missing(key: &core::RequestKey) -> core::Request {
    core::Request {
        key: key.clone(),
        outcome: core::Outcome::Missing,
        usage: None,
        finish: None,
        done: false,
        terminal_ms: None,
        ttft_from_scheduled_ms: None,
        visible_span_ms: None,
        visible_text_events: None,
        visible_gaps_ms: Vec::new(),
    }
}

/// Only roundoff tolerance, scaled to the arithmetic being checked; this is
/// not a latency/SLO tolerance and cannot excuse an absent measurement.
fn same(a: f64, b: f64) -> bool {
    same_scaled(a, b, 1.0)
}
fn same_scaled(a: f64, b: f64, operand_scale: f64) -> bool {
    a.is_finite()
        && b.is_finite()
        && operand_scale.is_finite()
        && (a - b).abs()
            <= 16.0 * f64::EPSILON * a.abs().max(b.abs()).max(operand_scale.abs()).max(1.0)
}
fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite() && *x >= 0.0)
}

fn map_record(r: &wire::Record, key: &core::RequestKey, log: &mut Log) -> (core::Request, bool) {
    let i = Some(r.index);
    let usage = match (r.prompt_tokens, r.completion_tokens, r.total_tokens) {
        (Some(prompt), Some(completion), Some(total)) => Some(core::Usage {
            prompt,
            completion,
            total,
        }),
        _ => None,
    };
    let usage_valid = usage.is_some_and(|u| {
        u.prompt.checked_add(u.completion) == Some(u.total)
            && u.completion <= key.output_cap
            && r.usage.as_ref().is_some_and(|v| {
                v["prompt_tokens"].as_u64() == Some(u.prompt)
                    && v["completion_tokens"].as_u64() == Some(u.completion)
                    && v["total_tokens"].as_u64() == Some(u.total)
            })
    });
    if r.success && !usage_valid {
        log.issue(Kind::Usage, i, false);
    }
    let finish = match r.finish_reason.as_deref() {
        Some("stop") => Some(core::Finish::Stop),
        Some("length") => Some(core::Finish::Length),
        _ => None,
    };
    let invalid_protocol = r.success
        && (!usage_valid
            || !r.done
            || finish.is_none()
            || r.timed_out
            || r.error.is_some()
            || !r.http_status.is_some_and(|s| (200..300).contains(&s))
            || !r
                .content_type
                .as_deref()
                .is_some_and(|s| s.starts_with("text/event-stream")));
    let outcome = if invalid_protocol {
        core::Outcome::Failed(core::FailureKind::InvalidOutput)
    } else if r.timed_out {
        core::Outcome::Failed(core::FailureKind::Timeout)
    } else if !r.success || r.error.is_some() {
        core::Outcome::Failed(core::FailureKind::Transport)
    } else {
        core::Outcome::Success
    };
    let terminal = (r.start_ms.is_finite()
        && r.start_ms >= key.scheduled_ms as f64
        && r.finish_ms.is_finite()
        && r.finish_ms >= r.start_ms)
        .then_some(r.finish_ms);
    if terminal.is_none() {
        log.issue(Kind::Timing, i, false);
    }
    let mut gaps = Vec::new();
    let mut count = None;
    let mut first = None;
    let mut last = None;
    let mut complete = false;
    if let (Some(t), Some(bytes), Some(chunks)) = (
        &r.visible_event_ms,
        &r.visible_event_utf8_bytes,
        &r.visible_event_transport_chunk,
    ) {
        if t.len() == bytes.len() && t.len() == chunks.len() {
            let text = r.text.as_ref().and_then(Value::as_str);
            let transport = r.transport_chunks.as_ref().and_then(Value::as_u64);
            let sse_events = r.all_sse_events.as_ref().and_then(Value::as_u64);
            let mut byte_offset = Some(0usize);
            let mut valid_bytes = Vec::with_capacity(bytes.len());
            for value in bytes {
                let extent = byte_offset
                    .zip(
                        value
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .filter(|n| *n > 0),
                    )
                    .and_then(|(start, len)| start.checked_add(len).map(|end| (start, end)));
                let valid = extent.is_some_and(|(start, end)| {
                    text.is_some_and(|text| {
                        end <= text.len()
                            && text.is_char_boundary(start)
                            && text.is_char_boundary(end)
                    })
                });
                valid_bytes.push(valid);
                // An invalid length, extent or UTF8 boundary destroys the prefix.
                // Later events cannot restart it at a guessed byte position.
                byte_offset = extent.filter(|_| valid).map(|(_, end)| end);
            }
            let text_complete = text.is_some_and(|text| byte_offset == Some(text.len()))
                && valid_bytes.iter().all(|valid| *valid);
            let counts_consistent = sse_events
                .is_some_and(|n| u64::try_from(t.len()).is_ok_and(|len| len <= n))
                && transport.is_some();
            let valid_event = |j: usize| -> Option<(f64, u64)> {
                let time = finite(&t[j])?;
                let chunk = chunks[j].as_u64()?;
                if !valid_bytes[j]
                    || chunk == 0
                    || !transport.is_some_and(|n| chunk <= n)
                    || sse_events.is_some_and(|n| j as u64 >= n)
                    || terminal.is_some() && time > r.finish_ms - r.start_ms
                {
                    return None;
                }
                Some((time, chunk))
            };
            // Retained capture redundancies establish the population independently
            // of timestamps. They do not prove authenticity beyond the pinned
            // raw artifact. Contradictions keep locally verified prefix gaps only.
            if text_complete
                && counts_consistent
                && chunks.iter().all(|v| {
                    v.as_u64()
                        .is_some_and(|c| c > 0 && transport.is_some_and(|n| c <= n))
                })
            {
                count = Some(t.len());
            }
            complete = count.is_some() && (0..t.len()).all(|j| valid_event(j).is_some());
            for j in 1..t.len() {
                let pair = valid_event(j - 1).zip(valid_event(j));
                let gap = pair.and_then(|((a, ca), (b, cb))| {
                    (b >= a && cb >= ca && (ca != cb || same(a, b))).then_some(b - a)
                });
                let stored = r
                    .itl_sse_text_event_ms
                    .as_ref()
                    .and_then(|v| v.get(j - 1))
                    .and_then(finite);
                if let Some(g) = gap.filter(|g| stored.is_some_and(|s| same(*g, s))) {
                    gaps.push(g);
                } else {
                    complete = false;
                }
            }
            if r.itl_sse_text_event_ms
                .as_ref()
                .is_none_or(|v| v.len() != t.len().saturating_sub(1))
            {
                complete = false;
                // Retain valid adjacent gaps but do not conceal a contradictory
                // or absent interval vector behind an apparently complete DTO.
                count = None;
            }
            first = (!t.is_empty())
                .then(|| valid_event(0).map(|x| x.0))
                .flatten();
            last = (!t.is_empty())
                .then(|| valid_event(t.len() - 1).map(|x| x.0))
                .flatten();
        }
    }
    if !complete {
        log.issue(Kind::Events, i, false);
    }
    // Scalar first/last timestamps are independently emitted by this client.
    // Missing event vectors do not erase otherwise consistent TTFT/TPOT.
    let first_clock = r.ttft_ms.filter(|f| {
        f.is_finite()
            && *f >= 0.0
            && r.start_ms.is_finite()
            && r.start_ms >= key.scheduled_ms as f64
            && r.first_visible_ms.is_some_and(|v| same(v, r.start_ms + f))
            && first.is_none_or(|actual| same(actual, *f))
    });
    let ttft = first_clock.and_then(|f| {
        r.ttft_from_scheduled_ms.filter(|v| {
            *v >= 0.0
                && same_scaled(
                    *v,
                    r.start_ms - key.scheduled_ms as f64 + f,
                    r.start_ms.abs().max(key.scheduled_ms as f64),
                )
        })
    });
    let span = first_clock.zip(r.last_visible_ms).and_then(|(f, l)| {
        let relative = l - r.start_ms;
        (relative.is_finite()
            && relative >= f
            && last.is_none_or(|actual| same(actual, relative))
            && terminal.is_none_or(|end| l <= end))
        .then_some(first.zip(last).map_or(relative - f, |(a, b)| b - a))
    });
    let span = span.filter(|s| match usage {
        Some(u) if u.completion > 1 => r.tpot_last_visible_usage_ms.is_some_and(|v| {
            same_scaled(
                v,
                *s / (u.completion - 1) as f64,
                r.last_visible_ms
                    .unwrap_or(r.start_ms)
                    .abs()
                    .max(r.start_ms.abs())
                    / (u.completion - 1) as f64,
            )
        }),
        _ => true,
    });
    if first.is_some() && ttft.is_none() || first.zip(last).is_some() && span.is_none() {
        log.issue(Kind::Timing, i, false);
    }
    (
        core::Request {
            key: key.clone(),
            outcome,
            usage,
            finish,
            done: r.done,
            terminal_ms: terminal,
            ttft_from_scheduled_ms: ttft,
            visible_span_ms: span,
            visible_text_events: count,
            visible_gaps_ms: gaps,
        },
        invalid_protocol,
    )
}

fn reconcile_summary(s: &Value, requests: &[core::Request], log: &mut Log) {
    let successful: Vec<_> = requests
        .iter()
        .filter(|r| r.outcome == core::Outcome::Success)
        .collect();
    let tokens = successful
        .iter()
        .try_fold(0u64, |n, r| n.checked_add(r.usage?.completion));
    let start = requests
        .iter()
        .map(|r| r.key.scheduled_ms)
        .min()
        .unwrap_or(0) as f64;
    let end = requests
        .iter()
        .try_fold(start, |end, r| Some(end.max(r.terminal_ms?)));
    if let (Some(tokens), Some(end)) = (tokens, end) {
        let seconds = (end - start) / 1000.0;
        if s["successful_output_tokens_known"].as_u64() != Some(tokens)
            || !s["throughput_denominator_scheduled_start_to_last_terminal_seconds"]
                .as_f64()
                .is_some_and(|v| same(v, seconds))
            || seconds > 0.0
                && !s["output_tok_s_successful_usage"]
                    .as_f64()
                    .is_some_and(|v| same(v, tokens as f64 / seconds))
        {
            log.issue(Kind::Summary, None, true);
        }
    }
    // Quantiles are recomputed by the core. Source distributions remain in the
    // audit; complete populations are compared without inventing missing rows.
    for (field, values, complete) in [
        (
            "ttft_ms_successful",
            successful
                .iter()
                .filter_map(|r| r.ttft_from_scheduled_ms)
                .collect::<Vec<_>>(),
            successful
                .iter()
                .all(|r| r.ttft_from_scheduled_ms.is_some()),
        ),
        (
            "tpot_last_visible_usage_ms_successful",
            successful
                .iter()
                .filter_map(|r| {
                    let u = r.usage?;
                    (u.completion > 1)
                        .then(|| r.visible_span_ms.map(|s| s / (u.completion - 1) as f64))
                        .flatten()
                })
                .collect(),
            successful.iter().all(|r| {
                r.usage
                    .is_some_and(|u| u.completion < 2 || r.visible_span_ms.is_some())
            }),
        ),
        (
            "itl_sse_text_event_ms_all_observed",
            requests
                .iter()
                .flat_map(|r| r.visible_gaps_ms.iter().copied())
                .collect(),
            requests.iter().all(|r| {
                r.visible_text_events
                    .is_some_and(|n| r.visible_gaps_ms.len() == n.saturating_sub(1))
            }),
        ),
    ] {
        // Partial evidence has an intentionally smaller descriptive population.
        // It cannot authenticate the source's claimed complete distribution.
        if s[field]["count"].as_u64() != Some(values.len() as u64) {
            if complete && !(values.is_empty() && s[field].is_null()) {
                log.issue(Kind::Summary, None, true);
            }
            continue;
        }
        if !matches_distribution(&s[field], values) {
            log.issue(Kind::Summary, None, true);
        }
    }
}

fn matches_distribution(s: &Value, mut values: Vec<f64>) -> bool {
    if values.is_empty() {
        return s.is_null() || s["count"].as_u64() == Some(0);
    }
    values.sort_unstable_by(f64::total_cmp);
    [("p50", 50usize), ("p95", 95), ("p99", 99)]
        .iter()
        .all(|(field, p)| {
            let Some(k) = values
                .len()
                .checked_mul(*p)
                .and_then(|v| v.checked_add(99))
                .map(|v| v / 100)
            else {
                return false;
            };
            s[*field].as_f64().is_some_and(|v| same(v, values[k - 1]))
        })
}

#[cfg(test)]
mod tests;
