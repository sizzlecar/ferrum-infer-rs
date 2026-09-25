//! Read-only audit of the existing calibrate-slo JSONL diagnostic stream.
//! JSON never becomes a live receipt, source population or qualified input.
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

const MAX_RAW_BYTES: u64 = 1 << 30;
const MAX_LINE_BYTES: u64 = 8 << 20;
const MAX_WAVES: usize = 65_536;
const MAX_DOMAINS: usize = 16_384;

fn key(v: &Value) -> String {
    serde_json::to_string(v).expect("JSON value serialization")
}
fn increment(map: &mut BTreeMap<String, u64>, key: String) {
    *map.entry(key).or_default() += 1;
}
fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or_default()
}
fn label(v: &Value) -> String {
    v.as_str().map(str::to_owned).unwrap_or_else(|| key(v))
}
fn shape(v: &Value) -> &Value {
    &v["host_stages"]["actual_shape"]["exact"]
}
fn recipe(v: &Value) -> &Value {
    &v["host_stages"]["structured_evidence"]["Ok"]["recipe"]
}
fn algorithms(v: &Value) -> BTreeMap<String, Value> {
    array(&recipe(v)["device"]["algorithm_work"]["Ok"]["entries"])
        .iter()
        .map(|row| (key(&json!([row["algorithm"], row["kind"]])), row.clone()))
        .collect()
}
fn context(v: &Value, line: usize) -> Value {
    json!({"line":line,"phase":v["phase"],"case":v["case"],"call_id":v["host_stages"]["call_id"],
        "submission":v["submission"],"evidence_kind":v["evidence"]["kind"],"evidence_reason":v["evidence"]["reason"],
        "error":v["error"],"host_completeness":v["host_stages"]["completeness"],
        "settlement_error":v["host_stages"]["structured_evidence"]["Err"],
        "graph":shape(v)["graph_state"],"kind":shape(v)["kind"],
        "decode_kv_tokens":shape(v)["decode_kv_tokens"],"prefill_chunks":shape(v)["prefill_chunks"],
        "numeric_rows":v["host_stages"]["actual_shape"]["numeric_features"]["rows"],
        "physical_host_rows":recipe(v)["physical_host_rows"],"replay_work":recipe(v)["device"]["replay_work"]})
}

#[derive(Serialize)]
struct Domain {
    owner: Value,
    domain_signature: Value,
    waves: u64,
    first_line: usize,
    last_line: usize,
    basis_axes: usize,
    support_axes: usize,
    pending_length_counts: BTreeMap<String, u64>,
    graph_states: BTreeMap<String, u64>,
    cases: BTreeMap<String, u64>,
    first: Value,
    last: Value,
}
#[derive(Default, Serialize)]
struct Group {
    waves: u64,
    domains: BTreeSet<String>,
    templates: BTreeSet<String>,
    algorithms: BTreeSet<String>,
    policies: BTreeSet<String>,
}
#[derive(Default)]
struct SummaryReplay {
    domains: BTreeMap<String, (String, usize, usize, BTreeSet<(usize, usize)>)>,
    coordinates: usize,
    joint_pairs: usize,
    retained: usize,
    omitted: BTreeMap<String, u64>,
}
impl SummaryReplay {
    fn observe(&mut self, d: &Value, pending: usize, length: usize) -> Result<()> {
        let k = key(&d["domain_signature"]);
        let n = array(&d["basis"]).len();
        let m = array(&d["support"]).len();
        let owner = key(&d["owner"]);
        if !self.domains.contains_key(&k) {
            let next = self
                .coordinates
                .checked_add(n)
                .and_then(|a| a.checked_add(m))
                .context("axis overflow")?;
            let domains_full = self.domains.len() == 128;
            let coordinates_full = next > 65_536;
            if domains_full || coordinates_full {
                increment(&mut self.omitted, format!("new_domain:domain_limit={domains_full},coordinate_limit={coordinates_full}"));
                return Ok(());
            }
            self.coordinates = next;
            self.domains
                .insert(k.clone(), (owner.clone(), n, m, BTreeSet::new()));
        }
        let (old_owner, old_n, old_m, pairs) = self.domains.get_mut(&k).unwrap();
        if old_owner != &owner || *old_n != n || *old_m != m {
            increment(&mut self.omitted, "identity_or_axis_mismatch".into());
            return Ok(());
        }
        if !pairs.contains(&(pending, length)) {
            if self.joint_pairs == 16_384 {
                increment(&mut self.omitted, "joint_pair_limit".into());
                return Ok(());
            }
            pairs.insert((pending, length));
            self.joint_pairs += 1;
        }
        self.retained += 1;
        Ok(())
    }
}

fn audit(raw: &Path, report_path: &Path) -> Result<Value> {
    ensure!(
        raw.metadata()?.len() <= MAX_RAW_BYTES,
        "raw file exceeds bound"
    );
    ensure!(
        report_path.metadata()?.len() <= 32 << 20,
        "report exceeds bound"
    );
    let report_bytes = std::fs::read(report_path)?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure!(
        report["kind"] == "real_manual_calibration",
        "unexpected report kind"
    );
    let mut input = BufReader::new(File::open(raw)?);
    let mut bytes = Vec::new();
    let mut hash = Sha256::new();
    let mut line = 0usize;
    let mut raw_bytes = 0u64;
    let mut wave_count = 0usize;
    let mut phases = BTreeMap::new();
    let mut unknown = BTreeMap::new();
    let mut unknown_examples = BTreeMap::<String, Value>::new();
    let mut domains = BTreeMap::<String, Domain>::new();
    let mut groups = BTreeMap::<String, Group>::new();
    let mut prior = BTreeMap::<String, (Value, BTreeMap<String, Value>, Value)>::new();
    let mut transitions = Vec::new();
    let mut transition_counts = BTreeMap::new();
    let mut summary = SummaryReplay::default();
    let mut request_records = Vec::new();
    let mut terminal_records = Vec::new();
    loop {
        bytes.clear();
        let read = (&mut input)
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        line += 1;
        raw_bytes += read as u64;
        ensure!(
            read as u64 <= MAX_LINE_BYTES && bytes.last() == Some(&b'\n'),
            "oversize or incomplete raw line {line}"
        );
        ensure!(raw_bytes <= MAX_RAW_BYTES, "raw grew beyond bound");
        hash.update(&bytes);
        let v: Value =
            serde_json::from_slice(&bytes).with_context(|| format!("JSON at line {line}"))?;
        if v["event"] == "request" || v["event"] == "request_evidence" {
            request_records.push(v.clone());
        }
        if v["event"] == "request_completed" {
            terminal_records.push(v.clone());
        }
        if v["event"] != "wave" {
            continue;
        }
        wave_count += 1;
        ensure!(wave_count <= MAX_WAVES, "wave population exceeds bound");
        let d = &v["structured_cost_discovery_v2"];
        let phase = label(&v["phase"]);
        increment(
            &mut phases,
            key(&json!([phase, v["submission"], d["status"], d["reason"]])),
        );
        if phase != "discovery" {
            continue;
        }
        match d["status"].as_str() {
            Some("unknown") => {
                let detail = json!({"reason":d["reason"],"submission":v["submission"],"legacy":v["evidence"]["reason"],
                    "completeness":v["host_stages"]["completeness"],"settlement":v["host_stages"]["structured_evidence"]["Err"],
                    "graph":shape(&v)["graph_state"],"kind":shape(&v)["kind"]});
                let k = key(&detail);
                increment(&mut unknown, k.clone());
                unknown_examples.entry(k).or_insert_with(|| json!({"diagnostic":detail,"context":context(&v,line),"host_stages":v["host_stages"]}));
            }
            Some("known") => {
                let rows = array(&d["physical_host_rows"]);
                let p = rows
                    .iter()
                    .filter(|r| r["pending_decoded_utf8"] == true)
                    .count();
                let l = rows
                    .iter()
                    .filter(|r| r["terminal_expectation"] == "length_boundary")
                    .count();
                summary.observe(d, p, l)?;
                let k = key(&d["domain_signature"]);
                ensure!(
                    domains.contains_key(&k) || domains.len() < MAX_DOMAINS,
                    "offline domain bound exceeded; no partial success"
                );
                let info = context(&v, line);
                let domain = domains.entry(k.clone()).or_insert_with(|| Domain {
                    owner: d["owner"].clone(),
                    domain_signature: d["domain_signature"].clone(),
                    waves: 0,
                    first_line: line,
                    last_line: line,
                    basis_axes: array(&d["basis"]).len(),
                    support_axes: array(&d["support"]).len(),
                    pending_length_counts: BTreeMap::new(),
                    graph_states: BTreeMap::new(),
                    cases: BTreeMap::new(),
                    first: info.clone(),
                    last: info.clone(),
                });
                ensure!(
                    domain.owner == d["owner"]
                        && domain.basis_axes == array(&d["basis"]).len()
                        && domain.support_axes == array(&d["support"]).len(),
                    "domain identity changed at line {line}"
                );
                domain.waves += 1;
                domain.last_line = line;
                domain.last = info.clone();
                increment(&mut domain.pending_length_counts, format!("{p},{l}"));
                increment(&mut domain.graph_states, label(&shape(&v)["graph_state"]));
                increment(&mut domain.cases, key(&v["case"]));
                let owner = &d["owner"];
                let g = key(&json!([
                    owner["rows"],
                    owner["role"],
                    owner["product"],
                    owner["readback"]
                ]));
                let group = groups.entry(g.clone()).or_default();
                group.waves += 1;
                group.domains.insert(k);
                group.templates.insert(key(&owner["provider_template"]));
                group.algorithms.insert(key(&owner["algorithm_domain"]));
                group.policies.insert(key(&owner["installed_policy"]));
                let current_algorithms = algorithms(&v);
                if let Some((old_owner, old_algorithms, old_info)) = prior.get(&g) {
                    let changed = ["provider_template", "algorithm_domain", "installed_policy"]
                        .into_iter()
                        .filter(|f| old_owner[*f] != owner[*f])
                        .collect::<Vec<_>>();
                    if !changed.is_empty() {
                        increment(&mut transition_counts, key(&json!([g, changed])));
                        if transitions.len() < 32 {
                            let removed = old_algorithms
                                .iter()
                                .filter(|(k, _)| !current_algorithms.contains_key(*k))
                                .map(|(_, v)| v)
                                .collect::<Vec<_>>();
                            let added = current_algorithms
                                .iter()
                                .filter(|(k, _)| !old_algorithms.contains_key(*k))
                                .map(|(_, v)| v)
                                .collect::<Vec<_>>();
                            transitions.push(json!({"group":g,"changed":changed,"before":old_info,"after":info,"removed_algorithms":removed,"added_algorithms":added}));
                        }
                    }
                }
                prior.insert(g, (owner.clone(), current_algorithms, info));
            }
            _ => bail!("missing/unsupported discovery status at line {line}"),
        }
    }
    let raw_sha = format!("{:x}", hash.finalize());
    ensure!(
        report["raw_sha256"] == raw_sha && report["raw_bytes"] == raw_bytes,
        "raw identity differs from original report"
    );
    let official = &report["summary"]["structured_discovery_v2"];
    let known: u64 = domains.values().map(|v| v.waves).sum();
    let unknown_count: u64 = unknown.values().sum();
    let omitted: u64 = summary.omitted.values().sum();
    ensure!(
        official["known_input_waves"] == known
            && official["unknown_input_reports"] == unknown_count
            && official["wave_reports_seen"] == known + unknown_count,
        "original summary counts differ"
    );
    ensure!(
        official["omitted_known_waves"] == omitted
            && array(&official["domains"]).len() == summary.domains.len(),
        "bounded-summary replay differs"
    );
    let mut frequency = BTreeMap::new();
    for d in domains.values() {
        increment(&mut frequency, d.waves.to_string());
    }
    let groups = groups.into_iter().map(|(k,v)| (k,json!({"waves":v.waves,"domains":v.domains.len(),"templates":v.templates.len(),"algorithm_domains":v.algorithms.len(),"installed_policies":v.policies.len()}))).collect::<BTreeMap<_,_>>();
    Ok(
        json!({"scope":"read-only diagnostic JSON audit; no live receipts, training membership, qualified model or future-demand completeness",
        "raw_sha256":raw_sha,"raw_bytes":raw_bytes,"report_sha256":format!("{:x}",Sha256::digest(&report_bytes)),
        "phases":phases,"discovery":{"known":known,"unknown":unknown_count,"domains":domains.len(),"domain_frequency":frequency,"groups":groups},
        "bounded_summary_replay":{"domains_retained":summary.domains.len(),"coordinates_retained":summary.coordinates,"joint_pairs_retained":summary.joint_pairs,"waves_retained":summary.retained,"omitted":summary.omitted},
        "unknown_groups":unknown,"unknown_examples":unknown_examples,"identity_transition_counts":transition_counts,"first_identity_transitions":transitions,
        "request_records":request_records,"output_records":terminal_records,"domains":domains}),
    )
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3,
        "usage: structured_discovery_audit RAW.jsonl REPORT.json OUT.json"
    );
    let result = audit(Path::new(&args[0]), Path::new(&args[1]))?;
    let bytes = serde_json::to_vec_pretty(&result)?;
    ensure!(bytes.len() <= 128 << 20, "audit output exceeds bound");
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    out.write_all(&bytes)?;
    out.write_all(b"\n")?;
    out.flush()?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"discovery":result["discovery"],"bounded_summary_replay":result["bounded_summary_replay"],"unknown_groups":result["unknown_groups"]})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_domains_still_update_after_new_domain_capacity_is_exhausted() {
        let mut s = SummaryReplay::default();
        let row = |id, n| json!({"domain_signature":id,"owner":{"rows":1},"basis":vec![0;n],"support":[]});
        let first = row(0, 1024);
        for id in 0..64 {
            s.observe(&row(id, 1024), 0, 0).unwrap();
        }
        s.observe(&row(64, 1024), 0, 0).unwrap();
        s.observe(&first, 0, 1).unwrap();
        assert_eq!(s.retained, 65);
        assert_eq!(s.coordinates, 65_536);
        assert_eq!(s.omitted.values().sum::<u64>(), 1);
        assert_eq!(s.joint_pairs, 65);
    }
}
