//! Offline necessary-condition comparison only. External labels are provenance
//! claims, never reconstructed catalogue, qualification, or execution authority.
use super::{invalid, read_bounded_line, wire::*, RequiredQueryAudit};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Read},
};

const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CANDIDATES: usize = 256;
const MAX_ALGORITHMS: usize = 4096 / 11;
const MAX_COMPARISONS: u64 = 4_000_000;
const MAX_EXAMPLES: usize = 128;
const MAX_MISSING_EXAMPLES: usize = 16;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Axis {
    signature: [u8; 32],
    kind: u8,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Universe {
    revision: u32,
    workload_domain: [u8; 32],
    algorithms: Vec<Axis>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    model_version: u64,
    child: String,
    capture: String,
    #[serde(default)]
    source_header_sha256: Option<String>,
    #[serde(default)]
    universe_signature: Option<[u8; 32]>,
    universe: Universe,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputWire {
    schema: String,
    run_id: String,
    input_jsonl_sha256: String,
    candidates: Vec<Candidate>,
}
/// Constructed only by bounded import; contains no live scheduler types.
#[derive(Debug)]
pub struct UniverseComparisonInput {
    wire: InputWire,
    sha256: String,
}

fn hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_axes(axes: &[Axis]) -> io::Result<()> {
    if axes.len() > MAX_ALGORITHMS
        || axes.windows(2).any(|p| p[0] >= p[1])
        || axes.iter().any(|a| a.signature == [0; 32] || a.kind > 5)
    {
        return Err(invalid(
            "invalid, unsorted, duplicate, or over-capacity algorithm roster",
        ));
    }
    Ok(())
}
fn universe_signature(u: &Universe) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ferrum.declared-algorithm-universe.v1\0");
    hash.update(u.workload_domain);
    hash.update((u.algorithms.len() as u64).to_le_bytes());
    for axis in &u.algorithms {
        hash.update(axis.signature);
        hash.update([axis.kind]);
    }
    hash.finalize().into()
}
/// Hashes original input bytes. The copied wire rules preserve the declaration's
/// revision, domain, pair ordering and signature algorithm without importing it.
pub fn read_universe_comparison_input(reader: impl Read) -> io::Result<UniverseComparisonInput> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(invalid("universe input byte bound exceeded"));
    }
    let mut wire: InputWire = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
    if wire.schema != "ferrum.required-query-universe-input.v1"
        || wire.run_id.is_empty()
        || wire.run_id.len() > 256
        || !hex_digest(&wire.input_jsonl_sha256)
        || wire.candidates.len() > MAX_CANDIDATES
    {
        return Err(invalid(
            "invalid universe input identity or candidate bound",
        ));
    }
    for candidate in &mut wire.candidates {
        let u = &candidate.universe;
        validate_axes(&u.algorithms)?;
        if u.revision != 1
            || u.workload_domain == [0; 32]
            || u.algorithms.is_empty()
            || candidate.child.is_empty()
            || candidate.child.len() > 256
            || candidate.capture.is_empty()
            || candidate.capture.len() > 256
            || candidate
                .source_header_sha256
                .as_deref()
                .is_some_and(|s| !hex_digest(s))
        {
            return Err(invalid("invalid candidate universe or provenance label"));
        }
        // At most 372 axes, each at most four numeric coordinates: both original
        // coordinate bounds (<4096 and 11*algorithm_count<4096) follow directly.
        let signature = universe_signature(u);
        if candidate.universe_signature.is_some_and(|s| s != signature) {
            return Err(invalid("candidate universe signature mismatch"));
        }
        candidate.universe_signature = Some(signature);
    }
    Ok(UniverseComparisonInput {
        wire,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

#[derive(Debug, Serialize)]
pub struct UniverseComparison {
    schema: &'static str,
    interpretation: &'static str,
    input_manifest_sha256: String,
    input_jsonl_sha256: String,
    run_id: String,
    limits: ComparisonLimits,
    /// Candidate index in examples refers to this externally supplied table.
    candidates: Vec<Candidate>,
    constructed_queries: u64,
    query_evidence: BTreeMap<&'static str, u64>,
    candidate_constructed_comparisons: Vec<BTreeMap<&'static str, u64>>,
    examples: Vec<Example>,
    examples_truncated: bool,
}
#[derive(Debug, Serialize)]
struct ComparisonLimits {
    candidates: usize,
    algorithms: usize,
    comparisons: u64,
    examples: usize,
    missing_axis_examples: usize,
}
#[derive(Debug, Serialize)]
struct Example {
    transaction: u64,
    attempt: u64,
    alternative: usize,
    ordinal: u64,
    line: u64,
    snapshot_model_version: Option<u64>,
    phase: String,
    depth: usize,
    candidate_index: Option<usize>,
    result: &'static str,
    missing_algorithm_count: usize,
    missing_algorithms: Vec<Axis>,
    missing_algorithms_truncated: bool,
}
#[derive(Default)]
struct Totals {
    constructed: u64,
    evidence: BTreeMap<&'static str, u64>,
    candidates: BTreeMap<(usize, &'static str), u64>,
    examples: Vec<Example>,
    example_count: u64,
}
impl Totals {
    fn example(&mut self, value: Example) {
        self.example_count += 1;
        if self.examples.len() < MAX_EXAMPLES
            && !self.examples.iter().any(|old| {
                old.candidate_index == value.candidate_index
                    && old.result == value.result
                    && old.missing_algorithm_count == value.missing_algorithm_count
                    && old.missing_algorithms == value.missing_algorithms
            })
        {
            self.examples.push(value);
        }
    }
}
#[derive(Default)]
struct Tx {
    epoch: Option<u64>,
    // Only the active attempt's immutable snapshot epoch is used for a query.
    attempts: BTreeMap<u64, (Option<u64>, String, usize)>,
    maximum_ordinal: u64,
    events: usize,
    totals: Totals,
}
struct Query {
    axes: Option<Vec<Axis>>,
    domain: Option<[u8; 32]>,
    // None is old/missing evidence; Some(None) is an explicit unbound query.
    prebound: Option<Option<[u8; 32]>>,
}
impl Query {
    fn read(data: &Value) -> io::Result<Self> {
        let axes: Option<Vec<Axis>> = data
            .get("algorithm_axes")
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value(v.clone()).map_err(|e| invalid(e.to_string())))
            .transpose()?;
        if let Some(axes) = &axes {
            validate_axes(axes)?;
        }
        let domain = data
            .get("identity")
            .and_then(|v| v.get("physical_domain_signature"))
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value(v.clone()).map_err(|e| invalid(e.to_string())))
            .transpose()?;
        let prebound = data
            .get("prebound_algorithm_universe")
            .map(|v| serde_json::from_value(v.clone()).map_err(|e| invalid(e.to_string())))
            .transpose()?;
        Ok(Self {
            axes,
            domain,
            prebound,
        })
    }
    fn missing(&self) -> Option<&'static str> {
        if self.axes.is_none() {
            Some("missing_algorithm_roster")
        } else if self.domain.is_none() {
            Some("missing_physical_domain")
        } else if self.prebound.is_none() {
            Some("missing_prebound_evidence")
        } else {
            None
        }
    }
}

/// Second bounded pass after the original complete-content audit. No query work
/// vectors are decoded or rebuilt. Counts cover constructions (including those
/// never looked up), not failed lookups. Complete-transaction cuts match audit.
pub fn compare_query_universes(
    mut reader: impl BufRead,
    audit: &RequiredQueryAudit,
    input: UniverseComparisonInput,
) -> io::Result<UniverseComparison> {
    if !audit.integrity.trace_content_complete {
        return Err(invalid(
            "universe comparison requires complete original integrity audit",
        ));
    }
    if audit.run_id.as_deref() != Some(input.wire.run_id.as_str())
        || audit.input_jsonl_sha256 != input.wire.input_jsonl_sha256
    {
        return Err(invalid(
            "universe input run or original trace hash mismatch",
        ));
    }
    let mut candidates_by_epoch = BTreeMap::<u64, Vec<usize>>::new();
    for (index, candidate) in input.wire.candidates.iter().enumerate() {
        candidates_by_epoch
            .entry(candidate.model_version)
            .or_default()
            .push(index);
    }
    let mut hash = Sha256::new();
    let mut active = BTreeMap::<u64, Tx>::new();
    let mut total = Totals::default();
    let mut comparisons = 0u64;
    let mut events = 0u64;
    let mut line_number = 0u64;
    loop {
        let line = read_bounded_line(&mut reader, audit.scope.maximum_line_bytes)?;
        if line.is_empty() {
            break;
        }
        hash.update(&line);
        line_number += 1;
        if line_number > audit.scope.maximum_events + 2 {
            return Err(invalid("comparison record bound exceeded"));
        }
        let value: Value = serde_json::from_slice(&line).map_err(|e| invalid(e.to_string()))?;
        if matches!(
            value.get("event").and_then(Value::as_str),
            Some("run_header" | "run_footer")
        ) {
            continue;
        }
        let envelope: Envelope =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        events += 1;
        if events > audit.scope.maximum_events {
            return Err(invalid("comparison event bound exceeded"));
        }
        let id = envelope.transaction;
        if matches!(envelope.event, Event::RunIdentity(_)) {
            continue;
        }
        if matches!(envelope.event, Event::TransactionBegin) {
            if active.len() >= audit.scope.maximum_active_transactions
                || active.insert(id, Tx::default()).is_some()
            {
                return Err(invalid("comparison active transaction bound or duplicate"));
            }
        }
        let tx = active
            .get_mut(&id)
            .ok_or_else(|| invalid("comparison event without transaction"))?;
        tx.maximum_ordinal = tx.maximum_ordinal.max(envelope.ordinal);
        tx.events += 1;
        if tx.events > audit.scope.maximum_transaction_events {
            return Err(invalid("comparison transaction event bound exceeded"));
        }
        match envelope.event {
            Event::Snapshot {
                cost_model_version, ..
            } => tx.epoch = Some(cost_model_version),
            Event::AttemptBegin {
                attempt,
                phase,
                depth,
                ..
            } => {
                if phase.len() > 256 {
                    return Err(invalid("comparison attempt phase byte bound exceeded"));
                }
                tx.attempts.insert(attempt, (tx.epoch, phase, depth));
            }
            Event::AttemptEnd { attempt, .. } => {
                tx.attempts.remove(&attempt);
            }
            Event::QueryConstructed {
                attempt,
                alternative,
                input_unknown,
                ..
            } => {
                let (epoch, phase, depth) = tx
                    .attempts
                    .get(&attempt)
                    .ok_or_else(|| invalid("comparison query without attempt"))?;
                let query = Query::read(&value["data"])?;
                tx.totals.constructed += 1;
                let missing = if input_unknown.is_some() {
                    Some("input_unknown")
                } else if epoch.is_none() {
                    Some("missing_snapshot_epoch")
                } else {
                    query.missing()
                };
                let make_example =
                    |candidate_index, result, missing: Vec<Axis>, missing_count| Example {
                        transaction: id,
                        attempt,
                        alternative,
                        ordinal: envelope.ordinal,
                        line: line_number,
                        snapshot_model_version: *epoch,
                        phase: phase.clone(),
                        depth: *depth,
                        candidate_index,
                        result,
                        missing_algorithm_count: missing_count,
                        missing_algorithms_truncated: missing_count > missing.len(),
                        missing_algorithms: missing,
                    };
                if let Some(reason) = missing {
                    *tx.totals.evidence.entry(reason).or_default() += 1;
                    tx.totals.example(make_example(None, reason, vec![], 0));
                    continue;
                }
                let mut found = false;
                for &index in candidates_by_epoch
                    .get(&epoch.unwrap())
                    .into_iter()
                    .flatten()
                {
                    let candidate = &input.wire.candidates[index];
                    found = true;
                    comparisons += 1;
                    if comparisons > MAX_COMPARISONS {
                        return Err(invalid("universe comparison work bound exceeded"));
                    }
                    let mut missing = Vec::new();
                    let mut missing_count = 0;
                    let result = if query.domain != Some(candidate.universe.workload_domain) {
                        "physical_domain_mismatch"
                    } else if query
                        .prebound
                        .flatten()
                        .is_some_and(|p| Some(p) != candidate.universe_signature)
                    {
                        "prebound_universe_mismatch"
                    } else {
                        for axis in query.axes.as_ref().unwrap() {
                            if candidate.universe.algorithms.binary_search(axis).is_err() {
                                missing_count += 1;
                                if missing.len() < MAX_MISSING_EXAMPLES {
                                    missing.push(*axis);
                                }
                            }
                        }
                        if missing_count == 0 {
                            "no_missing_algorithms"
                        } else {
                            "missing_algorithms"
                        }
                    };
                    *tx.totals.candidates.entry((index, result)).or_default() += 1;
                    tx.totals
                        .example(make_example(Some(index), result, missing, missing_count));
                }
                *tx.totals
                    .evidence
                    .entry(if found {
                        "compared_same_epoch_candidates"
                    } else {
                        "no_same_epoch_candidate_evidence"
                    })
                    .or_default() += 1;
                if !found {
                    tx.totals.example(make_example(
                        None,
                        "no_same_epoch_candidate_evidence",
                        vec![],
                        0,
                    ));
                }
            }
            Event::TransactionEnd(_) => {
                let tx = active.remove(&id).unwrap();
                if audit.scope.through_transaction.is_none_or(|cut| id <= cut)
                    && audit
                        .scope
                        .before_ordinal
                        .is_none_or(|cut| tx.maximum_ordinal < cut)
                {
                    total.constructed += tx.totals.constructed;
                    for (key, value) in tx.totals.evidence {
                        *total.evidence.entry(key).or_default() += value;
                    }
                    for (key, value) in tx.totals.candidates {
                        *total.candidates.entry(key).or_default() += value;
                    }
                    total.example_count += tx.totals.example_count;
                    for example in tx.totals.examples {
                        if total.examples.len() < MAX_EXAMPLES
                            && !total.examples.iter().any(|old| {
                                old.candidate_index == example.candidate_index
                                    && old.result == example.result
                                    && old.missing_algorithm_count
                                        == example.missing_algorithm_count
                                    && old.missing_algorithms == example.missing_algorithms
                            })
                        {
                            total.examples.push(example);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if format!("{:x}", hash.finalize()) != audit.input_jsonl_sha256 || !active.is_empty() {
        return Err(invalid(
            "trace changed between integrity and comparison passes",
        ));
    }
    let mut candidate_counts = vec![BTreeMap::new(); input.wire.candidates.len()];
    for ((index, reason), count) in total.candidates {
        candidate_counts[index].insert(reason, count);
    }
    Ok(UniverseComparison {
        schema: "ferrum.required-query-universe-comparison.v1",
        interpretation: "External epoch/child/capture labels are provenance claims, not authority. Counts are QueryConstructed, including unqueried alternatives; they are not lookup-failure denominators. Epoch is the original AttemptBegin snapshot. Every same-epoch candidate is compared. Missing evidence is not coverage failure. Nonempty difference excludes only that U; empty difference excludes only missing algorithms, not other membership, host-policy, selection, qualification or adoption failures. No successful-close attestation is added.",
        input_manifest_sha256: input.sha256, input_jsonl_sha256: audit.input_jsonl_sha256.clone(), run_id: input.wire.run_id,
        limits: ComparisonLimits { candidates: MAX_CANDIDATES, algorithms: MAX_ALGORITHMS, comparisons: MAX_COMPARISONS, examples: MAX_EXAMPLES, missing_axis_examples: MAX_MISSING_EXAMPLES },
        candidates: input.wire.candidates, constructed_queries: total.constructed, query_evidence: total.evidence,
        candidate_constructed_comparisons: candidate_counts,
        examples_truncated: total.example_count > total.examples.len() as u64, examples: total.examples,
    })
}
