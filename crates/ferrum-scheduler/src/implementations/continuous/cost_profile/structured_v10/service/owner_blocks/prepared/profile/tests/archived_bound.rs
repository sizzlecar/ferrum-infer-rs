//! Offline audit of an original checkpoint and a same-call observed query.
//! No live receipt is created; all qualification still comes from source replay.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::prepared as input_replay;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};

mod empirical_cells;
mod online;
mod seeded;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditSpec {
    checkpoint_source: PathBuf,
    checkpoint_bytes: usize,
    checkpoint_sha256: [u8; 32],
    parameters_sha256: [u8; 32],
    actual_source: PathBuf,
    actual_source_sha256: [u8; 32],
    query_journal: PathBuf,
    pairs: Vec<Pair>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    transaction: u64,
    attempt: u64,
    alternative: u64,
    call_id: u64,
}

#[test]
#[ignore = "requires original source8/source7/query artifacts via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_archived_source8_issued_bounds_without_refitting_or_clock_refresh() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").expect("audit spec path"))
            .unwrap(),
    )
    .unwrap();
    let limits = CostProfileLoadLimits::default();
    assert!(spec.checkpoint_bytes <= limits.max_file_bytes.get());
    let mut bytes = Vec::new();
    std::fs::File::open(&spec.checkpoint_source)
        .unwrap()
        .take(spec.checkpoint_bytes as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len(), spec.checkpoint_bytes);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&bytes)),
        spec.checkpoint_sha256
    );
    let h = header(&bytes);
    let checkpoint = replay_structured_source_v8(&bytes, &limits).unwrap();
    let original_closing = checkpoint.population.closing.clone();
    let catalog = checkpoint
        .activate_same_process_memory(original_closing, &limits)
        .unwrap();
    let child = catalog
        .children
        .iter()
        .find(|child| child.parameters_signature() == spec.parameters_sha256)
        .expect("original installed parameters from independently replayed populations");

    let mut wave_by_call = std::collections::BTreeMap::new();
    let mut actual_hash = Sha256::new();
    let mut actual_header = None;
    for (index, line) in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .split(b'\n')
        .enumerate()
    {
        let mut line = line.unwrap();
        assert!(line.len() < 8 * 1024 * 1024);
        line.push(b'\n');
        actual_hash.update(&line);
        if index == 0 {
            actual_header =
                Some(serde_json::from_slice::<StructuredServiceHeaderV7>(&line).unwrap());
            continue;
        }
        let value: Value = serde_json::from_slice(&line).unwrap();
        if value["kind"] != "completed" {
            continue;
        }
        let call = value["wave"]["host_stages"]["call_id"].as_u64().unwrap();
        if spec.pairs.iter().any(|pair| pair.call_id == call) {
            let StructuredServiceRecordV7::Completed { wave } =
                serde_json::from_slice(&line).unwrap()
            else {
                unreachable!()
            };
            assert!(wave_by_call.insert(call, wave).is_none());
        }
    }
    assert_eq!(
        <[u8; 32]>::from(actual_hash.finalize()),
        spec.actual_source_sha256
    );
    let actual_header = actual_header.unwrap();
    assert_eq!(actual_header.fingerprint, h.fingerprint);
    let mut demands = std::collections::BTreeMap::new();
    let mut lookups = std::collections::BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(&spec.query_journal).unwrap()).lines() {
        let line = line.unwrap();
        assert!(line.len() <= 1024 * 1024);
        let value: Value = serde_json::from_str(&line).unwrap();
        let Some(pair) = spec.pairs.iter().find(|pair| {
            value["transaction"] == pair.transaction
                && value["data"]["attempt"] == pair.attempt
                && value["data"]["alternative"] == pair.alternative
        }) else {
            continue;
        };
        match value["event"].as_str() {
            Some("query_constructed") => {
                assert!(demands
                    .insert(pair.transaction, value["data"]["demand"].clone())
                    .is_none());
            }
            Some("query_lookup") => {
                assert!(lookups
                    .insert(pair.transaction, value["data"].clone())
                    .is_none());
            }
            _ => {}
        }
    }
    let contract = h
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap();
    for pair in &spec.pairs {
        let wave = &wave_by_call[&pair.call_id];
        // Validate the complete original actual shape, recipe, host settlement
        // and time boundary before comparing any numerical projection.
        let (_, actual_ns, _) = physical::validate_parts(
            &h.fingerprint,
            actual_header.opening.monotonic_ns,
            Some(contract),
            actual_header.opening.monotonic_ns,
            wave.ticket,
            wave.fifo,
            wave.issued_at_ns,
            &wave.host_stages,
            wave.independent.as_ref(),
            &mut physical::Frontiers::default(),
        )
        .unwrap();
        let (prepared, offered) =
            physical::original_prepared(&wave.host_stages, wave.independent.as_ref()).unwrap();
        let raw = input_replay::project_service_actual_with_domain(
            &prepared,
            &offered,
            &contract.workload_domain,
        )
        .unwrap();
        let query = StructuredQueryV2::exact(raw);
        // This audit supports only a query whose entire public demand is the
        // same as the actual input. It cannot replace a pending/terminal upper
        // forecast with its cheaper realized outcome.
        assert_eq!(
            serde_json::to_value(query.required_coverage().unwrap()).unwrap(),
            demands[&pair.transaction],
            "same-call demand differs; do not reinterpret its issued forecast"
        );
        let lookup = &lookups[&pair.transaction];
        assert_eq!(lookup["outcome"]["kind"], "known");
        let original_now = lookup["cost_now_ns"].as_u64().unwrap();
        let prediction = child
            .predict_query_local(child.fingerprint(), &query, original_now)
            .unwrap();
        assert_eq!(
            prediction.fitted_upper_ns,
            lookup["outcome"]["cost"]["typical_ns"].as_u64().unwrap()
        );
        assert_eq!(
            prediction.planning_ns,
            lookup["outcome"]["cost"]["planning_ns"].as_u64().unwrap()
        );
        eprintln!(
            "ARCHIVED_BOUND {}",
            json!({
                "transaction": pair.transaction, "call_id": pair.call_id,
                "original_model_now_ns": original_now, "actual_ns": actual_ns,
                "prediction": {
                    "fitted_upper_ns": prediction.fitted_upper_ns,
                    "planning_ns": prediction.planning_ns,
                    "valid_until_ns": prediction.valid_until_ns,
                    "fit_samples": prediction.fit_samples,
                    "residual_samples": prediction.residual_samples,
                    "identified_rank": prediction.identified_rank,
                }, "uncertainty": child.model.uncertainty(),
                "decomposition": child.model.diagnose_archived_bound(&query),
                "source_sha256": catalog.source_sha256,
                "parameters_sha256": child.parameters_signature(),
            })
        );
    }
}

mod expiry;
mod rolling;
