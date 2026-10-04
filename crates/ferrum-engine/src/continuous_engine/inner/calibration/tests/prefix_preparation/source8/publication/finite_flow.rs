//! The finite production plan drives real CPU cohorts and the original source8
//! collector. Independent journal replay and an ordinary adopted wave are
//! separate checks; declared phase labels do not establish numerical phases.
use super::*;
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};
use ferrum_scheduler::implementations::continuous::cost_profile::{
    self as file, StructuredPreparedOwnerBlockCollectorV8 as Collector,
    StructuredPreparedOwnerBlockHeaderV8 as Header, StructuredPreparedOwnerBlockRecordV8 as Record,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrum-source8-finite-flow-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// These are the existing hash-bound opaque manifest envelopes, not a second
// scheduler or a reconstruction of any collected cost sample.
fn manifest_with<'a>(value: &'a Value, field: &str) -> Option<&'a Value> {
    if value.get(field).is_some() {
        return Some(value);
    }
    ["child", "parent"].into_iter().find_map(|name| {
        value
            .get(name)
            .and_then(|child| manifest_with(child, field))
    })
}

fn assert_finite_manifest(header: &Header) -> usize {
    let source: Value =
        serde_json::from_str(header.declaration.cohort_manifest_payload.get()).unwrap();
    let source_index = source["source"].as_u64().unwrap() as usize;
    let selected = &manifest_with(&source, "checked_selection").unwrap()["checked_selection"];
    let batch = selected["batches"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|batch| batch["scheduled"] == true)
        .nth(source_index)
        .unwrap();
    assert_eq!(batch["input_plan"]["kind"], "finite");
    let occurrences = batch["input_plan"]["plan"]["occurrence_case_indices"]
        .as_array()
        .unwrap();
    assert!(!occurrences.is_empty());
    let range = &source["original_range"];
    let start = range["start"].as_u64().unwrap() as usize;
    let end = range["end"].as_u64().unwrap() as usize;
    assert_eq!(
        &selected["execution_case_indices"].as_array().unwrap()[start..end],
        occurrences
    );
    assert_eq!(source["schedule"], batch["schedule"]);
    assert_eq!(
        source["schedule"],
        serde_json::to_value(&header.declaration.population.schedule).unwrap()
    );

    let cases = manifest_with(&source, "cases").unwrap()["cases"]
        .as_array()
        .unwrap();
    let cohorts = source["cohorts"].as_array().unwrap();
    let declared: Vec<_> = header
        .declaration
        .cohort_plan
        .phases
        .iter()
        .flatten()
        .collect();
    assert_eq!(cohorts.len(), occurrences.len());
    assert_eq!(declared.len(), occurrences.len());
    let mut seeds = BTreeSet::new();
    for ((index, actual), declared) in occurrences.iter().zip(cohorts).zip(declared) {
        let original = &cases[index.as_u64().unwrap() as usize];
        for field in [
            "width",
            "maximum_output",
            "suffix_tokens",
            "preset",
            "prefix",
            "route",
        ] {
            assert_eq!(
                actual[field], original[field],
                "finite occurrence changed {field}"
            );
        }
        assert_eq!(actual["reset_token_policy"], original["reset"]);
        assert!(
            seeds.insert(actual["seed"].as_u64().unwrap()),
            "each occurrence needs fresh request seeds"
        );
        assert_eq!(
            declared.requests.len(),
            actual["width"].as_u64().unwrap() as usize
        );
        assert!(declared.requests.iter().all(|request| {
            request.maximum_output == actual["maximum_output"].as_u64().unwrap()
        }));
    }
    occurrences.len()
}

#[tokio::test]
async fn source8_finite_production_plan_replays_independent_phases_and_adopts_ordinary_wave() {
    let directory = Directory::new();
    let diagnostics = ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
        directory: directory.0.clone(),
        maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
        maximum_retained_generations: NonZeroUsize::new(8).unwrap(),
    };
    let (mut session, executor) = automatic_session_with_diagnostics(
        2,
        Arc::new(AdvancingClock(AtomicU64::new(100))),
        diagnostics,
    )
    .await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let mut seed_request = request(&session, 3);
    seed_request.sampling_params.top_k =
        Some(session.engine.inner.model_executor.info().vocab_size);
    seed_request.sampling_params.stop_sequences.clear();
    let mut ordinary_request = seed_request.clone();
    ordinary_request.prompt = "ok".into();
    assert_ne!(ordinary_request.prompt, seed_request.prompt);
    assert_eq!(
        session
            .engine
            .inner
            .tokenizer
            .encode(&ordinary_request.prompt, true)
            .unwrap()
            .len(),
        1
    );
    let template =
        AutomaticCostProbeTemplate::new(seed_request, AutomaticCostProbeOutput::CliText).unwrap();
    let ordinary =
        AutomaticCostProbeTemplate::new(ordinary_request, AutomaticCostProbeOutput::CliText)
            .unwrap();
    let settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    let prepared = session
        .prepare_startup_cost(&settings, &[template])
        .await
        .unwrap();
    assert!(runtime.snapshot().is_none());
    assert!(
        session
            .collect_prepared_startup_cost(&settings, prepared)
            .await
            .unwrap()
            > 0
    );
    assert!(session.prepared_owner_capture.is_none());
    assert!(session.frontiers().unwrap().is_empty());
    let children = runtime.startup_series_children_for_test().unwrap();
    assert!(!children.is_empty());
    let numerical =
        crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(&settings);
    for child in &children {
        let phases = &child.provenance().phases;
        assert_eq!(phases.len(), 3);
        assert!(
            phases[0].members
                >= numerical
                    .min_phase_samples
                    .max(numerical.max_rank + numerical.min_fit_redundancy)
        );
        assert!(phases[1..]
            .iter()
            .all(|phase| phase.members >= numerical.min_phase_samples));
    }

    let mut matched = vec![false; children.len()];
    let mut cohort_count = 0;
    let mut archives = 0;
    // Directory is the existing optional original-source sink. Only completed
    // source.jsonl artifacts are consumed, including every footer event.
    for entry in fs::read_dir(directory.0.join("ferrum-automatic-v1")).unwrap() {
        let path = entry.unwrap().path().join("source.jsonl");
        if !path.is_file() {
            continue;
        }
        archives += 1;
        let bytes = fs::read(&path).unwrap();
        let mut lines = bytes.split_inclusive(|byte| *byte == b'\n');
        let first = lines.next().unwrap();
        let header: Header = serde_json::from_slice(first).unwrap();
        cohort_count += assert_finite_manifest(&header);
        let capture_identity = header.capture_identity;
        let limits = CostProfileLoadLimits::default();
        let mut collector = Collector::new(header, limits.clone()).unwrap();
        let mut end = first.len();
        for line in lines {
            let record: Record = serde_json::from_slice(line).unwrap();
            collector.push(&record).unwrap();
            end += line.len();
            let wire: Value = serde_json::from_slice(line).unwrap();
            if wire["kind"] != "checkpoint" {
                continue;
            }
            let digest: [u8; 32] = Sha256::digest(&bytes[..end]).into();
            let original: Vec<_> = children
                .iter()
                .enumerate()
                .filter(|(_, child)| {
                    child.provenance().capture_identity == capture_identity
                        && child.provenance().source_sha256 == digest
                })
                .collect();
            if original.is_empty() {
                continue;
            }
            let replayed = file::replay_structured_source_v8(&bytes[..end], &limits).unwrap();
            assert_eq!(replayed.source_receipt(), (end as u64, digest));
            let restored = replayed
                .activate_same_process_memory(
                    file::StructuredServiceClockV7 {
                        monotonic_ns: runtime.clock.now_ns().unwrap(),
                        wall_unix_ns: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos()
                            .try_into()
                            .unwrap(),
                    },
                    &limits,
                )
                .unwrap();
            for (index, child) in original {
                let replayed = restored
                    .children
                    .iter()
                    .find(|candidate| candidate.domain_signature() == child.domain_signature())
                    .expect("independent replay lost an actually installed child");
                assert_eq!(
                    replayed.parameters_signature(),
                    child.parameters_signature()
                );
                assert_eq!(
                    replayed.provenance().source_sha256,
                    child.provenance().source_sha256
                );
                assert_eq!(
                    serde_json::to_value(&replayed.provenance().phases).unwrap(),
                    serde_json::to_value(&child.provenance().phases).unwrap()
                );
                matched[index] = true;
            }
        }
        assert!(collector.audit().closed);
        assert!(!collector.audit().poisoned);
        assert_eq!(
            collector.source_receipt(),
            (bytes.len() as u64, Sha256::digest(&bytes).into())
        );
    }
    assert!(archives > 0 && cohort_count > 0);
    assert!(
        matched.iter().all(|matched| *matched),
        "every installed child must replay from its original finite source"
    );

    // Fresh ordinary CLI requests retain the installed model, but none of the
    // startup cohort/prefix authorities. Prepare two real tokens, then make
    // the original controller independently select and adopt terminal Decode.
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for seed in 0..2 {
        let (request, contract) = ordinary
            .instantiate(
                NonZeroUsize::new(3).unwrap(),
                seed,
                ferrum_types::SloAutomaticCostProbeSamplingPresetV1::Configured,
            )
            .unwrap();
        ids.push(request.id.clone());
        let mut output = session
            .add_request(request, InferenceRequestContext::capture(), contract)
            .await
            .unwrap();
        consumers.push(tokio::spawn(async move {
            let mut terminal = false;
            while let Some(frame) = output.frames.next().await {
                terminal |= frame.metadata().terminal;
                drop(frame);
            }
            assert!(terminal);
            assert!(matches!(
                output.completion.await.unwrap().payload(),
                OutputCompletion::Succeeded {
                    reason: ferrum_types::FinishReason::Length,
                    ..
                }
            ));
        }));
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    for _ in &ids {
        admit(&mut session).await;
    }
    for prefill in [true, false] {
        for id in &ids {
            ready(&session, id, false).await;
        }
        let rows = ids
            .iter()
            .map(|id| {
                let row = frontier(&session, id);
                if prefill {
                    row.prefill_work(NonZeroU32::MIN).unwrap()
                } else {
                    row.decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                        .unwrap()
                }
            })
            .collect();
        let report = wave(&mut session, &executor, rows).await;
        assert!(report.error.is_none(), "{report:?}");
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    future::submit_terminal_witness(&session, &executor, &ids).await;
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    session.shutdown().await.unwrap();
}
