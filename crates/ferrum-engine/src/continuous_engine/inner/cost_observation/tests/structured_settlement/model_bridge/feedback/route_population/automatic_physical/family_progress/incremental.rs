//! Real CPU calls queue an original block before the worker runs. Each worker
//! turn must consume feedback and append one record, never postpone conversion
//! until the final offer. Only the original complete block can freeze a phase.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2;
use sha2::{Digest, Sha256};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-owner-block-incremental-{}",
            uuid::Uuid::new_v4()
        )))
    }
    fn policy(&self) -> SloAutomaticCalibrationDiagnosticsV1 {
        SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: self.0.clone(),
            maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn queue(f: &mut Families, count: usize) {
    let before = f.runtime.audit_snapshot().sink.raw_resolved;
    for _ in 0..count {
        let stages = fixture::record_cohort_route(&f.runtime, &f.clock, Families::wave(A))
            .expect("real CPU submit and original complete settlement");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        f.recorded += 1;
    }
    assert_eq!(f.runtime.audit_snapshot().sink.raw_resolved, before);
}

#[tokio::test]
async fn automatic_owner_block_incremental_ingest_precedes_close_and_replays() {
    let directory = Directory::new();
    let mut f = Families::new_with_diagnostics(directory.policy());
    let started = std::time::Instant::now();
    let mut worker_elapsed = std::time::Duration::ZERO;
    let mut worker_turns = 0;
    let phases = [
        Some(StructuredPhaseV2::Fit),
        Some(StructuredPhaseV2::Residual),
        Some(StructuredPhaseV2::Qualification),
        None,
    ];
    for block in 0..4 {
        f.runtime.consume_samples();
        queue(&mut f, OFFERS);
        for n in 1..=OFFERS {
            let turn = std::time::Instant::now();
            f.runtime.consume_samples();
            worker_elapsed += turn.elapsed();
            worker_turns += 1;
            let expected = (block * OFFERS + n) as u64;
            let sink = f.runtime.audit_snapshot().sink;
            assert_eq!(
                sink.raw_resolved, expected,
                "one original FIFO per worker turn"
            );
            assert_eq!(sink.raw_pending, (OFFERS - n) as u64);
            assert_eq!(sink.raw_resolution_failed, 0);
            assert_eq!(sink.raw_lost, 0);
            let live = f.live().audit();
            let audit = live
                .automatic
                .as_ref()
                .unwrap()
                .owner_blocks
                .as_ref()
                .unwrap();
            assert_eq!(
                audit.offered, expected,
                "collector must progress before block close"
            );
            assert_eq!(audit.last_fifo, expected);
            assert_eq!(audit.block_offered, n);
            assert!(
                !audit.closed,
                "generation remains open until its original footer"
            );
            assert!(!audit.poisoned);
            assert_eq!(
                live.qualified_publications,
                u64::from(block == 3 && n == OFFERS)
            );
            if n == OFFERS {
                assert_eq!(audit.owners.len(), 1);
                assert_eq!(audit.owners[0].phase, phases[block]);
            } else if block != 0 {
                assert_eq!(audit.owners[0].phase, phases[block - 1]);
            }
        }
    }
    assert!(f.known(A));
    let mut children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(children.len(), 1);
    let child = children.pop().unwrap();
    assert_eq!(
        child.provenance().phases.each_ref().map(|p| p.members),
        [OFFERS; 3]
    );
    let original = child
        .predict_query_local(child.fingerprint(), &f.query(A), f.clock.now_ns().unwrap())
        .unwrap();
    f.runtime.shutdown().await.unwrap();
    let audit = f.live().audit();
    assert_eq!(audit.failed_generations, 0);
    let automatic = audit.automatic.as_ref().unwrap();
    assert_eq!(automatic.diagnostic_failures.count, 0);
    let receipt = &automatic
        .last_diagnostic_publication
        .as_ref()
        .unwrap()
        .source;
    let bytes = fs::read(&receipt.path).unwrap();
    assert_eq!(receipt.bytes, bytes.len() as u64);
    assert_eq!(receipt.digest, <[u8; 32]>::from(Sha256::digest(&bytes)));
    let mut end = 0;
    let mut checkpoint_end = None;
    let mut tickets = Vec::new();
    let mut closes = 0;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        end += line.len();
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        match value["kind"].as_str() {
            Some("completed") => tickets.push(value["wave"]["ticket"].as_u64().unwrap()),
            Some("block_close") => closes += 1,
            Some("checkpoint") => checkpoint_end = Some(end),
            Some("failed") => panic!("complete original block failed"),
            _ => {}
        }
    }
    assert_eq!(tickets, (1..=f.recorded).collect::<Vec<_>>());
    assert_eq!(closes, 4);
    let prefix = &bytes[..checkpoint_end.unwrap()];
    let limits = file::CostProfileLoadLimits::default();
    let replayed = file::replay_structured_source_v7(prefix, &limits).unwrap();
    assert_eq!(
        replayed.source_receipt(),
        (prefix.len() as u64, Sha256::digest(prefix).into())
    );
    assert_eq!(replayed.qualified_children(), 1);
    let activation = crate::continuous_engine::inner::cost_observation::profile_export::ExportClockReading::closing(f.clock.as_ref()).unwrap();
    let replayed = replayed
        .activate_same_process_memory(
            file::StructuredServiceClockV7 {
                monotonic_ns: activation.monotonic_ns,
                wall_unix_ns: activation.wall_unix_ns,
            },
            &limits,
        )
        .unwrap();
    let restored = &replayed.children[0];
    assert_eq!(
        restored.parameters_signature(),
        child.parameters_signature()
    );
    let after = restored
        .predict_query_local(child.fingerprint(), &f.query(A), f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(after.planning_ns, original.planning_ns);
    assert_eq!(after.fitted_upper_ns, original.fitted_upper_ns);
    eprintln!(
        "incremental source7: real_calls={}, worker_turns={}, worker_ns={}, total_ns={}, original_bytes={}",
        f.recorded, worker_turns, worker_elapsed.as_nanos(), started.elapsed().as_nanos(), bytes.len()
    );
}

#[tokio::test]
async fn automatic_owner_block_incremental_processing_keeps_real_observation_lag_guard() {
    let mut f = Families::new();
    for _ in 0..4 {
        f.block(A);
    }
    assert!(f.known(A));
    f.runtime.consume_samples();
    queue(&mut f, 1);
    // The next real consumption clock is deliberately late. Incremental source
    // conversion must neither predate it nor exempt this original observation.
    let lag_limit = SloAutomaticCalibrationSettingsV1::default()
        .feedback
        .maximum_consumption_lag_ns
        .get();
    f.clock.set(f.clock.now_ns().unwrap() + lag_limit + 1);
    f.runtime.consume_samples();
    let audit = f.runtime.audit_snapshot();
    assert_eq!(audit.sink.raw_resolved, f.recorded);
    assert_eq!(audit.sink.raw_lost, 0);
    assert_eq!(
        serde_json::to_value(audit.structured_feedback.unwrap().revoked).unwrap(),
        serde_json::json!("observation_lag")
    );
    assert!(f.runtime.snapshot().is_none());
    f.runtime.shutdown().await.unwrap();
}
