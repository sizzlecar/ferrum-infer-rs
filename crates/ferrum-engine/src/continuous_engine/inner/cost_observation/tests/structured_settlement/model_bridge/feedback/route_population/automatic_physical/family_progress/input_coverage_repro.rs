//! The same real CPU observations compare two explicit phase-support policies.
//! Legacy None keeps its numerical failure; the automatic intersection policy
//! waits for eligible members and can qualify only the covered input branch.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{OwnerPhaseSupportPolicyV1, StructuredUnknownV2},
    cost_profile::{
        record_bytes_v7, CostProfileLoadLimits, StructuredServiceCollectorV7,
        StructuredServiceHeaderV7, StructuredServiceRecordV7,
    },
};
use sha2::{Digest, Sha256};

struct ArchiveDirectory(PathBuf);
impl ArchiveDirectory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-phase-support-controls-{}",
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
impl Drop for ArchiveDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn branch_wave(at_length: bool) -> Wave {
    let mut host = wave(A).host;
    host.state.maximum_output_tokens = if at_length { 1 } else { 3 };
    wave_with_host_rows(A, 7, host, 2)
}

fn record_branch_block(f: &mut Families, at_length: bool) {
    f.runtime.consume_samples();
    for _ in 0..OFFERS {
        let w = branch_wave(at_length);
        let expected = w.actual.rows.clone();
        let stages = fixture::record_cohort_route(&f.runtime, &f.clock, w)
            .expect("actual CPU submission and complete original host settlement");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert_eq!(stages.rows.len(), expected.len());
        stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .validate_host_stages(&stages)
            .unwrap();
        for (original, settled) in expected.iter().zip(&stages.rows) {
            assert_eq!(settled.request_id, original.request_id);
            assert_eq!(settled.owner_incarnation, original.owner_incarnation);
            assert_eq!(settled.work_generation, original.work_generation);
            assert_eq!(settled.input_index, original.input_index);
            assert_eq!(settled.terminal.is_some(), at_length);
            if let Some(terminal) = &settled.terminal {
                assert_eq!(terminal.finish_reason, ferrum_types::FinishReason::Length);
            }
        }
        f.runtime.consume_samples();
        f.recorded += 1;
        let sink = f.runtime.audit_snapshot().sink;
        assert_eq!(sink.raw_offered, f.recorded);
        assert_eq!(sink.raw_accepted, f.recorded);
        assert_eq!(sink.raw_resolved, f.recorded);
        assert_eq!(sink.raw_resolution_failed, 0);
        assert_eq!(sink.raw_lost, 0);
        assert_eq!(sink.raw_pending, 0);
    }
    let live = f.live().audit();
    assert!(!live.population.failed, "{live:#?}");
    assert_eq!(live.population.issued, OFFERS);
    assert_eq!(live.population.retired, OFFERS);
    assert_eq!(live.population.eligible_route, OFFERS);
    assert!(live.first_settlement_failure.is_none(), "{live:#?}");
}

async fn diagnose_length_transition(fit_has_length: bool, residual_has_length: bool, reason: &str) {
    let directory = ArchiveDirectory::new();
    // CountOnly fixes the original stopping minimum, independently of the new
    // input-membership policy. The legacy comparator below explicitly has None.
    let mut f = Families::configured_with_input_readiness(
        directory.policy(),
        true,
        ferrum_types::SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {},
    );
    let domain = f.domain.clone();
    let query = |w: Wave| {
        StructuredQueryV2::from_future_with_domain(
            &w.prepared.exact,
            &w.prepared.selected,
            &w.prepared.recipe,
            &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
            &domain,
        )
        .unwrap()
    };
    assert_eq!(
        query(branch_wave(false)).owner(),
        query(branch_wave(true)).owner(),
        "Length changes numeric work, not the structural cost family"
    );

    // Original discovery, Fit and Residual populations; no ticket is retried
    // and no wave is selected or removed based on its measured wall.
    record_branch_block(&mut f, fit_has_length);
    record_branch_block(&mut f, fit_has_length);
    {
        let live = f.live().audit();
        let blocks = live
            .automatic
            .as_ref()
            .unwrap()
            .owner_blocks
            .as_ref()
            .unwrap();
        assert_eq!(blocks.owners.len(), 1);
        assert_eq!(blocks.owners[0].phase, Some(
            ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Residual));
        assert!(blocks.owners[0].failure.is_none(), "{blocks:#?}");
    }
    record_branch_block(&mut f, residual_has_length);
    let live = f.live().audit();
    let blocks = live
        .automatic
        .as_ref()
        .unwrap()
        .owner_blocks
        .as_ref()
        .unwrap();
    assert_eq!(blocks.offered, (3 * OFFERS) as u64);
    assert_eq!(blocks.owners.len(), 1);
    assert_eq!(blocks.owners[0].owner_offered, OFFERS);
    assert_eq!(blocks.owners[0].phase, Some(
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Residual));
    assert!(!blocks.owners[0].qualified);
    assert_eq!(
        blocks.owners[0].eligible, 0,
        "outside input cannot supply the minimum"
    );
    assert!(blocks.owners[0].failure.is_none(), "{blocks:#?}");
    assert_eq!(live.qualified_publications, 0);
    assert!(f.runtime.training.published_catalog_receipt().is_none());
    assert!(f.runtime.snapshot().is_none());
    assert_eq!(f.recorded, (3 * OFFERS) as u64);
    // Two subsequent complete original blocks supply independent eligible
    // Residual and Qualification. The excluded block is never reclassified.
    record_branch_block(&mut f, fit_has_length);
    assert_eq!(f.live().audit().qualified_publications, 0);
    record_branch_block(&mut f, fit_has_length);
    assert_eq!(f.live().audit().qualified_publications, 1);
    let snapshot = f.runtime.snapshot().unwrap();
    let now = f.clock.now_ns().unwrap();
    snapshot
        .audit_structured_query_v2(&query(branch_wave(fit_has_length)), now)
        .unwrap();
    assert_eq!(
        snapshot
            .audit_structured_query_v2(&query(branch_wave(!fit_has_length)), now)
            .unwrap_err(),
        StructuredUnknownV2::QualificationCoverage,
    );
    f.runtime.shutdown().await.unwrap();
    let final_audit = f.live().audit();
    assert_eq!(final_audit.qualified_publications, 1);
    assert_eq!(final_audit.failed_generations, 0);
    let automatic = final_audit.automatic.unwrap();
    assert_eq!(automatic.diagnostic_failures.count, 0);
    let source = automatic.last_diagnostic_publication.unwrap().source;
    let bytes = fs::read(&source.path).unwrap();
    assert_eq!(source.bytes, bytes.len() as u64);
    assert_eq!(source.digest, <[u8; 32]>::from(Sha256::digest(&bytes)));
    compare_explicit_legacy_policy(&bytes, reason);
}

fn compare_explicit_legacy_policy(bytes: &[u8], reason: &str) {
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines.next().unwrap();
    let original: StructuredServiceHeaderV7 = serde_json::from_slice(first).unwrap();
    assert_eq!(record_bytes_v7(&original).unwrap(), first);
    assert_eq!(original.declaration.schedule.input_readiness, None);
    assert_eq!(
        original.declaration.schedule.phase_support,
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1)
    );
    let budget = NonZeroU64::new(original.maximum_file_bytes).unwrap();
    let mut original_replay = StructuredServiceCollectorV7::new_streaming(
        original.clone(),
        CostProfileLoadLimits::default(),
        budget,
    )
    .unwrap();
    let mut declaration = original.declaration.clone();
    declaration.schedule.phase_support = None;
    let mut identity = Sha256::new();
    identity.update(b"ferrum.test.original-length-legacy-control.v1\0");
    identity.update(original.capture_identity);
    let legacy_header = StructuredServiceHeaderV7::new(
        identity.finalize().into(),
        original.generation,
        original.fingerprint.clone(),
        original.producer.clone(),
        original.opening,
        declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    assert_ne!(legacy_header.capture_identity, original.capture_identity);
    assert_ne!(legacy_header.protocol, original.protocol);
    let mut legacy = StructuredServiceCollectorV7::new_streaming(
        legacy_header,
        CostProfileLoadLimits::default(),
        budget,
    )
    .unwrap();
    let mut compared_blocks = 0;
    for line in lines {
        let record: StructuredServiceRecordV7 = serde_json::from_slice(line).unwrap();
        assert_eq!(record_bytes_v7(&record).unwrap(), line);
        original_replay.push(&record).unwrap();
        if compared_blocks == 3 {
            continue;
        }
        match record {
            StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            } => {
                legacy.open_block(opened_at_ns, fifo_cutoff).unwrap();
            }
            StructuredServiceRecordV7::Completed { .. }
            | StructuredServiceRecordV7::OutsideDeclaredRoute { .. }
            | StructuredServiceRecordV7::NotSubmitted { .. } => legacy.push(&record).unwrap(),
            StructuredServiceRecordV7::BlockClose { closing, .. } => {
                let result = legacy.close_block(closing).unwrap();
                compared_blocks += 1;
                if compared_blocks == 3 {
                    let StructuredServiceRecordV7::BlockClose { freezes, .. } = result else {
                        unreachable!()
                    };
                    assert_eq!(freezes.len(), 1);
                    assert_eq!(freezes[0].close.member_count, OFFERS);
                    assert_eq!(freezes[0].domain.owner_offered, OFFERS);
                    assert_eq!(freezes[0].domain.eligible, OFFERS);
                    assert_eq!(freezes[0].domain.outside_fit_support, 0);
                    assert_eq!(freezes[0].domain.outside_residual_support, 0);
                    assert!(
                        freezes[0]
                            .failure
                            .as_deref()
                            .is_some_and(|e| e.contains(reason)),
                        "{freezes:#?}"
                    );
                    legacy.stop(closing).unwrap();
                }
            }
            _ => panic!("no checkpoint or failed tail precedes the third original block"),
        }
    }
    assert_eq!(compared_blocks, 3);
    assert_eq!(legacy.offered(), (3 * OFFERS) as u64);
    assert_eq!(legacy.qualified_children(), 0);
    assert_eq!(legacy.audit().owners[0].phase, None);
    assert_eq!(original_replay.offered(), (5 * OFFERS) as u64);
    assert_eq!(original_replay.qualified_children(), 1);
    assert_eq!(
        original_replay.source_receipt(),
        (bytes.len() as u64, Sha256::digest(bytes).into())
    );
}

#[tokio::test]
async fn automatic_legacy_count_readiness_late_length_keeps_receipts_but_cannot_publish() {
    diagnose_length_transition(false, true, "UnidentifiedDirection").await;
}

#[tokio::test]
async fn automatic_legacy_count_readiness_missing_residual_length_cannot_publish() {
    diagnose_length_transition(true, false, "QualificationCoverage").await;
}
