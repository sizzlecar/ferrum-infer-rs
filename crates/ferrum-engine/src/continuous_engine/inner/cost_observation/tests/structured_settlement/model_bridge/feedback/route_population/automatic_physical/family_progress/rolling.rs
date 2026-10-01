//! Real private CPU submissions and the production worker coordinator. The
//! clock is fake; original tickets/FIFO/host settlement are not fabricated.
use super::*;

fn fixture(retained: usize) -> Families {
    Families::configured_with_settings(
        SloAutomaticCalibrationSettingsV1 {
            population_schedule:
                ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
            discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
            maximum_retained_generations: NonZeroUsize::new(retained).unwrap(),
            ..Default::default()
        },
        true,
    )
}

macro_rules! rolling {
    ($f:expr) => {
        $f.live()
            .audit()
            .automatic
            .unwrap()
            .rolling_owner_blocks
            .unwrap()
    };
}

#[path = "rolling/close_task.rs"]
mod close_task;

fn open_original_block(f: &Families) {
    let limit = rolling!(f).resource.maximum_source_slots + 3;
    for _ in 0..limit {
        let audit = f.live().audit();
        if !audit.population.closed && audit.population.issued == 0 {
            return;
        }
        f.runtime.consume_samples();
    }
    panic!(
        "worker did not open its next real block: {:#?}",
        f.live().audit()
    );
}

fn block(f: &mut Families, algorithm: &'static str) {
    open_original_block(f);
    let original = rolling!(f).original_block;
    for _ in 0..OFFERS {
        f.record(algorithm);
    }
    let limit = rolling!(f).resource.maximum_source_slots + 3;
    for _ in 0..limit {
        let audit = rolling!(f);
        if audit.at_boundary && audit.original_block == original {
            return;
        }
        f.runtime.consume_samples();
    }
    panic!(
        "original block close/publication ACK did not drain: {:#?}",
        f.live().audit()
    );
}

#[tokio::test]
async fn rolling_owner_blocks_cpu_repeats_successors_without_duplicate_original_offers() {
    let mut f = fixture(4);
    let mut captures = std::collections::BTreeSet::new();
    for _ in 0..12 {
        block(&mut f, A);
        let audit = rolling!(&f);
        assert!(audit.active_enrollments.len() <= audit.resource.maximum_source_slots);
        assert!(audit.pending_successor_parents.len() <= audit.resource.maximum_source_slots);
        for source in &audit.active_enrollments {
            captures.insert(source.capture_identity);
        }
        assert_eq!(f.runtime.audit_snapshot().sink.raw_offered, f.recorded);
        assert_eq!(
            f.live().audit().failed_generations,
            0,
            "{:#?}",
            f.live().audit()
        );
    }
    let live = f.live().audit();
    assert!(live.qualified_publications >= 3, "{live:#?}");
    assert!(
        captures.len() > rolling!(&f).resource.maximum_source_slots,
        "lifetime source creation must exceed the retained slot count"
    );
    assert!(f.known(A), "{live:#?}");
    let children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        children.len(),
        1,
        "same population replaces in original publication order"
    );
    assert_eq!(
        children[0]
            .provenance()
            .phases
            .iter()
            .map(|p| p.members)
            .collect::<Vec<_>>(),
        vec![OFFERS; 3]
    );
    let audit = rolling!(&f);
    assert_eq!(audit.resource.original_blocks_credited, 12);
    assert!(audit.resource.spent.canonical_source_bytes > 0);
    assert!(audit.resource.spent.readiness_scalar_visits > 0);
    assert!(audit.resource.spent.phase_transition_attempts >= 9);
}

#[tokio::test]
#[ignore = "reads declared runtime settings via FERRUM_ROLLING_CAPACITY_SETTINGS"]
async fn rolling_owner_blocks_declared_settings_capacity() {
    let settings: SloAutomaticCalibrationSettingsV1 = serde_json::from_slice(
        &fs::read(std::env::var_os("FERRUM_ROLLING_CAPACITY_SETTINGS").expect("settings path"))
            .unwrap(),
    )
    .unwrap();
    settings.validate().unwrap();
    assert_eq!(
        settings.population_schedule,
        ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2
    );
    // No fixture quota or owner override: use the actual production constructor
    // and its allocation sizes before any offered wave or numerical result.
    let f = Families::configured_with_settings(settings.clone(), true);
    let resource = rolling!(&f).resource;
    eprintln!(
        "ROLLING_PRODUCTION_CAPACITY {}",
        serde_json::json!({
            "settings":settings,"capacity":resource,
        })
    );
    assert_eq!(
        resource.collector_bytes_per_source
            + resource.catalog_bytes_per_source
            + resource.source_metadata_bytes,
        resource.source_slot_bytes
    );
}

#[tokio::test]
async fn rolling_owner_blocks_old_source_expiry_preserves_younger_ticket_roster() {
    let mut f = fixture(4);
    block(&mut f, A);
    // Fixed clock gap between original BlockOpen boundaries, before observing
    // either block's results. Each source still gets exactly its own 300s.
    f.clock.0.fetch_add(1_000_000_000, Ordering::SeqCst);
    block(&mut f, B);
    let before = rolling!(&f);
    let oldest = before
        .active_enrollments
        .iter()
        .find(|s| s.generation == 1)
        .unwrap();
    let younger = before
        .active_enrollments
        .iter()
        .find(|s| s.generation == 2)
        .unwrap();
    assert!(younger.deadline_ns > oldest.deadline_ns);
    let younger_capture = younger.capture_identity;
    f.clock.0.store(oldest.deadline_ns + 1, Ordering::SeqCst);
    open_original_block(&f);
    let after = rolling!(&f);
    assert!(after.active_enrollments.iter().all(|s| s.generation != 1));
    assert!(after
        .active_enrollments
        .iter()
        .any(|s| s.capture_identity == younger_capture));
    assert!(!f.live().audit().population.failed);
    block(&mut f, A);
    assert_eq!(
        f.live().audit().failed_generations,
        1,
        "{:#?}",
        f.live().audit()
    );
}

#[tokio::test]
async fn rolling_owner_blocks_full_slots_retry_same_declared_successor_after_release() {
    let mut f = fixture(1); // Two capture slots including active/catalog sources.
    block(&mut f, A);
    f.clock.0.fetch_add(1_000_000_000, Ordering::SeqCst);
    block(&mut f, B); // Sparse B holds the oldest source while A progresses.
    block(&mut f, A);
    let blocked = rolling!(&f);
    assert_eq!(blocked.resource.maximum_source_slots, 2);
    assert_eq!(blocked.active_enrollments.len(), 2);
    assert_eq!(blocked.pending_successor_parents, vec![2]);
    let expiry = blocked
        .active_enrollments
        .iter()
        .map(|s| s.deadline_ns)
        .min()
        .unwrap();
    f.clock.0.store(expiry + 1, Ordering::SeqCst);
    open_original_block(&f);
    let resumed = rolling!(&f);
    assert_eq!(resumed.active_enrollments.len(), 2);
    assert!(resumed.active_enrollments.iter().any(|s| s.generation == 3));
    assert!(resumed.pending_successor_parents.is_empty());
    assert_eq!(resumed.resource.original_blocks_credited, 3);
    block(&mut f, A);
    assert_eq!(rolling!(&f).resource.source_reservations, 2);
}

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn rolling_owner_blocks_original_journals_replay_independent_membership_and_work() {
    let directory = Directory(
        std::env::temp_dir().join(format!("ferrum-rolling-original-{}", uuid::Uuid::new_v4())),
    );
    let mut f = Families::configured_with_settings(
        SloAutomaticCalibrationSettingsV1 {
            population_schedule:
                ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
            discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
            diagnostics: SloAutomaticCalibrationDiagnosticsV1::Directory {
                directory: directory.0.clone(),
                maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
                maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
                maximum_retained_generations: NonZeroUsize::new(16).unwrap(),
            },
            ..Default::default()
        },
        true,
    );
    let mut audits = std::collections::BTreeMap::new();
    let mut paths = std::collections::BTreeSet::new();
    for _ in 0..6 {
        block(&mut f, A);
        let audit = rolling!(&f);
        for (enrollment, source) in audit.active_enrollments.iter().zip(&audit.active_sources) {
            audits.insert(enrollment.capture_identity, source.clone());
        }
        if let Some(published) = &f
            .live()
            .audit()
            .automatic
            .unwrap()
            .last_diagnostic_publication
        {
            paths.insert(published.source.path.clone());
        }
    }
    assert!(f.live().audit().qualified_publications >= 3);
    f.runtime.shutdown().await.unwrap();
    let audit = f.live().audit();
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    assert!(rolling!(&f).active_sources.is_empty(), "{audit:#?}");
    assert_eq!(rolling!(&f).resource.source_reservations, 0);
    assert!(
        !paths.is_empty(),
        "completed rolling attempts retain their own actual journals"
    );
    for path in paths {
        let bytes = fs::read(path).unwrap();
        let mut lines = bytes.split_inclusive(|b| *b == b'\n');
        let header: file::StructuredServiceHeaderV7 =
            serde_json::from_slice(lines.next().unwrap()).unwrap();
        let expected = audits.get(&header.capture_identity).unwrap();
        let limit = NonZeroU64::new(header.maximum_file_bytes).unwrap();
        let limits = file::CostProfileLoadLimits::default();
        let mut replay =
            file::StructuredServiceCollectorV7::new_streaming(header, limits.clone(), limit)
                .unwrap();
        let mut checkpoint_end = None;
        let mut offset = bytes.split_inclusive(|b| *b == b'\n').next().unwrap().len();
        let mut checkpoint_audit = None;
        for line in lines {
            offset += line.len();
            let record: file::StructuredServiceRecordV7 = serde_json::from_slice(line).unwrap();
            replay.push(&record).unwrap();
            if matches!(record, file::StructuredServiceRecordV7::Checkpoint { .. }) {
                checkpoint_end = Some(offset);
                checkpoint_audit = Some(replay.audit());
            }
        }
        let end = checkpoint_end.expect("published source has a complete independent checkpoint");
        let replay_audit = checkpoint_audit.unwrap();
        assert_eq!(replay_audit.offered, expected.offered);
        assert_eq!(replay_audit.source_bytes, expected.source_bytes);
        assert_eq!(
            replay_audit.readiness_scalar_visits,
            expected.readiness_scalar_visits
        );
        assert_eq!(
            replay_audit.phase_transition_attempts,
            expected.phase_transition_attempts
        );
        let checkpoint = file::replay_structured_source_v7(&bytes[..end], &limits).unwrap();
        assert_eq!(checkpoint.qualified_children(), 1);
    }
    assert_eq!(f.runtime.audit_snapshot().sink.raw_offered, f.recorded);
}
