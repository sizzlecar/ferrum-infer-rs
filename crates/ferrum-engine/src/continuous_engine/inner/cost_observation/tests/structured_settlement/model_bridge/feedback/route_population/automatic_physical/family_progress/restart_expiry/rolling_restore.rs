//! Clean restart -> exact retained-origin ownership -> bounded rolling renewal.
use super::*;
use std::collections::BTreeSet;

#[path = "rolling_restore/cache_failure.rs"]
mod cache_failure;

fn assert_capture_ownership(f: &Families) {
    let automatic = f.live().audit().automatic.unwrap();
    let mut captures = BTreeSet::new();
    if let Some(snapshot) = f.runtime.snapshot() {
        for child in snapshot.retained_catalog_children().unwrap() {
            captures.insert(child.provenance().capture_identity);
        }
    }
    assert_eq!(automatic.retained_origins, captures.len(), "{automatic:#?}");
    if let Some(rolling) = automatic.rolling_owner_blocks {
        captures.extend(
            rolling
                .active_enrollments
                .iter()
                .map(|source| source.capture_identity),
        );
        assert!(
            captures.len() <= rolling.resource.maximum_source_slots,
            "restored catalog and original enrolled sources share one slot bound: {rolling:#?}"
        );
        assert!(rolling.resource.source_reservations <= rolling.resource.maximum_source_slots);
    }
}

fn complete_block(f: &mut Families, algorithm: &'static str) {
    // Each actual block has at most K+1 source closes/publications and the
    // boundary turn; no retries are conditioned on measured numerical cost.
    let limit = 2 + 1 + 3;
    for _ in 0..limit {
        let population = f.live().audit().population;
        if !population.closed && population.issued == 0 {
            break;
        }
        f.runtime.consume_samples();
    }
    let opened = f.live().audit();
    assert!(!opened.population.closed, "{opened:#?}");
    assert_eq!(opened.population.issued, 0, "{opened:#?}");
    assert_capture_ownership(f);
    let block = opened
        .automatic
        .unwrap()
        .rolling_owner_blocks
        .unwrap()
        .original_block;
    for _ in 0..OFFERS {
        f.record(algorithm);
    }
    for _ in 0..limit {
        let audit = f
            .live()
            .audit()
            .automatic
            .unwrap()
            .rolling_owner_blocks
            .unwrap();
        if audit.at_boundary && audit.original_block == block {
            assert_capture_ownership(f);
            return;
        }
        f.runtime.consume_samples();
    }
    panic!(
        "original block did not finish publication ACK: {:#?}",
        f.live().audit()
    );
}

fn original_clean_generation(directory: &CacheDirectory) -> serde_json::Value {
    let root = directory.0.join("cache-v1");
    assert!(!root.join("dirty").exists());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema_version"], 2);
    let generation: [u8; 16] = serde_json::from_value(manifest["generation"].clone()).unwrap();
    let name = generation
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    use sha2::{Digest, Sha256};
    use std::io::Read;
    for source in manifest["sources"].as_array().unwrap() {
        assert_eq!(source["kind"], "owner_blocks_v7");
        assert_eq!(source["encoding"]["kind"], "gzip_v1");
        let receipt = &source["journal"];
        let path = root.join(format!("g-{name}")).join(format!(
            "artifact-{}.bin",
            receipt["index"].as_u64().unwrap()
        ));
        let stored = std::fs::read(path).unwrap();
        assert_eq!(stored.len() as u64, receipt["bytes"].as_u64().unwrap());
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(&stored)),
            serde_json::from_value::<[u8; 32]>(receipt["sha256"].clone()).unwrap()
        );
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(&stored[..])
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(
            raw.len() as u64,
            source["encoding"]["original_bytes"].as_u64().unwrap()
        );
        assert!(
            raw.len() > stored.len(),
            "real original source must exercise distinct raw/stored accounting"
        );
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(&raw)),
            serde_json::from_value::<[u8; 32]>(source["encoding"]["original_sha256"].clone())
                .unwrap()
        );
        let prefix = source["checkpoint_bytes"].as_u64().unwrap() as usize;
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(&raw[..prefix])),
            serde_json::from_value::<[u8; 32]>(source["checkpoint_sha256"].clone()).unwrap()
        );
    }
    manifest
}

#[tokio::test]
async fn rolling_warm_restored_origins_precede_enrollment_and_allow_one_family_renewal() {
    let directory = CacheDirectory(std::env::temp_dir().join(format!(
        "ferrum-rolling-warm-origins-{}",
        uuid::Uuid::new_v4()
    )));
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        unreachable!()
    };
    let domain = CostWorkloadDomainV1::new_vnext(
        executor,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap();
    let local = Arc::new(VirtualClock(AtomicU64::new(1)));
    let clock = Arc::new(BootClock {
        local: local.clone(),
        domain: CostMonotonicDomainV1::new_macos_continuous([74; 16]).unwrap(),
    });
    let mut settings = SloAutomaticCalibrationSettingsV1 {
        population_schedule:
            ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
        maximum_retained_generations: NonZeroUsize::new(2).unwrap(),
        discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
        phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
        ..Default::default()
    };
    let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, .. } =
        &mut settings.reuse
    else {
        panic!("default automatic reuse policy")
    };
    *location = SloAutomaticCalibrationCacheLocationV1::Directory {
        path: directory.0.clone(),
    };
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
    let runtime = EngineCostRuntime::build_with_manual_worker_and_reuse(
        identity.clone(),
        clock.clone(),
        &config,
        domain.clone(),
    )
    .unwrap();
    runtime.begin_automatic_calibration().unwrap();
    runtime.consume_samples();
    let mut cold = Families {
        runtime,
        clock: local.clone(),
        domain: domain.clone(),
        recorded: 0,
    };
    // A D/F/R/Q = 1/2/5/6, B D/F/R/Q = 3/4/7/8. These are
    // genuine original private CPU submissions from independent complete blocks.
    for algorithm in [A, A, B, B, A, A, B, B] {
        complete_block(&mut cold, algorithm);
    }
    assert!(cold.known(A) && cold.known(B), "{:#?}", cold.live().audit());
    cold.runtime.shutdown().await.unwrap();
    cold.runtime.acknowledge_restart_shutdown();
    let clean = cold.runtime.audit_snapshot().automatic_reuse.unwrap();
    assert!(
        clean.clean_shutdown && clean.first_failure.is_none(),
        "{clean:?}"
    );
    let first = original_clean_generation(&directory);
    drop(cold);

    let runtime = EngineCostRuntime::build_with_manual_worker_and_reuse(
        identity,
        clock,
        &config,
        domain.clone(),
    )
    .unwrap();
    assert!(
        runtime.reused_cost_is_fresh(),
        "must really restore before starting capture"
    );
    let mut warm = Families {
        runtime,
        clock: local.clone(),
        domain,
        recorded: 0,
    };
    assert!(warm.known(A) && warm.known(B));
    let restored = warm.runtime.snapshot().unwrap();
    let restored_children = restored.retained_catalog_children().unwrap();
    assert_eq!(
        restored_children.len(),
        2,
        "both genuine populations restored"
    );
    let restored_origins = restored_children
        .iter()
        .map(|child| child.provenance().capture_identity)
        .collect::<BTreeSet<_>>();
    assert!(!restored_origins.is_empty());
    let automatic = warm.live().audit().automatic.unwrap();
    assert_eq!(
        automatic.retained_origins,
        restored_origins.len(),
        "restored origins must be charged before first enrollment"
    );
    assert!(
        automatic.rolling_owner_blocks.is_none(),
        "no original source opened yet"
    );
    let a_query = warm.query(A);
    let b_query = warm.query(B);
    let original_a_source = restored.prospective_source(&a_query).unwrap();
    let original_a = original_a_source.domain;
    let original_a_child = restored_children
        .iter()
        .find(|child| *child.domain_signature() == original_a)
        .unwrap();
    let original_a_clock = original_a_child.provenance().clock.clone();
    let restored_epoch = restored.model_version();
    let warm_started_at = local.now_ns().unwrap();
    let original_b = restored.prospective_source(&b_query).unwrap().domain;
    let b = restored_children
        .iter()
        .find(|child| *child.domain_signature() == original_b)
        .unwrap();
    let expected_b_parameters = b.parameters_signature();
    let expected_b_clock = b.provenance().clock.clone();
    let expected_b_provenance = serde_json::to_value(b.provenance()).unwrap();

    warm.runtime.begin_automatic_calibration().unwrap();
    warm.runtime.consume_samples();
    assert_capture_ownership(&warm);
    for _ in 0..4 {
        complete_block(&mut warm, A);
    }
    let audit = warm.live().audit();
    assert!(audit.qualified_publications > 0, "{audit:#?}");
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    assert_eq!(audit.failed_generations, 0, "{audit:#?}");
    assert!(warm.known(A) && warm.known(B));
    let renewed = warm.runtime.snapshot().unwrap();
    let renewed_a_source = renewed.prospective_source(&a_query).unwrap();
    // Domain identifies the numerical family; renewal of that family keeps
    // this key. Original source identity, phase clocks and publication epoch
    // prove that actual new qualification replaced the restored model.
    assert_eq!(renewed_a_source.domain, original_a);
    assert_ne!(
        renewed_a_source.capture_identity,
        original_a_source.capture_identity
    );
    assert_ne!(
        renewed_a_source.source_sha256,
        original_a_source.source_sha256
    );
    assert!(renewed.model_version() > restored_epoch);
    assert!(
        !restored.current(),
        "the old runtime publication must be invalidated"
    );
    assert_eq!(
        renewed.prospective_source(&b_query).unwrap().domain,
        original_b
    );
    let children = renewed.retained_catalog_children().unwrap();
    let a = children
        .iter()
        .find(|child| *child.domain_signature() == original_a)
        .unwrap();
    let provenance = a.provenance();
    assert_eq!(
        provenance.capture_identity,
        renewed_a_source.capture_identity
    );
    assert!(provenance.clock.model_anchor_ns > original_a_clock.model_anchor_ns);
    assert_eq!(
        provenance.monotonic_domain,
        original_a_child.provenance().monotonic_domain
    );
    use ferrum_scheduler::implementations::continuous::cost_profile::StructuredProfilePhaseV10;
    assert_eq!(
        provenance.phases.each_ref().map(|phase| phase.phase),
        [
            StructuredProfilePhaseV10::Fit,
            StructuredProfilePhaseV10::Residual,
            StructuredProfilePhaseV10::Qualification
        ]
    );
    assert!(provenance
        .phases
        .iter()
        .all(|phase| phase.members == OFFERS && phase.frozen_at_ns > warm_started_at));
    assert!(provenance
        .phases
        .windows(2)
        .all(|phases| phases[0].frozen_at_ns < phases[1].frozen_at_ns
            && phases[0].member_cutoff < phases[1].member_cutoff
            && phases[0].accepted_fifo_cutoff < phases[1].accepted_fifo_cutoff
            && phases[0].source_prefix_bytes < phases[1].source_prefix_bytes));
    eprintln!(
        "ROLLING_WARM_ORIGIN_RENEWAL {}",
        serde_json::json!({
            "old_runtime_epoch": restored_epoch,
            "new_runtime_epoch": renewed.model_version(),
            "domain_unchanged": renewed_a_source.domain == original_a,
            "old_capture": original_a_source.capture_identity,
            "new_capture": renewed_a_source.capture_identity,
            "old_model_anchor_ns": original_a_clock.model_anchor_ns,
            "new_model_anchor_ns": provenance.clock.model_anchor_ns,
            "warm_started_at_ns": warm_started_at,
            "phase_members": provenance.phases.each_ref().map(|phase| phase.members),
            "phase_frozen_at_ns": provenance.phases.each_ref().map(|phase| phase.frozen_at_ns),
        })
    );
    let b = children
        .iter()
        .find(|child| *child.domain_signature() == original_b)
        .unwrap();
    assert_eq!(b.parameters_signature(), expected_b_parameters);
    assert_eq!(b.provenance().clock, expected_b_clock);
    assert_eq!(
        serde_json::to_value(b.provenance()).unwrap(),
        expected_b_provenance
    );
    assert_capture_ownership(&warm);
    warm.runtime.shutdown().await.unwrap();
    warm.runtime.acknowledge_restart_shutdown();
    let clean = warm.runtime.audit_snapshot().automatic_reuse.unwrap();
    assert!(
        clean.clean_shutdown && clean.first_failure.is_none(),
        "{clean:?}"
    );
    let second = original_clean_generation(&directory);
    assert_ne!(first["generation"], second["generation"]);
}
