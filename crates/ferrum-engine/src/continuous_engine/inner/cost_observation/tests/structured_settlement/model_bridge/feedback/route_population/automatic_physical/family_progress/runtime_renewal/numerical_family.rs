//! Original mixed-width CPU settlements drive two complete automatic source7
//! generations. Renewal and archive replay must use the numerical family even
//! when independent discovery chooses a different real owner representative.
use super::*;
use sha2::{Digest, Sha256};
#[path = "numerical_family/subset_shadow.rs"]
mod subset_shadow;

// Eight distinct input corners can have rank eight. Retain all corners and
// the original rank + 4 redundancy gate by collecting sixteen original waves.
const BLOCK_OFFERS: usize = 2 * OFFERS;

fn families(diagnostics: SloAutomaticCalibrationDiagnosticsV1) -> Families {
    families_before_or_after_start(diagnostics, true)
}

fn families_before_or_after_start(
    diagnostics: SloAutomaticCalibrationDiagnosticsV1,
    start: bool,
) -> Families {
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        panic!("fixture identity");
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
    let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    assert_eq!(config.model.min_samples.get(), OFFERS);
    assert!(BLOCK_OFFERS >= OFFERS + StructuredSettingsV2::default().min_fit_redundancy);
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
        settings: SloAutomaticCalibrationSettingsV1 {
            discovery_offered_waves: NonZeroUsize::new(BLOCK_OFFERS).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(BLOCK_OFFERS).unwrap(); 3],
            diagnostics,
            ..Default::default()
        },
    };
    let runtime = EngineCostRuntime::build_with_profile_and_domain(
        identity,
        clock.clone(),
        &config,
        false,
        None,
        None,
        Some(domain.clone()),
    )
    .unwrap();
    if start {
        runtime.begin_automatic_calibration().unwrap();
        runtime.consume_samples();
    }
    Families {
        runtime,
        clock,
        domain,
        recorded: 0,
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-numerical-family-renewal-{}",
            uuid::Uuid::new_v4()
        )))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn ordinary_wave(rows: u32, pending: bool, length: bool, context: u32) -> Wave {
    let mut host = wave(A).host;
    // The parent fixture's zero generated history is FirstDecode and cannot
    // exercise the homogeneous ordinary-decode population.
    host.state.generated_tokens_before = 1;
    host.state.sampling_history_tokens = 1;
    host.state.maximum_output_tokens = if length { 2 } else { 3 };
    host.state.pending_decoded_utf8 = pending;
    wave_with_host_rows(A, context, host, rows)
}

fn query(f: &Families, rows: u32, pending: bool, length: bool) -> StructuredQueryV2 {
    // A fresh future request at a context absent from the measured population.
    let w = ordinary_wave(rows, pending, length, 8);
    StructuredQueryV2::from_future_with_domain(
        &w.prepared.exact,
        &w.prepared.selected,
        &w.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
        &f.domain,
    )
    .unwrap()
}

fn block(f: &mut Families, discovery_width_one: bool) {
    f.runtime.consume_samples();
    assert_population_healthy(f);
    // Repeat all eight corners with new original calls. Eight members alone
    // have at least rank six and fail the unchanged redundancy requirement.
    for offer in 0..BLOCK_OFFERS {
        let i = offer % OFFERS;
        let rows = if discovery_width_one {
            1
        } else {
            1 + (i % 2) as u32
        };
        let pending = i / 2 % 2 != 0;
        let length = i / 4 != 0;
        let w = ordinary_wave(rows, pending, length, 7);
        let expected = w.actual.rows.clone();
        let stages = fixture::record_cohort_route(&f.runtime, &f.clock, w)
            .expect("original CPU cohort submission and host settlement");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert_eq!(stages.rows.len(), rows as usize);
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        assert!(stages.route_evidence.is_some());
        for (actual, settled) in expected.iter().zip(&stages.rows) {
            assert_eq!(settled.request_id, actual.request_id);
            assert_eq!(settled.owner_incarnation, actual.owner_incarnation);
            assert_eq!(settled.work_generation, actual.work_generation);
            assert_eq!(settled.input_index, actual.input_index);
            assert_eq!(
                settled.completeness,
                HostStageCompleteness::CompleteSingleWave
            );
            assert_eq!(settled.terminal.is_some(), length);
        }
        f.runtime.consume_samples();
        f.recorded += 1;
        assert_population_healthy(f);
    }
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_offered, f.recorded);
    assert_eq!(sink.raw_accepted, f.recorded);
    assert_eq!(sink.raw_resolved, f.recorded);
    assert_eq!(sink.raw_resolution_failed, 0);
    assert_eq!(sink.raw_lost, 0);
    assert_eq!(sink.raw_pending, 0);
}

fn assert_population_healthy(f: &Families) {
    let before = f.live().audit();
    let failed_owners: Vec<_> = before
        .automatic
        .as_ref()
        .and_then(|automatic| automatic.owner_blocks.as_ref())
        .into_iter()
        .flat_map(|blocks| &blocks.owners)
        .filter(|owner| owner.failure.is_some())
        .collect();
    if failed_owners.is_empty()
        && before.failed_generations == 0
        && !before.population.failed
        && before.publication_error.is_none()
    {
        return;
    }
    eprintln!("original owner freeze failures before reopening: {failed_owners:#?}");
    // Close the original failed source before Directory cleanup. Preserve its
    // actual block_close/freezes in test output; a later reserve error must
    // never replace the numerical failure that caused the population to stop.
    f.runtime.consume_samples();
    let after = f.live().audit();
    if let Some(path) = after
        .last_failed_source
        .as_ref()
        .and_then(|source| source.path.as_ref())
    {
        match fs::read(path) {
            Ok(bytes) => {
                for line in bytes
                    .split(|byte| *byte == b'\n')
                    .filter(|line| !line.is_empty())
                {
                    let record: serde_json::Value = serde_json::from_slice(line).unwrap();
                    if matches!(
                        record["kind"].as_str(),
                        Some("block_close" | "failed" | "footer")
                    ) {
                        eprintln!("original failed source {}: {record}", path.display());
                    }
                }
            }
            Err(error) => eprintln!(
                "original failed source {} unreadable: {error}",
                path.display()
            ),
        }
    }
    panic!("automatic family block failed before next offer; before={before:#?}; after={after:#?}");
}

fn assert_queries(f: &Families, child: &ImportedStructuredModelV2, queries: &[StructuredQueryV2]) {
    let now = f.clock.now_ns().unwrap();
    let snapshot = f.runtime.snapshot().unwrap();
    for query in queries {
        assert_eq!(
            child.numerical_family_key(),
            Some(
                &query
                    .input()
                    .numerical_family_key_for_universe(
                        child
                            .algorithm_universe()
                            .expect("automatic discovery froze its checked subset")
                    )
                    .unwrap()
            )
        );
        assert!(!snapshot.undeclared_structured_owner(query));
        let prediction = snapshot.audit_structured_query_v2(query, now).unwrap();
        assert!(prediction.planning_ns > 0);
        assert_eq!(
            snapshot.prospective_source(query).unwrap().domain,
            *child.domain_signature()
        );
    }
}

fn replay_archive(
    source: &crate::continuous_engine::inner::cost_observation::profile_export::PublishedFile,
    directory: &Directory,
    expected: &ImportedStructuredModelV2,
    queries: &[StructuredQueryV2],
    prediction_at: u64,
) {
    let bytes = fs::read(&source.path).unwrap();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!((source.bytes, source.digest), (bytes.len() as u64, digest));
    assert_eq!(source.sha256, format!("{:x}", Sha256::digest(&bytes)));
    let mut offset = 0usize;
    let mut checkpoint_end = None;
    let mut completed = 0;
    let mut closing_wall = None;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        offset += line.len();
        let record: serde_json::Value = serde_json::from_slice(line).unwrap();
        match record["kind"].as_str() {
            Some("completed") => completed += 1,
            Some("checkpoint") => checkpoint_end = Some(offset),
            Some("footer") => {
                assert!(!record["incomplete_block"].as_bool().unwrap());
                assert_eq!(offset, bytes.len());
                closing_wall = record["closing"]["wall_unix_ns"].as_u64();
            }
            Some("failed") => panic!("qualified original generation contains a failed block"),
            _ => {}
        }
    }
    assert_eq!(completed, 4 * BLOCK_OFFERS);
    let checkpoint_end = checkpoint_end.unwrap();
    assert!(checkpoint_end < bytes.len());
    let prefix = &bytes[..checkpoint_end];
    let prefix_digest: [u8; 32] = Sha256::digest(prefix).into();
    let limits = file::CostProfileLoadLimits::default();
    let checkpoint = file::replay_structured_source_v7(prefix, &limits).unwrap();
    assert_eq!(checkpoint.qualified_children(), 1);
    assert_eq!(
        checkpoint.source_receipt(),
        (prefix.len() as u64, prefix_digest)
    );
    drop(checkpoint);
    let profile_path = directory.0.join(format!(
        "qualified-family-owner-{}.json",
        expected.owner().rows
    ));
    let exported = file::export_structured_profile_v14(
        &source.path,
        digest,
        checkpoint_end as u64,
        &profile_path,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.schema_version, 14);
    assert_eq!(exported.children.len(), 1);
    assert_eq!(exported.source_bytes, prefix.len() as u64);
    assert_eq!(exported.source_sha256, prefix_digest);
    let load_now = 1_000;
    let imported = file::load_structured_profile_v14(
        &profile_path,
        expected.fingerprint(),
        &limits,
        file::ProfileLoadClock {
            wall_unix_ns: Some(closing_wall.unwrap() + 1),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: load_now,
        },
    )
    .unwrap();
    assert_eq!(imported.journal_bytes, source.bytes);
    assert_eq!(imported.offered_attempts, (4 * BLOCK_OFFERS) as u64);
    assert_eq!(imported.source_sha256, prefix_digest);
    assert_eq!(imported.children.len(), 1);
    let child = &imported.children[0];
    assert_eq!(child.owner(), expected.owner());
    assert_eq!(
        child.numerical_family_key(),
        expected.numerical_family_key()
    );
    assert_eq!(child.domain_signature(), expected.domain_signature());
    assert_eq!(
        child.parameters_signature(),
        expected.parameters_signature()
    );
    assert_eq!(child.provenance().storage, SloCostProfileStorage::File);
    assert_eq!(child.workload_domain(), expected.workload_domain());
    for query in queries {
        let before = expected
            .predict_query_local(expected.fingerprint(), query, prediction_at)
            .unwrap();
        let after = child
            .predict_query_local(expected.fingerprint(), query, load_now)
            .unwrap();
        assert_eq!(after.planning_ns, before.planning_ns);
        assert_eq!(after.fitted_upper_ns, before.fitted_upper_ns);
        assert_eq!(after.valid_until_ns, before.valid_until_ns);
    }
}

#[tokio::test]
async fn automatic_source7_numerical_family_renews_mixed_widths_and_replays_original_archives() {
    let directory = Directory::new();
    let mut f = families(SloAutomaticCalibrationDiagnosticsV1::Directory {
        directory: directory.0.clone(),
        maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
        maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
    });
    let queries: Vec<_> = [1, 2]
        .into_iter()
        .flat_map(|rows| {
            [false, true]
                .into_iter()
                .map(move |pending| (rows, pending))
        })
        .flat_map(|(rows, pending)| {
            [false, true]
                .into_iter()
                .map(move |length| (rows, pending, length))
        })
        .map(|(rows, pending, length)| query(&f, rows, pending, length))
        .collect();
    let family = queries[0].input().numerical_family_key().unwrap();
    for query in &queries {
        assert_eq!(query.input().numerical_family_key().unwrap(), family);
    }
    let events = Installations::default();
    for phase in 0..4 {
        events.during(|| block(&mut f, false));
        let audit = f.live().audit();
        assert_eq!(audit.failed_generations, 0, "{audit:#?}");
        assert_eq!(
            audit.qualified_publications,
            u64::from(phase == 3),
            "{audit:#?}"
        );
    }
    let first = only_child(&f);
    assert_eq!(first.owner().rows, 2);
    let projected_family = queries[0]
        .input()
        .numerical_family_key_for_universe(
            first.algorithm_universe().expect("first discovery subset"),
        )
        .unwrap();
    assert_eq!(first.numerical_family_key(), Some(&projected_family));
    // Freezing a numerical interpretation never rewrites the original query.
    for query in &queries {
        assert_eq!(query.input().numerical_family_key().unwrap(), family);
    }
    assert_queries(&f, &first, &queries);
    let first_provenance = serde_json::to_value(first.provenance()).unwrap();
    let first_snapshot = f.runtime.snapshot().unwrap();
    let first_epoch = first_snapshot.prospective_identity().unwrap();
    let first_expiry = expires_at(&first, &queries[0], f.clock.now_ns().unwrap());
    let first_prediction_at = f.clock.now_ns().unwrap();
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.offered_samples, 4 * BLOCK_OFFERS);
    assert_eq!(receipt.recorded_samples, 3 * BLOCK_OFFERS);
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);

    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.generation, 2);
    let first_source = f
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source;
    // This new independent discovery sees only B1. The real representative
    // owner changes; subsequent mixed-width members still belong to one family.
    for phase in 0..4 {
        events.during(|| block(&mut f, phase == 0));
        let audit = f.live().audit();
        assert_eq!(audit.failed_generations, 0, "{audit:#?}");
        assert_eq!(
            audit.qualified_publications,
            1 + u64::from(phase == 3),
            "{audit:#?}"
        );
        assert_eq!(first_epoch.current(), phase != 3);
    }
    let second = only_child(&f);
    assert_eq!(second.owner().rows, 1);
    assert_ne!(first.owner(), second.owner());
    assert_eq!(second.algorithm_universe(), first.algorithm_universe());
    assert_eq!(second.numerical_family_key(), Some(&projected_family));
    assert_eq!(first.domain_signature(), second.domain_signature());
    assert_queries(&f, &second, &queries);
    for child in [&first, &second] {
        assert_eq!(
            child
                .provenance()
                .phases
                .each_ref()
                .map(|phase| phase.members),
            [BLOCK_OFFERS; 3]
        );
    }
    assert!(
        second.provenance().phases[0].accepted_fifo_cutoff
            > first.provenance().phases[2].accepted_fifo_cutoff
    );
    assert_ne!(
        first.provenance().capture_identity,
        second.provenance().capture_identity
    );
    assert_ne!(
        first.provenance().source_sha256,
        second.provenance().source_sha256
    );
    assert_eq!(
        serde_json::to_value(first.provenance()).unwrap(),
        first_provenance
    );
    let now = f.clock.now_ns().unwrap();
    assert_eq!(expires_at(&first, &queries[0], now), first_expiry);
    assert!(expires_at(&second, &queries[0], now) > first_expiry);
    for query in &queries {
        assert_eq!(
            first_snapshot
                .audit_structured_query_v2(query, now)
                .unwrap_err(),
            StructuredUnknownV2::RuntimeValidity
        );
    }
    let records = events.records();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["changed_children"][0]["kind"], "added");
    let change = &records[1]["changed_children"][0];
    assert_eq!(change["kind"], "replaced");
    assert_eq!(change["same_domain"], true);
    assert_eq!(change["independent_source"], true);
    assert_eq!(change["previous_was_current_at_validation"], true);
    assert_eq!(records[1]["catalog_child_count"], 1);
    assert_ne!(
        change["before"]["owner_sha256"],
        change["after"]["owner_sha256"]
    );
    assert_eq!(f.recorded, (8 * BLOCK_OFFERS) as u64);

    f.runtime.shutdown().await.unwrap();
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 2, "{audit:#?}");
    assert_eq!(audit.failed_generations, 0, "{audit:#?}");
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    let automatic = audit.automatic.unwrap();
    assert!(automatic.stopped);
    assert_eq!(automatic.diagnostic_failures.count, 0);
    let second_source = automatic.last_diagnostic_publication.unwrap().source;
    assert_ne!(first_source.path, second_source.path);
    replay_archive(
        &first_source,
        &directory,
        &first,
        &queries,
        first_prediction_at,
    );
    replay_archive(&second_source, &directory, &second, &queries, now);
}
