//! Called from the existing real CPU source8 driver. No synthetic model or
//! unchecked profile field can stand in for an independently qualified child.
use super::super::{
    profile::EngineCostSnapshot,
    selected_feedback::{Comparison, FeedbackKind, FeedbackObservation, Monitor},
};
use super::*;
use ferrum_interfaces::execution_cost::{CostObservationClock, CostWorkloadDomainV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::ExecutionFingerprint, cost_profile::StructuredSourceRecordSinkV1,
};
use ferrum_types::{
    SloAutomaticCalibrationCacheLocationV1, SloAutomaticCalibrationSettingsV1,
    SloSelectedFeedbackSettingsV1, SloSelectedFeedbackStorageV1,
};
use std::num::{NonZeroU64, NonZeroUsize};

impl EngineCostSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::continuous_engine::inner) fn check_automatic_reuse_replay_for_test(
        &self,
        source: &Path,
        source_sha256: [u8; 32],
        checkpoint: (u64, [u8; 32]),
        fingerprint: &ExecutionFingerprint,
        workload: &CostWorkloadDomainV1,
        clock: &dyn CostObservationClock,
        cache_directory: &Path,
    ) {
        let automatic = SloAutomaticCalibrationSettingsV1::default();
        let import = super::super::profile::load_limits(&Default::default());
        let limits = CacheLimits {
            maximum_bytes: 256 * 1024 * 1024,
            maximum_source_bytes: automatic.maximum_encoded_source_bytes.get(),
            maximum_sources: automatic.maximum_retained_generations.get(),
            maximum_retained_bytes: automatic.maximum_retained_numeric_bytes.get()
                * automatic.maximum_retained_generations.get()
                + automatic.maximum_discovery_bytes.get()
                + automatic.reference_probe.maximum_source_bytes.get() as usize
                + automatic.feedback.maximum_state_bytes.get(),
            maximum_samples: import.max_samples.get(),
            maximum_rows: import.max_total_shape_rows.get(),
            maximum_duration: Duration::from_secs(30),
        };
        let mut policy = SloAutomaticCalibrationReuseV1::default();
        let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, .. } = &mut policy
        else {
            unreachable!()
        };
        // The product override isolates only the test's filesystem location.
        *location = SloAutomaticCalibrationCacheLocationV1::Directory {
            path: cache_directory.to_path_buf(),
        };
        let identity = CacheIdentity {
            binding: [9; 32],
            clock: clock.monotonic_domain().unwrap().clone(),
        };
        let opened =
            CacheSession::open(&policy, identity.clone(), clock.now_ns().unwrap(), limits).unwrap();
        assert_eq!(opened.candidate.unwrap_err(), CacheMiss::Missing);
        let session = opened.session;
        let mut transaction = opened.transaction;
        let mut journal = session.original_source().unwrap();
        let raw = fs::read(source).unwrap();
        assert_eq!(<[u8; 32]>::from(Sha256::digest(&raw)), source_sha256);
        let split = usize::try_from(checkpoint.0).unwrap();
        journal.append_original(&raw[..split], checkpoint);
        journal.checkpoint(checkpoint).unwrap();
        journal.append_original(&raw[split..], (raw.len() as u64, source_sha256));
        let archived = journal.seal((raw.len() as u64, source_sha256)).unwrap();
        assert!(archived.bytes < raw.len() as u64);
        assert_eq!(
            journal.encoding().unwrap().original(archived),
            (raw.len() as u64, source_sha256)
        );
        assert_eq!(session.0.ledger.lock().codec_retained, 0);

        let originals = self.live_children(clock.now_ns().unwrap()).unwrap();
        assert!(!originals.is_empty());
        let maximum_age = originals
            .iter()
            .map(|c| c.runtime_limits().1)
            .min()
            .unwrap();
        let maximum_wave = originals
            .iter()
            .map(|c| c.runtime_limits().0)
            .min()
            .unwrap();
        let feedback = SloSelectedFeedbackSettingsV1 {
            window_samples: NonZeroUsize::new(2).unwrap(),
            minimum_underestimates: NonZeroUsize::new(2).unwrap(),
            minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
            trigger_excess_ns: NonZeroU64::new(2).unwrap(),
            correction_padding_ns: 5,
            maximum_family_margin_ns: NonZeroU64::new(maximum_wave.min(100)).unwrap(),
            maximum_consumption_lag_ns: NonZeroU64::new(maximum_age.min(50)).unwrap(),
            maximum_uncomparable_observations: 0,
            maximum_failed_or_partial: 0,
            maximum_queue_drops: 0,
            maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
        };
        let context = ReplayContext {
            seed_limits: None,
            original_retained_bytes: self.restart_retained_bytes().unwrap(),
            fingerprint,
            workload,
            clock,
            policy: &feedback,
            import: &import,
        };
        let mut domains: Vec<_> = originals.iter().map(|c| *c.domain_signature()).collect();
        domains.sort_unstable();
        let cached = session
            .export_checkpoint(
                SourceKind::PreparedOwnersV8,
                archived,
                journal.encoding().unwrap(),
                checkpoint,
                domains,
                &context,
                &transaction,
            )
            .unwrap();
        let (binding, scopes) = self.live_feedback_binding(&feedback).unwrap();
        let scope = scopes[0];
        let mut monitor = Monitor::open_bound(
            &feedback,
            &SloSelectedFeedbackStorageV1::MemoryOnly,
            binding,
            scopes.len(),
            FeedbackKind::StructuredV2,
            Some(scopes),
        )
        .unwrap();
        for fifo in 1..=2 {
            let at = clock.now_ns().unwrap();
            monitor.observe_classified(
                fifo,
                FeedbackObservation::Compared(Comparison {
                    family: scope,
                    base_planning_ns: 100,
                    actual_ns: 120,
                    observed_at_ns: at,
                    consumed_at_ns: at,
                }),
            );
        }
        monitor.publish(Some(0)).unwrap();
        assert_eq!(monitor.current_view().margin(&scope), 25);
        session
            .finish_verified_catalog(
                vec![cached],
                self,
                &mut monitor,
                2,
                &context,
                &mut transaction,
            )
            .unwrap();
        drop(monitor);
        drop(journal);
        drop(session);

        let opened =
            CacheSession::open(&policy, identity.clone(), clock.now_ns().unwrap(), limits).unwrap();
        let candidate = opened.candidate.unwrap();
        assert_eq!(candidate.schema_version, 2);
        assert!(candidate
            .sources
            .iter()
            .all(|source| matches!(source.encoding, JournalEncoding::GzipV1 { .. })));
        let mut transaction = opened.transaction;
        let manifest_path = opened.session.0.root.join("manifest.json");
        let committed_manifest = fs::read(&manifest_path).unwrap();
        let assert_committed_unchanged = || {
            assert_eq!(fs::read(&manifest_path).unwrap(), committed_manifest);
            for receipt in candidate.artifacts() {
                let path = opened.session.artifact_path(candidate.generation, receipt);
                let verify = opened.session.transaction().unwrap();
                store::verify_file(&path, receipt, &verify).unwrap();
                for suffix in [".session", ".pending"] {
                    let mut marker = path.as_os_str().to_owned();
                    marker.push(suffix);
                    assert!(!Path::new(&marker).exists());
                }
            }
        };
        let mut restored = opened
            .session
            .restore_candidate(&candidate, &context, &mut transaction)
            .unwrap();
        // Real Monitor resume persists a new session and epoch. All those
        // mutations must be confined to the current private generation.
        assert_committed_unchanged();
        let adopted = opened
            .session
            .adopt_source(candidate.generation, &candidate.sources[0], &transaction)
            .unwrap();
        assert_committed_unchanged();
        assert!(
            restored.startup_seed.is_none(),
            "legacy cache has no cold inventory authority"
        );
        assert_eq!(restored.monitor.current_view().margin(&scope), 25);
        assert_eq!(restored.monitor.audit().compared, 2);
        assert!(
            !restored.monitor.current_view().current(),
            "installation gate remains closed"
        );
        let loaded = restored.seed.snapshot.unwrap();
        let children = loaded.live_children(clock.now_ns().unwrap()).unwrap();
        assert_eq!(children.len(), originals.len());
        for child in &children {
            let old = originals
                .iter()
                .find(|c| c.domain_signature() == child.domain_signature())
                .unwrap();
            assert_eq!(
                old.provenance().source_sha256,
                child.provenance().source_sha256
            );
            assert_eq!(
                old.provenance().parameters_sha256,
                child.provenance().parameters_sha256
            );
            assert_eq!(old.provenance().clock, child.provenance().clock);
        }
        assert!(loaded
            .validate_live_freshness(clock.now_ns().unwrap() + maximum_age + 1)
            .is_err());
        assert!(adopted.discard_if_unobserved().unwrap());
        assert_committed_unchanged();
        drop(adopted);
        let adopted = opened
            .session
            .adopt_source(candidate.generation, &candidate.sources[0], &transaction)
            .unwrap();
        let saved = opened
            .session
            .export_checkpoint(
                candidate.sources[0].kind,
                adopted.sealed().unwrap(),
                adopted.encoding().unwrap(),
                checkpoint,
                candidate.sources[0].domains.clone(),
                &context,
                &transaction,
            )
            .unwrap();
        assert_committed_unchanged();
        // A real second handoff reclaims the private resumed Store and carries
        // the same learned margin/provenance into another clean generation.
        opened
            .session
            .finish_verified_catalog(
                vec![saved],
                &loaded,
                &mut restored.monitor,
                0,
                &context,
                &mut transaction,
            )
            .unwrap();
        assert!(!opened.session.0.root.join("dirty").exists());
        let after_commit: CacheManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        assert_ne!(after_commit.generation, candidate.generation);
        restored.monitor.finish();
        drop(restored.monitor);
        for receipt in after_commit.artifacts() {
            store::verify_file(
                &opened
                    .session
                    .artifact_path(after_commit.generation, receipt),
                receipt,
                &opened.session.transaction().unwrap(),
            )
            .unwrap();
        }
        drop((adopted, opened.session));
        let next = CacheSession::open(&policy, identity, clock.now_ns().unwrap(), limits).unwrap();
        let mut next_transaction = next.transaction;
        let next_seed = next
            .session
            .restore_candidate(&next.candidate.unwrap(), &context, &mut next_transaction)
            .unwrap();
        assert_eq!(next_seed.monitor.current_view().margin(&scope), 25);
        assert_eq!(next_seed.monitor.audit().compared, 2);
        assert!(!next_seed.monitor.current_view().current());
        // An unacknowledged Monitor Drop only leaves this new transaction dirty.
        drop(next_seed);
        for receipt in after_commit.artifacts() {
            store::verify_file(
                &next.session.artifact_path(after_commit.generation, receipt),
                receipt,
                &next.session.transaction().unwrap(),
            )
            .unwrap();
        }
    }
}
