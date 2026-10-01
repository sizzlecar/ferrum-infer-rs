//! Shared run/serve cache lifecycle. The only authority comes from strict
//! original replay plus the preserved feedback monitor, never this index.
use super::super::{
    profile::{self, EngineCostSnapshot},
    selected_feedback::Monitor,
};
use super::*;
use ferrum_interfaces::execution_cost::{CostObservationClock, CostWorkloadDomainV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::ExecutionFingerprint, cost_profile,
};
use ferrum_types::{
    SloAutomaticCalibrationSettingsV1, SloCostObservationConfig, SloSelectedFeedbackSettingsV1,
};

#[derive(Clone, Copy)]
pub(in crate::continuous_engine::inner::cost_observation) struct RestartOrigin {
    pub capture: [u8; 32],
    pub protocol: [u8; 32],
    pub domain: [u8; 32],
    pub prefix: (u64, [u8; 32]),
    pub offered: u64,
    pub rows: u64,
}
struct Source {
    kind: SourceKind,
    capture: [u8; 32],
    protocol: [u8; 32],
    journal: OriginalSourceJournal,
}
pub(in crate::continuous_engine::inner::cost_observation) struct ReuseOpening {
    pub coordinator: Arc<AutomaticReuse>,
    pub restored: Option<RestoredCostSeed>,
    pub miss: Option<CacheMiss>,
}
pub(in crate::continuous_engine::inner::cost_observation) struct AutomaticReuse {
    cache: CacheSession,
    fingerprint: ExecutionFingerprint,
    workload: CostWorkloadDomainV1,
    clock: Arc<dyn CostObservationClock>,
    feedback: SloSelectedFeedbackSettingsV1,
    import: cost_profile::CostProfileLoadLimits,
    maximum_origins: usize,
    seed_limits: super::seed::SeedLimits,
    startup_seed: Mutex<Option<super::seed::DeclaredAlgorithmUniverseV1>>,
    sources: Mutex<Vec<Source>>,
    inventory: Mutex<Vec<RestartOrigin>>,
    finished: std::sync::atomic::AtomicBool,
    audit: Mutex<ReuseAudit>,
    pending_clean: Mutex<
        Option<(
            ColdTransaction,
            Arc<EngineCostSnapshot>,
            Arc<EngineCostSnapshot>,
            u64,
        )>,
    >,
}
impl AutomaticReuse {
    pub fn audit(&self) -> ReuseAudit {
        *self.audit.lock()
    }
    pub fn note_failure(&self, stage: ReuseStage, reason: CacheMiss) {
        self.audit.lock().failure(stage, reason);
    }
    fn at_stage<T>(&self, stage: ReuseStage, result: Result<T>) -> Result<T> {
        if let Err(reason) = &result {
            self.note_failure(stage, *reason);
        }
        result
    }

    pub fn open(
        config: &SloCostObservationConfig,
        settings: &SloAutomaticCalibrationSettingsV1,
        fingerprint: ExecutionFingerprint,
        workload: CostWorkloadDomainV1,
        clock: Arc<dyn CostObservationClock>,
    ) -> Result<ReuseOpening> {
        let clock_domain = clock
            .monotonic_domain()
            .ok_or(CacheMiss::UnsupportedClock)?;
        let feedback = match &config.structured_feedback {
            ferrum_types::SloStructuredFeedbackPolicy::Disabled => settings.feedback.clone(),
            ferrum_types::SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
                policy,
                ..
            } => policy.clone(),
        };
        // Serialize directly into a bounded hash writer. No per-query hashing,
        // no arbitrary JSON tree, and no authorization from the lookup key.
        struct BindingHash {
            hash: Sha256,
            bytes: usize,
        }
        impl Write for BindingHash {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.bytes = self
                    .bytes
                    .checked_add(bytes.len())
                    .filter(|n| *n <= 1024 * 1024)
                    .ok_or_else(|| std::io::Error::other("cache binding capacity"))?;
                self.hash.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut digest = BindingHash {
            hash: Sha256::new(),
            bytes: 0,
        };
        digest
            .hash
            .update(b"ferrum.automatic.sameboot-cache-binding.v1\0");
        serde_json::to_writer(
            &mut digest,
            &(
                cost_profile::ProfileFingerprint::from(&fingerprint),
                workload.sha256(),
                &config.model,
                settings,
                &feedback,
            ),
        )
        .map_err(|_| CacheMiss::Capacity)?;
        let identity = CacheIdentity {
            binding: digest.hash.finalize().into(),
            clock: clock_domain.clone(),
        };
        let mut import = profile::load_limits(&config.profile_import);
        // This is the existing, validated whole automatic retention budget.
        // A full original numeric collector and the earlier retained models can
        // coexist during replay. Do not mistake its per-source cap for the sum.
        let retained = settings
            .maximum_retained_numeric_bytes
            .get()
            .checked_mul(settings.maximum_retained_generations.get())
            .and_then(|n| n.checked_add(settings.maximum_discovery_bytes.get()))
            .and_then(|n| {
                n.checked_add(
                    usize::try_from(settings.reference_probe.maximum_source_bytes.get()).ok()?,
                )
            })
            .and_then(|n| n.checked_add(feedback.maximum_state_bytes.get()))
            .ok_or(CacheMiss::Capacity)?;
        let policy_limits = match &settings.reuse {
            SloAutomaticCalibrationReuseV1::Disabled {} => return Err(CacheMiss::Disabled),
            SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { limits, .. } => limits,
        };
        // Explicit manual imports keep their 16 MiB default. Automatic reuse
        // instead shares its already declared transaction-wide disk envelope;
        // replay_catalog deducts every source+metadata byte from this one sum.
        import.max_file_bytes = std::num::NonZeroUsize::new(
            usize::try_from(policy_limits.maximum_total_bytes.get())
                .map_err(|_| CacheMiss::Capacity)?
                .min(ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES),
        )
        .ok_or(CacheMiss::Capacity)?;
        let limits = CacheLimits {
            maximum_bytes: policy_limits.maximum_total_bytes.get(),
            maximum_source_bytes: settings
                .maximum_encoded_source_bytes
                .get()
                .min(policy_limits.maximum_total_bytes.get()),
            // At most 128 independently selected children/prefixes. Physical
            // original captures have the smaller original retention bound below.
            maximum_sources: 128,
            maximum_retained_bytes: retained,
            maximum_samples: import.max_samples.get(),
            maximum_rows: import.max_total_shape_rows.get(),
            maximum_duration: Duration::from_millis(
                policy_limits.maximum_operation_duration_ms.get(),
            ),
        };
        let seed_limits = super::seed::SeedLimits {
            maximum_axes: super::super::automatic_numerical_settings(settings).max_axes,
            maximum_bytes: settings.maximum_discovery_bytes.get(),
        };
        let mut opened = CacheSession::open(
            &settings.reuse,
            identity,
            clock.now_ns().ok_or(CacheMiss::UnsupportedClock)?,
            limits,
        )?;
        let context = ReplayContext {
            fingerprint: &fingerprint,
            workload: &workload,
            clock: clock.as_ref(),
            policy: &feedback,
            import: &import,
            original_retained_bytes: 0,
            seed_limits: Some(seed_limits),
        };
        let mut miss = opened.candidate.as_ref().err().copied();
        let mut restored = match opened.candidate.as_ref() {
            Ok(manifest) => {
                match opened
                    .session
                    .restore_candidate(manifest, &context, &mut opened.transaction)
                {
                    Ok(seed) if seed.monitor.audit().revoked.is_none() => Some(seed),
                    Ok(_) => {
                        miss = Some(CacheMiss::Revoked);
                        None
                    }
                    Err(reason) => {
                        miss = Some(reason);
                        None
                    }
                }
            }
            Err(_) => None,
        };
        let maximum_origins = settings
            .maximum_retained_generations
            .get()
            .checked_add(1)
            .ok_or(CacheMiss::Capacity)?;
        let coordinator = Arc::new(Self {
            cache: opened.session,
            fingerprint,
            workload,
            clock,
            feedback,
            import,
            maximum_origins,
            seed_limits,
            startup_seed: Mutex::new(None),
            sources: Mutex::new(Vec::with_capacity(maximum_origins)),
            inventory: Mutex::new(Vec::with_capacity(128)),
            finished: std::sync::atomic::AtomicBool::new(false),
            audit: Mutex::new(ReuseAudit::default()),
            pending_clean: Mutex::new(None),
        });
        // Preserve the actual source streams before the old generation can be
        // collected at the next clean commit. Private copies stay inside the
        // initial dirty transaction; old and new bytes are both charged.
        if let (Some(seed), Ok(manifest)) = (&restored, &opened.candidate) {
            let result = (|| {
                if let Some(seed) = &seed.startup_seed {
                    coordinator.note_startup_seed(seed)?;
                }
                coordinator
                    .note_catalog(seed.seed.snapshot.as_deref().ok_or(CacheMiss::Corrupt)?)?;
                coordinator.copy_restored_sources(manifest, &opened.transaction)
            })();
            if let Err(reason) = result {
                coordinator.note_failure(ReuseStage::ArchiveRestoredSources, reason);
                tracing::info!(
                    ?reason,
                    "Automatic cost cache seed remains valid but next-save archival is unavailable"
                );
            }
        }
        if let Some(seed) = &restored {
            let fresh = opened
                .transaction
                .poll()
                .and_then(|()| {
                    coordinator
                        .clock
                        .now_ns()
                        .ok_or(CacheMiss::UnsupportedClock)
                })
                .and_then(|now| {
                    seed.seed
                        .snapshot
                        .as_ref()
                        .ok_or(CacheMiss::Corrupt)?
                        .validate_live_freshness(now)
                        .map_err(|_| CacheMiss::Expired)
                });
            if let Err(reason) = fresh {
                miss = Some(reason);
                restored = None;
                coordinator.inventory.lock().clear();
                *coordinator.startup_seed.lock() = None;
            }
        }
        Ok(ReuseOpening {
            coordinator,
            restored,
            miss,
        })
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn note_startup_seed(
        &self,
        seed: &super::seed::DeclaredAlgorithmUniverseV1,
    ) -> Result<()> {
        super::seed::validate(seed, &self.workload, self.seed_limits)?;
        let mut current = self.startup_seed.lock();
        if let Some(old) = current.as_ref() {
            if seed == old {
                return Ok(());
            }
            if !seed.contains_universe(old) {
                return Err(CacheMiss::Identity);
            }
            old.retained_payload_bytes()
                .and_then(|n| n.checked_add(seed.retained_payload_bytes()?))
                .filter(|n| *n <= self.seed_limits.maximum_bytes)
                .ok_or(CacheMiss::Capacity)?;
        }
        *current = Some(seed.clone());
        Ok(())
    }
    pub fn open_source(
        &self,
        kind: SourceKind,
        capture: [u8; 32],
        protocol: [u8; 32],
    ) -> Result<OriginalSourceJournal> {
        self.prune()?;
        let mut sources = self.sources.lock();
        if sources.len() >= self.maximum_origins
            || sources
                .iter()
                .any(|s| s.capture == capture && s.protocol == protocol)
        {
            return Err(CacheMiss::Capacity);
        }
        let journal = self.cache.original_source()?;
        sources.push(Source {
            kind,
            capture,
            protocol,
            journal: journal.clone(),
        });
        Ok(journal)
    }
    pub fn note_catalog(&self, snapshot: &EngineCostSnapshot) -> Result<()> {
        self.at_stage(ReuseStage::TrackCatalog, self.note_catalog_inner(snapshot))
    }
    pub fn note_empty_catalog(&self) -> Result<()> {
        self.inventory.lock().clear();
        self.at_stage(ReuseStage::TrackCatalog, self.prune())
    }
    fn note_catalog_inner(&self, snapshot: &EngineCostSnapshot) -> Result<()> {
        let mut inventory = self.inventory.lock();
        inventory.clear();
        snapshot
            .restart_origins(&mut inventory)
            .map_err(|_| CacheMiss::Corrupt)?;
        drop(inventory);
        self.prune()
    }
    fn prune(&self) -> Result<()> {
        let inventory = self.inventory.lock();
        let mut sources = self.sources.lock();
        let mut i = 0;
        while i < sources.len() {
            let source = &sources[i];
            let keep = inventory
                .iter()
                .any(|v| v.capture == source.capture && v.protocol == source.protocol);
            if !keep && source.journal.discard_if_unobserved()? {
                sources.swap_remove(i);
            } else {
                i += 1;
            }
        }
        Ok(())
    }
    fn copy_restored_sources(
        &self,
        manifest: &CacheManifest,
        transaction: &ColdTransaction,
    ) -> Result<()> {
        let inventory = self.inventory.lock().clone();
        for source in &manifest.sources {
            transaction.poll()?;
            let selected = inventory
                .iter()
                .find(|v| source.domains.binary_search(&v.domain).is_ok())
                .ok_or(CacheMiss::Corrupt)?;
            if self
                .sources
                .lock()
                .iter()
                .any(|s| s.capture == selected.capture && s.protocol == selected.protocol)
            {
                continue;
            }
            // Full original replay succeeded, but the previous manifest stays
            // immutable even if this transaction fails or the model expires.
            // Copy its exact bytes under the shared quota, without renewing age.
            self.prune()?;
            let mut sources = self.sources.lock();
            if sources.len() >= self.maximum_origins {
                return Err(CacheMiss::Capacity);
            }
            let journal = self
                .cache
                .adopt_source(manifest.generation, source, transaction)?;
            sources.push(Source {
                kind: source.kind,
                capture: selected.capture,
                protocol: selected.protocol,
                journal,
            });
        }
        Ok(())
    }
    pub fn finish(
        &self,
        original: &Arc<EngineCostSnapshot>,
        monitor: &mut Monitor,
        processed_fifo: u64,
    ) -> Result<()> {
        self.at_stage(
            ReuseStage::PrepareSave,
            self.finish_inner(original, monitor, processed_fifo),
        )
    }
    fn finish_inner(
        &self,
        original: &Arc<EngineCostSnapshot>,
        monitor: &mut Monitor,
        processed_fifo: u64,
    ) -> Result<()> {
        use std::sync::atomic::Ordering;
        if self.finished.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let mut transaction = self.cache.transaction()?;
        self.at_stage(
            ReuseStage::VerifyFreshness,
            original
                .validate_live_freshness(self.clock.now_ns().ok_or(CacheMiss::UnsupportedClock)?)
                .map_err(|_| CacheMiss::Expired),
        )?;
        self.note_catalog(original)?;
        let inventory = self.inventory.lock();
        if inventory.is_empty() {
            return Err(CacheMiss::Missing);
        }
        // Before any exporter replays, authorize the complete unique original
        // populations using private imported/live provenance. Replay checks the
        // source hashes and exact prefix again; sources do not renew this sum.
        let mut counts = (0u64, 0u64);
        for (i, v) in inventory.iter().enumerate() {
            if !inventory[..i]
                .iter()
                .any(|p| p.capture == v.capture && p.protocol == v.protocol && p.prefix == v.prefix)
            {
                counts.0 = counts.0.checked_add(v.offered).ok_or(CacheMiss::Capacity)?;
                counts.1 = counts.1.checked_add(v.rows).ok_or(CacheMiss::Capacity)?;
            }
        }
        if counts.0 > self.import.max_samples.get() as u64
            || counts.1 > self.import.max_total_shape_rows.get() as u64
        {
            return Err(CacheMiss::Capacity);
        }
        let startup_seed = self.startup_seed.lock().clone();
        let seed_bytes = startup_seed
            .as_ref()
            .map_or(Some(0), |s| s.retained_payload_bytes())
            .ok_or(CacheMiss::Capacity)?;
        let context = ReplayContext {
            fingerprint: &self.fingerprint,
            workload: &self.workload,
            clock: self.clock.as_ref(),
            policy: &self.feedback,
            import: &self.import,
            original_retained_bytes: original
                .restart_retained_bytes()
                .map_err(|_| CacheMiss::Capacity)?
                .checked_add(seed_bytes)
                .ok_or(CacheMiss::Capacity)?,
            seed_limits: Some(self.seed_limits),
        };
        let sources = self.sources.lock();
        let mut exports = Vec::with_capacity(inventory.len());
        for (index, selected) in inventory.iter().enumerate() {
            if inventory[..index].iter().any(|p| {
                p.capture == selected.capture
                    && p.protocol == selected.protocol
                    && p.prefix == selected.prefix
            }) {
                continue;
            }
            let source = sources
                .iter()
                .find(|s| s.capture == selected.capture && s.protocol == selected.protocol)
                .ok_or(CacheMiss::Missing)?;
            let journal = source.journal.sealed()?;
            let mut domains = Vec::with_capacity(inventory.len());
            domains.extend(
                inventory
                    .iter()
                    .filter(|p| {
                        p.capture == selected.capture
                            && p.protocol == selected.protocol
                            && p.prefix == selected.prefix
                    })
                    .map(|p| p.domain),
            );
            domains.sort_unstable();
            exports.push(self.at_stage(
                ReuseStage::ExportSource,
                self.cache.export_checkpoint(
                    source.kind,
                    journal,
                    source.journal.encoding()?,
                    selected.prefix,
                    domains,
                    &context,
                    &transaction,
                ),
            )?);
        }
        drop(sources);
        drop(inventory);
        let seed_receipt = startup_seed
            .as_ref()
            .map(|seed| super::seed::persist(&self.cache, seed, &context, &transaction))
            .transpose()?;
        let replayed = self.at_stage(
            ReuseStage::ReplayCatalog,
            self.cache.prepare_verified_catalog(
                exports,
                original,
                monitor,
                processed_fifo,
                &context,
                &mut transaction,
                seed_receipt,
            ),
        )?;
        let checked_at = self.clock.now_ns().ok_or(CacheMiss::UnsupportedClock)?;
        self.at_stage(
            ReuseStage::VerifyFreshness,
            original
                .validate_live_freshness(checked_at)
                .map_err(|_| CacheMiss::Expired),
        )?;
        self.at_stage(
            ReuseStage::VerifyFreshness,
            replayed
                .validate_live_freshness(checked_at)
                .map_err(|_| CacheMiss::Expired),
        )?;
        *self.pending_clean.lock() = Some((transaction, original.clone(), replayed, checked_at));
        self.audit.lock().prepared = true;
        Ok(())
    }
    pub fn acknowledge_clean_shutdown(&self) -> Result<()> {
        let result = self.acknowledge_clean_shutdown_inner();
        if result.is_ok() {
            self.audit.lock().clean_shutdown = true;
        }
        self.at_stage(ReuseStage::CleanShutdown, result)
    }
    fn acknowledge_clean_shutdown_inner(&self) -> Result<()> {
        let (transaction, original, replayed, checked_at) =
            self.pending_clean.lock().take().ok_or_else(|| {
                self.audit
                    .lock()
                    .first_failure
                    .map_or(CacheMiss::Missing, |failure| failure.reason)
            })?;
        self.cache.complete_prepared_commit(&transaction, || {
            let now = self.clock.now_ns().ok_or(CacheMiss::UnsupportedClock)?;
            if now < checked_at {
                return Err(CacheMiss::Expired);
            }
            original
                .validate_live_freshness(now)
                .map_err(|_| CacheMiss::Expired)?;
            replayed
                .validate_live_freshness(now)
                .map_err(|_| CacheMiss::Expired)
        })
    }
}
