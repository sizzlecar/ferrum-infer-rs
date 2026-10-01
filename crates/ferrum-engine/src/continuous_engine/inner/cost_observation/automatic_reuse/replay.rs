//! Strict cold replay, using the existing source7/8 importer and the original
//! feedback handoff. Storage metadata alone can never construct this seed.
use super::super::{
    profile::{self, EngineCostSnapshot, TrainingSeed},
    selected_feedback::{Monitor, RestartFeedbackAllowance},
    structured_epoch,
};
use super::*;
use ferrum_interfaces::execution_cost::{CostObservationClock, CostWorkloadDomainV1};
use ferrum_scheduler::implementations::continuous::{
    cost_model::ExecutionFingerprint, cost_profile as file,
};
use ferrum_types::{SloCostProfileReceipt, SloSelectedFeedbackSettingsV1};
use std::{io::BufReader, num::NonZeroUsize};
mod capacity;

pub(in crate::continuous_engine::inner::cost_observation) struct ReplayContext<'a> {
    pub fingerprint: &'a ExecutionFingerprint,
    pub workload: &'a CostWorkloadDomainV1,
    pub clock: &'a dyn CostObservationClock,
    pub policy: &'a SloSelectedFeedbackSettingsV1,
    pub import: &'a file::CostProfileLoadLimits,
    /// Existing original models still alive during cold save/replay.
    pub original_retained_bytes: usize,
    pub seed_limits: Option<super::seed::SeedLimits>,
}
pub(in crate::continuous_engine::inner::cost_observation) struct RestoredCostSeed {
    pub seed: TrainingSeed,
    /// Move this existing Monitor into the new runtime. Do not open a fresh
    /// feedback State or rebind with an empty retained-domain list.
    pub monitor: Monitor,
    pub startup_seed: Option<super::seed::DeclaredAlgorithmUniverseV1>,
}
impl ReplayContext<'_> {
    fn now(&self) -> Result<file::StructuredServiceClockV7> {
        Ok(file::StructuredServiceClockV7 {
            monotonic_ns: self.clock.now_ns().ok_or(CacheMiss::UnsupportedClock)?,
            // Informational only in SameBootMonotonic; no wall error is made up.
            wall_unix_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|d| u64::try_from(d.as_nanos()).ok())
                .unwrap_or(0),
        })
    }
}

impl CacheSession {
    /// Export the exact original publication prefix using the existing replay
    /// constructor. The final journal is immutable and is never substituted
    /// for an earlier child's checkpoint merely because it is longer.
    pub fn export_checkpoint(
        &self,
        kind: SourceKind,
        journal: ArtifactReceipt,
        encoding: JournalEncoding,
        checkpoint: (u64, [u8; 32]),
        domains: Vec<[u8; 32]>,
        context: &ReplayContext<'_>,
        transaction: &ColdTransaction,
    ) -> Result<CachedSource> {
        transaction.poll()?;
        let (raw_bytes, raw_sha) = encoding.original(journal);
        if domains.is_empty()
            || domains.len() > 128
            || domains.windows(2).any(|p| p[0] >= p[1])
            || checkpoint.0 == 0
            || checkpoint.0 > raw_bytes
        {
            return Err(CacheMiss::Corrupt);
        }
        let path = self.artifact_path(self.0.generation, &journal);
        let raw = codec::decode(
            self,
            &path,
            journal,
            encoding,
            context.import.max_file_bytes.get(),
            context.original_retained_bytes,
            transaction,
        )?;
        let numeric = declared_numeric_bound_bytes(&raw, kind, transaction)?;
        let peak = capacity::replay_peak_bytes(
            &raw,
            numeric,
            context.original_retained_bytes,
            transaction,
        )?
        .checked_add(self.0.ledger.lock().codec_retained)
        .ok_or(CacheMiss::Capacity)?;
        if peak > self.0.limits.maximum_retained_bytes {
            return Err(CacheMiss::Capacity);
        }
        // publish_new writes one metadata temporary and then links/renames
        // it. Reserve both temporary/final logical copies conservatively using
        // the same encoder-enforced bound, never the input journal limit.
        let reserve = u64::try_from(file::structured_owner_block_metadata_maximum_bytes())
            .ok()
            .and_then(|n| n.checked_mul(2))
            .ok_or(CacheMiss::Capacity)?;
        let destination = self.reserve_external(reserve, transaction)?;
        let now = context.now()?;
        let source_sha = match kind {
            SourceKind::OwnerBlocksV7 => {
                file::export_structured_profile_v14_same_boot_selected_from_original_bytes(
                    &path,
                    &raw,
                    raw_sha,
                    checkpoint.0,
                    &destination.path,
                    &self.0.identity.clock,
                    now,
                    context.import,
                    &domains,
                )
                .map_err(|_| CacheMiss::Corrupt)?
                .source_sha256
            }
            SourceKind::PreparedOwnersV8 => {
                file::export_structured_profile_v15_same_boot_selected_from_original_bytes(
                    &path,
                    &raw,
                    raw_sha,
                    checkpoint.0,
                    &destination.path,
                    &self.0.identity.clock,
                    now,
                    context.import,
                    &domains,
                )
                .map_err(|_| CacheMiss::Corrupt)?
                .source_sha256
            }
        };
        transaction.poll()?;
        if source_sha != checkpoint.1 {
            return Err(CacheMiss::Corrupt);
        }
        let profile = destination.finish(transaction)?;
        Ok(CachedSource {
            kind,
            journal,
            encoding,
            checkpoint_bytes: checkpoint.0,
            checkpoint_sha256: checkpoint.1,
            profile,
            domains,
        })
    }

    /// Final worker drain -> independently replayed catalog -> original
    /// feedback handoff -> atomic outer clean marker. No empty feedback State
    /// is constructed and neither samples nor profile age is renewed.
    pub fn finish_verified_catalog(
        &self,
        sources: Vec<CachedSource>,
        original: &EngineCostSnapshot,
        monitor: &mut Monitor,
        processed_fifo: u64,
        context: &ReplayContext<'_>,
        transaction: &mut ColdTransaction,
    ) -> Result<()> {
        self.finish_verified_catalog_inner(
            sources,
            original,
            monitor,
            processed_fifo,
            context,
            transaction,
            false,
            None,
        )
        .map(|_| ())
    }
    pub(super) fn prepare_verified_catalog(
        &self,
        sources: Vec<CachedSource>,
        original: &EngineCostSnapshot,
        monitor: &mut Monitor,
        processed_fifo: u64,
        context: &ReplayContext<'_>,
        transaction: &mut ColdTransaction,
        startup_seed: Option<ArtifactReceipt>,
    ) -> Result<Arc<EngineCostSnapshot>> {
        self.finish_verified_catalog_inner(
            sources,
            original,
            monitor,
            processed_fifo,
            context,
            transaction,
            true,
            startup_seed,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn finish_verified_catalog_inner(
        &self,
        sources: Vec<CachedSource>,
        original: &EngineCostSnapshot,
        monitor: &mut Monitor,
        processed_fifo: u64,
        context: &ReplayContext<'_>,
        transaction: &mut ColdTransaction,
        defer_clean: bool,
        startup_seed: Option<ArtifactReceipt>,
    ) -> Result<Arc<EngineCostSnapshot>> {
        transaction.poll()?;
        let mut manifest = CacheManifest {
            schema_version: 2,
            generation: self.0.generation,
            identity: self.0.identity.clone(),
            sources,
            startup_seed,
            feedback: ArtifactReceipt {
                index: 0,
                bytes: 0,
                sha256: [0; 32],
            },
            // Replaying and final freshness checking are authoritative. Do not
            // reconstruct sample expiry from a load-time age and stable anchor.
            expires_at_ns: None,
        };
        let (replayed, _, retained) = self
            .replay_catalog(&manifest, context, transaction)
            .inspect_err(|reason| {
                tracing::info!(
                    stage = "catalog_replay",
                    ?reason,
                    "Automatic restart save rejected"
                )
            })?;
        let proof_memory = original
            .restart_verification_budget(&replayed, context.policy)
            .map_err(|_| CacheMiss::Capacity)?;
        if proof_memory
            .checked_add(retained)
            .is_none_or(|n| n > self.0.limits.maximum_retained_bytes)
        {
            return Err(CacheMiss::Capacity);
        }
        let proof = original
            .verify_restart_replay(
                &replayed,
                context.policy,
                &self.0.identity.clock,
                context.now()?.monotonic_ns,
                proof_memory,
            )
            .map_err(|reason| corrupt("catalog_equivalence", &reason))?;
        // The old private feedback file is no longer a writer after this
        // replay-proven final drain. Reclaim its slot before the final receipt
        // is allocated; committed predecessor files are never touched here.
        self.retire_resumed_feedback(monitor, transaction)?;
        // The path is fixed before the existing feedback preflight computes its
        // path/marker allowance. Reserving it creates no file or receipt.
        let (index, path) = self.0.artifact()?;
        let feedback = self.finish_feedback_artifact(
            index,
            path,
            monitor,
            &proof,
            processed_fifo,
            retained
                .checked_add(proof_memory)
                .ok_or(CacheMiss::Capacity)?,
            context,
            transaction,
        )?;
        manifest.feedback = feedback;
        transaction.poll()?;
        proof
            .validate_freshness(context.now()?.monotonic_ns)
            .map_err(|_| CacheMiss::Expired)?;
        self.prepare_commit(
            manifest.sources,
            feedback,
            manifest.startup_seed,
            None,
            context.now()?.monotonic_ns,
            transaction,
            || {
                proof
                    .validate_freshness(context.now()?.monotonic_ns)
                    .map_err(|_| CacheMiss::Expired)
            },
        )?;
        if !defer_clean {
            self.complete_prepared_commit(transaction, || {
                proof
                    .validate_freshness(context.now()?.monotonic_ns)
                    .map_err(|_| CacheMiss::Expired)
            })?;
        }
        Ok(replayed)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_feedback_artifact(
        &self,
        index: usize,
        path: PathBuf,
        monitor: &mut Monitor,
        proof: &super::super::profile::VerifiedRestartCatalog<'_>,
        processed_fifo: u64,
        retained: usize,
        context: &ReplayContext<'_>,
        transaction: &ColdTransaction,
    ) -> Result<ArtifactReceipt> {
        // artifact() already registered the outstanding writer. This RAII
        // handle closes that registration on every budget/proof/IO failure.
        let result = (|| {
            let budget = monitor
                .restart_budget(&path)
                .map_err(|_| CacheMiss::Capacity)?;
            if retained
                .checked_add(budget.maximum_transient_bytes)
                .is_none_or(|n| n > self.0.limits.maximum_retained_bytes)
            {
                return Err(CacheMiss::Capacity);
            }
            self.0.charge(budget.maximum_disk_bytes)?;
            transaction.poll()?;
            let receipt = monitor
                .finish_for_restart(
                    proof,
                    &path,
                    processed_fifo,
                    RestartFeedbackAllowance {
                        transient_bytes: budget.maximum_transient_bytes,
                        disk_bytes: budget.maximum_disk_bytes,
                    },
                    || {
                        transaction
                            .poll()
                            .ok()
                            .and_then(|()| context.clock.now_ns())
                    },
                )
                .map_err(|reason| corrupt("feedback_handoff", &reason))?;
            transaction.poll()?;
            let actual = ArtifactReceipt {
                index,
                bytes: receipt.bytes,
                sha256: receipt.sha256,
            };
            store::verify_file(&path, &actual, transaction)?;
            if receipt.path != path || receipt.bytes > budget.maximum_disk_bytes {
                return Err(CacheMiss::Corrupt);
            }
            // All pending/session markers were removed by the original Store.
            for suffix in [".pending", ".session"] {
                let mut name = path.as_os_str().to_owned();
                name.push(suffix);
                if Path::new(&name).exists() {
                    return Err(CacheMiss::Dirty);
                }
            }
            let mut ledger = self.0.ledger.lock();
            ledger.used = ledger
                .used
                .checked_sub(budget.maximum_disk_bytes - receipt.bytes)
                .ok_or(CacheMiss::Corrupt)?;
            Ok(actual)
        })();
        self.0.close_writer(index);
        result
    }

    pub fn restore_candidate(
        &self,
        manifest: &CacheManifest,
        context: &ReplayContext<'_>,
        transaction: &mut ColdTransaction,
    ) -> Result<RestoredCostSeed> {
        transaction.poll()?;
        if context.clock.monotonic_domain() != Some(&self.0.identity.clock) {
            return Err(CacheMiss::CrossBoot);
        }
        manifest.validate(&self.0.identity, context.now()?.monotonic_ns, self.0.limits)?;
        let (snapshot, receipt, mut retained) =
            self.replay_catalog(manifest, context, transaction)?;
        let startup_seed = super::seed::restore(self, manifest, context, retained, transaction)?;
        if let Some(seed) = &startup_seed {
            retained = retained
                .checked_add(seed.retained_payload_bytes().ok_or(CacheMiss::Capacity)?)
                .ok_or(CacheMiss::Capacity)?;
        }
        let committed_feedback = self.artifact_path(manifest.generation, &manifest.feedback);
        // Resume mutates its receipt/session markers. Its target must belong to
        // this private transaction, never the generation named by manifest.json.
        // Reserve the full mutable-store lifetime bound before copying; Drop
        // closes the allocation slot but intentionally does not refund it.
        let disk = Monitor::restart_resume_budget(context.policy, &committed_feedback)
            .map_err(|_| CacheMiss::Capacity)?
            .maximum_disk_bytes;
        let private_feedback = self.reserve_external(disk, transaction)?;
        let feedback_path = &private_feedback.path;
        let budget = Monitor::restart_resume_budget(context.policy, feedback_path)
            .map_err(|_| CacheMiss::Capacity)?;
        let binding_memory = snapshot
            .restart_verification_budget(&snapshot, context.policy)
            .map_err(|_| CacheMiss::Capacity)?;
        let peak = budget
            .maximum_transient_bytes
            .checked_add(binding_memory)
            .and_then(|n| n.checked_add(retained))
            .ok_or(CacheMiss::Capacity)?;
        if peak > self.0.limits.maximum_retained_bytes {
            return Err(CacheMiss::Capacity);
        }
        let (binding, scopes) = snapshot
            .live_feedback_binding(context.policy)
            .map_err(|_| CacheMiss::Corrupt)?;
        transaction.poll()?;
        private_feedback.copy_verified_from(
            &committed_feedback,
            &manifest.feedback,
            transaction,
        )?;
        let monitor = Monitor::resume_for_restart(
            context.policy,
            feedback_path,
            binding,
            scopes,
            RestartFeedbackAllowance {
                transient_bytes: budget.maximum_transient_bytes,
                disk_bytes: budget.maximum_disk_bytes,
            },
        )
        .map_err(|_| CacheMiss::Corrupt)?;
        transaction.poll()?;
        snapshot
            .validate_live_freshness(context.now()?.monotonic_ns)
            .map_err(|_| CacheMiss::Expired)?;
        private_feedback.retain_resumed_feedback()?;
        Ok(RestoredCostSeed {
            seed: TrainingSeed {
                trainer: None,
                snapshot: Some(snapshot),
                receipt: Some(receipt),
            },
            monitor,
            startup_seed,
        })
    }

    /// Also used before saving: every metadata artifact is independently loaded
    /// from its immutable original publication prefix, not the final tail.
    pub(super) fn replay_catalog(
        &self,
        manifest: &CacheManifest,
        context: &ReplayContext<'_>,
        transaction: &mut ColdTransaction,
    ) -> Result<(Arc<EngineCostSnapshot>, SloCostProfileReceipt, usize)> {
        let mut available_bytes = context.import.max_file_bytes.get();
        // Receipts retain only fixed-size phase records and two paths per child.
        // Reserve their full catalog capacity and the final map/Vec overlap
        // before either the first loader or the merged child vector allocates.
        let path_bytes = self
            .0
            .root
            .as_os_str()
            .len()
            .checked_add(256)
            .ok_or(CacheMiss::Capacity)?;
        let headers = 128usize
            .checked_mul(
                std::mem::size_of::<file::ImportedStructuredModelV2>()
                    .checked_mul(3)
                    .and_then(|n| {
                        n.checked_add(
                            std::mem::size_of::<ferrum_types::SloStructuredChildReceiptV2>() * 3,
                        )
                    })
                    .and_then(|n| n.checked_add(path_bytes.checked_mul(8)?))
                    .and_then(|n| n.checked_add(1024))
                    .ok_or(CacheMiss::Capacity)?,
            )
            .ok_or(CacheMiss::Capacity)?;
        let headers = headers
            .checked_add(context.original_retained_bytes)
            .ok_or(CacheMiss::Capacity)?;
        if headers > self.0.limits.maximum_retained_bytes {
            return Err(CacheMiss::Capacity);
        }
        let mut children = Vec::new();
        children
            .try_reserve_exact(128)
            .map_err(|_| CacheMiss::Capacity)?;
        if children.capacity() != 128 {
            return Err(CacheMiss::Capacity);
        }
        let mut combined: Option<SloCostProfileReceipt> = None;
        let mut retained = headers;
        for source in &manifest.sources {
            transaction.poll()?;
            if source.domains.is_empty() {
                return Err(CacheMiss::Corrupt);
            }
            let path = self.artifact_path(manifest.generation, &source.profile);
            let journal_path = self.artifact_path(manifest.generation, &source.journal);
            store::verify_file(&path, &source.profile, transaction)?;
            let (raw_bytes, _) = source.encoding.original(source.journal);
            let source_bytes = usize::try_from(
                raw_bytes
                    .checked_add(source.profile.bytes)
                    .ok_or(CacheMiss::Capacity)?,
            )
            .map_err(|_| CacheMiss::Capacity)?;
            if source_bytes > available_bytes {
                return Err(CacheMiss::Capacity);
            }
            let raw = codec::decode(
                self,
                &journal_path,
                source.journal,
                source.encoding,
                available_bytes - source.profile.bytes as usize,
                retained,
                transaction,
            )?;
            let numeric = declared_numeric_bound_bytes(&raw, source.kind, transaction)?;
            let scratch = capacity::replay_peak_bytes(&raw, numeric, retained, transaction)?
                .checked_add(self.0.ledger.lock().codec_retained)
                .ok_or(CacheMiss::Capacity)?;
            if scratch > self.0.limits.maximum_retained_bytes {
                return Err(CacheMiss::Capacity);
            }
            verify_profile_path(&path, &journal_path, source).inspect_err(|reason| {
                tracing::info!(
                    stage = "profile_source_path",
                    ?reason,
                    "Automatic restart replay rejected"
                )
            })?;
            let (remaining_samples, remaining_rows) = transaction.remaining_population();
            let limits = file::CostProfileLoadLimits {
                max_file_bytes: NonZeroUsize::new(available_bytes).ok_or(CacheMiss::Capacity)?,
                max_samples: NonZeroUsize::new(remaining_samples).ok_or(CacheMiss::Capacity)?,
                max_total_shape_rows: NonZeroUsize::new(remaining_rows)
                    .ok_or(CacheMiss::Capacity)?,
                ..context.import.clone()
            };
            let now = context.now()?;
            let (loaded, mut receipt, samples, rows) = match source.kind {
                SourceKind::OwnerBlocksV7 => {
                    let imported =
                        file::load_structured_profile_v14_same_boot_selected_from_original_bytes(
                            &path,
                            &raw,
                            &journal_path,
                            context.fingerprint,
                            &limits,
                            &self.0.identity.clock,
                            now,
                            &source.domains,
                        )
                        .map_err(|reason| corrupt("source7_strict_load", &reason))?;
                    check_receipts(
                        source,
                        imported.journal_bytes,
                        imported.source_bytes,
                        imported.source_sha256,
                    )?;
                    let counts = (imported.offered_attempts, imported.total_shape_rows);
                    let (children, receipt) =
                        EngineCostSnapshot::live_owner_block_catalog_with_domain(
                            Some(&path),
                            imported,
                            Some(context.workload),
                        )
                        .map_err(|_| CacheMiss::Corrupt)?;
                    (children, receipt, counts.0, counts.1)
                }
                SourceKind::PreparedOwnersV8 => {
                    let imported =
                        file::load_structured_profile_v15_same_boot_selected_from_original_bytes(
                            &path,
                            &raw,
                            &journal_path,
                            context.fingerprint,
                            &limits,
                            &self.0.identity.clock,
                            now,
                            &source.domains,
                        )
                        .map_err(|reason| corrupt("source8_strict_load", &reason))?;
                    check_receipts(
                        source,
                        imported.journal_bytes,
                        imported.source_bytes,
                        imported.source_sha256,
                    )?;
                    let counts = (imported.offered_attempts, imported.total_shape_rows);
                    let (children, receipt) =
                        EngineCostSnapshot::live_prepared_owner_block_catalog_with_domain(
                            Some(&path),
                            imported,
                            Some(context.workload),
                        )
                        .map_err(|_| CacheMiss::Corrupt)?;
                    (children, receipt, counts.0, counts.1)
                }
            };
            transaction.poll()?;
            transaction.charge_population(
                usize::try_from(samples).map_err(|_| CacheMiss::Capacity)?,
                usize::try_from(rows).map_err(|_| CacheMiss::Capacity)?,
            )?;
            available_bytes -= source_bytes;
            for domain in &source.domains {
                if !loaded.iter().any(|c| c.domain_signature() == domain) {
                    return Err(CacheMiss::Corrupt);
                }
            }
            for child in loaded
                .into_iter()
                .filter(|c| source.domains.binary_search(c.domain_signature()).is_ok())
            {
                if children.len() == 128
                    || children
                        .iter()
                        .any(|c: &file::ImportedStructuredModelV2| c.same_population(&child))
                {
                    return Err(CacheMiss::Corrupt);
                }
                retained = retained
                    // The final catalog clones provenance while constructing
                    // bounded receipts and feedback descriptors. Models/U use
                    // their original Arc and are conservatively counted twice.
                    .checked_add(
                        child
                            .retained_payload_bytes()
                            .ok_or(CacheMiss::Capacity)?
                            .checked_mul(2)
                            .ok_or(CacheMiss::Capacity)?,
                    )
                    .ok_or(CacheMiss::Capacity)?;
                children.push(child);
            }
            let v2 = receipt
                .structured_whole_wave_v2
                .as_mut()
                .ok_or(CacheMiss::Corrupt)?;
            v2.children
                .retain(|c| source.domains.binary_search(&c.domain_signature).is_ok());
            v2.child_count = v2.children.len();
            // The source receipt above remains the original complete replay;
            // this outer object reports the selected restart inventory.
            receipt.bucket_count = v2.child_count;
            receipt.recorded_samples = usize::try_from(
                v2.children
                    .iter()
                    .try_fold(0u64, |sum, c| sum.checked_add(c.reserved_members))
                    .ok_or(CacheMiss::Capacity)?,
            )
            .map_err(|_| CacheMiss::Capacity)?;
            v2.source_inventory_sha256 = None;
            if let Some(prior) = combined.as_mut() {
                let dst = prior
                    .structured_whole_wave_v2
                    .as_mut()
                    .ok_or(CacheMiss::Corrupt)?;
                dst.children.extend(v2.children.drain(..));
                dst.child_count = dst.children.len();
                dst.total_imported_bytes = dst
                    .total_imported_bytes
                    .checked_add(v2.total_imported_bytes)
                    .ok_or(CacheMiss::Capacity)?;
                dst.total_shape_rows = dst
                    .total_shape_rows
                    .checked_add(v2.total_shape_rows)
                    .ok_or(CacheMiss::Capacity)?;
                dst.artifact_kind = ferrum_types::SloStructuredArtifactKindV2::CatalogV1;
                dst.source_inventory_sha256 = None;
                prior.bucket_count = dst.child_count;
                prior.recorded_samples = prior
                    .recorded_samples
                    .checked_add(receipt.recorded_samples)
                    .ok_or(CacheMiss::Capacity)?;
                prior.offered_samples = prior
                    .offered_samples
                    .checked_add(receipt.offered_samples)
                    .ok_or(CacheMiss::Capacity)?;
                prior.generated_unix_ns = prior.generated_unix_ns.max(receipt.generated_unix_ns);
                prior.loaded_unix_ns = prior.loaded_unix_ns.max(receipt.loaded_unix_ns);
                prior.oldest_imported_age_ns = prior
                    .oldest_imported_age_ns
                    .zip(receipt.oldest_imported_age_ns)
                    .map(|(a, b)| a.max(b));
                prior.newest_imported_age_ns = prior
                    .newest_imported_age_ns
                    .zip(receipt.newest_imported_age_ns)
                    .map(|(a, b)| a.min(b));
            } else {
                combined = Some(receipt);
            }
        }
        let at = context.now()?.monotonic_ns;
        let snapshot = EngineCostSnapshot::live_catalog_with_domain(
            children,
            context.fingerprint.clone(),
            structured_epoch::View::initial(),
            at,
            Some(context.workload),
        )
        .map_err(|_| CacheMiss::Expired)?;
        let receipt = combined.ok_or(CacheMiss::Corrupt)?;
        snapshot
            .validate_live_receipt(&receipt)
            .map_err(|_| CacheMiss::Corrupt)?;
        transaction.poll()?;
        Ok((snapshot, receipt, retained))
    }
}

fn check_receipts(
    source: &CachedSource,
    journal_bytes: u64,
    prefix_bytes: u64,
    prefix_sha256: [u8; 32],
) -> Result<()> {
    if journal_bytes != source.encoding.original(source.journal).0
        || (prefix_bytes, prefix_sha256) != (source.checkpoint_bytes, source.checkpoint_sha256)
    {
        return Err(CacheMiss::Corrupt);
    }
    Ok(())
}
#[derive(Deserialize)]
struct ProfilePath {
    source_path: PathBuf,
    journal_bytes: u64,
    journal_sha256: [u8; 32],
}
fn verify_profile_path(path: &Path, expected: &Path, source: &CachedSource) -> Result<()> {
    let file = File::open(path).map_err(io)?;
    let parsed: ProfilePath =
        serde_json::from_reader(BufReader::new(file.take(source.profile.bytes)))
            .map_err(|_| CacheMiss::Corrupt)?;
    // The referenced path is still exactly the already verified managed file.
    // Canonicalization only normalizes OS path aliases used by the exporter.
    let expected = expected.canonicalize().map_err(io)?;
    if parsed.source_path != expected
        || (parsed.journal_bytes, parsed.journal_sha256) != source.encoding.original(source.journal)
    {
        return Err(CacheMiss::Corrupt);
    }
    Ok(())
}
fn corrupt(stage: &'static str, reason: &impl std::fmt::Debug) -> CacheMiss {
    struct Bounded(String);
    impl std::fmt::Write for Bounded {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            let mut end = value.len().min(512 - self.0.len());
            while !value.is_char_boundary(end) {
                end -= 1;
            }
            self.0.push_str(&value[..end]);
            Ok(())
        }
    }
    let mut text = Bounded(String::with_capacity(512));
    let _ = std::fmt::write(&mut text, format_args!("{reason:?}"));
    tracing::info!(stage, reason = %text.0, "Automatic restart evidence rejected");
    CacheMiss::Corrupt
}
#[derive(Default, Deserialize)]
struct NumericBound {
    #[serde(default)]
    maximum_retained_numeric_bytes: usize,
    #[serde(default)]
    maximum_discovery_bytes: usize,
}
#[derive(Deserialize)]
struct HeaderBound {
    declaration: DeclarationBound,
}
#[derive(Default, Deserialize)]
struct DeclarationBound {
    #[serde(default)]
    maximum_retained_numeric_bytes: usize,
    #[serde(default)]
    maximum_discovery_bytes: usize,
    #[serde(default)]
    population: NumericBound,
}
fn declared_numeric_bound(
    path: &Path,
    kind: SourceKind,
    maximum: u64,
    transaction: &ColdTransaction,
) -> Result<usize> {
    // This bounded selective parse grants no validity. The full original loader
    // still validates schema, signature, every raw member and numerical phase.
    let file = File::open(path).map_err(io)?;
    let mut decoder = serde_json::Deserializer::from_reader(BufReader::new(file.take(maximum)));
    let header = HeaderBound::deserialize(&mut decoder).map_err(|_| CacheMiss::Corrupt)?;
    let bound = match kind {
        SourceKind::OwnerBlocksV7 => NumericBound {
            maximum_retained_numeric_bytes: header.declaration.maximum_retained_numeric_bytes,
            maximum_discovery_bytes: header.declaration.maximum_discovery_bytes,
        },
        SourceKind::PreparedOwnersV8 => header.declaration.population,
    };
    transaction.poll()?;
    if bound.maximum_retained_numeric_bytes == 0 {
        return Err(CacheMiss::Corrupt);
    }
    bound
        .maximum_retained_numeric_bytes
        .checked_add(bound.maximum_discovery_bytes)
        .ok_or(CacheMiss::Capacity)
}

fn declared_numeric_bound_bytes(
    raw: &[u8],
    kind: SourceKind,
    transaction: &ColdTransaction,
) -> Result<usize> {
    let end = raw
        .iter()
        .position(|b| *b == b'\n')
        .ok_or(CacheMiss::Corrupt)?;
    if end > 8 * 1024 * 1024 {
        return Err(CacheMiss::Capacity);
    }
    let header: HeaderBound =
        serde_json::from_slice(&raw[..end]).map_err(|_| CacheMiss::Corrupt)?;
    let bound = match kind {
        SourceKind::OwnerBlocksV7 => NumericBound {
            maximum_retained_numeric_bytes: header.declaration.maximum_retained_numeric_bytes,
            maximum_discovery_bytes: header.declaration.maximum_discovery_bytes,
        },
        SourceKind::PreparedOwnersV8 => header.declaration.population,
    };
    transaction.poll()?;
    if bound.maximum_retained_numeric_bytes == 0 {
        return Err(CacheMiss::Corrupt);
    }
    bound
        .maximum_retained_numeric_bytes
        .checked_add(bound.maximum_discovery_bytes)
        .ok_or(CacheMiss::Capacity)
}
