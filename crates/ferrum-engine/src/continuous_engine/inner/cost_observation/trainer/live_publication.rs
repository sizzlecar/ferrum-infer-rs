//! Worker-only installation of a durably written and qualified catalog.
//! The hot path only reads one scalar origin and immutable snapshot/gate.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::ImportedStructuredModelV2;
use ferrum_types::{FerrumError, SloCostProfileReceipt};

enum CatalogPublication {
    QualifiedSource(SloCostProfileReceipt),
    ExistingCatalog { previous_epoch: u64 },
}

impl CostTrainingState {
    pub(super) fn install_startup_catalog(
        &self,
        activation: super::super::live_calibration::StartupCatalogActivation,
    ) -> Result<u64, FerrumError> {
        let live = self.live.as_ref().ok_or_else(|| {
            FerrumError::config("startup catalog requires automatic observation runtime")
        })?;
        live.begin_startup_catalog(activation.series.as_ref())?;
        let preserve_input_coverage = activation.series.is_some();
        let result = self
            .install_live_publication_guarded(
                live,
                Some(activation.publication),
                preserve_input_coverage,
                || {
                    activation
                        .series
                        .as_ref()
                        .map_or(Ok(()), |series| series.validate())
                },
            )
            .and_then(|installed| {
                if !installed {
                    return Err(FerrumError::config(
                        "startup catalog installation did not publish",
                    ));
                }
                self.snapshot()
                    .map(|snapshot| snapshot.model_version())
                    .ok_or_else(|| {
                        FerrumError::internal("installed startup catalog snapshot missing")
                    })
            });
        live.finish_startup_catalog(&result);
        if let (Ok(epoch), Some(series)) = (&result, activation.series.as_ref()) {
            series.installed(*epoch);
        }
        result
    }

    pub(super) fn finish_live_calibration(&self, processed: u64) -> bool {
        if let Some(live) = &self.live {
            let mut sealed = false;
            for _ in 0..live.automatic_finish_step_limit() {
                let result = live
                    .finish(self.clock.as_ref(), processed)
                    .and_then(|publication| self.install_live_publication(live, publication));
                let terminal = matches!(result, Ok(false));
                live.note_publication(result);
                // Waiting is not a completed source transition and cannot
                // consume the bounded finish-step allowance or seal feedback.
                if live.automatic_computation_pending() {
                    return false;
                }
                if terminal && !live.automatic_finish_pending() {
                    sealed = true;
                    break;
                }
            }
            if !sealed {
                live.note_publication(Err(FerrumError::config(
                    "automatic shutdown exceeded retained source completion bound",
                )));
            }
        }
        self.prune_expired_live_catalog_audited();
        true
    }

    pub(super) fn advance_live_calibration(&self, processed: u64) {
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.prune_expired_live_catalog_audited();
        #[cfg(test)]
        let after_prune = std::time::Instant::now();
        if let Some(live) = &self.live {
            let publication = live.advance(self.clock.as_ref(), processed);
            #[cfg(test)]
            let after_advance = std::time::Instant::now();
            let result = publication
                .and_then(|publication| self.install_live_publication(live, publication));
            #[cfg(test)]
            let after_install = std::time::Instant::now();
            live.note_publication(result);
            #[cfg(test)]
            {
                std::thread_local! {
                    static SLOW_TURNS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
                }
                let elapsed = started.elapsed();
                if elapsed >= std::time::Duration::from_millis(100)
                    && SLOW_TURNS.with(|count| {
                        let current = count.get();
                        if current >= 8 {
                            false
                        } else {
                            count.set(current + 1);
                            true
                        }
                    })
                {
                    eprintln!("original live advance phases: processed={processed} total_ns={} prune_ns={} advance_ns={} install_ns={} note_ns={}", elapsed.as_nanos(), after_prune.duration_since(started).as_nanos(), after_advance.duration_since(after_prune).as_nanos(), after_install.duration_since(after_advance).as_nanos(), after_install.elapsed().as_nanos());
                }
            }
        }
    }

    pub(super) fn prune_expired_live_catalog_audited(&self) {
        if let Err(error) = self.prune_expired_live_catalog() {
            tracing::warn!(
                target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                %error, "Expired structured catalog transition unavailable"
            );
        }
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn prune_expired_live_catalog(
        &self,
    ) -> Result<bool, FerrumError> {
        let Some(old) = self.snapshot() else {
            return Ok(false);
        };
        let now = self
            .clock
            .now_ns()
            .ok_or_else(|| FerrumError::config("catalog expiry clock unavailable"))?;
        let Some((previous_children, children)) = old.expired_live_subset(now)? else {
            return Ok(false);
        };
        let previous_epoch = old.model_version();
        let current_children = children.len();
        let next_epoch = if children.is_empty() {
            let _trainer = self.trainer.lock();
            let _feedback = self.feedback.lock();
            let mut current = self.snapshot.write();
            if !current.as_ref().is_some_and(|snapshot| {
                snapshot.current() && snapshot.model_version() == previous_epoch
            }) {
                return Err(FerrumError::config(
                    "catalog changed during expiry transition",
                ));
            }
            self.close_structured_epoch();
            *current = None;
            self.sink.set_source_generation(0);
            0
        } else {
            self.publish_catalog_guarded(
                children.clone(),
                CatalogPublication::ExistingCatalog { previous_epoch },
                now,
                || Ok(()),
            )?
        };
        self.audit.lock().catalog_expiry = Some(super::super::audit::CatalogExpiryTransition {
            observed_at_ns: now,
            previous_runtime_epoch: previous_epoch,
            current_runtime_epoch: next_epoch,
            previous_children,
            current_children,
            removed_expired_children: previous_children - current_children,
            reason: super::super::audit::CatalogExpiryReason::OriginalSampleAgeExpired,
        });
        // Preserve source-slot accounting only after the immutable catalog
        // commit, including the all-expired Unknown state. Its audit is already
        // committed if optional archive cleanup subsequently fails.
        if let Some(live) = &self.live {
            live.note_automatic_catalog_origins(&children)?;
        }
        if children.is_empty() {
            if let Some(reuse) = &self.reuse {
                reuse.note_empty_catalog().map_err(|reason| {
                    FerrumError::config(format!("empty restart catalog cleanup: {reason:?}"))
                })?;
            }
        }
        Ok(true)
    }

    fn install_live_publication(
        &self,
        live: &super::super::live_calibration::LiveCalibration,
        publication: Option<super::super::live_calibration::Publication>,
    ) -> Result<bool, FerrumError> {
        self.install_live_publication_guarded(live, publication, false, || Ok(()))
    }

    fn install_live_publication_guarded(
        &self,
        live: &super::super::live_calibration::LiveCalibration,
        publication: Option<super::super::live_calibration::Publication>,
        preserve_input_coverage: bool,
        validate_activation: impl Fn() -> Result<(), FerrumError>,
    ) -> Result<bool, FerrumError> {
        let Some(mut publication) = publication else {
            return Ok(false);
        };
        let now = self
            .clock
            .now_ns()
            .ok_or_else(|| FerrumError::config("live publication clock unavailable"))?;
        let mut children = self.live_catalog_children(now)?;
        // Only coverage conflicts protect an old origin from this activation's
        // retention pass. Other origins keep the existing bounded eviction rule.
        let mut protected_origins = [[0; 32]; 128];
        let mut protected_count = 0;
        if preserve_input_coverage {
            if publication.children.len() > protected_origins.len() {
                return Err(FerrumError::config("startup catalog child capacity"));
            }
            let mut keep = [true; 128];
            for (index, new) in publication.children.iter().enumerate() {
                new.is_current_local(now).map_err(|error| {
                    FerrumError::config(format!("startup source child is not current: {error:?}"))
                })?;
                if let Some(old) = children.iter().find(|old| old.same_population(new)) {
                    if !new.preserves_physical_input_coverage(old) {
                        keep[index] = false;
                        protected_origins[protected_count] = old.provenance().capture_identity;
                        protected_count += 1;
                        tracing::info!(
                            old_domain = ?old.domain_signature(),
                            old_capture = ?old.provenance().capture_identity,
                            uninstalled_domain = ?new.domain_signature(),
                            uninstalled_capture = ?new.provenance().capture_identity,
                            "Startup source retains original child whose structural coverage would shrink"
                        );
                    }
                }
            }
            if protected_count != 0 {
                EngineCostSnapshot::retain_startup_catalog_receipt(
                    &mut publication.receipt,
                    &publication.children,
                    &keep[..publication.children.len()],
                )?;
                let mut index = 0;
                publication.children.retain(|_| {
                    let retain = keep[index];
                    index += 1;
                    retain
                });
            }
        }
        let retained = if protected_count == 0 {
            live.prepare_retention(&children, &publication.children)
        } else {
            live.prepare_retention_preserving(
                &children,
                &publication.children,
                &protected_origins[..protected_count],
            )
        }?;
        if let Some(origins) = retained {
            if protected_origins[..protected_count]
                .iter()
                .any(|origin| !origins.contains(origin))
            {
                return Err(FerrumError::config(
                    "startup extension retention would remove protected original structural coverage",
                ));
            }
            children.retain(|child| origins.contains(&child.provenance().capture_identity));
        }
        children.retain(|old| {
            !publication
                .children
                .iter()
                .any(|new| old.same_population(new))
        });
        children.extend(publication.children);
        self.publish_live_catalog_guarded(children, publication.receipt, now, validate_activation)?;
        Ok(true)
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn close_structured_epoch(&self) {
        if let Some(epoch) = &self.structured_epoch {
            epoch.activate(0);
        }
    }
    /// Source6 may merge this current subset with freshly qualified owners.
    /// Expired/revoked entries are never carried forward by this helper.
    pub(in crate::continuous_engine::inner::cost_observation) fn live_catalog_children(
        &self,
        now: u64,
    ) -> Result<Vec<ImportedStructuredModelV2>, FerrumError> {
        self.snapshot()
            .map_or_else(|| Ok(Vec::new()), |value| value.live_children(now))
    }
    /// `children` is the complete intended next catalog. `receipt` describes
    /// the newly persisted source; preserved children keep their own originals.
    /// Only the original background worker calls this after dropping drain locks.
    pub(in crate::continuous_engine::inner::cost_observation) fn publish_live_catalog(
        &self,
        children: Vec<ImportedStructuredModelV2>,
        receipt: SloCostProfileReceipt,
        now: u64,
    ) -> Result<u64, FerrumError> {
        self.publish_live_catalog_guarded(children, receipt, now, || Ok(()))
    }

    fn publish_live_catalog_guarded(
        &self,
        children: Vec<ImportedStructuredModelV2>,
        receipt: SloCostProfileReceipt,
        now: u64,
        validate_activation: impl Fn() -> Result<(), FerrumError>,
    ) -> Result<u64, FerrumError> {
        self.publish_catalog_guarded(
            children,
            CatalogPublication::QualifiedSource(receipt),
            now,
            validate_activation,
        )
    }

    fn publish_catalog_guarded(
        &self,
        children: Vec<ImportedStructuredModelV2>,
        publication: CatalogPublication,
        now: u64,
        validate_activation: impl Fn() -> Result<(), FerrumError>,
    ) -> Result<u64, FerrumError> {
        validate_activation()?;
        // Same lock order as consume_batch. This also makes deterministic test
        // callers serialize with feedback; inference never takes this lock.
        let _trainer = self.trainer.lock();
        let epoch = self
            .structured_epoch
            .as_ref()
            .ok_or_else(|| FerrumError::config("live publication requires V2"))?;
        let fingerprint = self
            .source_fingerprint
            .clone()
            .ok_or_else(|| FerrumError::config("live publication requires executor fingerprint"))?;
        let local_now = self
            .clock
            .now_ns()
            .filter(|v| *v >= now)
            .ok_or_else(|| FerrumError::config("live publication clock moved backwards"))?;
        let next_epoch = epoch.reserve()?;
        let mut next = EngineCostSnapshot::live_catalog_with_domain(
            children,
            fingerprint,
            epoch.view(next_epoch),
            local_now,
            self.live.as_ref().and_then(|live| live.workload_domain()),
        )?;
        let mut feedback = self.feedback.lock();
        let old = self.snapshot.read().clone();
        match &publication {
            CatalogPublication::QualifiedSource(receipt) => next.validate_live_receipt(receipt)?,
            CatalogPublication::ExistingCatalog { previous_epoch } => {
                let old = old
                    .as_ref()
                    .filter(|old| old.model_version() == *previous_epoch)
                    .ok_or_else(|| FerrumError::config("expiry publication predecessor changed"))?;
                old.validate_retained_subset(&next)?;
            }
        }
        let retained = old
            .as_ref()
            .map_or_else(Vec::new, |old| old.retained_live_domains(&next));
        // Capture the original predecessor while its gate still describes the
        // pre-install state. Serialization is optional and worker-only; a later
        // validation/persistence failure must never emit an installed event.
        let installation_diagnostic = tracing::enabled!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            tracing::Level::DEBUG
        )
        .then(|| match &publication {
            CatalogPublication::QualifiedSource(receipt) => {
                Some(next.installation_diagnostic(old.as_deref(), receipt, local_now, &retained))
            }
            CatalogPublication::ExistingCatalog { .. } => None,
        })
        .flatten();
        let validate = || {
            validate_activation()?;
            let at = self
                .clock
                .now_ns()
                .filter(|v| *v >= local_now)
                .ok_or_else(|| {
                    FerrumError::config("publication clock unavailable after persistence")
                })?;
            next.validate_live_freshness(at)
        };
        if let Some(monitor) = feedback.as_mut() {
            let (binding, scopes) = next.live_feedback_binding(monitor.policy())?;
            let view =
                monitor.rebind_structured(binding, scopes, &retained, next_epoch, validate)?;
            next = next
                .with_feedback(view)
                .ok_or_else(|| FerrumError::config("replacement feedback adapter missing"))?;
        } else {
            validate()?;
            if let Some(policy) = &self.automatic_feedback {
                let mut monitor = next
                    .open_structured_feedback(policy)?
                    .ok_or_else(|| FerrumError::config("automatic feedback policy missing"))?;
                monitor.initialize_structured_epoch(epoch.clone(), next_epoch)?;
                // No sample has entered this monitor. Old queued source
                // generations remain excluded by the existing consume guard.
                next.validate_live_freshness(
                    self.clock
                        .now_ns()
                        .filter(|at| *at >= local_now)
                        .ok_or_else(|| FerrumError::config("initial feedback clock unavailable"))?,
                )?;
                validate_activation()?;
                next = next
                    .with_feedback(monitor.current_view())
                    .ok_or_else(|| FerrumError::config("initial feedback adapter missing"))?;
                *feedback = Some(monitor);
            }
        }
        // Readers either see the old gate or the complete new Arc. Old retained
        // Arcs and published plans fail the same final model-version/gate check.
        // No fallible state changes occur after this point. Optional diagnostic
        // output happens only after committing and releasing all these locks.
        let mut snapshot = self.snapshot.write();
        epoch.activate(0);
        *snapshot = Some(next);
        self.sink.set_source_generation(next_epoch);
        if let CatalogPublication::QualifiedSource(mut receipt) = publication {
            receipt.model_version = next_epoch;
            *self.published_receipt.lock() = Some(receipt);
        }
        epoch.activate(next_epoch);
        drop(snapshot);
        drop(feedback);
        drop(_trainer);
        if let (Some(reuse), Some(installed)) = (&self.reuse, self.snapshot()) {
            if let Err(reason) = reuse.note_catalog(&installed) {
                tracing::info!(?reason, "Automatic restart catalog tracking unavailable");
            }
        }
        if let Some(diagnostic) = installation_diagnostic {
            match diagnostic {
                Ok(payload) => tracing::debug!(
                    target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                    event = "structured_catalog_installed_v1",
                    installation = %payload,
                    "qualified structured catalog installed"
                ),
                Err(reason) => tracing::debug!(
                    target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                    event = "structured_catalog_installation_diagnostic_incomplete_v1",
                    installed_runtime_epoch = next_epoch,
                    reason,
                    "catalog installed; original receipt diagnostic unavailable"
                ),
            }
        }
        Ok(next_epoch)
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn published_catalog_receipt(
        &self,
    ) -> Option<SloCostProfileReceipt> {
        self.published_receipt.lock().clone()
    }
}
