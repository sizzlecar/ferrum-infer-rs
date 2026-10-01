//! Private source8 checkpoint -> original runtime clock/domain -> worker cut.
//! This entry accepts neither a wire DTO nor an arbitrarily assembled catalog.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    checkpoint::CostCheckpointWaiter, profile::EngineCostSnapshot, EngineCostRuntime,
};
use ferrum_scheduler::implementations::continuous::cost_profile::{
    StructuredPreparedOwnerBlockCheckpointV8, StructuredServiceClockV7,
};

/// Process-local authority issued only before live calibration starts. Each
/// activation still carries a complete independently replayable source8.
pub(in crate::continuous_engine::inner) struct StartupOwnerSeries {
    authority: Arc<StartupSeriesAuthority>,
    next_ordinal: usize,
}

pub(super) struct StartupSeriesAuthority {
    maximum_sources: usize,
    deadline: tokio::time::Instant,
    closed: AtomicBool,
    latest_installed_epoch: AtomicU64,
}
impl StartupSeriesAuthority {
    pub(super) fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

pub(in crate::continuous_engine::inner::cost_observation) struct StartupSeriesActivation {
    authority: Arc<StartupSeriesAuthority>,
    ordinal: usize,
}
impl StartupSeriesActivation {
    pub(in crate::continuous_engine::inner::cost_observation) fn validate(
        &self,
    ) -> Result<(), FerrumError> {
        if self.authority.closed()
            || tokio::time::Instant::now() >= self.authority.deadline
            || self.ordinal >= self.authority.maximum_sources
        {
            return Err(error("startup source series closed, expired or exhausted"));
        }
        Ok(())
    }
    pub(super) fn matches(&self, authority: &Arc<StartupSeriesAuthority>) -> bool {
        Arc::ptr_eq(&self.authority, authority)
    }
    pub(super) fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn installed(&self, epoch: u64) {
        debug_assert_ne!(epoch, 0);
        self.authority
            .latest_installed_epoch
            .store(epoch, Ordering::Release);
    }
}

pub(in crate::continuous_engine::inner::cost_observation) struct StartupCatalogActivation {
    pub publication: publication::Publication,
    pub series: Option<StartupSeriesActivation>,
}

impl StartupOwnerSeries {
    /// Sole-worker acknowledgement, independent of a caller keeping its waiter.
    /// Revocation cannot erase an installation that already completed.
    pub(in crate::continuous_engine::inner) fn installed_epoch(&self) -> Option<u64> {
        match self
            .authority
            .latest_installed_epoch
            .load(Ordering::Acquire)
        {
            0 => None,
            epoch => Some(epoch),
        }
    }
    pub(in crate::continuous_engine::inner) fn ensure_open(&self) -> Result<(), FerrumError> {
        self.activation()?.validate()
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn activation(
        &self,
    ) -> Result<StartupSeriesActivation, FerrumError> {
        let activation = StartupSeriesActivation {
            authority: Arc::clone(&self.authority),
            ordinal: self.next_ordinal,
        };
        activation.validate()?;
        Ok(activation)
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn accepted_activation(&mut self) {
        self.next_ordinal += 1;
    }
}

impl EngineCostRuntime {
    #[cfg(test)]
    pub(in crate::continuous_engine::inner) fn startup_series_children_for_test(
        &self,
    ) -> Result<Vec<ImportedStructuredModelV2>, FerrumError> {
        self.training.live_catalog_children(
            self.clock
                .now_ns()
                .ok_or_else(|| error("startup series test clock unavailable"))?,
        )
    }
    pub(in crate::continuous_engine::inner) fn begin_prepared_owner_series(
        &self,
        maximum_sources: usize,
        deadline: tokio::time::Instant,
    ) -> Result<StartupOwnerSeries, FerrumError> {
        if maximum_sources == 0
            || maximum_sources > 65_536
            || tokio::time::Instant::now() >= deadline
        {
            return Err(error("invalid bounded startup source series"));
        }
        let live = self
            .training
            .live
            .as_ref()
            .ok_or_else(|| error("startup series requires automatic calibration"))?;
        let controller = live
            .automatic
            .as_ref()
            .ok_or_else(|| error("startup series requires automatic calibration"))?;
        let mut state = controller.state.lock();
        let population = live.active.read().audit();
        if controller.requested.load(Ordering::Acquire)
            || controller.stopped.load(Ordering::Acquire)
            || state.generation != 0
            || state.install_pending
            || state.startup_publication.is_some()
            || state.startup_series.is_some()
            || population.issued != 0
            || population.retired != 0
        {
            return Err(error(
                "startup series requires an unused automatic lifecycle",
            ));
        }
        let authority = Arc::new(StartupSeriesAuthority {
            maximum_sources,
            deadline,
            closed: AtomicBool::new(false),
            latest_installed_epoch: AtomicU64::new(0),
        });
        state.startup_series = Some(Arc::clone(&authority));
        Ok(StartupOwnerSeries {
            authority,
            next_ordinal: 0,
        })
    }

    pub(in crate::continuous_engine::inner) fn finish_prepared_owner_series(
        &self,
        series: &StartupOwnerSeries,
    ) -> Result<(), FerrumError> {
        let controller = self
            .training
            .live
            .as_ref()
            .and_then(|live| live.automatic.as_ref())
            .ok_or_else(|| error("startup series requires automatic calibration"))?;
        let state = controller.state.lock();
        if state
            .startup_series
            .as_ref()
            .is_none_or(|authority| !Arc::ptr_eq(authority, &series.authority))
        {
            return Err(error("startup series belongs to another runtime"));
        }
        // A queued but not installed activation is revoked; already installed
        // children keep their original sources, age and catalog gate.
        series.authority.closed.store(true, Ordering::Release);
        Ok(())
    }
    /// Keep the first bounded startup failure visible in the automatic audit.
    /// This does not replace an accepted installation or a published epoch.
    pub(in crate::continuous_engine::inner) fn record_prepared_owner_failure(&self, reason: &str) {
        let Some(live) = self.training.live.as_ref() else {
            return;
        };
        let Some(controller) = live.automatic.as_ref() else {
            return;
        };
        let mut state = controller.state.lock();
        if state.generation != 0 || state.install_pending || state.startup_publication.is_some() {
            return;
        }
        let mut end = reason.len().min(2048);
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        let reason = reason[..end].to_owned();
        state.startup_publication = Some(Err(reason.clone()));
        *live.publication_error.lock() = Some(reason);
    }

    /// Prepare immutable source15 metadata on the exclusive startup caller,
    /// then let the sole worker install after the original accepted FIFO cut.
    /// A successful enqueue is not publication: the caller must wait for the
    /// checkpoint's explicit catalog_activation result before enabling service.
    pub(in crate::continuous_engine::inner) fn activate_prepared_owner_checkpoint(
        &self,
        checkpoint: StructuredPreparedOwnerBlockCheckpointV8,
    ) -> Result<CostCheckpointWaiter, FerrumError> {
        self.activate_prepared_owner_checkpoint_inner(checkpoint, None)
    }

    pub(in crate::continuous_engine::inner) fn activate_prepared_owner_checkpoint_in_series(
        &self,
        checkpoint: StructuredPreparedOwnerBlockCheckpointV8,
        series: &mut StartupOwnerSeries,
    ) -> Result<CostCheckpointWaiter, FerrumError> {
        self.activate_prepared_owner_checkpoint_inner(checkpoint, Some(series))
    }

    fn activate_prepared_owner_checkpoint_inner(
        &self,
        checkpoint: StructuredPreparedOwnerBlockCheckpointV8,
        series: Option<&mut StartupOwnerSeries>,
    ) -> Result<CostCheckpointWaiter, FerrumError> {
        let live =
            self.training.live.as_ref().ok_or_else(|| {
                error("prepared startup checkpoint requires automatic calibration")
            })?;
        let controller = live
            .automatic
            .as_ref()
            .ok_or_else(|| error("prepared startup checkpoint requires automatic calibration"))?;
        let ExecutorCostIdentityAvailability::Known(identity) = &self.identity else {
            return Err(error(
                "prepared startup checkpoint requires executor identity",
            ));
        };
        let domain = self.workload_domain().ok_or_else(|| {
            error("prepared startup checkpoint requires the original workload domain")
        })?;
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA
            || !domain.matches_execution_identity(identity)
        {
            return Err(error("prepared startup runtime identity/domain differs"));
        }
        let fingerprint = ExecutionFingerprint {
            model_weights: identity.model_weights,
            numerical_policy: identity.numerical_policy,
            device_runtime: identity.device_runtime,
            execution_config: identity.execution_config,
        };
        let cutoff = checkpoint.accepted_fifo_cutoff();
        let now = ExportClockReading::closing(self.clock.as_ref()).map_err(error)?;
        let installed = StructuredServiceClockV7 {
            monotonic_ns: now.monotonic_ns,
            wall_unix_ns: now.wall_unix_ns,
        };
        let limits = profile::load_limits(&controller.import);
        let budget = controller.settings.maximum_encoded_source_bytes;
        let imported = match self.clock.monotonic_domain() {
            Some(domain) => {
                checkpoint.activate_same_boot_memory_streaming(installed, domain, &limits, budget)
            }
            None => checkpoint.activate_same_process_memory_streaming(installed, &limits, budget),
        }
        .map_err(error)?;
        if imported
            .children
            .iter()
            .any(|child| child.fingerprint() != &fingerprint)
        {
            return Err(error(
                "prepared startup source executor fingerprint differs",
            ));
        }
        let (children, receipt) =
            EngineCostSnapshot::live_prepared_owner_block_catalog_with_domain(
                None,
                imported,
                Some(domain),
            )?;
        // The sink atomically checks Closing/Busy/current accepted cutoff.
        // The worker independently checks generation zero, no live tickets,
        // pending activation, source age, retention and final epoch freshness.
        let publication = publication::Publication { children, receipt };
        match series {
            Some(series) => {
                self.sink
                    .request_catalog_activation_in_series(cutoff, publication, series)
            }
            None => self
                .sink
                .request_catalog_activation(cutoff, publication)
                .map_err(error),
        }
    }
}

impl EngineCostRuntime {
    #[cfg(test)]
    pub(in crate::continuous_engine::inner) fn automatic_original_header_for_test(
        &self,
    ) -> Option<
        ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceHeaderV7,
    > {
        self.training
            .live
            .as_ref()?
            .automatic
            .as_ref()?
            .state
            .lock()
            .blocks
            .as_ref()
            .map(|blocks| blocks.header_for_test())
    }

    /// Declaration only: cache restoration cannot turn this into learned support.
    pub(in crate::continuous_engine::inner) fn startup_algorithm_seed(&self)
        -> Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>
    {
        self.training
            .live
            .as_ref()?
            .automatic
            .as_ref()?
            .state
            .lock()
            .cold_algorithm_seed
            .clone()
    }

    pub(in crate::continuous_engine::inner) fn install_startup_algorithm_seed(
        &self,
        seed: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1,
    ) -> Result<(), FerrumError> {
        let live = self
            .training
            .live
            .as_ref()
            .ok_or_else(|| error("algorithm seed requires automatic runtime"))?;
        live.install_startup_algorithm_seed(seed.clone())?;
        if let Some(reuse) = &self.reuse {
            if let Err(reason) = reuse.note_startup_seed(&seed) {
                tracing::warn!(
                    ?reason,
                    "Checked algorithm seed installed but restart persistence unavailable"
                );
            }
        }
        Ok(())
    }
}

impl LiveCalibration {
    pub(in crate::continuous_engine::inner::cost_observation) fn install_startup_algorithm_seed(
        &self,
        seed: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1,
    ) -> Result<(), FerrumError> {
        let live = self;
        let controller = live
            .automatic
            .as_ref()
            .ok_or_else(|| error("algorithm seed requires automatic lifecycle"))?;
        if live
            .workload_domain
            .as_ref()
            .is_none_or(|d| d.sha256() != seed.workload_domain_signature())
        {
            return Err(error(
                "algorithm seed belongs to another workload domain or fingerprint",
            ));
        }
        seed.validate_budget(
            live.declaration.settings.max_axes,
            controller.settings.maximum_discovery_bytes.get(),
        )
        .map_err(|e| error(format!("algorithm seed capacity: {e:?}")))?;
        let mut state = controller.state.lock();
        if state.generation != 0
            || controller.requested.load(Ordering::Acquire)
            || controller.stopped.load(Ordering::Acquire)
            || state.install_pending
            || state.startup_install_pending
        {
            return Err(error(
                "algorithm seed must precede the original online generation",
            ));
        }
        if let Some(old) = &state.cold_algorithm_seed {
            if old == &seed {
                return Ok(());
            }
            if !seed.contains_universe(old) {
                return Err(error("algorithm seed would remove checked classes"));
            }
            old.retained_payload_bytes()
                .and_then(|n| n.checked_add(seed.retained_payload_bytes()?))
                .filter(|n| *n <= controller.settings.maximum_discovery_bytes.get())
                .ok_or_else(|| {
                    error("algorithm seed replacement peak exceeds discovery capacity")
                })?;
        }
        state.cold_algorithm_seed = Some(seed);
        Ok(())
    }
}
