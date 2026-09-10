//! Publication of bounded, immutable reusable-program catalog snapshots.

use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn reusable_execution_catalog_snapshot(&self) -> serde_json::Value {
        let snapshot = self.reusable_execution_catalog.read().clone();
        let Some(catalog) = snapshot else {
            return serde_json::json!({"state": "not_installed"});
        };
        serde_json::json!({
            "state": "installed",
            "catalog_lifetime": self.reusable_execution_startup_plan.as_ref().map(|plan| plan.device_plan.catalog_lifetime()),
            "lane_epoch": catalog.lane_epoch,
            "current_lane_epoch": self.lane.reusable_execution_epoch(),
            "refresh_pending": self.reusable_execution_catalog_refresh_needed.load(Ordering::Acquire),
            "maximum_device_executables": self.reusable_execution_startup_plan.as_ref().map(|plan| plan.device_plan.maximum_executables()).unwrap_or(0),
            "programs": catalog.programs.len(),
            "programs_with_resident_segments": catalog.programs.values().filter(|program| program.has_resident_segments()).count(),
            "scope": "last_published_quiescent_catalog; physical segments may be shared by programs",
        })
    }

    pub(super) fn on_demand_reusable_execution_enabled(&self) -> bool {
        self.reusable_execution_startup_plan
            .as_ref()
            .is_some_and(|plan| {
                plan.device_plan.catalog_lifetime()
                    == ReusableExecutionCatalogLifetime::OnDemandBounded
            })
    }

    pub(super) fn install_reusable_execution_catalog(
        &self,
        catalog: VNextReusableExecutionCatalog,
    ) -> Result<()> {
        let mut installed = self.reusable_execution_catalog.write();
        if installed.is_some() {
            return Err(FerrumError::internal(
                "vNext reusable execution catalog was already installed",
            ));
        }
        *installed = Some(Arc::new(catalog));
        Ok(())
    }

    pub(super) fn prepare_on_demand_reusable_execution(
        &self,
        plan: &VNextReusableExecutionStartupPlan,
        started: Instant,
    ) -> Result<VNextReusableExecutionStartupReport> {
        let device_preparation = self
            .lane
            .configure_reusable_executables(plan.device_plan)
            .map_err(|error| {
                FerrumError::device(format!("vNext on-demand configuration failed: {error}"))
            })?;
        if device_preparation.state() != DeviceReusableExecutionPreparationState::Ready
            || device_preparation.resident_executables() != 0
            || device_preparation.captured_executables() != 0
            || device_preparation.maximum_executables()
                != plan.device_plan.maximum_executables() as u64
        {
            return Err(FerrumError::device(format!(
                "vNext on-demand configuration returned an invalid receipt: {device_preparation:?}"
            )));
        }
        let catalog = self
            .lane
            .reusable_execution_catalog()
            .map_err(|error| FerrumError::device(error.to_string()))?;
        let (lane_epoch, programs) = catalog.into_parts();
        if !programs.is_empty() {
            return Err(FerrumError::internal(
                "vNext on-demand catalog must start empty",
            ));
        }
        self.install_reusable_execution_catalog(VNextReusableExecutionCatalog {
            lane_epoch,
            programs: BTreeMap::new(),
        })?;
        Ok(VNextReusableExecutionStartupReport {
            enabled: true,
            supported: true,
            eager_fallback_required: true,
            resolved_runtime_policy_fingerprint: self.policy.fingerprint_str().to_owned(),
            resolved_program_policy: Some(plan.program_policy.clone()),
            decode_width_resolution: Some(plan.capture_resolution.clone()),
            maximum_device_executables: plan.device_plan.maximum_executables(),
            requested_descriptors: plan.descriptors.clone(),
            prepared_descriptors: Vec::new(),
            capture_case_receipts: Vec::new(),
            catalog_programs: Vec::new(),
            requested_decode_widths: plan.decode_widths(),
            prepared_decode_widths: Vec::new(),
            requested_prefill_token_counts: plan.prefill_token_counts(),
            prepared_prefill_token_counts: Vec::new(),
            requested_prefill_chunks: plan.prefill_chunks(),
            prepared_prefill_chunks: Vec::new(),
            synthetic_sequences: 0,
            eager_warmup_waves: 0,
            capture_waves: 0,
            replay_inventory_check_waves: 0,
            prepared_programs: 0,
            device_preparation,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }

    /// Called after a terminal completion. Clone snapshots on the read path and
    /// release the catalog guard before submitting work. Publication holds the
    /// write guard only while obtaining a quiescent device snapshot, never across
    /// a GPU wait. A different in-flight request defers publication.
    pub(super) fn refresh_on_demand_reusable_execution_catalog(&self) -> Result<()> {
        if !self.on_demand_reusable_execution_enabled()
            || !self
                .reusable_execution_catalog_refresh_needed
                .swap(false, Ordering::AcqRel)
        {
            return Ok(());
        }
        let Some(mut installed) = self.reusable_execution_catalog.try_write() else {
            self.reusable_execution_catalog_refresh_needed
                .store(true, Ordering::Release);
            return Ok(());
        };
        let Some(snapshot) = self
            .lane
            .try_reusable_execution_catalog()
            .map_err(|error| {
                FerrumError::device(format!("vNext on-demand catalog refresh failed: {error}"))
            })?
        else {
            self.reusable_execution_catalog_refresh_needed
                .store(true, Ordering::Release);
            return Ok(());
        };
        let (lane_epoch, programs) = snapshot.into_parts();
        let maximum_programs = self
            .reusable_execution_startup_plan
            .as_ref()
            .ok_or_else(|| {
                FerrumError::internal("vNext on-demand catalog has no preparation plan")
            })?
            .device_plan
            .maximum_executables();
        if programs.len() > maximum_programs {
            return Err(FerrumError::internal(
                "vNext on-demand program catalog exceeds its immutable capacity",
            ));
        }
        let mut catalog = BTreeMap::new();
        for program in programs {
            let id = program.program_id();
            if id.plan_hash() != self.resolved_plan.execution_plan().plan_hash()
                || id.runtime_implementation_fingerprint()
                    != self.runtime.descriptor().runtime_implementation_fingerprint
                || id.lane_id() != self.lane.id()
                || program.segments().iter().any(|segment| {
                    segment.end_node_index() as usize
                        > self.resolved_plan.execution_plan().payload().nodes().len()
                })
            {
                return Err(FerrumError::internal(
                    "vNext on-demand catalog differs from its immutable plan or lane",
                ));
            }
            if catalog.insert(id.clone(), program).is_some() {
                return Err(FerrumError::internal(
                    "vNext on-demand catalog contains a duplicate program identity",
                ));
            }
        }
        *installed = Some(Arc::new(VNextReusableExecutionCatalog {
            lane_epoch,
            programs: catalog,
        }));
        Ok(())
    }
}
