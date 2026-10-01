//! A prepared numeric directory owned by one real CUDA stream and execution lane.
//! Native capture/upload remain authoritative. Publication is optional metadata:
//! failures retract the directory, never change a native submission outcome.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCostGraphCatalogSource, DeviceObservationTemplateBudget, DevicePreparedCostGraphCatalog,
    DevicePreparedCostGraphCatalogAvailability, DevicePreparedCostGraphCatalogBuildError,
    ExecutionLaneId, VNextError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildStage {
    Begin,
    Program,
    Segment,
    Finish,
}

#[derive(Debug)]
enum PreparationFailure {
    Source(VNextError),
    Owner,
    State,
    Mutation,
    BuildRejected {
        stage: BuildStage,
        error: DevicePreparedCostGraphCatalogBuildError,
    },
}

#[derive(Clone, Copy, Debug, Default)]
enum PreparationRetry {
    #[default]
    NextBoundary,
    NextMutation,
    Capacity {
        required_exclusive_peak_bytes: usize,
    },
}

#[derive(Default)]
pub(super) struct PreparedCatalog {
    source: Option<DeviceCostGraphCatalogSource>,
    root: Option<Arc<DevicePreparedCostGraphCatalog>>,
    failure: Option<PreparationFailure>,
    retry: PreparationRetry,
    failure_logged: bool,
}

impl PreparedCatalog {
    pub(super) fn is_bound(&self) -> bool {
        self.source.is_some()
    }

    pub(super) fn invalidate(&mut self) {
        self.root = None;
        if let Some(source) = &mut self.source {
            source.invalidate();
        }
        self.failure = None;
        self.retry = PreparationRetry::NextBoundary;
        self.failure_logged = false;
    }
}

impl CudaExecutableCache {
    pub(crate) fn bind_cost_catalog_lane(
        &mut self,
        runtime_instance: u64,
        stream_instance: u64,
        runtime_fingerprint: &str,
        lane: ExecutionLaneId,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if let Some(source) = &self.prepared_catalog.source {
            if source.matches_owner(runtime_instance, stream_instance, runtime_fingerprint)
                && source.lane_id() == lane
            {
                return Ok(());
            }
            self.prepared_catalog.invalidate();
            self.prepared_catalog.failure = Some(PreparationFailure::Owner);
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA cost catalog cannot be rebound to another runtime, stream or lane",
            ));
        }
        match DeviceCostGraphCatalogSource::new(
            runtime_instance,
            stream_instance,
            runtime_fingerprint,
            lane,
        ) {
            Ok(source) => {
                self.prepared_catalog.source = Some(source);
                self.prepared_catalog.failure = None;
            }
            Err(error) => {
                // Optional cost metadata must not make a healthy lane unusable.
                tracing::debug!(
                    target: "ferrum::cost_catalog_diagnostics",
                    error = %error,
                    "CUDA prepared cost catalog source binding rejected"
                );
                self.prepared_catalog.failure = Some(PreparationFailure::Source(error));
            }
        }
        Ok(())
    }

    pub(crate) fn invalidate_prepared_cost_catalog(&mut self) {
        self.prepared_catalog.invalidate();
        self.prepared_catalog.failure = Some(PreparationFailure::Mutation);
    }

    pub(crate) fn cost_prepared_reusable_graph_catalog(
        &self,
        runtime_instance: u64,
        stream_instance: u64,
        runtime_fingerprint: &str,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<DevicePreparedCostGraphCatalogAvailability, CudaDeviceRuntimeError> {
        poll().map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        let ready = self
            .prepared_catalog
            .source
            .as_ref()
            .filter(|source| {
                source.matches_owner(runtime_instance, stream_instance, runtime_fingerprint)
            })
            .zip(self.cost_graph_stream_state())
            .and_then(|(source, state)| {
                self.prepared_catalog
                    .root
                    .as_ref()
                    .filter(|root| root.is_current(source, state))
            });
        let result = match ready {
            Some(root) => DevicePreparedCostGraphCatalogAvailability::Ready(Arc::clone(root)),
            None => DevicePreparedCostGraphCatalogAvailability::Unprepared,
        };
        poll().map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        Ok(result)
    }

    /// Called only after native preparation/registration has converged, or an
    /// explicit configure/seal/trim boundary. Query never enters this method.
    pub(crate) fn publish_prepared_cost_catalog(
        &mut self,
        runtime_instance: u64,
        stream_instance: u64,
        runtime_fingerprint: &str,
        budget: &Arc<DeviceObservationTemplateBudget>,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) {
        let Some(source) = self.prepared_catalog.source.as_ref() else {
            return;
        };
        if !source.matches_owner(runtime_instance, stream_instance, runtime_fingerprint) {
            self.prepared_catalog.invalidate();
            self.prepared_catalog.failure = Some(PreparationFailure::Owner);
            return;
        }
        let Some(state) = self.cost_graph_stream_state() else {
            self.prepared_catalog.invalidate();
            self.prepared_catalog.failure = Some(PreparationFailure::State);
            return;
        };
        if self
            .prepared_catalog
            .root
            .as_ref()
            .is_some_and(|root| root.is_current(source, state))
        {
            return;
        }
        match self.prepared_catalog.retry {
            PreparationRetry::NextMutation => return,
            PreparationRetry::Capacity {
                required_exclusive_peak_bytes,
            } if budget
                .maximum_bytes()
                .saturating_sub(budget.retained_payload_bytes())
                < required_exclusive_peak_bytes =>
            {
                return;
            }
            _ => {}
        }
        let prior_required = match self.prepared_catalog.retry {
            PreparationRetry::Capacity {
                required_exclusive_peak_bytes,
            } => required_exclusive_peak_bytes,
            _ => 0,
        };
        // Never keep the old revision alive as the cache's published root
        // while constructing its replacement. External numeric readers retain
        // their own old root and its full lease until they release it.
        self.prepared_catalog.root = None;
        let mut cancelled = false;
        let result = (|| {
            let mut checked_poll = || {
                let result = poll();
                cancelled |= result.is_err();
                result
            };
            let poll: &mut dyn FnMut() -> Result<(), VNextError> = &mut checked_poll;
            let mut builder = source.begin(state, budget, poll).map_err(|error| {
                PreparationFailure::BuildRejected {
                    stage: BuildStage::Begin,
                    error,
                }
            })?;
            for program in self.programs.values() {
                builder
                    .push_program(&program.descriptor, poll)
                    .map_err(|error| PreparationFailure::BuildRejected {
                        stage: BuildStage::Program,
                        error,
                    })?;
                for segment in &program.segments {
                    let Some(executable) = self.entries.get(&segment.key) else {
                        continue;
                    };
                    if !executable.uploaded {
                        continue;
                    }
                    let Some(logical) = segment.logical_commands.as_deref() else {
                        continue;
                    };
                    builder
                        .push_uploaded_segment(
                            &segment.descriptor,
                            segment.reusable_executable_fingerprint.as_ref(),
                            logical,
                            poll,
                        )
                        .map_err(|error| PreparationFailure::BuildRejected {
                            stage: BuildStage::Segment,
                            error,
                        })?;
                }
            }
            builder
                .finish(poll)
                .map_err(|error| PreparationFailure::BuildRejected {
                    stage: BuildStage::Finish,
                    error,
                })
        })();
        match result {
            Ok(root) if root.is_current(source, state) => {
                self.prepared_catalog.root = Some(root);
                self.prepared_catalog.failure = None;
                self.prepared_catalog.retry = PreparationRetry::NextBoundary;
            }
            Ok(_) => {
                self.prepared_catalog.failure = Some(PreparationFailure::State);
                self.prepared_catalog.retry = PreparationRetry::NextMutation;
            }
            Err(failure) => {
                // Opt-in diagnostics only; no full-registry rescan/format on
                // repeated queries, and no string classification of backend errors.
                if !cancelled
                    && !self.prepared_catalog.failure_logged
                    && tracing::enabled!(
                        target: "ferrum::cost_catalog_diagnostics",
                        tracing::Level::DEBUG
                    )
                {
                    tracing::debug!(
                        target: "ferrum::cost_catalog_diagnostics",
                        failure = ?failure,
                        programs = self.programs.len(),
                        resident_executables = self.entries.len(),
                        rejected_executables = self.rejected.len(),
                        retained_metadata_bytes = budget.retained_payload_bytes(),
                        maximum_metadata_bytes = budget.maximum_bytes(),
                        "CUDA prepared cost catalog publication rejected"
                    );
                    self.prepared_catalog.failure_logged = true;
                }
                self.prepared_catalog.retry = match &failure {
                    PreparationFailure::BuildRejected {
                        error:
                            DevicePreparedCostGraphCatalogBuildError::Capacity {
                                required_exclusive_peak_bytes,
                            },
                        ..
                    } => PreparationRetry::Capacity {
                        required_exclusive_peak_bytes: prior_required
                            .max(*required_exclusive_peak_bytes),
                    },
                    _ if cancelled && prior_required != 0 => PreparationRetry::Capacity {
                        required_exclusive_peak_bytes: prior_required,
                    },
                    _ if cancelled => PreparationRetry::NextBoundary,
                    _ => PreparationRetry::NextMutation,
                };
                self.prepared_catalog.failure = Some(failure);
            }
        }
    }
}

#[cfg(test)]
mod tests;
