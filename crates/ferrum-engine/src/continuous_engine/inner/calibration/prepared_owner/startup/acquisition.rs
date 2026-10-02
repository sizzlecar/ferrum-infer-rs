//! Resolve source preparation before numerical membership starts. Failed
//! setup keeps its original charges; only a rebuilt cold source may continue.
use super::*;
use crate::continuous_engine::inner::calibration::{
    cohort_driver::ProbeExecutionBudget,
    prepared_owner::plan::{ColdSourceRebuild, PreparedProbeSeries, PreparedProbeSource},
    startup::{AcquiredProbePrefix, ProbePrefixAcquisition, ProbePrefixFallback},
};

pub(in crate::continuous_engine::inner::calibration) enum StartupSourcePreparation<'a> {
    Ready {
        source: PreparedProbeSource<'a>,
        acquired: Vec<AcquiredProbePrefix>,
    },
    Skipped,
    Interrupted {
        error: FerrumError,
        acquired: Vec<AcquiredProbePrefix>,
    },
}

fn lease_vec_bytes(capacity: usize) -> Result<usize> {
    capacity
        .checked_mul(std::mem::size_of::<AcquiredProbePrefix>())
        .and_then(|n| n.checked_add(std::mem::size_of::<Vec<AcquiredProbePrefix>>()))
        .ok_or_else(|| FerrumError::resource_exhausted("native source lease vector overflow"))
}

impl CalibrationSession {
    pub(in crate::continuous_engine::inner::calibration) async fn prepare_startup_source<'a>(
        &mut self,
        series: &'a PreparedProbeSeries,
        source_index: usize,
        budget: &mut ProbeExecutionBudget,
        extra_offer_rows: &mut usize,
    ) -> Result<StartupSourcePreparation<'a>> {
        let mut source = series.source(source_index)?;
        let keys = source.acquisitions().len();
        if keys == 0 {
            source.ensure_acquisitions_bound()?;
            return Ok(StartupSourcePreparation::Ready {
                source,
                acquired: Vec::new(),
            });
        }
        let mut acquired = Vec::new();
        let mut fallback = source
            .acquisition_host_allowance(lease_vec_bytes(keys)?)?
            .is_none()
            .then_some(ProbePrefixFallback::HostCapacity);
        if fallback.is_none() && acquired.try_reserve_exact(keys).is_err() {
            fallback = Some(ProbePrefixFallback::HostCapacity);
        }
        let vec_bytes = lease_vec_bytes(acquired.capacity())?;
        if fallback.is_none() && source.acquisition_host_allowance(vec_bytes)?.is_none() {
            fallback = Some(ProbePrefixFallback::HostCapacity);
        }
        if fallback.is_none() {
            self.begin_startup_inventory()?;
            let built = async {
                for key in 0..keys {
                    budget.require_selection_time()?;
                    let declared = source.acquisitions()[key].plan();
                    let Some(allowance) = source.acquisition_host_allowance(vec_bytes)? else {
                        return Ok(Some(ProbePrefixFallback::HostCapacity));
                    };
                    let request = source.acquisition_request_for(key)?;
                    let ready = match self
                        .acquire_reserved_probe_prefix(request, declared, budget, allowance)
                        .await?
                    {
                        ProbePrefixAcquisition::Ready(ready) => ready,
                        ProbePrefixAcquisition::ColdFallback(reason) => return Ok(Some(reason)),
                    };
                    if !source.bind_verified_scope(key, &ready, vec_bytes)? {
                        return Ok(Some(ProbePrefixFallback::HostCapacity));
                    }
                    acquired.push(ready);
                    tracing::info!(
                        source_index,
                        acquisition_key = key,
                        boundary_tokens = declared.boundary(),
                        "Automatic source private prefix acknowledged before numerical collection"
                    );
                }
                source.ensure_acquisitions_bound()?;
                Ok::<_, FerrumError>(None)
            }
            .await;
            // Execute both cleanup operations even on setup failure. A failed
            // retirement never authorizes cold execution or the next source.
            let drained = self.drain_startup_geometry().await;
            let closed = self.end_startup_inventory().await;
            if drained.is_err() || closed.is_err() {
                return Ok(StartupSourcePreparation::Interrupted {
                    error: FerrumError::internal(format!(
                        "native source preparation cleanup failed: setup={:?}; drain={:?}; inventory={:?}",
                        built.as_ref().err(), drained.as_ref().err(), closed.as_ref().err()
                    )),
                    acquired,
                });
            }
            fallback = built?;
        }
        if fallback.is_none() && acquired.iter().any(|prefix| !prefix.ready()) {
            fallback = Some(ProbePrefixFallback::LeaseUnavailable);
        }
        let Some(reason) = fallback else {
            return Ok(StartupSourcePreparation::Ready { source, acquired });
        };

        // Seed owners and submitted transfers have retired. Drop all extra
        // pins before clearing native scope or allocating the cold rebuild.
        drop(acquired);
        match source.try_rebuild_cold()? {
            ColdSourceRebuild::Ready {
                additional_collection_actions,
                additional_offer_rows,
            } => {
                let Some(remaining_rows) = extra_offer_rows.checked_sub(additional_offer_rows)
                else {
                    tracing::warn!(source_index, ?reason, additional_offer_rows,
                        remaining_offer_rows = *extra_offer_rows,
                        "Automatic source skipped before collection: cold sample capacity unavailable");
                    return Ok(StartupSourcePreparation::Skipped);
                };
                if !budget.try_reserve_additional_source_actions(additional_collection_actions)? {
                    tracing::warn!(source_index, ?reason, additional_collection_actions,
                        remaining_actions = budget.selection_attempts_remaining(),
                        "Automatic source skipped before collection: cold action reservation unavailable");
                    return Ok(StartupSourcePreparation::Skipped);
                }
                *extra_offer_rows = remaining_rows;
                tracing::info!(
                    source_index,
                    ?reason,
                    additional_collection_actions,
                    additional_offer_rows,
                    "Automatic source rebuilt cold before numerical collection"
                );
                Ok(StartupSourcePreparation::Ready {
                    source,
                    acquired: Vec::new(),
                })
            }
            ColdSourceRebuild::Skip(skip) => {
                tracing::warn!(source_index, ?reason, ?skip,
                    "Automatic source skipped before collection: original members cannot support cold preparation");
                Ok(StartupSourcePreparation::Skipped)
            }
        }
    }
}
