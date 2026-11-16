//! One optional cardinality derived from this exact immutable model domain.
//! Only compiled metadata and terminal planning limits are retained here;
//! sessions, installation and capacity remain owned by the worker stage.
use super::*;
use std::{num::NonZeroU64, sync::OnceLock};

pub(super) struct OptionalDenoisingBatch {
    package: Arc<FastH3TransformerPackage>,
    layout: FastH3VsaLayout,
    schedule: FastH3ConditioningSchedule,
    tags: Vec<FastH3TextTagRun>,
    chunk_rows: NonZeroU64,
    partition: FeedForwardPartitionPolicy,
    prepared: OnceLock<Result<PreparedDenoisingBatch<CudaDeviceRuntime>, VNextError>>,
}

impl OptionalDenoisingBatch {
    pub(super) fn new(
        package: Arc<FastH3TransformerPackage>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        tags: &[FastH3TextTagRun],
        chunk_rows: NonZeroU64,
        partition: FeedForwardPartitionPolicy,
    ) -> Self {
        Self {
            package,
            layout: layout.clone(),
            schedule: schedule.clone(),
            tags: tags.to_vec(),
            chunk_rows,
            partition,
            prepared: OnceLock::new(),
        }
    }

    pub(super) fn ready(&self) -> Option<&PreparedDenoisingBatch<CudaDeviceRuntime>> {
        self.prepared.get().and_then(|p| p.as_ref().ok())
    }

    pub(super) fn prepare(
        &self,
        context: &FastH3CudaComposition,
        limits: FastH3DenoisingLimits,
        single: &PreparedDenoisingBatch<CudaDeviceRuntime>,
        weights: Option<&SharedComponentWeights<CudaDeviceRuntime>>,
    ) -> Result<&PreparedDenoisingBatch<CudaDeviceRuntime>> {
        if self.prepared.get().is_none() {
            let result = (|| -> Result<_> {
                let (config, source, regions) = OwnedFastH3DenoisingSource::prepare_batch(
                    self.package.clone(),
                    &self.layout,
                    &self.schedule,
                    &[&self.tags, &self.tags],
                    FastH3VsaSelection::Trained,
                    FastH3PrefixAttentionArithmetic::VendorFused,
                    &context.host,
                    limits,
                )?;
                let batch = compile_batch(
                    context,
                    config,
                    regions,
                    limits,
                    self.chunk_rows,
                    self.partition,
                    weights,
                )?;
                if batch.compiled().family().weights() != single.compiled().family().weights() {
                    return Err(invalid("B1/B2 physical source interpretations differ").into());
                }
                // Inventory validation still precedes publication. Execution
                // attaches the original B1 installation, never this source.
                drop(source);
                Ok(batch)
            })();
            // A transient backend or device-account refusal must remain
            // retryable. Only these fixed compiler ceilings belong to the
            // immutable domain. Repeated refused joins need not recompile it.
            match result {
                Ok(batch) => {
                    let _ = self.prepared.set(Ok(batch));
                }
                Err(error) => match error.downcast_ref::<VNextError>() {
                    Some(
                        limit @ (VNextError::ModelProgramNodeCapacityExceeded { .. }
                        | VNextError::PlanCapacityExceeded { .. }),
                    ) => {
                        let _ = self.prepared.set(Err(limit.clone()));
                    }
                    _ => return Err(error),
                },
            }
        }
        self.prepared
            .get()
            .expect("published optional plan")
            .as_ref()
            .map_err(|error| anyhow::Error::new(error.clone()))
    }
}

pub(super) fn compile_batch(
    context: &FastH3CudaComposition,
    config: FastH3DenoisingConfig,
    regions: Arc<CheckedVsaRegions>,
    limits: FastH3DenoisingLimits,
    chunk_rows: NonZeroU64,
    partition: FeedForwardPartitionPolicy,
    weights: Option<&SharedComponentWeights<CudaDeviceRuntime>>,
) -> Result<PreparedDenoisingBatch<CudaDeviceRuntime>> {
    let count = config.backbone().batches();
    let registry = context.registry(Some(regions.clone()))?;
    let catalog = context.catalog(&registry)?;
    let compile = |config| {
        PreparedDenoisingBatch::compile(
            config,
            limits,
            regions.clone(),
            &registry,
            &catalog,
            &context.policy,
            &context.materializers,
        )
    };
    let additional_peak = |batch: &PreparedDenoisingBatch<CudaDeviceRuntime>| {
        batch
            .compiled()
            .minimum_additional_execution_bytes(&context.runtime, weights)?
            .checked_add(context.initialization_stream_bytes)
            .ok_or_else(|| invalid("FFN additional plan peak overflow"))
    };
    let fits_current_capacity = |batch: &PreparedDenoisingBatch<CudaDeviceRuntime>| {
        let peak = additional_peak(batch)?;
        match context.capacity.check_available_bytes(peak) {
            Ok(()) => Ok(true),
            Err(VNextError::DeviceCapacityUnavailable(pressure)) => {
                tracing::debug!(
                    batch = count,
                    additional_peak_bytes = peak,
                    ?pressure,
                    "FastH3 compiled denoising candidate exceeds live capacity"
                );
                Ok(false)
            }
            Err(error) => Err(error),
        }
    };
    let bounded = || {
        let mut best = None;
        let mut capacity_failure = None;
        let mut next_rows = Some(chunk_rows);
        'partitions: while let Some(rows) = next_rows {
            tracing::info!(
                batch = count,
                maximum_chunk_rows = rows.get(),
                "FastH3 selects bounded FFN rows"
            );
            let bounded = config.clone().with_ffn_chunk_rows(Some(rows));
            for schedule in [
                ferrum_models::TensorProgramSchedule::DeclarationOrder,
                ferrum_models::TensorProgramSchedule::DelayedExpansion,
                ferrum_models::TensorProgramSchedule::EarlyCompaction,
                ferrum_models::TensorProgramSchedule::ConsumerChains,
            ] {
                match compile(bounded.clone().with_schedule(schedule)) {
                    Ok(candidate) => {
                        let bytes = additional_peak(&candidate)?;
                        if fits_current_capacity(&candidate)? {
                            tracing::info!(
                                batch = count,
                                ?schedule,
                                maximum_chunk_rows = rows.get(),
                                additional_peak_bytes = bytes,
                                "FastH3 bounded denoising schedule fits current capacity"
                            );
                            return Ok(candidate);
                        }
                        if best.as_ref().is_none_or(|(previous, _)| bytes < *previous) {
                            best = Some((bytes, candidate));
                        }
                    }
                    Err(error @ VNextError::PlanCapacityExceeded { .. }) => {
                        capacity_failure = Some(error);
                    }
                    Err(error @ VNextError::ModelProgramNodeCapacityExceeded { .. }) => {
                        // Smaller row partitions add nodes monotonically. Keep
                        // the best complete plan without exceeding metadata
                        // limits or repeating an impossible smaller graph.
                        capacity_failure.get_or_insert(error);
                        break 'partitions;
                    }
                    Err(error) => return Err(error),
                }
            }
            // The configured value is an upper bound, not the only allowed
            // partition. Try smaller real graphs against the original account;
            // arithmetic, input coverage, quality and all budgets stay fixed.
            next_rows = NonZeroU64::new(rows.get() / 2);
        }
        // A heuristic never replaces a smaller valid plan. Runtime admission
        // remains authoritative if every compiled candidate faces live pressure.
        match best {
            Some((_, candidate)) => Ok(candidate),
            None => Err(capacity_failure.expect("every candidate exceeded plan capacity")),
        }
    };
    if partition == FeedForwardPartitionPolicy::Always {
        return bounded().map_err(Into::into);
    }
    match compile(config.clone()) {
        Ok(batch) => match fits_current_capacity(&batch)? {
            true => {
                tracing::info!(
                    batch = count,
                    additional_peak_bytes = additional_peak(&batch)?,
                    shared_weights = weights.is_some(),
                    "FastH3 whole-row FFN fits current capacity"
                );
                Ok(batch)
            }
            false => bounded(),
        },
        Err(VNextError::PlanCapacityExceeded { .. }) => bounded(),
        Err(error) => Err(error),
    }
    .map_err(Into::into)
}
