//! Cold model domain and its exact sources. Runtime sessions live only in the
//! worker's DenoisingStage; this shared owner contains no mutable installation,
//! request-id lookup, or second device capacity authority.
use crate::iterative_service::{BatchError, DeviceCapacityExceeded, DeviceCapacityReport};
use crate::video_generation::fast_h3::{
    assets::FastH3Assets,
    cuda::FastH3CudaComposition,
    denoising::{
        DenoisingStage, DenoisingStageError, DenoisingWorkspacePolicy, FeedForwardPartitionPolicy,
        PredictionHandoffPolicy, PredictionInvocation, PreparedDenoisingBatch, PreparedPrediction,
    },
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{component::ComponentInitializationConfig, TensorSerialActor};
use ferrum_kernels::{
    backend::cuda::vnext_runtime::CudaDeviceRuntime, native_ops::sage_vsa_fp4::CheckedVsaRegions,
};
use ferrum_models::fast_h3::vsa::{
    FastH3PrefixAttentionArithmetic, FastH3VsaLayout, FastH3VsaSelection,
};
use ferrum_models::fast_h3::*;
use ferrum_runtime::*;
use std::{collections::BTreeMap, sync::Arc};

mod batch;
use batch::{compile_batch, OptionalDenoisingBatch};

pub(super) struct PreparedDenoisingDomain {
    context: Arc<FastH3CudaComposition>,
    config: FastH3DenoisingConfig,
    limits: FastH3DenoisingLimits,
    regions: Arc<CheckedVsaRegions>,
    source: Arc<OwnedFastH3DenoisingSource>,
    single: PreparedDenoisingBatch<CudaDeviceRuntime>,
    optional: OptionalDenoisingBatch,
    predictions: Vec<PreparedPrediction<CudaDeviceRuntime>>,
    prediction_sources: Vec<Arc<OwnedFastH3FinalOutputSource>>,
    maximum_prediction_output_bytes: usize,
    handoff: PredictionHandoffPolicy,
}
impl PreparedDenoisingDomain {
    /// B2 here repeats the explicit checked tag route supplied by this domain.
    /// Other same-shape tag routes do not join by shape: BatchPlan::bind rejects
    /// them. Each continuation's evaluation remains independent at every step.
    pub(super) fn prepare(
        assets: Arc<FastH3Assets>,
        context: Arc<FastH3CudaComposition>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        tags: &[FastH3TextTagRun],
        limits: FastH3DenoisingLimits,
        final_limits: FastH3FinalOutputLimits,
        handoff: PredictionHandoffPolicy,
        feed_forward_chunk_rows: std::num::NonZeroU64,
        feed_forward_partition: FeedForwardPartitionPolicy,
    ) -> Result<Self> {
        let expected = FastH3ConditioningSchedule::all_modalities(&assets.pdd)?;
        ensure!(
            schedule.evaluation_count() == expected.evaluation_count()
                && (0..expected.evaluation_count()).all(|evaluation| {
                    use FastH3SegmentKind::*;
                    [
                        Text,
                        ReferenceImage,
                        ReferenceVideo,
                        ReferenceAudio,
                        TargetVideo,
                        TargetAudio,
                    ]
                    .into_iter()
                    .all(|kind| {
                        schedule.segment_time_index(evaluation, kind)
                            == expected.segment_time_index(evaluation, kind)
                    })
                })
                && schedule
                    .timesteps()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.timesteps().iter().map(|v| v.to_bits())),
            "denoising schedule differs from the original PDD asset"
        );
        let (config, source, regions) = OwnedFastH3DenoisingSource::prepare_batch(
            assets.transformer.clone(),
            layout,
            schedule,
            &[tags],
            FastH3VsaSelection::Trained,
            FastH3PrefixAttentionArithmetic::VendorFused,
            &context.host,
            limits,
        )?;
        let single = compile_batch(
            &context,
            config.clone(),
            regions.clone(),
            limits,
            feed_forward_chunk_rows,
            feed_forward_partition,
            None,
        )?;
        let optional = OptionalDenoisingBatch::new(
            assets.transformer.clone(),
            layout,
            schedule,
            tags,
            feed_forward_chunk_rows,
            feed_forward_partition,
        );
        let registry = context.registry(None)?;
        let catalog = context.catalog(&registry)?;
        let mut predictions = Vec::with_capacity(assets.pdd.steps().len());
        let mut prediction_sources = Vec::with_capacity(assets.pdd.steps().len());
        let mut maximum_prediction_output_bytes = 0usize;
        for evaluation in 0..assets.pdd.steps().len() {
            let (c, source) = OwnedFastH3FinalOutputSource::prepare(
                assets.transformer.clone(),
                &assets.pdd,
                layout.packed(),
                evaluation,
                &context.host,
                final_limits,
            )?;
            let options = c.compile_options(final_limits)?;
            let mut bytes = 0usize;
            for id in [
                FastH3FinalOutputFamily::VIDEO_OUTPUT_ID,
                FastH3FinalOutputFamily::AUDIO_OUTPUT_ID,
            ] {
                let s = options
                    .tensor_specs()
                    .get(&ProgramValueId::new(id)?)
                    .context("missing prediction output")?;
                let n = ResolvedTensorSpec::new(
                    s.dimensions.clone(),
                    s.element_type,
                    s.layout.clone(),
                )?
                .minimum_storage_bytes()?;
                bytes = bytes
                    .checked_add(context.host.backing_bytes(usize::try_from(n)?)?)
                    .context("prediction output overflow")?;
            }
            maximum_prediction_output_bytes = maximum_prediction_output_bytes.max(bytes);
            predictions.push(PreparedPrediction::compile(
                c,
                final_limits,
                &registry,
                &catalog,
                &context.policy,
                &context.materializers,
            )?);
            prediction_sources.push(Arc::new(source));
        }
        Ok(Self {
            context,
            config,
            limits,
            regions,
            source: Arc::new(source),
            single,
            optional,
            predictions,
            prediction_sources,
            maximum_prediction_output_bytes,
            handoff,
        })
    }
    fn batch(
        &self,
        count: usize,
    ) -> Result<&PreparedDenoisingBatch<CudaDeviceRuntime>, VNextError> {
        match count {
            1 => Ok(&self.single),
            2 => self.optional.ready().ok_or_else(|| {
                invalid("optional batch must be prepared before membership admission")
            }),
            _ => Err(invalid("batch outside B1/B2 domain")),
        }
    }
    pub(super) fn check_batch_capacity(
        &self,
        count: usize,
        weights: Option<&SharedComponentWeights<CudaDeviceRuntime>>,
    ) -> Result<bool, BatchError> {
        let prepared = if count == 2 {
            self.optional
                .prepare(&self.context, self.limits, &self.single, weights)
        } else {
            self.batch(count).map_err(Into::into)
        };
        match prepared {
            Ok(_) => Ok(true),
            Err(error) if error.is::<HostMemoryError>() => {
                let pressure = error.downcast::<HostMemoryError>().unwrap();
                if pressure.is_temporary_capacity() {
                    Err(BatchError::AwaitHostCapacity {
                        account: self.context.host.clone(),
                        pressure,
                    })
                } else if count == 2 && matches!(pressure, HostMemoryError::Capacity { .. }) {
                    // This optional cardinality exceeds a hard Host ceiling.
                    // Keep the already admitted independent B1 continuations.
                    Ok(false)
                } else {
                    Err(super::rejected_error(pressure))
                }
            }
            Err(error) => match error.downcast_ref::<VNextError>() {
                Some(VNextError::ModelProgramNodeCapacityExceeded { .. }) if count == 2 => {
                    Ok(false)
                }
                Some(VNextError::PlanCapacityExceeded {
                    required_bytes,
                    usable_capacity_bytes,
                }) if count == 2 => {
                    let proof = DeviceCapacityExceeded {
                        report: DeviceCapacityReport {
                            complete_batch: false,
                            required_bytes: *required_bytes,
                            capacity_bytes: *usable_capacity_bytes,
                            retained_static_bytes: 0,
                            active_stage_peak_bytes: *required_bytes,
                            retained_dynamic_allowance_bytes: 0,
                            native_allowance_bytes: 0,
                        },
                    };
                    Err(BatchError::Capacity {
                        proof,
                        diagnostic: proof.to_string(),
                    })
                }
                _ => Err(super::rejected_error(error)),
            },
        }
    }
    pub(super) fn extend_stage(
        &self,
        stage: &mut DenoisingStage<CudaDeviceRuntime>,
        count: usize,
        base: &ComponentInitializationConfig,
    ) -> Result<(), DenoisingStageError<CudaDeviceRuntime>> {
        if stage.prepared_batch_counts().any(|n| n == count) {
            return Ok(());
        }
        let batch = self.batch(count)?.clone();
        stage.attach_batch(
            batch,
            derived_initialization(base, &format!("batch{count}"), None)?,
        )
    }
    pub(super) fn configuration(&self) -> &FastH3DenoisingConfig {
        &self.config
    }
    pub(super) fn limits(&self) -> FastH3DenoisingLimits {
        self.limits
    }
    pub(super) fn regions(&self) -> &Arc<CheckedVsaRegions> {
        &self.regions
    }
    pub(super) fn request_plan(&self) -> &FastH3DenoisingBatchPlan {
        self.single.plan()
    }
    pub(super) fn new_stage(&self) -> Result<DenoisingStage<CudaDeviceRuntime>, VNextError> {
        let stage = DenoisingStage::new(
            self.context.runtime.clone(),
            self.context.host.clone(),
            std::iter::once(self.single.clone())
                .chain(self.optional.ready().cloned())
                .collect(),
            self.predictions.clone(),
            std::iter::once(self.source.clone() as Arc<dyn WeightComponentSource>)
                .chain(
                    self.prediction_sources
                        .iter()
                        .map(|s| s.clone() as Arc<dyn WeightComponentSource>),
                )
                .collect(),
            DenoisingWorkspacePolicy::ActivePhaseOnly,
        )?;
        Ok(match self.handoff {
            PredictionHandoffPolicy::Host => stage,
            PredictionHandoffPolicy::DeviceIfCapacityAllows => stage
                .with_device_prediction_handoff(
                    self.context.capacity.clone(),
                    self.context.device_bytes,
                ),
        })
    }
    pub(super) fn restore(
        &self,
        stage: &mut DenoisingStage<CudaDeviceRuntime>,
        base: &ComponentInitializationConfig,
    ) -> Result<(), DenoisingStageError<CudaDeviceRuntime>> {
        let initialization = stage
            .prepared_batch_counts()
            .map(|count| {
                Ok((
                    count,
                    derived_initialization(base, &format!("batch{count}"), None)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
        stage.restore(&initialization)
    }
    /// `base` supplies a fresh monotonic execution identity from the sole worker.
    /// Actual member ids and current evaluations are read from the borrowed
    /// continuations; a caller cannot substitute a stale slot/evaluation table.
    pub(super) fn advance(
        &self,
        stage: &mut DenoisingStage<CudaDeviceRuntime>,
        actor: &mut TensorSerialActor<CudaDeviceRuntime>,
        batch_host: &HostMemoryAccount,
        base: ComponentInitializationConfig,
        requests: Vec<&mut FastH3DenoisingRequest>,
    ) -> Result<Vec<bool>, DenoisingStageError<CudaDeviceRuntime>> {
        let mut invocations = Vec::with_capacity(requests.len());
        for (slot, request) in requests.iter().enumerate() {
            let evaluation = request.evaluation();
            self.prediction_sources
                .get(evaluation)
                .ok_or_else(|| invalid("request evaluation has no original PDD head"))?;
            let suffix = format!("slot{slot}.evaluation{evaluation}");
            let initialization = derived_initialization(&base, &suffix, Some(request.request()))?;
            invocations.push(PredictionInvocation {
                run_id: RunId::new(format!("{}.prediction", initialization.run_id.as_str()))?,
                request_id: request.request().clone(),
                initialization,
            });
        }
        stage.advance(
            actor,
            batch_host,
            base.run_id,
            base.request_id,
            requests,
            invocations,
        )
    }

    /// Exact input backing upper bound from the declared external tensor ports,
    /// including physical Host alignment. B1 aliases may spend less. No weight
    /// or intermediate graph tensor is counted as a joined request input.
    pub(super) fn joined_input_bytes(&self, batches: usize) -> Result<usize> {
        let prepared = self.batch(batches)?;
        let family = prepared.compiled().family();
        family
            .program()
            .inputs()
            .iter()
            .try_fold(0usize, |sum, input| {
                let spec = prepared
                    .plan()
                    .compile_options()
                    .tensor_specs()
                    .get(input)
                    .context("missing external input specification")?;
                let bytes = ResolvedTensorSpec::new(
                    spec.dimensions.clone(),
                    spec.element_type,
                    spec.layout.clone(),
                )?
                .minimum_storage_bytes()?;
                let physical = self.context.host.backing_bytes(usize::try_from(bytes)?)?;
                sum.checked_add(physical)
                    .context("batch Host input bound overflow")
            })
    }
    pub(super) fn output_bytes(&self, batches: usize) -> Result<usize> {
        let prepared = self.batch(batches)?;
        let hidden = self.context.host.backing_bytes(usize::try_from(
            prepared
                .plan()
                .configuration()
                .backbone()
                .packed_rows()
                .checked_mul(batches as u64)
                .and_then(|n| n.checked_mul(self.config.media().hidden_size()))
                .and_then(|n| n.checked_mul(2))
                .context("hidden output extent overflow")?,
        )?)?;
        hidden
            .checked_add(
                self.maximum_prediction_output_bytes
                    .checked_mul(batches)
                    .context("prediction batch overflow")?,
            )
            .context("output admission overflow")
    }
}
fn derived_initialization(
    base: &ComponentInitializationConfig,
    suffix: &str,
    request: Option<&RequestIdentity>,
) -> Result<ComponentInitializationConfig, VNextError> {
    Ok(ComponentInitializationConfig {
        request_id: request.cloned().unwrap_or_else(|| base.request_id.clone()),
        run_id: RunId::new(format!("{}.{}", base.run_id.as_str(), suffix))?,
        transaction_id: TransactionId::new(format!("{}.{}", base.transaction_id.as_str(), suffix))?,
        static_policy: base.static_policy.clone(),
        executor: base.executor.clone(),
    })
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
