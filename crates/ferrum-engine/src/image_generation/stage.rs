use super::*;
use std::{fmt, time::Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStage {
    TextEncoder,
    VisionEncoder,
    VaeEncoder,
    Noise,
    Denoiser,
    Decoder,
}
impl ImageStage {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text",
            Self::VisionEncoder => "vision",
            Self::VaeEncoder => "encode",
            Self::Noise => "noise",
            Self::Denoiser => "denoise",
            Self::Decoder => "decode",
        }
    }
}

/// A bounded semantic plan slot within a component stage. This distinguishes
/// cooperating plans without deriving topology from a model or profile name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStagePlanRole {
    #[default]
    WholeStage,
    PrefixPrefill,
    PrefixDecode,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStagePlanReceipt {
    pub stage: ImageStage,
    pub role: ImageStagePlanRole,
    /// Actual fixed batch geometry compiled for each execution of this stage.
    pub batch_size: usize,
    pub family: ModelFamilyId,
    pub profile: NumericalProfileId,
    pub family_fingerprint: String,
    pub plan_hash: String,
    /// Compiled storage requirements, not an observation of peak device usage.
    pub static_bytes: u64,
    pub minimum_step_bytes: u64,
    /// Host wall time for family identity and plan compilation; not GPU time.
    pub compile_wall_duration_us: u64,
    /// Reused immutable compilation. Session reuse is reported separately.
    pub plan_cache_hit: bool,
    /// Compilation metadata retention, separate from device/session residency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_cache: Option<ImagePlanCacheReceipt>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStageExecutionReceipt {
    pub plan: ImageStagePlanReceipt,
    pub initialization: ComponentInitializationReceipt,
    /// The initialization receipt belongs to the retained session's original
    /// initialization. When true, no initialization occurred for this batch.
    pub session_reused: bool,
    /// This acquisition read an existing immutable installation, either through
    /// the same session or a newly bound compatible plan. No weight upload ran.
    pub weights_reused: bool,
    pub completed_executions: usize,
    /// Host wall time around component initialization, including no-static setup.
    pub initialization_wall_duration_us: u64,
    /// Host wall time per execution attempt, including admission, upload,
    /// submission, completion wait and readback. A failed final attempt is
    /// retained here; this is not a CUDA event or kernel-only measurement.
    pub execution_wall_durations_us: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStageReceipt {
    pub execution: ImageStageExecutionReceipt,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub close: Option<PlanRuntimeCloseReceipt>,
    pub retirement: ImageStageRetirement,
    pub close_wall_duration_us: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageStageRetirement {
    Closed,
    /// The last physical snapshot owner retired after all resident sessions.
    SharedWeightsClosed {
        receipt: SharedComponentWeightsCloseReceipt,
    },
    Resident {
        idle_release: DynamicPoolRebalanceReceipt,
    },
    ResidentBackingRetained {
        retention: IdleDynamicRetentionReceipt,
    },
}

/// Recovery owns the actual active session rather than a formatted diagnostic.
/// The caller can inspect its reaper, retire the execution error, then close.
pub struct ActiveImageStage<R: DeviceRuntime> {
    pub receipt: ImageStageExecutionReceipt,
    pub session: TensorComponentSession<R>,
}

#[must_use = "pipeline failure may own pending execution or plan cleanup"]
pub enum ImageGenerationFailure<R: DeviceRuntime> {
    Preflight(anyhow::Error),
    Unavailable,
    /// Retains every owner from a two-plan prefix handoff on failure. The inner
    /// failure owns the failed operation; these are the still-live peers.
    Prefix {
        source: Box<ImageGenerationFailure<R>>,
        request: Option<ferrum_execution::TensorRequestSession<R>>,
        stages: Vec<ActiveImageStage<R>>,
        weights: Option<SharedComponentWeights<R>>,
    },
    RequestExecution {
        source: TensorExecutionError<R>,
    },
    RequestClose {
        source: ferrum_execution::TensorRequestCloseFailure<R>,
    },
    Initialization {
        plan: ImageStagePlanReceipt,
        source: ComponentInitializationError<R>,
    },
    Execution {
        source: TensorExecutionError<R>,
        active: ActiveImageStage<R>,
    },
    Processing {
        source: anyhow::Error,
        active: Option<ActiveImageStage<R>>,
    },
    CloseReferenced {
        receipt: ImageStageExecutionReceipt,
        outcome: PlanRuntimeCloseOutcome<R>,
    },
    Close {
        receipt: ImageStageExecutionReceipt,
        source: PlanRuntimeCloseFailure<R>,
    },
    SharedWeightsReferenced {
        stage: ImageStage,
        outcome: SharedComponentWeightsCloseOutcome<R>,
    },
    SharedWeightsClose {
        stage: ImageStage,
        source: SharedComponentWeightsCloseFailure<R>,
    },
}

#[must_use = "keep recovery owners until quiescence and explicit retirement"]
pub struct ImageGenerationError<R: DeviceRuntime> {
    pub failure: ImageGenerationFailure<R>,
    pub completed_stages: Vec<ImageStageReceipt>,
    pub completed_steps: Vec<ImageDenoiseStepReceipt>,
}

impl<R: DeviceRuntime> fmt::Display for ImageGenerationFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageGenerationFailure::Preflight(e) => write!(f, "image preflight: {e:#}"),
            ImageGenerationFailure::Unavailable => f.write_str(
                "image pipeline stopped; recover any retained failure owners before replacing it",
            ),
            ImageGenerationFailure::Prefix { source, .. } => {
                write!(f, "image prefix: {source}")
            }
            ImageGenerationFailure::RequestExecution { source } => {
                write!(f, "image prefix request execution: {source}")
            }
            ImageGenerationFailure::RequestClose { source } => {
                write!(f, "image prefix request retirement: {source}")
            }
            ImageGenerationFailure::Initialization { plan, source } => {
                write!(f, "image {:?} initialization: {source}", plan.stage)
            }
            ImageGenerationFailure::Execution { source, active } => write!(
                f,
                "image {:?} execution: {source}",
                active.receipt.plan.stage
            ),
            ImageGenerationFailure::Processing { source, .. } => {
                write!(f, "image processing: {source}")
            }
            ImageGenerationFailure::CloseReferenced { receipt, .. } => write!(
                f,
                "image {:?} close still has live owners",
                receipt.plan.stage
            ),
            ImageGenerationFailure::Close { receipt, source } => write!(
                f,
                "image {:?} close: {:?}",
                receipt.plan.stage,
                source.failure()
            ),
            ImageGenerationFailure::SharedWeightsReferenced { stage, .. } => {
                write!(f, "image {stage:?} weight snapshot still has live owners")
            }
            ImageGenerationFailure::SharedWeightsClose { stage, source } => {
                write!(
                    f,
                    "image {stage:?} weight retirement: {:?}",
                    source.failure()
                )
            }
        }
    }
}
impl<R: DeviceRuntime> fmt::Display for ImageGenerationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.failure, f)
    }
}
impl<R: DeviceRuntime> fmt::Debug for ImageGenerationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for ImageGenerationError<R> {}

pub(super) struct PreparedStage<R: DeviceRuntime> {
    pub(super) component: Arc<CompiledTensorComponent<R>>,
    pub(super) plan: ImageStagePlanReceipt,
}

enum StageWeightInitialization<'a, R: DeviceRuntime> {
    Cold(ComponentWeightSource<'a>),
    Share(ComponentWeightSource<'a>),
    Attach(&'a SharedComponentWeights<R>),
}
impl<R: DeviceRuntime> PreparedStage<R> {
    pub(super) fn compile(
        stage: ImageStage,
        batch_size: usize,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
        pipeline: &QwenImage21Pipeline<R>,
    ) -> anyhow::Result<Self> {
        Self::compile_with_role(
            stage,
            ImageStagePlanRole::WholeStage,
            batch_size,
            family,
            options,
            pipeline,
        )
    }

    pub(super) fn compile_with_role(
        stage: ImageStage,
        role: ImageStagePlanRole,
        batch_size: usize,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
        pipeline: &QwenImage21Pipeline<R>,
    ) -> anyhow::Result<Self> {
        let started = Instant::now();
        ensure!(batch_size > 0, "stage batch size must be positive");
        let family_fingerprint = family.fingerprint()?;
        if pipeline.config.plan_cache != ImagePlanCachePolicy::Disabled {
            if let Some((component, mut plan)) = pipeline.plan_cache.get(
                pipeline.config.plan_cache,
                stage,
                role,
                batch_size,
                &family_fingerprint,
                options,
            )? {
                plan.compile_wall_duration_us = elapsed_us(started);
                plan.plan_cache_hit = true;
                return Ok(Self { component, plan });
            }
        }
        let family_id = family.family_id().clone();
        let profile = family.numerical_profile().id.clone();
        let component = CompiledTensorComponent::compile(
            family,
            &pipeline.registry,
            &pipeline.catalog,
            &pipeline.policy,
            &pipeline.materializers,
            options,
        )
        .with_context(|| {
            format!("image {stage:?} {role:?} plan compilation (batch {batch_size})")
        })?;
        let plan_hash = component
            .executable()
            .execution_plan()
            .plan_hash()
            .to_string();
        let memory = component.executable().execution_plan().payload().memory();
        let static_bytes = memory.static_bytes();
        let minimum_step_bytes = memory.minimum_step_bytes();
        let mut prepared = Self {
            component: Arc::new(component),
            plan: ImageStagePlanReceipt {
                stage,
                role,
                batch_size,
                family: family_id,
                profile,
                family_fingerprint,
                plan_hash,
                static_bytes,
                minimum_step_bytes,
                compile_wall_duration_us: elapsed_us(started),
                plan_cache_hit: false,
                plan_cache: None,
            },
        };
        if pipeline.config.plan_cache != ImagePlanCachePolicy::Disabled {
            prepared.plan.plan_cache = Some(pipeline.plan_cache.insert(
                pipeline.config.plan_cache,
                prepared.component.clone(),
                prepared.plan.clone(),
                options.clone(),
            )?);
        }
        prepared.plan.compile_wall_duration_us = elapsed_us(started);
        Ok(prepared)
    }

    pub(super) fn start(
        self,
        pipeline: &QwenImage21Pipeline<R>,
        job: u64,
        source: ComponentWeightSource<'_>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        self.start_with_mode(pipeline, job, StageWeightInitialization::Cold(source))
            .map(|(active, _)| active)
    }

    pub(super) fn start_shared(
        self,
        pipeline: &QwenImage21Pipeline<R>,
        job: u64,
        source: ComponentWeightSource<'_>,
    ) -> Result<(ActiveImageStage<R>, SharedComponentWeights<R>), ImageGenerationFailure<R>> {
        self.start_with_mode(pipeline, job, StageWeightInitialization::Share(source))
            .map(|(active, weights)| {
                (
                    active,
                    weights.expect("shared initialization returns its snapshot"),
                )
            })
    }

    pub(super) fn start_attached(
        self,
        pipeline: &QwenImage21Pipeline<R>,
        job: u64,
        weights: &SharedComponentWeights<R>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        self.start_with_mode(pipeline, job, StageWeightInitialization::Attach(weights))
            .map(|(active, _)| active)
    }

    fn start_with_mode(
        self,
        pipeline: &QwenImage21Pipeline<R>,
        job: u64,
        mode: StageWeightInitialization<'_, R>,
    ) -> Result<(ActiveImageStage<R>, Option<SharedComponentWeights<R>>), ImageGenerationFailure<R>>
    {
        let Self { component, plan } = self;
        let name = plan.stage.name();
        let config = (|| -> anyhow::Result<_> {
            Ok(ComponentInitializationConfig {
                request_id: id(format!("request.image.{job}.{name}.init"))?,
                run_id: id(format!("run.image.{job}.{name}.init"))?,
                transaction_id: id(format!("transaction.image.{job}.{name}.init"))?,
                static_policy: pipeline.config.static_initialization,
                executor: pipeline.config.executor.clone(),
            })
        })()
        .map_err(ImageGenerationFailure::Preflight)?;
        let started = Instant::now();
        let weights_reused = matches!(&mode, StageWeightInitialization::Attach(_));
        let initialized = match mode {
            StageWeightInitialization::Cold(source) => component
                .initialize(
                    pipeline.runtime.clone(),
                    pipeline.host.clone(),
                    source,
                    config,
                )
                .map(|session| (session, None)),
            StageWeightInitialization::Share(source) => component
                .initialize_shared_weights(
                    pipeline.runtime.clone(),
                    pipeline.host.clone(),
                    source,
                    config,
                )
                .map(|(session, weights)| (session, Some(weights))),
            StageWeightInitialization::Attach(weights) => component
                .initialize_with_shared_weights(
                    pipeline.runtime.clone(),
                    pipeline.host.clone(),
                    weights,
                    config,
                )
                .map(|session| (session, None)),
        };
        match initialized {
            Ok((session, weights)) => Ok((
                ActiveImageStage {
                    receipt: ImageStageExecutionReceipt {
                        plan,
                        initialization: session.initialization_receipt().clone(),
                        session_reused: false,
                        weights_reused,
                        completed_executions: 0,
                        initialization_wall_duration_us: elapsed_us(started),
                        execution_wall_durations_us: Vec::new(),
                    },
                    session,
                },
                weights,
            )),
            Err(source) => Err(ImageGenerationFailure::Initialization { plan, source }),
        }
    }
}

impl<R: DeviceRuntime> ActiveImageStage<R> {
    #[cfg(test)]
    pub(super) fn execute(
        self,
        job: u64,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<(Self, Vec<TensorExecutionOutput>), ImageGenerationFailure<R>> {
        self.execute_with(job, inputs, |session, run, request, inputs| {
            session.execute(run, request, inputs)
        })
    }

    pub(super) fn execute_on_actor(
        self,
        actor: &mut TensorSerialActor<R>,
        job: u64,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<(Self, Vec<TensorExecutionOutput>), ImageGenerationFailure<R>> {
        self.execute_with(job, inputs, |session, run, request, inputs| {
            session.execute_on_actor(actor, run, request, inputs)
        })
    }

    fn execute_with(
        mut self,
        job: u64,
        inputs: Vec<ManagedHostTensor>,
        execute: impl FnOnce(
            &mut TensorComponentSession<R>,
            RunId,
            RequestIdentity,
            Vec<ManagedHostTensor>,
        ) -> Result<Vec<TensorExecutionOutput>, TensorExecutionError<R>>,
    ) -> Result<(Self, Vec<TensorExecutionOutput>), ImageGenerationFailure<R>> {
        let stage = self.receipt.plan.stage.name();
        let ordinal = self.receipt.completed_executions;
        let ids = (|| -> anyhow::Result<(RunId, RequestIdentity)> {
            Ok((
                id(format!("run.image.{job}.{stage}.{ordinal}"))?,
                id(format!("request.image.{job}.{stage}.{ordinal}"))?,
            ))
        })();
        let (run, request) = match ids {
            Ok(ids) => ids,
            Err(source) => return Err(self.processing(source)),
        };
        let started = Instant::now();
        let result = execute(&mut self.session, run, request, inputs);
        self.receipt
            .execution_wall_durations_us
            .push(elapsed_us(started));
        match result {
            Ok(outputs) => {
                self.receipt.completed_executions += 1;
                Ok((self, outputs))
            }
            Err(source) => Err(ImageGenerationFailure::Execution {
                source,
                active: self,
            }),
        }
    }
    pub(super) fn processing(self, source: anyhow::Error) -> ImageGenerationFailure<R> {
        ImageGenerationFailure::Processing {
            source,
            active: Some(self),
        }
    }
    pub(super) fn close(self) -> Result<ImageStageReceipt, ImageGenerationFailure<R>> {
        let Self { receipt, session } = self;
        let started = Instant::now();
        match session.close() {
            Ok(PlanRuntimeCloseOutcome::Closed(close)) => Ok(ImageStageReceipt {
                execution: receipt,
                close: Some(close),
                retirement: ImageStageRetirement::Closed,
                close_wall_duration_us: elapsed_us(started),
            }),
            Ok(outcome @ PlanRuntimeCloseOutcome::Referenced { .. }) => {
                Err(ImageGenerationFailure::CloseReferenced { receipt, outcome })
            }
            Err(source) => Err(ImageGenerationFailure::Close { receipt, source }),
        }
    }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}
