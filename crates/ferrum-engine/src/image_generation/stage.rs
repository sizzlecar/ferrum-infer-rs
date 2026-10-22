use super::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStage {
    TextEncoder,
    Noise,
    Denoiser,
    Decoder,
}
impl ImageStage {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text",
            Self::Noise => "noise",
            Self::Denoiser => "denoise",
            Self::Decoder => "decode",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStagePlanReceipt {
    pub stage: ImageStage,
    pub family: ModelFamilyId,
    pub profile: NumericalProfileId,
    pub family_fingerprint: String,
    pub plan_hash: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStageExecutionReceipt {
    pub plan: ImageStagePlanReceipt,
    pub initialization: ComponentInitializationReceipt,
    pub completed_executions: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageStageReceipt {
    pub execution: ImageStageExecutionReceipt,
    pub close: PlanRuntimeCloseReceipt,
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
}

#[must_use = "keep recovery owners until quiescence and explicit retirement"]
pub struct ImageGenerationError<R: DeviceRuntime> {
    pub failure: ImageGenerationFailure<R>,
    pub completed_stages: Vec<ImageStageReceipt>,
    pub completed_steps: Vec<ImageDenoiseStepReceipt>,
}

impl<R: DeviceRuntime> fmt::Display for ImageGenerationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            ImageGenerationFailure::Preflight(e) => write!(f, "image preflight: {e}"),
            ImageGenerationFailure::Unavailable => f.write_str("image pipeline stopped after a stage failure; recover retained owners before replacing it"),
            ImageGenerationFailure::Initialization { plan, source } => write!(f, "image {:?} initialization: {source}", plan.stage),
            ImageGenerationFailure::Execution { source, active } => write!(f, "image {:?} execution: {source}", active.receipt.plan.stage),
            ImageGenerationFailure::Processing { source, .. } => write!(f, "image processing: {source}"),
            ImageGenerationFailure::CloseReferenced { receipt, .. } => write!(f, "image {:?} close still has live owners", receipt.plan.stage),
            ImageGenerationFailure::Close { receipt, source } => write!(f, "image {:?} close: {:?}", receipt.plan.stage, source.failure()),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for ImageGenerationError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for ImageGenerationError<R> {}

pub(super) struct PreparedStage<R: DeviceRuntime> {
    component: CompiledTensorComponent<R>,
    plan: ImageStagePlanReceipt,
}
impl<R: DeviceRuntime> PreparedStage<R> {
    pub(super) fn compile(
        stage: ImageStage,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
        pipeline: &QwenImage21Pipeline<R>,
    ) -> anyhow::Result<Self> {
        let family_fingerprint = family.fingerprint()?;
        let family_id = family.family_id().clone();
        let profile = family.numerical_profile().id.clone();
        let component = CompiledTensorComponent::compile(
            family,
            &pipeline.registry,
            &pipeline.catalog,
            &pipeline.policy,
            &pipeline.materializers,
            options,
        )?;
        let plan_hash = component
            .executable()
            .execution_plan()
            .plan_hash()
            .to_string();
        Ok(Self {
            component,
            plan: ImageStagePlanReceipt {
                stage,
                family: family_id,
                profile,
                family_fingerprint,
                plan_hash,
            },
        })
    }

    pub(super) fn start(
        self,
        pipeline: &QwenImage21Pipeline<R>,
        job: u64,
        source: ComponentWeightSource<'_>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
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
        match component.initialize(
            pipeline.runtime.clone(),
            pipeline.host.clone(),
            source,
            config,
        ) {
            Ok(session) => Ok(ActiveImageStage {
                receipt: ImageStageExecutionReceipt {
                    plan,
                    initialization: session.initialization_receipt().clone(),
                    completed_executions: 0,
                },
                session,
            }),
            Err(source) => Err(ImageGenerationFailure::Initialization { plan, source }),
        }
    }
}

impl<R: DeviceRuntime> ActiveImageStage<R> {
    pub(super) fn execute(
        mut self,
        job: u64,
        inputs: Vec<ManagedHostTensor>,
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
        match self.session.execute(run, request, inputs) {
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
        match session.close() {
            Ok(PlanRuntimeCloseOutcome::Closed(close)) => Ok(ImageStageReceipt {
                execution: receipt,
                close,
            }),
            Ok(outcome @ PlanRuntimeCloseOutcome::Referenced { .. }) => {
                Err(ImageGenerationFailure::CloseReferenced { receipt, outcome })
            }
            Err(source) => Err(ImageGenerationFailure::Close { receipt, source }),
        }
    }
}
