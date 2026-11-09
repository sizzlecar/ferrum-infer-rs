//! One visual input on the pipeline's original serial actor. Binding returns
//! the original phase after actual visual retirement, before causal install.
use super::*;
use crate::image_generation::p_h3::expansion::vision::{
    ExpansionVisionFailure, ExpansionVisionHostRequirements, ExpansionVisionOptions,
    PreparedExpansionVision,
};
use crate::image_generation::{ImageEditRequest, ImagePipelineConfig};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::{
    qwen3_vl_vision::hf::Qwen3VLVisionHfImagePlan,
    qwen_image21::image_input::{png_dimensions_hint, ImageInputLimits, ManagedRgba8Image},
};

pub(super) struct PreparedVisual<R: DeviceRuntime> {
    producer: PreparedExpansionVision<R>,
    encoded: ManagedHostBytes,
    dimensions: (u32, u32),
    limits: ImageInputLimits,
}
impl<R: DeviceRuntime> PreparedVisual<R> {
    pub fn prepare(
        assets: &NativeTextExpansion,
        composition: &RuntimeComposition<R>,
        host: &HostMemoryAccount,
        config: &ImagePipelineConfig,
        request: &ImageEditRequest,
    ) -> Result<Self> {
        ensure!(
            host.owns(&request.image) && request.image.class() == HostMemoryClass::InputPreprocess,
            "prompt expansion PNG requires its original Input Host owner"
        );
        let visual = assets.visual_assets().context(
            "image-conditioned prompt expansion requires validated visual sources and processor",
        )?;
        let limits = visual.processor().image_limits();
        let dimensions = png_dimensions_hint(request.image.bytes(), limits)
            .context("prompt expansion requires a bounded static RGB/RGBA8 PNG header")?;
        let producer = PreparedExpansionVision::compile_assets(
            composition,
            visual.clone(),
            dimensions.0,
            dimensions.1,
            ExpansionVisionOptions {
                limits: visual.limits(),
                profile: visual.profile().clone(),
                device_capacity_bytes: config.device_capacity_bytes,
                static_policy: config.static_initialization,
                executor: config.executor.clone(),
            },
        )?;
        Ok(Self {
            producer,
            encoded: request.image.clone(),
            dimensions,
            limits,
        })
    }
    pub fn plan(&self) -> &Qwen3VLVisionHfImagePlan {
        self.producer.plan()
    }
    pub fn requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<ExpansionVisionHostRequirements> {
        self.producer.host_requirements(host)
    }
    pub fn decoded_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let bytes = u64::from(self.dimensions.0)
            .checked_mul(u64::from(self.dimensions.1))
            .and_then(|n| n.checked_mul(4))
            .context("decoded visual PNG capacity overflow")?;
        Ok(host.backing_bytes(usize::try_from(bytes)?)?)
    }

    fn same_image(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions
            && self.producer.same_reuse_domain(&other.producer)
            // Original encoded bytes are immutable and compared in full.
            // Different encodings of equal pixels may miss; no hash collision
            // or prompt/file-name identity can produce a hit.
            && self.encoded.bytes() == other.encoded.bytes()
    }
}

/// At most one completed Output owner, held by the cohort rather than its
/// first member. Drop before a miss acquires a new producer or causal install.
pub(super) struct RetainedCohortVisual<R: DeviceRuntime> {
    assets: Arc<NativeTextExpansion>,
    visual: PreparedVisual<R>,
    output: TensorExecutionOutput,
    producer_job: u64,
    producer_request_id: Option<uuid::Uuid>,
}
impl<R: DeviceRuntime> RetainedCohortVisual<R> {
    pub(super) fn new(
        assets: Arc<NativeTextExpansion>,
        visual: PreparedVisual<R>,
        output: TensorExecutionOutput,
        producer_job: u64,
        correlation: Option<ferrum_types::ImageRequestCorrelation>,
    ) -> Self {
        Self {
            assets,
            visual,
            output,
            producer_job,
            producer_request_id: correlation.map(|c| c.request_id),
        }
    }

    pub(super) fn matches(
        &self,
        assets: &Arc<NativeTextExpansion>,
        visual: &PreparedVisual<R>,
    ) -> bool {
        Arc::ptr_eq(&self.assets, assets) && self.visual.same_image(visual)
    }
}

pub(super) struct VisionInitialization<R: DeviceRuntime> {
    assets: Arc<NativeTextExpansion>,
    prepared: PreparedTextExpansion,
    visual: PreparedVisual<R>,
    phase: HostMemoryScope,
    _requirements: InitializationRequirements,
}
impl<R: DeviceRuntime> VisionInitialization<R> {
    pub fn new(
        assets: Arc<NativeTextExpansion>,
        prepared: PreparedTextExpansion,
        visual: PreparedVisual<R>,
        phase: HostMemoryScope,
        requirements: InitializationRequirements,
    ) -> Self {
        Self {
            assets,
            prepared,
            visual,
            phase,
            _requirements: requirements,
        }
    }
}

/// Recoverable rejection returns the same grant. A cohort can drop this
/// member's preparation and use the grant for another member's visual input;
/// an uncertain native retirement instead follows the ordinary fatal owner.
pub(super) enum VisualInputBinding {
    Bound {
        phase: HostMemoryScope,
        output: TensorExecutionOutput,
    },
    Rejected {
        phase: HostMemoryScope,
        diagnostic: String,
    },
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn initialize_expansion(
        &mut self,
        assets: Arc<NativeTextExpansion>,
        mut prepared: PreparedTextExpansion,
        phase: HostMemoryScope,
        job: u64,
        correlation: Option<ferrum_types::ImageRequestCorrelation>,
    ) -> InitResult<ProductExpansionPhase<R>> {
        assets
            .configure_request_timing(
                &mut prepared,
                self.execution.config.executor.device_timing_mode,
            )
            .map_err(|error| preparation_error(error.into()))?;
        let diagnostics = match self.prompt_expansion_diagnostics {
            super::super::super::PH3ImagePromptExpansionDiagnosticPolicy::Disabled => None,
            policy => self.execution.continuous_trace.as_ref().map(|trace| {
                crate::image_generation::p_h3::expansion::DiagnosticContext {
                    policy,
                    trace: trace.handle(),
                    job,
                    request_id: correlation.map(|c| c.request_id),
                }
            }),
        };
        ProductExpansionPhase::initialize_with_diagnostics(
            assets,
            self.expansion_composition(),
            prepared,
            phase,
            diagnostics,
        )
        .map_err(|error| {
            if error.cleanup().is_some() {
                self.execution.stopped = true;
                ImageBatchError::Fatal(Box::new(error))
            } else {
                ImageBatchError::Rejected(error.to_string())
            }
        })
    }

    pub(super) fn advance_visual_initialization(
        &mut self,
        mut pending: VisionInitialization<R>,
        job: u64,
        correlation: Option<ferrum_types::ImageRequestCorrelation>,
    ) -> InitResult<std::result::Result<ProductExpansionPhase<R>, String>> {
        let phase = match self.bind_visual_input(
            &pending.assets,
            &mut pending.prepared,
            &pending.visual,
            pending.phase,
            job,
            correlation,
        )? {
            VisualInputBinding::Bound { phase, output } => {
                drop(output);
                phase
            }
            VisualInputBinding::Rejected { phase, diagnostic } => {
                drop(phase);
                return Ok(Err(diagnostic));
            }
        };
        match self.initialize_expansion(pending.assets, pending.prepared, phase, job, correlation) {
            Ok(phase) => Ok(Ok(phase)),
            Err(ImageBatchError::Rejected(diagnostic)) => Ok(Err(diagnostic)),
            Err(error) => Err(error),
        }
    }

    /// Produce and bind one image without installing causal weights. The
    /// caller retains source requirements and owns subsequent admission.
    pub(super) fn bind_visual_input(
        &mut self,
        assets: &NativeTextExpansion,
        prepared: &mut PreparedTextExpansion,
        visual: &PreparedVisual<R>,
        phase: HostMemoryScope,
        job: u64,
        correlation: Option<ferrum_types::ImageRequestCorrelation>,
    ) -> InitResult<VisualInputBinding> {
        self.bind_visual_input_with_reuse(assets, prepared, visual, phase, job, correlation, None)
    }

    pub(super) fn bind_visual_input_with_reuse(
        &mut self,
        assets: &NativeTextExpansion,
        prepared: &mut PreparedTextExpansion,
        visual: &PreparedVisual<R>,
        phase: HostMemoryScope,
        job: u64,
        correlation: Option<ferrum_types::ImageRequestCorrelation>,
        retained: Option<&RetainedCohortVisual<R>>,
    ) -> InitResult<VisualInputBinding> {
        // Compilation completed before this producer boundary. Keep the
        // original result/phase owner intact, including recoverable rejection.
        let span = self.trace_phase_started(
            correlation,
            None,
            crate::image_generation::continuous_trace::ImageTracePhase::PromptExpansionVision,
        );
        let result = self.bind_visual_input_inner(assets, prepared, visual, phase, job, retained);
        if let Ok(VisualInputBinding::Bound { output, .. }) = &result {
            prepared.record_vision_binding(
                crate::image_generation::p_h3::expansion::VisionBindingReceipt {
                    reused: retained.is_some(),
                    producer_job: retained.map_or(job, |r| r.producer_job),
                    producer_request_id: retained
                        .map(|r| r.producer_request_id)
                        .unwrap_or_else(|| correlation.map(|c| c.request_id)),
                    output_bytes: output.bytes().len(),
                },
            );
        }
        self.trace_phase_finished(
            span,
            if matches!(&result, Ok(VisualInputBinding::Bound { .. })) {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
            } else {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
            },
        );
        result
    }

    fn bind_visual_input_inner(
        &mut self,
        assets: &NativeTextExpansion,
        prepared: &mut PreparedTextExpansion,
        visual: &PreparedVisual<R>,
        phase: HostMemoryScope,
        job: u64,
        retained: Option<&RetainedCohortVisual<R>>,
    ) -> InitResult<VisualInputBinding> {
        let host = phase.account();
        if let Some(retained) = retained {
            ensure_reuse_identity(assets, visual, retained).map_err(preparation_error)?;
            return Ok(
                match assets.bind_image_outputs(
                    prepared,
                    &host,
                    std::slice::from_ref(&retained.output),
                ) {
                    Ok(()) => VisualInputBinding::Bound {
                        phase,
                        output: retained.output.clone(),
                    },
                    Err(error) => VisualInputBinding::Rejected {
                        phase,
                        diagnostic: format!("reused visual prompt causal bridge: {error}"),
                    },
                },
            );
        }
        // Strict decode is before any native acquisition. Bad CRC/chunks or
        // a changed geometry rejects just this request and returns its grant.
        let decoded = match ManagedRgba8Image::decode_png(&host, &visual.encoded, visual.limits) {
            Ok(decoded) if (decoded.width(), decoded.height()) == visual.dimensions => decoded,
            Ok(_) => {
                return Ok(VisualInputBinding::Rejected {
                    phase,
                    diagnostic: "visual PNG differs from prepared dimensions".into(),
                });
            }
            Err(error) => {
                return Ok(VisualInputBinding::Rejected {
                    phase,
                    diagnostic: format!("visual PNG decode: {error:#}"),
                });
            }
        };
        let run = RunId::new(format!("run.image.h3.{job}.prompt_vision"))
            .map_err(|e| preparation_error(e.into()))?;
        let request = RequestIdentity::new(format!("request.image.h3.{job}"))
            .map_err(|e| preparation_error(e.into()))?;
        let transaction = TransactionId::new(format!("transaction.image.h3.{job}.prompt_vision"))
            .map_err(|e| preparation_error(e.into()))?;
        let Some(actor) = self.execution.serial_actor.as_mut() else {
            return Ok(VisualInputBinding::Rejected {
                phase,
                diagnostic: "image serial actor is closed".into(),
            });
        };
        let completed = match visual.producer.execute_on_actor(
            phase,
            &decoded,
            actor,
            run,
            request,
            transaction,
        ) {
            Ok(completed) => completed,
            Err(ExpansionVisionFailure::Preflight { source, phase }) => {
                return Ok(VisualInputBinding::Rejected {
                    phase,
                    diagnostic: format!("visual prompt preprocessing: {source:#}"),
                });
            }
            Err(failure) => {
                self.execution.stopped = true;
                return Err(ImageBatchError::Fatal(Box::new(failure)));
            }
        };
        drop(decoded);
        // The completed value can only exist after original session.close()
        // acknowledges Closed. Its exact same phase is used for the bridge.
        let (output, phase, _close) = completed.into_parts();
        if let Err(error) =
            assets.bind_image_outputs(prepared, &host, std::slice::from_ref(&output))
        {
            return Ok(VisualInputBinding::Rejected {
                phase,
                diagnostic: format!("visual prompt causal bridge: {error}"),
            });
        }
        Ok(VisualInputBinding::Bound { phase, output })
    }
}

fn ensure_reuse_identity<R: DeviceRuntime>(
    assets: &NativeTextExpansion,
    visual: &PreparedVisual<R>,
    retained: &RetainedCohortVisual<R>,
) -> Result<()> {
    ensure!(
        std::ptr::eq(assets, retained.assets.as_ref()) && retained.visual.same_image(visual),
        "visual reuse differs from its original cohort assets or exact image"
    );
    Ok(())
}

#[cfg(test)]
pub(super) mod tests;
