//! Owners moved out of model scratch during one physical device execution.
//!
//! Encoding borrows this frame. In particular it must not move a scratch
//! buffer into an inner closure whose unwind would free it before completion
//! can be observed. The enclosing model task retains the remaining weights,
//! scratch and KV owners until this frame is terminal.

use ferrum_kernels::backend::{
    BackendCompletion, BackendMoeFused, KvLayer, MoeLlmBackend, QuantLlmBackend,
};
use ferrum_types::{FerrumError, Result};

use super::{LlamaFamilyModel, LlamaFamilyScratch};

#[derive(Clone, Copy)]
pub(in crate::models) enum FrameLayout {
    Single,
    Unified,
}

pub(in crate::models) struct LlamaExecutionFrame<B: QuantLlmBackend + BackendMoeFused> {
    pub(in crate::models) context: B::Context,
    pub(in crate::models) residual: Option<B::Buffer>,
    pub(in crate::models) residual_f32: Option<B::Buffer>,
    pub(in crate::models) branch_f32: Option<B::Buffer>,
    pub(in crate::models) norm_out: Option<B::Buffer>,
    /// A produced stage buffer can itself be referenced by the submission.
    /// It is retained only on the cold unresolved path; successful execution
    /// returns it directly without a box or another buffer allocation.
    produced_owners: Option<Box<dyn Send + Sync>>,
    layout: Option<FrameLayout>,
    pub(in crate::models) completed: Option<QuiescentContext>,
}

/// Produced only by observing the actual context, consumed only after the
/// core's final readback returned. No further encoding may follow that mark.
pub(in crate::models) struct QuiescentContext;

pub(in crate::models) fn checked_sync<B: QuantLlmBackend + BackendMoeFused>(
    context: &mut B::Context,
) -> QuiescentContext {
    match B::observe_completion(context) {
        BackendCompletion::Succeeded => QuiescentContext,
        BackendCompletion::Failed(error) | BackendCompletion::Unresolved(error) => {
            std::panic::panic_any(error)
        }
    }
}

/// Both outcomes retain the original frame. Only an actual terminal backend
/// observation constructs the type that permits restoring/releasing scratch.
pub(in crate::models) enum FrameObservation<B: QuantLlmBackend + BackendMoeFused> {
    Terminal {
        owners: QuiescentLlamaFrame<B>,
        error: Option<FerrumError>,
    },
    Unresolved {
        owners: LlamaExecutionFrame<B>,
        error: FerrumError,
    },
}

pub(in crate::models) struct QuiescentLlamaFrame<B: QuantLlmBackend + BackendMoeFused> {
    frame: LlamaExecutionFrame<B>,
}

pub(in crate::models) enum DeviceFrameResult<T> {
    /// Nothing from this invocation was encoded. This does not certify that
    /// an already-held model frontier is reusable.
    Rejected(FerrumError),
    Terminal(Result<T>),
    /// The exact frame, product owners and model stay in the recovery service.
    Held(FerrumError),
}

impl<T> DeviceFrameResult<T> {
    /// Compatibility for retained numerical callers, until their common
    /// token consumer takes the typed result directly. This never discards
    /// an unresolved frame: it remains with the original model owner.
    pub(in crate::models) fn into_legacy_result(self) -> T {
        match self {
            Self::Terminal(Ok(value)) => value,
            Self::Terminal(Err(error)) | Self::Rejected(error) | Self::Held(error) => {
                std::panic::panic_any(error)
            }
        }
    }
}

impl<B: QuantLlmBackend + BackendMoeFused> LlamaExecutionFrame<B> {
    /// The context is owned before prefix/block-table preparation can encode
    /// anything. Scratch can be attached later after the exact suffix/batch
    /// extent is known; that does not create another submission context.
    pub(in crate::models) fn new() -> Self {
        Self {
            context: B::new_context(),
            residual: None,
            residual_f32: None,
            branch_f32: None,
            norm_out: None,
            produced_owners: None,
            layout: None,
            completed: None,
        }
    }

    /// Validate the whole take before moving the first owner. No placeholder
    /// device allocations or successful-path buffer clones are introduced.
    pub(in crate::models) fn take_scratch(
        &mut self,
        scratch: &mut LlamaFamilyScratch<B>,
        layout: FrameLayout,
        device_residual_shadow: bool,
    ) -> Result<()> {
        if self.layout.is_some() {
            return Err(FerrumError::model(
                "device frame already owns model scratch",
            ));
        }
        let (residual, shadow, branch, norm) = match layout {
            FrameLayout::Single => (
                &mut scratch.residual,
                &mut scratch.residual_f32_shadow,
                &mut scratch.sandwich_branch_f32,
                None::<&mut Option<B::Buffer>>,
            ),
            FrameLayout::Unified => (
                &mut scratch.unified_residual,
                &mut scratch.unified_residual_f32_shadow,
                &mut scratch.unified_sandwich_branch_f32,
                Some(&mut scratch.unified_norm_out),
            ),
        };
        if residual.is_none()
            || device_residual_shadow && (shadow.is_none() || branch.is_none())
            || norm.as_ref().is_some_and(|buffer| buffer.is_none())
        {
            return Err(FerrumError::model(
                "device execution scratch is still owned by another frame or was not prepared",
            ));
        }
        self.residual = residual.take();
        self.residual_f32 = device_residual_shadow.then(|| shadow.take()).flatten();
        self.branch_f32 = device_residual_shadow.then(|| branch.take()).flatten();
        self.norm_out = norm.and_then(Option::take);
        self.layout = Some(layout);
        Ok(())
    }

    pub(in crate::models) fn observe(mut self) -> FrameObservation<B> {
        if self.completed.take().is_some() {
            return FrameObservation::Terminal {
                owners: QuiescentLlamaFrame { frame: self },
                error: None,
            };
        }
        // Keep self outside the unwind scope. A backend observation panic is
        // not permission to destroy its command buffer, graph lease or scratch.
        let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            B::observe_completion(&mut self.context)
        }))
        .unwrap_or_else(|payload| BackendCompletion::Unresolved(panic_error(payload)));
        match observed {
            BackendCompletion::Succeeded => FrameObservation::Terminal {
                owners: QuiescentLlamaFrame { frame: self },
                error: None,
            },
            BackendCompletion::Failed(error) => FrameObservation::Terminal {
                owners: QuiescentLlamaFrame { frame: self },
                error: Some(error),
            },
            BackendCompletion::Unresolved(error) => FrameObservation::Unresolved {
                owners: self,
                error,
            },
        }
    }
}

impl<B: QuantLlmBackend + BackendMoeFused> QuiescentLlamaFrame<B> {
    pub(in crate::models) fn restore(self, scratch: &mut LlamaFamilyScratch<B>) {
        let LlamaExecutionFrame {
            context,
            residual,
            residual_f32,
            branch_f32,
            norm_out,
            produced_owners,
            layout,
            completed: _,
        } = self.frame;
        match layout {
            Some(FrameLayout::Single) => {
                scratch.residual = residual;
                if residual_f32.is_some() {
                    scratch.residual_f32_shadow = residual_f32;
                    scratch.sandwich_branch_f32 = branch_f32;
                }
            }
            Some(FrameLayout::Unified) => {
                scratch.unified_residual = residual;
                if residual_f32.is_some() {
                    scratch.unified_residual_f32_shadow = residual_f32;
                    scratch.unified_sandwich_branch_f32 = branch_f32;
                }
                scratch.unified_norm_out = norm_out;
            }
            None => {}
        }
        drop(produced_owners);
        drop(context);
    }
}

fn panic_error(payload: Box<dyn std::any::Any + Send>) -> FerrumError {
    if let Some(error) = payload.downcast_ref::<FerrumError>() {
        return error.clone();
    }
    let detail = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload");
    FerrumError::backend(format!("device model operation panicked: {detail}"))
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyModel<B, K> {
    /// Numerical encoding borrows the actual frame. The catch boundary is
    /// outside every moved scratch owner, and terminal observation happens on
    /// that same context before restoration or release is permitted.
    pub(in crate::models) fn run_device_frame<T: Send + Sync + 'static>(
        &mut self,
        encode: impl FnOnce(&mut Self, &mut LlamaExecutionFrame<B>) -> T,
    ) -> DeviceFrameResult<T> {
        if self.pending_device_frame.is_some() {
            return DeviceFrameResult::Rejected(FerrumError::backend(
                "model still owns an unresolved device execution frame",
            ));
        }
        let mut frame = LlamaExecutionFrame::new();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(self, &mut frame)))
                .map_err(panic_error);
        match frame.observe() {
            FrameObservation::Terminal { owners, error } => {
                owners.restore(&mut self.scratch);
                let result = match (result, error) {
                    (Ok(value), None) => Ok(value),
                    (Ok(_), Some(error)) => Err(error),
                    (Err(primary), secondary) => {
                        if let Some(error) = secondary {
                            tracing::error!(%primary, %error, "device encoding and terminal observation both failed");
                        }
                        Err(primary)
                    }
                };
                DeviceFrameResult::Terminal(result)
            }
            FrameObservation::Unresolved { mut owners, error } => {
                let primary = match result {
                    Ok(product) => {
                        owners.produced_owners = Some(Box::new(product));
                        error
                    }
                    Err(primary) => {
                        tracing::error!(%primary, %error, "device lifetime remains unresolved after encoding failure");
                        primary
                    }
                };
                self.pending_device_frame = Some(owners);
                DeviceFrameResult::Held(primary)
            }
        }
    }

    /// Called only by the actual model's recovery worker. This does not retry
    /// numerical work, publish logits, or advance any participant frontier.
    pub(crate) fn recover_device_frame(&mut self) -> Result<bool> {
        let Some(frame) = self.pending_device_frame.take() else {
            return Ok(true);
        };
        match frame.observe() {
            FrameObservation::Terminal { owners, error } => {
                owners.restore(&mut self.scratch);
                // A terminal error proves lifetime only. The worker still
                // reports the original failure and closes affected sequences.
                if let Some(error) = error {
                    tracing::error!(%error, "retained device frame reached terminal failure");
                }
                Ok(true)
            }
            FrameObservation::Unresolved { owners, error } => {
                self.pending_device_frame = Some(owners);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests;
