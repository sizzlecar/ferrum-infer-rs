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

pub(super) use crate::common::construction::ConstructionContext;

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
    /// Upload made by a stage/head entry remains with the same submission on
    /// failure; an unwind cannot release the input before observing its work.
    pub(in crate::models) input: Option<B::Buffer>,
    /// A produced stage buffer can itself be referenced by the submission.
    /// It is retained only on the cold unresolved path; successful execution
    /// returns it directly without a box or another buffer allocation.
    produced_owners: Option<Box<dyn Send + Sync>>,
    layout: Option<FrameLayout>,
    preparation_only: bool,
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
    pub(in crate::models) fn new(context: B::Context) -> Self {
        Self {
            context,
            residual: None,
            residual_f32: None,
            branch_f32: None,
            norm_out: None,
            input: None,
            produced_owners: None,
            layout: None,
            preparation_only: false,
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
        if self.completed.take().is_some()
            || self.preparation_only && B::preparation_is_host_only(&self.context)
        {
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
    pub(in crate::models) fn restore(self, scratch: &mut LlamaFamilyScratch<B>) -> B::Context {
        let LlamaExecutionFrame {
            context,
            residual,
            residual_f32,
            branch_f32,
            norm_out,
            input,
            produced_owners,
            layout,
            preparation_only: _,
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
        drop(input);
        context
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
    pub(in crate::models) fn ensure_preparation_available(&self) -> Result<()> {
        if self.pending_device_frame.is_some() {
            return Err(FerrumError::backend(
                "model still owns an unresolved device execution frame",
            ));
        }
        if self.execution_context.is_none() {
            return Err(FerrumError::backend(
                "model execution context is already borrowed",
            ));
        }
        Ok(())
    }

    /// Numerical encoding borrows the actual frame. The catch boundary is
    /// outside every moved scratch owner, and terminal observation happens on
    /// that same context before restoration or release is permitted.
    pub(in crate::models) fn run_device_frame<T: Send + Sync + 'static>(
        &mut self,
        encode: impl FnOnce(&mut Self, &mut LlamaExecutionFrame<B>) -> T,
    ) -> DeviceFrameResult<T> {
        self.run_owned_frame(false, encode)
    }

    /// One retained owner for allocation, table initialization and physical
    /// reservation. Numerical callers already owning a frame use the
    /// in-context helpers instead of nesting this boundary.
    pub(in crate::models) fn run_preparation_frame<T: Send + Sync + 'static>(
        &mut self,
        prepare: impl FnOnce(&mut Self, &mut LlamaExecutionFrame<B>) -> T,
    ) -> DeviceFrameResult<T> {
        self.run_owned_frame(true, prepare)
    }

    fn run_owned_frame<T: Send + Sync + 'static>(
        &mut self,
        preparation_only: bool,
        encode: impl FnOnce(&mut Self, &mut LlamaExecutionFrame<B>) -> T,
    ) -> DeviceFrameResult<T> {
        if let Err(error) = self.ensure_preparation_available() {
            return DeviceFrameResult::Rejected(error);
        }
        let context = self
            .execution_context
            .take()
            .expect("validated original context");
        let mut frame = LlamaExecutionFrame::new(context);
        // Beginning an invocation must not manufacture a replacement owner or
        // erase an unresolved status from the preceding invocation.
        let beginning = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            B::begin_execution(&mut frame.context)
        }))
        .map_err(panic_error)
        .and_then(|result| result);
        if let Err(error) = beginning {
            self.pending_device_frame = Some(frame);
            return DeviceFrameResult::Held(error);
        }
        frame.preparation_only = preparation_only;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(self, &mut frame)))
                .map_err(panic_error);
        if result.is_err() {
            // Encoding failure may leave a partially recorded graph. Explicitly
            // discard it before observing the original stream; ordinary
            // completion observation must never silently cancel a capture.
            // Third-party hooks can panic too: keep every frame owner outside
            // this boundary and retain the original encoding error.
            let cancellation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                B::abort_graph_capture(&mut frame.context)
            }))
            .map_err(panic_error)
            .and_then(|result| result);
            if let Err(error) = cancellation {
                tracing::error!(%error, "graph cancellation after encoding failure failed");
            }
        }
        match frame.observe() {
            FrameObservation::Terminal { owners, error } => {
                self.execution_context = Some(owners.restore(&mut self.scratch));
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
        match self.recover_token_frame() {
            crate::common::llm::ModelExecutionRecovery::Quiescent { error } => {
                if let Some(error) = error {
                    tracing::error!(%error, "retained device frame reached terminal failure");
                }
                Ok(true)
            }
            crate::common::llm::ModelExecutionRecovery::Held { error } => Err(error),
        }
    }

    pub(crate) fn recover_token_frame(&mut self) -> crate::common::llm::ModelExecutionRecovery {
        use crate::common::llm::ModelExecutionRecovery;
        let Some(frame) = self.pending_device_frame.take() else {
            return ModelExecutionRecovery::Quiescent { error: None };
        };
        match frame.observe() {
            FrameObservation::Terminal { owners, error } => {
                self.execution_context = Some(owners.restore(&mut self.scratch));
                ModelExecutionRecovery::Quiescent { error }
            }
            FrameObservation::Unresolved { owners, error } => {
                self.pending_device_frame = Some(owners);
                ModelExecutionRecovery::Held { error }
            }
        }
    }

    /// Retire this instance's original context and captured programs. A failed
    /// retirement keeps the model intact for recovery; a successful retirement
    /// makes the instance permanently unavailable for further execution.
    pub fn close(&mut self) -> Result<()> {
        use crate::common::llm::ModelExecutionRecovery;
        let prior_error = match self.recover_token_frame() {
            ModelExecutionRecovery::Held { error } => return Err(error),
            ModelExecutionRecovery::Quiescent { error } => error,
        };
        let Some(context) = self.execution_context.as_mut() else {
            return prior_error.map_or(Ok(()), Err);
        };
        let retirement =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| B::retire_context(context)))
                .map_err(panic_error)
                .and_then(|result| result);
        retirement?;
        // Graphs have retired before any captured model buffers can be freed.
        self.execution_context = None;
        prior_error.map_or(Ok(()), Err)
    }

    pub(super) fn truncate_token_cache(&mut self, cache_id: &str, new_len: usize) {
        self.validate_prepared_truncate(cache_id, new_len)
            .unwrap_or_else(|error| std::panic::panic_any(error));
        self.recover_for_mutation();
        self.run_preparation_frame(|model, frame| {
            B::reset_graph(&mut frame.context, super::SINGLE_ITEM_GRAPH_KEY);
            if let Some(caches) = model.kv_caches.get_mut(cache_id) {
                for cache in caches {
                    if new_len < K::len(cache) {
                        K::set_len(cache, new_len);
                    }
                }
            }
            model.graph_warmup = 0;
            model.graph_capture_failed = false;
        })
        .into_legacy_result();
    }

    pub(super) fn release_token_cache(&mut self, cache_id: &str) {
        use crate::common::llm::ModelTokenCloseOutcome;
        match self.close_managed_token_cache(cache_id) {
            ModelTokenCloseOutcome::Ended { error: None } => {}
            ModelTokenCloseOutcome::Ended { error: Some(error) }
            | ModelTokenCloseOutcome::Held { error } => std::panic::panic_any(error),
        }
    }

    fn recover_for_mutation(&mut self) {
        use crate::common::llm::ModelExecutionRecovery;
        match self.recover_token_frame() {
            ModelExecutionRecovery::Quiescent { error: None } => {}
            ModelExecutionRecovery::Quiescent { error: Some(error) }
            | ModelExecutionRecovery::Held { error } => std::panic::panic_any(error),
        }
        self.ensure_preparation_available()
            .unwrap_or_else(|error| std::panic::panic_any(error));
    }

    pub(super) fn reset_token_caches(&mut self) {
        self.recover_for_mutation();
        self.run_preparation_frame(|model, frame| {
            // Destroy captured programs while their original buffers still live.
            B::retire_graphs_for_storage_mutation(&mut frame.context)
                .unwrap_or_else(|error| std::panic::panic_any(error));
            model.graph_warmup = 0;
            model.graph_capture_failed = false;
            model.batched_graph_keys_seen.clear();
            model.batched_graph_warmup = 0;
            model.batched_graph_failed = false;
            model.batched_pointers_for = None;
            model.unified_graph_keys_seen.clear();
            model.unified_graph_warmup = 0;
            model.unified_graph_failed = false;
            model.prepared_prefixes.clear();
            model.kv_caches.clear();
            model.kv_free_pool.clear();
            model.paged_pools = None;
            model.paged_block_alloc = None;
            model.paged_dims = None;
            model.lora_cache_adapters.clear();
        })
        .into_legacy_result();
    }
}

impl<B: MoeLlmBackend, K: KvLayer<B>> Drop for LlamaFamilyModel<B, K> {
    fn drop(&mut self) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.close()));
        if self.pending_device_frame.is_some() || self.execution_context.is_some() {
            eprintln!("cannot drop model with unretired original device context: {result:?}");
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod preparation_tests;

#[cfg(test)]
mod instance_context_tests;
