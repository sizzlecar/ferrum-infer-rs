//! Aggregate observations of the original stage owners, without another
//! execution context, resource registry, mathematical retry or worker.
use super::*;
use crate::common::llm::{
    ModelExecutionRecovery, ModelKvReservation, ModelTokenBatchOutcome, ModelTokenCloseOutcome,
    ModelTokenExecutionMode, ModelTokenInput, ModelTokenOutcome, ModelTokenProduct, ModelTokenWork,
};
use ferrum_execution::model_executor::{KvSlotRequest, LogitsReturnPolicy};

// Stage operations remain private, as in the original pipeline decode impl.
// Every method here is private to the pipeline; no public bound is exposed.
#[allow(private_bounds)]
impl<B, K> LlamaFamilyPipelineModel<B, K>
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM + LlamaPipelineStageBatchOps<B> + Send,
{
    pub(super) fn recover_pipeline_stages(&mut self) -> ModelExecutionRecovery {
        let mut first_error = None;
        let mut held = false;
        for stage in &mut self.stages {
            match stage.recover_token_execution() {
                ModelExecutionRecovery::Quiescent { error } => {
                    if first_error.is_none() {
                        first_error = error;
                    }
                }
                ModelExecutionRecovery::Held { error } => {
                    held = true;
                    first_error.get_or_insert(error);
                }
            }
        }
        if held {
            ModelExecutionRecovery::Held {
                error: first_error.expect("held stage supplies an error"),
            }
        } else {
            ModelExecutionRecovery::Quiescent { error: first_error }
        }
    }

    pub(super) fn close_pipeline_sequence(&mut self, cache_id: &str) -> ModelTokenCloseOutcome {
        let mut first_error = None;
        let mut held = false;
        // Each stage's idempotent close retains its own cache/frame on failure.
        // Host bridges are owned Vecs: a retired stage cannot release another
        // stage's pending input. Retry observes those same owners only.
        for stage in &mut self.stages {
            match stage.close_token_execution(cache_id) {
                ModelTokenCloseOutcome::Ended { error } => {
                    if first_error.is_none() {
                        first_error = error;
                    }
                }
                ModelTokenCloseOutcome::Held { error } => {
                    held = true;
                    first_error.get_or_insert(error);
                }
            }
        }
        if held {
            ModelTokenCloseOutcome::Held {
                error: first_error.expect("held stage supplies an error"),
            }
        } else {
            ModelTokenCloseOutcome::Ended { error: first_error }
        }
    }

    fn validate_pipeline_work(&self, input: &ModelTokenInput<'_>) -> Result<()> {
        if self.token_execution_mode() == ModelTokenExecutionMode::Unsupported {
            return Err(FerrumError::unsupported(
                "pipeline backend has not qualified retained execution",
            ));
        }
        if self.stages.is_empty() || self.stages.len() != self.placement.len() {
            return Err(FerrumError::request_validation(
                "pipeline no longer has its complete stage owners",
            ));
        }
        // Check the original incarnation/prompt/frontier in every stage before
        // allowing the first stage to encode any token computation.
        for stage in &self.stages {
            stage.validate_managed_token_work(input.cache_id, input.work)?;
        }
        Ok(())
    }

    fn prepare_pipeline_capacity(
        &mut self,
        inputs: &[ModelTokenInput<'_>],
    ) -> ModelTokenOutcome<()> {
        let requests: Vec<_> = inputs
            .iter()
            .map(|input| KvSlotRequest {
                cache_id: input.cache_id.to_owned(),
                target_len: match input.work {
                    ModelTokenWork::Prefill { tokens, position } => position + tokens.len(),
                    ModelTokenWork::Decode { position, .. } => position as usize + 1,
                },
                admission_target_len: None,
            })
            .collect();
        match self.reserve_token_kv_slots(&requests) {
            Ok(ModelKvReservation::Reserved(_)) => ModelTokenOutcome::Terminal(Ok(())),
            Ok(ModelKvReservation::CapacityDeferred(deferred)) => {
                ModelTokenOutcome::CapacityDeferred(deferred)
            }
            Err(error) => match self.recover_pipeline_stages() {
                ModelExecutionRecovery::Held { .. } => ModelTokenOutcome::Held(error),
                // Preparation has not run token work or advanced its frontier.
                ModelExecutionRecovery::Quiescent { .. } => ModelTokenOutcome::NotSubmitted(error),
            },
        }
    }

    fn observe_pipeline_work<T>(
        &mut self,
        execute: impl FnOnce(&mut Self) -> T,
    ) -> ModelTokenOutcome<T> {
        // Each stage retains its frame before propagating an error; the scoped
        // overlap thread has joined before this catch returns. Catch alone is
        // never terminal evidence: recovery must observe every original stage.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)))
            .map_err(pipeline_panic_error);
        match self.recover_pipeline_stages() {
            ModelExecutionRecovery::Held { error } => {
                ModelTokenOutcome::Held(result.err().unwrap_or(error))
            }
            ModelExecutionRecovery::Quiescent { error } => ModelTokenOutcome::Terminal(
                result.and_then(|product| error.map_or(Ok(product), Err)),
            ),
        }
    }

    pub(super) fn execute_pipeline_token_work(
        &mut self,
        input: &ModelTokenInput<'_>,
    ) -> ModelTokenOutcome<ModelTokenProduct> {
        if let Err(error) = self.validate_pipeline_work(input) {
            return ModelTokenOutcome::NotSubmitted(error);
        }
        match self.prepare_pipeline_capacity(std::slice::from_ref(input)) {
            ModelTokenOutcome::Terminal(Ok(())) => {}
            ModelTokenOutcome::NotSubmitted(error) | ModelTokenOutcome::Terminal(Err(error)) => {
                return ModelTokenOutcome::NotSubmitted(error)
            }
            ModelTokenOutcome::CapacityDeferred(deferred) => {
                return ModelTokenOutcome::CapacityDeferred(deferred)
            }
            ModelTokenOutcome::Held(error) => return ModelTokenOutcome::Held(error),
        }
        self.observe_pipeline_work(|model| {
            let logits = match input.work {
                ModelTokenWork::Prefill { tokens, .. } => model.prefill(input.cache_id, tokens),
                ModelTokenWork::Decode { token, position } => {
                    model.decode(input.cache_id, token, position)
                }
            };
            input
                .sample
                .then_some(ExecutorSamplingOutput::FullLogits(logits))
        })
    }

    pub(super) fn execute_pipeline_token_batch(
        &mut self,
        inputs: &[ModelTokenInput<'_>],
    ) -> ModelTokenBatchOutcome {
        // CPU and mixed/prefill retain the existing per-participant route.
        // Device decode retains the original stage Batch/Overlapped execution.
        let batch_decode = !B::host_execution_is_synchronous()
            && inputs.len() > 1
            && inputs
                .iter()
                .all(|input| matches!(input.work, ModelTokenWork::Decode { .. }));
        if !batch_decode {
            let mut rows = Vec::with_capacity(inputs.len());
            let mut held = false;
            for input in inputs {
                let result = if held {
                    ModelTokenOutcome::NotSubmitted(FerrumError::backend(
                        "earlier participant retains a pipeline stage",
                    ))
                } else {
                    self.execute_token_work(input)
                };
                held |= matches!(result, ModelTokenOutcome::Held(_));
                rows.push(result);
            }
            return ModelTokenBatchOutcome::Serial(rows);
        }
        for input in inputs {
            if let Err(error) = self.validate_pipeline_work(input) {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error));
            }
        }
        match self.prepare_pipeline_capacity(inputs) {
            ModelTokenOutcome::Terminal(Ok(())) => {}
            ModelTokenOutcome::NotSubmitted(error) | ModelTokenOutcome::Terminal(Err(error)) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error))
            }
            ModelTokenOutcome::CapacityDeferred(deferred) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::CapacityDeferred(
                    deferred,
                ))
            }
            ModelTokenOutcome::Held(error) => {
                return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Held(error))
            }
        }
        let batch: Vec<_> = inputs
            .iter()
            .map(|input| {
                let ModelTokenWork::Decode { token, position } = input.work else {
                    unreachable!()
                };
                (input.cache_id.to_owned(), token, position)
            })
            .collect();
        // The existing native head fast path has no mask/penalty input. Full
        // readback preserves downstream processors for these policies.
        let full = inputs.iter().any(|input| {
            !matches!(
                input.logits_policy,
                LogitsReturnPolicy::GreedyArgmax {
                    token_mask: None,
                    repetition_penalty: None
                }
            )
        });
        ModelTokenBatchOutcome::Cohort(self.observe_pipeline_work(|model| {
            let products = if model.pipeline_mode() == LlamaPipelineMode::Overlapped {
                model.decode_batch_overlapped_two_stage(&batch, full)
            } else {
                model.decode_batch_sequential_internal(&batch, full)
            };
            assert_eq!(
                products.len(),
                inputs.len(),
                "pipeline produced a different row count"
            );
            products
                .into_iter()
                .zip(inputs)
                .map(|(product, input)| input.sample.then_some(product))
                .collect()
        }))
    }
}

fn pipeline_panic_error(payload: Box<dyn std::any::Any + Send>) -> FerrumError {
    if let Some(error) = payload.downcast_ref::<FerrumError>() {
        return error.clone();
    }
    let detail = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload");
    FerrumError::backend(format!("pipeline stage operation panicked: {detail}"))
}
