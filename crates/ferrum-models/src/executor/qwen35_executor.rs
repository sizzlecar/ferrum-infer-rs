//! Qwen3.5 / Qwen3.6 W3 executor boundary.
//!
//! This skeleton is intentionally not registered as a product executor yet. It
//! exposes the recurrent-state allocation contract that the real W3 executor
//! will use, while prefill/decode remain explicitly unsupported.

use async_trait::async_trait;
use ferrum_interfaces::{
    model_executor::{
        DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage, ExecutorState,
        ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    ModelExecutor, RecurrentStateSpec,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, RequestId, Result, TokenId};
use std::path::Path;

use crate::{
    definition::ModelDefinition,
    qwen35_config::Qwen35TextConfig,
    qwen35_weights::{Qwen35ResolvedWeightPlan, Qwen35WeightInventory, Qwen35WeightValidation},
};

const UNSUPPORTED_EXECUTION_MESSAGE: &str = "Qwen3.5/Qwen3.6 W3 executor exposes recurrent-state \
spec only; prefill/decode are not wired yet";

#[derive(Debug, Clone)]
pub struct Qwen35W3Executor {
    info: ModelInfo,
    config: Qwen35TextConfig,
    dtype: DataType,
    device: Device,
    weight_validation: Option<Qwen35WeightValidation>,
    weight_plan: Option<Qwen35ResolvedWeightPlan>,
}

impl Qwen35W3Executor {
    pub fn from_definition(
        model_id: impl Into<String>,
        def: &ModelDefinition,
        dtype: DataType,
        device: Device,
    ) -> Result<Self> {
        let config = Qwen35TextConfig::from_model_definition(def)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5/Qwen3.6 config: {err}")))?;
        let mut info = def.to_model_info(model_id);
        info.dtype = dtype;
        info.device = device.clone();
        Ok(Self {
            info,
            config,
            dtype,
            device,
            weight_validation: None,
            weight_plan: None,
        })
    }

    pub fn from_definition_with_weight_preflight(
        model_id: impl Into<String>,
        def: &ModelDefinition,
        model_dir: &Path,
        dtype: DataType,
        device: Device,
    ) -> Result<Self> {
        let mut executor = Self::from_definition(model_id, def, dtype, device)?;
        let inventory = Qwen35WeightInventory::from_safetensors_dir(model_dir)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight inventory failed: {err}")))?;
        let weight_plan = inventory
            .detect_prefix_and_resolve(&executor.config)
            .map_err(|err| FerrumError::model(format!("Qwen3.5 weight preflight failed: {err}")))?;
        let validation = weight_plan.validation();
        executor.weight_validation = Some(validation);
        executor.weight_plan = Some(weight_plan);
        Ok(executor)
    }

    pub fn qwen35_config(&self) -> &Qwen35TextConfig {
        &self.config
    }

    pub fn weight_validation(&self) -> Option<&Qwen35WeightValidation> {
        self.weight_validation.as_ref()
    }

    pub fn weight_plan(&self) -> Option<&Qwen35ResolvedWeightPlan> {
        self.weight_plan.as_ref()
    }

    fn unsupported_execution() -> FerrumError {
        FerrumError::unsupported(UNSUPPORTED_EXECUTION_MESSAGE)
    }
}

#[async_trait]
impl ModelExecutor for Qwen35W3Executor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        self.config
            .to_recurrent_state_spec(request_id.clone(), self.dtype, self.device.clone(), 1)
            .map(Some)
            .map_err(|err| FerrumError::model(format!("invalid Qwen3.5 recurrent spec: {err}")))
    }

    async fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        Err(Self::unsupported_execution())
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(Self::unsupported_execution())
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let recurrent_state_bytes = self
            .config
            .recurrent_state_elements_per_slot()
            .unwrap_or(0)
            .saturating_mul(self.dtype.size_bytes());
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: Vec::new(),
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![self.dtype],
            supported_devices: vec![self.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: self.info.num_parameters.saturating_mul(2),
                activation_memory_per_token: self.info.hidden_size,
                kv_cache_memory_per_token: 0,
                overhead_memory: recurrent_state_bytes as u64,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ExecutorState::Error,
            is_ready: false,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: None,
        }
    }
}
