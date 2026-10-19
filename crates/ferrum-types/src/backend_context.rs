//! Immutable backend policy, resolved once before constructing an execution owner.

use crate::{FerrumError, Result, RuntimeConfigSnapshot};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendContextConfig {
    pub device_ordinal: usize,
    pub cuda: CudaBackendConfig,
    pub metal: MetalBackendConfig,
    pub moe: MoeExecutionConfig,
    pub collective: BackendCollectiveConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaBackendConfig {
    pub max_kv: Option<usize>,
    pub graph_skip_upload: bool,
    pub graph_skip_sync: bool,
    /// Kernel capacity hint, independent of the Engine allocator's default.
    pub kv_capacity: usize,
    pub paged_flash_splits: Option<usize>,
    pub split_k_attn: Option<bool>,
    pub fa2_source: bool,
    pub fa2_direct_ffi_shim: Option<String>,
    pub paged_flash: bool,
    pub vllm_paged_attn_v1_short: bool,
    pub triton_int4: bool,
    pub vllm_marlin: bool,
    pub vllm_marlin_sms: i32,
    pub vllm_atomic_add: bool,
    pub vllm_fp32_reduce: bool,
    pub marlin_skip_ws_zero: bool,
}

impl Default for CudaBackendConfig {
    fn default() -> Self {
        Self {
            max_kv: None,
            graph_skip_upload: false,
            graph_skip_sync: false,
            kv_capacity: 512,
            paged_flash_splits: None,
            split_k_attn: None,
            fa2_source: false,
            fa2_direct_ffi_shim: None,
            paged_flash: true,
            vllm_paged_attn_v1_short: true,
            triton_int4: false,
            vllm_marlin: false,
            vllm_marlin_sms: 128,
            vllm_atomic_add: false,
            vllm_fp32_reduce: false,
            marlin_skip_ws_zero: false,
        }
    }
}

impl CudaBackendConfig {
    pub fn shared_kv_for(&self, kv_len: usize) -> usize {
        self.kv_capacity.max(kv_len).max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalBackendConfig {
    pub prefer_f16_weights: bool,
    pub mmap_trace: bool,
    pub force_legacy_attention: bool,
    pub allow_decode_widen: bool,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            prefer_f16_weights: false,
            mmap_trace: false,
            force_legacy_attention: false,
            allow_decode_widen: true,
        }
    }
}

/// The model's expert dispatch and the backend's weight packing read this
/// same value from the actual execution context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeExecutionConfig {
    pub vllm_moe: bool,
    pub vllm_moe_zero_ws: bool,
    pub vllm_moe_pair_ids: bool,
    pub moe_load_trace: bool,
    pub moe_block_size: Option<usize>,
    pub moe_large_m_block_size: Option<usize>,
    pub moe_large_m_min_pairs: usize,
    pub moe_host_route: bool,
    pub moe_streams: usize,
    pub moe_fused: bool,
}

impl Default for MoeExecutionConfig {
    fn default() -> Self {
        Self {
            vllm_moe: false,
            vllm_moe_zero_ws: false,
            vllm_moe_pair_ids: false,
            moe_load_trace: false,
            moe_block_size: None,
            moe_large_m_block_size: None,
            moe_large_m_min_pairs: 1024,
            moe_host_route: false,
            moe_streams: 4,
            moe_fused: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendCollectiveConfig {
    pub world_size: usize,
    pub rank: usize,
}

impl Default for BackendCollectiveConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }
}

impl BackendContextConfig {
    pub fn validate(&self) -> Result<()> {
        if self.cuda.max_kv == Some(0)
            || self.cuda.kv_capacity == 0
            || self.cuda.paged_flash_splits == Some(0)
            || self.cuda.vllm_marlin_sms <= 0
            || self.moe.moe_streams == 0
            || self.collective.world_size == 0
            || self.collective.rank >= self.collective.world_size
        {
            return Err(FerrumError::config(
                "invalid backend capacity, stream count, or collective rank",
            ));
        }
        for size in [self.moe.moe_block_size, self.moe.moe_large_m_block_size]
            .into_iter()
            .flatten()
        {
            if !matches!(size, 8 | 16 | 32 | 48 | 64) {
                return Err(FerrumError::config(
                    "MoE block size must be 8, 16, 32, 48, or 64",
                ));
            }
        }
        if self
            .cuda
            .fa2_direct_ffi_shim
            .as_ref()
            .is_some_and(|path| path.is_empty() || path.contains('\0'))
        {
            return Err(FerrumError::config(
                "FA2 shim path must be nonempty and contain no NUL",
            ));
        }
        Ok(())
    }

    /// Cold assembly only. Backends never read the process or parse a snapshot.
    /// Parse into a candidate so invalid input cannot partly mutate this owner.
    pub fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &RuntimeConfigSnapshot,
    ) -> Result<()> {
        let mut next = self.clone();
        for entry in &snapshot.entries {
            let key = entry.key.as_str();
            let value = entry.effective_value.as_str();
            let numeric = || {
                value
                    .parse::<usize>()
                    .map_err(|_| FerrumError::config(format!("{key} requires an unsigned integer")))
            };
            match key {
                "FERRUM_CUDA_DEVICE" => next.device_ordinal = numeric()?,
                "FERRUM_CUDA_MAX_KV" => next.cuda.max_kv = Some(numeric()?),
                "FERRUM_GRAPH_SKIP_UPLOAD" => next.cuda.graph_skip_upload = value == "1",
                "FERRUM_GRAPH_SKIP_SYNC" => next.cuda.graph_skip_sync = value == "1",
                "FERRUM_KV_CAPACITY" => next.cuda.kv_capacity = numeric()?,
                "FERRUM_PAGED_FLASH_SPLITS" => next.cuda.paged_flash_splits = Some(numeric()?),
                "FERRUM_SPLIT_K_ATTN" => {
                    next.cuda.split_k_attn = match value {
                        "1" => Some(true),
                        "0" => Some(false),
                        _ => None,
                    }
                }
                "FERRUM_FA2_SOURCE" => {
                    next.cuda.fa2_source = matches!(value, "1" | "true" | "TRUE" | "on" | "ON")
                }
                "FERRUM_FA2_DIRECT_FFI_SHIM" => {
                    next.cuda.fa2_direct_ffi_shim = Some(value.to_owned())
                }
                "FERRUM_PAGED_FLASH" => next.cuda.paged_flash = value != "0",
                "FERRUM_VLLM_PAGED_ATTN_V1_SHORT" => {
                    next.cuda.vllm_paged_attn_v1_short = value != "0"
                }
                "FERRUM_TRITON_INT4" => next.cuda.triton_int4 = value == "1",
                "FERRUM_VLLM_MARLIN" => next.cuda.vllm_marlin = value == "1",
                "FERRUM_VLLM_MARLIN_SMS" => {
                    next.cuda.vllm_marlin_sms = value.parse().map_err(|_| {
                        FerrumError::config("FERRUM_VLLM_MARLIN_SMS requires an integer")
                    })?
                }
                "FERRUM_VLLM_ATOMIC_ADD" => next.cuda.vllm_atomic_add = value == "1",
                "FERRUM_VLLM_FP32_REDUCE" => next.cuda.vllm_fp32_reduce = value == "1",
                "FERRUM_MARLIN_SKIP_WS_ZERO" => next.cuda.marlin_skip_ws_zero = value == "1",
                "FERRUM_VLLM_MOE" => next.moe.vllm_moe = value == "1",
                "FERRUM_VLLM_MOE_ZERO_WS" => next.moe.vllm_moe_zero_ws = value == "1",
                "FERRUM_VLLM_MOE_PAIR_IDS" => next.moe.vllm_moe_pair_ids = value == "1",
                "FERRUM_MOE_LOAD_TRACE" => next.moe.moe_load_trace = true,
                "FERRUM_MOE_BLOCK_SIZE" => next.moe.moe_block_size = Some(numeric()?),
                "FERRUM_MOE_LARGE_M_BLOCK_SIZE" => {
                    next.moe.moe_large_m_block_size = Some(numeric()?)
                }
                "FERRUM_MOE_LARGE_M_MIN_PAIRS" => next.moe.moe_large_m_min_pairs = numeric()?,
                "FERRUM_MOE_HOST_ROUTE" => next.moe.moe_host_route = value == "1",
                "FERRUM_MOE_STREAMS" => next.moe.moe_streams = numeric()?.max(1),
                "FERRUM_MOE_FUSED" => next.moe.moe_fused = value != "0",
                "FERRUM_TP" => next.collective.world_size = numeric()?,
                "FERRUM_RANK" => next.collective.rank = numeric()?,
                "FERRUM_METAL_DTYPE" => {
                    next.metal.prefer_f16_weights = value.eq_ignore_ascii_case("f16")
                }
                "FERRUM_MMAP_TRACE" => next.metal.mmap_trace = true,
                "FERRUM_FA_LEGACY" => next.metal.force_legacy_attention = value == "1",
                "FERRUM_FA_DECODE" => next.metal.allow_decode_widen = value != "0",
                _ => {}
            }
        }
        next.validate()?;
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, EngineConfig};

    #[test]
    fn backend_context_defaults_preserve_existing_kernel_policies() {
        let config = BackendContextConfig::default();
        config.validate().unwrap();
        assert_eq!(config.device_ordinal, 0);
        assert_eq!(config.cuda.kv_capacity, 512);
        assert_eq!(config.cuda.max_kv, None);
        assert!(config.cuda.paged_flash && config.cuda.vllm_paged_attn_v1_short);
        assert_eq!(config.cuda.paged_flash_splits, None);
        assert_eq!(config.cuda.split_k_attn, None);
        assert!(!config.cuda.graph_skip_upload && !config.cuda.graph_skip_sync);
        assert!(!config.cuda.triton_int4 && !config.cuda.vllm_marlin);
        assert_eq!(config.cuda.vllm_marlin_sms, 128);
        assert!(!config.cuda.marlin_skip_ws_zero);
        assert!(!config.cuda.vllm_atomic_add && !config.cuda.vllm_fp32_reduce);
        assert!(!config.cuda.fa2_source && config.cuda.fa2_direct_ffi_shim.is_none());
        assert_eq!(config.moe.moe_streams, 4);
        assert!(config.moe.moe_fused);
        assert!(!config.moe.vllm_moe && !config.moe.moe_host_route);
        assert_eq!(config.moe.moe_large_m_min_pairs, 1024);
        assert!(!config.metal.prefer_f16_weights && !config.metal.force_legacy_attention);
        assert!(config.metal.allow_decode_widen);
        assert_eq!(
            config.collective,
            BackendCollectiveConfig {
                world_size: 1,
                rank: 0
            }
        );
        assert_eq!(
            EngineConfig::default()
                .llama_execution_config()
                .backend_context,
            config
        );
    }

    #[test]
    fn backend_context_snapshot_freezes_every_configuration_domain() {
        let mut first = BackendContextConfig::default();
        first
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_CUDA_DEVICE", "2"),
                ("FERRUM_CUDA_MAX_KV", "4096"),
                ("FERRUM_GRAPH_SKIP_UPLOAD", "1"),
                ("FERRUM_GRAPH_SKIP_SYNC", "1"),
                ("FERRUM_KV_CAPACITY", "256"),
                ("FERRUM_PAGED_FLASH_SPLITS", "4"),
                ("FERRUM_SPLIT_K_ATTN", "0"),
                ("FERRUM_FA2_SOURCE", "on"),
                ("FERRUM_FA2_DIRECT_FFI_SHIM", "explicit-shim.so"),
                ("FERRUM_PAGED_FLASH", "0"),
                ("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "0"),
                ("FERRUM_TRITON_INT4", "1"),
                ("FERRUM_VLLM_MARLIN", "1"),
                ("FERRUM_VLLM_MARLIN_SMS", "132"),
                ("FERRUM_VLLM_ATOMIC_ADD", "1"),
                ("FERRUM_VLLM_FP32_REDUCE", "1"),
                ("FERRUM_MARLIN_SKIP_WS_ZERO", "1"),
                ("FERRUM_VLLM_MOE", "1"),
                ("FERRUM_VLLM_MOE_ZERO_WS", "1"),
                ("FERRUM_VLLM_MOE_PAIR_IDS", "1"),
                ("FERRUM_MOE_LOAD_TRACE", "1"),
                ("FERRUM_MOE_BLOCK_SIZE", "16"),
                ("FERRUM_MOE_LARGE_M_BLOCK_SIZE", "32"),
                ("FERRUM_MOE_LARGE_M_MIN_PAIRS", "513"),
                ("FERRUM_MOE_HOST_ROUTE", "1"),
                ("FERRUM_MOE_STREAMS", "2"),
                ("FERRUM_MOE_FUSED", "0"),
                ("FERRUM_METAL_DTYPE", "f16"),
                ("FERRUM_MMAP_TRACE", "1"),
                ("FERRUM_FA_LEGACY", "1"),
                ("FERRUM_FA_DECODE", "0"),
                ("FERRUM_TP", "3"),
                ("FERRUM_RANK", "1"),
            ]))
            .unwrap();
        let frozen = first.clone();
        let mut peer = BackendContextConfig::default();
        peer.apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
            ("FERRUM_CUDA_DEVICE", "0"),
            ("FERRUM_MOE_STREAMS", "0"),
            ("FERRUM_METAL_DTYPE", "f32"),
            ("FERRUM_VLLM_MOE", "0"),
        ]))
        .unwrap();
        assert_eq!(peer.moe.moe_streams, 1);
        assert_eq!(first, frozen);
        assert_eq!(first.device_ordinal, 2);
        assert_eq!(first.cuda.max_kv, Some(4096));
        assert_eq!(first.cuda.kv_capacity, 256);
        assert_eq!(first.cuda.paged_flash_splits, Some(4));
        assert_eq!(first.cuda.split_k_attn, Some(false));
        assert!(first.cuda.graph_skip_upload && first.cuda.graph_skip_sync);
        assert!(first.cuda.fa2_source && first.cuda.fa2_direct_ffi_shim.is_some());
        assert!(!first.cuda.paged_flash && !first.cuda.vllm_paged_attn_v1_short);
        assert!(first.cuda.triton_int4 && first.cuda.vllm_marlin);
        assert_eq!(first.cuda.vllm_marlin_sms, 132);
        assert!(
            first.cuda.vllm_atomic_add
                && first.cuda.vllm_fp32_reduce
                && first.cuda.marlin_skip_ws_zero
        );
        assert!(
            first.moe.vllm_moe
                && first.moe.vllm_moe_zero_ws
                && first.moe.vllm_moe_pair_ids
                && first.moe.moe_load_trace
        );
        assert_eq!(first.moe.moe_block_size, Some(16));
        assert_eq!(first.moe.moe_large_m_block_size, Some(32));
        assert_eq!(first.moe.moe_large_m_min_pairs, 513);
        assert!(first.moe.moe_host_route && !first.moe.moe_fused);
        assert_eq!(first.moe.moe_streams, 2);
        assert!(
            first.metal.prefer_f16_weights
                && first.metal.mmap_trace
                && first.metal.force_legacy_attention
        );
        assert!(!first.metal.allow_decode_widen);
        assert_eq!(
            first.collective,
            BackendCollectiveConfig {
                world_size: 3,
                rank: 1
            }
        );
        let roundtrip: BackendContextConfig =
            serde_json::from_value(serde_json::to_value(&first).unwrap()).unwrap();
        assert_eq!(roundtrip, first);
    }

    #[test]
    fn backend_context_invalid_snapshot_is_rejected_without_partial_mutation() {
        let mut config = BackendContextConfig::default();
        for (key, value) in [
            ("FERRUM_KV_CAPACITY", "0"),
            ("FERRUM_PAGED_FLASH_SPLITS", "0"),
            ("FERRUM_CUDA_DEVICE", "invalid"),
            ("FERRUM_MOE_BLOCK_SIZE", "7"),
            ("FERRUM_RANK", "1"),
            ("FERRUM_FA2_DIRECT_FFI_SHIM", ""),
        ] {
            assert!(
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                        ("FERRUM_GRAPH_SKIP_SYNC", "1"),
                        (key, value),
                    ]))
                    .is_err(),
                "{key}"
            );
            assert_eq!(config, BackendContextConfig::default());
        }
    }

    #[test]
    fn backend_context_product_projection_has_explicit_placement_and_legacy_precedence() {
        let mut engine = EngineConfig::default();
        engine.backend.device = Device::CUDA(3);
        engine.runtime.backend_context.device_ordinal = 7;
        engine.runtime.backend_context.cuda.kv_capacity = 8192;
        engine.runtime.backend_context.metal.prefer_f16_weights = true;
        engine.runtime.backend_context.collective.world_size = 4;
        engine.runtime.metal_dtype = Some("f32".into());
        engine.runtime.tp = Some(1);
        engine.runtime.kv_capacity = Some(64);
        engine.runtime.llama.split_k_attn = Some(false);
        let actual = engine.llama_execution_config().backend_context;
        assert_eq!(actual.device_ordinal, 3);
        assert_eq!(actual.cuda.kv_capacity, 64);
        assert_eq!(actual.cuda.split_k_attn, Some(false));
        assert!(!actual.metal.prefer_f16_weights);
        assert_eq!(actual.collective.world_size, 1);
        engine
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_CUDA_DEVICE",
                "2",
            )]))
            .unwrap();
        assert_eq!(engine.backend.device, Device::CUDA(2));
        assert_eq!(
            engine
                .llama_execution_config()
                .backend_context
                .device_ordinal,
            2
        );
        for device in [
            Device::CPU,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal,
        ] {
            engine.backend.device = device.clone();
            engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    "FERRUM_CUDA_DEVICE",
                    "2",
                )]))
                .unwrap();
            assert_eq!(engine.backend.device, device);
            assert_eq!(
                engine
                    .llama_execution_config()
                    .backend_context
                    .device_ordinal,
                0
            );
        }

        let mut engine = EngineConfig::default();
        engine.runtime.tp = Some(4);
        // The raw backend value still defaults to one rank. Cold validation
        // must use the explicit Engine policy before applying this rank.
        assert_eq!(engine.runtime.backend_context.collective.world_size, 1);
        engine
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_RANK",
                "2",
            )]))
            .unwrap();
        assert_eq!(
            engine.runtime.backend_context.collective,
            BackendCollectiveConfig {
                world_size: 4,
                rank: 2
            }
        );
        assert_eq!(
            engine.llama_execution_config().backend_context.collective,
            BackendCollectiveConfig {
                world_size: 4,
                rank: 2
            }
        );
        // An explicit snapshot TP replaces the prior Engine setting while a
        // missing rank preserves the previously selected rank.
        engine
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_TP",
                "6",
            )]))
            .unwrap();
        assert_eq!(engine.runtime.tp, Some(6));
        assert_eq!(
            engine.runtime.backend_context.collective,
            BackendCollectiveConfig {
                world_size: 6,
                rank: 2
            }
        );
        assert_eq!(
            engine.llama_execution_config().backend_context.collective,
            BackendCollectiveConfig {
                world_size: 6,
                rank: 2
            }
        );
    }
}
