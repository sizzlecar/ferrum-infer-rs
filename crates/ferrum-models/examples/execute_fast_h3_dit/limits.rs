use super::config::Config;
use ferrum_models::fast_h3::{vsa::*, *};
pub(super) fn model_limits(a: &Config, p: &FastH3TransformerPackage) -> FastH3DitBackboneLimits {
    let rows = a.maximum_rows as u64;
    let metadata = a.maximum_metadata_bytes;
    let bytes = a.device_bytes;
    let columns =
        (p.config()
            .hidden_size()
            .max(p.config().attention_heads() * p.config().attention_head_dim())) as u64;
    let transport = FastH3VsaRowTransportLimits {
        maximum_rows: rows,
        maximum_columns: columns,
        maximum_batches: 1,
        maximum_map_bytes: a.host_bytes as u64,
        maximum_metadata_bytes: metadata,
    };
    let pooling = FastH3VsaPoolLimits {
        maximum_batches: 1,
        maximum_tiles: a.maximum_tiles as u64,
        maximum_columns: columns,
        maximum_count_bytes: a.host_bytes as u64,
        maximum_metadata_bytes: metadata,
    };
    FastH3DitBackboneLimits {
        maximum_layers: a.maximum_layers,
        maximum_plan_nodes: a.maximum_plan_nodes,
        maximum_stored_bytes: bytes,
        maximum_metadata_bytes: metadata,
        block: FastH3DitBlockLimits {
            modulation: FastH3ModulationLimits {
                maximum_rows: rows,
                maximum_time_rows: a.maximum_steps * 4,
                maximum_metadata_bytes: metadata,
                maximum_stored_bytes: bytes,
            },
            attention_input: FastH3AttentionInputLimits {
                projection: FastH3AttentionProjectionLimits {
                    maximum_rows: rows,
                    maximum_metadata_bytes: metadata,
                    maximum_stored_bytes: bytes,
                },
                qk: FastH3QkNormRopeLimits {
                    maximum_rows: rows,
                    maximum_metadata_bytes: metadata,
                    maximum_stored_bytes: bytes,
                },
                transport,
                pooling,
                maximum_plan_nodes: a.maximum_plan_nodes,
                maximum_stored_bytes: bytes,
                maximum_metadata_bytes: metadata,
            },
            attention_core: FastH3AttentionCoreLimits {
                prefix: FastH3PrefixAttentionLimits {
                    center: FastH3KeyCenterLimits {
                        maximum_batches: 1,
                        maximum_rows: rows,
                        maximum_columns: columns,
                        maximum_tensor_bytes: bytes,
                        maximum_metadata_bytes: metadata,
                    },
                    transport,
                    maximum_plan_nodes: a.maximum_plan_nodes,
                    maximum_stored_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                quantize: FastH3VsaQkvQuantizeLimits {
                    maximum_batches: 1,
                    maximum_heads: p.config().attention_heads() as u64,
                    maximum_rows: rows,
                    maximum_input_bytes: bytes,
                    maximum_output_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                pooled: FastH3PooledAttentionLimits {
                    maximum_batches: 1,
                    maximum_heads: p.config().attention_heads() as u64,
                    maximum_tiles: a.maximum_tiles as u64,
                    maximum_head_dim: p.config().attention_head_dim() as u64,
                    maximum_score_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                pooling,
                transport,
                head_pack: FastH3HeadAxisPackLimits {
                    maximum_batches: 1,
                    maximum_rows: rows,
                    maximum_columns: columns,
                    maximum_input_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                projection: FastH3AttentionOutputLimits {
                    maximum_rows: rows,
                    maximum_metadata_bytes: metadata,
                    maximum_stored_bytes: bytes,
                },
                maximum_region_bytes: a.host_bytes as u64,
                maximum_plan_nodes: a.maximum_plan_nodes,
                maximum_stored_bytes: bytes,
                maximum_metadata_bytes: metadata,
            },
            ffn: FastH3FfnLimits {
                maximum_rows: rows,
                maximum_plan_nodes: a.maximum_plan_nodes,
                maximum_metadata_bytes: metadata,
                maximum_stored_bytes: bytes,
            },
            maximum_plan_nodes: a.maximum_plan_nodes,
            maximum_stored_bytes: bytes,
            maximum_metadata_bytes: metadata,
        },
    }
}
