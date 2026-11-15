use super::super::denoising::{
    DenoisingResidencyPolicy, FeedForwardPartitionPolicy, PredictionHandoffPolicy,
};
use super::{conditioning::ConditioningLimits, decode::FastH3DecodingLimits};
use crate::video_generation::fast_h3::assets::FastH3AssetLimits;
use ferrum_models::fast_h3::{vsa::*, *};
use ferrum_models::spatial_tiles::SpatialTileBatch;
use ferrum_models::{
    fast_h3::{audio_vae::FastH3AudioDecoderLimits, video_vae::*},
    qwen3_vl_vision::Qwen3VLVisionProcessorLimits,
    qwen_image21::image_input::ImageInputLimits,
};

/// Explicit admission/compilation limits for the ordered-image B1/B2
/// path. These are capacities, not claims that every admitted shape fits GPU.
#[derive(Clone, Copy, Debug)]
pub struct FastH3PipelineLimits {
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub maximum_frames: u32,
    pub maximum_rows: usize,
    pub maximum_tiles: usize,
    pub maximum_layers: usize,
    pub maximum_steps: usize,
    pub maximum_plan_nodes: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_auxiliary_bytes: u64,
    pub request_input_bytes: usize,
    pub request_weight_bytes: usize,
    pub request_transfer_bytes: usize,
    pub request_output_floor_bytes: usize,
    pub conditioning: ConditioningLimits,
    pub video_decoder_tile_batch: SpatialTileBatch,
    pub video_postprocessing: super::super::decode::FastH3VideoPostprocessing,
    pub denoiser_residency: DenoisingResidencyPolicy,
    pub prediction_handoff: PredictionHandoffPolicy,
    pub feed_forward_partition: FeedForwardPartitionPolicy,
    /// Upper bound when row partitioning is selected; no reduction axis is split.
    pub feed_forward_chunk_rows: std::num::NonZeroU64,
}
impl FastH3PipelineLimits {
    pub fn rtx5090() -> Self {
        Self {
            maximum_width: 1920,
            maximum_height: 1088,
            // Fifteen seconds at 24 FPS occupies 362 frames after the model's
            // 5 + 17*n alignment. Per-request plans still admit actual memory.
            maximum_frames: 362,
            maximum_rows: 262_144,
            maximum_tiles: 4096,
            maximum_layers: 50,
            maximum_steps: 32,
            maximum_plan_nodes: 16384,
            maximum_metadata_bytes: 64 * 1024 * 1024,
            maximum_stored_bytes: 32 * 1024 * 1024 * 1024,
            maximum_auxiliary_bytes: 4 * 1024 * 1024 * 1024,
            request_input_bytes: 512 * 1024 * 1024,
            request_weight_bytes: 128 * 1024 * 1024,
            request_transfer_bytes: 64 * 1024 * 1024,
            request_output_floor_bytes: 384 * 1024 * 1024,
            conditioning: ConditioningLimits::rtx5090(),
            video_decoder_tile_batch: SpatialTileBatch::default(),
            video_postprocessing: super::super::decode::FastH3VideoPostprocessing::default(),
            denoiser_residency: DenoisingResidencyPolicy::default(),
            prediction_handoff: PredictionHandoffPolicy::default(),
            feed_forward_chunk_rows: std::num::NonZeroU64::new(4096).unwrap(),
            feed_forward_partition: FeedForwardPartitionPolicy::default(),
        }
    }
    pub fn asset_limits(&self) -> FastH3AssetLimits {
        FastH3AssetLimits {
            metadata: FastH3MetadataLimits {
                max_file_bytes: self.maximum_metadata_bytes,
                max_pdd_steps: self.maximum_steps,
            },
            maximum_tokenizer_bytes: 32 * 1024 * 1024,
            maximum_shards_per_archive: 128,
            maximum_tensors_per_archive: 32_768,
            maximum_total_source_bytes: 96 * 1024 * 1024 * 1024,
            maximum_exported_text_layers: 64,
            transformer: FastH3WeightLimits {
                maximum_tensors: 32_768,
                maximum_source_bytes: 64 * 1024 * 1024 * 1024,
                maximum_skip_metadata_bytes: self.maximum_metadata_bytes,
            },
            video_decoder: FastH3VideoDecoderWeightLimits {
                maximum_tensors: 32_768,
                maximum_source_bytes: self.maximum_stored_bytes as usize,
                maximum_metadata_bytes: self.maximum_metadata_bytes,
            },
            processor: processor_limits(self.maximum_metadata_bytes),
        }
    }
    pub(super) fn input(&self) -> FastH3InputLimits {
        FastH3InputLimits {
            maximum_rows: self.maximum_rows as u64,
            maximum_segments: 32,
            maximum_metadata_bytes: self.maximum_metadata_bytes,
            maximum_stored_bytes: self.maximum_stored_bytes,
        }
    }
    pub(super) fn denoising(&self, package: &FastH3TransformerPackage) -> FastH3DenoisingLimits {
        FastH3DenoisingLimits {
            input: self.input(),
            backbone: backbone_limits(self, package),
            maximum_plan_nodes: self.maximum_plan_nodes,
            maximum_stored_bytes: self.maximum_stored_bytes,
            maximum_metadata_bytes: self.maximum_metadata_bytes,
        }
    }
    pub(super) fn output_conditioning(&self) -> FastH3OutputConditioningLimits {
        FastH3OutputConditioningLimits {
            maximum_rows: self.maximum_rows as u64,
            maximum_time_rows: self.maximum_steps * 4,
            maximum_metadata_bytes: self.maximum_metadata_bytes,
            maximum_stored_bytes: self.maximum_stored_bytes,
        }
    }
    pub(super) fn final_output(&self) -> FastH3FinalOutputLimits {
        FastH3FinalOutputLimits {
            conditioning: self.output_conditioning(),
            projection: FastH3PddProjectionLimits {
                maximum_rows: self.maximum_rows as u64,
                maximum_metadata_bytes: self.maximum_metadata_bytes,
                maximum_source_bytes: self.maximum_stored_bytes,
                maximum_fused_bytes: self.maximum_auxiliary_bytes,
            },
            maximum_metadata_bytes: self.maximum_metadata_bytes,
            maximum_plan_nodes: self.maximum_plan_nodes,
            maximum_stored_bytes: self.maximum_stored_bytes,
            maximum_pdd_steps: self.maximum_steps,
        }
    }
    pub(super) fn time(&self) -> FastH3TimeConditioningLimits {
        FastH3TimeConditioningLimits {
            maximum_time_rows: (self.maximum_steps * 4) as u64,
            maximum_metadata_bytes: self.maximum_metadata_bytes,
            maximum_stored_bytes: self.maximum_stored_bytes,
        }
    }
    pub(super) fn decoding(&self) -> FastH3DecodingLimits {
        FastH3DecodingLimits {
            video: FastH3VideoClipLimits {
                blocks: FastH3VideoDecoderBlocksLimits {
                    maximum_rows: self.maximum_rows as u64,
                    maximum_blocks: 36,
                    maximum_stored_bytes: self.maximum_stored_bytes,
                    maximum_metadata_bytes: self.maximum_metadata_bytes,
                },
                maximum_plan_nodes: self.maximum_plan_nodes,
                maximum_activation_bytes: self.maximum_stored_bytes,
            },
            audio: FastH3AudioDecoderLimits {
                maximum_metadata_bytes: self.maximum_metadata_bytes,
                maximum_source_bytes: self.maximum_stored_bytes,
                maximum_stored_bytes: self.maximum_stored_bytes,
                maximum_plan_nodes: self.maximum_plan_nodes,
                maximum_activation_bytes: self.maximum_stored_bytes,
            },
            temporal: FastH3VideoTemporalLimits {
                maximum_output_frames: self.maximum_frames as usize,
                maximum_buffer_bytes: self.maximum_auxiliary_bytes as usize,
            },
            maximum_spatial_tiles: 1024,
            tile_batch: self.video_decoder_tile_batch,
            postprocessing: self.video_postprocessing,
        }
    }
}
fn processor_limits(metadata: usize) -> Qwen3VLVisionProcessorLimits {
    Qwen3VLVisionProcessorLimits {
        maximum_metadata_bytes: metadata,
        image: ImageInputLimits::default(),
        maximum_patch_rows: 65_536,
        maximum_patch_bytes: 512 * 1024 * 1024,
    }
}
fn backbone_limits(
    a: &FastH3PipelineLimits,
    p: &FastH3TransformerPackage,
) -> FastH3DitBackboneLimits {
    let rows = a.maximum_rows as u64;
    let metadata = a.maximum_metadata_bytes;
    let bytes = a.maximum_stored_bytes;
    let columns =
        (p.config()
            .hidden_size()
            .max(p.config().attention_heads() * p.config().attention_head_dim())) as u64;
    let transport = FastH3VsaRowTransportLimits {
        maximum_rows: rows,
        maximum_columns: columns,
        maximum_batches: 2,
        maximum_map_bytes: a.maximum_auxiliary_bytes,
        maximum_metadata_bytes: metadata,
    };
    let pooling = FastH3VsaPoolLimits {
        maximum_batches: 2,
        maximum_tiles: a.maximum_tiles as u64,
        maximum_columns: columns,
        maximum_count_bytes: a.maximum_auxiliary_bytes,
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
                        maximum_batches: 2,
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
                    maximum_batches: 2,
                    maximum_heads: p.config().attention_heads() as u64,
                    maximum_rows: rows,
                    maximum_input_bytes: bytes,
                    maximum_output_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                pooled: FastH3PooledAttentionLimits {
                    maximum_batches: 2,
                    maximum_heads: p.config().attention_heads() as u64,
                    maximum_tiles: a.maximum_tiles as u64,
                    maximum_head_dim: p.config().attention_head_dim() as u64,
                    maximum_score_bytes: bytes,
                    maximum_metadata_bytes: metadata,
                },
                pooling,
                transport,
                head_pack: FastH3HeadAxisPackLimits {
                    maximum_batches: 2,
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
                maximum_region_bytes: a.maximum_auxiliary_bytes,
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
