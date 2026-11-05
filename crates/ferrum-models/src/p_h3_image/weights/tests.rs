use super::*;
use crate::p_h3_image::test_fixture::tiny_transformer;
use serde_json::json;

fn limits() -> PH3ImageWeightLimits {
    PH3ImageWeightLimits {
        max_tensors: 1000,
        max_payload_bytes: 70_000_000_000,
    }
}

fn released_config() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"MiniMaxH3Transformer3DModel",
            "attention_head_dim":128,"audio_in_channels":32,"ffn_dim":14336,
            "final_norm_eps":1e-5,"freq_dim":256,"hidden_size":5376,
            "in_channels":24,"norm_eps":1e-5,"num_attention_heads":56,
            "num_layers":50,"num_refiner_layers":2,"patch_size":[1,2,2],
            "qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.0,
            "text_dim":5120,"time_embed_dim":2688,"time_embed_hidden_dim":5376
        }))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn pinned_checkpoint_schema_matches_published_payload_and_mixed_precision() {
    let config = released_config();
    let weights = PH3ImageTransformerWeights::from_config(&config, limits()).unwrap();
    // Independently published total_size in the pinned Hugging Face shard
    // index; this checks all dense shapes and F32/BF16 storage accounting.
    assert_eq!(weights.payload_bytes(), 66_280_430_080);
    assert_eq!(
        weights.tensor("proj_in.weight").unwrap().dimensions(),
        &[5376, 96]
    );
    assert_eq!(
        weights.tensor("proj_in.weight").unwrap().element_type(),
        ElementType::F32
    );
    assert_eq!(
        weights
            .tensor("time_embedder.linear_2.weight")
            .unwrap()
            .element_type(),
        ElementType::F32
    );
    let adaln = weights
        .tensor("transformer_blocks.0.adaln_proj.linear.weight")
        .unwrap();
    assert_eq!(adaln.dimensions(), &[96768, 2688]);
    assert_eq!(adaln.element_type(), ElementType::Bf16);
    assert_eq!(
        weights
            .tensor("transformer_blocks.0.attn.to_q.weight")
            .unwrap()
            .dimensions(),
        &[7168, 5376]
    );
    assert_eq!(
        weights
            .tensor("transformer_blocks.0.ff.net.0.proj.weight")
            .unwrap()
            .dimensions(),
        &[28672, 5376]
    );
    assert!(weights.tensor("rope.inv_freq").is_none());
    assert!(weights
        .tensor("transformer_blocks.0.attn.to_q.bias")
        .is_none());
}

#[test]
fn original_tensor_header_dtype_and_shape_must_match_the_declared_route() {
    let weights =
        PH3ImageTransformerWeights::from_config(&tiny_transformer(8, 1), limits()).unwrap();
    let tensor = weights.tensor("audio_proj_out.weight").unwrap();
    tensor.validate_header(&[2, 8], ElementType::F32).unwrap();
    assert!(tensor.validate_header(&[2, 8], ElementType::Bf16).is_err());
    assert!(tensor.validate_header(&[8, 2], ElementType::F32).is_err());
    assert!(weights
        .tensor("token_refiner.refiner_blocks.0.adaln_proj.linear.weight")
        .is_none());
}

#[test]
fn tensor_and_payload_capacities_bound_the_metadata_inventory() {
    let config = tiny_transformer(8, 1);
    let weights = PH3ImageTransformerWeights::from_config(&config, limits()).unwrap();
    let capacity = PH3ImageWeightLimits {
        max_tensors: weights.tensors().len(),
        max_payload_bytes: weights.payload_bytes(),
    };
    assert!(PH3ImageTransformerWeights::from_config(&config, capacity).is_ok());
    assert!(PH3ImageTransformerWeights::from_config(
        &config,
        PH3ImageWeightLimits {
            max_tensors: capacity.max_tensors - 1,
            ..capacity
        }
    )
    .is_err());
    assert!(PH3ImageTransformerWeights::from_config(
        &config,
        PH3ImageWeightLimits {
            max_payload_bytes: capacity.max_payload_bytes - 1,
            ..capacity
        }
    )
    .is_err());
}

#[test]
fn real_comfy_curve_header_structure_keeps_audio_and_f32_modulation_without_dense_time_mlp() {
    let config = released_config();
    let curve = PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap();
    let weights =
        PH3ImageTransformerWeights::from_comfy_curve_config(&config, curve, limits()).unwrap();
    // Last data offset in the real BF16 header at Comfy revision
    // 44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543, independently read with HTTP Range.
    assert_eq!(weights.payload_bytes(), 40_468_355_200);
    assert_eq!(
        weights.checkpoint_layout(),
        PH3ImageTransformerCheckpointLayout::ComfyCurve
    );
    assert_eq!(
        weights.tensor("adaln_t_table").unwrap().dimensions(),
        &[4097, 16]
    );
    let adaln = weights.tensor("blocks.0.adaln_proj.linear.weight").unwrap();
    assert_eq!(adaln.dimensions(), &[96768, 16]);
    assert_eq!(adaln.element_type(), ElementType::F32);
    assert_eq!(
        weights
            .tensor("final_layer.adaln_proj.linear.weight")
            .unwrap()
            .dimensions(),
        &[10752, 16]
    );
    assert_eq!(
        weights
            .tensor("blocks.0.attn.qkv_proj.weight")
            .unwrap()
            .dimensions(),
        &[21504, 5376]
    );
    assert!(weights.tensor("audio_patch_proj.weight").is_some());
    assert!(weights.tensor("final_layer.audio_out.weight").is_some());
    assert!(weights.tensor("rope.inv_freq").is_some());
    assert!(weights.tensor("time_embedder.linear_1.weight").is_none());
    assert!(weights.tensor("blocks.0.attn.to_q.weight").is_none());
}

#[test]
fn curve_width_is_explicit_and_capacity_does_not_fall_back_to_dense_schema() {
    let config = tiny_transformer(8, 1);
    let curve = PH3ImageCurveTimeEmbedding::new(5, 7).unwrap();
    let weights =
        PH3ImageTransformerWeights::from_comfy_curve_config(&config, curve, limits()).unwrap();
    assert_eq!(
        weights
            .tensor("blocks.0.adaln_proj.linear.weight")
            .unwrap()
            .dimensions(),
        &[144, 7]
    );
    let capacity = PH3ImageWeightLimits {
        max_tensors: weights.tensors().len(),
        max_payload_bytes: weights.payload_bytes(),
    };
    assert!(PH3ImageTransformerWeights::from_comfy_curve_config(&config, curve, capacity).is_ok());
    assert!(PH3ImageTransformerWeights::from_comfy_curve_config(
        &config,
        curve,
        PH3ImageWeightLimits {
            max_tensors: capacity.max_tensors - 1,
            ..capacity
        }
    )
    .is_err());
    assert!(PH3ImageTransformerWeights::from_comfy_curve_config(
        &config,
        curve,
        PH3ImageWeightLimits {
            max_payload_bytes: capacity.max_payload_bytes - 1,
            ..capacity
        }
    )
    .is_err());
}
