//! Actual uploads and numerical dispatch through distinct original policies.

use super::*;
use crate::attention::metal::pipelines::MetalAttentionKernel;

#[test]
fn metal_context_rejects_unsupported_device_before_resource_creation() {
    let config = BackendContextConfig {
        device_ordinal: 1,
        ..Default::default()
    };
    assert!(MetalBackend::new_context(&config).is_err());
}

fn weight_gemv(ctx: &mut MetalContext, weight: &MetalBuf, host: &[f32], half: bool) {
    const WIDTH: usize = 1024;
    let mut input = vec![0.0; WIDTH];
    input[3] = 0.5;
    input[517] = -2.0;
    let input = MetalBackend::from_slice_in_context(ctx, &input);
    let mut output = MetalBackend::alloc_in_context(ctx, WIDTH);
    MetalBackend::begin_execution(ctx).unwrap();
    MetalBackend::gemm(ctx, &input, weight, &mut output, 1, WIDTH, WIDTH);
    MetalBackend::sync_checked(ctx).unwrap();
    let actual = MetalBackend::to_vec_in_context(ctx, &output, WIDTH);
    let stored = |x: f32| if half { f16::from_f32(x).to_f32() } else { x };
    for (row, value) in actual.iter().enumerate() {
        let expected = 0.5 * stored(host[row * WIDTH + 3]) - 2.0 * stored(host[row * WIDTH + 517]);
        assert!(value.is_finite());
        assert!(
            (value - expected).abs() < 1.0e-5,
            "row={row}: {value} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires actual Metal uploads and F16/F32 weight GEMV"]
fn metal_instance_weight_policy_is_frozen_across_interleaved_loads_and_peer_retirement() {
    let mut half_config = BackendContextConfig::default();
    half_config.metal.prefer_f16_weights = true;
    let mut half_context = MetalBackend::new_context(&half_config).unwrap();
    let mut full_context = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    // Mutating the caller's assembly value cannot change an existing owner.
    half_config.metal.prefer_f16_weights = false;
    let host: Vec<_> = (0..F16_MIN_ELEMS)
        .map(|i| ((i % 997) as f32 - 498.0) * 0.001)
        .collect();
    let bytes: Vec<_> = host.iter().flat_map(|x| x.to_le_bytes()).collect();
    let threshold_minus_one = &bytes[..(F16_MIN_ELEMS - 1) * 4];
    let small = MetalBackend::from_weight_bytes_in_context(
        &mut half_context,
        threshold_minus_one,
        SrcDtype::F32,
    );
    assert_eq!(small.dtype(), Dtype::F32);
    assert_eq!(
        MetalBackend::to_vec(&small, small.len()),
        host[..F16_MIN_ELEMS - 1]
    );
    let half_weight =
        MetalBackend::from_weight_bytes_in_context(&mut half_context, &bytes, SrcDtype::F32);
    let full_weight =
        MetalBackend::from_weight_bytes_in_context(&mut full_context, &bytes, SrcDtype::F32);
    assert_eq!(half_weight.dtype(), Dtype::F16);
    assert_eq!(full_weight.dtype(), Dtype::F32);
    let rounded: Vec<_> = host.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
    assert!(host.iter().zip(&rounded).any(|(a, b)| a != b));
    assert_eq!(
        MetalBackend::to_vec(&half_weight, half_weight.len()),
        rounded
    );
    assert_eq!(MetalBackend::to_vec(&full_weight, full_weight.len()), host);
    weight_gemv(&mut half_context, &half_weight, &host, true);
    weight_gemv(&mut full_context, &full_weight, &host, false);
    MetalBackend::retire_context(&mut full_context).unwrap();
    drop(full_context);
    drop(full_weight);
    // Load again after the other original owner retires, then execute again.
    let second_half =
        MetalBackend::from_weight_bytes_in_context(&mut half_context, &bytes, SrcDtype::F32);
    assert_eq!(second_half.dtype(), Dtype::F16);
    weight_gemv(&mut half_context, &second_half, &host, true);
    MetalBackend::retire_context(&mut half_context).unwrap();
}

fn attention(
    ctx: &mut MetalContext,
    q_len: usize,
    kv_len: usize,
    expected_kernel: MetalAttentionKernel,
) {
    let params = AttentionParams {
        batch: 2,
        num_heads: 4,
        num_kv_heads: 2,
        q_len,
        kv_len,
        head_dim: 128,
        causal: true,
        pos_offset: kv_len - q_len,
        sliding_window: 0,
    };
    let query: Vec<_> = (0..params.batch * params.num_heads * q_len * params.head_dim)
        .map(|i| ((i * 7 % 41) as f32 - 20.0) / 32.0)
        .collect();
    let key: Vec<_> = (0..params.batch * params.num_kv_heads * params.kv_len * params.head_dim)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 32.0)
        .collect();
    let value: Vec<_> = (0..key.len())
        .map(|i| ((i * 13 % 43) as f32 - 21.0) / 32.0)
        .collect();
    let mut expected = vec![0.0; query.len()];
    crate::attention::attention_cpu(&query, &key, &value, &mut expected, &params);
    let query = MetalBackend::from_slice_in_context(ctx, &query);
    let key = MetalBackend::from_slice_in_context(ctx, &key);
    let value = MetalBackend::from_slice_in_context(ctx, &value);
    let mut output = MetalBackend::from_slice_in_context(ctx, &vec![f32::NAN; expected.len()]);
    let config = AttnConfig {
        num_heads: params.num_heads,
        num_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        causal: params.causal,
        scale: 1.0 / (params.head_dim as f32).sqrt(),
        ..Default::default()
    };
    MetalBackend::begin_execution(ctx).unwrap();
    MetalBackend::flash_attention(
        ctx,
        &query,
        &key,
        &value,
        &mut output,
        params.batch,
        params.q_len,
        params.kv_len,
        params.pos_offset,
        &config,
    );
    MetalBackend::sync_checked(ctx).unwrap();
    // Recorded from the dispatch that actually encoded this command buffer,
    // rather than predicting a route independently in the test.
    assert_eq!(ctx.last_attention_kernel, Some(expected_kernel));
    let actual = MetalBackend::to_vec_in_context(ctx, &output, expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite(),
            "kernel={expected_kernel:?} q_len={q_len} kv_len={kv_len} index={index}: actual={actual}, expected={expected}");
        assert!((actual - expected).abs() < 1.0e-3,
            "kernel={expected_kernel:?} q_len={q_len} kv_len={kv_len} index={index}: {actual} != {expected}");
    }
}

#[test]
#[ignore = "requires actual Metal attention dispatch for independent policies"]
fn metal_instance_attention_routes_and_full_outputs_survive_interleaving_and_peer_retirement() {
    let mut legacy_config = BackendContextConfig::default();
    legacy_config.metal.force_legacy_attention = true;
    let mut legacy = MetalBackend::new_context(&legacy_config).unwrap();
    let mut fast = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    let mut no_decode_config = BackendContextConfig::default();
    no_decode_config.metal.allow_decode_widen = false;
    let mut no_decode = MetalBackend::new_context(&no_decode_config).unwrap();
    // Deliberately alternate original owners and query shapes.
    attention(&mut fast, 1, 19, MetalAttentionKernel::DecodeWide);
    attention(&mut legacy, 8, 19, MetalAttentionKernel::Legacy);
    attention(&mut no_decode, 1, 19, MetalAttentionKernel::Legacy);
    attention(&mut fast, 8, 19, MetalAttentionKernel::QueryTiled);
    attention(&mut legacy, 1, 19, MetalAttentionKernel::Legacy);
    attention(&mut no_decode, 8, 19, MetalAttentionKernel::QueryTiled);
    MetalBackend::retire_context(&mut legacy).unwrap();
    drop(legacy);
    attention(&mut fast, 1, 19, MetalAttentionKernel::DecodeWide);
    attention(&mut no_decode, 1, 19, MetalAttentionKernel::Legacy);
    MetalBackend::retire_context(&mut fast).unwrap();
    MetalBackend::retire_context(&mut no_decode).unwrap();
}

#[test]
#[ignore = "requires actual Metal query-tiled attention with exact-length KV storage"]
fn metal_query_tiled_partial_kv_tiles_match_cpu_without_padding() {
    let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
    // Keep exact KV extents: no backing padding may make a final matrix
    // load legal. Cover an all-tail tile, full tiles, and a tail following
    // full tiles, with a partial query tile and independent batch/GQA heads.
    for (q_len, kv_len) in [(8, 8), (8, 19), (8, 32), (9, 33), (8, 64), (9, 65)] {
        attention(&mut ctx, q_len, kv_len, MetalAttentionKernel::QueryTiled);
    }
    MetalBackend::retire_context(&mut ctx).unwrap();
}
