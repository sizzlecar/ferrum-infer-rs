use std::ffi::c_void;

use half::f16;
use metal::{Buffer, BufferRef, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

use super::super::numerical_tolerance;
use super::*;

const TOKENS: usize = 2;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const ROPE_DIM: usize = 16;
const QUERY_FEATURES: usize = QUERY_HEADS * HEAD_DIM;
const QUERY_PROJECTION_FEATURES: usize = QUERY_FEATURES * 2;
const KV_FEATURES: usize = KV_HEADS * HEAD_DIM;
const TEST_PAGE_ELEMENTS: usize = 2 * KV_FEATURES;
const CPU_OUTPUT_TOLERANCE_ID: &str =
    "runtime-vnext.metal.causal-attention.v2.operation.fp16.none.fixed-page-split";
const CPU_KV_STATE_TOLERANCE_ID: &str =
    "runtime-vnext.metal.causal-attention.v2.state.kv.fp16.none.fixed-page-split";
// This stricter diagnostic never substitutes for a catalog-bound release comparison.
const SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS: f32 = 0.001;
const CPU_KV_STATE_DIAGNOSTIC_MAX_ABS: f32 = 0.001;

#[test]
fn fixed_page_attention_matches_cpu_and_preserves_split_decode_state_on_real_metal() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping causal-attention conformance");
        return;
    };
    let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    assert_eq!(pipelines.prepare.thread_execution_width(), SIMD_THREADS);
    assert_eq!(pipelines.attention.thread_execution_width(), SIMD_THREADS);
    let queue = device.new_command_queue();

    let query_raw = half_values(TOKENS * QUERY_PROJECTION_FEATURES, 0.037, 0.31);
    let key_raw = half_values(TOKENS * KV_FEATURES, 0.043, 0.27);
    let value_raw = half_values(TOKENS * KV_FEATURES, 0.029, 0.22);
    let query_norm = half_values(HEAD_DIM, 0.019, 0.94);
    let key_norm = half_values(HEAD_DIM, 0.023, 0.89);

    let full_pages = test_pages(&device, TOKENS);
    assert_ne!(full_pages[0].contents(), full_pages[1].contents());
    let full_output = run_segment(
        &device,
        &queue,
        &pipelines,
        SegmentInputs {
            query_raw: &query_raw,
            key_raw: &key_raw,
            value_raw: &value_raw,
        },
        &query_norm,
        &key_norm,
        &full_pages,
        0,
    );

    let split_pages = test_pages(&device, TOKENS);
    let mut split_output = run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&query_raw, &key_raw, &value_raw, 0),
        &query_norm,
        &key_norm,
        &split_pages[..1],
        0,
    );
    split_output.extend(run_segment(
        &device,
        &queue,
        &pipelines,
        segment(&query_raw, &key_raw, &value_raw, 1),
        &query_norm,
        &key_norm,
        &split_pages,
        1,
    ));

    let cpu_output = cpu_attention(&query_raw, &key_raw, &value_raw, &query_norm, &key_norm);
    let cpu_kv_state = cpu_fixed_page_kv_state(&key_raw, &value_raw, &key_norm);
    let full_kv_state = read_pages(&full_pages);
    let split_kv_state = read_pages(&split_pages);
    assert!(full_output.iter().any(|value| value.abs() > 1.0e-4));
    numerical_tolerance::assert_matches(
        "Metal/CPU causal output",
        &full_output,
        &[TOKENS, QUERY_HEADS, HEAD_DIM],
        &cpu_output,
        &[TOKENS, QUERY_HEADS, HEAD_DIM],
        numerical_tolerance::LogicalDtype::Fp16,
        CPU_OUTPUT_TOLERANCE_ID,
    )
    .expect("reviewed causal-attention numerical contract");
    assert_close(
        "full/cpu causal KV state",
        &full_kv_state,
        &cpu_kv_state,
        CPU_KV_STATE_DIAGNOSTIC_MAX_ABS,
    );
    numerical_tolerance::assert_matches(
        "Metal/CPU causal KV state after split decode",
        &split_kv_state,
        &[TOKENS, 2, KV_HEADS, HEAD_DIM],
        &cpu_kv_state,
        &[TOKENS, 2, KV_HEADS, HEAD_DIM],
        numerical_tolerance::LogicalDtype::Fp16,
        CPU_KV_STATE_TOLERANCE_ID,
    )
    .expect("reviewed causal KV-state numerical contract");
    assert_close(
        "full/split causal output",
        &full_output,
        &split_output,
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split first KV page",
        &read_f16(&full_pages[0], TEST_PAGE_ELEMENTS),
        &read_f16(&split_pages[0], TEST_PAGE_ELEMENTS),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
    assert_close(
        "full/split second KV page",
        &read_f16(&full_pages[1], TEST_PAGE_ELEMENTS),
        &read_f16(&split_pages[1], TEST_PAGE_ELEMENTS),
        SPLIT_CONTINUITY_DIAGNOSTIC_MAX_ABS,
    );
}

#[test]
fn grouped_decode_head128_without_gate_matches_direct_and_cpu_across_page_boundary_on_real_metal() {
    run_decode_cpu_case(
        "head128 ratio8 grouped decode",
        128,
        32,
        4,
        false,
        257,
        AttentionDispatchKind::GroupedDecode,
    );
}

#[test]
fn grouped_decode_head256_with_gate_matches_direct_and_cpu_across_page_boundary_on_real_metal() {
    run_decode_cpu_case(
        "head256 ratio4 grouped decode",
        256,
        16,
        4,
        true,
        257,
        AttentionDispatchKind::GroupedDecode,
    );
}

#[test]
fn direct_decode_ratio_one_matches_general_and_cpu_across_page_boundary_on_real_metal() {
    run_decode_cpu_case(
        "head128 ratio1 direct decode",
        128,
        4,
        4,
        false,
        33,
        AttentionDispatchKind::DirectDecode,
    );
}

#[test]
fn attention_dispatch_plan_routes_supported_prefill_and_preserves_decode() {
    for (head_dim, tokens, expected) in [
        (128, 1, AttentionDispatchKind::GroupedDecode),
        (256, 1, AttentionDispatchKind::GroupedDecode),
        (64, 1, AttentionDispatchKind::General),
        (128, 2, AttentionDispatchKind::General),
        (128, 7, AttentionDispatchKind::General),
        (128, 8, AttentionDispatchKind::TiledPrefill),
        (128, 9, AttentionDispatchKind::TiledPrefill),
        (256, 8, AttentionDispatchKind::GqaTiledPrefill),
        (64, 9, AttentionDispatchKind::General),
    ] {
        assert_eq!(
            attention_dispatch_plan(&dispatch_test_params(tokens, head_dim)).kind,
            expected,
            "head_dim={head_dim} tokens={tokens}",
        );
    }

    let exact_tile = attention_dispatch_plan(&dispatch_test_params(8, 128));
    assert_eq!(exact_tile.threadgroups, [1, 16, 1]);
    assert_eq!(
        exact_tile.threads_per_threadgroup,
        [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
    );
    assert!(exact_tile
        .threadgroup_memory_bytes
        .iter()
        .all(|bytes| *bytes > 0));
    assert_eq!(exact_tile.threadgroup_memory_bytes, [2560, 5120]);
    assert_eq!(grouped_decode_reduce_threadgroup_memory_bytes(), 48);

    let head256_tile = attention_dispatch_plan(&dispatch_test_params(8, 256));
    assert_eq!(head256_tile.threadgroup_memory_bytes, [10240, 20480]);
    assert_eq!(
        head256_tile.threadgroup_memory_bytes.iter().sum::<u64>(),
        30_720,
    );
    let exact_memory_limit =
        attention_dispatch_plan_with_memory_limit(&dispatch_test_params(8, 256), 30_720);
    assert_eq!(
        exact_memory_limit.kind,
        AttentionDispatchKind::GqaTiledPrefill,
    );
    let memory_limited =
        attention_dispatch_plan_with_memory_limit(&dispatch_test_params(8, 256), 30_720 - 1);
    assert_eq!(memory_limited.kind, AttentionDispatchKind::TiledPrefill);
    assert_eq!(memory_limited.threads_per_threadgroup, [32, 4, 1]);
    assert_eq!(memory_limited.threadgroup_memory_bytes, [4608, 9216]);

    let query_tail = attention_dispatch_plan(&dispatch_test_params(9, 128));
    assert_eq!(query_tail.threadgroups, [2, 16, 1]);

    let mut qwen35_eight_query_two_kv = dispatch_test_params(8, 256);
    qwen35_eight_query_two_kv.query_heads = 8;
    qwen35_eight_query_two_kv.key_value_heads = 2;
    qwen35_eight_query_two_kv.query_projection_stride =
        2 * qwen35_eight_query_two_kv.query_heads * qwen35_eight_query_two_kv.head_dim;
    qwen35_eight_query_two_kv.kv_projection_stride =
        qwen35_eight_query_two_kv.key_value_heads * qwen35_eight_query_two_kv.head_dim;
    let qwen35_eight_query_two_kv_plan = attention_dispatch_plan(&qwen35_eight_query_two_kv);
    assert_eq!(
        qwen35_eight_query_two_kv_plan.kind,
        AttentionDispatchKind::GqaTiledPrefill,
    );
    assert_eq!(qwen35_eight_query_two_kv_plan.threadgroups, [1, 4, 1]);
    assert_eq!(
        qwen35_eight_query_two_kv_plan.threadgroup_memory_bytes,
        [10240, 20480],
    );

    let mut ratio_two_prefill_params = dispatch_test_params(8, 256);
    ratio_two_prefill_params.query_heads = 8;
    ratio_two_prefill_params.query_projection_stride =
        2 * ratio_two_prefill_params.query_heads * ratio_two_prefill_params.head_dim;
    let ratio_two_prefill = attention_dispatch_plan(&ratio_two_prefill_params);
    assert_eq!(
        ratio_two_prefill.kind,
        AttentionDispatchKind::GqaTiledPrefill,
    );
    assert_eq!(ratio_two_prefill.threadgroups, [1, 4, 1]);

    let mut split_token_page = dispatch_test_params(8, 128);
    split_token_page.page_elements -= 1;
    assert_eq!(
        attention_dispatch_plan(&split_token_page).kind,
        AttentionDispatchKind::General,
    );

    let mut split_decode_token_page = dispatch_test_params(1, 128);
    split_decode_token_page.page_elements -= 1;
    assert_eq!(
        attention_dispatch_plan(&split_decode_token_page).kind,
        AttentionDispatchKind::General,
    );

    let mut partial_matrix_page = dispatch_test_params(8, 128);
    let token_stride = 2 * partial_matrix_page.key_value_heads * partial_matrix_page.head_dim;
    partial_matrix_page.page_elements = 7 * token_stride;
    assert_eq!(
        attention_dispatch_plan(&partial_matrix_page).kind,
        AttentionDispatchKind::General,
    );

    let mut seven_token_decode_page = dispatch_test_params(1, 128);
    seven_token_decode_page.page_elements = 7 * token_stride;
    assert_eq!(
        attention_dispatch_plan(&seven_token_decode_page).kind,
        AttentionDispatchKind::DirectDecode,
    );

    let mut one_token_page = dispatch_test_params(1, 128);
    one_token_page.page_elements = token_stride;
    assert_eq!(
        attention_dispatch_plan(&one_token_page).kind,
        AttentionDispatchKind::DirectDecode,
    );

    let decode = attention_dispatch_plan(&dispatch_test_params(1, 256));
    assert_eq!(decode.kind, AttentionDispatchKind::GroupedDecode);
    assert_eq!(decode.threadgroups, [GROUPED_DECODE_PARTITIONS, 4, 1]);
    assert_eq!(
        decode.threads_per_threadgroup,
        [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
    );
    assert_eq!(decode.threadgroup_memory_bytes, [4608, 9216]);

    let mut ratio_one_params = dispatch_test_params(1, 128);
    ratio_one_params.key_value_heads = ratio_one_params.query_heads;
    ratio_one_params.kv_projection_stride =
        ratio_one_params.key_value_heads * ratio_one_params.head_dim;
    let ratio_one = attention_dispatch_plan(&ratio_one_params);
    assert_eq!(ratio_one.kind, AttentionDispatchKind::DirectDecode);
    assert_eq!(ratio_one.threadgroups, [1, 16, 1]);

    let mut ratio_one_prefill_params = dispatch_test_params(8, 128);
    ratio_one_prefill_params.key_value_heads = ratio_one_prefill_params.query_heads;
    ratio_one_prefill_params.kv_projection_stride =
        ratio_one_prefill_params.key_value_heads * ratio_one_prefill_params.head_dim;
    let ratio_one_prefill = attention_dispatch_plan(&ratio_one_prefill_params);
    assert_eq!(ratio_one_prefill.kind, AttentionDispatchKind::TiledPrefill);
    assert_eq!(ratio_one_prefill.threadgroups, [1, 16, 1]);
    assert_eq!(
        ratio_one_prefill.threads_per_threadgroup,
        [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
    );

    let mut odd_ratio_prefill_params = dispatch_test_params(8, 128);
    odd_ratio_prefill_params.query_heads = 12;
    odd_ratio_prefill_params.query_projection_stride =
        2 * odd_ratio_prefill_params.query_heads * odd_ratio_prefill_params.head_dim;
    let odd_ratio_prefill = attention_dispatch_plan(&odd_ratio_prefill_params);
    assert_eq!(odd_ratio_prefill.kind, AttentionDispatchKind::TiledPrefill);
    assert_eq!(odd_ratio_prefill.threadgroups, [1, 12, 1]);

    let mut invalid_gqa = dispatch_test_params(1, 128);
    invalid_gqa.query_heads = 10;
    assert_eq!(
        attention_dispatch_plan(&invalid_gqa).kind,
        AttentionDispatchKind::General,
    );

    let mut invalid_prefill_gqa = dispatch_test_params(8, 128);
    invalid_prefill_gqa.query_heads = 10;
    assert_eq!(
        attention_dispatch_plan(&invalid_prefill_gqa).kind,
        AttentionDispatchKind::General,
    );
}

#[test]
fn packed_mixed_decode_and_prefill_keep_participant_local_dispatch_plans() {
    // Packed projections are shared, but attention is dispatched once per
    // participant. A decode participant must not inherit its prefill peer's
    // tiled launch shape (or vice versa).
    let participant_params = [dispatch_test_params(1, 256), dispatch_test_params(9, 256)];
    let plans = participant_params.map(|params| attention_dispatch_plan(&params));
    assert_eq!(plans[0].kind, AttentionDispatchKind::GroupedDecode);
    assert_eq!(plans[0].threadgroups[0], GROUPED_DECODE_PARTITIONS);
    assert_eq!(plans[1].kind, AttentionDispatchKind::GqaTiledPrefill);
    assert_eq!(plans[1].threadgroups[0], 2);
    let grouped_decode_reductions = plans
        .iter()
        .filter(|plan| plan.kind == AttentionDispatchKind::GroupedDecode)
        .count() as u64;
    assert_eq!(
        physical_dispatch_count(plans.len(), true) + grouped_decode_reductions,
        11,
    );
}

#[test]
fn tiled_prefill_head128_without_gate_matches_general_and_cpu_for_fresh_exact_tile_on_real_metal() {
    run_prefill_cpu_case(
        "head128 ungated fresh",
        128,
        32,
        4,
        false,
        0,
        8,
        AttentionDispatchKind::TiledPrefill,
    );
}

#[test]
fn gqa_tiled_prefill_head256_with_gate_matches_general_and_cpu_across_prefix_page_and_tail_on_real_metal(
) {
    run_prefill_cpu_case(
        "head256 gated continuation",
        256,
        16,
        4,
        true,
        62,
        9,
        AttentionDispatchKind::GqaTiledPrefill,
    );
}

#[test]
fn gqa_tiled_prefill_head256_matches_general_and_cpu_at_exact_64_key_boundary_on_real_metal() {
    run_prefill_cpu_case(
        "head256 gated exact 64 keys",
        256,
        16,
        4,
        true,
        56,
        8,
        AttentionDispatchKind::GqaTiledPrefill,
    );
}

#[test]
fn gqa_tiled_prefill_head256_matches_general_and_cpu_across_65th_key_on_real_metal() {
    run_prefill_cpu_case(
        "head256 gated 65 keys",
        256,
        16,
        4,
        true,
        57,
        8,
        AttentionDispatchKind::GqaTiledPrefill,
    );
}

#[test]
fn gqa_tiled_prefill_head256_matches_general_and_cpu_across_129_keys_on_real_metal() {
    run_prefill_cpu_case(
        "head256 gated 129 through 136 keys",
        256,
        16,
        4,
        true,
        128,
        8,
        AttentionDispatchKind::GqaTiledPrefill,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_cpu_case(
    label: &str,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    output_gate: bool,
    prefix: usize,
    tokens: usize,
    expected_kind: AttentionDispatchKind,
) {
    let context = prefix + tokens;

    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping tiled causal-attention conformance");
        return;
    };
    let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let query_features = query_heads * head_dim;
    let query_projection_features = query_features * if output_gate { 2 } else { 1 };
    let kv_features = kv_heads * head_dim;
    let query = (0..tokens * query_features)
        .map(|index| f16::from_f32(((index as f32 * 0.013) + 0.3).sin() * 0.2))
        .collect::<Vec<_>>();
    let query_raw = (0..tokens * query_projection_features)
        .map(|index| f16::from_f32(((index as f32 * 0.017) + 0.7).cos() * 0.3))
        .collect::<Vec<_>>();
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / std::mem::size_of::<f16>();
    let state_elements = context * 2 * kv_features;
    let page_count = state_elements.div_ceil(page_elements);
    // Poison unused page slack so a speculative matrix load beyond `context`
    // cannot be hidden by a zero-initialized test allocation.
    let mut state = vec![f16::NAN; page_count * page_elements];
    for position in 0..context {
        for kind in 0..2 {
            for head in 0..kv_heads {
                for dim in 0..head_dim {
                    let semantic = (((position * 2 + kind) * kv_heads + head) * head_dim) + dim;
                    let phase = semantic as f32 * if kind == 0 { 0.0091 } else { 0.0117 };
                    state[semantic] = f16::from_f32(
                        if kind == 0 {
                            phase.cos()
                        } else {
                            (phase + 1.7).sin()
                        } * 0.25,
                    );
                }
            }
        }
    }
    let pages = state
        .chunks_exact(page_elements)
        .map(|page| shared_buffer(&device, page))
        .collect::<Vec<_>>();
    let query_buffer = shared_buffer(&device, &query);
    let query_raw_buffer = shared_buffer(&device, &query_raw);
    let params = CausalAttentionParams {
        page_elements: page_elements as u32,
        page_count: page_count as u32,
        position_start: prefix as u32,
        tokens: tokens as u32,
        query_heads: query_heads as u32,
        key_value_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        rope_dim: head_dim as u32,
        query_projection_stride: query_projection_features as u32,
        query_head_stride: (head_dim * if output_gate { 2 } else { 1 }) as u32,
        kv_projection_stride: kv_features as u32,
        output_gate: u32::from(output_gate),
        rope_interleaved: 0,
        attention_simdgroups: pipelines.attention_simdgroups_for_context(context as u64),
        epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    };
    let selected_plan = attention_dispatch_plan(&params);
    assert_eq!(selected_plan.kind, expected_kind);
    let selected = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        selected_plan,
    );
    let tiled = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        tiled_prefill_attention_dispatch_plan(&params),
    );
    let general = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        general_attention_dispatch_plan(&params),
    );
    let expected = cpu_tiled_prefill_attention(
        &query,
        &query_raw,
        &state,
        prefix,
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        output_gate,
    );
    let selected_cpu_error = assert_close(
        &format!("{label} selected/cpu"),
        &selected,
        &expected,
        0.001,
    );
    let general_cpu_error =
        assert_close(&format!("{label} general/cpu"), &general, &expected, 0.001);
    let selected_tiled_error =
        assert_close(&format!("{label} selected/tiled"), &selected, &tiled, 0.001);
    let selected_general_error = assert_close(
        &format!("{label} selected/general"),
        &selected,
        &general,
        0.001,
    );
    eprintln!(
        "{label}: max_abs selected/cpu={selected_cpu_error} general/cpu={general_cpu_error} selected/tiled={selected_tiled_error} selected/general={selected_general_error}",
    );
}

fn run_decode_cpu_case(
    label: &str,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    output_gate: bool,
    context: usize,
    expected_kind: AttentionDispatchKind,
) {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping decode conformance");
        return;
    };
    let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let query_features = query_heads * head_dim;
    let query_projection_features = query_features * if output_gate { 2 } else { 1 };
    let kv_features = kv_heads * head_dim;
    let query = (0..query_features)
        .map(|index| f16::from_f32(((index as f32 * 0.013) + 0.3).sin() * 0.2))
        .collect::<Vec<_>>();
    let query_raw = (0..query_projection_features)
        .map(|index| f16::from_f32(((index as f32 * 0.017) + 0.7).cos() * 0.3))
        .collect::<Vec<_>>();
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / std::mem::size_of::<f16>();
    let state_elements = context * 2 * kv_features;
    let page_count = state_elements.div_ceil(page_elements);
    // Only valid K/V rows are initialized. A direct kernel must not sample
    // the unused tail of the final page, even when the context crosses a page.
    let mut state = vec![f16::NAN; page_count * page_elements];
    for position in 0..context {
        for kind in 0..2 {
            for head in 0..kv_heads {
                for dim in 0..head_dim {
                    let semantic = (((position * 2 + kind) * kv_heads + head) * head_dim) + dim;
                    let phase = semantic as f32 * if kind == 0 { 0.0091 } else { 0.0117 };
                    state[semantic] = f16::from_f32(
                        if kind == 0 {
                            phase.cos()
                        } else {
                            (phase + 1.7).sin()
                        } * 0.25,
                    );
                }
            }
        }
    }
    let pages = state
        .chunks_exact(page_elements)
        .map(|page| shared_buffer(&device, page))
        .collect::<Vec<_>>();
    let query_buffer = shared_buffer(&device, &query);
    let query_raw_buffer = shared_buffer(&device, &query_raw);
    let params = CausalAttentionParams {
        page_elements: page_elements as u32,
        page_count: page_count as u32,
        position_start: (context - 1) as u32,
        tokens: 1,
        query_heads: query_heads as u32,
        key_value_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        rope_dim: head_dim as u32,
        query_projection_stride: query_projection_features as u32,
        query_head_stride: (head_dim * if output_gate { 2 } else { 1 }) as u32,
        kv_projection_stride: kv_features as u32,
        output_gate: u32::from(output_gate),
        rope_interleaved: 0,
        attention_simdgroups: pipelines.attention_simdgroups_for_context(context as u64),
        epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    };
    let selected_plan = attention_dispatch_plan(&params);
    assert_eq!(selected_plan.kind, expected_kind);
    let selected = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        selected_plan,
    );
    let direct = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        direct_decode_attention_dispatch_plan(&params),
    );
    let general = run_attention_plan(
        &device,
        &queue,
        &pipelines,
        &query_buffer,
        &query_raw_buffer,
        &pages,
        &params,
        general_attention_dispatch_plan(&params),
    );
    let expected = cpu_tiled_prefill_attention(
        &query,
        &query_raw,
        &state,
        context - 1,
        1,
        query_heads,
        kv_heads,
        head_dim,
        output_gate,
    );
    assert!(selected.iter().all(|value| value.is_finite()));
    assert!(direct.iter().all(|value| value.is_finite()));
    assert!(general.iter().all(|value| value.is_finite()));
    let selected_cpu_error = assert_close(
        &format!("{label} selected/cpu"),
        &selected,
        &expected,
        0.001,
    );
    let direct_cpu_error = assert_close(&format!("{label} direct/cpu"), &direct, &expected, 0.001);
    let general_cpu_error =
        assert_close(&format!("{label} general/cpu"), &general, &expected, 0.001);
    let selected_direct_error = assert_close(
        &format!("{label} selected/direct"),
        &selected,
        &direct,
        0.001,
    );
    let selected_general_error = assert_close(
        &format!("{label} selected/general"),
        &selected,
        &general,
        0.001,
    );
    eprintln!(
        "{label}: max_abs selected/cpu={selected_cpu_error} direct/cpu={direct_cpu_error} general/cpu={general_cpu_error} selected/direct={selected_direct_error} selected/general={selected_general_error}",
    );
}

#[allow(clippy::too_many_arguments)]
fn run_attention_plan(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalCausalAttentionPipelines,
    query: &BufferRef,
    query_raw: &BufferRef,
    pages: &[Buffer],
    params: &CausalAttentionParams,
    plan: AttentionDispatchPlan,
) -> Vec<f32> {
    let output_elements =
        params.tokens as usize * params.query_heads as usize * params.head_dim as usize;
    let output = output_buffer::<f16>(device, output_elements);
    let grouped_partials = (plan.kind == AttentionDispatchKind::GroupedDecode).then(|| {
        output_buffer::<f32>(
            device,
            GROUPED_DECODE_PARTITIONS as usize
                * params.query_heads as usize
                * (params.head_dim as usize + 2),
        )
    });
    let argument_buffer = device.new_buffer(
        pipelines.binding_slot_bytes().unwrap(),
        MTLResourceOptions::StorageModeShared,
    );
    let argument_encoder = pipelines.new_binding_encoder();
    argument_encoder.set_argument_buffer(&argument_buffer, 0);
    let page_refs = pages.iter().map(|page| &**page).collect::<Vec<_>>();
    let page_offsets = vec![0; pages.len()];
    argument_encoder.set_buffers(0, &page_refs, &page_offsets);

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    set_raw(encoder, 0, query);
    set_raw(encoder, 1, query_raw);
    set_raw(encoder, 2, grouped_partials.as_ref().unwrap_or(&output));
    encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&argument_buffer), 0);
    set_raw_params(encoder, 4, params);
    use_raw_pages(encoder, pages);
    encode_attention_dispatch(pipelines, encoder, plan);
    if let Some(grouped_partials) = grouped_partials.as_ref() {
        encoder.set_compute_pipeline_state(&pipelines.grouped_decode_reduce_attention);
        set_raw(encoder, 0, grouped_partials);
        set_raw(encoder, 1, query_raw);
        set_raw(encoder, 2, &output);
        set_raw_params(encoder, 4, params);
        encoder.set_threadgroup_memory_length(0, grouped_decode_reduce_threadgroup_memory_bytes());
        encoder.set_threadgroup_memory_length(1, 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.query_heads), 1, 1),
            MTLSize::new(SIMD_THREADS, 1, 1),
        );
        encoder.set_threadgroup_memory_length(0, 0);
        encoder.set_threadgroup_memory_length(1, 0);
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    read_f16(&output, output_elements)
}

fn dispatch_test_params(tokens: u32, head_dim: u32) -> CausalAttentionParams {
    const QUERY_HEADS: u32 = 16;
    const KV_HEADS: u32 = 4;
    CausalAttentionParams {
        page_elements: (VNEXT_KV_PAGE_BYTES / std::mem::size_of::<f16>() as u64) as u32,
        page_count: 1,
        position_start: (GROUPED_DECODE_MINIMUM_CONTEXT - 1) as u32,
        tokens,
        query_heads: QUERY_HEADS,
        key_value_heads: KV_HEADS,
        head_dim,
        rope_dim: head_dim,
        query_projection_stride: 2 * QUERY_HEADS * head_dim,
        query_head_stride: 2 * head_dim,
        kv_projection_stride: KV_HEADS * head_dim,
        output_gate: 1,
        rope_interleaved: 0,
        attention_simdgroups: 3,
        epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_tiled_prefill_attention(
    query: &[f16],
    query_raw: &[f16],
    state: &[f16],
    position_start: usize,
    tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    output_gate: bool,
) -> Vec<f32> {
    let query_features = query_heads * head_dim;
    let query_projection_features = query_features * if output_gate { 2 } else { 1 };
    let mut output = vec![0.0_f32; tokens * query_features];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for token in 0..tokens {
        for query_head in 0..query_heads {
            let kv_head = query_head / (query_heads / kv_heads);
            let query_base = (token * query_heads + query_head) * head_dim;
            let mut probabilities = (0..=position_start + token)
                .map(|position| {
                    let key_base = ((position * 2) * kv_heads + kv_head) * head_dim;
                    (0..head_dim)
                        .map(|dim| {
                            f32::from(query[query_base + dim]) * f32::from(state[key_base + dim])
                        })
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            let maximum = probabilities
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator = probabilities
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            for dim in 0..head_dim {
                let context = probabilities
                    .iter()
                    .enumerate()
                    .map(|(position, probability)| {
                        let value = (((position * 2 + 1) * kv_heads + kv_head) * head_dim) + dim;
                        probability * f32::from(state[value])
                    })
                    .sum::<f32>()
                    / denominator;
                let gate = if output_gate {
                    let gate_index = token * query_projection_features
                        + query_head * 2 * head_dim
                        + head_dim
                        + dim;
                    1.0 / (1.0 + (-f32::from(query_raw[gate_index])).exp())
                } else {
                    1.0
                };
                output[query_base + dim] = context * gate;
            }
        }
    }
    output
}

fn read_pages(pages: &[Buffer]) -> Vec<f32> {
    pages
        .iter()
        .flat_map(|page| read_f16(page, TEST_PAGE_ELEMENTS))
        .collect()
}

struct SegmentInputs<'a> {
    query_raw: &'a [f16],
    key_raw: &'a [f16],
    value_raw: &'a [f16],
}

fn segment<'a>(
    query_raw: &'a [f16],
    key_raw: &'a [f16],
    value_raw: &'a [f16],
    token: usize,
) -> SegmentInputs<'a> {
    SegmentInputs {
        query_raw: &query_raw
            [token * QUERY_PROJECTION_FEATURES..(token + 1) * QUERY_PROJECTION_FEATURES],
        key_raw: &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
        value_raw: &value_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
    }
}

#[allow(clippy::too_many_arguments)]
fn run_segment(
    device: &Device,
    queue: &CommandQueueRef,
    pipelines: &MetalCausalAttentionPipelines,
    inputs: SegmentInputs<'_>,
    query_norm_values: &[f16],
    key_norm_values: &[f16],
    pages: &[Buffer],
    position_start: usize,
) -> Vec<f32> {
    let tokens = inputs.query_raw.len() / QUERY_PROJECTION_FEATURES;
    let params = CausalAttentionParams {
        page_elements: TEST_PAGE_ELEMENTS as u32,
        page_count: pages.len() as u32,
        position_start: position_start as u32,
        tokens: tokens as u32,
        query_heads: QUERY_HEADS as u32,
        key_value_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        rope_dim: ROPE_DIM as u32,
        query_projection_stride: QUERY_PROJECTION_FEATURES as u32,
        query_head_stride: (2 * HEAD_DIM) as u32,
        kv_projection_stride: KV_FEATURES as u32,
        output_gate: 1,
        rope_interleaved: 0,
        attention_simdgroups: pipelines
            .attention_simdgroups_for_context((position_start + tokens) as u64),
        epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    };
    let query_raw = shared_buffer(device, inputs.query_raw);
    let key_raw = shared_buffer(device, inputs.key_raw);
    let value_raw = shared_buffer(device, inputs.value_raw);
    let query_norm = shared_buffer(device, query_norm_values);
    let key_norm = shared_buffer(device, key_norm_values);
    let query = output_buffer::<f16>(device, tokens * QUERY_FEATURES);
    let output = output_buffer::<f16>(device, tokens * QUERY_FEATURES);
    let argument_buffer = device.new_buffer(
        pipelines.binding_slot_bytes().unwrap(),
        MTLResourceOptions::StorageModeShared,
    );
    let argument_encoder = pipelines.new_binding_encoder();
    argument_encoder.set_argument_buffer(&argument_buffer, 0);
    let page_refs = pages.iter().map(|page| &**page).collect::<Vec<_>>();
    let page_offsets = vec![0; pages.len()];
    argument_encoder.set_buffers(0, &page_refs, &page_offsets);

    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.prepare);
    for (index, buffer) in [
        &*query_raw,
        &*key_raw,
        &*value_raw,
        &*query_norm,
        &*key_norm,
        &*query,
    ]
    .into_iter()
    .enumerate()
    {
        set_raw(encoder, index as u64, buffer);
    }
    encoder.set_buffer(PREPARE_PAGE_TABLE_INDEX, Some(&argument_buffer), 0);
    set_raw_params(encoder, 7, &params);
    use_raw_pages(encoder, pages);
    encoder.set_threadgroup_memory_length(0, 0);
    encoder.set_threadgroup_memory_length(1, 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(tokens as u64, (QUERY_HEADS + 2 * KV_HEADS) as u64, 1),
        MTLSize::new(SIMD_THREADS, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.attention);
    set_raw(encoder, 0, &query);
    set_raw(encoder, 1, &query_raw);
    set_raw(encoder, 2, &output);
    encoder.set_buffer(ATTENTION_PAGE_TABLE_INDEX, Some(&argument_buffer), 0);
    set_raw_params(encoder, 4, &params);
    use_raw_pages(encoder, pages);
    encoder.set_threadgroup_memory_length(0, attention_threadgroup_memory_bytes(&params));
    encoder.set_threadgroup_memory_length(1, 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(tokens as u64, QUERY_HEADS as u64, 1),
        MTLSize::new(
            SIMD_THREADS,
            u64::from(pipelines.attention_simdgroups_for_context((position_start + tokens) as u64)),
            1,
        ),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    read_f16(&output, tokens * QUERY_FEATURES)
}

fn use_raw_pages(encoder: &ComputeCommandEncoderRef, pages: &[Buffer]) {
    for page in pages {
        encoder.use_resource(&**page, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
}

fn set_raw(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &BufferRef) {
    encoder.set_buffer(index, Some(buffer), 0);
}

fn set_raw_params(encoder: &ComputeCommandEncoderRef, index: u64, params: &CausalAttentionParams) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<CausalAttentionParams>() as u64,
        params as *const _ as *const c_void,
    );
}

fn test_pages(device: &Device, count: usize) -> Vec<Buffer> {
    (0..count)
        .map(|_| output_buffer::<f16>(device, TEST_PAGE_ELEMENTS))
        .collect()
}

fn cpu_attention(
    query_raw: &[f16],
    key_raw: &[f16],
    value_raw: &[f16],
    query_norm: &[f16],
    key_norm: &[f16],
) -> Vec<f32> {
    let mut query = vec![0.0_f32; TOKENS * QUERY_FEATURES];
    let mut key = vec![0.0_f32; TOKENS * KV_FEATURES];
    for token in 0..TOKENS {
        for head in 0..QUERY_HEADS {
            let source = token * QUERY_PROJECTION_FEATURES + head * 2 * HEAD_DIM;
            let destination = token * QUERY_FEATURES + head * HEAD_DIM;
            prepare_head(
                &query_raw[source..source + HEAD_DIM],
                query_norm,
                token,
                &mut query[destination..destination + HEAD_DIM],
            );
        }
        prepare_head(
            &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
            key_norm,
            token,
            &mut key[token * KV_FEATURES..(token + 1) * KV_FEATURES],
        );
    }

    let mut output = vec![0.0_f32; TOKENS * QUERY_FEATURES];
    for token in 0..TOKENS {
        for head in 0..QUERY_HEADS {
            let query_row = &query[token * QUERY_FEATURES + head * HEAD_DIM
                ..token * QUERY_FEATURES + (head + 1) * HEAD_DIM];
            let mut scores = (0..=token)
                .map(|position| {
                    query_row
                        .iter()
                        .zip(key[position * KV_FEATURES..(position + 1) * KV_FEATURES].iter())
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        / (HEAD_DIM as f32).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            for dim in 0..HEAD_DIM {
                let context = scores
                    .iter()
                    .enumerate()
                    .map(|(position, score)| {
                        score / denominator * f32::from(value_raw[position * KV_FEATURES + dim])
                    })
                    .sum::<f32>();
                let gate_index =
                    token * QUERY_PROJECTION_FEATURES + head * 2 * HEAD_DIM + HEAD_DIM + dim;
                let gate = 1.0 / (1.0 + (-f32::from(query_raw[gate_index])).exp());
                output[token * QUERY_FEATURES + head * HEAD_DIM + dim] = context * gate;
            }
        }
    }
    output
}

fn cpu_fixed_page_kv_state(key_raw: &[f16], value_raw: &[f16], key_norm: &[f16]) -> Vec<f32> {
    let mut state = vec![0.0_f32; TOKENS * TEST_PAGE_ELEMENTS];
    for token in 0..TOKENS {
        let page = token * TEST_PAGE_ELEMENTS;
        prepare_head(
            &key_raw[token * KV_FEATURES..(token + 1) * KV_FEATURES],
            key_norm,
            token,
            &mut state[page..page + KV_FEATURES],
        );
        for dim in 0..KV_FEATURES {
            state[page + KV_FEATURES + dim] = f32::from(value_raw[token * KV_FEATURES + dim]);
        }
    }
    state
}

fn prepare_head(source: &[f16], weight: &[f16], position: usize, output: &mut [f32]) {
    let sum_squares = source
        .iter()
        .map(|value| f32::from(*value).powi(2))
        .sum::<f32>();
    let scale = 1.0 / (sum_squares / HEAD_DIM as f32 + 1.0e-6).sqrt();
    for pair in 0..ROPE_DIM / 2 {
        let low = pair;
        let high = pair + ROPE_DIM / 2;
        let x0 = f32::from(source[low]) * scale * f32::from(weight[low]);
        let x1 = f32::from(source[high]) * scale * f32::from(weight[high]);
        let angle = position as f32 * 10_000.0_f32.powf(-((2 * pair) as f32) / ROPE_DIM as f32);
        output[low] = x0 * angle.cos() - x1 * angle.sin();
        output[high] = x1 * angle.cos() + x0 * angle.sin();
    }
    for dim in ROPE_DIM..HEAD_DIM {
        output[dim] = f32::from(source[dim]) * scale * f32::from(weight[dim]);
    }
}

fn half_values(length: usize, step: f32, base: f32) -> Vec<f16> {
    (0..length)
        .map(|index| f16::from_f32(base + ((index * 17 + 5) % 41) as f32 * step / 41.0))
        .collect()
}

fn shared_buffer<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn output_buffer<T>(device: &Device, elements: usize) -> Buffer {
    device.new_buffer(
        (elements * std::mem::size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f16(buffer: &BufferRef, elements: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f16>(), elements) }
        .iter()
        .map(|value| f32::from(*value))
        .collect()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (index, maximum) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap();
    assert!(
        maximum <= tolerance,
        "{label} maximum absolute error {maximum} at {index}: actual={} expected={} tolerance={tolerance}",
        actual[index],
        expected[index]
    );
    maximum
}
