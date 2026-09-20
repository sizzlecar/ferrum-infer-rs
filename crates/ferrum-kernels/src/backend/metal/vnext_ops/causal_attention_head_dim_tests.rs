//! Real function-constant dispatch versus the retained dynamic fallback.
use super::*;

fn dynamic_pipelines(device: &Device) -> MetalCausalAttentionPipelines {
    let mut pipelines = MetalCausalAttentionPipelines::new(device).unwrap();
    pipelines.specialized = std::array::from_fn(|_| SpecializedAttentionPipelines::default());
    pipelines
}

fn assert_specialized_selected(
    pipelines: &MetalCausalAttentionPipelines,
    params: &CausalAttentionParams,
    route: AttentionDispatchKind,
) {
    let index = match params.head_dim {
        128 => 0,
        256 => 1,
        _ => unreachable!(),
    };
    let expected = match route {
        AttentionDispatchKind::GroupedDecode => {
            let (partial, reduce) = pipelines.specialized[index]
                .grouped
                .as_ref()
                .expect("device must support grouped specialization for this numerical comparison");
            assert!(std::ptr::eq(
                pipelines.grouped_reduce_pipeline(params),
                reduce
            ));
            partial
        }
        AttentionDispatchKind::TiledPrefill => pipelines.specialized[index].tiled.as_ref().unwrap(),
        AttentionDispatchKind::GqaTiledPrefill => {
            pipelines.specialized[index].gqa.as_ref().unwrap()
        }
        _ => unreachable!(),
    };
    assert!(std::ptr::eq(
        pipelines.attention_pipeline(params, route),
        expected
    ));
}

fn compare_outputs(
    runner: &PairedAttention<'_>,
    dynamic: &MetalCausalAttentionPipelines,
    plan: AttentionDispatchPlan,
) -> Value {
    runner.run_with_pipelines(dynamic, plan, 1);
    let baseline = runner.output.read_after_completion(plan.kind);
    runner.run(plan, 1);
    let candidate = runner.output.read_after_completion(plan.kind);
    let difference = assert_close(
        "production function constant/dynamic fallback",
        &candidate,
        &baseline,
        0.001,
    );
    json!({
        "max_abs_candidate_production_difference": difference,
        "different_output_bits": candidate.iter().zip(&baseline).filter(|(a,b)| a.to_bits() != b.to_bits()).count(),
        "full_output_reference_checked": true,
        "output_guard_unchanged": true,
    })
}

#[test]
fn attention_static_head_dim_preserves_paged_decode_and_prefill_tails() {
    for (dim, query_heads, gate, prefix, tokens, route) in [
        (128, 32, false, 256, 1, AttentionDispatchKind::GroupedDecode),
        (256, 24, true, 1024, 1, AttentionDispatchKind::GroupedDecode),
        (128, 32, false, 251, 11, AttentionDispatchKind::TiledPrefill),
        (
            256,
            24,
            true,
            507,
            11,
            AttentionDispatchKind::GqaTiledPrefill,
        ),
    ] {
        autoreleasepool(|| {
            let case = run_prefill_cpu_case(
                "static head dimension boundaries",
                dim,
                query_heads,
                4,
                gate,
                prefix,
                tokens,
                route,
            );
            let dynamic = dynamic_pipelines(&case.device);
            let plan = attention_dispatch_plan(&case.params);
            assert_specialized_selected(&case.pipelines, &case.params, plan.kind);
            let runner = PairedAttention::new(&case);
            let numeric = compare_outputs(&runner, &dynamic, plan);
            // Drop just the selected specialization, leaving other heads/routes
            // available. The real encoder must execute the matching fallback.
            let mut missing = MetalCausalAttentionPipelines::new(&case.device).unwrap();
            let slot = &mut missing.specialized[usize::from(dim == 256)];
            match route {
                AttentionDispatchKind::GroupedDecode => slot.grouped = None,
                AttentionDispatchKind::TiledPrefill => slot.tiled = None,
                AttentionDispatchKind::GqaTiledPrefill => slot.gqa = None,
                _ => unreachable!(),
            }
            runner.run_with_pipelines(&missing, plan, 1);
            let fallback = runner.output.read_after_completion(route);
            runner.run_with_pipelines(&dynamic, plan, 1);
            assert_eq!(fallback, runner.output.read_after_completion(route));
            case.assert_kv_unchanged("static head dimension boundaries");
            println!(
                "{}",
                json!({"kind":"attention_static_head_dim_correctness", "head_dim":dim,
                "prefix_tokens":prefix,"query_tokens":tokens,"route":format!("{:?}",route),"numerics":numeric})
            );
        });
    }
}

#[test]
fn attention_static_head_dim_keeps_int8_and_other_dimensions_dynamic() {
    let case = run_prefill_cpu_case(
        "non-specialized dimension",
        64,
        8,
        4,
        false,
        32,
        3,
        AttentionDispatchKind::General,
    );
    let mut params = case.params;
    for route in [
        AttentionDispatchKind::General,
        AttentionDispatchKind::DirectDecode,
        AttentionDispatchKind::TiledPrefill,
    ] {
        let expected = match route {
            AttentionDispatchKind::General => &case.pipelines.attention,
            AttentionDispatchKind::DirectDecode => &case.pipelines.direct_decode_attention,
            _ => &case.pipelines.tiled_prefill_attention,
        };
        assert!(std::ptr::eq(
            case.pipelines.attention_pipeline(&params, route),
            expected
        ));
    }
    let int8 = MetalCausalAttentionPipelines::new_int8(&case.device).unwrap();
    assert!(int8
        .specialized
        .iter()
        .all(|s| s.grouped.is_none() && s.tiled.is_none() && s.gqa.is_none()));
    params.head_dim = 256;
    params.tokens = 8;
    let plan = int8.dispatch_plan(&params);
    let original = match plan.kind {
        AttentionDispatchKind::GqaTiledPrefill => &int8.gqa_tiled_prefill_attention,
        AttentionDispatchKind::TiledPrefill => &int8.tiled_prefill_attention,
        AttentionDispatchKind::General => &int8.attention,
        _ => unreachable!(),
    };
    assert!(std::ptr::eq(
        int8.attention_pipeline(&params, plan.kind),
        original
    ));
}

#[test]
#[ignore = "paired attention GPU diagnostic; requires exclusive device access"]
fn attention_static_head_dim_paired_microbench() {
    for (dim, query_heads, gate, prefix, tokens, route) in [
        (
            128,
            32,
            false,
            8192,
            1,
            AttentionDispatchKind::GroupedDecode,
        ),
        (256, 24, true, 8192, 1, AttentionDispatchKind::GroupedDecode),
        (128, 32, false, 7680, 8, AttentionDispatchKind::TiledPrefill),
        (
            256,
            24,
            true,
            7680,
            8,
            AttentionDispatchKind::GqaTiledPrefill,
        ),
    ] {
        autoreleasepool(|| {
            let case = run_prefill_cpu_case(
                "static head dimension timing",
                dim,
                query_heads,
                4,
                gate,
                prefix,
                tokens,
                route,
            );
            let dynamic = dynamic_pipelines(&case.device);
            let plan = attention_dispatch_plan(&case.params);
            assert_specialized_selected(&case.pipelines, &case.params, plan.kind);
            let runner = PairedAttention::new(&case);
            let before = compare_outputs(&runner, &dynamic, plan);
            let pso = match route {
                AttentionDispatchKind::GroupedDecode => vec![
                    GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL,
                    GROUPED_DECODE_REDUCE_ATTENTION_KERNEL,
                ],
                AttentionDispatchKind::TiledPrefill => vec![TILED_PREFILL_ATTENTION_KERNEL],
                AttentionDispatchKind::GqaTiledPrefill => vec![GQA_TILED_PREFILL_ATTENTION_KERNEL],
                _ => unreachable!(),
            };
            let mut samples = Vec::new();
            for (phase, pairs) in [("warmup", 2), ("measured", 5)] {
                for pair in 0..pairs {
                    for candidate in if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let sample = runner.run_with_pipelines(
                            if candidate { &case.pipelines } else { &dynamic },
                            plan,
                            4,
                        );
                        samples.push(json!({"phase":phase,"pair":pair,"specialized":candidate,"sample":sample}));
                    }
                }
            }
            let after = compare_outputs(&runner, &dynamic, plan);
            case.assert_kv_unchanged("static head dimension timing");
            println!(
                "{}",
                json!({
                    "kind":"attention_static_head_dim_paired_microbench","device":case.device.name(),
                    "head_dim":dim,"query_heads":query_heads,"kv_heads":4,
                    "prefix_tokens":prefix,"query_tokens":tokens,"route":format!("{:?}",route),"pso":pso,
                    "precision":"existing F16 Q/K/V and SIMD operands, F32 accumulation/softmax, F16 output",
                    "scope":"prepared attention only; excludes all projections, RoPE, cache writes and scheduling; 8-query prefill is a tile microbenchmark, not full 512-query throughput",
                    "warmup_pairs":2,"measured_pairs":5,"dispatches_per_command":4,
                    "specialized_pipeline_initialization_ns":case.pipelines.specialized.iter().map(|p|p.initialization_ns).collect::<Vec<_>>(),
                    "numerics_before":before,"numerics_after":after,"samples":samples,
                })
            );
        });
    }
}

fn large_prefill_case(
    dim: usize,
    heads: usize,
    prefix: usize,
) -> (ValidatedPrefillCase, MetalCausalAttentionPipelines) {
    const TOKENS: usize = 512;
    // Build and independently validate the complete paged KV history once.
    // Repeating this already validated query row keeps CPU sampling affordable
    // while preserving the full 512-row launch and causal masking geometry.
    let mut case = run_prefill_cpu_case(
        "large prefill KV fixture",
        dim,
        heads,
        4,
        dim == 256,
        prefix + TOKENS - 1,
        1,
        AttentionDispatchKind::GroupedDecode,
    );
    let query = read_f16(&case.query, heads * dim)
        .into_iter()
        .map(f16::from_f32)
        .collect::<Vec<_>>()
        .repeat(TOKENS);
    let raw_features = heads * dim * if dim == 256 { 2 } else { 1 };
    let raw = read_f16(&case.query_raw, raw_features)
        .into_iter()
        .map(f16::from_f32)
        .collect::<Vec<_>>()
        .repeat(TOKENS);
    case.query = shared_buffer(&case.device, &query);
    case.query_raw = shared_buffer(&case.device, &raw);
    case.params.position_start = prefix as u32;
    case.params.tokens = TOKENS as u32;
    let plan = case.pipelines.dispatch_plan(&case.params);
    let dynamic = dynamic_pipelines(&case.device);
    case.expected = run_attention_plan(
        &case.device,
        &case.queue,
        &dynamic,
        &case.query,
        &case.query_raw,
        &case.pages,
        &case.params,
        plan,
    );
    for token in [0, 255, 511] {
        let expected = cpu_tiled_prefill_attention(
            &query[token * heads * dim..(token + 1) * heads * dim],
            &raw[token * raw_features..(token + 1) * raw_features],
            &case.state,
            prefix + token,
            1,
            heads,
            4,
            dim,
            dim == 256,
        );
        assert_close(
            "large prefill sampled CPU row",
            &case.expected[token * heads * dim..(token + 1) * heads * dim],
            &expected,
            0.001,
        );
    }
    case.reference_scope="full original dynamic output; independent CPU checks at query rows 0,255,511 plus complete small/tail CPU fixtures";
    (case, dynamic)
}

#[test]
#[ignore = "512-query attention GPU diagnostic; requires exclusive device access"]
fn attention_static_head_dim_512_query_paired_microbench() {
    for (dim, heads, prefix) in [(128, 32, 0), (256, 24, 0), (128, 32, 7680), (256, 24, 7680)] {
        autoreleasepool(|| {
            let (case, dynamic) = large_prefill_case(dim, heads, prefix);
            let plan = case.pipelines.dispatch_plan(&case.params);
            assert_specialized_selected(&case.pipelines, &case.params, plan.kind);
            let runner = PairedAttention::new(&case);
            let before = compare_outputs(&runner, &dynamic, plan);
            let mut samples = Vec::new();
            for (phase, pairs) in [("warmup", 2), ("measured", 5)] {
                for pair in 0..pairs {
                    for candidate in if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let sample = runner.run_with_pipelines(
                            if candidate { &case.pipelines } else { &dynamic },
                            plan,
                            1,
                        );
                        samples.push(json!({"phase":phase,"pair":pair,"specialized":candidate,"sample":sample}));
                    }
                }
            }
            let after = compare_outputs(&runner, &dynamic, plan);
            case.assert_kv_unchanged("large prefill final");
            println!(
                "{}",
                json!({"kind":"attention_static_head_dim_512_query_paired_microbench","device":case.device.name(),"head_dim":dim,"query_heads":heads,"kv_heads":4,"prefix_tokens":prefix,"query_tokens":512,"route":format!("{:?}",plan.kind),"dispatches_per_command":1,"warmup_pairs":2,"measured_pairs":5,"scope":"prepared attention only; fixed original dynamic PSOs versus production function constants; repeated synthetic query rows","reference_scope":case.reference_scope,"numerics_before":before,"numerics_after":after,"samples":samples})
            );
        });
    }
}
