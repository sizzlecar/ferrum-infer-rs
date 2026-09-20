//! Complete-output PQ2 production dispatch versus the fixed guarded kernel.
use super::*;
use crate::backend::metal::vnext_ops::native_blocks::supports_pq2_gemv_threadgroup;
use metal::CommandQueueRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    Guarded,
    Complete,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Guarded => "guarded",
            Self::Complete => "production_complete",
        }
    }
}

fn select(
    pipelines: &MetalLinearPipelines,
    params: LinearParams,
    output: ElementType,
    variant: Variant,
) -> &ComputePipelineState {
    if variant == Variant::Complete {
        let (pipeline, dispatch) =
            pipelines.hadamard_native_dispatch(GgufBlockFormat::Pq2_0, output, params);
        assert_eq!(dispatch, LinearDispatchKind::Pq2CooperativeGemv);
        return pipeline;
    }
    match output {
        ElementType::F16 => &pipelines.native.pq2_linear_f32_f16,
        ElementType::F32 => &pipelines.native.pq2_linear_f32,
        _ => unreachable!(),
    }
}

fn run(
    pipelines: &MetalLinearPipelines,
    fixture: &Fixture,
    output: ElementType,
    variant: Variant,
    queue: &CommandQueueRef,
    repetitions: usize,
) -> f64 {
    fixture.run_pipeline(
        select(pipelines, fixture.params(), output, variant),
        LinearDispatchKind::Pq2CooperativeGemv,
        queue,
        repetitions,
    )
}

fn complete_outputs(params: LinearParams) -> bool {
    pq2_complete_outputs_supported(params.rows, params.in_features, params.out_features)
}

#[test]
fn pq2_complete_decode_shape_and_capability_guard() {
    assert!(supports_pq2_gemv_threadgroup(32, 64, 0, 0));
    assert!(supports_pq2_gemv_threadgroup(32, 64, 128, 128));
    assert!(!supports_pq2_gemv_threadgroup(16, 64, 0, 32768));
    assert!(!supports_pq2_gemv_threadgroup(64, 64, 0, 32768));
    assert!(!supports_pq2_gemv_threadgroup(32, 63, 0, 32768));
    assert!(!supports_pq2_gemv_threadgroup(32, 64, 129, 128));
    assert!(!supports_pq2_gemv_threadgroup(32, 64, u64::MAX, 32768));
    let params = LinearParams {
        rows: 1,
        in_features: 384,
        out_features: 32,
        output_stride: 37,
        output_column_offset: 2,
    };
    assert!(complete_outputs(params));
    for (rows, width, outputs) in [
        (0, 384, 32),
        (1, 0, 32),
        (1, 127, 32),
        (1, 129, 32),
        (1, 384, 0),
        (1, 384, 1),
        (1, 384, 15),
        (1, 384, 17),
        (1, 384, 33),
    ] {
        assert!(!complete_outputs(LinearParams {
            rows,
            in_features: width,
            out_features: outputs,
            ..params
        }));
    }
}

#[test]
fn pq2_complete_decode_production_selector_keeps_other_routes() {
    let device = Device::system_default().expect("PQ2 dispatch conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let params = LinearParams {
        rows: 1,
        in_features: 384,
        out_features: 32,
        output_stride: 37,
        output_column_offset: 2,
    };
    for output in [ElementType::F16, ElementType::F32] {
        let complete = match output {
            ElementType::F16 => pipelines.native.pq2_linear_f32_f16_complete.as_ref(),
            ElementType::F32 => pipelines.native.pq2_linear_f32_complete.as_ref(),
            _ => unreachable!(),
        }
        .expect("complete-output PSO must be available on this Metal test device");
        assert!(std::ptr::eq(
            select(&pipelines, params, output, Variant::Complete),
            complete,
        ));
        for width in [0, 127, 129] {
            let invalid = LinearParams {
                in_features: width,
                ..params
            };
            assert!(std::ptr::eq(
                select(&pipelines, invalid, output, Variant::Complete),
                select(&pipelines, invalid, output, Variant::Guarded),
            ));
        }
        let (plain, _) = pipelines.plain_linear_dispatch(
            LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
            output,
            params,
        );
        assert!(!std::ptr::eq(plain, complete));
        let (other_format, _) =
            pipelines.hadamard_native_dispatch(GgufBlockFormat::Iq4Xs, output, params);
        assert!(!std::ptr::eq(other_format, complete));
        for (rows, outputs) in [(32, 32), (64, 4096)] {
            let prefill = LinearParams {
                rows,
                out_features: outputs,
                output_stride: outputs + 5,
                ..params
            };
            let (pipeline, dispatch) =
                pipelines.hadamard_native_dispatch(GgufBlockFormat::Pq2_0, output, prefill);
            assert!(!std::ptr::eq(pipeline, complete));
            assert_ne!(dispatch, LinearDispatchKind::Pq2CooperativeGemv);
        }
    }
}

#[test]
fn pq2_complete_decode_preserves_full_tiles_tails_offsets_and_f32_ranges() {
    let device = Device::system_default().expect("PQ2 address conformance requires Metal");
    let mut production = MetalLinearPipelines::new(&device).unwrap();
    let reference = ReferencePipelines::new(&device);
    let queue = device.new_command_queue();
    let mut check = |fixture: &Fixture, output, cpu| {
        fixture.poison_output();
        fixture.run(&production, &queue, true, 1);
        let baseline = fixture.read();
        assert!(baseline.iter().all(|x| x.is_finite()));
        if cpu {
            fixture.assert_cpu();
        }
        fixture.poison_output();
        fixture.run_pipeline(
            reference.get(output),
            LinearDispatchKind::Pq2CooperativeGemv,
            &queue,
            1,
        );
        assert_same_bits(&fixture.read(), &baseline);
        let pipeline = select(&production, fixture.params(), output, Variant::Complete);
        let control = select(&production, fixture.params(), output, Variant::Guarded);
        assert_eq!(
            std::ptr::eq(pipeline, control),
            !complete_outputs(fixture.params())
        );
        if complete_outputs(fixture.params()) {
            let expected = match output {
                ElementType::F16 => production.native.pq2_linear_f32_f16_complete.as_ref(),
                ElementType::F32 => production.native.pq2_linear_f32_complete.as_ref(),
                _ => unreachable!(),
            }
            .expect("complete-output PSO must be available on this Metal test device");
            assert!(std::ptr::eq(pipeline, expected));
        }
        fixture.poison_output();
        run(&production, fixture, output, Variant::Complete, &queue, 1);
        assert_same_bits(&fixture.read(), &baseline);
        if cpu {
            fixture.assert_cpu();
        }
        let saved = match output {
            ElementType::F16 => production.native.pq2_linear_f32_f16_complete.take(),
            ElementType::F32 => production.native.pq2_linear_f32_complete.take(),
            _ => unreachable!(),
        };
        assert!(std::ptr::eq(
            select(&production, fixture.params(), output, Variant::Complete),
            select(&production, fixture.params(), output, Variant::Guarded),
        ));
        fixture.poison_output();
        run(&production, fixture, output, Variant::Complete, &queue, 1);
        assert_same_bits(&fixture.read(), &baseline);
        fixture.assert_inputs_unchanged();
        match output {
            ElementType::F16 => production.native.pq2_linear_f32_f16_complete = saved,
            ElementType::F32 => production.native.pq2_linear_f32_complete = saved,
            _ => unreachable!(),
        }
        baseline
    };
    for output in [ElementType::F16, ElementType::F32] {
        // K384 has 102-byte packed row strides; fixture buffers start at input
        // byte16 / weight byte18 and output byte16, with N+5 stride / column2.
        for (rows, width, outputs) in [
            (1, 128, 16),
            (3, 384, 32),
            (31, 640, 48),
            (1, 128, 1),
            (3, 384, 17),
            (31, 640, 33),
        ] {
            check(
                &Fixture::new(&device, rows, width, outputs, output, true),
                output,
                true,
            );
        }
        for scale in [0x3555, 0xb955, 0x0001, 0, 0x8000, 0x7bff, 0xfbff] {
            let mut fixture = Fixture::new(&device, 1, 1024, 16, output, true);
            let divisor = if scale & 0x7fff == 0x7bff {
                65536.0
            } else {
                1.0
            };
            fixture.replace_payload(
                &device,
                (0..1024)
                    .map(|i| (i as f32 * 0.0171).sin() / divisor)
                    .collect(),
                weights(1024, 16, scale, |i| i as u8),
            );
            check(&fixture, output, true);
        }
        let mut zeros = Fixture::new(&device, 1, 128, 16, output, true);
        zeros.replace_payload(
            &device,
            (0..128).map(|i| (i as f32 * 1.37).cos()).collect(),
            weights(128, 16, 0x3555, |_| 0x55),
        );
        check(&zeros, output, true);
        zeros.replace_payload(
            &device,
            (0..128)
                .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                .collect(),
            weights(128, 16, 0xb955, |i| i as u8),
        );
        check(&zeros, output, true);
        let mut wide = Fixture::new(&device, 1, 384, 16, output, true);
        wide.replace_payload(
            &device,
            (0..384)
                .map(|i| if i % 2 == 0 { 131008.0 } else { -131008.0 })
                .collect(),
            weights(384, 16, 0x0001, |i| if i % 2 == 0 { 0xff } else { 0xe4 }),
        );
        check(&wide, output, true);
    }
    let mut extreme = Fixture::new(&device, 1, 384, 16, ElementType::F32, true);
    extreme.replace_payload(
        &device,
        (0..384)
            .map(|i| 2.0_f32.powi(125) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        weights(384, 16, 0x0001, |i| if i % 2 == 0 { 0xff } else { 0xe4 }),
    );
    check(&extreme, ElementType::F32, true);
    let mut subnormal = Fixture::new(&device, 1, 128, 16, ElementType::F32, true);
    subnormal.replace_payload(
        &device,
        (0..128).map(|i| f32::from_bits(1 + i)).collect(),
        weights(128, 16, 0x7bff, |_| 0xff),
    );
    // Preserve current subnormal treatment rather than imposing host F64 FTZ rules.
    check(&subnormal, ElementType::F32, false);
    let mut cancellation = Fixture::new(&device, 1, 640, 16, ElementType::F32, true);
    let mut input = vec![0.0; 640];
    input[..16].fill(2.0_f32.powi(108));
    input[512..528].fill(2.0_f32.powi(108));
    cancellation.replace_payload(
        &device,
        input,
        weights(
            640,
            16,
            0x7800,
            |byte| {
                if (byte / 32) % 5 == 4 {
                    0xff
                } else {
                    0
                }
            },
        ),
    );
    let actual = check(&cancellation, ElementType::F32, false);
    assert_eq!(actual[6], 2.0_f32.powi(127));
}

#[test]
#[ignore = "isolated Metal paired GPU timing; exclude compiler and other GPU work"]
fn pq2_complete_decode_isolated_gpu_timing() {
    const DISPATCHES: usize = 64;
    let device = Device::system_default().expect("PQ2 address timing requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for (width, outputs) in [(5120, 17408), (17408, 5120), (6144, 5120)] {
        for output in [ElementType::F16, ElementType::F32] {
            let mut fixture = Fixture::new(&device, 1, width, outputs, output, false);
            fixture.replace_payload(
                &device,
                (0..width)
                    .map(|i| ((i * 37 % 63) as i32 - 31) as f32 / 512.0)
                    .collect(),
                weights(width, outputs, 0x3000, |i| {
                    i.wrapping_mul(73).wrapping_add(i >> 8) as u8
                }),
            );
            let expected_pipeline = match output {
                ElementType::F16 => production.native.pq2_linear_f32_f16_complete.as_ref(),
                ElementType::F32 => production.native.pq2_linear_f32_complete.as_ref(),
                _ => unreachable!(),
            }
            .expect("timing requires the complete-output production pipeline");
            assert!(std::ptr::eq(
                select(&production, fixture.params(), output, Variant::Complete),
                expected_pipeline,
            ));
            assert!(!std::ptr::eq(
                select(&production, fixture.params(), output, Variant::Guarded),
                expected_pipeline,
            ));
            let run = |variant, repetitions| {
                fixture.poison_output();
                run(&production, &fixture, output, variant, &queue, repetitions)
            };
            run(Variant::Guarded, 1);
            let expected = fixture.read();
            assert!(expected.iter().all(|x| x.is_finite()));
            for candidate in [Variant::Complete] {
                run(candidate, 1);
                assert_same_bits(&fixture.read(), &expected);
                for _ in 0..2 {
                    run(Variant::Guarded, DISPATCHES);
                    run(candidate, DISPATCHES);
                }
                let mut control_us = Vec::new();
                let mut candidate_us = Vec::new();
                for pair in 0..5 {
                    for variant in if pair % 2 == 0 {
                        [Variant::Guarded, candidate]
                    } else {
                        [candidate, Variant::Guarded]
                    } {
                        let us = run(variant, DISPATCHES) * 1e6 / DISPATCHES as f64;
                        assert!(us.is_finite() && us > 0.0);
                        if variant == Variant::Guarded {
                            control_us.push(us);
                        } else {
                            candidate_us.push(us);
                        }
                    }
                }
                assert_same_bits(&fixture.read(), &expected);
                fixture.assert_inputs_unchanged();
                let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
                eprintln!(
                    "PQ2_COMPLETE_GPU {}",
                    serde_json::json!({
                        "rows": 1, "width": width, "outputs": outputs,
                        "output": format!("{output:?}"),
                        "control": Variant::Guarded.name(),
                        "candidate": candidate.name(), "timing": "gpu",
                        "warmups": 2, "pairs": 5, "dispatches": DISPATCHES,
                        "control_us": control_us, "candidate_us": candidate_us,
                        "ratio": mean(&candidate_us) / mean(&control_us),
                    })
                );
            }
        }
    }
}
