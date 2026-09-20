//! Test-only output tiling and independent-sum interleaving experiments.
//! Production selection remains unchanged.
use super::*;
use metal::CommandQueueRef;

struct OutputPipelines {
    complete: ComputePipelineState,
    guarded: Option<ComputePipelineState>,
}

struct TilePipelines {
    f16: OutputPipelines,
    f32: OutputPipelines,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeCandidate {
    Four,
    Sixteen,
    InterleavedEight,
}

impl DecodeCandidate {
    const fn outputs_per_simd(self) -> u32 {
        match self {
            Self::Four => 4,
            Self::Sixteen => 16,
            Self::InterleavedEight => 8,
        }
    }

    const fn outputs_per_group(self) -> u32 {
        2 * self.outputs_per_simd()
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Four => "four",
            Self::Sixteen => "sixteen",
            Self::InterleavedEight => "interleaved_eight",
        }
    }

    fn supports(self, params: LinearParams) -> bool {
        params.rows > 0
            && params.in_features > 0
            && params.in_features.is_multiple_of(128)
            && params.out_features > 0
            && (self != Self::InterleavedEight || params.out_features.is_multiple_of(16))
    }
}

struct CandidatePipelines {
    four: TilePipelines,
    sixteen: TilePipelines,
    interleaved_eight: TilePipelines,
}

impl CandidatePipelines {
    fn new(device: &Device) -> Self {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(include_str!("output_tiles.metal"), &options)
            .unwrap();
        let interleaved_library = device
            .new_library_with_source(include_str!("interleaved_eight.metal"), &options)
            .unwrap();
        let pipeline = |candidate: DecodeCandidate, suffix| {
            let (library, name) = if candidate == DecodeCandidate::InterleavedEight {
                (
                    &interleaved_library,
                    format!("pq2_interleaved_eight_{suffix}"),
                )
            } else {
                (
                    &library,
                    format!("pq2_output_tile_{}_{suffix}", candidate.name()),
                )
            };
            let pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library.get_function(&name, None).unwrap(),
                )
                .unwrap();
            assert_eq!(pipeline.thread_execution_width(), 32);
            assert!(pipeline.max_total_threads_per_threadgroup() >= 64);
            pipeline
        };
        let tile_pipelines = |tile: DecodeCandidate| TilePipelines {
            f16: OutputPipelines {
                complete: pipeline(tile, "f16_complete"),
                guarded: (tile != DecodeCandidate::InterleavedEight)
                    .then(|| pipeline(tile, "f16_guarded")),
            },
            f32: OutputPipelines {
                complete: pipeline(tile, "f32_complete"),
                guarded: (tile != DecodeCandidate::InterleavedEight)
                    .then(|| pipeline(tile, "f32_guarded")),
            },
        };
        Self {
            four: tile_pipelines(DecodeCandidate::Four),
            sixteen: tile_pipelines(DecodeCandidate::Sixteen),
            interleaved_eight: tile_pipelines(DecodeCandidate::InterleavedEight),
        }
    }

    fn get(
        &self,
        tile: DecodeCandidate,
        output: ElementType,
        params: LinearParams,
    ) -> Option<&ComputePipelineState> {
        if !tile.supports(params) {
            return None;
        }
        let tile_pipelines = match tile {
            DecodeCandidate::Four => &self.four,
            DecodeCandidate::Sixteen => &self.sixteen,
            DecodeCandidate::InterleavedEight => &self.interleaved_eight,
        };
        let pipelines = match output {
            ElementType::F16 => &tile_pipelines.f16,
            ElementType::F32 => &tile_pipelines.f32,
            _ => unreachable!(),
        };
        if params.out_features.is_multiple_of(tile.outputs_per_group()) {
            Some(&pipelines.complete)
        } else {
            pipelines.guarded.as_ref()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    ProductionEight,
    Candidate(DecodeCandidate),
}

impl Variant {
    const fn outputs_per_group(self) -> u64 {
        match self {
            Self::ProductionEight => 16,
            Self::Candidate(tile) => tile.outputs_per_group() as u64,
        }
    }
}

fn run(
    production: &MetalLinearPipelines,
    candidate: &CandidatePipelines,
    fixture: &Fixture,
    output: ElementType,
    variant: Variant,
    queue: &CommandQueueRef,
    repetitions: usize,
) -> f64 {
    let pipeline = match variant {
        Variant::ProductionEight => {
            let (pipeline, dispatch) = production.hadamard_native_dispatch(
                GgufBlockFormat::Pq2_0,
                output,
                fixture.params(),
            );
            // Do not accidentally compare a prefill GEMM with this GEMV experiment.
            assert_eq!(dispatch, LinearDispatchKind::Pq2CooperativeGemv);
            pipeline
        }
        Variant::Candidate(tile) => candidate
            .get(tile, output, fixture.params())
            .expect("candidate must support this shape before dispatch"),
    };
    fixture.poison_output();
    fixture.run_pipeline_with_dispatch(pipeline, queue, repetitions, |encoder, params| {
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.out_features).div_ceil(variant.outputs_per_group()),
                u64::from(params.rows),
                1,
            ),
            MTLSize::new(32, 2, 1),
        );
    })
}

fn check_candidates(
    production: &MetalLinearPipelines,
    candidate: &CandidatePipelines,
    fixture: &Fixture,
    output: ElementType,
    queue: &CommandQueueRef,
    cpu: bool,
    kinds: &[DecodeCandidate],
) -> Vec<f32> {
    run(
        production,
        candidate,
        fixture,
        output,
        Variant::ProductionEight,
        queue,
        1,
    );
    let expected = fixture.read();
    assert!(expected.iter().all(|value| value.is_finite()));
    if cpu {
        fixture.assert_cpu();
    }
    for &kind in kinds {
        if candidate.get(kind, output, fixture.params()).is_none() {
            // Interleaving has no guarded kernel. Never disguise production or
            // another candidate as a successful interleaved tail dispatch.
            let params = fixture.params();
            assert_eq!(kind, DecodeCandidate::InterleavedEight);
            assert!(params.rows > 0 && params.in_features > 0);
            assert!(params.in_features.is_multiple_of(128));
            assert!(params.out_features > 0 && !params.out_features.is_multiple_of(16));
            eprintln!(
                "PQ2_DECODE_CANDIDATE_SKIP candidate={kind:?} M={} K={} N={} reason=requires_complete_N16",
                params.rows, params.in_features, params.out_features,
            );
            continue;
        }
        run(
            production,
            candidate,
            fixture,
            output,
            Variant::Candidate(kind),
            queue,
            1,
        );
        assert_same_bits(&fixture.read(), &expected);
        fixture.assert_inputs_unchanged();
    }
    expected
}

#[test]
fn pq2_output_tile_decode_preserves_tails_offsets_and_f32_ranges() {
    check_ranges(&[DecodeCandidate::Four, DecodeCandidate::Sixteen]);
}

#[test]
fn pq2_interleaved_eight_decode_preserves_complete_tiles_offsets_and_f32_ranges() {
    check_ranges(&[DecodeCandidate::InterleavedEight]);
}

fn check_ranges(kinds: &[DecodeCandidate]) {
    let device = Device::system_default().expect("PQ2 output-tile conformance requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let candidate = CandidatePipelines::new(&device);
    let queue = device.new_command_queue();
    let check = |fixture: &Fixture, output, cpu| {
        check_candidates(&production, &candidate, fixture, output, &queue, cpu, kinds)
    };
    for output in [ElementType::F16, ElementType::F32] {
        // Exercise both sides of the 4/8/16-output SIMD and 8/16/32-output group
        // boundaries, including shapes complete only for smaller tiles. Fixture
        // bindings start at input byte16 / weight byte18 / output byte16 and
        // retain a nonzero output column offset, stride padding, and canaries.
        for outputs in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33] {
            let fixture = Fixture::new(&device, 3, 384, outputs, output, true);
            check(&fixture, output, true);
        }
        for (rows, width, outputs) in [
            (1, 128, 16),
            (4, 640, 24),
            (4, 640, 32),
            (31, 640, 32),
            (31, 640, 33),
        ] {
            let fixture = Fixture::new(&device, rows, width, outputs, output, true);
            check(&fixture, output, true);
        }
        // All packed byte values and half scales, without rounding F32 inputs.
        for scale in [0x3555, 0xb955, 0x0001, 0, 0x8000, 0x7bff, 0xfbff] {
            let mut fixture = Fixture::new(&device, 1, 1024, 32, output, true);
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
                weights(1024, 32, scale, |i| i as u8),
            );
            check(&fixture, output, true);
        }
        for (scale, zero_inputs) in [(0x3555, false), (0xb955, true)] {
            for outputs in [16, 17] {
                let mut fixture = Fixture::new(&device, 1, 128, outputs, output, true);
                fixture.replace_payload(
                    &device,
                    (0..128)
                        .map(|i| {
                            if zero_inputs {
                                if i % 2 == 0 {
                                    0.0
                                } else {
                                    -0.0
                                }
                            } else {
                                (i as f32 * 1.37).cos()
                            }
                        })
                        .collect(),
                    weights(128, outputs, scale, |i| {
                        if zero_inputs {
                            i as u8
                        } else {
                            0x55
                        }
                    }),
                );
                check(&fixture, output, true);
            }
        }
    }
    for magnitude in [131008.0, 2.0_f32.powi(125)] {
        let mut fixture = Fixture::new(&device, 1, 384, 32, ElementType::F32, true);
        fixture.replace_payload(
            &device,
            (0..384)
                .map(|i| if i % 2 == 0 { magnitude } else { -magnitude })
                .collect(),
            weights(384, 32, 0x0001, |i| if i % 2 == 0 { 0xff } else { 0xe4 }),
        );
        check(&fixture, ElementType::F32, true);
    }
    for outputs in [16, 17] {
        let mut subnormal = Fixture::new(&device, 1, 128, outputs, ElementType::F32, true);
        subnormal.replace_payload(
            &device,
            (0..128).map(|i| f32::from_bits(1 + i)).collect(),
            weights(128, outputs, 0x7bff, |_| 0xff),
        );
        // Compare existing GPU FTZ behavior rather than imposing host F64 FTZ rules.
        check(&subnormal, ElementType::F32, false);
    }
    let mut cancellation = Fixture::new(&device, 1, 640, 32, ElementType::F32, true);
    let mut input = vec![0.0; 640];
    input[..16].fill(2.0_f32.powi(108));
    input[512..528].fill(2.0_f32.powi(108));
    cancellation.replace_payload(
        &device,
        input,
        weights(
            640,
            32,
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

/// These are workload samples, never production eligibility conditions.
const PROJECTION_SAMPLES: [(&str, u32, u32); 4] = [
    ("ffn_gate_or_up", 5120, 17408),
    ("ffn_down", 17408, 5120),
    ("gdn_qkv", 5120, 10240),
    ("gdn_output", 6144, 5120),
];

fn projection_fixture(
    device: &Device,
    rows: u32,
    width: u32,
    outputs: u32,
    output: ElementType,
) -> Fixture {
    let mut fixture = Fixture::new(device, rows, width, outputs, output, false);
    fixture.replace_payload(
        device,
        (0..rows * width)
            .map(|i| ((i * 37 % 63) as i32 - 31) as f32 / 512.0)
            .collect(),
        weights(width, outputs, 0x3000, |i| {
            i.wrapping_mul(73).wrapping_add(i >> 8) as u8
        }),
    );
    fixture
}

#[test]
fn pq2_output_tile_decode_matches_f64_on_projection_shapes() {
    check_projection_shapes(&[DecodeCandidate::Four, DecodeCandidate::Sixteen]);
}

#[test]
fn pq2_interleaved_eight_decode_matches_f64_on_projection_shapes() {
    check_projection_shapes(&[DecodeCandidate::InterleavedEight]);
}

fn check_projection_shapes(kinds: &[DecodeCandidate]) {
    let device = Device::system_default().expect("PQ2 projection conformance requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let candidate = CandidatePipelines::new(&device);
    let queue = device.new_command_queue();
    for (_, width, outputs) in PROJECTION_SAMPLES {
        for output in [ElementType::F16, ElementType::F32] {
            let fixture = projection_fixture(&device, 1, width, outputs, output);
            check_candidates(
                &production,
                &candidate,
                &fixture,
                output,
                &queue,
                true,
                kinds,
            );
        }
    }
}

#[test]
#[ignore = "isolated Metal paired GPU timing; exclude compiler and other GPU work"]
fn pq2_sixteen_output_decode_isolated_gpu_timing() {
    paired_gpu_timing(DecodeCandidate::Sixteen);
}

#[test]
#[ignore = "isolated Metal paired GPU timing; exclude compiler and other GPU work"]
fn pq2_interleaved_eight_decode_isolated_gpu_timing() {
    paired_gpu_timing(DecodeCandidate::InterleavedEight);
}

fn pipeline_properties(pipeline: &ComputePipelineState) -> serde_json::Value {
    serde_json::json!({
        "thread_execution_width": pipeline.thread_execution_width(),
        "max_total_threads_per_threadgroup": pipeline.max_total_threads_per_threadgroup(),
        "static_threadgroup_memory_length": pipeline.static_threadgroup_memory_length(),
    })
}

fn paired_gpu_timing(timed_candidate: DecodeCandidate) {
    const DISPATCHES: usize = 64;
    const WARMUPS: usize = 2;
    const PAIRS: usize = 5;
    let device = Device::system_default().expect("PQ2 output tiling timing requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let candidate = CandidatePipelines::new(&device);
    let queue = device.new_command_queue();
    let timed_variant = Variant::Candidate(timed_candidate);
    for rows in [1, 4] {
        for (shape, width, outputs) in PROJECTION_SAMPLES {
            for output in [ElementType::F16, ElementType::F32] {
                let fixture = projection_fixture(&device, rows, width, outputs, output);
                // Full F64 projection checks live in the non-ignored test. This
                // independently runnable timing test checks every output bit,
                // canary and immutable input against production before and after.
                // Dispatch only the requested candidate in this experiment.
                let expected = check_candidates(
                    &production,
                    &candidate,
                    &fixture,
                    output,
                    &queue,
                    false,
                    &[timed_candidate],
                );
                let (production_pipeline, dispatch) = production.hadamard_native_dispatch(
                    GgufBlockFormat::Pq2_0,
                    output,
                    fixture.params(),
                );
                assert_eq!(dispatch, LinearDispatchKind::Pq2CooperativeGemv);
                let candidate_pipeline = candidate
                    .get(timed_candidate, output, fixture.params())
                    .expect("timing shape must support the requested candidate");
                let run = |variant| {
                    run(
                        &production,
                        &candidate,
                        &fixture,
                        output,
                        variant,
                        &queue,
                        DISPATCHES,
                    ) * 1e6
                        / DISPATCHES as f64
                };
                for _ in 0..WARMUPS {
                    run(Variant::ProductionEight);
                    run(timed_variant);
                }
                let mut samples = Vec::new();
                for pair in 0..PAIRS {
                    let order = if pair % 2 == 0 {
                        [Variant::ProductionEight, timed_variant]
                    } else {
                        [timed_variant, Variant::ProductionEight]
                    };
                    let mut production_us = 0.0;
                    let mut candidate_us = 0.0;
                    for variant in order {
                        let us = run(variant);
                        assert!(us.is_finite() && us > 0.0);
                        assert_same_bits(&fixture.read(), &expected);
                        match variant {
                            Variant::ProductionEight => production_us = us,
                            Variant::Candidate(_) => candidate_us = us,
                        }
                    }
                    samples.push(serde_json::json!({
                        "pair": pair,
                        "order": order.map(|variant| format!("{variant:?}")),
                        "production_gpu_us": production_us,
                        "candidate_gpu_us": candidate_us,
                        "candidate_over_production": candidate_us / production_us,
                    }));
                }
                fixture.assert_inputs_unchanged();
                eprintln!(
                    "PQ2_OUTPUT_TILE_GPU {}",
                    serde_json::json!({
                        "schema_version": 1, "device": device.name(),
                        "kind": "pq2_output_tile_decode_paired_gpu_microbench",
                        "rows": rows, "shape": shape, "in_features": width,
                        "out_features": outputs, "output_dtype": format!("{output:?}"),
                        "input_dtype": "f32", "operand_dtype": "f32",
                        "accumulator_dtype": "f32", "weight_format": "quantization.gguf.pq2-0",
                        "production_outputs_per_simd": 8,
                        "candidate": timed_candidate.name(),
                        "candidate_outputs_per_simd": timed_candidate.outputs_per_simd(),
                        "production_outputs_per_group": Variant::ProductionEight.outputs_per_group(),
                        "candidate_outputs_per_group": timed_variant.outputs_per_group(),
                        "threads_per_group": 64, "output_stride": fixture.params().output_stride,
                        "production_pipeline": pipeline_properties(production_pipeline),
                        "candidate_pipeline": pipeline_properties(candidate_pipeline),
                        "output_column_offset": fixture.params().output_column_offset,
                        "input_offset_bytes": 16, "weight_offset_bytes": 18,
                        "output_offset_bytes": 16, "warmup_pairs": WARMUPS,
                        "pairs": PAIRS, "dispatches_per_sample": DISPATCHES,
                        "working_set": "one_resident_matrix_per_shape",
                        "timing_scope": "gpu_gemv_without_hadamard_or_upload_or_readback",
                        "validation": "all_output_bits_and_guards_vs_production_before_and_after",
                        "production_selection_changed": false, "samples": samples,
                    })
                );
            }
        }
    }
}
