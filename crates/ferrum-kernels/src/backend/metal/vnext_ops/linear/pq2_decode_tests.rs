//! Production PQ2 field decoding versus the original direct-bit kernel and F64 dot.
use super::pq2_tests::Fixture;
use super::*;

mod complete_outputs;
mod output_tiles;

struct ReferencePipelines {
    f16: ComputePipelineState,
    f32: ComputePipelineState,
}

impl ReferencePipelines {
    fn new(device: &Device) -> Self {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(include_str!("pq2_decode_reference.metal"), &options)
            .unwrap();
        let pipeline = |name| {
            device
                .new_compute_pipeline_state_with_function(
                    &library.get_function(name, None).unwrap(),
                )
                .unwrap()
        };
        Self {
            f16: pipeline("pq2_direct_bits_f16"),
            f32: pipeline("pq2_direct_bits_f32"),
        }
    }

    fn get(&self, output: ElementType) -> &ComputePipelineState {
        match output {
            ElementType::F16 => &self.f16,
            ElementType::F32 => &self.f32,
            _ => unreachable!(),
        }
    }
}

fn weights(width: u32, outputs: u32, scale: u16, packed: impl Fn(u32) -> u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    for block in 0..outputs * (width / 128) {
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend((0..32).map(|byte| packed(block * 32 + byte)));
    }
    bytes
}

fn assert_same_bits(actual: &[f32], reference: &[f32]) {
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn pq2_float_floor_decoding_preserves_reference_tails_codes_and_f32_ranges() {
    let device = Device::system_default().expect("PQ2 decoding conformance requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let reference = ReferencePipelines::new(&device);
    let queue = device.new_command_queue();
    for output in [ElementType::F16, ElementType::F32] {
        for (rows, width, outputs) in [(1, 128, 1), (3, 384, 17), (31, 640, 33)] {
            let fixture = Fixture::new(&device, rows, width, outputs, output, true);
            fixture.run(&production, &queue, true, 1);
            let baseline = fixture.read();
            fixture.run_pipeline(
                reference.get(output),
                LinearDispatchKind::Pq2CooperativeGemv,
                &queue,
                1,
            );
            fixture.assert_cpu();
            assert_same_bits(&fixture.read(), &baseline);
        }
        // All 256 packed bytes, non-dyadic inputs, positive and negative scales.
        for scale in [0x3555, 0xb955, 0x0001, 0, 0x8000] {
            let mut fixture = Fixture::new(&device, 1, 1024, 17, output, true);
            fixture.replace_payload(
                &device,
                (0..1024).map(|i| (i as f32 * 0.0171).sin()).collect(),
                weights(1024, 17, scale, |i| i as u8),
            );
            fixture.run(&production, &queue, true, 1);
            let baseline = fixture.read();
            fixture.run_pipeline(
                reference.get(output),
                LinearDispatchKind::Pq2CooperativeGemv,
                &queue,
                1,
            );
            fixture.assert_cpu();
            assert_same_bits(&fixture.read(), &baseline);
        }
        // Every zero-coded weight must remain exactly zero even with non-dyadic
        // activation inputs. The existing forward bound is zero in this case.
        let mut zeros = Fixture::new(&device, 1, 128, 17, output, true);
        zeros.replace_payload(
            &device,
            (0..128).map(|i| (i as f32 * 1.37).cos()).collect(),
            weights(128, 17, 0x3555, |_| 0x55),
        );
        zeros.run(&production, &queue, true, 1);
        let baseline = zeros.read();
        zeros.run_pipeline(
            reference.get(output),
            LinearDispatchKind::Pq2CooperativeGemv,
            &queue,
            1,
        );
        zeros.assert_cpu();
        assert_same_bits(&zeros.read(), &baseline);
        // Preserve signed zero through scale products and SIMD reduction too.
        zeros.replace_payload(
            &device,
            (0..128)
                .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                .collect(),
            weights(128, 17, 0xb955, |i| i as u8),
        );
        zeros.run(&production, &queue, true, 1);
        let baseline = zeros.read();
        zeros.run_pipeline(
            reference.get(output),
            LinearDispatchKind::Pq2CooperativeGemv,
            &queue,
            1,
        );
        zeros.assert_cpu();
        assert_same_bits(&zeros.read(), &baseline);
    }
    // F32 input must stay wide: beyond half range, and finite products for
    // extreme inputs with the smallest half scale. Compare every output and
    // canary to the original arithmetic order.
    for (magnitude, scale) in [(131008.0, 0x0001), (2.0_f32.powi(125), 0x0001)] {
        let mut fixture = Fixture::new(&device, 1, 384, 17, ElementType::F32, true);
        fixture.replace_payload(
            &device,
            (0..384)
                .map(|i| if i % 2 == 0 { magnitude } else { -magnitude })
                .collect(),
            weights(384, 17, scale, |i| if i % 2 == 0 { 0xff } else { 0xe4 }),
        );
        fixture.run(&production, &queue, true, 1);
        let baseline = fixture.read();
        fixture.assert_cpu();
        fixture.run_pipeline(
            reference.get(ElementType::F32),
            LinearDispatchKind::Pq2CooperativeGemv,
            &queue,
            1,
        );
        fixture.assert_cpu();
        assert_same_bits(&fixture.read(), &baseline);
    }
    // Preserve the existing treatment of F32 subnormal activation inputs too.
    let mut subnormal = Fixture::new(&device, 1, 128, 17, ElementType::F32, true);
    subnormal.replace_payload(
        &device,
        (0..128).map(|i| f32::from_bits(1 + i)).collect(),
        weights(128, 17, 0x7bff, |_| 0xff),
    );
    subnormal.run(&production, &queue, true, 1);
    let baseline = subnormal.read();
    subnormal.run_pipeline(
        reference.get(ElementType::F32),
        LinearDispatchKind::Pq2CooperativeGemv,
        &queue,
        1,
    );
    assert_same_bits(&subnormal.read(), &baseline);

    // A scaled span can overflow even though each old product and every
    // running sum remain finite: -2^127 followed by sixteen +2^124 terms.
    // Blocks 0 and 4 belong to the same SIMD lane; other lanes read zero.
    let mut cancellation = Fixture::new(&device, 1, 640, 17, ElementType::F32, true);
    let mut input = vec![0.0; 640];
    input[..16].fill(2.0_f32.powi(108));
    input[512..528].fill(2.0_f32.powi(108));
    cancellation.replace_payload(
        &device,
        input,
        weights(
            640,
            17,
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
    cancellation.run(&production, &queue, true, 1);
    let baseline = cancellation.read();
    assert!(baseline.iter().all(|value| value.is_finite()));
    assert_eq!(baseline[6], 2.0_f32.powi(127));
    cancellation.run_pipeline(
        reference.get(ElementType::F32),
        LinearDispatchKind::Pq2CooperativeGemv,
        &queue,
        1,
    );
    assert_same_bits(&cancellation.read(), &baseline);
}

#[test]
#[ignore = "isolated Metal GPU timestamps; exclude other GPU and compiler work"]
fn pq2_float_floor_decoding_isolated_gpu_timing() {
    let device = Device::system_default().expect("PQ2 decoding timing requires Metal");
    let production = MetalLinearPipelines::new(&device).unwrap();
    let reference = ReferencePipelines::new(&device);
    let queue = device.new_command_queue();
    for (width, outputs) in [(5120, 17408), (17408, 5120), (6144, 5120)] {
        for output in [ElementType::F16, ElementType::F32] {
            let mut fixture = Fixture::new(&device, 1, width, outputs, output, false);
            // Cover all packed byte values, with a deterministic phase shift
            // across 256-byte groups. Both decoders receive identical buffers.
            fixture.replace_payload(
                &device,
                (0..width)
                    .map(|i| ((i * 37 % 63) as i32 - 31) as f32 / 512.0)
                    .collect(),
                weights(width, outputs, 0x3000, |i| {
                    i.wrapping_mul(73).wrapping_add(i >> 8) as u8
                }),
            );
            let run = |new, repetitions| {
                if new {
                    fixture.run(&production, &queue, true, repetitions)
                } else {
                    fixture.run_pipeline(
                        reference.get(output),
                        LinearDispatchKind::Pq2CooperativeGemv,
                        &queue,
                        repetitions,
                    )
                }
            };
            run(false, 1);
            let expected = fixture.read();
            run(true, 1);
            assert_same_bits(&fixture.read(), &expected);
            for _ in 0..2 {
                run(false, 8);
                run(true, 8);
            }
            let mut old_us = Vec::new();
            let mut new_us = Vec::new();
            for sample in 0..10 {
                for new in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let us = run(new, 8) * 1e6 / 8.0;
                    assert!(us.is_finite() && us > 0.0);
                    if new {
                        new_us.push(us);
                    } else {
                        old_us.push(us);
                    }
                }
            }
            let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
            eprintln!("PQ2_FLOOR_GPU K={width} N={outputs} output={output:?} old_us={old_us:?} new_us={new_us:?} ratio={:.6}", mean(&new_us) / mean(&old_us));
        }
    }
}
