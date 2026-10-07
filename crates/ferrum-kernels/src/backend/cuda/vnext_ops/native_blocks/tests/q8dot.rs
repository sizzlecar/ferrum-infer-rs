//! Approximate activation-Q8/DP4A experiment, reachable only from these tests.
//! It is not the F32 production numerical contract or ggml's half-scale Q8_1.

use super::*;

mod fixture;
mod oracle;
mod q4k_reference;
mod q5k_reference;

use fixture::{Fixture, Route};

struct Experiment {
    quantize: CudaFunction,
    iq4_scalar: CudaFunction,
    iq4_tiled: CudaFunction,
    q4_scalar: CudaFunction,
    q4_tiled: CudaFunction,
    iq4_group32_scalar: CudaFunction,
    iq4_group32_tiled: CudaFunction,
    q4_group32_scalar: CudaFunction,
    q4_group32_tiled: CudaFunction,
    q5_group32_scalar: CudaFunction,
    q5_group32_tiled: CudaFunction,
}

impl Experiment {
    fn load(context: &Arc<CudaContext>) -> Self {
        use cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };
        let major = context
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap();
        let minor = context
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap();
        assert!(
            major > 6 || (major == 6 && minor >= 1),
            "experiment requires native DP4A support"
        );
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .unwrap();
        let load = |name| module.load_function(name).unwrap();
        Self {
            quantize: load("vnext_gguf_q8_f32scale_pack_f16_prototype"),
            iq4_scalar: load("vnext_gguf_iq4xs_q8_f32scale_dp4a_lane_f16_prototype"),
            iq4_tiled: load("vnext_gguf_iq4xs_q8_f32scale_dp4a_lane_tiled_f16_prototype"),
            q4_scalar: load("vnext_gguf_q4k_q8_f32scale_dp4a_lane_f16_prototype"),
            q4_tiled: load("vnext_gguf_q4k_q8_f32scale_dp4a_lane_tiled_f16_prototype"),
            iq4_group32_scalar: load("vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_f16_prototype"),
            iq4_group32_tiled: load(
                "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_tiled_f16_prototype",
            ),
            q4_group32_scalar: load("vnext_gguf_q4k_q8_f32scale_dp4a_group32_f16_prototype"),
            q4_group32_tiled: load("vnext_gguf_q4k_q8_f32scale_dp4a_group32_tiled_f16_prototype"),
            q5_group32_scalar: load("vnext_gguf_q5k_q8_f32scale_dp4a_group32_f16_prototype"),
            q5_group32_tiled: load("vnext_gguf_q5k_q8_f32scale_dp4a_group32_tiled_f16_prototype"),
        }
    }

    fn dot(
        &self,
        format: GgufBlockFormat,
        row_tile: u32,
        integer_partial_values: u32,
    ) -> &CudaFunction {
        match (format, row_tile, integer_partial_values) {
            (GgufBlockFormat::Iq4Xs, 1, 4) => &self.iq4_scalar,
            (GgufBlockFormat::Iq4Xs, 8, 4) => &self.iq4_tiled,
            (GgufBlockFormat::Q4K, 1, 4) => &self.q4_scalar,
            (GgufBlockFormat::Q4K, 8, 4) => &self.q4_tiled,
            (GgufBlockFormat::Iq4Xs, 1, 32) => &self.iq4_group32_scalar,
            (GgufBlockFormat::Iq4Xs, 8, 32) => &self.iq4_group32_tiled,
            (GgufBlockFormat::Q4K, 1, 32) => &self.q4_group32_scalar,
            (GgufBlockFormat::Q4K, 8, 32) => &self.q4_group32_tiled,
            (GgufBlockFormat::Q5K, 1, 32) => &self.q5_group32_scalar,
            (GgufBlockFormat::Q5K, 8, 32) => &self.q5_group32_tiled,
            _ => panic!("unsupported test-only activation-Q8 route"),
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn q8dot_activation_contract_and_dot_oracles_on_cuda() {
    let context = CudaContext::new(0).expect("activation-Q8 conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let experiment = Experiment::load(&context);
    let stream = context.default_stream();
    // Amax=127 produces exact d=1 and explicit signed half-integer ties.
    // Other blocks cover signed zero, F16 subnormal and finite endpoints.
    let blocks = [
        [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5],
        [0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
        [65504.0, -65504.0, 1.0, -1.0, 32.0, -32.0, 0.0, -0.0],
        [
            f16::from_bits(1).to_f32(),
            -f16::from_bits(1).to_f32(),
            f16::from_bits(0x03ff).to_f32(),
            -f16::from_bits(0x03ff).to_f32(),
            f16::from_bits(0x0400).to_f32(),
            -f16::from_bits(0x0400).to_f32(),
            0.0,
            -0.0,
        ],
    ];
    for (rows, inputs) in [(1, 32), (3, 96), (9, 256)] {
        let values = (0..rows * inputs)
            .map(|i| f16::from_f32(blocks[(i / 32) % blocks.len()][i % 8]))
            .collect::<Vec<_>>();
        Fixture::pack_probe(&stream, &experiment, rows, inputs, &values);
    }
    for format in [GgufBlockFormat::Iq4Xs, GgufBlockFormat::Q4K] {
        for (rows, inputs, outputs) in [(1, 256, 7), (8, 768, 17), (9, 512, 7), (8, 5120, 49)] {
            let mut fixture = Fixture::new(&stream, format, rows, inputs, outputs);
            for tile in [1, 8] {
                for route in Route::ALL {
                    let mut previous = None;
                    for _ in 0..2 {
                        fixture.run(&stream, &kernels, &experiment, route, tile, 1);
                        let (bits, errors) = fixture.validate(&stream, route);
                        if let Some(previous) = previous {
                            assert_eq!(
                                bits, previous,
                                "route {route:?} tile {tile} not repeatable"
                            );
                        }
                        previous = Some(bits);
                        println!(
                            "{}",
                            serde_json::json!({
                                "experiment":"activation_q8_f32_scale_conformance",
                                "format":format!("{format:?}"), "rows":rows,
                                "input":inputs, "output":outputs, "row_tile":tile,
                                "route":route.name(), "integer_partial_values":route.integer_partial_values(), "errors":errors
                            })
                        );
                    }
                }
            }
        }
        for (inputs, outputs) in [(768, 17), (5120, 49)] {
            Fixture::batch_row_equivalence(&stream, &kernels, &experiment, format, inputs, outputs);
        }
    }
    Fixture::edge_probes(&stream, &kernels, &experiment);
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn q8dot_activation_inclusive_paired_microbench() {
    let context = CudaContext::new(0).expect("activation-Q8 timing requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let experiment = Experiment::load(&context);
    let stream = context.default_stream();
    let routes = Route::ALL;
    const WARM_ROUNDS: usize = 4;
    const MEASURED_PAIRS: usize = 4;
    const ITERATIONS: u32 = 32;
    for format in [GgufBlockFormat::Iq4Xs, GgufBlockFormat::Q4K] {
        // Same physical projection shapes as the pinned dense GGUF inventory:
        // K5120/N6144, FFN up/gate K5120/N17408, down K17408/N5120.
        for (inputs, outputs) in [(5120, 6144), (5120, 17408), (17408, 5120)] {
            for rows in [1, 8] {
                let tile = if rows == 1 { 1 } else { 8 };
                let mut fixture = Fixture::new(&stream, format, rows, inputs, outputs);
                let mut stable = vec![None; routes.len()];
                for round in 0..WARM_ROUNDS + MEASURED_PAIRS {
                    for position in 0..routes.len() {
                        let index = (round + position) % routes.len();
                        let route = routes[index];
                        let (wall_ns, gpu_ns) =
                            fixture.run(&stream, &kernels, &experiment, route, tile, ITERATIONS);
                        let (bits, errors) = fixture.validate(&stream, route);
                        if let Some(previous) = &stable[index] {
                            assert_eq!(&bits, previous, "timed route {route:?} changed bits");
                        } else {
                            stable[index] = Some(bits);
                        }
                        if round >= WARM_ROUNDS {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "experiment":"activation_q8_f32_scale_paired",
                                    "format":format!("{format:?}"), "rows":rows,
                                    "input":inputs, "output":outputs, "row_tile":tile,
                                    "pair":round-WARM_ROUNDS, "order":position, "route":route.name(),
                                    "warm_rounds":WARM_ROUNDS, "measured_pairs":MEASURED_PAIRS,
                                    "iterations":ITERATIONS, "wall_ns":wall_ns, "gpu_ns":gpu_ns,
                                    "quantization_included":route.includes_quantization(),
                                    "integer_partial_values":route.integer_partial_values(),
                                    "errors":errors
                                })
                            );
                        }
                    }
                }
            }
        }
    }
}
