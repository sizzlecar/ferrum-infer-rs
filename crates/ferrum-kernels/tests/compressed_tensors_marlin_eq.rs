//! CUDA parity fixtures for the exact compressed-tensors W4A16 subset used by
//! the locked Qwen3.8 checkpoint.
//!
//! Run on an sm89 CUDA host with the versioned native-operator lock:
//! `cargo test -p ferrum-kernels --features vllm-marlin --release \
//!   --test compressed_tensors_marlin_eq -- --ignored --nocapture --test-threads=1`

#![cfg(all(feature = "cuda", feature = "vllm-marlin"))]

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_kernels::marlin_repack::{
    repack_compressed_tensors_zero_points_to_marlin, repack_gptq_to_marlin, repack_scales_to_marlin,
};
use ferrum_kernels::vllm_marlin::{
    launch_marlin_mm_f16_weight, MarlinF16WeightType, MarlinMmBuffers, MarlinMmExecution,
    MarlinMmF16WeightRequest, MarlinMmProblem,
};
use half::f16;
use std::os::raw::c_void;

#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    rows: usize,
    input_features: usize,
    output_features: usize,
}

const GROUP_SIZE: usize = 32;

const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "asymmetric-packing",
        rows: 3,
        input_features: 128,
        output_features: 64,
    },
    Fixture {
        name: "single-projection",
        rows: 1,
        input_features: 256,
        output_features: 128,
    },
    Fixture {
        name: "fused-qkv-gate-up",
        rows: 2,
        input_features: 256,
        output_features: 256,
    },
    Fixture {
        name: "mixed-dense-linear-attention-segment",
        rows: 4,
        input_features: 128,
        output_features: 192,
    },
];

struct HostFixture {
    input: Vec<f16>,
    qweight_gptq: Vec<i32>,
    scales_grouped: Vec<f16>,
    zero_points_compressed_tensors: Vec<i32>,
    logical_weights: Vec<u8>,
    logical_scales: Vec<f16>,
    logical_zero_points: Vec<u8>,
}

fn build_host_fixture(fixture: Fixture) -> HostFixture {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let groups = k / GROUP_SIZE;
    assert!(k.is_multiple_of(128));
    assert!(n.is_multiple_of(64));

    let input = (0..m * k)
        .map(|index| {
            let centered = ((index * 29 + m * 17) % 257) as f32 - 128.0;
            f16::from_f32(centered / 113.0)
        })
        .collect::<Vec<_>>();
    let logical_weights = (0..k * n)
        .map(|index| ((index * 7 + index / 11 + n / 64) & 0x0f) as u8)
        .collect::<Vec<_>>();
    let logical_scales = (0..groups * n)
        .map(|index| f16::from_f32(0.0125 + ((index * 13 % 17) as f32) * 0.00075))
        .collect::<Vec<_>>();
    let logical_zero_points = (0..groups * n)
        .map(|index| ((index * 5 + index / 9 + 3) & 0x0f) as u8)
        .collect::<Vec<_>>();

    // Marlin weight repacking consumes GPTQ's `[K / 8, N]` word layout.
    // The production source adapter reaches this layout by transposing the
    // checkpoint's compressed-tensors `[N, K / 8]` storage.
    let mut qweight_gptq = vec![0_i32; (k / 8) * n];
    for packed_input in 0..k / 8 {
        for output in 0..n {
            qweight_gptq[packed_input * n + output] = (0..8).fold(0_u32, |word, lane| {
                word | (u32::from(logical_weights[(packed_input * 8 + lane) * n + output])
                    << (lane * 4))
            }) as i32;
        }
    }

    // compressed-tensors packs zero points as `[N / 8, K / G]` with
    // little-endian output-channel nibbles.
    let mut zero_points_compressed_tensors = vec![0_i32; (n / 8) * groups];
    for packed_output in 0..n / 8 {
        for group in 0..groups {
            zero_points_compressed_tensors[packed_output * groups + group] =
                (0..8).fold(0_u32, |word, lane| {
                    word | (u32::from(logical_zero_points[group * n + packed_output * 8 + lane])
                        << (lane * 4))
                }) as i32;
        }
    }

    HostFixture {
        input,
        qweight_gptq,
        scales_grouped: logical_scales.clone(),
        zero_points_compressed_tensors,
        logical_weights,
        logical_scales,
        logical_zero_points,
    }
}

fn cpu_reference(fixture: Fixture, host: &HostFixture) -> Vec<f16> {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let mut output = vec![f16::ZERO; m * n];
    for row in 0..m {
        for output_feature in 0..n {
            let mut sum = 0.0_f32;
            for input_feature in 0..k {
                let group = input_feature / GROUP_SIZE;
                let quantized = host.logical_weights[input_feature * n + output_feature] as i32;
                let zero_point = host.logical_zero_points[group * n + output_feature] as i32;
                let scale = host.logical_scales[group * n + output_feature].to_f32();
                sum += host.input[row * k + input_feature].to_f32()
                    * (quantized - zero_point) as f32
                    * scale;
            }
            output[row * n + output_feature] = f16::from_f32(sum);
        }
    }
    output
}

fn run_fixture(context: &std::sync::Arc<CudaContext>, fixture: Fixture) -> (f64, usize) {
    let stream = context.default_stream();
    let host = build_host_fixture(fixture);
    let reference = cpu_reference(fixture, &host);
    let k = fixture.input_features;
    let n = fixture.output_features;
    let groups = k / GROUP_SIZE;

    let packed_weight = repack_gptq_to_marlin(&host.qweight_gptq, k, n);
    let packed_scales = repack_scales_to_marlin(&host.scales_grouped, k, n, GROUP_SIZE);
    let packed_zero_points = repack_compressed_tensors_zero_points_to_marlin(
        &host.zero_points_compressed_tensors,
        groups,
        n,
    );

    let input_device: CudaSlice<f16> = stream.clone_htod(&host.input).expect("upload input");
    let weight_device: CudaSlice<i32> = stream
        .clone_htod(&packed_weight)
        .expect("upload packed weight");
    let scales_device: CudaSlice<f16> = stream
        .clone_htod(&packed_scales)
        .expect("upload packed scales");
    let zero_points_device: CudaSlice<i32> = stream
        .clone_htod(&packed_zero_points)
        .expect("upload packed zero points");
    let mut output_device: CudaSlice<f16> = stream
        .alloc_zeros(fixture.rows * n)
        .expect("allocate output");
    let sms = context
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("query SM count");
    let workspace: CudaSlice<i32> = stream
        .alloc_zeros(usize::try_from(sms).expect("positive SM count"))
        .expect("allocate workspace");

    let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
    let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
    let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
    let (zero_points_pointer, _zero_points_guard) = zero_points_device.device_ptr(&stream);
    let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
    let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
    unsafe {
        launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
            weight_type: MarlinF16WeightType::U4,
            buffers: MarlinMmBuffers {
                a: input_pointer as *const c_void,
                b: weight_pointer as *const c_void,
                c: output_pointer as *mut c_void,
                c_tmp: std::ptr::null_mut(),
                a_scales: std::ptr::null_mut(),
                b_scales: scales_pointer as *mut c_void,
                zero_points: zero_points_pointer as *mut c_void,
                group_index: std::ptr::null_mut(),
                permutation: std::ptr::null_mut(),
                a_tmp: std::ptr::null_mut(),
                workspace: workspace_pointer as *mut c_void,
            },
            problem: MarlinMmProblem {
                m: fixture.rows as i32,
                n: n as i32,
                k: k as i32,
                lda: k as i32,
                num_groups: groups as i32,
                group_size: GROUP_SIZE as i32,
            },
            execution: MarlinMmExecution {
                device: 0,
                stream: stream.cu_stream(),
                sms,
                has_act_order: false,
                is_k_full: true,
                use_atomic_add: false,
                use_fp32_reduce: false,
            },
        });
    }
    stream.synchronize().expect("Marlin synchronize");
    let actual = stream
        .memcpy_dtov(&output_device)
        .expect("download Marlin output");

    let mut reference_squared = 0.0_f64;
    let mut error_squared = 0.0_f64;
    let mut non_finite = 0;
    for (actual, expected) in actual.iter().zip(reference.iter()) {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        if !actual.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = f64::from(actual - expected);
        reference_squared += f64::from(expected) * f64::from(expected);
        error_squared += error * error;
    }
    let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
    (relative_l2, non_finite)
}

#[test]
#[ignore = "requires an sm89 CUDA host and the versioned vLLM Marlin native artifact"]
fn qwen38_compressed_tensors_w4a16_matches_cpu_reference_for_four_fixed_fixtures() {
    let context = CudaContext::new(0).expect("CUDA context");
    for fixture in FIXTURES {
        let (relative_l2, non_finite) = run_fixture(&context, fixture);
        eprintln!(
            "QWEN38_CT_MARLIN_FIXTURE name={} rel_err={relative_l2:.8} non_finite={non_finite}",
            fixture.name
        );
        assert_eq!(non_finite, 0, "{} emitted NaN/Inf", fixture.name);
        assert!(
            relative_l2 < 0.05,
            "{} rel_err={relative_l2:.8} exceeds 0.05",
            fixture.name
        );
    }
}
