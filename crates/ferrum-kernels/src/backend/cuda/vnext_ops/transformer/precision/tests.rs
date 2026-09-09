use super::*;
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::f16;

#[test]
#[ignore = "requires an actual CUDA device"]
fn master_rms_norm_matches_f64_with_half_weights_on_cuda() {
    let context = CudaContext::new(0).expect("normalization conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::RMS_NORM))
        .unwrap();
    for width in [1_usize, 17, 33, 128, 4096] {
        let rows = 3_usize;
        // Include master values outside F16's finite range and values that
        // would disappear if the provider first rounded the residual to F16.
        let input = (0..rows * width)
            .map(|i| match i / width {
                0 => ((i * 7 % 31) as f32 - 15.0) * 10000.125,
                1 => ((i * 13 % 41) as f32 - 20.0) * 0.000000125,
                _ => 1.0001 + (i % 7) as f32 * 0.00003125,
            })
            .collect::<Vec<_>>();
        let weights = (0..width)
            .map(|i| f16::from_f32(((i * 11 % 17) as f32 - 8.0) / 8.0))
            .collect::<Vec<_>>();
        let epsilon = 1e-6_f32;
        let weight_gpu = stream.clone_htod(&weights).unwrap();
        for precision in [
            RmsNormPrecision::F16,
            RmsNormPrecision::F32ToF16,
            RmsNormPrecision::F32,
        ] {
            let represented = if matches!(precision, RmsNormPrecision::F16) {
                input
                    .iter()
                    .map(|&x| f16::from_f32(x.clamp(-60000.0, 60000.0)).to_f32())
                    .collect::<Vec<_>>()
            } else {
                input.clone()
            };
            let input_bytes = if matches!(precision, RmsNormPrecision::F16) {
                represented
                    .iter()
                    .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                represented
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let input_gpu = stream.clone_htod(&input_bytes).unwrap();
            let reference = represented
                .chunks_exact(width)
                .flat_map(|row| {
                    let variance =
                        row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / width as f64;
                    row.iter().zip(&weights).map(move |(&x, &w)| {
                        f64::from(x) / (variance + f64::from(epsilon)).sqrt() * w.to_f64()
                    })
                })
                .collect::<Vec<_>>();
            let function = module.load_function(precision.kernel()).unwrap();
            let scalar_bytes = if matches!(precision, RmsNormPrecision::F32) {
                4
            } else {
                2
            };
            let mut output = stream
                .clone_htod(&vec![0xCD_u8; 16 + rows * width * scalar_bytes + 16])
                .unwrap();
            let mut span = output.slice_mut(16..16 + rows * width * scalar_bytes);
            let (out, _out_guard) = span.device_ptr_mut(&stream);
            let hidden = width as i32;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&input_gpu)
                .arg(&weight_gpu)
                .arg(&out)
                .arg(&hidden)
                .arg(&epsilon);
            // SAFETY: Retained input/weight allocations have the exact typed
            // row extents; output includes disjoint canaries, and the complete
            // warps include neutral lanes for partial hidden dimensions.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (super::super::rms_norm_threads(hidden), 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            drop(_out_guard);
            drop(span);
            let actual = stream.clone_dtoh(&output).unwrap();
            assert!(actual[..16]
                .iter()
                .chain(&actual[actual.len() - 16..])
                .all(|&x| x == 0xCD));
            for (i, bytes) in actual[16..actual.len() - 16]
                .chunks_exact(scalar_bytes)
                .enumerate()
            {
                let (value, rounding) = if scalar_bytes == 4 {
                    (
                        f64::from(f32::from_le_bytes(bytes.try_into().unwrap())),
                        0.0,
                    )
                } else {
                    (
                        f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f64(),
                        0.0009765625,
                    )
                };
                let bound =
                    (32.0 * f64::from(f32::EPSILON) + rounding) * reference[i].abs().max(1.0);
                assert!(
                    (value - reference[i]).abs() <= bound,
                    "width={width} element={i} actual={value} expected={} bound={bound}",
                    reference[i]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn master_residual_preserves_f32_and_declared_inplace_alias_on_cuda() {
    let context = CudaContext::new(0).expect("residual conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD))
        .unwrap();
    let function = module
        .load_function(ResidualPrecision::F32F16.kernel())
        .unwrap();
    for count in [1_usize, 33, 513] {
        let residual = (0..count)
            .map(|i| match i % 8 {
                0 => 70000.125_f32,
                1 => 1.0001,
                2 => -65000.125,
                3 => 0.000000125,
                4 => f32::from_bits(1),
                5 => f32::from_bits(0x807fffff),
                6 => -0.0,
                _ => f32::MIN_POSITIVE,
            })
            .collect::<Vec<_>>();
        let update = (0..count)
            .map(|i| {
                f16::from_f32(if i % 8 >= 4 {
                    0.0
                } else {
                    ((i * 7 % 23) as f32 - 11.0) / 32.0
                })
            })
            .collect::<Vec<_>>();
        let expected = residual
            .iter()
            .zip(&update)
            .map(|(&a, &b)| a + b.to_f32())
            .collect::<Vec<_>>();
        let update_gpu = stream.clone_htod(&update).unwrap();
        for inplace in [false, true] {
            let mut padded = vec![-12345.0_f32; count + 8];
            padded[4..4 + count].copy_from_slice(&residual);
            let mut left = stream.clone_htod(&padded).unwrap();
            let mut out = stream.clone_htod(&vec![-12345.0_f32; count + 8]).unwrap();
            let (left_ptr, _left_guard) = left.device_ptr_mut(&stream);
            let (out_ptr, _out_guard) = out.device_ptr_mut(&stream);
            let left_ptr = left_ptr + 16;
            let output_ptr = if inplace { left_ptr } else { out_ptr + 16 };
            let n = count as i32;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&left_ptr)
                .arg(&update_gpu)
                .arg(&output_ptr)
                .arg(&n);
            // SAFETY: Each scalar reads one F32 and one F16 at equal logical
            // indices. The only alias is the declared output/residual alias;
            // every allocation and pointer guard remains live through launch.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.unwrap();
            drop(_left_guard);
            drop(_out_guard);
            let actual = stream
                .clone_dtoh(if inplace { &left } else { &out })
                .unwrap();
            assert_eq!(&actual[..4], &[-12345.0; 4]);
            assert_eq!(&actual[count + 4..], &[-12345.0; 4]);
            for (actual, expected) in actual[4..4 + count].iter().zip(&expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            if !inplace {
                assert_eq!(stream.clone_dtoh(&left).unwrap(), padded);
            }
        }
    }
}
