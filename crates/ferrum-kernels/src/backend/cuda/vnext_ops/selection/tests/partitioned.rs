use super::*;
use cudarc::driver::{sys::CUevent_flags, DevicePtr};
use std::time::Instant;

fn bytes(case: &Case, precision: ArgmaxPrecision) -> Vec<u8> {
    case.logits
        .iter()
        .flat_map(|&value| match precision {
            ArgmaxPrecision::F16 => f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
            ArgmaxPrecision::F32 => value.to_le_bytes().to_vec(),
        })
        .collect()
}

fn large_cases(n: usize) -> Vec<Case> {
    let mut tied = vec![-8.0; n];
    tied[255] = 4.0;
    tied[256] = 4.0;
    tied[n - 1] = 4.0;
    tied[3] = f32::NAN;
    tied[4] = f32::INFINITY;
    tied[5] = f32::NEG_INFINITY;
    let mut mask = vec![1; n];
    mask[255] = 0;
    let base = Case {
        logits: tied,
        valid: mask,
        repetition: vec![255, 256, n as u32 - 1, n as u32 + 3],
        offsets: [1, u32::MAX],
    };
    let mut rounding = vec![-8.0; n];
    rounding[256] = 1.0001;
    rounding[n - 1] = 1.0002;
    let mut negative = vec![-10.0; n];
    negative[256] = -0.0;
    negative[n - 1] = 0.0;
    negative[1] = f32::from_bits(0x80000001);
    vec![
        Case {
            logits: base.logits.clone(),
            valid: vec![0; n],
            repetition: base.repetition.clone(),
            offsets: [0, 0],
        },
        Case {
            logits: base.logits.clone(),
            valid: base.valid.clone(),
            repetition: base.repetition.clone(),
            offsets: [4, 1],
        },
        Case {
            logits: base.logits.clone(),
            valid: base.valid.clone(),
            repetition: base.repetition.clone(),
            offsets: [9, u32::MAX],
        },
        base,
        Case {
            logits: rounding,
            valid: vec![1; n],
            repetition: vec![256],
            offsets: [0, 1],
        },
        Case {
            logits: negative,
            valid: vec![1; n],
            repetition: vec![256],
            offsets: [0, 1],
        },
    ]
}

#[test]
fn partitioned_argmax_dispatch_threshold_preserves_small_rows() {
    assert_eq!(argmax_dispatches(8191), 1);
    assert_eq!(argmax_dispatches(8192), 1);
    assert_eq!(argmax_dispatches(8193), 1);
    assert_eq!(argmax_dispatches(65535), 1);
    assert_eq!(argmax_dispatches(65536), 2);
    assert_eq!(argmax_dispatches(65537), 2);
    assert_eq!(argmax_dispatches(248320), 2);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn partitioned_argmax_matches_scalar_and_old_kernel_with_guarded_windows_on_cuda() {
    let context = CudaContext::new(0).expect("partitioned selection requires CUDA");
    let stream = context.default_stream();
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let functions = ArgmaxFunctions::load(&context, precision).unwrap();
        for n in [8191, 8192, 8193, 65535, 65536, 65537, 248320] {
            let cases = large_cases(n);
            let stride =
                masked_argmax_scratch_stride(n as u64, precision.element()).unwrap() as usize;
            for penalty in [1.0_f32, 4.0, 1.1] {
                let penalty_gpu = stream.clone_htod(&[penalty]).unwrap();
                for parallel in [false, argmax_dispatches(n as i32) == 2] {
                    // Independent participant windows share one invocation backing.
                    // Both outside guards and each inter-participant guard survive.
                    let scratch = stream
                        .clone_htod(&vec![0xCD_u8; (stride + 32) * cases.len()])
                        .unwrap();
                    let output = stream
                        .clone_htod(&vec![0xDEADBEEF_u32; cases.len() * 3])
                        .unwrap();
                    let mut retained = Vec::new();
                    for (slot, case) in cases.iter().enumerate() {
                        let original = bytes(case, precision);
                        let logits = stream.clone_htod(&original).unwrap();
                        let valid = stream.clone_htod(&case.valid).unwrap();
                        let repetition = stream.clone_htod(&case.repetition).unwrap();
                        let offsets = stream.clone_htod(&case.offsets).unwrap();
                        let args = ArgmaxArguments {
                            logits: logits.device_ptr(&stream).0,
                            scratch: scratch.device_ptr(&stream).0
                                + (slot * (stride + 32) + 16) as u64,
                            valid_mask: valid.device_ptr(&stream).0,
                            repetition_offsets: offsets.device_ptr(&stream).0,
                            repetition_token_ids: repetition.device_ptr(&stream).0,
                            repetition_penalty: penalty_gpu.device_ptr(&stream).0,
                            output: output.device_ptr(&stream).0 + (slot * 3 + 1) as u64 * 4,
                            vocabulary_size: n as i32,
                            repetition_capacity: case.repetition.len() as i32,
                        };
                        functions.launch(&stream, args, parallel).unwrap();
                        retained.push((logits, valid, repetition, offsets, original));
                    }
                    stream.synchronize().unwrap();
                    let selected = stream.clone_dtoh(&output).unwrap();
                    let workspace = stream.clone_dtoh(&scratch).unwrap();
                    for (slot, case) in cases.iter().enumerate() {
                        assert_eq!(
                            &selected[slot * 3..slot * 3 + 3],
                            &[0xDEADBEEF, reference(case, precision, penalty), 0xDEADBEEF],
                            "n={n} slot={slot} penalty={penalty} parallel={parallel}"
                        );
                        assert_eq!(
                            stream.clone_dtoh(&retained[slot].0).unwrap(),
                            retained[slot].4,
                            "semantic logits changed"
                        );
                        let start = slot * (stride + 32);
                        assert!(
                            workspace[start..start + 16]
                                .iter()
                                .chain(&workspace[start + 16 + stride..start + 32 + stride])
                                .all(|&b| b == 0xCD),
                            "participant scratch guard changed"
                        );
                        let active = penalty != 1.0
                            && case.offsets[0].min(case.repetition.len() as u32)
                                < case.offsets[1].min(case.repetition.len() as u32);
                        let written = if active {
                            bytes(case, precision).len()
                        } else if parallel {
                            ARGMAX_PARTITIONS as usize * 8
                        } else {
                            0
                        };
                        assert!(
                            workspace[start + 16 + written..start + 16 + stride]
                                .iter()
                                .all(|&b| b == 0xCD),
                            "selection wrote beyond its declared partial/penalty region"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device; emits CUDA-event microbenchmark JSON"]
fn partitioned_argmax_dispatch_microbench() {
    argmax_microbench(&[8192, 248320]);
}

#[test]
#[ignore = "requires an actual CUDA device; emits CUDA-event threshold microbenchmark JSON"]
fn partitioned_argmax_threshold_microbench() {
    argmax_microbench(&[65536]);
}

fn argmax_microbench(vocabularies: &[usize]) {
    let context = CudaContext::new(0).expect("argmax microbenchmark requires CUDA");
    let stream = context.default_stream();
    const ITERATIONS: usize = 32;
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let functions = ArgmaxFunctions::load(&context, precision).unwrap();
        for &n in vocabularies {
            let case = large_cases(n).remove(3);
            let logits = stream.clone_htod(&bytes(&case, precision)).unwrap();
            let valid = stream.clone_htod(&case.valid).unwrap();
            let repetition = stream.clone_htod(&case.repetition).unwrap();
            let offsets = stream.clone_htod(&case.offsets).unwrap();
            let scratch = stream
                .alloc_zeros::<u8>(
                    masked_argmax_scratch_stride(n as u64, precision.element()).unwrap() as usize,
                )
                .unwrap();
            let output = stream.alloc_zeros::<u32>(1).unwrap();
            for penalty in [1.0_f32, 1.1] {
                let penalty_gpu = stream.clone_htod(&[penalty]).unwrap();
                let args = ArgmaxArguments {
                    logits: logits.device_ptr(&stream).0,
                    scratch: scratch.device_ptr(&stream).0,
                    valid_mask: valid.device_ptr(&stream).0,
                    repetition_offsets: offsets.device_ptr(&stream).0,
                    repetition_token_ids: repetition.device_ptr(&stream).0,
                    repetition_penalty: penalty_gpu.device_ptr(&stream).0,
                    output: output.device_ptr(&stream).0,
                    vocabulary_size: n as i32,
                    repetition_capacity: case.repetition.len() as i32,
                };
                for round in 0..10 {
                    for parallel in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        stream.synchronize().unwrap();
                        let wall = Instant::now();
                        let start = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        for _ in 0..ITERATIONS {
                            functions.launch(&stream, args, parallel).unwrap();
                        }
                        let end = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        end.synchronize().unwrap();
                        let wall_ns = wall.elapsed().as_nanos() as f64 / ITERATIONS as f64;
                        let gpu_ns =
                            f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / ITERATIONS as f64;
                        assert_eq!(
                            stream.clone_dtoh(&output).unwrap()[0],
                            reference(&case, precision, penalty)
                        );
                        if round >= 2 {
                            println!(
                                "{}",
                                serde_json::json!({"benchmark":"partitioned_argmax","precision": if matches!(precision,ArgmaxPrecision::F16) {"f16"} else {"f32"},
                                "vocabulary":n,"penalty":penalty,"parallel":parallel,"round":round-2,"iterations":ITERATIONS,"gpu_ns":gpu_ns,"wall_ns":wall_ns})
                            );
                        }
                    }
                }
            }
        }
    }
}
