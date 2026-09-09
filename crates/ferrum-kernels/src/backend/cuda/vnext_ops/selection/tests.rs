use super::*;
use cudarc::driver::{CudaContext, DevicePtrMut};
use half::f16;

struct Case {
    logits: Vec<f32>,
    valid: Vec<u8>,
    repetition: Vec<u32>,
    offsets: [u32; 2],
}

fn rounded(value: f32, precision: ArgmaxPrecision) -> f32 {
    match precision {
        ArgmaxPrecision::F16 => f16::from_f32(value).to_f32(),
        ArgmaxPrecision::F32 => value,
    }
}

fn reference(case: &Case, precision: ArgmaxPrecision, penalty: f32) -> u32 {
    let mut row = case
        .logits
        .iter()
        .map(|&x| rounded(x, precision))
        .collect::<Vec<_>>();
    let start = (case.offsets[0] as usize).min(case.repetition.len());
    let end = (case.offsets[1] as usize).min(case.repetition.len());
    if penalty != 1.0 && start < end {
        for &id in &case.repetition[start..end] {
            if let Some(value) = row.get_mut(id as usize).filter(|value| value.is_finite()) {
                *value = rounded(
                    if *value > 0.0 {
                        *value / penalty
                    } else {
                        *value * penalty
                    },
                    precision,
                );
            }
        }
    }
    // Iterate in vocabulary order so an equal finite value retains the
    // smallest permitted token, without sharing the kernel's reduction tree.
    let mut selected = None;
    for (id, &value) in row.iter().enumerate() {
        if case.valid[id] != 0
            && value.is_finite()
            && selected.is_none_or(|(_, previous)| value > previous)
        {
            selected = Some((id as u32, value));
        }
    }
    selected.map_or(u32::MAX, |(id, _)| id)
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn masked_selection_matches_scalar_semantics_and_preserves_logits_on_cuda() {
    let context = CudaContext::new(0).expect("selection conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::ARGMAX_ROWS))
        .unwrap();
    let mut tail = vec![-10.0; 513];
    tail[254] = 4.0;
    tail[512] = 4.0;
    let mut tail_mask = vec![1; 513];
    tail_mask[254] = 0;
    let cases = [
        Case {
            logits: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0, 2.0, 1.0],
            valid: vec![1; 6],
            repetition: vec![3, 4],
            offsets: [0, 2],
        },
        Case {
            logits: vec![1.0, 1.0001, 1.0002],
            valid: vec![1; 3],
            repetition: vec![0],
            offsets: [0, 0],
        },
        Case {
            logits: vec![0.0, f32::from_bits(1), f32::from_bits(0x80000001)],
            valid: vec![1; 3],
            repetition: vec![1],
            offsets: [0, 1],
        },
        Case {
            logits: vec![4.0, 3.0, -2.0, -1.0, 100.0],
            valid: vec![1, 1, 1, 1, 0],
            repetition: vec![1, 0, 900],
            offsets: [1, 10],
        },
        Case {
            logits: vec![-4.0, -2.0, -3.0],
            valid: vec![1; 3],
            repetition: vec![1],
            offsets: [0, 1],
        },
        Case {
            logits: vec![1.0, 2.0],
            valid: vec![0; 2],
            repetition: vec![0],
            offsets: [0, 1],
        },
        Case {
            logits: tail,
            valid: tail_mask,
            repetition: vec![512],
            offsets: [1, 0],
        },
    ];
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let function = module.load_function(precision.kernel()).unwrap();
        for (index, case) in cases.iter().enumerate() {
            let original = case
                .logits
                .iter()
                .flat_map(|&value| match precision {
                    ArgmaxPrecision::F16 => f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                    ArgmaxPrecision::F32 => value.to_le_bytes().to_vec(),
                })
                .collect::<Vec<_>>();
            let logits = stream.clone_htod(&original).unwrap();
            let valid = stream.clone_htod(&case.valid).unwrap();
            let repetition = stream.clone_htod(&case.repetition).unwrap();
            let offsets = stream.clone_htod(&case.offsets).unwrap();
            let n = case.logits.len() as i32;
            let capacity = case.repetition.len() as i32;
            for penalty in [1.0_f32, 1.75, 0.75] {
                let penalty_gpu = stream.clone_htod(&[penalty]).unwrap();
                let stride =
                    masked_argmax_scratch_stride(case.logits.len() as u64, precision.element())
                        .unwrap() as usize;
                let mut scratch = stream.clone_htod(&vec![0xCD_u8; stride + 32]).unwrap();
                let mut output = stream.clone_htod(&[0xDEADBEEF_u32; 3]).unwrap();
                let (scratch_ptr, scratch_guard) = scratch.device_ptr_mut(&stream);
                let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
                let scratch_ptr = scratch_ptr + 16;
                let output_ptr = output_ptr + 4;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&logits)
                    .arg(&scratch_ptr)
                    .arg(&n)
                    .arg(&valid)
                    .arg(&n)
                    .arg(&offsets)
                    .arg(&repetition)
                    .arg(&penalty_gpu)
                    .arg(&capacity)
                    .arg(&output_ptr);
                // SAFETY: Exact scalar type/length per contract, one complete
                // 256-thread block, separate retained scratch and immutable
                // logits, and a single output slot surrounded by canaries.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                drop(scratch_guard);
                drop(output_guard);
                let actual = stream.clone_dtoh(&output).unwrap();
                assert_eq!(
                    actual,
                    [0xDEADBEEF, reference(case, precision, penalty), 0xDEADBEEF],
                    "case={index} penalty={penalty}"
                );
                assert_eq!(
                    stream.clone_dtoh(&logits).unwrap(),
                    original,
                    "semantic logits changed"
                );
                let scratch = stream.clone_dtoh(&scratch).unwrap();
                assert!(scratch[..16]
                    .iter()
                    .chain(&scratch[16 + original.len()..])
                    .all(|&byte| byte == 0xCD));
                if penalty == 1.0 || case.offsets[0] >= case.offsets[1] {
                    assert!(
                        scratch.iter().all(|&byte| byte == 0xCD),
                        "inactive penalty wrote scratch"
                    );
                }
            }
        }
    }
}

#[test]
fn selection_scratch_rejects_empty_or_overflowing_vocabularies() {
    for element in [ElementType::F16, ElementType::F32] {
        assert!(masked_argmax_scratch_stride(0, element).is_err());
        assert!(masked_argmax_scratch_stride(u64::MAX, element).is_err());
        let bytes = masked_argmax_scratch_stride(33, element).unwrap();
        assert!(bytes >= 33 * element.size_bytes());
        assert_eq!(bytes % VALUE_ALIGNMENT_BYTES, 0);
    }
}
