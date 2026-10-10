//! Device-content regression for the two neutral repetition inputs. Cache
//! publication and fresh authority validation are tested at their own layers;
//! this fixture does not claim to execute the model executor's residency ledger.

use super::*;
use cudarc::driver::DevicePtr;

#[derive(Clone, Copy, Debug)]
enum RepetitionWave {
    WriteNeutral,
    KeepNeutral,
    WriteActive,
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn neutral_repetition_resident_windows_preserve_logits_and_greedy_selection_on_cuda() {
    let context = CudaContext::new(0).expect("neutral repetition conformance requires CUDA");
    let stream = context.default_stream();
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let functions = ArgmaxFunctions::load(&context, precision).unwrap();
        for n in [513_usize, 65_536] {
            let mut positive = vec![-8.0; n];
            positive[0] = 9.0;
            positive[n - 1] = 10.0;
            let mut negative = vec![-8.0; n];
            negative[0] = -2.0;
            negative[1] = -3.0;
            let cases = [
                Case {
                    logits: positive,
                    valid: vec![1; n],
                    repetition: vec![n as u32 - 1],
                    offsets: [0, 0],
                },
                Case {
                    logits: negative,
                    valid: vec![1; n],
                    repetition: vec![0],
                    offsets: [0, 0],
                },
            ];
            let originals = cases
                .iter()
                .map(|case| {
                    case.logits
                        .iter()
                        .flat_map(|value| match precision {
                            ArgmaxPrecision::F16 => {
                                f16::from_f32(*value).to_bits().to_le_bytes().to_vec()
                            }
                            ArgmaxPrecision::F32 => value.to_le_bytes().to_vec(),
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let logits = originals
                .iter()
                .map(|bytes| stream.clone_htod(bytes).unwrap())
                .collect::<Vec<_>>();
            let masks = cases
                .iter()
                .map(|case| stream.clone_htod(&case.valid).unwrap())
                .collect::<Vec<_>>();
            let ids = cases
                .iter()
                .map(|case| stream.clone_htod(&case.repetition).unwrap())
                .collect::<Vec<_>>();
            // Every physical participant owns distinct guarded input windows.
            // Deliberately dirty initial contents require the first rewrite.
            let mut offsets = stream.clone_htod(&[0xa5_u8; 32]).unwrap();
            let mut penalties = stream.clone_htod(&[0xa5_u8; 24]).unwrap();
            let stride =
                masked_argmax_scratch_stride(n as u64, precision.element()).unwrap() as usize;
            let mut scratch = stream
                .clone_htod(&vec![0xcd_u8; 2 * (stride + 32)])
                .unwrap();
            let mut output = stream.clone_htod(&[0xdeadbeef_u32; 6]).unwrap();

            for wave in [
                RepetitionWave::WriteNeutral,
                RepetitionWave::KeepNeutral,
                RepetitionWave::WriteActive,
                RepetitionWave::WriteNeutral,
                RepetitionWave::KeepNeutral,
            ] {
                let active = matches!(wave, RepetitionWave::WriteActive);
                let penalty = if active { 2.0_f32 } else { 1.0 };
                let count = u32::from(active);
                let expected_offsets = [0_u32, count]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>();
                let mut input_copies = 0;
                if !matches!(wave, RepetitionWave::KeepNeutral) {
                    for p in 0..2 {
                        stream
                            .memcpy_htod(
                                &expected_offsets,
                                &mut offsets.slice_mut(p * 16 + 4..p * 16 + 12),
                            )
                            .unwrap();
                        stream
                            .memcpy_htod(
                                &penalty.to_le_bytes(),
                                &mut penalties.slice_mut(p * 12 + 4..p * 12 + 8),
                            )
                            .unwrap();
                        input_copies += 2;
                    }
                }
                stream
                    .memcpy_htod(&vec![0xcd_u8; 2 * (stride + 32)], &mut scratch)
                    .unwrap();
                stream
                    .memcpy_htod(&[0xdeadbeef_u32; 6], &mut output)
                    .unwrap();
                for p in 0..2 {
                    functions
                        .launch(
                            &stream,
                            ArgmaxArguments {
                                logits: logits[p].device_ptr(&stream).0,
                                scratch: scratch.device_ptr(&stream).0
                                    + (p * (stride + 32) + 16) as u64,
                                valid_mask: masks[p].device_ptr(&stream).0,
                                repetition_offsets: offsets.device_ptr(&stream).0
                                    + (p * 16 + 4) as u64,
                                repetition_token_ids: ids[p].device_ptr(&stream).0,
                                repetition_penalty: penalties.device_ptr(&stream).0
                                    + (p * 12 + 4) as u64,
                                output: output.device_ptr(&stream).0 + (p * 3 + 1) as u64 * 4,
                                vocabulary_size: n as i32,
                                repetition_capacity: 1,
                            },
                            argmax_dispatches(n as i32) == 2,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                let selected = stream.clone_dtoh(&output).unwrap();
                let actual_offsets = stream.clone_dtoh(&offsets).unwrap();
                let actual_penalties = stream.clone_dtoh(&penalties).unwrap();
                let workspace = stream.clone_dtoh(&scratch).unwrap();
                for (p, case) in cases.iter().enumerate() {
                    let reference_case = Case {
                        logits: case.logits.clone(),
                        valid: case.valid.clone(),
                        repetition: case.repetition.clone(),
                        offsets: [0, count],
                    };
                    assert_eq!(
                        selected[p * 3 + 1],
                        reference(&reference_case, precision, penalty)
                    );
                    assert_eq!(selected[p * 3], 0xdeadbeef);
                    assert_eq!(selected[p * 3 + 2], 0xdeadbeef);
                    assert_eq!(stream.clone_dtoh(&logits[p]).unwrap(), originals[p]);
                    assert_eq!(&actual_offsets[p * 16 + 4..p * 16 + 12], expected_offsets);
                    assert_eq!(
                        &actual_penalties[p * 12 + 4..p * 12 + 8],
                        penalty.to_le_bytes()
                    );
                    assert!(actual_offsets[p * 16..p * 16 + 4]
                        .iter()
                        .chain(&actual_offsets[p * 16 + 12..p * 16 + 16])
                        .all(|b| *b == 0xa5));
                    assert!(actual_penalties[p * 12..p * 12 + 4]
                        .iter()
                        .chain(&actual_penalties[p * 12 + 8..p * 12 + 12])
                        .all(|b| *b == 0xa5));
                    let start = p * (stride + 32);
                    assert!(workspace[start..start + 16]
                        .iter()
                        .chain(&workspace[start + 16 + stride..start + 32 + stride])
                        .all(|b| *b == 0xcd));
                }
                println!("neutral_repetition_device_contents n={n} wave={wave:?} participants=2 offsets_and_penalty_copies={input_copies}");
            }
        }
    }
}
