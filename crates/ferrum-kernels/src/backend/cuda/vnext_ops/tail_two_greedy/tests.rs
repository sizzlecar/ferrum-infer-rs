use super::*;
use cudarc::driver::{CudaContext, DevicePtr};
use half::f16;

#[test]
fn tail_head_workspace_is_one_bounded_row_and_rejects_invalid_dimensions() {
    assert_eq!(dimensions(8, 17).unwrap(), (8, 17, 48));
    assert_eq!(dimensions(2048, 248320).unwrap().2, 496640);
    for (h, v) in [(0, 1), (1, 0), (u64::MAX, 1), (1, u64::MAX)] {
        assert!(dimensions(h, v).is_err());
    }
    assert!(dimensions(1, i32::MAX as u64).is_err());
    assert!(dimensions(1, i32::MAX as u64 - 255).is_ok());
    assert!(byte_range(0, 4).is_err());
    assert!(byte_range(u64::MAX - 2, 4).is_err());
    let reads = [100..120, 200..240];
    assert!(validate_disjoint(&reads, &[300..308, 400..432, 500..548]).is_ok());
    // Current output/input, cross-participant output/input, output/output,
    // scratch/output: all rejected before any CUDA command is created.
    for writes in [
        vec![110..118],
        vec![230..238],
        vec![300..308, 304..336],
        vec![300..308, 300..348],
    ] {
        assert!(validate_disjoint(&reads, &writes).is_err());
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_tail_two_greedy_b2_matches_original_head_and_preserves_hidden() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let blas = CudaBlas::new(Arc::clone(&stream)).unwrap();
    let function = load(&context).unwrap();
    let ordinary = ArgmaxFunctions::load(&context, ArgmaxPrecision::F16).unwrap();
    let h = 8_usize;
    let v = 33_usize;
    let weights = (0..h * v)
        .map(|i| f16::from_f32(((i * 13 % 41) as f32 - 20.0) / 32.0))
        .collect::<Vec<_>>();
    let weight = stream.clone_htod(&weights).unwrap();
    let valid = stream.clone_htod(&vec![1_u8; v]).unwrap();
    let offsets = stream.clone_htod(&[0_u32, 0]).unwrap();
    let repetition = stream.clone_htod(&[0_u32]).unwrap();
    let penalty = stream.clone_htod(&[1.0_f32]).unwrap();
    let old_scratch = stream.clone_htod(&vec![0_u8; v * 2]).unwrap();
    let old_logits = stream.clone_htod(&vec![f16::ZERO; v]).unwrap();
    let old_decision = stream.clone_htod(&[u32::MAX]).unwrap();
    for counts in [[1, 2], [2, 1], [7, 2]] {
        let raw = (0..counts.iter().sum::<usize>() * h)
            .map(|i| f16::from_f32(((i * 7 % 29) as f32 - 14.0) / 16.0))
            .collect::<Vec<_>>();
        let input = stream.clone_htod(&raw).unwrap();
        let mut decisions_original = vec![0xDEADBEEF_u32; 7];
        let decisions = stream.clone_htod(&decisions_original).unwrap();
        let carrier_original = vec![f16::from_bits(0x5522); 4 * h + 3];
        let carrier = stream.clone_htod(&carrier_original).unwrap();
        let (_, _, bytes) = dimensions(h as u64, v as u64).unwrap();
        let scratch = stream
            .clone_htod(&vec![0xCD_u8; bytes as usize + 32])
            .unwrap();
        // The native command helper has two different participant lengths;
        // this exercises its actual CUDA arithmetic, not Core wave admission.
        // Output owners are reversed relative to packed input token order.
        let output_slots = [4, 1];
        let carrier_slots = [2 * h + 2, 1];
        let mut start = 0;
        let mut launches = Vec::new();
        let mut expected_carrier = carrier_original.clone();
        for (index, &n) in counts.iter().enumerate() {
            let rows = n.min(2);
            let tail = start + n - rows;
            launches.push(TailLaunch {
                input: input.device_ptr(&stream).0 + (tail * h * 2) as u64,
                weight: weight.device_ptr(&stream).0,
                decisions: decisions.device_ptr(&stream).0 + (output_slots[index] * 4) as u64,
                carrier: carrier.device_ptr(&stream).0 + (carrier_slots[index] * 2) as u64,
                rows: rows as u64,
            });
            for slot in 0..2 {
                let row = tail + slot.min(rows - 1);
                transformer::launch_gemm_f16(
                    &blas,
                    input.device_ptr(&stream).0 + (row * h * 2) as u64,
                    weight.device_ptr(&stream).0,
                    old_logits.device_ptr(&stream).0,
                    1,
                    v as i32,
                    h as i32,
                    "ordinary head reference",
                )
                .unwrap();
                ordinary
                    .launch(
                        &stream,
                        ArgmaxArguments {
                            logits: old_logits.device_ptr(&stream).0,
                            scratch: old_scratch.device_ptr(&stream).0,
                            valid_mask: valid.device_ptr(&stream).0,
                            repetition_offsets: offsets.device_ptr(&stream).0,
                            repetition_token_ids: repetition.device_ptr(&stream).0,
                            repetition_penalty: penalty.device_ptr(&stream).0,
                            output: old_decision.device_ptr(&stream).0,
                            vocabulary_size: v as i32,
                            repetition_capacity: 1,
                        },
                        false,
                    )
                    .unwrap();
                decisions_original[output_slots[index] + slot] =
                    stream.clone_dtoh(&old_decision).unwrap()[0];
                expected_carrier
                    [carrier_slots[index] + slot * h..carrier_slots[index] + (slot + 1) * h]
                    .copy_from_slice(&raw[row * h..(row + 1) * h]);
            }
            start += n;
        }
        launch_batch(
            &stream,
            &blas,
            &function,
            h as i32,
            v as i32,
            scratch.device_ptr(&stream).0 + 16,
            &launches,
        )
        .unwrap();
        assert_eq!(
            stream.clone_dtoh(&decisions).unwrap(),
            decisions_original,
            "B2 {counts:?}"
        );
        let bits = |values: Vec<f16>| values.into_iter().map(f16::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(stream.clone_dtoh(&carrier).unwrap()),
            bits(expected_carrier)
        );
        assert_eq!(bits(stream.clone_dtoh(&input).unwrap()), bits(raw));
        assert_eq!(
            bits(stream.clone_dtoh(&weight).unwrap()),
            bits(weights.clone())
        );
        let actual = stream.clone_dtoh(&scratch).unwrap();
        assert!(actual[..16]
            .iter()
            .chain(&actual[16 + v * 2..])
            .all(|x| *x == 0xCD));
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_tail_greedy_reduction_preserves_f16_rounding_finite_and_total_order() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let function = load(&context).unwrap();
    for values in [
        vec![-0.0, 0.0],
        vec![0.0, -0.0],
        vec![1.0, 1.0001, 1.0002],
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        vec![f32::NAN, 3.0, 3.0, 2.0],
    ] {
        let values = values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let expected = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(a, x), (b, y)| x.to_f32().total_cmp(&y.to_f32()).then_with(|| b.cmp(a)))
            .map_or(u32::MAX, |(i, _)| i as u32);
        let input = stream.clone_htod(&values).unwrap();
        let output = stream.clone_htod(&[17_u32, 17, 17]).unwrap();
        launch_argmax(
            &stream,
            &function,
            input.device_ptr(&stream).0,
            values.len() as i32,
            output.device_ptr(&stream).0 + 4,
        )
        .unwrap();
        assert_eq!(stream.clone_dtoh(&output).unwrap(), [17, expected, 17]);
        assert_eq!(
            stream
                .clone_dtoh(&input)
                .unwrap()
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>(),
            values.into_iter().map(f16::to_bits).collect::<Vec<_>>()
        );
    }
}
