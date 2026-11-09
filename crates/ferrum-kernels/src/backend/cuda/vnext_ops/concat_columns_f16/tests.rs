use super::*;
use cudarc::driver::{CudaContext, DevicePtr};

#[test]
fn concat_columns_checks_widths_and_wave_wide_aliases_without_workspace() {
    assert_eq!(checked_widths(3, 5, 8).unwrap(), [6, 10, 16]);
    for (l, r, o) in [
        (0, 1, 1),
        (1, 0, 1),
        (3, 5, 9),
        (u64::MAX, 1, 0),
        (isize::MAX as u64, 1, isize::MAX as u64 + 1),
    ] {
        assert!(checked_widths(l, r, o).is_err());
    }
    assert!(validate_spans(&[10..20, 30..40], &[50..70, 80..100]).is_ok());
    for writes in [vec![15..25], vec![35..45], vec![50..70, 60..80]] {
        assert!(validate_spans(&[10..20, 30..40], &writes).is_err());
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_concat_columns_b2_copies_original_bits_with_different_token_lengths() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let [l, r, o] = [3_usize, 5, 8];
    let values = [
        0_u16, 0x8000, 0x7E01, 0x7C00, 0xFC00, 0x0001, 0x3C00, 0xFE55,
    ];
    for counts in [[1_usize, 2], [5, 1], [2, 7]] {
        let tokens = counts.iter().sum::<usize>();
        let left = (0..tokens * l)
            .map(|i| values[i % values.len()])
            .collect::<Vec<_>>();
        let right = (0..tokens * r)
            .map(|i| values[(i + 3) % values.len()])
            .collect::<Vec<_>>();
        let x = stream.clone_htod(&left).unwrap();
        let y = stream.clone_htod(&right).unwrap();
        let mut expected = vec![0x5522_u16; tokens * o + 3];
        let output = stream.clone_htod(&expected).unwrap();
        let slots = [counts[1] * o + 2, 1];
        let mut start = 0;
        let mut copies = Vec::new();
        for (index, n) in counts.into_iter().enumerate() {
            copies.push(CopyRows {
                left: x.device_ptr(&stream).0 + (start * l * 2) as u64,
                right: y.device_ptr(&stream).0 + (start * r * 2) as u64,
                output: output.device_ptr(&stream).0 + (slots[index] * 2) as u64,
                rows: n as u64,
            });
            for row in 0..n {
                let dst = slots[index] + row * o;
                expected[dst..dst + l]
                    .copy_from_slice(&left[(start + row) * l..(start + row + 1) * l]);
                expected[dst + l..dst + o]
                    .copy_from_slice(&right[(start + row) * r..(start + row + 1) * r]);
            }
            start += n;
        }
        // Actual two-participant native transport; Core permits/admission are
        // not fabricated or claimed by this direct CUDA helper fixture.
        copy_rows(
            &stream,
            checked_widths(l as u64, r as u64, o as u64).unwrap(),
            &copies,
        )
        .unwrap();
        assert_eq!(stream.clone_dtoh(&output).unwrap(), expected);
        assert_eq!(stream.clone_dtoh(&x).unwrap(), left);
        assert_eq!(stream.clone_dtoh(&y).unwrap(), right);
    }
}
