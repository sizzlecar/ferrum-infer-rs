use super::*;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, DevicePtr};
use half::f16;

mod gather_microbench;
mod gather_tests;

fn row_layouts(count: usize) -> Vec<LastTokenF16Row> {
    (0..count)
        .map(|row| LastTokenF16Row {
            input_pointer: 0x1000 + row as u64 * 32,
            input_bytes: 32,
            output_pointer: 0x2000 + row as u64 * 64,
            output_bytes: 64,
        })
        .collect()
}

fn select(ranges: &[std::ops::Range<u64>], rows: &[LastTokenF16Row]) -> Option<i32> {
    packed_last_token_rows(true, 16, 32, ranges.iter().cloned(), rows.iter().copied())
}

#[test]
fn packed_last_token_rows_requires_unit_ranges_and_exact_matrix_coverage() {
    let rows = row_layouts(2);
    assert_eq!(select(&[0..1, 1..2], &rows), Some(2));
    assert_eq!(select(&[0..1], &rows[..1]), None);
    assert_eq!(select(&[], &[]), None);
    assert_eq!(select(&[0..1], &rows), None);
    for ranges in [[0..2, 2..3], [0..1, 2..3], [1..2, 0..1], [0..0, 0..1]] {
        assert_eq!(select(&ranges, &rows), None);
    }
    assert_eq!(
        packed_last_token_rows(
            false,
            16,
            32,
            [0..1, 1..2].into_iter(),
            rows.iter().copied()
        ),
        None
    );
    for bad in [
        LastTokenF16Row {
            input_pointer: rows[1].input_pointer + 2,
            ..rows[1]
        },
        LastTokenF16Row {
            output_pointer: rows[1].output_pointer + 16,
            ..rows[1]
        },
        LastTokenF16Row {
            output_pointer: rows[0].output_pointer,
            ..rows[1]
        },
        LastTokenF16Row {
            input_bytes: 30,
            ..rows[1]
        },
        LastTokenF16Row {
            output_bytes: 80,
            ..rows[1]
        },
    ] {
        assert_eq!(select(&[0..1, 1..2], &[rows[0], bad]), None);
    }
    let overlapping = rows
        .iter()
        .enumerate()
        .map(|(index, row)| LastTokenF16Row {
            output_pointer: rows[0].input_pointer + index as u64 * 64,
            ..*row
        })
        .collect::<Vec<_>>();
    assert_eq!(select(&[0..1, 1..2], &overlapping), None);
}

#[test]
fn packed_last_token_rows_rejects_pointer_width_and_launch_overflow() {
    let rows = row_layouts(2);
    for (hidden, outputs) in [(0, 32), (16, 0), (u64::MAX, 32), (16, u64::MAX)] {
        assert_eq!(
            packed_last_token_rows(
                true,
                hidden,
                outputs,
                [0..1, 1..2].into_iter(),
                rows.iter().copied()
            ),
            None
        );
    }
    for base in [0, u64::MAX - 63] {
        let overflow = [
            LastTokenF16Row {
                input_pointer: base,
                ..rows[0]
            },
            LastTokenF16Row {
                input_pointer: base + 32,
                ..rows[1]
            },
        ];
        assert_eq!(select(&[0..1, 1..2], &overflow), None);
    }
    let overflow = [
        LastTokenF16Row {
            output_pointer: u64::MAX - 127,
            ..rows[0]
        },
        LastTokenF16Row {
            output_pointer: u64::MAX - 63,
            ..rows[1]
        },
    ];
    assert_eq!(select(&[0..1, 1..2], &overflow), None);
    let too_many = i32::MAX as usize + 1;
    assert_eq!(
        packed_last_token_rows(
            true,
            16,
            32,
            std::iter::repeat_n(0..1, too_many),
            std::iter::repeat_n(rows[0], too_many),
        ),
        None
    );
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    input_gap: usize,
    output_gap: usize,
    reverse_outputs: bool,
    first_prefill_tokens: usize,
}

fn values(count: usize, salt: usize) -> Vec<f16> {
    (0..count)
        .map(|index| f16::from_f32(((index * 13 + salt * 7) % 41) as f32 / 32.0 - 0.625))
        .collect()
}

fn assert_dot(actual: f16, input: &[f16], weight: &[f16]) {
    let products = input
        .iter()
        .zip(weight)
        .map(|(x, w)| x.to_f64() * w.to_f64());
    let expected = products.clone().sum::<f64>();
    // F32 accumulation error, one F16 relative spacing and minimum subnormal:
    // the same independent numerical bound as the existing F16 CUDA oracles.
    let bound = (input.len() as f64 * f64::from(f32::EPSILON) + 0.0009765625)
        * products.map(f64::abs).sum::<f64>()
        + f16::from_bits(1).to_f64();
    assert!(
        actual.is_finite() && (actual.to_f64() - expected).abs() <= bound,
        "projection={}, F64={expected}, bound={bound}",
        actual.to_f32()
    );
}

fn projection_case(rows: usize, hidden: usize, outputs: usize, layout: Layout) {
    projection_case_with_gather(rows, hidden, outputs, layout, false);
}

fn projection_case_with_gather(
    rows: usize,
    hidden: usize,
    outputs: usize,
    layout: Layout,
    allow_gather: bool,
) {
    let context = CudaContext::new(0).expect("LM head oracle requires an actual CUDA device");
    let stream = context.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    let guard = f16::from_f32(-117.0);
    let input_stride = hidden + layout.input_gap;
    let output_stride = outputs + layout.output_gap;
    let prefix = (layout.first_prefill_tokens - 1) * hidden;
    let mut input = vec![guard; 16 + prefix + rows * input_stride];
    let mut weight = vec![guard; 8];
    weight.extend(values(outputs * hidden, 9));
    weight.extend([guard; 8]);
    let mut output = vec![guard; 16 + rows * output_stride];
    let mut oracle_output = vec![guard; 16 + rows * outputs];
    for row in 0..rows {
        // Distinct inputs in reverse semantic order expose accidental row sorting.
        let source = 8 + prefix + row * input_stride;
        input[source..source + hidden].copy_from_slice(&values(hidden, rows - row));
        let physical_row = if layout.reverse_outputs {
            rows - 1 - row
        } else {
            row
        };
        let destination = 8 + physical_row * output_stride;
        output[destination..destination + outputs].fill(f16::NAN);
        oracle_output[8 + row * outputs..8 + (row + 1) * outputs].fill(f16::NAN);
    }
    let x = stream.clone_htod(&input).unwrap();
    let w = stream.clone_htod(&weight).unwrap();
    let y = stream.clone_htod(&output).unwrap();
    let scalar = stream.clone_htod(&oracle_output).unwrap();
    let scratch_template = vec![guard; 16 + rows * hidden];
    let scratch = stream.clone_htod(&scratch_template).unwrap();
    let (xp, _x_guard) = x.device_ptr(&stream);
    let (wp, _w_guard) = w.device_ptr(&stream);
    let (yp, _y_guard) = y.device_ptr(&stream);
    let (sp, _scalar_guard) = scalar.device_ptr(&stream);
    let (scratch_pointer, _scratch_guard) = scratch.device_ptr(&stream);
    let spans = (0..rows)
        .map(|row| {
            let physical_row = if layout.reverse_outputs {
                rows - 1 - row
            } else {
                row
            };
            LastTokenF16Row {
                input_pointer: xp + ((8 + prefix + row * input_stride) * 2) as u64,
                input_bytes: (hidden * 2) as u64,
                output_pointer: yp + ((8 + physical_row * output_stride) * 2) as u64,
                output_bytes: (outputs * 2) as u64,
            }
        })
        .collect::<Vec<_>>();
    let weight_span = LastTokenByteSpan {
        pointer: wp + 16,
        bytes: (outputs * hidden * 2) as u64,
    };
    validate_last_token_f16_access(hidden as u64, outputs as u64, weight_span, &spans).unwrap();
    let ranges = (0..rows)
        .map(|row| {
            if row == 0 {
                0..layout.first_prefill_tokens as u64
            } else {
                let start = (row + layout.first_prefill_tokens - 1) as u64;
                start..start + 1
            }
        })
        .collect::<Vec<_>>();
    let packed = packed_last_token_rows(
        true,
        hidden as u64,
        outputs as u64,
        ranges.into_iter(),
        spans.iter().copied(),
    );
    let expected_packed = rows > 1
        && layout.input_gap == 0
        && layout.output_gap == 0
        && !layout.reverse_outputs
        && layout.first_prefill_tokens == 1;
    assert_eq!(packed, expected_packed.then_some(rows as i32), "{layout:?}");
    let gather = allow_gather
        && packed.is_none()
        && last_token_output_rows_are_contiguous(outputs as u64, &spans)
        && last_token_gather_scratch_is_disjoint(
            hidden as u64,
            outputs as u64,
            weight_span,
            LastTokenByteSpan {
                pointer: scratch_pointer + 16,
                bytes: (rows * hidden * 2) as u64,
            },
            &spans,
        );
    assert_eq!(
        gather,
        allow_gather
            && !expected_packed
            && rows > 1
            && layout.output_gap == 0
            && !layout.reverse_outputs
    );
    if gather {
        copy_last_token_rows_f16(
            &stream,
            scratch_pointer + 16,
            hidden * 2,
            spans.iter().map(|span| span.input_pointer),
        )
        .unwrap();
        transformer::launch_gemm_f16(
            &blas,
            scratch_pointer + 16,
            wp + 16,
            spans[0].output_pointer,
            rows as i32,
            outputs as i32,
            hidden as i32,
            "test gathered last-token batch",
        )
        .unwrap();
    }
    for (row, span) in spans.iter().enumerate() {
        if !gather && (row == 0 || packed.is_none()) {
            transformer::launch_gemm_f16(
                &blas,
                span.input_pointer,
                wp + 16,
                span.output_pointer,
                packed.unwrap_or(1),
                outputs as i32,
                hidden as i32,
                "test LM head batch",
            )
            .unwrap();
        }
        transformer::launch_gemm_f16(
            &blas,
            span.input_pointer,
            wp + 16,
            sp + ((8 + row * outputs) * 2) as u64,
            1,
            outputs as i32,
            hidden as i32,
            "test LM head scalar oracle",
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let actual = stream.clone_dtoh(&y).unwrap();
    let reference = stream.clone_dtoh(&scalar).unwrap();
    for row in 0..rows {
        let source = 8 + prefix + row * input_stride;
        let physical_row = if layout.reverse_outputs {
            rows - 1 - row
        } else {
            row
        };
        let destination = 8 + physical_row * output_stride;
        for column in 0..outputs {
            let weights = &weight[8 + column * hidden..8 + (column + 1) * hidden];
            let input_row = &input[source..source + hidden];
            assert_dot(actual[destination + column], input_row, weights);
            assert_dot(reference[8 + row * outputs + column], input_row, weights);
        }
    }
    for (actual, original) in actual.iter().zip(&output) {
        if !original.is_nan() {
            assert_eq!(actual, original, "output padding/guard changed");
        }
    }
    for (actual, original) in reference.iter().zip(&oracle_output) {
        if !original.is_nan() {
            assert_eq!(actual, original, "scalar guard changed");
        }
    }
    assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
    assert_eq!(stream.clone_dtoh(&w).unwrap(), weight);
    let scratch_actual = stream.clone_dtoh(&scratch).unwrap();
    assert_eq!(&scratch_actual[..8], &scratch_template[..8]);
    assert_eq!(
        &scratch_actual[8 + rows * hidden..],
        &scratch_template[8 + rows * hidden..]
    );
    if gather {
        for row in 0..rows {
            let source = 8 + prefix + row * input_stride;
            assert_eq!(
                &scratch_actual[8 + row * hidden..8 + (row + 1) * hidden],
                &input[source..source + hidden]
            );
        }
    } else {
        assert_eq!(
            scratch_actual, scratch_template,
            "unused gather scratch changed"
        );
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn packed_f16_last_token_projection_matches_scalar_and_f64_on_cuda() {
    let layout = Layout {
        input_gap: 0,
        output_gap: 0,
        reverse_outputs: false,
        first_prefill_tokens: 1,
    };
    for (hidden, outputs) in [(17, 19), (256, 128)] {
        for rows in [1, 2, 4, 8] {
            projection_case(rows, hidden, outputs, layout);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn f16_last_token_projection_fallback_preserves_gaps_order_and_final_rows_on_cuda() {
    for layout in [
        Layout {
            input_gap: 3,
            output_gap: 0,
            reverse_outputs: false,
            first_prefill_tokens: 1,
        },
        Layout {
            input_gap: 0,
            output_gap: 5,
            reverse_outputs: false,
            first_prefill_tokens: 1,
        },
        Layout {
            input_gap: 0,
            output_gap: 0,
            reverse_outputs: true,
            first_prefill_tokens: 1,
        },
        Layout {
            input_gap: 0,
            output_gap: 0,
            reverse_outputs: false,
            first_prefill_tokens: 3,
        },
    ] {
        projection_case(4, 17, 19, layout);
    }
}
