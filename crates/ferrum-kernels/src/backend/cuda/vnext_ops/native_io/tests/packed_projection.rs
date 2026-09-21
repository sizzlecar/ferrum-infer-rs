use super::*;
use std::ops::Range;

fn projection_matrix(format: MatrixFormat) -> (Vec<u8>, Vec<f32>, usize) {
    let (mut bytes, mut decoded, width) = matrix(format, 7);
    if let MatrixFormat::Block(format) = format {
        let scale_offset = match format {
            GgufBlockFormat::Q3K => 108,
            GgufBlockFormat::Q6K => 208,
            _ => 0,
        };
        for (index, block) in bytes.chunks_exact_mut(format.block_bytes()).enumerate() {
            // Each output row has a distinct legal scale; reusing the two
            // base fixture blocks alone would repeat every other weight row.
            let magnitude = ((index / 3 + 1) * 3 + index % 3) as f32 / 128.0;
            let scale = if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            block[scale_offset..scale_offset + 2]
                .copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        format.decode(&bytes, &mut decoded).unwrap();
    }
    for row in 0..7 {
        for prior in 0..row {
            assert_ne!(
                decoded[row * width..(row + 1) * width],
                decoded[prior * width..(prior + 1) * width]
            );
        }
    }
    (bytes, decoded, width)
}

fn rows(precision: TokenPrecision, count: usize) -> Vec<NativeProjectionRow> {
    let bytes = precision.element().size_bytes();
    (0..count)
        .map(|index| NativeProjectionRow {
            input: 0x10000 + index as u64 * 33 * bytes,
            input_bytes: 33 * bytes,
            output: 0x100000 + index as u64 * 11 * bytes,
            output_bytes: 11 * bytes,
        })
        .collect()
}

fn select(
    precision: TokenPrecision,
    rows: &[NativeProjectionRow],
    spans: &[Range<u64>],
) -> Option<u32> {
    packed_projection_rows(
        precision,
        true,
        33,
        11,
        &[part(MatrixFormat::DenseF16, 11, 33)],
        spans.iter().cloned(),
        rows.iter().copied(),
        &[0x200000..0x201000],
    )
}

#[test]
fn native_projection_packing_requires_exact_physical_rows_and_nonaliasing() {
    let spans = [0..1, 1..2, 2..3];
    for precision in [TokenPrecision::F16, TokenPrecision::F32] {
        let valid = rows(precision, 3);
        assert_eq!(select(precision, &valid, &spans), Some(3));
        let mut invalid = valid.clone();
        invalid[1].input += precision.element().size_bytes();
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[1].output += precision.element().size_bytes();
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid.swap(0, 1);
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input_bytes -= 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].output_bytes += 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input = 0;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].output |= 1;
        assert_eq!(select(precision, &invalid, &spans), None);
        invalid = valid.clone();
        invalid[0].input = u64::MAX - 3;
        assert_eq!(select(precision, &invalid, &spans), None);
        for start in [valid[0].input, 0x200000] {
            invalid = valid.clone();
            for (index, row) in invalid.iter_mut().enumerate() {
                row.output = start + index as u64 * row.output_bytes;
            }
            assert_eq!(select(precision, &invalid, &spans), None);
        }
        assert_eq!(select(precision, &valid, &[0..2, 2..3, 3..6]), None);
        assert_eq!(select(precision, &valid, &[2..3, 0..1, 1..2]), None);
        assert_eq!(select(precision, &valid[..2], &spans), None);
        assert_eq!(select(precision, &valid[..1], &spans[..1]), None);
    }
}

#[test]
fn native_projection_packing_preserves_capacity_and_transform_fallback() {
    use ferrum_interfaces::vnext::{
        HadamardApplication, HadamardSigns, HadamardTransformSpec, PhysicalWeightComponentBinding,
    };
    let table = part(MatrixFormat::DenseF16, 11, 33);
    let capacity = |count: usize, input_packed, parts: &[MatrixPart]| {
        packed_projection_rows(
            TokenPrecision::F32,
            input_packed,
            33,
            11,
            parts,
            (0..count).map(|i| i as u64..i as u64 + 1),
            (0..count).map(|i| NativeProjectionRow {
                input: 0x10000 + i as u64 * 132,
                input_bytes: 132,
                output: 0x10000000 + i as u64 * 44,
                output_bytes: 44,
            }),
            &[0x20000000..0x20001000],
        )
    };
    assert_eq!(
        capacity(65535, true, std::slice::from_ref(&table)),
        Some(65535)
    );
    assert_eq!(capacity(65536, true, std::slice::from_ref(&table)), None);
    assert_eq!(capacity(8, false, std::slice::from_ref(&table)), None);
    let transformed = MatrixPart {
        transform: Some(HadamardTransformSpec {
            block_size: std::num::NonZeroU32::new(1).unwrap(),
            signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new("component.shared-signs").unwrap(),
            )),
            application: HadamardApplication::BeforeMatmul {
                input_permutation: None,
            },
        }),
        ..table
    };
    assert_eq!(capacity(8, true, &[transformed]), None);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn packed_native_projection_matches_f64_and_scalar_with_guarded_fallback_on_cuda() {
    let context = CudaContext::new(0).expect("native projection conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for format in std::iter::once(MatrixFormat::DenseF16).chain(FORMATS.map(MatrixFormat::Block)) {
        let (weight_bytes, decoded, width) = projection_matrix(format);
        let mut weights = vec![0xAB; 16];
        weights.extend_from_slice(&weight_bytes);
        weights.extend_from_slice(&[0xAB; 16]);
        let weight = stream.clone_htod(&weights).unwrap();
        let (wp, _weight_guard) = weight.device_ptr(&stream);
        let table = MatrixPart {
            output_offset: 2,
            ..part(format, 7, width as u32)
        };
        for precision in [TokenPrecision::F16, TokenPrecision::F32] {
            let element_bytes = precision.element().size_bytes() as usize;
            // Canonical decode rows cross the native tile tail. Mixed spans,
            // output padding and reversed output locations exercise fallback.
            let mut cases = [2, 3, 8, 9, 32]
                .map(|count| {
                    (
                        (0..count)
                            .map(|i| i as u64..i as u64 + 1)
                            .collect::<Vec<_>>(),
                        (0..count).collect::<Vec<_>>(),
                        true,
                    )
                })
                .to_vec();
            cases.push((vec![0..2, 2..3, 3..6], vec![0, 1, 2], false));
            cases.push((vec![0..1, 1..2, 2..3], vec![0, 2, 4], false));
            cases.push((vec![0..1, 1..2, 2..3], vec![2, 1, 0], false));
            for (spans, output_rows, expect_packed) in cases {
                let input_rows = spans.iter().map(|range| range.end).max().unwrap() as usize;
                let mut input = vec![0xAB; 16];
                input.extend((0..input_rows * width).flat_map(|i| {
                    scalar_bytes(((i * 13 % 31) as f32 - 15.0) / 32.0 + 0.0001, precision)
                }));
                input.extend_from_slice(&[0xAB; 16]);
                if precision == TokenPrecision::F32 {
                    assert!(input[16..input.len() - 16].chunks_exact(4).any(|bytes| {
                        let value = scalar_value(bytes, precision);
                        f16::from_f32(value).to_f32() != value
                    }));
                }
                let x = stream.clone_htod(&input).unwrap();
                let (xp, _input_guard) = x.device_ptr(&stream);
                let output_bytes = (output_rows.iter().max().unwrap() + 1) * 11 * element_bytes;
                let blank = vec![0xCD; output_bytes + 32];
                let mut y = stream.clone_htod(&blank).unwrap();
                let mut scalar = stream.clone_htod(&blank).unwrap();
                let (yp, y_guard) = y.device_ptr_mut(&stream);
                let (sp, scalar_guard) = scalar.device_ptr_mut(&stream);
                let selected = spans
                    .iter()
                    .zip(&output_rows)
                    .map(|(span, &out)| NativeProjectionRow {
                        input: xp
                            + 16
                            + last_token(span.clone()).unwrap() * (width * element_bytes) as u64,
                        input_bytes: (width * element_bytes) as u64,
                        output: yp + 16 + (out * 11 * element_bytes) as u64,
                        output_bytes: (11 * element_bytes) as u64,
                    })
                    .collect::<Vec<_>>();
                let packed = packed_projection_rows(
                    precision,
                    true,
                    width as u64,
                    11,
                    std::slice::from_ref(&table),
                    spans.iter().cloned(),
                    selected.iter().copied(),
                    &[wp + 16..wp + 16 + weight_bytes.len() as u64],
                );
                assert_eq!(
                    packed.is_some(),
                    expect_packed,
                    "{format:?}, {} rows",
                    spans.len()
                );
                if let Some(count) = packed {
                    kernels
                        .transformed_linear(
                            &stream,
                            selected[0].input,
                            wp + 16,
                            selected[0].output,
                            &table,
                            count,
                            11,
                            precision.element(),
                            0,
                            0,
                        )
                        .unwrap();
                } else {
                    for row in &selected {
                        kernels
                            .transformed_linear(
                                &stream,
                                row.input,
                                wp + 16,
                                row.output,
                                &table,
                                1,
                                11,
                                precision.element(),
                                0,
                                0,
                            )
                            .unwrap();
                    }
                }
                for row in &selected {
                    kernels
                        .transformed_linear(
                            &stream,
                            row.input,
                            wp + 16,
                            sp + (row.output - yp),
                            &table,
                            1,
                            11,
                            precision.element(),
                            0,
                            0,
                        )
                        .unwrap();
                }
                drop(y_guard);
                drop(scalar_guard);
                let actual = stream.clone_dtoh(&y).unwrap();
                let scalar = stream.clone_dtoh(&scalar).unwrap();
                let mut written = vec![false; actual.len()];
                for (span, &out) in spans.iter().zip(&output_rows) {
                    let last = last_token(span.clone()).unwrap() as usize;
                    for column in 0..7 {
                        let products = (0..width).map(|k| {
                            let offset = 16 + (last * width + k) * element_bytes;
                            f64::from(scalar_value(
                                &input[offset..offset + element_bytes],
                                precision,
                            )) * f64::from(decoded[column * width + k])
                        });
                        let reference = products.clone().sum::<f64>();
                        let rounding = if precision == TokenPrecision::F16 {
                            0.0009765625
                        } else {
                            f32::EPSILON as f64
                        };
                        let bound = (width as f64 * f32::EPSILON as f64 + rounding)
                            * products.map(f64::abs).sum::<f64>()
                            + 1e-6;
                        let offset = 16 + (out * 11 + 2 + column) * element_bytes;
                        written[offset..offset + element_bytes].fill(true);
                        for (label, bytes) in [("candidate", &actual), ("scalar", &scalar)] {
                            let value = f64::from(scalar_value(
                                &bytes[offset..offset + element_bytes],
                                precision,
                            ));
                            assert!(value.is_finite() && (value - reference).abs() <= bound,
                                "{label} {format:?} rows {} output {out}/{column}: {value} vs {reference}, bound {bound}", spans.len());
                        }
                    }
                }
                for (index, &is_written) in written.iter().enumerate() {
                    if !is_written {
                        assert_eq!(actual[index], 0xCD, "candidate output guard {index}");
                        assert_eq!(scalar[index], 0xCD, "scalar output guard {index}");
                    }
                }
                assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
                assert_eq!(stream.clone_dtoh(&weight).unwrap(), weights);
            }
        }
    }
}
