use super::*;
use half::bf16;

#[cfg(feature = "cuda-sage-vsa-fp4")]
mod cuda;

fn limits() -> FastH3VsaQkvQuantizeLimits {
    FastH3VsaQkvQuantizeLimits {
        maximum_batches: 2,
        maximum_heads: 64,
        maximum_rows: 4096,
        maximum_input_bytes: 128 << 20,
        maximum_output_bytes: 64 << 20,
        maximum_metadata_bytes: 4096,
    }
}
fn config(batches: u64, heads: u64, query_rows: u64, key_rows: u64) -> FastH3VsaQkvQuantizeConfig {
    FastH3VsaQkvQuantizeConfig {
        batches,
        heads,
        query_rows,
        key_rows,
    }
}

#[test]
fn quantize_family_checks_all_six_extents_and_actual_capacity() {
    let c = config(2, 3, 256, 384);
    let family = FastH3VsaQkvQuantizeFamily::new(limits()).unwrap();
    let p = family.numerical_profiles(&c).unwrap();
    let p = p
        .resolve(&NumericalProfileId::new(FastH3VsaQkvQuantizeFamily::PROFILE_ID).unwrap())
        .unwrap();
    let graph = family.semantic_program(&c, p).unwrap();
    assert_eq!(graph.outputs().len(), 6);
    assert_eq!(
        c.input_shapes().unwrap(),
        [vec![2, 256, 384], vec![2, 384, 384], vec![2, 384, 384]]
    );
    assert_eq!(
        c.output_shapes(),
        [
            vec![2, 3, 256, 64],
            vec![2, 3, 384, 64],
            vec![2, 3, 128, 192],
            vec![2, 3, 256, 8],
            vec![2, 3, 384, 8],
            vec![2, 3, 128, 24]
        ]
    );
    let mut cap = limits();
    let input_bytes = 2 * 3 * 128 * (256 + 384 * 2) * 2;
    let output_bytes = 2 * 3 * (256 + 384 * 2) * 72;
    cap.maximum_input_bytes = input_bytes;
    cap.maximum_output_bytes = output_bytes;
    assert!(c.compile_options(cap).is_ok());
    cap.maximum_input_bytes -= 1;
    assert!(c.compile_options(cap).is_err());
    cap.maximum_input_bytes += 1;
    cap.maximum_output_bytes -= 1;
    assert!(c.compile_options(cap).is_err());
    for invalid in [
        config(0, 3, 256, 384),
        config(2, 0, 256, 384),
        config(2, 3, 255, 384),
        config(2, 3, 384, 384),
        config(2, 3, 512, 384),
    ] {
        assert!(invalid.compile_options(limits()).is_err());
    }
}

#[test]
fn quantize_layout_uses_video_query_suffix_and_all_padded_keys() {
    use crate::fast_h3::*;
    let cp = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let layout = super::super::tests::layout();
    let c =
        FastH3VsaQkvQuantizeConfig::from_layout(&layout, 2, cp.transformer(), limits()).unwrap();
    assert_eq!(
        c.query_rows(),
        (layout.padded_rows() - layout.prefix_tiles() * 128) as u64
    );
    assert_eq!(c.key_rows(), layout.padded_rows() as u64);
    assert!(c.key_rows() > layout.packed().sequence_rows() as u64);
    let f = TypedFamilyRegistration::new(FastH3VsaQkvQuantizeFamily::new(limits()).unwrap());
    assert!(f
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &NumericalProfileId::new("wrong.profile").unwrap()
        )
        .is_err());
}

const MAGNITUDES: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
// Explicit physical-to-logical K row table, one 32-row atom. It is deliberately
// separate from the native thread/lane arithmetic.
const KEY_ROWS: [usize; 32] = [
    0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7,
    14, 15, 22, 23, 30, 31,
];

/// Exact-domain fixture: group maxima are 6 * 2^n and every normalized value is
/// on the E2M1 lattice. It qualifies byte layouts, scale addressing and owners,
/// not arbitrary-data fast division or complete Sage numerical equivalence.
fn operands(c: &FastH3VsaQkvQuantizeConfig, trial: usize) -> [Vec<u16>; 3] {
    std::array::from_fn(|which| {
        let rows = if which == 0 { c.query_rows } else { c.key_rows } as usize;
        let heads = c.heads as usize;
        (0..c.batches as usize * rows * heads * 128)
            .map(|i| {
                let b = i / (rows * heads * 128);
                let r = i / (heads * 128) % rows;
                let h = i / 128 % heads;
                let d = i % 128;
                let (group, lane) = if which == 2 {
                    (r / 16 + d * 7, r % 16)
                } else {
                    (d / 16 + r * 7, d % 16)
                };
                let tag = b * 11 + h * 13 + group + trial * 17 + which * 19;
                let exponent = (tag % 5) as i32 - 2;
                let code = if tag % 11 == 0 {
                    0
                } else if lane == 15 {
                    7
                } else {
                    (lane + tag) % 8
                };
                let sign = if code != 0 && (tag + lane) % 3 == 0 {
                    -1.
                } else {
                    1.
                };
                bf16::from_f32(sign * MAGNITUDES[code] * 2f32.powi(exponent)).to_bits()
            })
            .collect()
    })
}

fn scalar_code(v: f32, scale: f32) -> u8 {
    if scale == 0. {
        assert_eq!(v, 0.);
        return 0;
    }
    let magnitude = f64::from(v.abs()) / f64::from(scale);
    let index = MAGNITUDES
        .iter()
        .position(|&x| f64::from(x) == magnitude)
        .expect("exact E2M1 fixture");
    index as u8 | if v < 0. { 8 } else { 0 }
}
fn scale_code(scale: f32) -> u8 {
    if scale == 0. {
        return 0;
    }
    [
        (0.25, 0x28),
        (0.5, 0x30),
        (1., 0x38),
        (2., 0x40),
        (4., 0x48),
    ]
    .into_iter()
    .find(|&(v, _)| v == scale)
    .expect("exact power-of-two scale")
    .1
}
// Construct the destination traversal from the storage axes: 64-row atoms,
// groups of four columns, 16 row lanes, four row subgroups, four scale columns.
fn swizzle_scales(logical: &[u8], rows: usize, columns: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(rows * columns);
    for row_atom in 0..rows / 64 {
        for column_atom in 0..columns / 4 {
            for lane in 0..16 {
                for row_group in 0..4 {
                    for column in 0..4 {
                        result.push(
                            logical[(row_atom * 64 + row_group * 16 + lane) * columns
                                + column_atom * 4
                                + column],
                        );
                    }
                }
            }
        }
    }
    assert_eq!(result.len(), logical.len());
    result
}
fn oracle(c: &FastH3VsaQkvQuantizeConfig, input: &[Vec<u16>; 3]) -> [Vec<u8>; 6] {
    let mut outputs: [Vec<u8>; 6] = std::array::from_fn(|_| Vec::new());
    for (which, values) in input.iter().enumerate() {
        let sequence = if which == 0 { c.query_rows } else { c.key_rows } as usize;
        for b in 0..c.batches as usize {
            for h in 0..c.heads as usize {
                let (rows, columns) = if which == 2 {
                    (128, sequence)
                } else {
                    (sequence, 128)
                };
                let mut scales = Vec::with_capacity(rows * columns / 16);
                for physical_row in 0..rows {
                    for group in 0..columns / 16 {
                        let values: Vec<f32> = (0..16)
                            .map(|lane| {
                                let col = group * 16 + lane;
                                let (r, d) = match which {
                                    0 => (physical_row, col),
                                    1 => {
                                        (physical_row / 32 * 32 + KEY_ROWS[physical_row % 32], col)
                                    }
                                    _ => (col, physical_row),
                                };
                                bf16::from_bits(
                                    values[((b * sequence + r) * c.heads as usize + h) * 128 + d],
                                )
                                .to_f32()
                            })
                            .collect();
                        let amax = values.iter().map(|x| x.abs()).fold(0f32, f32::max);
                        let scale = amax / 6.;
                        scales.push(scale_code(scale));
                        for pair in values.chunks_exact(2) {
                            outputs[which].push(
                                scalar_code(pair[0], scale) | scalar_code(pair[1], scale) << 4,
                            );
                        }
                    }
                }
                outputs[which + 3].extend(swizzle_scales(&scales, rows, columns / 16));
            }
        }
    }
    outputs
}

#[test]
fn quantize_independent_fixture_covers_permutation_scale_atoms_and_zero_groups() {
    let mut permutation = KEY_ROWS;
    permutation.sort_unstable();
    assert_eq!(permutation, std::array::from_fn(|i| i));
    for c in [config(1, 2, 128, 256), config(2, 3, 256, 384)] {
        let first = oracle(&c, &operands(&c, 0));
        let second = oracle(&c, &operands(&c, 1));
        for ((a, b), shape) in first.iter().zip(&second).zip(c.output_shapes()) {
            assert_eq!(a.len() as u64, shape.iter().product::<u64>());
            assert_ne!(a, b);
        }
        for scale in &first[3..] {
            for code in [0, 0x28, 0x30, 0x38, 0x40, 0x48] {
                assert!(scale.contains(&code));
            }
        }
    }
}
