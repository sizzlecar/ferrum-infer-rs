//! Hardware-independent numerical oracle for the declared activation basis.
//! Accumulates in F64 and rounds only the completed transform to F32. This is
//! a test reference, not a production CPU model provider. Small transforms
//! below are also checked against the direct Walsh matrix, independently of
//! the butterfly scheduling used by device kernels.

use ferrum_interfaces::vnext::{HadamardApplication, HadamardSigns, HadamardTransformSpec};

pub(crate) fn apply_reference(
    input: &[f32],
    rows: usize,
    width: usize,
    spec: &HadamardTransformSpec,
    signs: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    spec.validate(u64::try_from(width).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    if rows == 0 || rows.checked_mul(width) != Some(input.len()) {
        return Err("Hadamard reference requires complete nonempty rows".into());
    }
    match (&spec.signs, signs) {
        (HadamardSigns::Identity, None) => {}
        (HadamardSigns::Explicit(_), Some(values))
            if values.len() == width && values.iter().all(|value| matches!(*value, -1.0 | 1.0)) => {
        }
        _ => return Err("Hadamard signs must match the declared full-width binding".into()),
    }
    let block = spec.block_size.get() as usize;
    let normalization = (block as f64).sqrt().recip();
    let sign = |feature: usize| signs.map_or(1.0, |values| f64::from(values[feature]));
    let mut result = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mut values = vec![0.0_f64; width];
        match &spec.application {
            HadamardApplication::BeforeMatmul { input_permutation } => {
                if let Some(permutation) = input_permutation {
                    let inner = permutation.inner_extent as usize;
                    let first = permutation.first_outer_extent as usize;
                    let second = permutation.second_outer_extent as usize;
                    for a in 0..first {
                        for b in 0..second {
                            for c in 0..inner {
                                let source = c + inner * (a + first * b);
                                let destination = c + inner * (b + second * a);
                                values[destination] = f64::from(row[source]) * sign(destination);
                            }
                        }
                    }
                } else {
                    for (feature, value) in row.iter().enumerate() {
                        values[feature] = f64::from(*value) * sign(feature);
                    }
                }
            }
            HadamardApplication::AfterEmbeddingLookup => {
                for (destination, value) in values.iter_mut().zip(row) {
                    *destination = f64::from(*value);
                }
            }
        }
        for chunk in values.chunks_exact_mut(block) {
            let mut span = 1;
            while span < block {
                for pair in chunk.chunks_exact_mut(span * 2) {
                    let (left, right) = pair.split_at_mut(span);
                    for (left, right) in left.iter_mut().zip(right) {
                        let a = *left;
                        let b = *right;
                        *left = a + b;
                        *right = a - b;
                    }
                }
                span *= 2;
            }
        }
        for (feature, value) in values.into_iter().enumerate() {
            let inverse_sign = match spec.application {
                HadamardApplication::AfterEmbeddingLookup => sign(feature),
                HadamardApplication::BeforeMatmul { .. } => 1.0,
            };
            result.push((value * normalization * inverse_sign) as f32);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{GroupedFeatureTranspose, PhysicalWeightComponentBinding};

    fn spec(block: u32, signed: bool, inverse: bool) -> HadamardTransformSpec {
        HadamardTransformSpec {
            block_size: block.try_into().unwrap(),
            signs: if signed {
                HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                    ferrum_interfaces::vnext::WeightId::new("test.signs").unwrap(),
                ))
            } else {
                HadamardSigns::Identity
            },
            application: if inverse {
                HadamardApplication::AfterEmbeddingLookup
            } else {
                HadamardApplication::BeforeMatmul {
                    input_permutation: None,
                }
            },
        }
    }

    fn direct_walsh(input: &[f32], block: usize) -> Vec<f32> {
        input
            .chunks_exact(block)
            .flat_map(|chunk| {
                (0..block).map(move |row| {
                    let sum: f64 = chunk
                        .iter()
                        .enumerate()
                        .map(|(column, value)| {
                            let sign = if (row & column).count_ones() % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            f64::from(*value) * sign
                        })
                        .sum();
                    (sum / (block as f64).sqrt()) as f32
                })
            })
            .collect()
    }

    #[test]
    fn butterflies_match_independent_walsh_matrix_and_direction() {
        let values: Vec<f32> = (0..48).map(|i| (i as f32 - 19.0) * 0.25).collect();
        let signs: Vec<f32> = (0..24)
            .map(|i| if i % 5 < 2 { -1.0 } else { 1.0 })
            .collect();
        for block in [1, 2, 4, 8] {
            let signed: Vec<f32> = values
                .iter()
                .enumerate()
                .map(|(i, x)| x * signs[i % 24])
                .collect();
            let forward =
                apply_reference(&values, 2, 24, &spec(block, true, false), Some(&signs)).unwrap();
            assert_eq!(forward, direct_walsh(&signed, block as usize));
            let expected: Vec<f32> = direct_walsh(&values, block as usize)
                .iter()
                .enumerate()
                .map(|(i, x)| x * signs[i % 24])
                .collect();
            let inverse =
                apply_reference(&values, 2, 24, &spec(block, true, true), Some(&signs)).unwrap();
            assert_eq!(inverse, expected);
            if block > 1 {
                assert_ne!(inverse, forward);
            }
        }
    }

    #[test]
    fn permutation_precedes_full_width_signs_and_block_transform() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let signs = [
            1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
        ];
        let mut contract = spec(4, true, false);
        contract.application = HadamardApplication::BeforeMatmul {
            input_permutation: Some(GroupedFeatureTranspose {
                inner_extent: 2,
                first_outer_extent: 2,
                second_outer_extent: 3,
            }),
        };
        // Literal transpose of fastest-axis-first [2,2,3], deliberately
        // asymmetric so an inverse permutation cannot satisfy this fixture.
        let reordered = [0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0];
        let signed: Vec<_> = reordered
            .iter()
            .zip(signs)
            .map(|(x, sign)| x * sign)
            .collect();
        assert_eq!(
            apply_reference(&input, 1, 12, &contract, Some(&signs)).unwrap(),
            direct_walsh(&signed, 4)
        );
    }

    #[test]
    fn multiple_1024_blocks_restore_signs_without_half_narrowing() {
        let width = 3072;
        let input: Vec<f32> = (0..width * 2)
            .map(|i| {
                if i % 701 == 0 {
                    131008.0
                } else {
                    (i % 41) as f32 - 20.0
                }
            })
            .collect();
        let signs: Vec<f32> = (0..width)
            .map(|i| {
                if (i / 1024 + i % 13) % 3 == 0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();
        let transformed =
            apply_reference(&input, 2, width, &spec(1024, true, false), Some(&signs)).unwrap();
        let restored = apply_reference(
            &transformed,
            2,
            width,
            &spec(1024, true, true),
            Some(&signs),
        )
        .unwrap();
        for (actual, expected) in restored.iter().zip(input) {
            assert!(actual.is_finite());
            assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
        }
    }

    #[test]
    fn malformed_geometry_or_sign_payload_is_rejected() {
        let input = [1.0; 8];
        assert!(apply_reference(&input, 1, 8, &spec(3, false, false), None).is_err());
        assert!(apply_reference(&input, 1, 6, &spec(2, false, false), None).is_err());
        assert!(apply_reference(&input, 1, 8, &spec(4, true, false), Some(&[1.0; 4])).is_err());
        assert!(apply_reference(&input, 1, 8, &spec(4, true, false), Some(&[0.0; 8])).is_err());
        assert!(apply_reference(&input, 1, 8, &spec(4, false, false), Some(&[1.0; 8])).is_err());
        assert!(apply_reference(&input, 1, 8, &spec(4, true, false), None).is_err());
        let mut contract = spec(4, false, false);
        contract.application = HadamardApplication::BeforeMatmul {
            input_permutation: Some(GroupedFeatureTranspose {
                inner_extent: u64::MAX,
                first_outer_extent: 2,
                second_outer_extent: 2,
            }),
        };
        assert!(apply_reference(&input, 1, 8, &contract, None).is_err());
    }
}
