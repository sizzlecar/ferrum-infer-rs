//! Reuse the same canonical Stored test family for token-selected INT8 rows.
use super::*;
pub(super) const TOKENS: u64 = 6;
pub(super) fn ids(trial: usize) -> [u32; 6] {
    if trial == 0 {
        [1, 0, 1, 2, 4, u32::MAX]
    } else {
        [u32::MAX; 6]
    }
}
pub(super) fn weight_bytes(name: &str) -> Vec<u8> {
    match name {
        "weight" => {
            let mut q = vec![0; (N * K) as usize];
            q[..K as usize].fill(1);
            q[K as usize] = (-128_i8) as u8;
            q[2 * K as usize - 1] = 127;
            q[2 * K as usize..2 * K as usize + 4].copy_from_slice(&[3, 1, 1, 1]);
            q[3 * K as usize..].fill(2);
            q
        }
        "weight_scale" => [1.000244140625_f32, 0.03125, 0.1, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        _ => unreachable!("embedding uses exact I8 table and F32[V,1] scales"),
    }
}
#[test]
fn stored_embedding_source_dtype_and_inverse_hadamard_match_independent_impulse_oracle() {
    for dtype in [
        ComfyEmbeddingDequantizedDtype::F32,
        ComfyEmbeddingDequantizedDtype::Bf16ThenF32,
    ] {
        let c = Config {
            w4: false,
            embedding: Some(dtype),
        };
        let (_directory, source) = checkpoint(c);
        assert_eq!(source.modules().get(MODULE), Some(&c.metadata()));
        c.schema().validate(Family.family_id()).unwrap();
        TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(c.profile()))
            .unwrap();
        let matrix = ComfyQuantizedMatrix::from_archive(source.archive(), MODULE).unwrap();
        let actual = matrix.embedding_f32_reference(&ids(0), dtype).unwrap();
        assert_eq!(actual.len(), (TOKENS * K) as usize);
        let signs = [
            [1_i32, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ];
        for col in 0..K as usize {
            let (mut r, mut input, mut sign) = (col % 256, if col < 256 { 0 } else { 255 }, 1_i32);
            for _ in 0..4 {
                sign *= signs[r % 4][input % 4];
                r /= 4;
                input /= 4;
            }
            let value = if col < 256 {
                -128.0_f64 * 0.03125
            } else {
                127.0_f64 * 0.03125
            };
            let reference = (value * f64::from(sign) / 16.0) as f32;
            let reference = if dtype == ComfyEmbeddingDequantizedDtype::Bf16ThenF32 {
                half::bf16::from_f32(reference).to_f32()
            } else {
                reference
            };
            assert_eq!(actual[col], reference);
            assert_eq!(actual[2 * K as usize + col], reference);
        }
        let constant = if dtype == ComfyEmbeddingDequantizedDtype::F32 {
            1.000244140625
        } else {
            1.0
        };
        assert!(actual[K as usize..2 * K as usize]
            .iter()
            .all(|v| *v == constant));
        assert_eq!(actual[3 * K as usize + 3], -2.0_f32.powi(-30));
        assert!(actual[4 * K as usize..].iter().all(|v| *v == 0.0));
        assert!(matrix
            .embedding_f32_reference(&ids(1), dtype)
            .unwrap()
            .iter()
            .all(|v| *v == 0.0));
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host; real canonical Stored gather/scale/inverse-H256 and explicit dtype boundary"]
fn canonical_stored_int8_embedding_executes_repeated_and_invalid_ids_with_explicit_source_dtype() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for dtype in [
        ComfyEmbeddingDequantizedDtype::F32,
        ComfyEmbeddingDequantizedDtype::Bf16ThenF32,
    ] {
        run_cuda(Config {
            w4: false,
            embedding: Some(dtype),
        });
    }
}
