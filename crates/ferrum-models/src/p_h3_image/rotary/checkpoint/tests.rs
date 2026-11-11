use super::*;
use crate::p_h3_image::test_fixture::tiny_transformer;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

struct Fixture {
    _directory: tempfile::TempDir,
    source: ComfyCanonicalWeightSource,
}
fn fixture(dtype: safetensors::Dtype, shape: Vec<usize>, bytes: &[u8]) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let tensor = safetensors::tensor::TensorView::new(dtype, shape, bytes).unwrap();
    safetensors::tensor::serialize_to_file(
        [("rope.inv_freq", tensor)],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        ComfyCanonicalWeightSource::new(SafetensorsArchive::open(directory.path()).unwrap())
            .unwrap();
    Fixture {
        _directory: directory,
        source,
    }
}
fn original() -> Fixture {
    let bytes: Vec<_> = [1.0_f32, 0.125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    fixture(safetensors::Dtype::F32, vec![2], &bytes)
}
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn id(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}
fn values(t: &ManagedHostTensor) -> Vec<u16> {
    t.bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn original_frequency_parameters_drive_both_managed_tables() {
    let data = original();
    let config = tiny_transformer(16, 2);
    let rope = PH3ImageCheckpointRotary::new(&config, &data.source).unwrap();
    assert!(std::ptr::eq(rope.source(), &data.source));
    assert_eq!(
        rope.inverse_bytes.as_ptr(),
        data.source
            .archive()
            .tensor("rope.inv_freq")
            .unwrap()
            .bytes()
            .as_ptr()
    );
    let host = account(96);
    let positions = [[8.0, 2.0, -4.0], [1.0, 0.0, 0.5]];
    let (cos, sin) = rope
        .managed_bf16_frequencies(&host, &positions, id("value.cos"), id("value.sin"))
        .unwrap();
    assert_eq!(cos.tensor().dimensions(), [2, 12]);
    assert_eq!(cos.tensor().element_type(), ElementType::Bf16);
    assert_eq!(host.usage().unwrap().committed_bytes(), 96);
    let cos_values = values(&cos);
    let sin_values = values(&sin);
    for row in 0..2 {
        for axis in 0..3 {
            for (f, inverse) in [1.0_f64, 0.125].into_iter().enumerate() {
                // Independent F64 trig at exactly representable angles. The
                // selected inputs are away from BF16 rounding ties.
                let angle = positions[row][axis] * inverse;
                let c = bf16::from_f32(libm::cos(angle) as f32).to_bits();
                let s = bf16::from_f32(libm::sin(angle) as f32).to_bits();
                for offset in [0, 6] {
                    assert_eq!(cos_values[row * 12 + axis * 2 + f + offset], c);
                    assert_eq!(sin_values[row * 12 + axis * 2 + f + offset], s);
                }
            }
        }
    }
    let mut regenerated_cos = [0.0; 24];
    let mut regenerated_sin = [0.0; 24];
    PH3ImageRotaryEmbedding::new(&config)
        .unwrap()
        .write_frequencies(&positions, &mut regenerated_cos, &mut regenerated_sin)
        .unwrap();
    assert_ne!(
        cos_values[1],
        bf16::from_f32(regenerated_cos[1]).to_bits(),
        "must not regenerate theta^-1/2 instead of reading .125"
    );
    drop((cos, sin));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn second_allocation_failure_and_late_invalid_position_leave_no_retained_pair() {
    let data = original();
    let config = tiny_transformer(16, 2);
    let rope = PH3ImageCheckpointRotary::new(&config, &data.source).unwrap();
    let host = account(95);
    assert!(rope
        .managed_bf16_frequencies(&host, &[[0.0; 3]; 2], id("value.cos"), id("value.sin"))
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    for positions in [vec![[0.0; 3], [0.0, f64::MAX, 0.0]], vec![]] {
        assert!(rope
            .managed_bf16_frequencies(&host, &positions, id("value.cos"), id("value.sin"))
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert!(rope
        .managed_bf16_frequencies(&host, &[[0.0; 3]], id("value.same"), id("value.same"))
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn checkpoint_dtype_shape_and_all_frequency_values_are_validated() {
    let config = tiny_transformer(16, 2);
    let wrong_dtype = fixture(safetensors::Dtype::BF16, vec![2], &[0; 4]);
    assert!(PH3ImageCheckpointRotary::new(&config, &wrong_dtype.source).is_err());
    let wrong_shape = fixture(safetensors::Dtype::F32, vec![1, 2], &[0; 8]);
    assert!(PH3ImageCheckpointRotary::new(&config, &wrong_shape.source).is_err());
    for value in [f32::NAN, f32::INFINITY] {
        let bytes: Vec<_> = [1.0_f32, value]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let invalid = fixture(safetensors::Dtype::F32, vec![2], &bytes);
        assert!(PH3ImageCheckpointRotary::new(&config, &invalid.source).is_err());
    }
}
