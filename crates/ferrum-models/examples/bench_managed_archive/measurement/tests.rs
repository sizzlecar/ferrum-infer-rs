use super::*;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{
    ElementType, HostMemoryClassLimits, HostMemoryLimits, WeightComponentRole, WeightEncoding,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

#[test]
fn fresh_and_reused_measurements_report_real_idle_charge_and_separate_optional_hashes() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = [1.0_f32, -0.0, 2.0, 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "value",
            TensorView::new(Dtype::F32, vec![1, 4], &bytes).unwrap(),
        )],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let component = WeightComponentSpec {
        id: "component.fixture.value".to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["value".into()],
        dimensions: vec![1, 4],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
        required: true,
    };
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            127,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 127,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap();
    let mut expected_hash = None;
    for backing in [Backing::Fresh, Backing::Reuse] {
        for hash_output in [false, true] {
            let args = Arguments {
                semantic_root: directory.path().to_owned(),
                tokenizer_root: directory.path().to_owned(),
                weights_file: directory.path().join("model.safetensors"),
                numerical_profile: "qwen3_5.f16".into(),
                host_bytes: 127,
                host_alignment: 64,
                iterations: 1,
                mode: Mode::Diagnostic,
                backing,
                touch_stride_bytes: 0,
                component_id: None,
                hash_output,
                conversion_workers: std::num::NonZeroUsize::MIN,
            };
            let installation = Installation::new(&archive, &host, backing, 8).unwrap();
            // Two components in the same installation exercise actual reuse;
            // each outer iteration constructs a new independently billed slot.
            for _ in 0..2 {
                let sample = measure(&installation, &host, &component, &args)
                    .unwrap_or_else(|failure| panic!("{}", failure.error));
                assert_eq!(sample.payload_bytes, 8);
                assert_eq!(sample.live_host_backing_bytes, 127);
                assert_eq!(
                    sample.host_committed_after_release_bytes,
                    if backing == Backing::Reuse { 127 } else { 0 }
                );
                assert_eq!(sample.output_sha256.is_some(), hash_output);
                assert_eq!(sample.output_hash_wall_ns.is_some(), hash_output);
                assert!(sample.diagnostic.as_ref().unwrap().completed);
                if let Some(hash) = sample.output_sha256 {
                    let expected = expected_hash.get_or_insert_with(|| hash.clone());
                    assert_eq!(&hash, &*expected);
                }
            }
            drop(installation);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        }
    }
}
