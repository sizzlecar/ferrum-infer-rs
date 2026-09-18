//! Optional local-source validation; no device allocation or model inference.

use super::*;
use ferrum_interfaces::vnext::{ModelSourceKind, OriginalModelSource, OriginalModelSources};

#[test]
#[ignore = "requires local Hadamard GGUF and matching semantic/tokenizer source directories"]
fn prepares_real_hadamard_gguf_with_shared_signs_and_packed_payloads() {
    let semantic_root =
        std::env::var("FERRUM_TEST_SEMANTIC_DIR").expect("FERRUM_TEST_SEMANTIC_DIR");
    let tokenizer_root =
        std::env::var("FERRUM_TEST_TOKENIZER_DIR").unwrap_or_else(|_| semantic_root.clone());
    let gguf_path = std::env::var("FERRUM_TEST_GGUF_PATH").expect("FERRUM_TEST_GGUF_PATH");
    let original = |kind, location| OriginalModelSource {
        kind,
        location,
        requested_revision: None,
    };
    let native = NativeGgufFile::open(&gguf_path).unwrap();
    let metadata = native
        .hadamard()
        .expect("source-declared Hadamard metadata");
    let sources = Arc::new(
        ProductionModelSourceBundle::open(
            &semantic_root,
            &tokenizer_root,
            ProductionWeightArtifact::gguf_file(&gguf_path),
            OriginalModelSources {
                semantic: original(ModelSourceKind::LocalDirectory, semantic_root.clone()),
                tokenizer: original(ModelSourceKind::LocalDirectory, tokenizer_root.clone()),
                weights: original(ModelSourceKind::LocalFile, gguf_path.clone()),
            },
        )
        .unwrap(),
    );
    let defined = define_from_sources(sources).unwrap();
    let master = defined
        .prepare(&F32_MASTER_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let f16 = defined
        .prepare(&F16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    assert!(Arc::ptr_eq(master.weight_source(), f16.weight_source()));
    assert_eq!(
        master.family().weight_schema(),
        f16.family().weight_schema()
    );
    let schema = master.family().weight_schema();
    let mut packed_components = 0_usize;
    let mut packed_bytes = 0_u64;
    let mut sign_components = 0_usize;
    for component in &schema.components {
        let payload = master.weights().component(component).unwrap();
        assert_eq!(
            payload.bytes().len() as u64,
            component.physical_bytes().unwrap()
        );
        if component.role == WeightComponentRole::TransformSigns {
            sign_components += 1;
            let GgufHadamardSigns::Explicit(by_width) = metadata.signs() else {
                panic!("identity signs must not allocate a component");
            };
            let expected = by_width.get(&component.dimensions[0]).unwrap();
            let actual: Vec<_> = payload
                .bytes()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(
                actual,
                expected.iter().map(|sign| *sign as f32).collect::<Vec<_>>()
            );
        } else if matches!(component.encoding, WeightEncoding::BlockQuantized(_)) {
            packed_components += 1;
            packed_bytes += component.physical_bytes().unwrap();
            let [name] = component.external_names.as_slice() else {
                panic!("native packed components must retain their individual source tensors");
            };
            let info = native.tensor_info(name).unwrap();
            assert_eq!(component.encoding, info.encoding);
            assert_eq!(
                payload.bytes().len(),
                native.tensor_byte_slice(name).unwrap().len()
            );
            let second = master.weights().component(component).unwrap();
            assert_eq!(payload.bytes().as_ptr(), second.bytes().as_ptr());
        }
    }
    let expected_signs = match metadata.signs() {
        GgufHadamardSigns::Identity => 0,
        GgufHadamardSigns::Explicit(_) => metadata
            .weights()
            .values()
            .map(|weight| weight.input_width())
            .collect::<BTreeSet<_>>()
            .len(),
    };
    assert_eq!(sign_components, expected_signs);
    assert!(
        packed_components > 0,
        "fixture must exercise native packed storage"
    );
    eprintln!(
        "prepared source: tensors={}, transforms={}, logical_weights={}, packed_components={packed_components}, packed_bytes={packed_bytes}, shared_sign_components={sign_components}",
        native.tensor_count(), metadata.weights().len(), schema.tensors.len(),
    );
}
