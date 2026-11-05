//! Actual safetensors storage provenance, independent of a device backend.
use ferrum_quantization::{ManagedDenseArchiveSource, SafetensorsArchive};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

fn fixture() -> (tempfile::TempDir, SafetensorsArchive) {
    let directory = tempfile::tempdir().unwrap();
    let bytes: Vec<_> = [0x3f80_0001u32, 0x8000_0000]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    serialize_to_file(
        [("w", TensorView::new(Dtype::F32, vec![2], &bytes).unwrap())],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive)
}
fn component(dtype: ElementType) -> WeightComponentSpec {
    WeightComponentSpec {
        id: "component.readonly.fixture".to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["w".into()],
        dimensions: vec![2],
        encoding: WeightEncoding::Dense {
            element_type: dtype,
        },
        required: true,
    }
}

#[test]
fn readonly_mapped_safetensors_payload_keeps_exact_bytes_after_archive_drop() {
    let (_directory, archive) = fixture();
    let spec = component(ElementType::F32);
    let payload = archive.component(&spec).unwrap();
    let retained = payload.retained_host_memory().unwrap();
    assert!(retained.managed_memory_class().is_none());
    let mapped = retained.readonly_file_mapping().unwrap();
    assert!(std::ptr::eq(mapped.bytes(), payload.bytes()));
    assert_eq!(mapped.length_bytes(), 8);
    assert!(mapped.offset_bytes() > 0); // The safetensors header precedes data.
    let tail = mapped.slice(4..8).unwrap();
    assert_eq!(tail.offset_bytes(), mapped.offset_bytes() + 4);
    for range in [0..0, 7..9, 9..8, 0..usize::MAX] {
        assert!(mapped.slice(range).is_err());
    }
    drop(payload);
    drop(archive);
    assert_eq!(mapped.bytes(), &[1, 0, 0x80, 0x3f, 0, 0, 0, 0x80]);
    drop(mapped);
    assert_eq!(tail.bytes(), &[0, 0, 0, 0x80]);
}

#[test]
fn readonly_mapped_proof_never_attaches_to_equal_content_other_mapping() {
    let (directory, archive) = fixture();
    let other = SafetensorsArchive::open(directory.path()).unwrap();
    let spec = component(ElementType::F32);
    let payload = archive.component(&spec).unwrap();
    let other_payload = other.component(&spec).unwrap();
    assert_eq!(payload.bytes(), other_payload.bytes());
    assert!(!std::ptr::eq(payload.bytes(), other_payload.bytes()));
    let wrong = other_payload.retained_host_memory().unwrap().clone();
    assert!(payload.with_retained_host_memory(wrong).is_err());
    let mut invalid = spec;
    invalid.dimensions = vec![3];
    assert!(archive.component(&invalid).is_err());
}

#[test]
fn converted_heap_and_accounted_weight_payloads_never_inherit_mapped_proof() {
    let (_directory, archive) = fixture();
    let spec = component(ElementType::F16);
    // Original unmanaged conversion owns a small Vec; being derived from a
    // file mapping does not make that converted allocation a mapping.
    let ordinary = archive.component(&spec).unwrap();
    assert!(ordinary.retained_host_memory().is_none());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 4,
            },
        )
        .unwrap(),
    );
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let managed = source.component(&spec).unwrap();
    assert_eq!(managed.bytes(), ordinary.bytes());
    let retained = managed.retained_host_memory().unwrap();
    assert_eq!(
        retained.managed_memory_class(),
        Some(ferrum_runtime::HostMemoryClass::WeightBacking)
    );
    assert!(retained.readonly_file_mapping().is_none());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 4);
    drop(managed);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
