use std::sync::Arc;

use ferrum_quantization::{ArchiveTensor, SafetensorsArchive, TensorArchive};
use ferrum_runtime::{
    BlockQuantizationSpec, ElementType, RetainedHostMemoryRegion, StableHostMemory, WeightEncoding,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

struct Bytes(Vec<u8>);
// SAFETY: No API mutates or resizes these bytes after creating a retained owner.
unsafe impl StableHostMemory for Bytes {
    fn stable_bytes(&self) -> &[u8] {
        &self.0
    }
}
fn region(bytes: usize) -> RetainedHostMemoryRegion {
    RetainedHostMemoryRegion::new(Arc::new(Bytes(vec![0; bytes])), 0, bytes).unwrap()
}
fn block() -> WeightEncoding {
    WeightEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: "quantization.gguf.q8-0".to_owned().try_into().unwrap(),
        logical_values_per_block: 32,
        bytes_per_block: 34,
    })
}

#[test]
fn archive_view_checks_rows_payload_length_and_overflow_without_conversion() {
    let valid = ArchiveTensor::new("w", "weights.gguf", &[2, 32], block(), region(68)).unwrap();
    assert_eq!(valid.shape(), &[2, 32]);
    assert_eq!(valid.element_type(), None);
    assert_eq!(valid.bytes().len(), 68);
    for (shape, bytes) in [
        (vec![2, 32], 67),
        (vec![32, 2], 68),
        (vec![0, 32], 34),
        (vec![], 34),
        (vec![u64::MAX, 32], 34),
    ] {
        assert!(ArchiveTensor::new("w", "w.gguf", &shape, block(), region(bytes)).is_err());
    }
    let dense = WeightEncoding::Dense {
        element_type: ElementType::Bf16,
    };
    assert!(ArchiveTensor::new("w", "w", &[], dense.clone(), region(2)).is_ok());
    assert!(ArchiveTensor::new("w", "w", &[u64::MAX], dense, region(2)).is_err());
}

#[test]
fn safetensors_inventory_and_payload_share_the_original_retained_mapping() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = [0x80, 0x3f, 0, 0xc0];
    serialize_to_file(
        [(
            "w",
            TensorView::new(Dtype::BF16, vec![1, 2], &bytes).unwrap(),
        )],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let pointer = archive.tensor("w").unwrap().bytes().as_ptr();
    let archive: Arc<dyn TensorArchive> = Arc::new(archive);
    assert_eq!(archive.tensor_count(), 1);
    assert_eq!(archive.tensor_names().collect::<Vec<_>>(), ["w"]);
    let tensor = archive.tensor("w").unwrap();
    assert_eq!(tensor.external_name(), "w");
    assert_eq!(tensor.source_file(), "model.safetensors");
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.bytes().as_ptr(), pointer);
    assert_eq!(tensor.element_type(), Some(ElementType::Bf16));
    let retained = tensor.retained_host_memory().clone();
    drop(tensor);
    drop(archive);
    assert_eq!(retained.bytes(), &bytes);
}

#[test]
fn dense_normalization_reserves_exact_backing_before_rounding_and_retains_its_charge() {
    use ferrum_quantization::ManagedDenseArchiveSource;
    use ferrum_runtime::{
        HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
        WeightComponentRole, WeightComponentSource, WeightComponentSpec,
    };
    let account = |bytes| {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    transfer_staging_bytes: 0,
                    output_bytes: 0,
                    weight_backing_bytes: bytes,
                },
            )
            .unwrap(),
        )
    };
    let directory = tempfile::tempdir().unwrap();
    // Adjacent BF16 midpoints select the even significand in both directions.
    let values = [1.00390625_f32, 1.01171875, -1.00390625, -1.01171875];
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    serialize_to_file(
        [(
            "w",
            TensorView::new(Dtype::F32, vec![1, 4], &bytes).unwrap(),
        )],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let component = WeightComponentSpec {
        id: "component.w".to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["w".into()],
        dimensions: vec![1, 4],
        required: true,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
    };
    let too_small = account(7);
    assert!(ManagedDenseArchiveSource::new(&archive, &too_small)
        .component(&component)
        .is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    let exact = account(8);
    let source = ManagedDenseArchiveSource::new(&archive, &exact);
    let payload = source.component(&component).unwrap();
    let expected: Vec<_> = [0x3f80_u16, 0x3f82, 0xbf80, 0xbf82]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(payload.bytes(), expected);
    assert_eq!(
        exact
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::WeightBacking),
        8
    );
    let retained = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    assert_eq!(exact.usage().unwrap().committed_bytes(), 8);
    drop(retained);
    assert_eq!(exact.usage().unwrap().committed_bytes(), 0);
}
