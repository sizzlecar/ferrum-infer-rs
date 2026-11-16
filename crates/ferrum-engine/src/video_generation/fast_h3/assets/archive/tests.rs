use super::*;
use crate::video_generation::fast_h3::assets::*;
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::{collections::BTreeMap, sync::Arc};

fn limits() -> FastH3AssetLimits {
    FastH3AssetLimits {
        metadata: FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
        maximum_tokenizer_bytes: 4096,
        maximum_shards_per_archive: 2,
        maximum_tensors_per_archive: 4,
        maximum_total_source_bytes: 16,
        maximum_exported_text_layers: 64,
        transformer: FastH3WeightLimits {
            maximum_tensors: 4,
            maximum_source_bytes: 16,
            maximum_skip_metadata_bytes: 4096,
        },
        video_decoder: FastH3VideoDecoderWeightLimits {
            maximum_tensors: 4,
            maximum_source_bytes: 16,
            maximum_metadata_bytes: 4096,
        },
        processor: Qwen3VLVisionProcessorLimits {
            maximum_metadata_bytes: 4096,
            image: ImageInputLimits::default(),
            maximum_patch_rows: 16,
            maximum_patch_bytes: 4096,
        },
    }
}
fn write(directory: &Path, name: &str, tensor: &str) {
    let bytes = [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat();
    serialize_to_file(
        BTreeMap::from([(
            tensor,
            TensorView::new(Dtype::F32, vec![2], &bytes).unwrap(),
        )]),
        &None,
        &directory.join(name),
    )
    .unwrap();
}

#[test]
fn imported_mapping_is_shared_and_extent_failure_does_not_replace_owner() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "one.safetensors", "one");
    write(dir.path(), "two.safetensors", "two");
    let mut cap = limits();
    cap.maximum_total_source_bytes = 8;
    let mut extent = ImportExtent::new(cap);
    let original = Arc::new(
        extent
            .open(&FastH3ArchivePath::File {
                path: dir.path().join("one.safetensors"),
            })
            .unwrap(),
    );
    let shared = original.clone();
    assert!(Arc::ptr_eq(&original, &shared));
    assert!(extent
        .open(&FastH3ArchivePath::File {
            path: dir.path().join("two.safetensors")
        })
        .is_err());
    assert_eq!(extent.payload_bytes(), 8);
    drop(original);
    assert_eq!(
        shared.tensor("one").unwrap().bytes(),
        [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()
    );
}

#[test]
fn index_shards_and_physical_inventory_are_bounded_before_mapping() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "one.safetensors", "one");
    let path = dir.path().join("weights.index.json");
    std::fs::write(&path, br#"{"weight_map":{"one":"../one.safetensors"}}"#).unwrap();
    let mut extent = ImportExtent::new(limits());
    assert!(extent
        .open(&FastH3ArchivePath::Index { path: path.clone() })
        .is_err());
    assert_eq!(extent.payload_bytes(), 0);
    std::fs::write(&path, br#"{"weight_map":{"one":"one.safetensors"}}"#).unwrap();
    let archive = extent.open(&FastH3ArchivePath::Index { path }).unwrap();
    assert_eq!(archive.tensor_count(), 1);
    assert_eq!(extent.payload_bytes(), 8);
    let mut cap = limits();
    cap.metadata.max_file_bytes = 8;
    assert!(ImportExtent::new(cap)
        .open(&FastH3ArchivePath::File {
            path: dir.path().join("one.safetensors")
        })
        .is_err());
}
