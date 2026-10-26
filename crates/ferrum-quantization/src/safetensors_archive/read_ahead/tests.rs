use super::*;
use crate::SafetensorsArchive;
use ferrum_runtime::{
    ElementType, WeightComponentRole, WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

#[test]
fn read_ahead_only_contains_complete_tensor_pages_with_a_whole_payload_cap() {
    let page = 4096;
    let start = 123;
    let end = 3 * page + 5;
    let length = (end - start) as u64;
    let actual = full_page_range(start, end, 4 * page, page, length)
        .unwrap()
        .unwrap();
    assert_eq!(actual, page..3 * page);
    assert!(actual.start >= start && actual.end <= end);
    assert_eq!(
        full_page_range(start, end, 4 * page, page, length - 1).unwrap(),
        None
    );
    assert_eq!(
        full_page_range(page, 2 * page, 2 * page, page, page as u64).unwrap(),
        Some(page..2 * page)
    );
    for (start, end) in [(1, page), (page + 1, 2 * page - 1), (page, page)] {
        assert_eq!(
            full_page_range(start, end, 2 * page, page, (2 * page) as u64).unwrap(),
            None
        );
    }
    for (start, end, mapping_bytes, page_size) in [
        (3, 2, 10, page),
        (0, 11, 10, page),
        (0, 10, 10, 0),
        (0, 10, usize::MAX, page),
    ] {
        assert!(full_page_range(start, end, mapping_bytes, page_size, 32).is_err());
    }
}

#[test]
fn read_ahead_policy_rejects_zero_unaddressable_and_unknown_wire_fields() {
    assert_eq!(
        SafetensorsReadAheadPolicy::default(),
        SafetensorsReadAheadPolicy::Off
    );
    for maximum_range_bytes in [0, isize::MAX as u64 + 1, u64::MAX] {
        let policy = SafetensorsReadAheadPolicy::WillNeed {
            maximum_range_bytes,
        };
        assert!(policy.validate().is_err());
        assert!(
            serde_json::from_value::<SafetensorsReadAheadPolicy>(serde_json::json!({
                "will_need": {"maximum_range_bytes": maximum_range_bytes}
            }))
            .is_err()
        );
    }
    for invalid in [
        serde_json::json!({"will_need":{"maximum_range_bytes":-1}}),
        serde_json::json!({"will_need":{"maximum_range_bytes":1,"whole_map":true}}),
        serde_json::json!({"will_need":{}}),
        serde_json::json!("sequential"),
    ] {
        assert!(serde_json::from_value::<SafetensorsReadAheadPolicy>(invalid).is_err());
    }
}

#[test]
fn bounded_read_ahead_preserves_real_component_payload_and_retained_mmap_owner() {
    let directory = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    // SAFETY: sysconf has no pointers or ownership effects.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
    #[cfg(not(unix))]
    let page = 4096;
    assert!(page > 0);
    let bytes = [0x80, 0x3f].repeat(3 * page / 2);
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::BF16, vec![bytes.len() / 2], &bytes).unwrap(),
        )],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let component = WeightComponentSpec {
        id: "component.weight".to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["weight".into()],
        dimensions: vec![(bytes.len() / 2) as u64],
        required: true,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
    };
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    assert_eq!(archive.read_ahead_policy(), SafetensorsReadAheadPolicy::Off);
    let before = archive.component(&component).unwrap();
    let original_pointer = before.bytes().as_ptr();
    let original_region = before.retained_host_memory().unwrap().clone();
    drop(before);
    let policy = SafetensorsReadAheadPolicy::WillNeed {
        maximum_range_bytes: 128 * 1024 * 1024,
    };
    let archive = archive.with_read_ahead(policy).unwrap();
    let metadata = archive.tensors.get("weight").unwrap();
    let range = full_page_range(
        metadata.data_start,
        metadata.data_end,
        archive.shards[metadata.shard].mmap.len(),
        page,
        128 * 1024 * 1024,
    )
    .unwrap()
    .unwrap();
    assert!(range.start >= metadata.data_start && range.end <= metadata.data_end);

    let payload = archive.component(&component).unwrap();
    assert_eq!(payload.bytes(), bytes);
    assert_eq!(payload.bytes().as_ptr(), original_pointer);
    let retained = payload.retained_host_memory().unwrap().clone();
    assert_eq!(retained.bytes().as_ptr(), original_region.bytes().as_ptr());
    assert_eq!(retained.offset_bytes(), original_region.offset_bytes());
    assert_eq!(retained.length_bytes(), original_region.length_bytes());
    drop(payload);

    let mut invalid = component.clone();
    invalid.dimensions[0] += 1;
    assert!(archive.component(&invalid).is_err());
    let mut transcoded = component.clone();
    transcoded.encoding = WeightEncoding::Dense {
        element_type: ElementType::F32,
    };
    let copied = archive.component(&transcoded).unwrap();
    assert!(copied.retained_host_memory().is_none());
    assert!(copied
        .bytes()
        .chunks_exact(4)
        .all(|x| x == 1_f32.to_le_bytes()));
    drop(copied);
    let archive = archive
        .with_read_ahead(SafetensorsReadAheadPolicy::WillNeed {
            maximum_range_bytes: bytes.len() as u64 - 1,
        })
        .unwrap();
    assert_eq!(
        archive.component(&component).unwrap().bytes().as_ptr(),
        original_pointer
    );
    drop(archive);
    drop(original_region);
    assert_eq!(retained.bytes(), bytes);
    for maximum_range_bytes in [0, u64::MAX] {
        assert!(SafetensorsArchive::open(directory.path())
            .unwrap()
            .with_read_ahead(SafetensorsReadAheadPolicy::WillNeed {
                maximum_range_bytes
            })
            .is_err());
    }
}
