use super::super::arguments::{Backing, Mode};
use super::*;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{
    ElementType, HostMemoryClassLimits, HostMemoryLimits, WeightComponentRole, WeightEncoding,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

fn host(total: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: total,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn component(names: &[&str], shape: &[u64], dtype: ElementType) -> WeightComponentSpec {
    WeightComponentSpec {
        id: format!("component.{}", names[0]).try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: names.iter().map(|s| s.to_string()).collect(),
        dimensions: shape.to_vec(),
        encoding: WeightEncoding::Dense {
            element_type: dtype,
        },
        required: true,
    }
}
fn args(total: usize) -> Arguments {
    Arguments {
        semantic_root: "unused".into(),
        tokenizer_root: "unused".into(),
        weights_file: "unused".into(),
        numerical_profile: "qwen3_5.f16".into(),
        host_bytes: total,
        host_alignment: 1,
        iterations: 2,
        mode: Mode::Diagnostic,
        backing: Backing::BoundedStream,
        touch_stride_bytes: 0,
        component_id: None,
        hash_output: true,
        conversion_workers: NonZeroUsize::MIN,
    }
}

#[test]
fn bounded_ranges_cover_composite_and_tail_with_original_hash_and_release_between_passes() {
    let directory = tempfile::tempdir().unwrap();
    let first = [0x3f80u16, 0x8000, 0x4000, 0x4040, 0x4080, 0x40a0]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let second = [0x40c0u16, 0x40e0, 0x4100, 0x4110, 0x4120, 0x4130]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let fallback = 0.25f32.to_le_bytes();
    serialize_to_file(
        [
            (
                "first",
                TensorView::new(Dtype::BF16, vec![2, 3], &first).unwrap(),
            ),
            (
                "second",
                TensorView::new(Dtype::BF16, vec![2, 3], &second).unwrap(),
            ),
            (
                "fallback",
                TensorView::new(Dtype::F32, vec![1], &fallback).unwrap(),
            ),
        ],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let combined = component(&["second", "first"], &[4, 3], ElementType::F16);
    let tail = component(&["first"], &[2, 3], ElementType::F16);
    let full = component(&["fallback"], &[1], ElementType::F16);
    let owner = host(10, 1); // 8-byte slot plus the real 2-byte full fallback.
    let mut output = Vec::new();
    run_with_limit(
        &archive,
        &owner,
        &[&combined, &tail, &full],
        &[24, 12, 2],
        &args(10),
        &mut output,
        8,
    )
    .unwrap();
    let records = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
        .collect::<Vec<_>>();
    let samples = records
        .iter()
        .filter(|r| r["kind"] == "streaming_component")
        .collect::<Vec<_>>();
    assert_eq!(samples.len(), 6);
    let expected = [
        0x4600u16, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x3c00, 0x8000, 0x4000, 0x4200, 0x4400,
        0x4500,
    ]
    .into_iter()
    .flat_map(u16::to_le_bytes)
    .collect::<Vec<_>>();
    assert_eq!(
        samples[0]["sample"]["output_sha256"],
        format!("{:x}", Sha256::digest(&expected))
    );
    assert_eq!(
        samples[0]["sample"]["output_sha256"],
        samples[3]["sample"]["output_sha256"]
    );
    assert_eq!(samples[1]["sample"]["ranges"][1]["offset_bytes"], 8);
    assert_eq!(samples[1]["sample"]["ranges"][1]["length_bytes"], 4);
    assert_eq!(samples[2]["sample"]["route"], "original_full_fallback");
    assert_eq!(
        samples[2]["sample"]["output_sha256"],
        format!("{:x}", Sha256::digest([0x00, 0x34]))
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r["kind"] == "streaming_pass_completed")
            .count(),
        2
    );
    assert_eq!(owner.usage().unwrap().committed_bytes(), 0);
    assert_eq!(owner.usage().unwrap().allocated_bytes(), 0);
    let mut output = Vec::new();
    let too_small = host(9, 1);
    assert!(run_with_limit(
        &archive,
        &too_small,
        &[&combined, &full],
        &[24, 2],
        &args(9),
        &mut output,
        8
    )
    .is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn capacity_counts_padding_and_checksum_consumes_last_bytes() {
    let owner = host(1024, 64);
    let capacity = slot_capacity(&owner, 512, 128).unwrap().get();
    assert_eq!(capacity, 64);
    assert_eq!(owner.backing_bytes(capacity).unwrap(), 127);
    assert!(slot_capacity(&owner, 512, 64).is_err());
    let mut first = 0;
    let mut second = 0;
    consume(&[0, 0, 0, 0, 0, 0, 0, 0, 1], &mut first, None);
    consume(&[0, 0, 0, 0, 0, 0, 0, 0, 2], &mut second, None);
    assert_ne!(first, second);
}
