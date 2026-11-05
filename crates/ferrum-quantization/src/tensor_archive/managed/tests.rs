use super::*;
use crate::SafetensorsArchive;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits, WeightComponentRole};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

mod affine;
mod bf16_simd;
mod diagnostics;

struct Tensor {
    name: &'static str,
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}
fn bf(name: &'static str, rows: usize, columns: usize, values: &[f32]) -> Tensor {
    Tensor {
        name,
        shape: vec![rows, columns],
        dtype: Dtype::BF16,
        bytes: values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
    }
}
fn f32_tensor(name: &'static str, shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name,
        shape: shape.to_vec(),
        dtype: Dtype::F32,
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}
fn archive(tensors: &[Tensor]) -> (tempfile::TempDir, SafetensorsArchive) {
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name,
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive)
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn component(names: &[&str], shape: &[u64], target: ElementType) -> WeightComponentSpec {
    WeightComponentSpec {
        id: "component.fixture.composite".to_owned().try_into().unwrap(),
        role: WeightComponentRole::Values,
        external_names: names.iter().map(|name| (*name).to_owned()).collect(),
        dimensions: shape.to_vec(),
        encoding: WeightEncoding::Dense {
            element_type: target,
        },
        required: true,
    }
}
fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

#[test]
fn four_declared_gdn_projection_parts_use_one_exact_managed_backing_and_retry_after_rejection() {
    // Small QKV/Z/B/A row counts retain the real four-part GDN packing rule.
    // Neutral source names demonstrate that only the typed order/shape matters.
    let tensors = [
        bf("a", 4, 2, &[1., 2., 3., 4., 5., 6., 7., 8.]),
        bf("b", 2, 2, &[9., 10., 11., 12.]),
        bf("c", 1, 2, &[13., 14.]),
        bf("d", 1, 2, &[15., 16.]),
    ];
    let (_directory, original) = archive(&tensors);
    let c = component(&["a", "b", "c", "d"], &[8, 2], ElementType::F16);
    let h = host();
    let physical = h
        .backing_bytes(c.physical_bytes().unwrap() as usize)
        .unwrap();
    let short = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical - 1)])
        .unwrap();
    assert!(ManagedDenseArchiveSource::new(&original, &short.account())
        .component(&c)
        .is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), physical - 1);
    short.close().unwrap();
    let scope = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let view = scope.account();
    let source = ManagedDenseArchiveSource::new(&original, &view);
    let payload = source.component(&c).unwrap();
    assert_eq!(
        payload.bytes(),
        f16_bytes(&(1..=16).map(|v| v as f32).collect::<Vec<_>>())
    );
    assert_eq!(h.usage().unwrap().allocated_bytes(), physical);
    let retained = payload.retained_host_memory().unwrap().clone();
    scope.close().unwrap();
    assert!(
        source.component(&c).is_err(),
        "a closed install view cannot grant another payload"
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), physical);
    drop(retained);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let fresh = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let fresh_view = fresh.account();
    let fresh_source = ManagedDenseArchiveSource::new(&original, &fresh_view);
    let retried = fresh_source.component(&c).unwrap();
    assert_eq!(
        retried.bytes(),
        f16_bytes(&(1..=16).map(|v| v as f32).collect::<Vec<_>>())
    );
    drop(retried);
    fresh.close().unwrap();
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn two_mlp_parts_convert_directly_in_declared_order_and_match_the_existing_archive_contract() {
    let tensors = [
        bf("gate", 2, 2, &[1., -0., 0.5, -2.]),
        f32_tensor(
            "up",
            &[2, 2],
            &[1.00048828125, 1.00146484375, -1.00048828125, -1.00146484375],
        ),
    ];
    let (_directory, original) = archive(&tensors);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let c = component(&["up", "gate"], &[4, 2], ElementType::F16);
    let payload = source.component(&c).unwrap();
    // F16 tie-to-even results, followed by the separately declared gate rows.
    let bits = [
        0x3c00u16, 0x3c02, 0xbc00, 0xbc02, 0x3c00, 0x8000, 0x3800, 0xc000,
    ];
    let expected = bits
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes(), expected);
    assert_eq!(payload.bytes(), original.component(&c).unwrap().bytes());
    assert_eq!(
        payload.source_files(),
        ["model.safetensors", "model.safetensors"]
    );
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        h.backing_bytes(expected.len()).unwrap()
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn exact_leading_axis_stack_keeps_f32_low_bits_and_signed_zero_without_borrowing_a_source() {
    let first = [
        f32::from_bits(0x3f80_0001),
        -0.0,
        f32::from_bits(1),
        -1.000123,
    ];
    let second = [2.000123, f32::from_bits(0x8000_0001), 0.0, 0.500123];
    let tensors = [
        f32_tensor("first", &[2, 2], &first),
        f32_tensor("second", &[2, 2], &second),
    ];
    let (_directory, original) = archive(&tensors);
    let h = host();
    let c = component(&["second", "first"], &[2, 2, 2], ElementType::F32);
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let payload = source.component(&c).unwrap();
    let expected = second
        .into_iter()
        .chain(first)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes(), expected);
    assert_ne!(
        payload.bytes().as_ptr(),
        original.tensor("second").unwrap().bytes().as_ptr()
    );
    assert!(payload.retained_host_memory().is_some());
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        h.backing_bytes(expected.len()).unwrap()
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn incompatible_ranks_axes_extents_or_encoding_reject_before_allocating_any_destination() {
    let tensors = [
        bf("a", 2, 3, &[1.; 6]),
        bf("b", 3, 2, &[2.; 6]),
        bf("c", 1, 3, &[3.; 3]),
        Tensor {
            name: "integer",
            shape: vec![1, 3],
            dtype: Dtype::I8,
            bytes: vec![1; 3],
        },
    ];
    let (_directory, original) = archive(&tensors);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    for invalid in [
        component(&["a", "b"], &[2, 2, 3], ElementType::F16), // same bytes, wrong stack shape
        component(&["a", "b"], &[4, 3], ElementType::F16),    // same bytes, wrong concat axis
        component(&["a", "c"], &[4, 3], ElementType::F16),    // wrong total rows
        component(&["a", "c"], &[3, 3, 1], ElementType::F16), // same elements, wrong rank
        component(&["a", "missing"], &[3, 3], ElementType::F16),
        component(&["a", "integer"], &[3, 3], ElementType::F16),
        component(&["a", "c"], &[3, 3], ElementType::I8),
        component(&[], &[1], ElementType::F16),
        component(&["a", "c"], &[], ElementType::F16),
        component(&["a", "c"], &[u64::MAX, 3], ElementType::F16),
    ] {
        assert!(
            source.component(&invalid).is_err(),
            "accepted invalid component {invalid:?}"
        );
        assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn late_conversion_overflow_refunds_the_whole_backing_and_single_identity_still_borrows() {
    let tensors = [
        f32_tensor("first", &[1, 2], &[1., -0.]),
        f32_tensor("last", &[1, 2], &[2., f32::MAX]),
    ];
    let (_directory, original) = archive(&tensors);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let narrow = component(&["first", "last"], &[2, 2], ElementType::F16);
    assert!(source.component(&narrow).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    let wide = component(&["first", "last"], &[2, 2], ElementType::F32);
    let payload = source.component(&wide).unwrap();
    assert_eq!(
        payload.bytes(),
        [1., -0., 2., f32::MAX]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let single = component(&["first"], &[1, 2], ElementType::F32);
    let borrowed = source.component(&single).unwrap();
    assert_eq!(
        borrowed.bytes().as_ptr(),
        original.tensor("first").unwrap().bytes().as_ptr()
    );
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
