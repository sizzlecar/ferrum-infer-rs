use super::*;

fn affine_component(
    names: &[&str],
    shape: &[u64],
    target: ElementType,
    scale: (i64, u64),
    bias: (i64, u64),
) -> WeightComponentSpec {
    let mut c = component(names, shape, target);
    c.encoding = WeightEncoding::DenseAffine {
        element_type: target,
        scale: CanonicalRational::new(scale.0, scale.1).unwrap(),
        bias: CanonicalRational::new(bias.0, bias.1).unwrap(),
    };
    c
}

#[test]
fn bf16_affine_to_f16_uses_one_charged_destination_and_independent_rne_ties() {
    let values = [0., 0.5, -1., -2., 1. / 2048., 3. / 2048.];
    let (_directory, original) = archive(&[bf("offset", 1, 6, &values)]);
    let h = host();
    let c = affine_component(&["offset"], &[1, 6], ElementType::F16, (1, 1), (1, 1));
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let payload = source.component(&c).unwrap();
    // 1 + 2^-11 ties down to even 0x3c00; 1 + 3*2^-11 ties
    // up to even 0x3c02. The other values are exact hand calculations.
    let expected = [0x3c00u16, 0x3e00, 0x0000, 0xbc00, 0x3c00, 0x3c02]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes(), expected);
    assert_eq!(payload.bytes(), original.component(&c).unwrap().bytes());
    assert_ne!(
        payload.bytes().as_ptr(),
        original.tensor("offset").unwrap().bytes().as_ptr()
    );
    assert_eq!(payload.external_names(), ["offset"]);
    assert_eq!(payload.source_files(), ["model.safetensors"]);
    let physical = h.backing_bytes(expected.len()).unwrap();
    assert_eq!(h.usage().unwrap().allocated_bytes(), physical);
    let retained = payload.retained_host_memory().unwrap().clone();
    assert_eq!(
        retained.managed_memory_class(),
        Some(HostMemoryClass::WeightBacking)
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), physical);
    assert_eq!(retained.bytes(), expected);
    drop(retained);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn same_dtype_affine_keeps_f64_coefficients_and_never_borrows_even_identity_affine() {
    let tensors = [
        f32_tensor("cancellation", &[1], &[16_777_216.]),
        f32_tensor("identity", &[2], &[-0., f32::from_bits(0x3f80_0001)]),
    ];
    let (_directory, original) = archive(&tensors);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let cancellation = affine_component(
        &["cancellation"],
        &[1],
        ElementType::F32,
        (16_777_217, 16_777_216),
        (-16_777_216, 1),
    );
    let payload = source.component(&cancellation).unwrap();
    // (2^24 * (1 + 2^-24)) - 2^24 = 1 in F64. A rounded
    // F32 coefficient would become 1 and incorrectly produce zero.
    assert_eq!(payload.bytes(), 1.0_f32.to_le_bytes());
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        h.backing_bytes(4).unwrap()
    );
    drop(payload);
    let identity = affine_component(&["identity"], &[2], ElementType::F32, (1, 1), (0, 1));
    let payload = source.component(&identity).unwrap();
    // The explicit multiply/add maps -0 * 1 + +0 to +0; it is not
    // interchangeable with the ordinary same-dtype Dense borrowing path.
    let expected = [0.0, f32::from_bits(0x3f80_0001)]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes(), expected);
    assert_ne!(
        payload.bytes().as_ptr(),
        original.tensor("identity").unwrap().bytes().as_ptr()
    );
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        h.backing_bytes(8).unwrap()
    );
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn affine_capacity_uses_physical_alignment_and_a_foreign_scope_cannot_supply_credit() {
    let (_directory, original) = archive(&[f32_tensor("a", &[3], &[1., 2., 3.])]);
    let c = affine_component(&["a"], &[3], ElementType::F32, (1, 1), (1, 1));
    let h = host();
    let physical = h.backing_bytes(12).unwrap();
    assert!(physical > 12);
    let short = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical - 1)])
        .unwrap();
    let short_view = short.account();
    assert!(ManagedDenseArchiveSource::new(&original, &short_view)
        .component(&c)
        .is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), physical - 1);
    short.close().unwrap();
    let closed = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let closed_view = closed.account();
    closed.close().unwrap();
    let other = host();
    let live = other
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let live_view = live.account();
    assert!(!closed_view.same_parent(&live_view));
    assert!(ManagedDenseArchiveSource::new(&original, &closed_view)
        .component(&c)
        .is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(other.usage().unwrap().allocated_bytes(), 0);
    let source = ManagedDenseArchiveSource::new(&original, &live_view);
    let payload = source.component(&c).unwrap();
    assert_eq!(
        payload.bytes(),
        [2.0_f32, 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(other.usage().unwrap().allocated_bytes(), physical);
    live.close().unwrap();
    drop(payload);
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn late_affine_and_destination_overflow_refund_the_entire_destination() {
    let (_directory, original) = archive(&[
        f32_tensor("f32_overflow", &[2], &[1., f32::MAX]),
        f32_tensor("f16_overflow", &[2], &[1., 65_504.]),
        f32_tensor("nonfinite", &[2], &[1., f32::NAN]),
    ]);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    for (name, target) in [
        ("f32_overflow", ElementType::F32),
        ("f16_overflow", ElementType::F16),
        ("nonfinite", ElementType::F32),
    ] {
        let c = affine_component(&[name], &[2], target, (2, 1), (1, 1));
        assert!(source.component(&c).is_err(), "accepted {name}");
        assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    let finite = affine_component(&["f16_overflow"], &[2], ElementType::F16, (1, 1), (0, 1));
    let retry = source.component(&finite).unwrap();
    assert_eq!(
        retry.bytes(),
        [0x3c00u16, 0x7bff]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    drop(retry);
    // Ordinary Dense retains its existing non-finite/same-dtype semantics.
    let plain = source
        .component(&component(&["nonfinite"], &[2], ElementType::F32))
        .unwrap();
    assert!(f32::from_le_bytes(plain.bytes()[4..8].try_into().unwrap()).is_nan());
    assert_eq!(
        plain.bytes().as_ptr(),
        original.tensor("nonfinite").unwrap().bytes().as_ptr()
    );
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn affine_rejects_wrong_shapes_integer_sources_and_composites_before_allocation() {
    let (_directory, original) = archive(&[
        bf("a", 1, 2, &[1., 2.]),
        bf("b", 1, 2, &[3., 4.]),
        Tensor {
            name: "integer",
            shape: vec![2],
            dtype: Dtype::I8,
            bytes: vec![1, 2],
        },
    ]);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    for c in [
        affine_component(&["a"], &[2, 1], ElementType::F16, (1, 1), (1, 1)),
        affine_component(&["a"], &[u64::MAX, 2], ElementType::F16, (1, 1), (1, 1)),
        affine_component(&["integer"], &[2], ElementType::F16, (1, 1), (1, 1)),
        affine_component(&["a"], &[1, 2], ElementType::I8, (1, 1), (1, 1)),
        affine_component(&["a", "b"], &[2, 2], ElementType::F16, (1, 1), (1, 1)),
    ] {
        assert!(source.component(&c).is_err(), "accepted {c:?}");
        assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn affine_conversion_checks_element_extents_before_writing_any_value() {
    let affine = Some((
        CanonicalRational::new(1, 1).unwrap(),
        CanonicalRational::new(1, 1).unwrap(),
    ));
    for input in [&[0u8, 0, 0][..], &[0u8, 0, 0, 0, 0, 0][..]] {
        let mut output = [0xa5u8; 4];
        assert!(convert_floating(
            ElementType::F32,
            ElementType::F32,
            input,
            &mut output,
            affine
        )
        .is_err());
        assert_eq!(output, [0xa5; 4]);
    }
    let mut wrong_output = [0xa5u8; 3];
    assert!(convert_floating(
        ElementType::F32,
        ElementType::F16,
        &[0; 4],
        &mut wrong_output,
        affine
    )
    .is_err());
    assert_eq!(wrong_output, [0xa5; 3]);
}

#[test]
fn native_blocks_stay_borrowed_and_never_become_an_affine_floating_source() {
    use ferrum_runtime::{BlockQuantizationSpec, ManagedHostBytes, RetainedHostMemoryRegion};
    use std::sync::Arc;

    // A genuine Q8_0 physical block remains an immutable archive view. This
    // fixture tests source ownership/encoding, not a device quantized kernel.
    struct NativeArchive {
        owner: Arc<ManagedHostBytes>,
        encoding: WeightEncoding,
    }
    impl NativeArchive {
        fn retained(&self) -> RetainedHostMemoryRegion {
            RetainedHostMemoryRegion::new(Arc::clone(&self.owner), 0, 34).unwrap()
        }
    }
    impl WeightComponentSource for NativeArchive {
        fn component<'source>(
            &'source self,
            c: &WeightComponentSpec,
        ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
            if c.encoding != self.encoding || c.external_names != ["native"] {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: "native encoding/source changed".into(),
                });
            }
            WeightComponentPayload::new(
                c,
                "native",
                "fixture.gguf",
                c.dimensions.clone(),
                ElementType::U8,
                self.owner.bytes(),
            )?
            .with_retained_host_memory(self.retained())
        }
    }
    impl crate::TensorArchive for NativeArchive {
        fn tensor(&self, name: &str) -> ferrum_types::Result<crate::ArchiveTensor<'_>> {
            if name != "native" {
                return Err(ferrum_types::FerrumError::model("missing native tensor"));
            }
            crate::ArchiveTensor::new(
                "native",
                "fixture.gguf",
                &[1, 32],
                self.encoding.clone(),
                self.retained(),
            )
        }
        fn tensor_count(&self) -> usize {
            1
        }
        fn tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
            Box::new(std::iter::once("native"))
        }
    }
    let original_host = host();
    let mut bytes = original_host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 34)
        .unwrap();
    bytes.as_mut()[..2].copy_from_slice(&f16::from_f32(1.).to_le_bytes());
    for (index, value) in bytes.as_mut()[2..].iter_mut().enumerate() {
        *value = (index as i8 - 16) as u8;
    }
    let original = NativeArchive {
        owner: Arc::new(bytes.freeze()),
        encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: "quantization.gguf.q8-0".to_owned().try_into().unwrap(),
            logical_values_per_block: 32,
            bytes_per_block: 34,
        }),
    };
    let h = host();
    let source = ManagedDenseArchiveSource::new(&original, &h);
    let mut c = component(&["native"], &[1, 1], ElementType::U8);
    c.encoding = original.encoding.clone();
    let borrowed = source.component(&c).unwrap();
    assert_eq!(borrowed.bytes(), original.owner.bytes());
    assert_eq!(borrowed.bytes().as_ptr(), original.owner.bytes().as_ptr());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let affine = affine_component(&["native"], &[1, 32], ElementType::F16, (1, 1), (1, 1));
    assert!(source.component(&affine).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let retained = borrowed.retained_host_memory().unwrap().clone();
    drop(borrowed);
    drop(source);
    drop(original);
    assert_eq!(
        original_host.usage().unwrap().allocated_bytes(),
        original_host.backing_bytes(34).unwrap()
    );
    drop(retained);
    assert_eq!(original_host.usage().unwrap().committed_bytes(), 0);
}
