use super::*;

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            8192,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8192,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn spec(shape: &[u64], element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        element_type,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn source(
    account: &HostMemoryAccount,
    tensor: ResolvedTensorSpec,
    bytes: &[u8],
) -> ManagedHostTensor {
    let mut backing = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    backing.as_mut().copy_from_slice(bytes);
    ManagedHostTensor::new(
        ProgramValueId::new("value.fixture.reference.work_latent").unwrap(),
        tensor,
        backing.freeze(),
    )
    .unwrap()
}

#[test]
fn reference_copy_preserves_f32_bits_and_original_owner_across_destination_capacity_retry() {
    let h = host();
    // Non-BF16 low bits, signed zero and finite subnormals survive this
    // ownership/shape transition without a numerical conversion.
    let words = [
        0x3f80_0001u32,
        0x8000_0000,
        0x0000_0001,
        0xbf80_0001,
        0x3eaa_aaab,
        0x0080_0001,
        0x3f00_0123,
        0x8000_0001,
        0x3f7f_ffff,
        0x4000_0001,
        0x0000_0000,
        0xc000_0001,
    ];
    let bytes = words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let physical = h.backing_bytes(bytes.len()).unwrap();
    assert!(
        physical > bytes.len(),
        "the boundary must include real alignment padding"
    );
    let work = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, physical)])
        .unwrap();
    let original = source(
        &work.account(),
        spec(&[1, 2, 2, 3], ElementType::F32),
        &bytes,
    );
    let target = spec(&[1, 2, 1, 2, 3], ElementType::F32);
    let short = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, physical - 1)])
        .unwrap();
    let before = h.usage().unwrap();
    let error = copy_reference_latent(&short.account(), &original, &target).unwrap_err();
    assert!(
        error.chain().any(|e| matches!(
            e.downcast_ref::<HostMemoryError>(),
            Some(HostMemoryError::ScopeCapacity {
                class: HostMemoryClass::InputPreprocess,
                ..
            })
        )),
        "expected the actual destination scope failure: {error:#}"
    );
    assert_eq!(
        h.usage().unwrap().allocated_bytes(),
        before.allocated_bytes()
    );
    assert_eq!(
        h.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    assert_eq!(original.bytes().bytes(), bytes);
    drop(short);
    // Closing the old work authority does not invalidate its actual source
    // allocation, and does not supply a new destination's authorization.
    work.close().unwrap();
    let persistent = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, physical)])
        .unwrap();
    let copied = copy_reference_latent(&persistent.account(), &original, &target).unwrap();
    assert_eq!(copied.tensor(), &target);
    assert_eq!(copied.bytes().class(), HostMemoryClass::InputPreprocess);
    assert!(h.owns(copied.bytes()));
    assert!(!copied.bytes().same_region(original.bytes()));
    assert_eq!(copied.bytes().bytes(), bytes);
    assert_eq!(original.bytes().bytes(), bytes);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 2 * physical);
    persistent.close().unwrap();
    drop(original);
    assert_eq!(h.usage().unwrap().committed_bytes(), physical);
    assert_eq!(copied.bytes().bytes(), bytes);
    drop(copied);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reference_copy_rejects_foreign_precision_or_shape_without_destination_allocation() {
    let h = host();
    let foreign = host();
    let target = spec(&[1, 2, 1, 2, 3], ElementType::F32);
    let foreign_source = source(&foreign, spec(&[1, 2, 2, 3], ElementType::F32), &[0; 48]);
    let bf16 = source(&h, spec(&[1, 2, 2, 3], ElementType::Bf16), &[0; 24]);
    let transposed_extent = source(&h, spec(&[1, 2, 3, 2], ElementType::F32), &[0; 48]);
    let persistent = h
        .reserve_scope(&[(
            HostMemoryClass::InputPreprocess,
            h.backing_bytes(48).unwrap(),
        )])
        .unwrap();
    let before = h.usage().unwrap();
    let foreign_before = foreign.usage().unwrap().committed_bytes();
    for invalid in [&foreign_source, &bf16, &transposed_extent] {
        assert!(copy_reference_latent(&persistent.account(), invalid, &target).is_err());
        assert_eq!(
            h.usage().unwrap().allocated_bytes(),
            before.allocated_bytes()
        );
        assert_eq!(
            h.usage().unwrap().committed_bytes(),
            before.committed_bytes()
        );
        assert_eq!(foreign.usage().unwrap().committed_bytes(), foreign_before);
    }
    persistent.close().unwrap();
    drop((bf16, transposed_extent, foreign_source));
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}
