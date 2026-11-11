use super::*;
use std::num::NonZeroUsize;

fn capacity(bytes: usize) -> NonZeroUsize {
    NonZeroUsize::new(bytes).unwrap()
}

#[test]
fn native_bf16_stream_preserves_bits_and_composite_order_with_bounded_backing() {
    // Include signed zero, BF16-only finite range, infinities and NaN payloads.
    let bits = [0x8000u16, 0x7f7f, 0xff7f, 0x7f80, 0x7fc1, 0x0001];
    let bytes = bits
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let (_directory, archive) = archive(&[
        Tensor {
            name: "first",
            shape: vec![1, 3],
            dtype: Dtype::BF16,
            bytes: bytes[..6].to_vec(),
        },
        Tensor {
            name: "second",
            shape: vec![1, 3],
            dtype: Dtype::BF16,
            bytes: bytes[6..].to_vec(),
        },
    ]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let spec = component(&["second", "first"], &[2, 3], ElementType::Bf16);
    assert!(source.prepare_bf16_f16_stream(&spec).unwrap().is_none());
    let prepared = source.prepare_dense_stream(&spec).unwrap().unwrap();
    let slot = source.bounded_streaming_slot(capacity(4)).unwrap();
    let mut output = Vec::new();
    for offset in (0..12).step_by(4) {
        output.extend_from_slice(
            prepared
                .materialize_range(&slot, offset, 4)
                .unwrap()
                .bytes(),
        );
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            slot.backing_bytes()
        );
    }
    assert_eq!(output, [&bytes[6..], &bytes[..6]].concat());
    drop(slot);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn metadata_survives_source_wrapper_and_ranges_reuse_one_original_owner() {
    let (_directory, archive) = archive(&[bf("a", 1, 9, &[1., -0., 2., 3., 4., 5., 6., 7., 8.])]);
    let spec = component(&["a"], &[1, 9], ElementType::F16);
    let host = host();
    let prepared = {
        let source = ManagedDenseArchiveSource::new(&archive, &host);
        source.prepare_bf16_f16_stream(&spec).unwrap().unwrap()
    };
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(prepared.component(), &spec);
    assert_eq!(prepared.source_files(), ["model.safetensors"]);
    assert_eq!(prepared.full_bytes(), 18);
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let expected = source.component(&spec).unwrap().bytes().to_vec();
    let slot = source.bounded_streaming_slot(capacity(8)).unwrap();
    assert_eq!(slot.logical_bytes(), 8);
    assert_eq!(slot.backing_bytes(), host.backing_bytes(8).unwrap());
    let mut collected = Vec::new();
    let mut address = None;
    for (offset, length) in [(0, 8), (8, 8), (16, 2)] {
        let payload = prepared.materialize_range(&slot, offset, length).unwrap();
        assert_eq!(payload.len(), length);
        assert_eq!(payload.backing_bytes(), slot.backing_bytes());
        assert_eq!(
            payload.bytes().as_ptr(),
            *address.get_or_insert(payload.bytes().as_ptr())
        );
        collected.extend_from_slice(payload.bytes());
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            slot.backing_bytes()
        );
    }
    assert_eq!(collected, expected);
    drop(slot);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn composite_ranges_cross_source_boundaries_in_declared_stack_or_concat_order() {
    let (_directory, archive) = archive(&[
        bf("first", 2, 3, &[1., -0., 2., 3., 4., 5.]),
        bf("second", 2, 3, &[6., 7., 8., 9., 10., 11.]),
    ]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let slot = source.bounded_streaming_slot(capacity(10)).unwrap();
    for shape in [&[4, 3][..], &[2, 2, 3][..]] {
        let spec = component(&["second", "first"], shape, ElementType::F16);
        let expected = source.component(&spec).unwrap().bytes().to_vec();
        let prepared = source.prepare_bf16_f16_stream(&spec).unwrap().unwrap();
        let mut output = Vec::new();
        for offset in (0..prepared.full_bytes()).step_by(10) {
            let count = 10.min(prepared.full_bytes() - offset);
            output.extend_from_slice(
                prepared
                    .materialize_range(&slot, offset, count)
                    .unwrap()
                    .bytes(),
            );
        }
        assert_eq!(output, expected);
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            slot.backing_bytes()
        );
    }
}

#[test]
fn every_bf16_bit_pattern_preserves_half_or_finite_overflow_contract() {
    let bits = (0..=u16::MAX)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let (_directory, archive) = archive(&[Tensor {
        name: "all",
        shape: vec![65536],
        dtype: Dtype::BF16,
        bytes: bits,
    }]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let spec = component(&["all"], &[65536], ElementType::F16);
    let prepared = source.prepare_bf16_f16_stream(&spec).unwrap().unwrap();
    let slot = source.bounded_streaming_slot(capacity(2)).unwrap();
    for bits in 0..=u16::MAX {
        let value = bf16::from_bits(bits).to_f32();
        let expected = f16::from_f32(value);
        let actual = prepared.materialize_range(&slot, usize::from(bits) * 2, 2);
        if value.is_finite() && !expected.is_finite() {
            assert!(actual.is_err(), "finite overflow {bits:04x}");
        } else {
            assert_eq!(
                actual.unwrap().bytes(),
                expected.to_le_bytes(),
                "BF16 {bits:04x}"
            );
        }
    }
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        slot.backing_bytes()
    );
}

#[test]
fn malformed_metadata_is_rejected_and_other_recipes_stay_on_the_full_path() {
    let (_directory, archive) = archive(&[
        bf("a", 1, 2, &[1., 2.]),
        f32_tensor("other", &[1, 2], &[3., 4.]),
    ]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    for spec in [
        component(&[], &[1, 2], ElementType::F16),
        component(&["missing"], &[1, 2], ElementType::F16),
        component(&["a"], &[2, 1], ElementType::F16),
        component(&["a", "a"], &[1, 4], ElementType::F16),
        component(&["a"], &[0, 2], ElementType::F16),
        component(&["a"], &[u64::MAX, 2], ElementType::F16),
    ] {
        assert!(source.prepare_bf16_f16_stream(&spec).is_err());
    }
    let mut affine = component(&["a"], &[1, 2], ElementType::F16);
    affine.encoding = WeightEncoding::DenseAffine {
        element_type: ElementType::F16,
        scale: CanonicalRational::new(3, 2).unwrap(),
        bias: CanonicalRational::new(0, 1).unwrap(),
    };
    for spec in [
        affine,
        component(&["a"], &[1, 2], ElementType::Bf16),
        component(&["other"], &[1, 2], ElementType::F16),
        component(&["a", "other"], &[2, 2], ElementType::F16),
    ] {
        assert!(source.prepare_bf16_f16_stream(&spec).unwrap().is_none());
        assert!(source.component(&spec).is_ok());
    }
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn ranges_and_capacity_fail_without_second_allocation_or_mutating_published_bytes() {
    let (_directory, archive) = archive(&[bf("a", 1, 8, &[0.5; 8])]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let prepared = source
        .prepare_bf16_f16_stream(&component(&["a"], &[1, 8], ElementType::F16))
        .unwrap()
        .unwrap();
    assert!(source.bounded_streaming_slot(capacity(3)).is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    let slot = source.bounded_streaming_slot(capacity(8)).unwrap();
    for (offset, len) in [(0, 0), (1, 2), (0, 3), (0, 10), (16, 2), (usize::MAX, 2)] {
        assert!(prepared.materialize_range(&slot, offset, len).is_err());
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            slot.backing_bytes()
        );
    }
    let payload = prepared.materialize_range(&slot, 0, 8).unwrap();
    let alias = payload.slice(2..4).unwrap();
    drop(payload);
    assert!(prepared.materialize_range(&slot, 8, 8).is_err());
    assert_eq!(alias.bytes(), f16_bytes(&[0.5]));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        slot.backing_bytes()
    );
    drop(alias);
    assert!(prepared.materialize_range(&slot, 8, 8).is_ok());
}

#[test]
fn exact_scoped_physical_capacity_and_retained_owner_survive_slot_drop() {
    let (_directory, archive) = archive(&[bf("a", 1, 8, &[0.5; 8])]);
    let host = host();
    let physical = host.backing_bytes(8).unwrap();
    for capacity_bytes in [physical - 1, physical] {
        let scope = host
            .reserve_scope(&[(HostMemoryClass::WeightBacking, capacity_bytes)])
            .unwrap();
        let view = scope.account();
        let source = ManagedDenseArchiveSource::new(&archive, &view);
        let slot = source.bounded_streaming_slot(capacity(8));
        assert_eq!(slot.is_ok(), capacity_bytes == physical);
        if let Ok(slot) = slot {
            let prepared = source
                .prepare_bf16_f16_stream(&component(&["a"], &[1, 8], ElementType::F16))
                .unwrap()
                .unwrap();
            let retained = prepared.materialize_range(&slot, 0, 2).unwrap();
            assert!(view.owns(&retained));
            scope.close().unwrap();
            drop(slot);
            assert_eq!(host.usage().unwrap().committed_bytes(), physical);
            assert_eq!(retained.bytes(), f16_bytes(&[0.5]));
            drop(retained);
        } else {
            scope.close().unwrap();
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn late_overflow_returns_no_range_and_reuses_only_after_every_alias_is_gone() {
    let (_directory, archive) = archive(&[bf("a", 1, 8, &[1., 2., 3., 4., 5., 6., 7., 65536.])]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let prepared = source
        .prepare_bf16_f16_stream(&component(&["a"], &[1, 8], ElementType::F16))
        .unwrap()
        .unwrap();
    let slot = source.bounded_streaming_slot(capacity(8)).unwrap();
    let first = prepared.materialize_range(&slot, 0, 8).unwrap();
    let address = first.bytes().as_ptr();
    assert!(prepared.materialize_range(&slot, 8, 8).is_err());
    assert_eq!(first.bytes(), f16_bytes(&[1., 2., 3., 4.]));
    drop(first);
    assert!(prepared.materialize_range(&slot, 8, 8).is_err());
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        slot.backing_bytes()
    );
    let restored = prepared.materialize_range(&slot, 0, 8).unwrap();
    assert_eq!(restored.bytes().as_ptr(), address);
    assert_eq!(restored.bytes(), f16_bytes(&[1., 2., 3., 4.]));
    // Core aborts the failed installation; this local conversion test does not
    // authorize retrying an indeterminate native submission.
}

fn parallel_host(bytes: usize) -> HostMemoryAccount {
    // The backing rounds payload size before adding allocation alignment slack.
    let limit = bytes.next_multiple_of(64) + 63;
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            limit,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: limit,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}

fn valid_bf16_bytes(bytes: usize) -> Vec<u8> {
    (0..bytes / 2)
        .flat_map(|index| {
            let mut bits = index as u16;
            let value = bf16::from_bits(bits).to_f32();
            if value.is_finite() && !f16::from_f32(value).is_finite() {
                bits = 0;
            }
            bits.to_le_bytes()
        })
        .collect()
}

#[test]
fn parallel_ranges_match_serial_bits_across_large_parts_tail_and_retained_alias() {
    let part_bytes = 4 * Bf16F16ConversionPolicy::MINIMUM_BYTES_PER_WORKER + 10;
    let small_bytes = 2 * Bf16F16ConversionPolicy::MINIMUM_BYTES_PER_WORKER + 6;
    let (_directory, archive) = archive(&[
        Tensor {
            name: "large",
            shape: vec![part_bytes / 2],
            dtype: Dtype::BF16,
            bytes: valid_bf16_bytes(part_bytes),
        },
        Tensor {
            name: "prefix",
            shape: vec![small_bytes / 2],
            dtype: Dtype::BF16,
            bytes: valid_bf16_bytes(small_bytes),
        },
    ]);
    let total = part_bytes + small_bytes;
    let owner = parallel_host(total);
    let source = ManagedDenseArchiveSource::new(&archive, &owner);
    let slot = source.bounded_streaming_slot(capacity(total)).unwrap();
    let policy = Bf16F16ConversionPolicy::new(capacity(4)).unwrap();
    assert_eq!(
        Bf16F16ConversionPolicy::default().maximum_workers().get(),
        1
    );
    assert!(Bf16F16ConversionPolicy::new(capacity(17)).is_err());

    let single = source
        .prepare_bf16_f16_stream(&component(
            &["large"],
            &[(part_bytes / 2) as u64],
            ElementType::F16,
        ))
        .unwrap()
        .unwrap();
    let serial = single
        .materialize_range(&slot, 0, part_bytes)
        .unwrap()
        .bytes()
        .to_vec();
    let parallel = single
        .materialize_range_with_policy(&slot, 0, part_bytes, policy)
        .unwrap();
    assert_eq!(parallel.bytes(), serial);
    let alias = parallel.slice(2..8).unwrap();
    drop(parallel);
    assert!(single
        .materialize_range_with_policy(&slot, 0, part_bytes, policy)
        .is_err());
    assert_eq!(alias.bytes(), &serial[2..8]);
    drop(alias);

    let combined = source
        .prepare_bf16_f16_stream(&component(
            &["prefix", "large"],
            &[(total / 2) as u64],
            ElementType::F16,
        ))
        .unwrap()
        .unwrap();
    // Cross the component boundary with an unaligned-to-SIMD, whole-element
    // offset and a tail. Each large continuous part uses the original kernel.
    let serial = combined
        .materialize_range(&slot, 2, total - 8)
        .unwrap()
        .bytes()
        .to_vec();
    let parallel = combined
        .materialize_range_with_policy(&slot, 2, total - 8, policy)
        .unwrap();
    assert_eq!(parallel.bytes(), serial);
    assert_eq!(
        owner.usage().unwrap().committed_bytes(),
        slot.backing_bytes()
    );
    drop(parallel);
    drop(slot);
    assert_eq!(owner.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn parallel_overflow_publishes_nothing_and_original_slot_can_be_rewritten() {
    let bytes = 4 * Bf16F16ConversionPolicy::MINIMUM_BYTES_PER_WORKER + 14;
    let mut input = valid_bf16_bytes(bytes);
    input[bytes - 2..].copy_from_slice(&bf16::from_f32(65536.).to_le_bytes());
    let (_directory, archive) = archive(&[Tensor {
        name: "overflow",
        shape: vec![bytes / 2],
        dtype: Dtype::BF16,
        bytes: input,
    }]);
    let owner = parallel_host(bytes);
    let source = ManagedDenseArchiveSource::new(&archive, &owner);
    let prepared = source
        .prepare_bf16_f16_stream(&component(
            &["overflow"],
            &[(bytes / 2) as u64],
            ElementType::F16,
        ))
        .unwrap()
        .unwrap();
    let slot = source.bounded_streaming_slot(capacity(bytes)).unwrap();
    let policy = Bf16F16ConversionPolicy::new(capacity(4)).unwrap();
    assert!(prepared
        .materialize_range_with_policy(&slot, 0, bytes, policy)
        .is_err());
    assert_eq!(
        owner.usage().unwrap().committed_bytes(),
        slot.backing_bytes()
    );
    let expected = prepared
        .materialize_range(&slot, 0, bytes - 2)
        .unwrap()
        .bytes()
        .to_vec();
    let recovered = prepared
        .materialize_range_with_policy(&slot, 0, bytes - 2, policy)
        .unwrap();
    assert_eq!(recovered.bytes(), expected);
    drop(recovered);
    drop(slot);
    assert_eq!(owner.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn failed_launch_and_panics_join_started_workers_before_slot_reuse() {
    use super::super::streaming::convert_part_with;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let bytes = 2 * Bf16F16ConversionPolicy::MINIMUM_BYTES_PER_WORKER;
    let input = valid_bf16_bytes(bytes);
    let owner = parallel_host(bytes);
    let backing = ReusableBacking::new(&owner, capacity(bytes)).unwrap();
    let policy = Bf16F16ConversionPolicy::new(capacity(2)).unwrap();
    #[derive(Clone, Copy)]
    enum Failure {
        Spawn,
        WorkerPanic,
        LaunchPanic,
    }
    for failure in [Failure::Spawn, Failure::WorkerPanic, Failure::LaunchPanic] {
        let completed = AtomicUsize::new(0);
        let result = (|| {
            let mut destination = BackingLease::allocate(&owner, bytes, Some(&backing))
                .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
            convert_part_with(
                &input,
                destination.as_mut(),
                policy,
                |worker, input, output| {
                    if matches!(failure, Failure::WorkerPanic) && worker == 0 {
                        panic!("injected converter panic");
                    }
                    super::super::bf16_to_f16::convert(input, output)?;
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                |worker| {
                    if worker == 1 {
                        match failure {
                            Failure::Spawn => {
                                return Err(std::io::Error::other("injected launch failure"))
                            }
                            Failure::LaunchPanic => panic!("injected launch panic"),
                            Failure::WorkerPanic => {}
                        }
                    }
                    Ok(())
                },
            )?;
            destination.freeze()
        })();
        assert!(result.is_err());
        // In all three cases one successful worker must have finished before
        // an error returns and a new mutable lease of the same slot is allowed.
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        assert_eq!(
            owner.usage().unwrap().committed_bytes(),
            backing.physical_bytes()
        );
        let mut destination = BackingLease::allocate(&owner, bytes, Some(&backing)).unwrap();
        convert_part_with(
            &input,
            destination.as_mut(),
            policy,
            |_, input, output| super::super::bf16_to_f16::convert(input, output),
            |_| Ok(()),
        )
        .unwrap();
        let recovered = destination.freeze().unwrap();
        assert_eq!(recovered.len(), bytes);
        drop(recovered);
    }
    drop(backing);
    assert_eq!(owner.usage().unwrap().committed_bytes(), 0);
}
