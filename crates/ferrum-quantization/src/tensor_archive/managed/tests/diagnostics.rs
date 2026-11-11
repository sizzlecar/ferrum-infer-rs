use super::*;
use std::num::NonZeroUsize;

#[test]
fn unattributed_time_saturates_before_subtracting_public_phase_totals() {
    let mut report = DenseArchiveMaterializationReport {
        total_wall_ns: u64::MAX,
        ..Default::default()
    };
    report.metadata_validation.wall_ns = u64::MAX - 5;
    report.source_touch.wall_ns = 1;
    report.reserve_allocate_zeroed.wall_ns = 1;
    report.conversion_and_direct_packing.wall_ns = 1;
    report.payload_assembly.wall_ns = 1;
    assert_eq!(report.unattributed_wall_ns(), 1);
    report.metadata_validation.wall_ns = u64::MAX;
    assert_eq!(report.unattributed_wall_ns(), 0);
    report.source_touch.wall_ns = u64::MAX;
    report.reserve_allocate_zeroed.wall_ns = u64::MAX;
    report.conversion_and_direct_packing.wall_ns = u64::MAX;
    report.payload_assembly.wall_ns = u64::MAX;
    assert_eq!(report.unattributed_wall_ns(), 0);
    report.total_wall_ns = 17;
    assert_eq!(report.unattributed_wall_ns(), 0);
}

#[test]
fn counted_composite_preserves_order_one_backing_and_real_host_refunds() {
    let tensors = [
        bf("first", 1, 9, &[1., 2., 3., 4., 5., 6., 7., 8., -0.]),
        bf("second", 1, 9, &[0.00001, 2., 3., 4., 5., 6., 7., 8., 9.]),
    ];
    let (_directory, archive) = archive(&tensors);
    let spec = component(&["second", "first"], &[2, 9], ElementType::F16);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&archive, &h);
    let expected = source.component(&spec).unwrap().bytes().to_vec();
    for touch in [
        DenseArchiveSourceTouch::AsMapped,
        DenseArchiveSourceTouch::Stride(NonZeroUsize::new(7).unwrap()),
    ] {
        let mut report = DenseArchiveMaterializationReport::default();
        let payload = source
            .component_with_diagnostics(&spec, touch, &mut report)
            .unwrap();
        assert_eq!(payload.bytes(), expected);
        assert!(report.completed && report.composite && !report.borrowed && !report.affine);
        assert_eq!(report.source_parts, 2);
        assert_eq!((report.source_bytes, report.destination_bytes), (36, 36));
        assert_eq!(
            h.usage().unwrap().allocated_bytes(),
            h.backing_bytes(36).unwrap()
        );
        if cfg!(all(
            target_endian = "little",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )) {
            assert_eq!(report.bf16_to_f16.simd_blocks_attempted, 2);
            assert_eq!(report.bf16_to_f16.simd_blocks_converted, 1);
            assert_eq!(report.bf16_to_f16.scalar_fallback_blocks, 1);
            assert_eq!(report.bf16_to_f16.scalar_tail_elements, 2);
        }
        assert_eq!(
            report.source_touch_loads,
            if matches!(touch, DenseArchiveSourceTouch::AsMapped) {
                0
            } else {
                8
            }
        );
        assert!(report.total_wall_ns >= report.conversion_and_direct_packing.wall_ns);
        assert!(report.total_wall_ns >= report.source_touch.wall_ns);
        drop(payload);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    let bytes = h.backing_bytes(36).unwrap();
    for capacity in [bytes - 1, bytes] {
        let scope = h
            .reserve_scope(&[(HostMemoryClass::WeightBacking, capacity)])
            .unwrap();
        let account = scope.account();
        let scoped = ManagedDenseArchiveSource::new(&archive, &account);
        let mut report = DenseArchiveMaterializationReport::default();
        let result = scoped.component_with_diagnostics(
            &spec,
            DenseArchiveSourceTouch::AsMapped,
            &mut report,
        );
        assert_eq!(result.is_ok(), capacity == bytes);
        assert_eq!(report.completed, capacity == bytes);
        drop(result);
        scope.close().unwrap();
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn diagnostic_affine_and_borrowed_paths_do_not_claim_bf16_fast_blocks() {
    let (_directory, archive) = archive(&[bf("value", 1, 8, &[0.125; 8])]);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&archive, &h);
    let mut affine = component(&["value"], &[1, 8], ElementType::F16);
    affine.encoding = WeightEncoding::DenseAffine {
        element_type: ElementType::F16,
        scale: CanonicalRational::new(1, 1).unwrap(),
        bias: CanonicalRational::new(1, 1).unwrap(),
    };
    let expected = source.component(&affine).unwrap().bytes().to_vec();
    let mut report = DenseArchiveMaterializationReport::default();
    let payload = source
        .component_with_diagnostics(&affine, DenseArchiveSourceTouch::AsMapped, &mut report)
        .unwrap();
    assert_eq!(payload.bytes(), expected);
    assert!(report.affine && !report.borrowed);
    assert_eq!(report.bf16_to_f16, Bf16ToF16BlockCounts::default());
    drop(payload);
    let borrowed = component(&["value"], &[1, 8], ElementType::Bf16);
    let payload = source
        .component_with_diagnostics(&borrowed, DenseArchiveSourceTouch::AsMapped, &mut report)
        .unwrap();
    assert!(report.borrowed && !report.affine && report.completed);
    assert_eq!(report.reserve_allocate_zeroed.wall_ns, 0);
    assert_eq!(report.conversion_and_direct_packing.wall_ns, 0);
    assert_eq!(report.bf16_to_f16, Bf16ToF16BlockCounts::default());
    assert_eq!(payload.bytes(), archive.tensor("value").unwrap().bytes());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn counted_late_overflow_keeps_original_error_and_refunds_output() {
    let mut values = vec![1.0; 16];
    values[15] = 65536.0;
    let (_directory, archive) = archive(&[bf("value", 2, 8, &values)]);
    let spec = component(&["value"], &[2, 8], ElementType::F16);
    let h = host();
    let source = ManagedDenseArchiveSource::new(&archive, &h);
    let original = source.component(&spec).err().unwrap().to_string();
    let mut report = DenseArchiveMaterializationReport::default();
    let error = source
        .component_with_diagnostics(&spec, DenseArchiveSourceTouch::AsMapped, &mut report)
        .err()
        .unwrap();
    assert_eq!(error.to_string(), original);
    assert!(!report.completed);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
