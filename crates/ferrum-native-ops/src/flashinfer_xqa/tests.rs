use super::*;
fn request(batch: u32) -> XqaRequestV1 {
    XqaRequestV1 {
        abi: 1,
        size: size_of::<XqaRequestV1>() as u32,
        batch,
        max_pages_per_sequence: 128,
        binding_stride_bytes: 1040,
        sliding_window: 0,
        device_ordinal: 0,
        architecture: 1200,
        multiprocessors: 170,
        reserved: 0,
        maximum_dynamic_shared_bytes: 101376,
    }
}
#[test]
fn workspace_covers_vendor_partial_reduction_and_affine_budget() {
    for batch in [1, 3, 8, 16, 17, 128, 65535] {
        let r = request(batch);
        let p = XqaPlanV1::checked(r, 65536).unwrap();
        assert_eq!(p.semaphore_bytes, u64::from(batch) * 32);
        assert!(p.scratch_bytes <= u64::from(r.multiprocessors) * 1280);
        if p.subsequences == 1 {
            assert_eq!(p.scratch_bytes, 0);
        } else {
            assert_eq!(
                p.scratch_bytes,
                u64::from(batch) * 8 * u64::from(p.subsequences) * 1280
            );
        }
        assert_eq!(p.fingerprint_material().len(), 72);
        p.validate().unwrap();
    }
}
#[test]
fn reject_unsupported_and_overflowing_boundary_geometry() {
    for mutate in [
        |r: &mut XqaRequestV1| r.architecture = 890,
        |r: &mut XqaRequestV1| r.batch = 0,
        |r: &mut XqaRequestV1| r.batch = 65536,
        |r: &mut XqaRequestV1| r.max_pages_per_sequence = u32::MAX,
        |r: &mut XqaRequestV1| r.binding_stride_bytes = 1032,
        |r: &mut XqaRequestV1| r.binding_stride_bytes = 1041,
        |r: &mut XqaRequestV1| r.sliding_window = 64,
    ] {
        let mut r = request(3);
        mutate(&mut r);
        assert!(r.validate().is_err());
    }
    let mut p = XqaPlanV1::checked(request(3), 65536).unwrap();
    p.scratch_bytes += 1024;
    assert!(p.validate().is_err());
}
#[test]
fn spans_reject_wrong_extents_alignment_and_write_aliases() {
    let p = XqaPlanV1::checked(request(3), 65536).unwrap();
    let bindings = XqaBindingsV1 {
        query: XqaSpan {
            address: 0x100000,
            bytes: p.request.query_bytes(),
        },
        sinks_f32: XqaSpan {
            address: 0x200000,
            bytes: 256,
        },
        packed_metadata: XqaSpan {
            address: 0x300000,
            bytes: p.request.metadata_bytes().unwrap(),
        },
        output: XqaSpan {
            address: 0x400000,
            bytes: p.request.query_bytes(),
        },
        semaphores: XqaSpan {
            address: 0x500000,
            bytes: p.semaphore_bytes,
        },
        scratch: XqaSpan {
            address: 0x600000,
            bytes: p.scratch_bytes,
        },
    };
    bindings.validate(&p).unwrap();
    let mut invalid = bindings;
    invalid.output.address = bindings.query.address;
    assert!(invalid.validate(&p).is_err());
    invalid = bindings;
    invalid.scratch.address += 256;
    assert!(invalid.validate(&p).is_err());
    invalid = bindings;
    invalid.packed_metadata.bytes -= 8;
    assert!(invalid.validate(&p).is_err());
    invalid = bindings;
    invalid.scratch.address = u64::MAX - 1023;
    assert!(invalid.validate(&p).is_err());
}
