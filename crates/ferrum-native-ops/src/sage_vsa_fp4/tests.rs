use super::*;

fn device() -> SageVsaFp4Device {
    SageVsaFp4Device {
        ordinal: 0,
        architecture: 1200,
        multiprocessors: 4,
        maximum_dynamic_shared_bytes: 131072,
    }
}
fn request() -> SageVsaFp4RequestV1 {
    SageVsaFp4RequestV1::new(2, 3, 2, 4, 0.125, device()).unwrap()
}
// A synthetic structurally valid return record, not a claim about compiled
// shared memory or hardware support. Native Validate rederives the real plan.
fn plan() -> SageVsaFp4PlanV1 {
    let request = request();
    SageVsaFp4PlanV1 {
        request,
        abi: 1,
        size: size_of::<SageVsaFp4PlanV1>() as u32,
        kernel_id: 1,
        shared_bytes: 4096,
        threads: 384,
        grid_blocks: request.multiprocessors,
        upload_count: 3,
        memset_count: 1,
        bytes: request.buffer_sizes().unwrap(),
    }
}
fn bindings(p: &SageVsaFp4PlanV1) -> SageVsaFp4BindingsV1 {
    let mut address = 0x10000;
    let mut span = |bytes| {
        let result = SageVsaFp4SpanV1 { address, bytes };
        address += bytes + 256;
        address = (address + 255) & !255;
        result
    };
    let s = p.bytes;
    SageVsaFp4BindingsV1 {
        query: span(s.query),
        key: span(s.key),
        value: span(s.value),
        query_scales: span(s.query_scales),
        key_scales: span(s.key_scales),
        value_scales: span(s.value_scales),
        output: span(s.output),
        lse: span(s.lse),
        zero_delta: span(s.zero_delta),
        indices: span(s.indices),
        counts: span(s.counts),
        half_valid: span(s.half_valid),
    }
}

#[test]
fn rectangular_query_and_key_extents_include_all_uploads_and_delta() {
    let p = plan();
    p.validate().unwrap();
    let s = p.bytes;
    assert_eq!(s.query, 2 * 3 * 256 * 64);
    assert_eq!(s.key, 2 * 3 * 512 * 64);
    assert_eq!(s.value, s.key);
    assert_eq!(s.query_scales, 2 * 3 * 256 * 8);
    assert_eq!(s.key_scales, s.value_scales);
    assert_eq!(s.output, 2 * 3 * 256 * 128 * 2);
    assert_eq!(s.zero_delta, 2 * 3 * 512 * 4);
    assert_eq!(s.indices, 2 * 3 * 2 * 4 * 4);
    assert_eq!(s.counts, 2 * 3 * 2 * 4);
    assert_eq!(s.half_valid, 4 * 2 * 4);
    bindings(&p).validate(&p).unwrap();
}

#[test]
fn unsupported_and_nonfinite_raw_requests_cannot_bypass_constructor() {
    let r = request();
    for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(SageVsaFp4RequestV1 {
            softmax_scale_bits: scale.to_bits(),
            ..r
        }
        .validate()
        .is_err());
    }
    for bad in [
        SageVsaFp4RequestV1 { head_dim: 256, ..r },
        SageVsaFp4RequestV1 { tile_rows: 64, ..r },
        SageVsaFp4RequestV1 {
            query_blocks: 5,
            ..r
        },
        SageVsaFp4RequestV1 { reserved: 1, ..r },
        SageVsaFp4RequestV1 { profile: 2, ..r },
        SageVsaFp4RequestV1 {
            architecture: 1210,
            ..r
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn persistent_scheduler_terminal_advance_and_lse_stride_are_bounded() {
    let mut r = request();
    r.heads = 1;
    r.query_blocks = 1;
    r.key_blocks = 1;
    // Synthetic metadata isolates the scheduler boundary from the separate
    // CuTe shape bound; native Plan also verifies the real device's SM count.
    r.multiprocessors = i32::MAX as u32 - 1;
    r.batches = 2;
    r.validate().unwrap();
    r.batches += 1;
    assert_eq!(r.validate(), Err(SageVsaFp4Error::Extent));
    r = request();
    r.batches = 1;
    r.heads = i32::MAX as u32 / 256 + 1;
    assert_eq!(r.validate(), Err(SageVsaFp4Error::Extent));
    r = request();
    r.key_blocks = i32::MAX as u32 / 128 + 1;
    assert_eq!(r.validate(), Err(SageVsaFp4Error::Extent));
    r = request();
    r.batches = 1;
    r.heads = 1;
    r.query_blocks = 1;
    r.key_blocks = i32::MAX as u32 / (128 * 128);
    r.validate().unwrap();
    r.key_blocks += 1;
    assert_eq!(r.validate(), Err(SageVsaFp4Error::Extent));
}

#[test]
fn every_device_write_is_exclusive_and_span_end_cannot_wrap() {
    let p = plan();
    let b = bindings(&p);
    let mut bad = b;
    bad.indices.address = b.query.address;
    assert_eq!(bad.validate(&p), Err(SageVsaFp4Error::Span));
    bad = b;
    bad.counts.address = b.indices.address;
    assert_eq!(bad.validate(&p), Err(SageVsaFp4Error::Span));
    bad = b;
    bad.key.bytes -= 1;
    assert_eq!(bad.validate(&p), Err(SageVsaFp4Error::Span));
    bad = b;
    bad.output.address = u64::MAX - 15;
    assert_eq!(bad.validate(&p), Err(SageVsaFp4Error::Span));
    bad = b;
    bad.value.address += 1;
    assert_eq!(bad.validate(&p), Err(SageVsaFp4Error::Span));
    let mut forged = p;
    forged.bytes.zero_delta *= 128;
    assert_eq!(b.validate(&forged), Err(SageVsaFp4Error::Plan));
    forged = p;
    forged.upload_count = 0;
    assert_eq!(forged.validate(), Err(SageVsaFp4Error::Plan));
}

#[test]
fn checked_lists_keep_each_batch_and_head_rectangular_row_independent() {
    let r = request();
    let mut idx = [3, 2, 1, 0].repeat(12);
    let mut counts = vec![4; 12];
    // A distinct sparse row in the last batch/head. Inactive slots have no
    // meaning; the kernel must use the checked count for traversal.
    idx[44..48].copy_from_slice(&[2, 0, i32::MIN, i32::MAX]);
    counts[11] = 2;
    let half = [64, 64, 64, 7, 17, 0, 64, 0];
    let checked = CheckedCpuBlockLists::new(r, &idx, &counts, &half).unwrap();
    assert_eq!(checked.indices(), idx);
    assert_eq!(checked.counts()[11], 2);
    assert_eq!(checked.half_valid(), half);
    assert_eq!(
        checked.host_addresses().indices.bytes,
        r.buffer_sizes().unwrap().indices
    );
    // A full [B,H,KV,KV] source cannot masquerade as compact [B,H,Q,KV].
    let full = [3, 2, 1, 0].repeat(24);
    assert!(CheckedCpuBlockLists::new(r, &full, &counts, &half).is_err());
}

#[test]
fn malformed_dynamic_lists_reject_before_any_submission() {
    let r = SageVsaFp4RequestV1::new(1, 1, 1, 4, 0.125, device()).unwrap();
    let half = [64, 64, 64, 7, 17, 0, 64, 0];
    for row in [
        [3, 2, 2, 0],
        [0, 1, 2, 3],
        [4, 2, 1, 0],
        [3, 2, 1, -1],
        [3, 2, 1, 1],
    ] {
        assert!(CheckedCpuBlockLists::new(r, &row, &[4], &half).is_err());
    }
    for count in [0, -1, 5] {
        assert!(CheckedCpuBlockLists::new(r, &[3, 2, 1, 0], &[count], &half).is_err());
    }
    for pair in [[0, 0], [65, 0], [64, 65], [63, 1], [64, -1]] {
        let mut invalid = half;
        invalid[..2].copy_from_slice(&pair);
        assert!(CheckedCpuBlockLists::new(r, &[3, 2, 1, 0], &[4], &invalid).is_err());
    }
}
