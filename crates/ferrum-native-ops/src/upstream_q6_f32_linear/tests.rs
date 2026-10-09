use super::*;

fn request(m: u32, k: u32, n: u32) -> UpstreamQ6F32RequestV1 {
    UpstreamQ6F32RequestV1::new(
        m,
        k,
        n,
        UpstreamLinearDevice {
            architecture: 1200,
            multiprocessors: 170,
            maximum_dynamic_shared_bytes: 101376,
        },
    )
    .unwrap()
}

#[test]
fn q6_f32_request_is_independent_and_rejects_wrong_boundary() {
    for m in 1..=32 {
        request(m, 5120, 248320).validate().unwrap();
    }
    let good = request(1, 256, 7);
    for (m, k, n) in [
        (0, 256, 7),
        (33, 256, 7),
        (1, 0, 7),
        (1, 32, 7),
        (1, 256, 0),
        (32, 5120, u32::MAX),
    ] {
        let mut bad = good;
        bad.rows = m;
        bad.inputs = k;
        bad.outputs = n;
        assert!(bad.validate().is_err());
    }
    for (format, layout) in [(12, 0), (13, 0), (23, 0), (14, 1)] {
        let mut bad = good;
        bad.format = format;
        bad.layout = layout;
        assert!(bad.validate().is_err());
    }
    let mut bad = good;
    bad.reserved = 1;
    assert!(bad.validate().is_err());
    bad = good;
    bad.abi += 1;
    assert!(bad.validate().is_err());
    bad = good;
    bad.size -= 4;
    assert!(bad.validate().is_err());
}

#[test]
fn q6_f32_plan_rejects_original_tail_extent_and_f16_output() {
    // M17/Ntail chooses J32: the last cooperative load spans 5120 bytes,
    // while the old 16 guard blocks only provide (17+16)*144=4752 bytes.
    let r = request(17, 5120, 49);
    let body = 17 * 5120 * 9 / 8;
    let p = UpstreamQ6F32PlanV1 {
        request: r,
        abi: 1,
        size: size_of::<UpstreamQ6F32PlanV1>() as u32,
        algorithm: 1,
        pack_abi: 1,
        padded_inputs: 5120,
        padded_outputs: 49,
        guard_blocks: 19,
        j: 32,
        i: 128,
        nthreads: 256,
        shared_bytes: 32768,
        blocks: 1,
        tiles_y: 1,
        weight_bytes: 49 * 20 * 210,
        converted_bytes: 17 * 5120 * 4,
        packed_bytes: body + 19 * 144,
        output_bytes: 17 * 49 * 4,
        ..Default::default()
    };
    p.validate_identity(&r).unwrap();
    let mut bad = p;
    bad.guard_blocks = 16;
    bad.packed_bytes = body + 16 * 144;
    assert_eq!(
        bad.validate_identity(&r),
        Err(UpstreamLinearError::AbiMismatch)
    );
    bad = p;
    bad.output_bytes /= 2;
    assert!(bad.validate_identity(&r).is_err());
    bad = p;
    bad.weight_bytes -= 1;
    assert!(bad.validate_identity(&r).is_err());
    bad = p;
    bad.pack_abi = 2;
    assert!(bad.validate_identity(&r).is_err());
    bad = p;
    bad.request.rows = 16;
    assert!(bad.validate_identity(&r).is_err());
    bad = p;
    bad.fixup = 1;
    assert!(bad.validate_identity(&r).is_err());
    bad = p;
    bad.packed_bytes = u64::MAX;
    assert!(bad.validate_identity(&r).is_err());
}
