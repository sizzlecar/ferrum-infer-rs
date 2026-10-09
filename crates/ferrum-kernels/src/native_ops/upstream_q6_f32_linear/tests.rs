use super::*;

#[test]
fn q6_workspace_keeps_rounded_loader_tail_and_row_flags_disjoint() {
    for rows in 1..=32 {
        for outputs in [7, 128, 129] {
            let mut plan = UpstreamQ6F32PlanV1::default();
            plan.request.rows = rows;
            plan.converted_bytes = u64::from(rows) * 512 * 4;
            // Include a non-8-multiple rounded read guard, not only logical codes.
            plan.packed_bytes = u64::from(rows) * 512 * 9 / 8 + 19 * 144;
            plan.output_bytes = u64::from(rows) * outputs * 4;
            plan.fixup_bytes = if outputs == 128 { 0 } else { 128 * 32 * 4 };
            let w = Q6F32Workspace::new(&plan).unwrap();
            let ranges = [
                (w.converted, plan.converted_bytes),
                (w.packed, plan.packed_bytes),
                (w.raw, plan.output_bytes),
                (w.fixup, plan.fixup_bytes),
                (w.row_flags, u64::from(rows) * 4),
            ];
            for (offset, bytes) in ranges {
                assert_eq!(offset % 16, 0);
                assert!(offset + bytes <= w.bytes);
            }
            for pair in ranges.windows(2) {
                assert!(pair[0].0 + pair[0].1 <= pair[1].0);
            }
            assert_eq!(w.bytes, w.row_flags + u64::from(rows) * 4);
        }
    }
    let huge = UpstreamQ6F32PlanV1 {
        converted_bytes: u64::MAX,
        packed_bytes: 1,
        ..Default::default()
    };
    assert_eq!(Q6F32Workspace::new(&huge), Err(Error::Extent));
}

#[test]
fn q6_launch_rejects_cross_stage_alias_before_native_access() {
    let mut raw = UpstreamQ6F32PlanV1::default();
    raw.request.rows = 1;
    raw.request.inputs = 256;
    raw.request.outputs = 7;
    raw.weight_bytes = 1470;
    raw.converted_bytes = 2048;
    raw.packed_bytes = 576 + 19 * 144;
    raw.output_bytes = 28;
    let prepared = PreparedQ6F32Linear {
        workspace: Q6F32Workspace::new(&raw).unwrap(),
        raw,
    };
    let input = DeviceSpan {
        address: 0x1000,
        bytes: 1024,
    };
    let weights = DeviceSpan {
        address: 0x2000,
        bytes: 1470,
    };
    let output = DeviceSpan {
        address: 0x4000,
        bytes: 28,
    };
    let work = DeviceSpan {
        address: 0x8000,
        bytes: prepared.workspace_bytes(),
    };
    let flag = DeviceSpan {
        address: 0x2000,
        bytes: 4,
    }; // aliases weight storage
    let result = unsafe {
        prepared.launch(
            input,
            256,
            weights,
            output,
            7,
            work,
            flag,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, Err(Error::Span));
    let short = DeviceSpan {
        bytes: work.bytes - 1,
        ..work
    };
    let flag = DeviceSpan {
        address: 0x6000,
        bytes: 4,
    };
    assert_eq!(
        unsafe {
            prepared.launch(
                input,
                256,
                weights,
                output,
                7,
                short,
                flag,
                std::ptr::null_mut(),
            )
        },
        Err(Error::Span)
    );
}
