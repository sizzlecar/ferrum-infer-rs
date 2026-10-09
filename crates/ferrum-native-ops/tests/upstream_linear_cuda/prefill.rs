//! Fixed-J32 native geometry, actual-pack oracle, and admitted scratch bounds.
//! Uses the same MarkerV2 arithmetic fixtures as the small-row native gate.
use super::*;

const FORMATS: [UpstreamLinearFormat; 3] = [
    UpstreamLinearFormat::Iq4Xs,
    UpstreamLinearFormat::Q4K,
    UpstreamLinearFormat::Q5K,
];

fn assert_scratch_bound(p: &UpstreamLinearPlanV1) {
    assert_eq!((p.j, p.i), (32, 128));
    assert_eq!(
        p.converted_bytes,
        u64::from(p.request.rows) * u64::from(p.padded_inputs) * 4
    );
    assert_eq!(
        p.packed_bytes,
        u64::from(p.request.rows) * u64::from(p.padded_inputs) * 9 / 8
            + u64::from(p.guard_blocks) * 144
    );
    assert!(p.guard_blocks <= 512);
    if p.fixup != 0 {
        assert_eq!(p.blocks, p.request.sm_count);
        assert_eq!(p.fixup_bytes, u64::from(p.blocks) * 32 * 128 * 4);
    } else {
        assert_eq!(p.fixup_bytes, 0);
    }
    // The actual native lengths are packed exactly as the typed wave layout.
    // This compares the analytic admission bound with returned native geometry,
    // rather than constructing a plan from a copy of the estimator formula.
    let mut extent = 0u64;
    for bytes in [
        p.converted_bytes,
        p.packed_bytes,
        p.output_bytes,
        p.fixup_bytes,
        u64::from(p.request.rows) * 4,
    ] {
        if bytes != 0 {
            extent = extent.checked_add(15).unwrap() & !15;
            extent = extent.checked_add(bytes).unwrap();
        }
    }
    assert!(extent <= p.request.mmq_prefill_marker_scratch_upper_bound().unwrap());
}

#[test]
#[ignore = "requires the new source-built MMQ prefill capability and a CUDA device"]
fn mmq_prefill_native_plans_obey_scratch_bound_and_capability_domains() {
    let device = caps();
    for f in FORMATS {
        for m in [33, 54, 63, 64, 65, 155, 747, 1024, 1025, 2048] {
            for (k, n) in [(256, 17), (768, 129), (5120, 17408), (17408, 5120)] {
                let p = plan(
                    UpstreamLinearAlgorithm::Mmq,
                    f,
                    UpstreamLinearLayout::Columns,
                    m,
                    k,
                    n,
                );
                assert_scratch_bound(&p);
            }
        }
        for m in [1, 4, 8, 9, 16, 17, 24, 25, 32] {
            for n in [17, 128] {
                let p = plan(
                    UpstreamLinearAlgorithm::Mmq,
                    f,
                    UpstreamLinearLayout::Columns,
                    m,
                    768,
                    n,
                );
                let old_j = if n % 128 == 0 {
                    m.div_ceil(8) * 8
                } else if m <= 8 {
                    8
                } else if m <= 16 {
                    16
                } else {
                    32
                };
                assert_eq!(p.j, old_j, "legacy small-row tile selection");
                let mut out = UpstreamLinearPlanV1::default();
                unsafe {
                    assert_eq!(
                        ffi::ferrum_upstream_mmq_prefill_plan_v1(&p.request, &mut out),
                        -1
                    );
                }
            }
        }
        let r = UpstreamLinearRequestV1::new(f, UpstreamLinearLayout::Columns, 33, 256, 17, device)
            .unwrap();
        let mut out = UpstreamLinearPlanV1::default();
        unsafe {
            assert_eq!(ffi::ferrum_upstream_mmq_plan_v1(&r, &mut out), -1);
            assert_eq!(ffi::ferrum_upstream_mmvq_plan_v1(&r, &mut out), -1);
            for bad in [
                UpstreamLinearRequestV1 { rows: 2049, ..r },
                UpstreamLinearRequestV1 {
                    layout: UpstreamLinearLayout::Channels as u32,
                    ..r
                },
                UpstreamLinearRequestV1 {
                    rows: 2048,
                    inputs: 1 << 24,
                    ..r
                },
                UpstreamLinearRequestV1 {
                    rows: 2048,
                    outputs: 1 << 24,
                    ..r
                },
            ] {
                assert_eq!(ffi::ferrum_upstream_mmq_prefill_plan_v1(&bad, &mut out), -1);
            }
            let shared_too_small = UpstreamLinearRequestV1 {
                shared_limit: 1,
                ..r
            };
            assert_eq!(
                ffi::ferrum_upstream_mmq_prefill_plan_v1(&shared_too_small, &mut out),
                -2
            );
        }
    }
}

#[test]
#[ignore = "requires the new source-built MMQ prefill capability and a CUDA device"]
fn marker_v2_mmq_prefill_preserves_pack_oracle_tails_flags_and_repeat() {
    for f in FORMATS {
        for (index, m) in [33, 54, 63, 64, 65, 155, 747, 1024, 1025, 2048]
            .into_iter()
            .enumerate()
        {
            let k = if index % 2 == 0 { 768 } else { 1280 };
            let n = if index % 3 == 0 { 129 } else { 17 };
            let p = check_geometry(
                UpstreamLinearAlgorithm::Mmq,
                f,
                UpstreamLinearLayout::Columns,
                [m, k, n],
                Input::Finite,
                false,
                true,
            );
            assert_scratch_bound(&p);
        }
        for kind in [
            Input::Zero,
            Input::Nonfinite,
            Input::SumOverflow,
            Input::Subnormal,
        ] {
            check_geometry(
                UpstreamLinearAlgorithm::Mmq,
                f,
                UpstreamLinearLayout::Columns,
                [33, 768, 17],
                kind,
                false,
                true,
            );
        }
        check_geometry(
            UpstreamLinearAlgorithm::Mmq,
            f,
            UpstreamLinearLayout::Columns,
            [2048, 256, 17],
            Input::Nonfinite,
            false,
            true,
        );
        check_geometry(
            UpstreamLinearAlgorithm::Mmq,
            f,
            UpstreamLinearLayout::Columns,
            [65, 256, 129],
            Input::Finite,
            true,
            true,
        );

        // Select boundaries from the real device, not a hard-coded SM count.
        // Small N produces stream-K fixup on multi-SM devices. Increasing the
        // number of output tiles eventually fills a wave and removes fixup.
        let sm = caps().multiprocessors;
        let mut fixed = false;
        for m in [33, 65] {
            let p = check_geometry(
                UpstreamLinearAlgorithm::Mmq,
                f,
                UpstreamLinearLayout::Columns,
                [m, 256, 17],
                Input::Finite,
                false,
                true,
            );
            fixed |= p.fixup != 0;
        }
        assert!(
            sm == 1 || fixed,
            "multi-SM stream-K fixup must actually execute"
        );
        let n = (1..=sm)
            .map(|tiles| tiles.checked_mul(128).unwrap())
            .find(|&n| {
                plan(
                    UpstreamLinearAlgorithm::Mmq,
                    f,
                    UpstreamLinearLayout::Columns,
                    33,
                    256,
                    n,
                )
                .fixup
                    == 0
            })
            .expect("an integral full SM wave admits no-fixup MMQ");
        let p = check_geometry(
            UpstreamLinearAlgorithm::Mmq,
            f,
            UpstreamLinearLayout::Columns,
            [33, 256, n],
            Input::Finite,
            false,
            true,
        );
        assert_eq!(p.fixup, 0);
        assert_scratch_bound(&p);
    }
}
