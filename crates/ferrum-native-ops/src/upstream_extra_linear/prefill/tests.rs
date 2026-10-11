use super::*;
fn device() -> UpstreamLinearDevice {
    UpstreamLinearDevice {
        architecture: 800,
        multiprocessors: 20,
        maximum_dynamic_shared_bytes: 98304,
    }
}
fn request(
    f: UpstreamExtraLinearFormat,
    m: u32,
    k: u32,
    n: u32,
) -> UpstreamExtraLinearPrefillRequestV2 {
    UpstreamExtraLinearPrefillRequestV2::new(f, m, k, n, device()).unwrap()
}
#[test]
fn extra_prefill_request_keeps_small_and_mmvq_domains_closed() {
    for f in [
        UpstreamExtraLinearFormat::Q3K,
        UpstreamExtraLinearFormat::Iq3S,
        UpstreamExtraLinearFormat::Iq4Nl,
    ] {
        for m in [33, 63, 155, 747, 2048] {
            let p = request(f, m, 768, 17);
            assert_eq!(p.as_raw().layout, UpstreamLinearLayout::Columns as u32);
            assert!(p.as_raw().validate().is_err());
            assert!(
                p.0.validate().is_err(),
                "small extra constructor was widened"
            );
            let mut channels = p;
            channels.0 .0.layout = 1;
            assert!(channels.validate().is_err());
            let mut old_format = p;
            old_format.0 .0.format = 12;
            assert!(old_format.validate().is_err());
        }
        for m in [0, 1, 32, 2049, u32::MAX] {
            assert!(UpstreamExtraLinearPrefillRequestV2::new(f, m, 768, 17, device()).is_err());
        }
        for (m, k, n) in [
            (33, 257, 17),
            (2048, 1 << 24, 17),
            (2048, 256, u32::MAX),
            (2048, 1 << 20, 1 << 20),
        ] {
            assert!(UpstreamExtraLinearPrefillRequestV2::new(f, m, k, n, device()).is_err());
        }
        assert!(UpstreamExtraLinearRequestV1::new(
            f,
            UpstreamLinearLayout::Columns,
            32,
            768,
            17,
            device()
        )
        .is_ok());
    }
}
#[test]
fn extra_prefill_plan_validates_d4_extents_and_nonmonotone_fixup() {
    for f in [
        UpstreamExtraLinearFormat::Q3K,
        UpstreamExtraLinearFormat::Iq3S,
        UpstreamExtraLinearFormat::Iq4Nl,
    ] {
        for m in [33, 63, 155, 747, 2048] {
            // N128 tiles produce a full SM wave; N17 produces partial waves.
            for n in [17, 128 * device().multiprocessors] {
                let r = request(f, m, 768, n);
                let tiles = m.div_ceil(32) * n.div_ceil(128);
                let blocks = if tiles % device().multiprocessors == 0 {
                    tiles
                } else {
                    device().multiprocessors
                };
                let fixup = u32::from(tiles % blocks != 0);
                let mut p = UpstreamLinearPlanV1 {
                    request: *r.as_raw(),
                    abi: 1,
                    size: size_of::<UpstreamLinearPlanV1>() as u32,
                    algorithm: 1,
                    pack_abi: 1,
                    padded_inputs: 1024,
                    padded_outputs: n,
                    guard_blocks: 32,
                    j: 32,
                    i: 128,
                    nthreads: 256,
                    shared_bytes: 32768,
                    blocks,
                    tiles_y: n.div_ceil(128),
                    fixup,
                    weight_bytes: u64::from(n)
                        * u64::from(768 / f.block_elements())
                        * u64::from(f.block_bytes()),
                    converted_bytes: u64::from(m) * 1024 * 4,
                    packed_bytes: u64::from(m) * 1024 * 9 / 8 + 32 * 144,
                    output_bytes: u64::from(m) * u64::from(n) * 4,
                    fixup_bytes: if fixup != 0 {
                        u64::from(blocks) * 16384
                    } else {
                        0
                    },
                    ..Default::default()
                };
                r.validate_plan_identity(&p).unwrap();
                let mut used = 0_u64;
                for b in [
                    p.converted_bytes,
                    p.packed_bytes,
                    p.output_bytes,
                    p.fixup_bytes,
                    u64::from(m) * 4,
                ] {
                    used = used.div_ceil(16) * 16 + b;
                }
                assert!(used <= r.marker_scratch_upper_bound().unwrap());
                p.pack_abi = 2;
                assert!(r.validate_plan_identity(&p).is_err());
                p.pack_abi = 1;
                p.fixup_bytes += 4;
                assert!(r.validate_plan_identity(&p).is_err());
                p.fixup_bytes -= 4;
                p.j = 16;
                assert!(r.validate_plan_identity(&p).is_err());
                p.j = 32;
                p.weight_bytes += 1;
                assert!(r.validate_plan_identity(&p).is_err());
            }
        }
    }
}

#[test]
fn extra_prefill_plan_checks_cooperative_tail_extent_without_guard_rounding_assumptions() {
    for format in [
        UpstreamExtraLinearFormat::Q3K,
        UpstreamExtraLinearFormat::Iq3S,
        UpstreamExtraLinearFormat::Iq4Nl,
    ] {
        // Whole K512 rows expose the last cooperative load. K768 has a
        // padded K1024 body: its logical K endpoint needs no extra block.
        for (m, k, guard) in [
            (33_u32, 512_u32, 35_u32),
            (34, 512, 34),
            (35, 512, 33),
            (64, 512, 4),
            (34, 5120, 34),
            (34, 768, 0),
        ] {
            let n = 17;
            let r = request(format, m, k, n);
            let kp = k.div_ceil(512) * 512;
            let blocks = device().multiprocessors;
            let body = u64::from(m) * u64::from(kp) * 9 / 8;
            let p = UpstreamLinearPlanV1 {
                request: *r.as_raw(),
                abi: 1,
                size: size_of::<UpstreamLinearPlanV1>() as u32,
                algorithm: 1,
                pack_abi: 1,
                padded_inputs: kp,
                padded_outputs: n,
                guard_blocks: guard,
                j: 32,
                i: 128,
                nthreads: 256,
                shared_bytes: 32768,
                blocks,
                tiles_y: 1,
                fixup: 1,
                weight_bytes: u64::from(n)
                    * u64::from(k / format.block_elements())
                    * u64::from(format.block_bytes()),
                converted_bytes: u64::from(m) * u64::from(kp) * 4,
                packed_bytes: body + u64::from(guard) * 144,
                output_bytes: u64::from(m) * u64::from(n) * 4,
                fixup_bytes: u64::from(blocks) * 16384,
                ..Default::default()
            };
            r.validate_plan_identity(&p).unwrap();
            if guard > 0 {
                let short = UpstreamLinearPlanV1 {
                    guard_blocks: guard - 1,
                    packed_bytes: p.packed_bytes - 144,
                    ..p
                };
                assert_eq!(r.validate_plan_identity(&short), Err(UpstreamLinearError::AbiMismatch), "{format:?} M{m} K{k}: matching fields cannot authorize an undersized cooperative tail");
            }
            let inconsistent = UpstreamLinearPlanV1 {
                packed_bytes: p.packed_bytes - 1,
                ..p
            };
            assert!(r.validate_plan_identity(&inconsistent).is_err());
            let oversized = UpstreamLinearPlanV1 {
                guard_blocks: 513,
                packed_bytes: body + 513 * 144,
                ..p
            };
            assert!(r.validate_plan_identity(&oversized).is_err());
        }
    }
}
