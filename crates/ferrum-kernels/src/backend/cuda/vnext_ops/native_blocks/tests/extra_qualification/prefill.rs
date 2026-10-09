//! New archive qualification only; no product profile selects this route.
use super::*;
use ferrum_native_ops::upstream_extra_linear::UpstreamExtraLinearPrefillRequestV2;

const ROWS: [usize; 5] = [33, 63, 155, 747, 2048];
fn prefill_plan(
    s: &Arc<CudaStream>,
    f: GgufBlockFormat,
    m: usize,
    k: usize,
    n: usize,
) -> PreparedUpstreamLinear {
    let r = request(s.context(), f, m as u32, k as u32, n as u32);
    let device = Device {
        architecture: r.cc,
        multiprocessors: r.sm_count,
        maximum_dynamic_shared_bytes: r.shared_limit,
    };
    let typed = UpstreamExtraLinearPrefillRequestV2::new(
        extra_format(f),
        r.rows,
        r.inputs,
        r.outputs,
        device,
    )
    .unwrap();
    let p = PreparedUpstreamLinear::new_extra_prefill(
        extra_format(f),
        r.rows,
        r.inputs,
        r.outputs,
        device,
    )
    .unwrap();
    typed.validate_plan_identity(p.geometry()).unwrap();
    assert_eq!(
        p.operator(),
        ferrum_native_ops::upstream_extra_linear::UPSTREAM_EXTRA_LINEAR_OPERATOR
    );
    assert_eq!(p.arithmetic(), Arithmetic::MarkerV2);
    let g = p.geometry();
    let mut actual = 0_u64;
    for bytes in [
        g.converted_bytes,
        g.packed_bytes,
        g.output_bytes,
        g.fixup_bytes,
        p.row_poison_bytes(),
    ] {
        actual = actual.checked_add(15).unwrap() / 16 * 16;
        actual = actual.checked_add(bytes).unwrap();
    }
    assert!(actual <= typed.marker_scratch_upper_bound().unwrap());
    println!(
        "{}",
        serde_json::json!({"kind":"extra_prefill_native_plan","format":format!("{f:?}"),"m":m,"k":k,"n":n,"j":g.j,"i":g.i,"blocks":g.blocks,"fixup":g.fixup,"guard_blocks":g.guard_blocks,"actual_scratch":actual,"scratch_bound":typed.marker_scratch_upper_bound().unwrap(),"native_capability":"ferrum_upstream_extra_mmq_prefill_plan_v2"})
    );
    p
}
fn case(s: &Arc<CudaStream>, f: GgufBlockFormat, m: usize, k: usize, n: usize) -> Case {
    let p = prefill_plan(s, f, m, k, n);
    let c = Case::with_plans(s, f, m, k, n, true, 0, vec![p]);
    let r = &c.routes[0];
    let mut ranges = [
        &c.bx,
        &c.bw,
        &c.baseline,
        &r.converted,
        &r.packed,
        &r.raw,
        &r.fixup,
        &r.output,
        &r.rows,
        &r.weight_flag,
    ]
    .into_iter()
    .map(|b| b.span(s))
    .filter(|span| span.bytes != 0)
    .map(|span| (span.address, span.address.checked_add(span.bytes).unwrap()))
    .collect::<Vec<_>>();
    ranges.sort_unstable();
    assert!(
        ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "actual device spans overlap"
    );
    c
}
fn repeat(s: &Arc<CudaStream>, c: &mut Case) {
    let mut old = None;
    for _ in 0..2 {
        c.reset(s);
        c.native(s, 0);
        let bits = c.validate(s, Some(0));
        if let Some(previous) = old {
            assert_eq!(bits, previous, "same route repeated bits");
        }
        old = Some(bits);
    }
}

#[test]
#[ignore = "requires fresh extra prefill v2 archive and exclusive CUDA"]
fn extra_prefill_actual_pack_oracles_tails_fixup_and_marker_rows_on_cuda() {
    let (ctx, s, _) = context();
    for f in FORMATS {
        let mut fixup_seen = [false; 2];
        for m in ROWS {
            let mut c = case(&s, f, m, 256, 17);
            fixup_seen[c.routes[0].p.fixup as usize] = true;
            repeat(&s, &mut c); // all M*N outputs have an independent oracle
        }
        for m in [33, 63] {
            let mut c = case(&s, f, m, 768, 129); // K512 padding + N128 output tail
            fixup_seen[c.routes[0].p.fixup as usize] = true;
            repeat(&s, &mut c);
        }
        // Force a whole number of actual-SM waves, without a machine ID.
        let sm = request(&ctx, f, 33, 256, 17).sm_count;
        let n = sm.checked_mul(128).unwrap() as usize;
        let mut complete_wave = case(&s, f, 33, 256, n);
        fixup_seen[complete_wave.routes[0].p.fixup as usize] = true;
        repeat(&s, &mut complete_wave);
        assert!(fixup_seen[0], "actual no-fixup kernel was not exercised");
        // If a very small device happens to have only exact waves above,
        // choose a partial-wave N using the actual authoritative planner.
        if !fixup_seen[1] {
            for tiles_y in 1..=sm + 1 {
                let n = tiles_y.checked_mul(128).unwrap() as usize;
                let p = prefill_plan(&s, f, 33, 256, n);
                if p.geometry().fixup == 1 {
                    let mut c = Case::with_plans(&s, f, 33, 256, n, true, 0, vec![p]);
                    repeat(&s, &mut c);
                    fixup_seen[1] = true;
                    break;
                }
            }
        }
        // On one/two-SM devices every two-column tile wave can be exact;
        // use odd ceil(M/32) rather than pretending a fixup was executed.
        if !fixup_seen[1] && sm > 1 {
            let mut c = case(&s, f, 65, 256, 17);
            fixup_seen[c.routes[0].p.fixup as usize] = true;
            repeat(&s, &mut c);
        }
        assert!(
            fixup_seen[1] || sm == 1,
            "actual stream-K fixup was not exercised"
        );
        println!(
            "{}",
            serde_json::json!({"kind":"extra_prefill_fixup_coverage","format":format!("{f:?}"),"multiprocessors":sm,"no_fixup_executed":fixup_seen[0],"fixup_executed":fixup_seen[1],"single_sm_fixup_impossible":sm==1})
        );
        for m in [63, 2048] {
            let mut c = case(&s, f, m, 768, 17);
            for row in 0..m {
                for k in 0..c.k {
                    c.x[row * c.xstride + k] = f16::from_bits(0x8000);
                }
            }
            c.upload(&s);
            repeat(&s, &mut c); // q/scale/padding must overwrite poison
            for row in 0..m {
                for k in 0..c.k {
                    c.x[row * c.xstride + k] = f16::from_bits(1 + ((row + k) % 1023) as u16);
                }
            }
            c.upload(&s);
            repeat(&s, &mut c); // finite half subnormals
            for row in [0, 31, 32, m - 1] {
                c.x[row * c.xstride] = f16::NAN;
            }
            c.upload(&s);
            repeat(&s, &mut c); // tile boundary and last row flag
            c.x.fill(f16::from_f32(0.01));
            let d = if f == GgufBlockFormat::Q3K { 108 } else { 0 };
            c.w[d..d + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            c.upload(&s);
            repeat(&s, &mut c); // retained whole-leaf marker
        }
        let mut c = case(&s, f, 33, 768, 17);
        c.reset(&s);
        c.native(&s, 0);
        c.validate(&s, Some(0));
        // Invalid spans cannot partially overwrite output or poison flags.
        let r = &c.routes[0];
        let before = r.output.read(&s);
        let raw = r.raw.span(&s);
        assert!(unsafe {
            r.native.cast(
                DeviceSpan {
                    bytes: raw.bytes - 1,
                    ..raw
                },
                r.output.span(&s),
                c.ystride as u32,
                r.rows.span(&s),
                r.weight_flag.span(&s),
                s.cu_stream().cast(),
            )
        }
        .is_err());
        assert_eq!(r.output.read(&s), before);
        let before = r.packed.read(&s);
        let rows = r.rows.span(&s);
        assert!(unsafe {
            r.native.pack(
                c.bx.span(&s),
                c.xstride as u32,
                r.converted.span(&s),
                r.packed.span(&s),
                DeviceSpan {
                    bytes: rows.bytes - 1,
                    ..rows
                },
                s.cu_stream().cast(),
            )
        }
        .is_err());
        assert_eq!(r.packed.read(&s), before);
        // Direct final-cast overflow is a distinct MarkerV2 boundary.
        let r = &mut c.routes[0];
        let mut raw = vec![0_f32; c.m * c.n];
        raw[0] = f32::INFINITY;
        raw[(c.m - 1) * c.n] = 65536.0;
        r.raw.write(
            &s,
            &raw.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        );
        unsafe {
            r.native
                .cast(
                    r.raw.span(&s),
                    r.output.span(&s),
                    c.ystride as u32,
                    r.rows.span(&s),
                    r.weight_flag.span(&s),
                    s.cu_stream().cast(),
                )
                .unwrap();
        }
        s.synchronize().unwrap();
        let bits = half_read(&r.output.read(&s));
        assert_eq!(bits[0], 0x7e00);
        assert_eq!(bits[(c.m - 1) * c.ystride], 0x7e00);
        r.guards(&s);
    }
    s.synchronize().unwrap();
}

#[test]
#[ignore = "requires fresh extra prefill archive; actual projection shapes with sparse F64 oracle"]
fn extra_prefill_real_projection_shapes_preserve_oracle_and_repeated_bits_on_cuda() {
    let (_, s, _) = context();
    for f in FORMATS {
        let mut shapes = vec![(5120, 17408), (17408, 5120)];
        if f == GgufBlockFormat::Iq4Nl {
            shapes.push((5120, 10240));
        }
        for (k, n) in shapes {
            for m in ROWS {
                let mut c = case(&s, f, m, k, n);
                repeat(&s, &mut c);
            }
        }
    }
    s.synchronize().unwrap();
}
