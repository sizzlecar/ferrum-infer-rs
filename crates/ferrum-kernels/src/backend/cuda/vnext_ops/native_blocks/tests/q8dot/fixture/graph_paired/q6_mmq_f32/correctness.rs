use super::*;
fn check_pack(c: &Case, s: &Arc<CudaStream>) {
    let p = c.packed.read(s);
    let flags = c.row_flags.read(s);
    assert!(flags.iter().all(|v| *v == 0));
    for r in 0..c.m {
        for g in 0..c.p.padded_inputs as usize / 32 {
            let base = ((g / 4) * c.m + r) * 144;
            let start = base + (g % 4) * 4;
            let d = f32::from_le_bytes(p[start..start + 4].try_into().unwrap());
            let q = &p[base + 16 + g % 4 * 32..base + 16 + (g % 4 + 1) * 32];
            assert!(d.is_finite() && d >= 0.0 && q.iter().all(|v| *v as i8 != i8::MIN));
            if g * 32 >= c.k {
                assert_eq!(d.to_bits(), 0);
                assert!(q.iter().all(|v| *v == 0));
            } else {
                let x = &c.x[r * c.xs + g * 32..r * c.xs + (g + 1) * 32];
                let amax = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
                if amax == 0.0 {
                    assert_eq!(d.to_bits(), 0);
                    assert!(q.iter().all(|v| *v == 0));
                } else {
                    assert!((d - amax / 127.0).abs() <= 8.0 * f32::EPSILON * (amax / 127.0));
                    for (i, &v) in x.iter().enumerate() {
                        assert!(
                            (f64::from(v) - oracle::activation(&p, c.m, r, g * 32 + i)).abs()
                                <= f64::from(d) * 0.501
                                    + f64::from(amax) * 8.0 * f64::from(f32::EPSILON)
                        );
                    }
                }
            }
        }
    }
    assert!(p[c.m * c.p.padded_inputs as usize * 9 / 8..]
        .iter()
        .all(|v| *v == 0));
}

fn check_packed_read_extent(c: &Case) {
    // Independent expansion of the pinned loader's two K128 loads per K256
    // iteration, including the unpredicated final nthreads-wide copy batch.
    let m = c.m as u64;
    let j = u64::from(c.p.j);
    let threads = u64::from(c.p.nthreads);
    let last_tile = m.div_ceil(j) - 1;
    let copy_bytes = (j * 36).div_ceil(threads) * threads * 4;
    let last_read_end = ((c.k as u64 / 128 - 1) * m + last_tile * j) * 144 + copy_bytes;
    assert!(
        c.p.packed_bytes >= last_read_end,
        "packed allocation must cover all tile-loader reads"
    );
    assert_eq!(
        c.p.packed_bytes,
        m * u64::from(c.p.padded_inputs) * 9 / 8 + u64::from(c.p.guard_blocks) * 144
    );
}
#[test]
#[ignore = "requires the locked production Q6 F32 artifact and actual CUDA"]
fn q6_mmq_f32_actual_pack_matches_f64_and_marker_boundaries_on_cuda() {
    let (ctx, s, kernels) = context();
    for m in 1..=32 {
        // K512 has no padding credit; N7 and N128 exercise both template
        // branches, while K768/N129 exercises K padding and the output tail.
        for (k, n) in [(512, 7), (512, 128), (768, 129)] {
            let mut c = Case::new(&s, m, k, n, true);
            check_packed_read_extent(&c);
            c.x[..32].fill(0.0);
            c.upload(&s);
            c.scan(&s);
            c.inclusive(&s);
            c.strict(&s, &kernels);
            s.synchronize().unwrap();
            check_pack(&c, &s);
            c.guards(&s);
            let evidence = c.oracle(&s, false);
            let before = c.output.read(&s);
            let graph = Captured::new(&s, || c.inclusive(&s));
            graph.launch();
            graph.launch();
            s.synchronize().unwrap();
            assert_eq!(
                before,
                c.output.read(&s),
                "same plan repeated/captured output"
            );
            c.guards(&s);
            println!(
                "{}",
                serde_json::json!({"kind":"q6_mmq_f32_correctness","m":m,"k":k,"n":n,"j":c.p.j,"fixup":c.p.fixup,"guard_blocks":c.p.guard_blocks,"packed_bytes":c.p.packed_bytes,"oracle":evidence})
            );
            if n == 7 {
                // Exercise the last physical row of every admitted width.
                // A row marker must not poison healthy rows or output padding.
                let affected = m - 1;
                c.x[affected * c.xs..affected * c.xs + k].fill(f32::NAN);
                c.upload(&s);
                c.inclusive(&s);
                s.synchronize().unwrap();
                let output = c.output.read(&s);
                let packed = c.packed.read(&s);
                let flags = c.row_flags.read(&s);
                for r in 0..m {
                    assert_eq!(
                        u32::from_le_bytes(flags[r * 4..r * 4 + 4].try_into().unwrap()),
                        u32::from(r == affected)
                    );
                    if r == affected {
                        assert!(floats(&output[r * c.ys * 4..(r * c.ys + n) * 4])
                            .iter()
                            .all(|v| v.to_bits() == 0x7fc00000));
                        for g in 0..k / 32 {
                            let block = (g / 4 * m + r) * 144;
                            let d = block + g % 4 * 4;
                            assert_eq!(
                                u32::from_le_bytes(packed[d..d + 4].try_into().unwrap()),
                                0x7fc00000
                            );
                            assert!(
                                packed[block + 16 + g % 4 * 32..block + 16 + (g % 4 + 1) * 32]
                                    .iter()
                                    .all(|v| *v == 0)
                            );
                        }
                    } else {
                        assert_eq!(
                            &output[r * c.ys * 4..(r * c.ys + n) * 4],
                            &before[r * c.ys * 4..(r * c.ys + n) * 4]
                        );
                    }
                }
                assert!(packed[m * c.p.padded_inputs as usize * 9 / 8..]
                    .iter()
                    .all(|v| *v == 0));
                c.guards(&s);
                println!(
                    "{}",
                    serde_json::json!({"kind":"q6_f32_last_row_marker_validated","m":m,"affected_row":affected})
                );
            }
        }
    }
    // Device-derived whole-SM tiles exercise the no-fixup branch without
    // hardcoding a GPU model. Keep expensive F64 checks sparse at larger N.
    for m in [1, 9, 17, 32] {
        let n = request(&ctx, m, 256, 128).sm_count as usize * 128;
        let c = Case::new(&s, m as usize, 256, n, true);
        check_packed_read_extent(&c);
        assert_eq!(c.p.fixup, 0);
        c.inclusive(&s);
        c.strict(&s, &kernels);
        s.synchronize().unwrap();
        check_pack(&c, &s);
        let evidence = c.oracle(&s, true);
        let before = c.output.read(&s);
        assert!(floats(&before)
            .chunks(c.ys)
            .all(|r| r[..c.n].iter().all(|v| v.is_finite())));
        c.inclusive(&s);
        s.synchronize().unwrap();
        assert_eq!(before, c.output.read(&s));
        c.guards(&s);
        println!(
            "{}",
            serde_json::json!({"kind":"q6_f32_whole_sm_no_fixup","m":m,"n":n,"j":c.p.j,"fixup":c.p.fixup,"oracle":evidence})
        );
    }
    // Exercise the production adapter's combined workspace layout as well as
    // the independently guarded buffers used by the numerical oracle above.
    let combined = Case::new(&s, 9, 5120, 49, true);
    combined.inclusive(&s);
    s.synchronize().unwrap();
    let expected = combined.output.read(&s);
    let workspace = Buffer::new(&s, combined.native.workspace_bytes() as usize);
    unsafe {
        combined.native.launch(
            combined.input.span(&s),
            combined.xs as u32,
            combined.weights.span(&s),
            combined.output.span(&s),
            combined.ys as u32,
            workspace.span(&s),
            combined.weight_flag.span(&s),
            s.cu_stream().cast(),
        )
    }
    .unwrap();
    s.synchronize().unwrap();
    assert_eq!(combined.output.read(&s), expected);
    workspace.guards(&s);
    let flags_before = combined.row_flags.read(&s);
    unsafe {
        assert!(combined
            .native
            .pack(
                DeviceSpan {
                    bytes: 1,
                    ..combined.input.span(&s)
                },
                combined.xs as u32,
                combined.padded.span(&s),
                combined.packed.span(&s),
                combined.row_flags.span(&s),
                s.cu_stream().cast(),
            )
            .is_err());
        // Packed storage is large enough for raw output but may not alias it.
        assert!(combined
            .native
            .dot(
                combined.weights.span(&s),
                combined.packed.span(&s),
                combined.packed.span(&s),
                combined.fixup.span(&s),
                s.cu_stream().cast(),
            )
            .is_err());
        assert!(combined
            .native
            .publish(
                combined.raw.span(&s),
                combined.output.span(&s),
                combined.n as u32 - 1,
                combined.row_flags.span(&s),
                combined.weight_flag.span(&s),
                s.cu_stream().cast(),
            )
            .is_err());
    }
    s.synchronize().unwrap();
    assert_eq!(combined.row_flags.read(&s), flags_before);
    assert_eq!(combined.output.read(&s), expected);
    combined.guards(&s);
    for m in [0, 33] {
        let r = request(&ctx, m, 5120, 49);
        let mut p = Plan::default();
        assert_ne!(unsafe { ferrum_upstream_q6_f32_plan_v1(&r, &mut p) }, 0);
    }
    let mut c = Case::new(&s, 17, 768, 17, true);
    let mut altered = c.p;
    altered.packed_bytes += 16;
    assert_eq!(
        unsafe {
            ferrum_upstream_q6_f32_dot_v1(
                &altered,
                c.weights.ptr(&s),
                c.packed.ptr(&s),
                c.raw.ptr(&s),
                c.fixup.ptr(&s),
                s.cu_stream().cast(),
            )
        },
        -1
    );
    assert_eq!(
        unsafe {
            ferrum_upstream_q6_f32_dot_v1(
                &c.p,
                (c.weights.ptr(&s) as usize + 1) as *const c_void,
                c.packed.ptr(&s),
                c.raw.ptr(&s),
                c.fixup.ptr(&s),
                s.cu_stream().cast(),
            )
        },
        -5
    );
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        1e-38,
    ] {
        let old = c.x.clone();
        c.x[3 * c.xs..3 * c.xs + 32].fill(bad);
        c.upload(&s);
        c.scan(&s);
        c.inclusive(&s);
        s.synchronize().unwrap();
        let flags = c.row_flags.read(&s);
        let y = floats(&c.output.read(&s));
        assert_ne!(u32::from_le_bytes(flags[12..16].try_into().unwrap()), 0);
        for r in 0..c.m {
            for j in 0..c.n {
                if r == 3 {
                    assert_eq!(y[r * c.ys + j].to_bits(), 0x7fc00000);
                } else {
                    assert!(y[r * c.ys + j].is_finite());
                }
            }
        }
        c.guards(&s);
        c.x = old;
    }
    // Finite tiny normal input can yield a subnormal scale that fast math
    // flushes to zero. Record actual pack evidence, not an ideal-scale gate.
    let original = c.x.clone();
    c.x[3 * c.xs..3 * c.xs + 32].fill(4e-37);
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    c.strict(&s, &kernels);
    s.synchronize().unwrap();
    assert!(c.row_flags.read(&s).iter().all(|v| *v == 0));
    let packed = c.packed.read(&s);
    let tiny_scale = f32::from_le_bytes(packed[3 * 144..3 * 144 + 4].try_into().unwrap());
    assert!(tiny_scale.is_finite() && tiny_scale >= 0.0);
    println!(
        "{}",
        serde_json::json!({"kind":"q6_f32_tiny_normal_scale_observation","input":4e-37f32,
        "scale_bits":tiny_scale.to_bits(),"scale_underflow_to_zero_allowed":true,"oracle":c.oracle(&s, false)})
    );
    c.guards(&s);
    c.x = original;
    // Actual pinned PTX is div.approx.ftz (127/amax), then rcp.approx.ftz.
    // PTX's div.approx large-divisor domain returns zero above 2^126.
    // The real marker therefore poisons those groups before integer conversion.
    // This is a compiled arithmetic boundary, not the ideal IEEE quotient.
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    s.synchronize().unwrap();
    let healthy = c.output.read(&s);
    let original = c.x.clone();
    let threshold = f32::from_bits(0x7e800000); // exactly 2^126
    for (value, poison) in [
        (threshold, false),
        (-threshold, false),
        (f32::from_bits(0x7e800001), true),
        (-f32::from_bits(0x7e800001), true),
        (f32::MAX, true),
        (-f32::MAX, true),
    ] {
        const AFFECTED: usize = 3;
        c.x = original.clone();
        c.x[AFFECTED * c.xs..AFFECTED * c.xs + c.k].fill(value);
        c.upload(&s);
        c.scan(&s);
        c.inclusive(&s);
        s.synchronize().unwrap();
        let flags: Vec<_> = c
            .row_flags
            .read(&s)
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let packed = c.packed.read(&s);
        let output = c.output.read(&s);
        let published = floats(&output);
        let raw = floats(&c.raw.read(&s));
        let affected_pack: Vec<_> = (0..c.p.padded_inputs as usize / 128)
            .flat_map(|b| {
                packed[(b * c.m + AFFECTED) * 144..(b * c.m + AFFECTED + 1) * 144]
                    .iter()
                    .copied()
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"kind":"q6_f32_div_approx_boundary",
            "input_bits":value.to_bits(),"affected_row":AFFECTED,"expected_poison":poison,
            "row_flags":flags,"affected_row_packed_bytes":affected_pack,
            "affected_row_output_bits":published[AFFECTED * c.ys..AFFECTED * c.ys + c.n].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "validation_pending":true})
        );
        assert!(c.weight_flag.read(&s).iter().all(|v| *v == 0));
        for r in 0..c.m {
            assert_eq!(flags[r], u32::from(poison && r == AFFECTED));
            for g in 0..c.p.padded_inputs as usize / 32 {
                let base = (g / 4 * c.m + r) * 144;
                let offset = base + g % 4 * 4;
                let d = u32::from_le_bytes(packed[offset..offset + 4].try_into().unwrap());
                let q = &packed[base + 16 + g % 4 * 32..base + 16 + (g % 4 + 1) * 32];
                if g * 32 >= c.k {
                    assert_eq!(d, 0);
                    assert!(q.iter().all(|v| *v == 0));
                } else if poison && r == AFFECTED {
                    assert_eq!(d, 0x7fc00000);
                    assert!(q.iter().all(|v| *v == 0));
                } else {
                    assert!(f32::from_bits(d).is_finite());
                    assert!(q.iter().all(|v| *v as i8 != i8::MIN));
                }
            }
            for j in 0..c.n {
                let v = raw[r * c.n + j];
                let expected = if flags[r] != 0 || !v.is_finite() {
                    0x7fc00000
                } else {
                    v.to_bits()
                };
                assert_eq!(published[r * c.ys + j].to_bits(), expected);
            }
            if r != AFFECTED {
                assert_eq!(
                    &output[r * c.ys * 4..(r * c.ys + c.n) * 4],
                    &healthy[r * c.ys * 4..(r * c.ys + c.n) * 4],
                    "healthy row changed"
                );
                assert!(published[r * c.ys..r * c.ys + c.n]
                    .iter()
                    .all(|v| v.is_finite()));
            }
        }
        if !poison {
            check_pack(&c, &s);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"q6_f32_div_approx_boundary_validated",
            "input_bits":value.to_bits(),"healthy_rows_unchanged":true})
        );
        c.guards(&s);
    }
    c.x = original;
    // F32 domain survives values that cannot pass through an F16 boundary.
    c.x[..c.k].fill(1e6);
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    c.strict(&s, &kernels);
    s.synchronize().unwrap();
    check_pack(&c, &s);
    c.oracle(&s, false);
    let good_weights = c.w.clone();
    c.w[208..210].copy_from_slice(&0x7c00u16.to_le_bytes());
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    s.synchronize().unwrap();
    let y = floats(&c.output.read(&s));
    assert!(y
        .chunks(c.ys)
        .all(|r| r[..c.n].iter().all(|v| v.to_bits() == 0x7fc00000)));
    c.w = good_weights;
    c.upload(&s);
    c.scan(&s);
    c.pack(&s);
    s.synchronize().unwrap();
    let mut raw = vec![1.0; c.m * c.n];
    raw[0] = f32::INFINITY;
    raw[1] = f32::NAN;
    raw[2] = f32::MAX;
    c.raw.write(&s, &bytes(&raw));
    c.publish(&s);
    s.synchronize().unwrap();
    let y = floats(&c.output.read(&s));
    assert_eq!(y[0].to_bits(), 0x7fc00000);
    assert_eq!(y[1].to_bits(), 0x7fc00000);
    assert_eq!(y[2].to_bits(), f32::MAX.to_bits());
    c.guards(&s);
}
