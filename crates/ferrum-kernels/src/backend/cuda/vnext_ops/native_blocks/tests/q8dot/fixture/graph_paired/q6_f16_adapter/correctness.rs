use super::*;

fn check_pack(c: &Case, s: &Arc<CudaStream>) {
    let packed = c.packed.read(s);
    let converted = floats(&c.converted.read(s));
    assert!(c.row_flags.read(s).iter().all(|b| *b == 0));
    for r in 0..c.m {
        for i in 0..c.p.padded_inputs as usize {
            let expected = if i < c.k {
                c.x[r * c.xs + i].to_f32()
            } else {
                0.0
            };
            assert_eq!(
                converted[r * c.p.padded_inputs as usize + i].to_bits(),
                expected.to_bits(),
                "exact F16 widening and K padding"
            );
        }
        for g in 0..c.p.padded_inputs as usize / 32 {
            let base = ((g / 4) * c.m + r) * 144;
            let scale = f32::from_le_bytes(
                packed[base + g % 4 * 4..base + g % 4 * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let codes = &packed[base + 16 + g % 4 * 32..base + 16 + (g % 4 + 1) * 32];
            assert!(scale.is_finite() && scale >= 0.0);
            assert!(codes.iter().all(|v| *v as i8 != i8::MIN));
            if g * 32 >= c.k {
                assert_eq!(scale.to_bits(), 0);
                assert!(codes.iter().all(|v| *v == 0));
                continue;
            }
            let x = &c.x[r * c.xs + g * 32..r * c.xs + (g + 1) * 32];
            let amax = x.iter().map(|x| x.to_f32().abs()).fold(0f32, f32::max);
            if amax == 0.0 {
                assert_eq!(scale.to_bits(), 0);
                assert!(codes.iter().all(|v| *v == 0));
            } else {
                assert!((scale - amax / 127.0).abs() <= 8.0 * f32::EPSILON * (amax / 127.0));
                for (i, x) in x.iter().enumerate() {
                    assert!(
                        (x.to_f64() - oracle::activation(&packed, c.m, r, g * 32 + i)).abs()
                            <= f64::from(scale) * 0.501
                                + f64::from(amax) * 8.0 * f64::from(f32::EPSILON)
                    );
                }
            }
        }
    }
    assert!(
        packed[c.m * c.p.padded_inputs as usize * 9 / 8..]
            .iter()
            .all(|b| *b == 0),
        "packed loader guard reset"
    );
}

#[test]
#[ignore = "exclusive CUDA; requires the diagnostic Q6 F16 ABI2 artifact"]
fn q6_f16_adapter_actual_pack_f64_markers_strides_and_bounds() {
    let (ctx, s, kernels) = context();
    for m in [1, 4, 7, 8, 9, 16, 17, 31, 32] {
        for (k, n) in [(512, 7), (768, 129)] {
            let mut c = Case::new(&s, m, k, n, true, false);
            c.x[..32].fill(f16::ZERO);
            c.x[32..64].fill(f16::from_bits(1));
            c.upload(&s);
            c.scan(&s);
            c.inclusive(&s);
            c.strict(&s, &kernels);
            s.synchronize().unwrap();
            check_pack(&c, &s);
            let oracle = c.validate(&s, true);
            let before = c.output.read(&s);
            let graph = Captured::new(&s, || c.inclusive(&s));
            graph.launch();
            graph.launch();
            s.synchronize().unwrap();
            assert_eq!(
                before,
                c.output.read(&s),
                "same-route captured full output bits"
            );
            c.guards(&s);
            println!(
                "{}",
                serde_json::json!({"kind":"q6_f16_adapter_correctness","m":m,"k":k,"n":n,
                "input_stride":c.xs,"output_stride":c.ys,"output_offset":c.output_offset,"weight_byte_offset":c.weight_offset,
                "j":c.p.j,"fixup":c.p.fixup,"guard_blocks":c.p.guard_blocks,"oracle":oracle})
            );
            drop(graph);
        }
    }
    // The real final-layer up projection occupies the second half of a joined
    // gate/up tensor. Neither algorithm may touch its neighboring first leaf.
    let c = Case::new(&s, 4, 5120, 17408, true, true);
    c.scan(&s);
    c.inclusive(&s);
    c.strict(&s, &kernels);
    s.synchronize().unwrap();
    check_pack(&c, &s);
    println!(
        "{}",
        serde_json::json!({"kind":"q6_f16_adapter_ffn_leaf_stride","m":c.m,"k":c.k,"n":c.n,
        "output_offset":c.output_offset,"output_stride":c.ys,"oracle":c.validate(&s, false)})
    );
    drop(c);

    let mut c = Case::new(&s, 7, 768, 17, true, false);
    c.scan(&s);
    c.inclusive(&s);
    c.strict(&s, &kernels);
    s.synchronize().unwrap();
    let healthy = c.output.read(&s);
    let original_x = c.x.clone();
    for bad in [0x7e01, 0x7c00, 0xfc00] {
        c.x = original_x.clone();
        c.x[6 * c.xs..6 * c.xs + 32].fill(f16::from_bits(bad));
        c.upload(&s);
        c.inclusive(&s);
        s.synchronize().unwrap();
        let flags = c.row_flags.read(&s);
        let output = c.output.read(&s);
        for row in 0..c.m {
            assert_eq!(
                u32::from_le_bytes(flags[row * 4..row * 4 + 4].try_into().unwrap()),
                u32::from(row == 6)
            );
            let range =
                (row * c.ys + c.output_offset) * 2..(row * c.ys + c.output_offset + c.n) * 2;
            if row == 6 {
                assert!(halves(&output[range]).iter().all(|x| x.to_bits() == 0x7e00));
            } else {
                assert_eq!(&output[range.clone()], &healthy[range]);
            }
        }
        let packed = c.packed.read(&s);
        let base = 6 * 144;
        assert_eq!(
            u32::from_le_bytes(packed[base..base + 4].try_into().unwrap()),
            0x7fc00000
        );
        assert!(packed[base + 16..base + 48].iter().all(|v| *v == 0));
        assert!(packed[c.m * c.p.padded_inputs as usize * 9 / 8..]
            .iter()
            .all(|b| *b == 0));
        c.guards(&s);
    }
    c.x = original_x;
    c.upload(&s);
    c.inclusive(&s);
    s.synchronize().unwrap();
    assert_eq!(
        healthy,
        c.output.read(&s),
        "row poison clears on fresh pack"
    );
    let original_w = c.w.clone();
    // Last physical block validates scanner coverage, not just its first tile.
    let last = c.w.len() - 2;
    c.w[last..].copy_from_slice(&0x7c00u16.to_le_bytes());
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    s.synchronize().unwrap();
    assert_ne!(
        u32::from_le_bytes(c.weight_flag.read(&s).try_into().unwrap()),
        0
    );
    let poisoned = halves(&c.output.read(&s));
    for row in 0..c.m {
        assert!(
            poisoned[row * c.ys + c.output_offset..row * c.ys + c.output_offset + c.n]
                .iter()
                .all(|v| v.to_bits() == 0x7e00)
        );
    }
    c.guards(&s);
    c.w = original_w;
    c.upload(&s);
    c.scan(&s);
    c.inclusive(&s);
    s.synchronize().unwrap();
    assert_eq!(
        healthy,
        c.output.read(&s),
        "fresh weight scan clears prior poison"
    );
    let values = [
        0.0,
        -0.0,
        65504.0,
        -65504.0,
        65519.0,
        65520.0,
        -65520.0,
        f32::INFINITY,
        f32::NAN,
        f32::MAX,
        2f32.powi(-24),
    ];
    let raw: Vec<_> = (0..c.m * c.n).map(|i| values[i % values.len()]).collect();
    c.raw.write(&s, &float_bytes(&raw));
    c.publish(&s);
    s.synchronize().unwrap();
    let output = halves(&c.output.read(&s));
    for row in 0..c.m {
        for column in 0..c.n {
            let raw = raw[row * c.n + column];
            let cast = f16::from_f32(raw);
            let expected = if !raw.is_finite() || !cast.is_finite() {
                0x7e00
            } else {
                cast.to_bits()
            };
            assert_eq!(
                output[row * c.ys + c.output_offset + column].to_bits(),
                expected
            );
        }
    }
    c.guards(&s);
    invalid_boundaries(&ctx, &s, &c);
    println!(
        "{}",
        serde_json::json!({"kind":"q6_f16_adapter_boundaries_complete","input_nonfinite_row_isolation":true,
        "last_weight_block_poison":true,"fresh_row_and_weight_reset":true,"finite_overflow_canonical_nan":true,
        "signed_zero_subnormal_cast":true,"invalid_abi_plan_stride_alignment_rejected_before_writes":true})
    );
}

fn invalid_boundaries(ctx: &CudaContext, s: &Arc<CudaStream>, c: &Case) {
    for (m, k, n, abi_tag) in [
        (0, 512, 7, 2),
        (33, 512, 7, 2),
        (4, 128, 7, 2),
        (4, 512, 0, 2),
        (4, 512, 7, 1),
    ] {
        let mut request = abi::request(ctx, m, k, n);
        request.abi = abi_tag;
        let mut p = Plan::default();
        assert_ne!(
            unsafe { abi::ferrum_upstream_q6_f16_plan_v1(&request, &mut p) },
            0
        );
    }
    let before = [
        c.converted.read(s),
        c.packed.read(s),
        c.raw.read(s),
        c.output.read(s),
        c.row_flags.read(s),
        c.weight_flag.read(s),
    ];
    let mut wrong = c.p;
    wrong.packed_bytes += 16;
    unsafe {
        assert_ne!(
            abi::ferrum_upstream_q6_f16_dot_v1(
                &wrong,
                c.weight_ptr(s),
                c.packed.ptr(s),
                c.raw.ptr(s),
                c.fixup.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        wrong = c.p;
        wrong.abi = 1;
        assert_ne!(
            abi::ferrum_upstream_q6_f16_pack_v1(
                &wrong,
                c.input.ptr(s),
                c.xs as u32,
                c.converted.ptr(s),
                c.packed.ptr(s),
                c.row_flags.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        assert_ne!(
            abi::ferrum_upstream_q6_f16_pack_v1(
                &c.p,
                c.input.ptr(s),
                c.k as u32 - 1,
                c.converted.ptr(s),
                c.packed.ptr(s),
                c.row_flags.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        assert_ne!(
            abi::ferrum_upstream_q6_f16_pack_v1(
                &c.p,
                (c.input.ptr(s) as usize + 1) as *const c_void,
                c.xs as u32,
                c.converted.ptr(s),
                c.packed.ptr(s),
                c.row_flags.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        assert_ne!(
            abi::ferrum_upstream_q6_f16_cast_v1(
                &c.p,
                c.raw.ptr(s),
                c.output_ptr(s),
                c.n as u32 - 1,
                c.row_flags.ptr(s),
                c.weight_flag.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        assert_ne!(
            abi::ferrum_upstream_q6_f16_cast_v1(
                &c.p,
                c.raw.ptr(s),
                (c.output_ptr(s) as usize + 1) as *mut c_void,
                c.ys as u32,
                c.row_flags.ptr(s),
                c.weight_flag.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
        assert_ne!(
            abi::ferrum_upstream_q6_f16_check_weights_v1(
                &c.p,
                (c.weight_ptr(s) as usize + 1) as *const c_void,
                c.weight_flag.ptr(s),
                s.cu_stream().cast()
            ),
            0
        );
    }
    s.synchronize().unwrap();
    let after = [
        c.converted.read(s),
        c.packed.read(s),
        c.raw.read(s),
        c.output.read(s),
        c.row_flags.read(s),
        c.weight_flag.read(s),
    ];
    assert_eq!(before, after, "contract rejection must not enqueue writes");
    c.guards(s);
}
