use super::*;
#[test]
#[ignore = "requires the actual production extra operator lock and exclusive CUDA"]
fn production_extra_actual_pack_oracles_markers_bounds_and_repeated_execution_on_cuda() {
    let (ctx, s, kernels) = context();
    for f in FORMATS {
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32] {
            let algorithms = if [1, 4, 8].contains(&m) {
                vec![1, 2]
            } else {
                vec![1]
            };
            let mut c = Case::new(&s, f, m, 768, 17, true, 0, &algorithms);
            for r in 0..c.routes.len() {
                let mut previous = None;
                for _ in 0..2 {
                    c.reset(&s);
                    c.native(&s, r);
                    let actual = c.validate(&s, Some(r));
                    if let Some(old) = previous {
                        assert_eq!(actual, old)
                    }
                    previous = Some(actual)
                }
            }
        }
        // Actual projection widths and a 512-pack tail. Large N is exercised
        // separately by the real-projection qualification; every output here.
        for k in [5120, 17408, 5120 + 256, 17408 + 256] {
            let mut c = Case::new(&s, f, 8, k, 49, false, 1, &[1, 2]);
            c.reset(&s);
            c.strict(&s, &kernels);
            c.validate(&s, None);
            for r in 0..2 {
                c.native(&s, r);
                c.validate(&s, Some(r));
            }
        }
        // Typed production planner rejects unsupported geometry; these are not
        // raw test-only FFI requests or success inferred from an old artifact.
        for algorithm in [Algorithm::Mmq, Algorithm::Mmvq] {
            for (m, k) in [(0, 256), (33, 256), (8, 32), (8, 96), (8, 257)] {
                let request = request(&ctx, f, 8, 256, 17);
                assert!(PreparedUpstreamLinear::new_extra(
                    algorithm,
                    Arithmetic::MarkerV2,
                    extra_format(f),
                    Layout::Columns,
                    m,
                    k,
                    17,
                    Device {
                        architecture: request.cc,
                        multiprocessors: request.sm_count,
                        maximum_dynamic_shared_bytes: request.shared_limit
                    }
                )
                .is_err());
            }
        }
        let mut c = Case::new(&s, f, 8, 768, 17, false, 2, &[1, 2]);
        for r in &c.routes {
            let before = r.weight_flag.read(&s);
            let good = c.bw.span(&s);
            for bad in [
                DeviceSpan {
                    address: good.address + 1,
                    bytes: good.bytes - 1,
                },
                DeviceSpan {
                    address: good.address + 2,
                    bytes: good.bytes - 2,
                },
                DeviceSpan {
                    address: good.address + 3,
                    bytes: good.bytes - 3,
                },
                DeviceSpan {
                    address: good.address,
                    bytes: r.p.weight_bytes - 1,
                },
                DeviceSpan {
                    address: u64::MAX - 3,
                    bytes: 8,
                },
            ] {
                assert!(unsafe {
                    r.native
                        .check_weights(bad, r.weight_flag.span(&s), s.cu_stream().cast())
                }
                .is_err());
                assert_eq!(r.weight_flag.read(&s), before);
            }
            assert!(
                unsafe {
                    r.native.check_weights(
                        good,
                        DeviceSpan {
                            address: good.address,
                            bytes: 4,
                        },
                        s.cu_stream().cast(),
                    )
                }
                .is_err(),
                "weight/flag alias"
            );
            let packed_before = r.packed.read(&s);
            assert!(unsafe {
                r.native.pack(
                    c.bx.span(&s),
                    c.k as u32 - 1,
                    r.converted.span(&s),
                    r.packed.span(&s),
                    r.rows.span(&s),
                    s.cu_stream().cast(),
                )
            }
            .is_err());
            assert_eq!(r.packed.read(&s), packed_before);
            let raw_before = r.raw.read(&s);
            let packed = r.packed.span(&s);
            assert!(unsafe {
                r.native.dot(
                    good,
                    DeviceSpan {
                        address: packed.address,
                        bytes: packed.bytes - 1,
                    },
                    r.raw.span(&s),
                    r.fixup.span(&s),
                    s.cu_stream().cast(),
                )
            }
            .is_err());
            assert_eq!(r.raw.read(&s), raw_before);
            let output_before = r.output.read(&s);
            let output = r.output.span(&s);
            assert!(unsafe {
                r.native.cast(
                    r.raw.span(&s),
                    DeviceSpan {
                        address: output.address,
                        bytes: 2,
                    },
                    c.ystride as u32,
                    r.rows.span(&s),
                    r.weight_flag.span(&s),
                    s.cu_stream().cast(),
                )
            }
            .is_err());
            assert_eq!(r.output.read(&s), output_before);
        }
        for kind in [0, 1, 2, 3] {
            for row in 0..c.m {
                for i in 0..c.k {
                    c.x[row * c.xstride + i] = match kind {
                        0 => f16::from_bits(0x8000),
                        1 => f16::from_bits(1 + ((row + i) % 1023) as u16),
                        2 => f16::from_f32(65504.0),
                        _ => f16::from_f32(((i % 7) as f32 - 3.0) / 64.0),
                    }
                }
            }
            if kind == 3 {
                c.x[0] = f16::INFINITY;
                c.x[c.xstride] = f16::NEG_INFINITY;
                c.x[7 * c.xstride] = f16::NAN;
            }
            c.upload(&s);
            for r in 0..2 {
                c.reset(&s);
                c.native(&s, r);
                c.validate(&s, Some(r));
            }
        }
        // Nonfinite static scale at the format's true offset must poison all
        // outputs, even with finite inputs and a previously successful scan.
        c.x.fill(f16::from_f32(0.01));
        let offset = if f == GgufBlockFormat::Q3K { 108 } else { 0 };
        c.w[offset..offset + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
        c.upload(&s);
        for r in 0..2 {
            c.reset(&s);
            c.native(&s, r);
            c.validate(&s, Some(r));
        }
    }
    s.synchronize().unwrap();
}

#[test]
#[ignore = "requires production extra artifact; real shapes with sparse F64 oracle and full guards"]
fn production_extra_real_projection_shapes_preserve_oracle_and_repeat_bits_on_cuda() {
    let (_, s, kernels) = context();
    for f in FORMATS {
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            for m in [1, 4, 8, 16, 32] {
                let algorithms = if [1, 4, 8].contains(&m) {
                    vec![1, 2]
                } else {
                    vec![1]
                };
                let mut c = Case::new(&s, f, m, k, n, false, 0, &algorithms);
                c.reset(&s);
                c.strict(&s, &kernels);
                c.validate(&s, None);
                for r in 0..c.routes.len() {
                    let mut previous = None;
                    for _ in 0..2 {
                        c.reset(&s);
                        c.native(&s, r);
                        let bits = c.validate(&s, Some(r));
                        if let Some(ref old) = previous {
                            assert_eq!(&bits, old);
                        }
                        previous = Some(bits);
                    }
                }
            }
        }
    }
    s.synchronize().unwrap();
}
