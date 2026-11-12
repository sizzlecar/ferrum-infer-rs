//! Run only against an explicitly linked, source-builder-produced native archive.
//! No runtime library loading, model, provider registration or timing threshold.
#![cfg(feature = "cuda-upstream-linear-tests")]
use ferrum_native_ops::upstream_linear::*;
use half::f16;
use std::{
    ffi::c_void,
    mem::{size_of, size_of_val},
    ptr,
};
#[path = "../../ferrum-kernels/src/native_ops/upstream_linear/ffi.rs"]
mod ffi;
#[path = "upstream_linear_cuda/oracle.rs"]
mod oracle;
#[path = "upstream_linear_cuda/prefill.rs"]
mod prefill;
#[path = "upstream_linear_cuda/timing.rs"]
mod timing;
#[link(name = "ferrum_upstream_linear", kind = "static")]
unsafe extern "C" {}
#[link(name = "cudart")]
#[link(name = "stdc++")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
}
fn ok(status: i32) {
    assert_eq!(status, 0, "CUDA/native ABI status {status}");
}
struct Buffer {
    base: *mut c_void,
    ptr: *mut c_void,
    bytes: usize,
}
impl Buffer {
    fn new(bytes: usize) -> Self {
        unsafe {
            let mut base = ptr::null_mut();
            ok(cudaMalloc(&mut base, bytes + 64));
            ok(cudaMemset(base, 0x35, bytes + 64));
            Self {
                base,
                ptr: base.add(32),
                bytes,
            }
        }
    }
    fn write<T: Copy>(&self, values: &[T]) {
        assert!(size_of_val(values) <= self.bytes);
        unsafe {
            ok(cudaMemcpy(
                self.ptr,
                values.as_ptr().cast(),
                size_of_val(values),
                1,
            ));
        }
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        assert!(n * size_of::<T>() <= self.bytes);
        let mut v = vec![T::default(); n];
        unsafe {
            ok(cudaMemcpy(
                v.as_mut_ptr().cast(),
                self.ptr,
                n * size_of::<T>(),
                2,
            ));
        }
        v
    }
    fn guards(&self) {
        unsafe {
            let mut b = [0u8; 32];
            ok(cudaMemcpy(b.as_mut_ptr().cast(), self.base, 32, 2));
            assert_eq!(b, [0x35; 32]);
            ok(cudaMemcpy(
                b.as_mut_ptr().cast(),
                self.ptr.add(self.bytes),
                32,
                2,
            ));
            assert_eq!(b, [0x35; 32]);
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaFree(self.base);
        }
    }
}
struct Stream(*mut c_void);
impl Stream {
    fn new() -> Self {
        let mut s = ptr::null_mut();
        unsafe {
            ok(cudaStreamCreate(&mut s));
        }
        Self(s)
    }
    fn sync(&self) {
        unsafe {
            ok(cudaStreamSynchronize(self.0));
        }
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaStreamDestroy(self.0);
        }
    }
}
fn caps() -> UpstreamLinearDevice {
    unsafe {
        ok(cudaSetDevice(0));
        let get = |attr| {
            let mut v = 0;
            ok(cudaDeviceGetAttribute(&mut v, attr, 0));
            v as u32
        };
        UpstreamLinearDevice {
            architecture: get(75) * 100 + get(76) * 10,
            multiprocessors: get(16),
            maximum_dynamic_shared_bytes: get(97) as u64,
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Input {
    Finite,
    Zero,
    Nonfinite,
    SumOverflow,
    Subnormal,
}
fn weights(format: UpstreamLinearFormat, k: usize, n: usize) -> Vec<u8> {
    let bytes = match format {
        UpstreamLinearFormat::Iq4Xs => 136,
        UpstreamLinearFormat::Q4K => 144,
        UpstreamLinearFormat::Q5K => 176,
    };
    let mut out = vec![0; bytes * (k / 256) * n];
    for block in out.chunks_exact_mut(bytes) {
        block[..2].copy_from_slice(&f16::from_f32(0.125).to_bits().to_le_bytes());
        match format {
            UpstreamLinearFormat::Iq4Xs => {
                block[2..4].copy_from_slice(&0xaaaau16.to_le_bytes());
                block[4..8].fill(0x11);
                block[8..].fill(0x85);
            }
            UpstreamLinearFormat::Q4K | UpstreamLinearFormat::Q5K => {
                block[2..4].copy_from_slice(&f16::from_f32(0.03125).to_bits().to_le_bytes());
                block[4..8].fill(2);
                block[8..12].fill(3);
                block[12..16].fill(0x32);
                let offset = if format == UpstreamLinearFormat::Q5K {
                    block[16..48].fill(0xa5);
                    48
                } else {
                    16
                };
                block[offset..].fill(0x21);
            }
        }
    }
    out
}
fn plan(
    a: UpstreamLinearAlgorithm,
    f: UpstreamLinearFormat,
    l: UpstreamLinearLayout,
    m: u32,
    k: u32,
    n: u32,
) -> UpstreamLinearPlanV1 {
    let r = UpstreamLinearRequestV1::new(f, l, m, k, n, caps()).unwrap();
    let mut p = UpstreamLinearPlanV1::default();
    unsafe {
        ok(match a {
            UpstreamLinearAlgorithm::Mmq if m > 32 => {
                ffi::ferrum_upstream_mmq_prefill_plan_v1(&r, &mut p)
            }
            UpstreamLinearAlgorithm::Mmq => ffi::ferrum_upstream_mmq_plan_v1(&r, &mut p),
            UpstreamLinearAlgorithm::Mmvq => ffi::ferrum_upstream_mmvq_plan_v1(&r, &mut p),
        })
    };
    p.validate_identity(&r, a).unwrap();
    p
}
fn check(
    a: UpstreamLinearAlgorithm,
    f: UpstreamLinearFormat,
    l: UpstreamLinearLayout,
    m: u32,
    k: u32,
    input_kind: Input,
    weight_bad: bool,
) {
    check_geometry(a, f, l, [m, k, 17], input_kind, weight_bad, false);
}

fn check_geometry(
    a: UpstreamLinearAlgorithm,
    f: UpstreamLinearFormat,
    l: UpstreamLinearLayout,
    [m, k, n]: [u32; 3],
    input_kind: Input,
    weight_bad: bool,
    distinct_rows_and_columns: bool,
) -> UpstreamLinearPlanV1 {
    let p = plan(a, f, l, m, k, n);
    let s = Stream::new();
    let stride = k as usize + 3;
    let outstride = n as usize + 5;
    let mut x = vec![0x3c00u16; stride * m as usize];
    for row in 0..m as usize {
        for col in 0..k as usize {
            x[row * stride + col] = match input_kind {
                Input::Zero => 0x8000,
                Input::Subnormal => {
                    if col % 2 == 0 {
                        1
                    } else {
                        0x8001
                    }
                }
                Input::SumOverflow => 0x7bff,
                _ => {
                    let index = col
                        + if distinct_rows_and_columns {
                            row * 7
                        } else {
                            0
                        };
                    f16::from_f32(((index % 19) as f32 - 9.0) / 16.0).to_bits()
                }
            };
        }
    }
    if matches!(input_kind, Input::Nonfinite) {
        x[(m as usize - 1) * stride] = 0x7e00;
        if m > 1 {
            x[0] = 0x7c00;
        }
        if m > 2 {
            x[stride] = 0xfc00;
        }
    }
    let mut w = weights(f, k as usize, p.padded_outputs as usize);
    let row_bytes = w.len() / p.padded_outputs as usize;
    w[n as usize * row_bytes..].fill(0);
    if distinct_rows_and_columns {
        let block_bytes = row_bytes / (k as usize / 256);
        for (column, row) in w[..n as usize * row_bytes]
            .chunks_exact_mut(row_bytes)
            .enumerate()
        {
            for block in row.chunks_exact_mut(block_bytes) {
                block[..2].copy_from_slice(
                    &f16::from_f32(0.0625 + (column % 5) as f32 / 64.0)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
    }
    if weight_bad {
        w[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
        if f == UpstreamLinearFormat::Iq4Xs || a == UpstreamLinearAlgorithm::Mmvq {
            w[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        }
    }
    let bx = Buffer::new(x.len() * 2);
    bx.write(&x);
    let bw = Buffer::new(w.len());
    bw.write(&w);
    let converted = Buffer::new(p.converted_bytes as usize);
    let packed = Buffer::new(p.packed_bytes as usize);
    let output = Buffer::new(p.output_bytes as usize);
    let fixup = Buffer::new(p.fixup_bytes.max(4) as usize);
    let result = Buffer::new(m as usize * outstride * 2);
    let rowflag = Buffer::new(m as usize * 4);
    let weightflag = Buffer::new(4);
    unsafe {
        ok(match a {
            UpstreamLinearAlgorithm::Mmq => {
                ffi::ferrum_upstream_mmq_check_weights_v2(&p, bw.ptr, weightflag.ptr, s.0)
            }
            UpstreamLinearAlgorithm::Mmvq => {
                ffi::ferrum_upstream_mmvq_check_weights_v2(&p, bw.ptr, weightflag.ptr, s.0)
            }
        });
    }
    let mut previous = None;
    for _ in 0..2 {
        unsafe {
            result.write(&vec![0x3555u16; m as usize * outstride]);
            ok(cudaMemset(packed.ptr, 0x35, packed.bytes));
            ok(match a {
                UpstreamLinearAlgorithm::Mmq => ffi::ferrum_upstream_mmq_pack_v2(
                    &p,
                    bx.ptr,
                    stride as u32,
                    converted.ptr,
                    packed.ptr,
                    rowflag.ptr,
                    s.0,
                ),
                UpstreamLinearAlgorithm::Mmvq => ffi::ferrum_upstream_mmvq_pack_v2(
                    &p,
                    bx.ptr,
                    stride as u32,
                    converted.ptr,
                    packed.ptr,
                    rowflag.ptr,
                    s.0,
                ),
            });
            let fix = if p.fixup_bytes > 0 {
                fixup.ptr
            } else {
                ptr::null_mut()
            };
            ok(match a {
                UpstreamLinearAlgorithm::Mmq => {
                    ffi::ferrum_upstream_mmq_dot_v1(&p, bw.ptr, packed.ptr, output.ptr, fix, s.0)
                }
                UpstreamLinearAlgorithm::Mmvq => {
                    ffi::ferrum_upstream_mmvq_dot_v1(&p, bw.ptr, packed.ptr, output.ptr, fix, s.0)
                }
            });
            ok(match a {
                UpstreamLinearAlgorithm::Mmq => ffi::ferrum_upstream_mmq_cast_v2(
                    &p,
                    output.ptr,
                    result.ptr,
                    outstride as u32,
                    rowflag.ptr,
                    weightflag.ptr,
                    s.0,
                ),
                UpstreamLinearAlgorithm::Mmvq => ffi::ferrum_upstream_mmvq_cast_v2(
                    &p,
                    output.ptr,
                    result.ptr,
                    outstride as u32,
                    rowflag.ptr,
                    weightflag.ptr,
                    s.0,
                ),
            });
        }
        s.sync();
        let rows = rowflag.read::<u32>(m as usize);
        let leaf = weightflag.read::<u32>(1)[0];
        let expected_leaf = w
            .chunks_exact(match f {
                UpstreamLinearFormat::Iq4Xs => 136,
                UpstreamLinearFormat::Q4K => 144,
                UpstreamLinearFormat::Q5K => 176,
            })
            .any(|b| oracle::classify_weight(a, f, b));
        assert_eq!(leaf != 0, expected_leaf);
        let q = packed.read::<u8>(p.packed_bytes as usize);
        if a == UpstreamLinearAlgorithm::Mmq {
            let tail = m as usize * p.padded_inputs as usize * 9 / 8;
            assert!(
                q[tail..].iter().all(|&byte| byte == 0),
                "MMQ guard tail written"
            );
        }
        let y = result.read::<u16>(m as usize * outstride);
        let yf = output.read::<f32>(m as usize * n as usize);
        for row in 0..m as usize {
            let mut row_bad = false;
            for group in 0..p.padded_inputs as usize / 32 {
                let logical = group * 32 < k as usize;
                let values = std::array::from_fn(|i| {
                    if logical {
                        x[row * stride + group * 32 + i]
                    } else {
                        0
                    }
                });
                let (zero, bad) = oracle::marker_classify_pack(a, f, &values);
                row_bad |= bad;
                let (metadata, codes) = if a == UpstreamLinearAlgorithm::Mmq {
                    let block = (group / 4) * m as usize + row;
                    (
                        &q[block * 144 + (group % 4) * 4..][..4],
                        &q[block * 144 + 16 + (group % 4) * 32..][..32],
                    )
                } else {
                    let block = row * (p.padded_inputs as usize / 32) + group;
                    (&q[block * 36..][..4], &q[block * 36 + 4..][..32])
                };
                if zero {
                    assert_eq!(metadata, &[0; 4]);
                    assert!(codes.iter().all(|&v| v == 0));
                }
                if bad {
                    assert!(codes.iter().all(|&v| v == 0));
                    if p.pack_abi == 1 {
                        assert_eq!(u32::from_le_bytes(metadata.try_into().unwrap()), 0x7fc00000);
                    } else {
                        assert_eq!(
                            u16::from_le_bytes(metadata[..2].try_into().unwrap()),
                            0x7e00
                        );
                    }
                }
            }
            assert_eq!(rows[row] != 0, row_bad);
            for col in 0..n as usize {
                let actual = yf[row * n as usize + col];
                assert_eq!(
                    y[row * outstride + col],
                    oracle::final_cast_bits(actual, row_bad, expected_leaf)
                );
                if !row_bad && !expected_leaf {
                    let mut target = 0.0;
                    let mut magnitude = 0.0;
                    let wb = match f {
                        UpstreamLinearFormat::Iq4Xs => 136,
                        UpstreamLinearFormat::Q4K => 144,
                        UpstreamLinearFormat::Q5K => 176,
                    };
                    for group in 0..k as usize / 32 {
                        let (meta, codes) = if a == UpstreamLinearAlgorithm::Mmq {
                            let b = (group / 4) * m as usize + row;
                            (
                                &q[b * 144 + (group % 4) * 4..][..4],
                                &q[b * 144 + 16 + (group % 4) * 32..][..32],
                            )
                        } else {
                            let b = row * (p.padded_inputs as usize / 32) + group;
                            (&q[b * 36..][..4], &q[b * 36 + 4..][..32])
                        };
                        let scale = if p.pack_abi == 1 {
                            f32::from_le_bytes(meta.try_into().unwrap())
                        } else {
                            f16::from_bits(u16::from_le_bytes(meta[..2].try_into().unwrap()))
                                .to_f32()
                        };
                        let original_sum = if p.pack_abi == 1 {
                            0.0
                        } else {
                            f16::from_bits(u16::from_le_bytes(meta[2..].try_into().unwrap()))
                                .to_f32()
                        };
                        let offset = (col * (k as usize / 256) + group / 8) * wb;
                        let (t, mag) = oracle::declared_group(
                            a,
                            f,
                            &w[offset..offset + wb],
                            group % 8,
                            &std::array::from_fn(|i| codes[i] as i8),
                            scale,
                            original_sum,
                        );
                        target += t;
                        magnitude += mag;
                    }
                    let operations = (k as f64 / 32.0 + 32.0) * f32::EPSILON as f64;
                    let bound = operations / (1.0 - operations) * magnitude
                        + 32.0 * f32::MIN_POSITIVE as f64;
                    assert!((actual as f64-target).abs()<=bound,"actual-pack F64 {a:?}/{f:?} {input_kind:?}: {actual} vs {target}, bound {bound}");
                }
            }
            assert!(y[row * outstride + n as usize..(row + 1) * outstride]
                .iter()
                .all(|&v| v == 0x3555));
        }
        if let Some(prev) = previous {
            assert_eq!(y, prev, "repeat bits {a:?}/{f:?}/{input_kind:?}");
        }
        previous = Some(y);
        for b in [
            &bx,
            &bw,
            &converted,
            &packed,
            &output,
            &fixup,
            &result,
            &rowflag,
            &weightflag,
        ] {
            b.guards();
        }
    }
    if matches!(input_kind, Input::Finite) && !weight_bad {
        let candidate = output.read::<u32>(m as usize * n as usize);
        unsafe {
            ok(match a {
                UpstreamLinearAlgorithm::Mmq => ffi::ferrum_upstream_mmq_pack_v1(
                    &p,
                    bx.ptr,
                    stride as u32,
                    converted.ptr,
                    packed.ptr,
                    s.0,
                ),
                UpstreamLinearAlgorithm::Mmvq => ffi::ferrum_upstream_mmvq_pack_v1(
                    &p,
                    bx.ptr,
                    stride as u32,
                    converted.ptr,
                    packed.ptr,
                    s.0,
                ),
            });
            ok(match a {
                UpstreamLinearAlgorithm::Mmq => ffi::ferrum_upstream_mmq_dot_v1(
                    &p,
                    bw.ptr,
                    packed.ptr,
                    output.ptr,
                    if p.fixup_bytes > 0 {
                        fixup.ptr
                    } else {
                        ptr::null_mut()
                    },
                    s.0,
                ),
                UpstreamLinearAlgorithm::Mmvq => ffi::ferrum_upstream_mmvq_dot_v1(
                    &p,
                    bw.ptr,
                    packed.ptr,
                    output.ptr,
                    ptr::null_mut(),
                    s.0,
                ),
            });
        }
        s.sync();
        assert_eq!(
            candidate,
            output.read::<u32>(m as usize * n as usize),
            "qualified V1/V2 dot bits"
        );
    }
    eprintln!("native_marker_case algorithm={a:?} format={f:?} layout={l:?} M={m} K={k} N={n} input={input_kind:?} weight_bad={weight_bad}");
    p
}
#[test]
#[ignore = "requires a source-builder-produced CUDA upstream linear artifact and GPU"]
fn marker_v2_real_pack_dot_cast_and_weight_flags() {
    for a in [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq] {
        for f in [
            UpstreamLinearFormat::Iq4Xs,
            UpstreamLinearFormat::Q4K,
            UpstreamLinearFormat::Q5K,
        ] {
            let widths: &[u32] = if a == UpstreamLinearAlgorithm::Mmq {
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32]
            } else {
                &[1, 4, 8]
            };
            for &m in widths {
                for kind in [
                    Input::Finite,
                    Input::Zero,
                    Input::Nonfinite,
                    Input::SumOverflow,
                    Input::Subnormal,
                ] {
                    check(a, f, UpstreamLinearLayout::Columns, m, 768, kind, false);
                }
            }
            check(
                a,
                f,
                UpstreamLinearLayout::Columns,
                4,
                256,
                Input::Finite,
                true,
            );
        }
    }
    check(
        UpstreamLinearAlgorithm::Mmvq,
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearLayout::Channels,
        32,
        256,
        Input::Finite,
        false,
    );
}
