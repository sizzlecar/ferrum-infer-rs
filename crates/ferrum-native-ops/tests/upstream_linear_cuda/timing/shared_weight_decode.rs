//! Test-only Q4K/Q5K Columns M8 dot alternatives. Both arms retain the same
//! MarkerV2 pack, static weight check, cast and physical working set.
use super::*;

unsafe extern "C" {
    fn ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_upstream_mmvq_q4_shared_weight_decode_m4_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_upstream_mmvq_q5_shared_weight_decode_m4_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

const DOTS: [MmvqDot; 2] = [
    ffi::ferrum_upstream_mmvq_dot_v1,
    ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1,
];
const Q5_DOTS: [MmvqDot; 2] = [
    ffi::ferrum_upstream_mmvq_dot_v1,
    ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1,
];
const LABELS: [&str; 2] = ["baseline", "shared_weight_decode"];
const FORMAT: UpstreamLinearFormat = UpstreamLinearFormat::Q4K;
const ALGORITHM: UpstreamLinearAlgorithm = UpstreamLinearAlgorithm::Mmvq;

pub(super) fn selected_dot(format: UpstreamLinearFormat) -> MmvqDot {
    match format {
        UpstreamLinearFormat::Q4K => DOTS[1],
        UpstreamLinearFormat::Q5K => Q5_DOTS[1],
        UpstreamLinearFormat::Iq4Xs => ffi::ferrum_upstream_mmvq_dot_v1,
    }
}

pub(super) fn validate_selected_markers(format: UpstreamLinearFormat) {
    assert_marker_bits_and_boundaries(
        format,
        [ffi::ferrum_upstream_mmvq_dot_v1, selected_dot(format)],
        &[(256, 17), (512, 33), (768, 17)],
    );
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    converted: Vec<u32>,
    packed: Vec<u8>,
    output: Vec<u32>,
    result: Vec<u16>,
    rows: Vec<u32>,
    weight_flag: Vec<u32>,
}

/// Nonzero aligned view offsets and odd element strides expose accidental
/// use of allocation bases, logical N in place of stride, or padded N writes.
struct Case {
    format: UpstreamLinearFormat,
    plan: UpstreamLinearPlanV1,
    input_stride: usize,
    output_stride: usize,
    host_input: Vec<u16>,
    host_weights: Vec<u8>,
    input: Buffer,
    weights: Buffer,
    converted: Buffer,
    packed: Buffer,
    output: Buffer,
    result: Buffer,
    rows: Buffer,
    weight_flag: Buffer,
}

impl Case {
    fn new(
        format: UpstreamLinearFormat,
        k: usize,
        n: usize,
        input_kind: Input,
        bad_weight: bool,
    ) -> Self {
        Self::new_rows(format, 8, k, n, input_kind, bad_weight)
    }

    fn new_rows(
        format: UpstreamLinearFormat,
        m: usize,
        k: usize,
        n: usize,
        input_kind: Input,
        bad_weight: bool,
    ) -> Self {
        assert!(m >= 2);
        let p = plan(
            ALGORITHM,
            format,
            UpstreamLinearLayout::Columns,
            m as u32,
            k as u32,
            n as u32,
        );
        let input_stride = k + 3;
        let output_stride = n + 5;
        let mut x = vec![0x3555; m * input_stride];
        for row in 0..m {
            for col in 0..k {
                x[row * input_stride + col] = match input_kind {
                    Input::Zero => 0x8000,
                    Input::Subnormal => {
                        if col % 2 == 0 {
                            1
                        } else {
                            0x8001
                        }
                    }
                    Input::SumOverflow => 0x7bff,
                    _ => f16::from_f32(((col + row * 7) % 19) as f32 / 16.0 - 0.5625).to_bits(),
                };
            }
        }
        if matches!(input_kind, Input::Nonfinite) {
            x[0] = 0x7c00;
            x[input_stride] = 0xfc00;
            x[(m - 1) * input_stride] = 0x7e00;
        }
        let mut w = weights(format, k, p.padded_outputs as usize);
        let block_bytes = quant_block_bytes(format);
        let row_bytes = k / 256 * block_bytes;
        for (column, row) in w[..n * row_bytes].chunks_exact_mut(row_bytes).enumerate() {
            for (block_index, block) in row.chunks_exact_mut(block_bytes).enumerate() {
                block[..2].copy_from_slice(
                    &f16::from_f32(0.0625 + (column % 5) as f32 / 64.0)
                        .to_bits()
                        .to_le_bytes(),
                );
                let qs_offset = if matches!(format, UpstreamLinearFormat::Iq4Xs) {
                    // IQ4_XS: d[2], scales_h[2], scales_l[4], qs[128].
                    for (i, scale) in block[2..8].iter_mut().enumerate() {
                        *scale = ((column * 29 + block_index * 31 + i * 53) % 256) as u8;
                    }
                    8
                } else if matches!(format, UpstreamLinearFormat::Q5K) {
                    // Exercise both scale encodings and nonconstant fifth bits,
                    // including qh words with the sign bit set.
                    for (i, code) in block[4..48].iter_mut().enumerate() {
                        *code = ((column * 29 + block_index * 31 + i * 53) % 256) as u8;
                    }
                    48
                } else {
                    16
                };
                for (i, code) in block[qs_offset..].iter_mut().enumerate() {
                    *code = ((column * 13 + block_index * 7 + i * 17) % 256) as u8;
                }
            }
        }
        w[n * row_bytes..].fill(0);
        if bad_weight {
            w[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        }
        let input = Buffer::new(x.len() * 2 + 2);
        let weights = Buffer::new(w.len() + 4);
        unsafe {
            ok(cudaMemcpy(
                input.ptr.add(2),
                x.as_ptr().cast(),
                x.len() * 2,
                1,
            ));
            ok(cudaMemcpy(
                weights.ptr.add(4),
                w.as_ptr().cast(),
                w.len(),
                1,
            ));
        }
        Self {
            format,
            plan: p,
            input_stride,
            output_stride,
            host_input: x,
            host_weights: w,
            input,
            weights,
            converted: Buffer::new(p.converted_bytes as usize),
            packed: Buffer::new(p.packed_bytes as usize),
            output: Buffer::new(p.output_bytes as usize + 4),
            result: Buffer::new(m * output_stride * 2 + 2),
            rows: Buffer::new(m * 4),
            weight_flag: Buffer::new(4),
        }
    }

    fn run(&self, stream: &Stream, dot: MmvqDot) -> Snapshot {
        let p = &self.plan;
        unsafe {
            ok(cudaMemset(self.result.ptr, 0x35, self.result.bytes));
            ok(cudaMemset(self.packed.ptr, 0x35, self.packed.bytes));
            ok(ffi::ferrum_upstream_mmvq_check_weights_v2(
                p,
                self.weights.ptr.add(4),
                self.weight_flag.ptr,
                stream.0,
            ));
            ok(ffi::ferrum_upstream_mmvq_pack_v2(
                p,
                self.input.ptr.add(2),
                self.input_stride as u32,
                self.converted.ptr,
                self.packed.ptr,
                self.rows.ptr,
                stream.0,
            ));
            ok(dot(
                p,
                self.weights.ptr.add(4),
                self.packed.ptr,
                self.output.ptr.add(4),
                ptr::null_mut(),
                stream.0,
            ));
            ok(ffi::ferrum_upstream_mmvq_cast_v2(
                p,
                self.output.ptr.add(4),
                self.result.ptr.add(2),
                self.output_stride as u32,
                self.rows.ptr,
                self.weight_flag.ptr,
                stream.0,
            ));
        }
        stream.sync();
        let output = self.output.read::<u32>(self.output.bytes / 4);
        let result = self.result.read::<u16>(self.result.bytes / 2);
        assert_eq!(output[0], 0x35353535);
        assert_eq!(result[0], 0x3535);
        let input = self.input.read::<u16>(self.input.bytes / 2);
        assert_eq!(input[0], 0x3535);
        assert_eq!(&input[1..], self.host_input);
        let weights = self.weights.read::<u8>(self.weights.bytes);
        assert_eq!(&weights[..4], &[0x35; 4]);
        assert_eq!(&weights[4..], self.host_weights);
        for b in [
            &self.input,
            &self.weights,
            &self.converted,
            &self.packed,
            &self.output,
            &self.result,
            &self.rows,
            &self.weight_flag,
        ] {
            b.guards();
        }
        Snapshot {
            converted: self.converted.read::<u32>(self.converted.bytes / 4),
            packed: self.packed.read::<u8>(self.packed.bytes),
            output: output[1..].to_vec(),
            result: result[1..].to_vec(),
            rows: self.rows.read::<u32>(p.request.rows as usize),
            weight_flag: self.weight_flag.read::<u32>(1),
        }
    }

    fn validate(&self, snapshot: &Snapshot) {
        let p = &self.plan;
        let format = self.format;
        let block_bytes = quant_block_bytes(format);
        let (k, n) = (p.request.inputs as usize, p.request.outputs as usize);
        let bad_weight = self
            .host_weights
            .chunks_exact(block_bytes)
            .any(|b| oracle::classify_weight(ALGORITHM, format, b));
        assert_eq!(snapshot.weight_flag[0] != 0, bad_weight);
        for row in 0..p.request.rows as usize {
            let mut row_bad = false;
            for group in 0..p.padded_inputs as usize / 32 {
                let values = std::array::from_fn(|i| {
                    if group * 32 < k {
                        self.host_input[row * self.input_stride + group * 32 + i]
                    } else {
                        0
                    }
                });
                let (zero, bad) = oracle::marker_classify_pack(ALGORITHM, format, &values);
                row_bad |= bad;
                let b = row * (p.padded_inputs as usize / 32) + group;
                let metadata = &snapshot.packed[b * 36..][..4];
                let codes = &snapshot.packed[b * 36 + 4..][..32];
                if zero {
                    assert_eq!(metadata, &[0; 4]);
                }
                if zero || bad {
                    assert!(codes.iter().all(|&v| v == 0));
                }
                if bad {
                    assert_eq!(
                        u16::from_le_bytes(metadata[..2].try_into().unwrap()),
                        0x7e00
                    );
                }
            }
            assert_eq!(snapshot.rows[row] != 0, row_bad);
            for col in 0..n {
                let value = f32::from_bits(snapshot.output[row * n + col]);
                assert_eq!(
                    snapshot.result[row * self.output_stride + col],
                    oracle::final_cast_bits(value, row_bad, bad_weight)
                );
                if !row_bad && !bad_weight {
                    assert!(value.is_finite());
                    let (mut target, mut magnitude) = (0.0, 0.0);
                    for group in 0..k / 32 {
                        let b = row * (p.padded_inputs as usize / 32) + group;
                        let meta = &snapshot.packed[b * 36..][..4];
                        let codes = &snapshot.packed[b * 36 + 4..][..32];
                        let scale =
                            f16::from_bits(u16::from_le_bytes(meta[..2].try_into().unwrap()))
                                .to_f32();
                        let sum = f16::from_bits(u16::from_le_bytes(meta[2..].try_into().unwrap()))
                            .to_f32();
                        let offset = (col * (k / 256) + group / 8) * block_bytes;
                        let (v, mag) = oracle::declared_group(
                            ALGORITHM,
                            format,
                            &self.host_weights[offset..offset + block_bytes],
                            group % 8,
                            &std::array::from_fn(|i| codes[i] as i8),
                            scale,
                            sum,
                        );
                        target += v;
                        magnitude += mag;
                    }
                    let nu = (k as f64 / 32.0 + 32.0) * f32::EPSILON as f64;
                    let bound = nu / (1.0 - nu) * magnitude + 32.0 * f32::MIN_POSITIVE as f64;
                    assert!(
                        (f64::from(value) - target).abs() <= bound,
                        "actual-pack F64 row{row} col{col}: {value} vs {target}, bound{bound}"
                    );
                }
            }
            assert!(
                snapshot.result[row * self.output_stride + n..(row + 1) * self.output_stride]
                    .iter()
                    .all(|&v| v == 0x3535)
            );
        }
    }
}

fn quant_block_bytes(format: UpstreamLinearFormat) -> usize {
    match format {
        UpstreamLinearFormat::Iq4Xs => 136,
        UpstreamLinearFormat::Q4K => 144,
        UpstreamLinearFormat::Q5K => 176,
    }
}

/// Exercise mantissas in both half coefficients, all packed scale bytes, and
/// signed activations. Padded weight rows and input padding remain untouched.
fn overwrite_mixed_finite_scales(case: &mut Case, seed: u64) {
    let mixed = |mut value: u64| {
        value = value.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    };
    let (m, k, n) = (
        case.plan.request.rows as usize,
        case.plan.request.inputs as usize,
        case.plan.request.outputs as usize,
    );
    for row in 0..m {
        for col in 0..k {
            let bits = mixed(seed ^ ((row * k + col) as u64));
            case.host_input[row * case.input_stride + col] =
                0x2400 | (bits as u16 & 0x0fff) | ((bits >> 32) as u16 & 0x8000);
        }
    }
    let block_bytes = quant_block_bytes(case.format);
    let real_bytes = n * (k / 256) * block_bytes;
    for (index, block) in case.host_weights[..real_bytes]
        .chunks_exact_mut(block_bytes)
        .enumerate()
    {
        let bits = mixed(seed.wrapping_add(index as u64));
        // Nearby positive coefficients combine with signed dot terms and the
        // subtractive minimum term to exercise cancellation. All are finite
        // normal halves with nonzero mantissas, varying at every block.
        let d = 0x1400 | (bits as u16 & 0x0fff) | 1;
        let dmin = 0x1400 | ((bits >> 16) as u16 & 0x0fff) | 1;
        block[..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        for (offset, byte) in block[4..].iter_mut().enumerate() {
            *byte = mixed(bits.wrapping_add(offset as u64)) as u8;
        }
    }
    unsafe {
        ok(cudaMemcpy(
            case.input.ptr.add(2),
            case.host_input.as_ptr().cast(),
            case.host_input.len() * 2,
            1,
        ));
        ok(cudaMemcpy(
            case.weights.ptr.add(4),
            case.host_weights.as_ptr().cast(),
            case.host_weights.len(),
            1,
        ));
    }
}

#[test]
#[ignore = "requires the production Q4/Q5 shared M4 and M8 CUDA exports"]
fn m4_m8_shared_weight_decode_preserves_mixed_finite_scale_bits() {
    let _ = caps();
    let stream = Stream::new();
    let mut failures = Vec::new();
    let mut cases = 0;
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for m in [4, 8] {
            let dot = match (format, m) {
                (UpstreamLinearFormat::Q4K, 4) => {
                    ferrum_upstream_mmvq_q4_shared_weight_decode_m4_dot_v1 as MmvqDot
                }
                (UpstreamLinearFormat::Q5K, 4) => {
                    ferrum_upstream_mmvq_q5_shared_weight_decode_m4_dot_v1 as MmvqDot
                }
                (UpstreamLinearFormat::Q4K, 8) => DOTS[1],
                (UpstreamLinearFormat::Q5K, 8) => Q5_DOTS[1],
                _ => unreachable!(),
            };
            for (k, n) in [
                (5120, 10240),
                (6144, 5120),
                (5120, 17408),
                (17408, 5120),
                (512, 17),
            ] {
                for seed in [0, 1] {
                    let mut case = Case::new_rows(format, m, k, n, Input::Finite, false);
                    overwrite_mixed_finite_scales(&mut case, seed);
                    let reference = case.run(&stream, ffi::ferrum_upstream_mmvq_dot_v1);
                    let actual = case.run(&stream, dot);
                    let mut raw_mismatches = 0;
                    let mut max_abs = 0.0_f64;
                    let mut max_ulp = 0_u32;
                    let mut first = Vec::new();
                    let ordered = |bits: u32| {
                        if bits & 0x80000000 == 0 {
                            bits | 0x80000000
                        } else {
                            !bits
                        }
                    };
                    for (index, (&a, &b)) in reference.output.iter().zip(&actual.output).enumerate()
                    {
                        if a != b {
                            raw_mismatches += 1;
                            let abs =
                                (f64::from(f32::from_bits(a)) - f64::from(f32::from_bits(b))).abs();
                            let ulp = ordered(a).abs_diff(ordered(b));
                            max_abs = max_abs.max(abs);
                            max_ulp = max_ulp.max(ulp);
                            if first.len() < 6 {
                                first.push(serde_json::json!({"row":index/n,"column":index%n,"baseline_bits":a,"shared_bits":b,"absolute_error":abs,"ulp":ulp}));
                            }
                        }
                    }
                    let cast_mismatches = reference
                        .result
                        .iter()
                        .zip(&actual.result)
                        .filter(|(a, b)| a != b)
                        .count();
                    let same_pack_and_flags = reference.converted == actual.converted
                        && reference.packed == actual.packed
                        && reference.rows == actual.rows
                        && reference.weight_flag == actual.weight_flag;
                    let finite = reference
                        .output
                        .iter()
                        .chain(&actual.output)
                        .all(|&x| f32::from_bits(x).is_finite());
                    let flags_clear = reference
                        .rows
                        .iter()
                        .chain(&actual.rows)
                        .chain(&reference.weight_flag)
                        .chain(&actual.weight_flag)
                        .all(|&x| x == 0);
                    let mut cast_and_padding_errors = 0;
                    for snapshot in [&reference, &actual] {
                        for row in 0..m {
                            for column in 0..n {
                                cast_and_padding_errors += usize::from(
                                    snapshot.result[row * case.output_stride + column]
                                        != oracle::final_cast_bits(
                                            f32::from_bits(snapshot.output[row * n + column]),
                                            false,
                                            false,
                                        ),
                                );
                            }
                            cast_and_padding_errors += snapshot.result
                                [row * case.output_stride + n..(row + 1) * case.output_stride]
                                .iter()
                                .filter(|&&x| x != 0x3535)
                                .count();
                        }
                    }
                    let pass = raw_mismatches == 0
                        && cast_mismatches == 0
                        && same_pack_and_flags
                        && finite
                        && flags_clear
                        && cast_and_padding_errors == 0;
                    let report = serde_json::json!({"kind":"mixed_finite_scale_bits","format":format!("{format:?}"),"M":m,"K":k,"N":n,"seed":seed,
                        "raw_f32_mismatches":raw_mismatches,"cast_f16_mismatches":cast_mismatches,"max_absolute_error":max_abs,"max_ulp":max_ulp,
                        "same_actual_pack_and_flags":same_pack_and_flags,"finite":finite,"flags_clear":flags_clear,"cast_and_padding_errors":cast_and_padding_errors,"first":first,"pass":pass});
                    println!("{report}");
                    cases += 1;
                    if !pass {
                        failures.push(report);
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "mixed finite scale differential: {}/{} cases failed; first={}",
        failures.len(),
        cases,
        serde_json::Value::Array(failures.into_iter().take(4).collect())
    );
}

#[test]
#[ignore = "requires the source-built shared-weight-decode CUDA experiment"]
fn q4_shared_weight_decode_preserves_marker_bits_and_boundaries() {
    assert_marker_bits_and_boundaries(FORMAT, DOTS, &[(256, 17), (512, 33), (768, 17), (5120, 17)]);
}

#[test]
#[ignore = "requires the source-built Q5 shared-weight-decode CUDA experiment"]
fn q5_shared_weight_decode_preserves_marker_bits_and_boundaries() {
    assert_marker_bits_and_boundaries(
        UpstreamLinearFormat::Q5K,
        Q5_DOTS,
        &[(256, 17), (512, 33), (768, 17), (6144, 17), (5120, 17)],
    );
}

pub(super) fn assert_marker_bits_and_boundaries(
    format: UpstreamLinearFormat,
    dots: [MmvqDot; 2],
    shapes: &[(usize, usize)],
) {
    let _ = caps();
    let stream = Stream::new();
    for &(k, n) in shapes {
        for input in [
            Input::Finite,
            Input::Zero,
            Input::Subnormal,
            Input::Nonfinite,
            Input::SumOverflow,
        ] {
            for bad_weight in [false, true] {
                let case = Case::new(format, k, n, input, bad_weight);
                let reference = case.run(&stream, dots[0]);
                case.validate(&reference);
                for dot in [dots[1], dots[1], dots[0]] {
                    let actual = case.run(&stream, dot);
                    case.validate(&actual);
                    assert_eq!(
                        actual, reference,
                        "full buffers/repeat {format:?} {k}x{n} {input:?} bad_weight={bad_weight}"
                    );
                }
            }
        }
    }
    // Valid-but-ineligible plans and malformed inputs fail before a launch.
    let case = Case::new(format, 512, 17, Input::Finite, false);
    let invoke = |p: &UpstreamLinearPlanV1, weights, packed, output, fixup| unsafe {
        dots[1](p, weights, packed, output, fixup, stream.0)
    };
    let w = unsafe { case.weights.ptr.add(4) }.cast_const();
    let q = case.packed.ptr.cast_const();
    let o = unsafe { case.output.ptr.add(4) };
    let mut malformed = case.plan;
    malformed.nwarps += 1;
    assert_eq!(invoke(&malformed, w, q, o, ptr::null_mut()), -1);
    assert_eq!(invoke(&case.plan, w, q, o, o), -1);
    for p in [
        plan(ALGORITHM, format, UpstreamLinearLayout::Columns, 4, 512, 17),
        plan(
            ALGORITHM,
            format,
            UpstreamLinearLayout::Channels,
            8,
            512,
            17,
        ),
        plan(
            ALGORITHM,
            if matches!(format, UpstreamLinearFormat::Q4K) {
                UpstreamLinearFormat::Q5K
            } else {
                UpstreamLinearFormat::Q4K
            },
            UpstreamLinearLayout::Columns,
            8,
            512,
            17,
        ),
    ] {
        assert_eq!(invoke(&p, w, q, o, ptr::null_mut()), -2);
    }
    assert_eq!(invoke(&case.plan, ptr::null(), q, o, ptr::null_mut()), -5);
    assert_eq!(invoke(&case.plan, w, ptr::null(), o, ptr::null_mut()), -5);
    assert_eq!(
        invoke(&case.plan, w, q, ptr::null_mut(), ptr::null_mut()),
        -5
    );
    unsafe {
        assert_eq!(dots[1](ptr::null(), w, q, o, ptr::null_mut(), stream.0), -1);
        assert_eq!(invoke(&case.plan, w.add(1), q, o, ptr::null_mut()), -5);
        assert_eq!(invoke(&case.plan, w, q.add(1), o, ptr::null_mut()), -5);
        assert_eq!(invoke(&case.plan, w, q, o.add(1), ptr::null_mut()), -5);
    }
    stream.sync();
    assert!(case
        .output
        .read::<u8>(case.output.bytes)
        .iter()
        .all(|&byte| byte == 0x35));
    for buffer in [
        &case.input,
        &case.weights,
        &case.converted,
        &case.packed,
        &case.output,
        &case.result,
        &case.rows,
        &case.weight_flag,
    ] {
        buffer.guards();
    }
}

#[test]
#[ignore = "exclusive CUDA full-projection paired diagnostic; run correctness first"]
fn q4_shared_weight_decode_paired_working_sets() {
    paired_working_set(FORMAT, DOTS, 5120, 10240);
}

#[test]
#[ignore = "exclusive CUDA full-projection Q5 paired diagnostic; run correctness first"]
fn q5_shared_weight_decode_paired_working_sets() {
    // Saved prepared-leaf metadata: GDN/causal output K6144,N5120 (44 leaves),
    // and GDN qkv K5120,N10240 (12 leaves). These are physical leaf dimensions.
    for (k, n) in [(6144, 5120), (5120, 10240)] {
        paired_working_set(UpstreamLinearFormat::Q5K, Q5_DOTS, k, n);
    }
}

fn paired_working_set(format: UpstreamLinearFormat, dots: [MmvqDot; 2], k: usize, n: usize) {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const WARM: usize = 8;
    const PAIRS: usize = 16;
    let m = 8;
    let host_weights = weights(format, k, n);
    let weight_bytes = host_weights.len();
    let experiment = match format {
        UpstreamLinearFormat::Q4K => "q4_shared_weight_decode",
        UpstreamLinearFormat::Q5K => "q5_shared_weight_decode",
        _ => unreachable!("shared-weight fixture format"),
    };
    let format_label = format!("{format:?}");
    println!(
        "{}",
        serde_json::json!({"experiment":experiment, "kind":"plan",
        "M":m,"K":k,"N":n,"format":format_label,"warm_rounds":WARM,"pairs":PAIRS,
        "order":"alternating AB/BA pairs","modes":["resident","arena_offset_ring"],
        "timed":"same MarkerV2 F16 convert/row checks/pack + selected dot + MarkerV2 cast",
        "excluded":"allocation, input upload, static plan/weight check, graph capture, oracle/readback",
        "scope":"synthetic same physical weights; fixed input; no provider/driver conditional checks or model throughput claim"})
    );
    for mode in ["resident", "arena_offset_ring"] {
        let count = if mode == "resident" {
            1
        } else {
            3 * l2 as usize / weight_bytes + 1
        };
        let stride = weight_bytes.div_ceil(256) * 256;
        let allocation = Buffer::new(count * stride);
        let addresses = (0..count)
            .map(|i| unsafe { allocation.ptr.add(i * stride) })
            .collect::<Vec<_>>();
        for &address in &addresses {
            unsafe {
                ok(cudaMemcpy(
                    address,
                    host_weights.as_ptr().cast(),
                    weight_bytes,
                    1,
                ));
            }
        }
        let input = Buffer::new(m * k * 2);
        input.write(
            &(0..m * k)
                .map(|i| f16::from_f32(((i * 17) % 101) as f32 / 2048.0 - 0.024).to_bits())
                .collect::<Vec<_>>(),
        );
        let stream = Stream::new();
        let routes =
            dots.map(|_| Route::new(ALGORITHM, format, m as u32, k as u32, n as u32, count));
        let mut stable = Vec::new();
        let mut raw = Vec::new();
        for (arm, route) in routes.iter().enumerate() {
            for (index, &address) in addresses.iter().enumerate() {
                route.scan(&stream, address, index);
                route.enqueue_with_mmvq_dot(&stream, &input, address, index, dots[arm]);
            }
            stream.sync();
            stable.push(route.validate(format, &host_weights, count));
            raw.push(route.output.read::<u32>(m * n));
        }
        assert_eq!(stable[0], stable[1]);
        assert_eq!(raw[0], raw[1]);
        assert_eq!(
            routes[0].packed.read::<u8>(routes[0].packed.bytes),
            routes[1].packed.read::<u8>(routes[1].packed.bytes)
        );
        let laps = 32_usize.div_ceil(count);
        let graphs = [0, 1].map(|arm| {
            Graph::capture(&stream, || {
                for _ in 0..laps {
                    for (index, &address) in addresses.iter().enumerate() {
                        routes[arm]
                            .enqueue_with_mmvq_dot(&stream, &input, address, index, dots[arm]);
                    }
                }
            })
        });
        let (start, end) = (Event::new(), Event::new());
        let mut samples = Vec::with_capacity(PAIRS * 2);
        for round in 0..WARM + PAIRS {
            for order in 0..2 {
                let arm = (round + order) % 2;
                let wall = Instant::now();
                unsafe {
                    ok(cudaEventRecord(start.0, stream.0));
                }
                graphs[arm].launch(&stream);
                unsafe {
                    ok(cudaEventRecord(end.0, stream.0));
                    ok(cudaEventSynchronize(end.0));
                }
                let wall_ns = wall.elapsed().as_nanos();
                let mut ms = 0.0;
                unsafe {
                    ok(cudaEventElapsedTime(&mut ms, start.0, end.0));
                }
                if round >= WARM {
                    samples.push((
                        round - WARM,
                        order,
                        arm,
                        f64::from(ms) * 1_000_000.0,
                        wall_ns,
                    ));
                }
            }
        }
        stream.sync();
        for (arm, route) in routes.iter().enumerate() {
            assert_eq!(route.validate(format, &host_weights, count), stable[arm]);
            assert_eq!(route.output.read::<u32>(m * n), raw[arm]);
        }
        input.guards();
        allocation.guards();
        for (pair, order, arm, gpu_ns, wall_ns) in samples {
            println!(
                "{}",
                serde_json::json!({"experiment":experiment, "kind":"sample",
                "M":m,"K":k,"N":n,"format":format_label,"weight_mode":mode,
                "weight_bytes":weight_bytes,"matrix_count":count,"working_set_bytes":count*weight_bytes,
                "device_l2_bytes":l2,"matrix_contents":"same deterministic bytes; different ring addresses",
                "laps":laps,"projections":laps*count,"warm_rounds":WARM,"pairs":PAIRS,
                "same_physical_weights":true,"between_timed_routes":"events/graph submission only; no reset/readback",
                "pair":pair,"order":order,"arm":LABELS[arm],"gpu_ns":gpu_ns,"wall_ns":wall_ns})
            );
        }
        // Captured graphs relinquish device addresses before their allocations.
        drop(graphs);
    }
}

#[cfg(feature = "cuda-upstream-small-shared-weight-decode-tests")]
pub(super) fn assert_small_shared_markers(format: UpstreamLinearFormat, m: usize, dot: MmvqDot) {
    let _ = caps();
    let stream = Stream::new();
    for (k, n) in [(256, 17), (768, 33)] {
        for input in [
            Input::Finite,
            Input::Zero,
            Input::Subnormal,
            Input::Nonfinite,
            Input::SumOverflow,
        ] {
            for bad in [false, true] {
                let case = Case::new_rows(format, m, k, n, input, bad);
                let expected = case.run(&stream, ffi::ferrum_upstream_mmvq_dot_v1);
                case.validate(&expected);
                for _ in 0..2 {
                    let actual = case.run(&stream, dot);
                    case.validate(&actual);
                    assert_eq!(
                        actual, expected,
                        "M{m} {format:?} {k}x{n} {input:?} bad_weight={bad}"
                    );
                }
            }
        }
    }
    let case = Case::new_rows(format, m, 512, 17, Input::Finite, false);
    let w = unsafe { case.weights.ptr.add(4) }.cast_const();
    let q = case.packed.ptr.cast_const();
    let o = unsafe { case.output.ptr.add(4) };
    let invoke =
        |p: &UpstreamLinearPlanV1, w, q, o, fixup| unsafe { dot(p, w, q, o, fixup, stream.0) };
    let mut malformed = case.plan;
    malformed.nwarps += 1;
    assert_eq!(invoke(&malformed, w, q, o, ptr::null_mut()), -1);
    assert_eq!(invoke(&case.plan, w, q, o, o), -1);
    for p in [
        plan(ALGORITHM, format, UpstreamLinearLayout::Columns, 1, 512, 17),
        plan(
            ALGORITHM,
            format,
            UpstreamLinearLayout::Channels,
            4,
            512,
            17,
        ),
        plan(
            ALGORITHM,
            UpstreamLinearFormat::Iq4Xs,
            UpstreamLinearLayout::Columns,
            4,
            512,
            17,
        ),
    ] {
        assert_eq!(invoke(&p, w, q, o, ptr::null_mut()), -2);
    }
    assert_eq!(invoke(&case.plan, ptr::null(), q, o, ptr::null_mut()), -5);
    assert_eq!(invoke(&case.plan, w, ptr::null(), o, ptr::null_mut()), -5);
    assert_eq!(
        invoke(&case.plan, w, q, ptr::null_mut(), ptr::null_mut()),
        -5
    );
    assert_eq!(
        invoke(&case.plan, unsafe { w.add(1) }, q, o, ptr::null_mut()),
        -5
    );
}
