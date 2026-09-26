use super::*;
use crate::gguf_blocks::q4k_q8_reference::pack_rows;
pub(super) fn check_pack(
    s: &Arc<CudaStream>,
    x: &[f16],
    rows: usize,
    k: usize,
    q: &CudaSlice<u32>,
    d: &CudaSlice<f32>,
    sum: &CudaSlice<i32>,
) {
    let q = s.clone_dtoh(q).unwrap();
    let d = s.clone_dtoh(d).unwrap();
    let sum = s.clone_dtoh(sum).unwrap();
    assert!(q[..PAD]
        .iter()
        .chain(&q[q.len() - PAD..])
        .all(|v| *v == 0xdeadbeef));
    assert!(d[..PAD]
        .iter()
        .chain(&d[d.len() - PAD..])
        .all(|v| *v == SENTINEL));
    assert!(sum[..PAD]
        .iter()
        .chain(&sum[sum.len() - PAD..])
        .all(|v| *v == i32::MIN));
    check_pack_values(
        x,
        rows,
        k,
        &q[PAD..q.len() - PAD],
        &d[PAD..d.len() - PAD],
        &sum[PAD..sum.len() - PAD],
    );
}
pub(super) fn check_pack_values(
    x: &[f16],
    rows: usize,
    k: usize,
    q: &[u32],
    d: &[f32],
    sum: &[i32],
) {
    assert_eq!(q.len(), rows * k / 4);
    assert_eq!(d.len(), rows * k / 32);
    assert_eq!(sum.len(), d.len());
    let expected = pack_rows(x, rows, k);
    for row in 0..rows {
        for group in 0..k / 32 {
            let i = row * (k / 32) + group;
            let pg = ((group / 8) * rows + row) * 8 + group % 8;
            if expected.scales[i].is_nan() {
                assert!(d[pg].is_nan());
            } else {
                assert_eq!(
                    d[pg].to_bits(),
                    expected.scales[i].to_bits(),
                    "scale {row}/{group}"
                );
            }
            assert_eq!(
                sum[pg],
                expected.quants[i * 32..i * 32 + 32]
                    .iter()
                    .map(|v| i32::from(*v))
                    .sum::<i32>(),
                "integer sum"
            );
            for word in 0..8 {
                let bytes = std::array::from_fn(|j| expected.quants[i * 32 + word * 4 + j] as u8);
                assert_eq!(
                    q[pg * 8 + word],
                    u32::from_le_bytes(bytes),
                    "q {row}/{group}/{word}"
                );
            }
        }
    }
}
fn pack_fixture(
    s: &Arc<CudaStream>,
    c: &Candidate,
    x: &[f16],
    rows: usize,
    k: usize,
) -> (CudaSlice<u32>, CudaSlice<f32>, CudaSlice<i32>) {
    let dx = s.clone_htod(x).unwrap();
    let mut q = s
        .clone_htod(&vec![0xdeadbeefu32; PAD + rows * k / 4 + PAD])
        .unwrap();
    let mut d = s
        .clone_htod(&vec![SENTINEL; PAD + rows * k / 32 + PAD])
        .unwrap();
    let mut sums = s
        .clone_htod(&vec![i32::MIN; PAD + rows * k / 32 + PAD])
        .unwrap();
    {
        let mut qv = q.slice_mut(PAD..q.len() - PAD);
        let mut dv = d.slice_mut(PAD..d.len() - PAD);
        let mut sv = sums.slice_mut(PAD..sums.len() - PAD);
        let (xp, xg) = dx.device_ptr(s);
        let (qp, qg) = qv.device_ptr_mut(s);
        let (dp, dg) = dv.device_ptr_mut(s);
        let (sp, sg) = sv.device_ptr_mut(s);
        c.pack(s, xp, qp, dp, sp, rows, k);
        drop((xg, qg, dg, sg));
    }
    s.synchronize().unwrap();
    check_pack(s, x, rows, k, &q, &d, &sums);
    assert_eq!(
        s.clone_dtoh(&dx)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    (q, d, sums)
}
#[test]
#[ignore = "requires actual CUDA device"]
fn q4_stream_mmq_pack_metadata_tails_and_full_ffn_conformance_on_cuda() {
    let ctx = CudaContext::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    let c = Candidate::load(&ctx);
    let k = CudaNativeBlockKernels::load(&ctx).unwrap();
    let mut special = vec![f16::NEG_ZERO; 256];
    for (i, v) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
        .into_iter()
        .enumerate()
    {
        special[32 + i] = f16::from_f32(v);
    }
    special[64] = f16::from_bits(1);
    special[65] = -f16::from_bits(1);
    special[96] = f16::MAX;
    special[97] = f16::MIN;
    special[128] = f16::NAN;
    special[160] = f16::INFINITY;
    special[192] = f16::ONE;
    pack_fixture(&s, &c, &special, 1, 256);
    for (rows, inputs, outputs) in [
        (1usize, 256usize, 1usize),
        (4, 512, 65),
        (8, 768, 129),
        (9, 256, 67),
    ] {
        c.report_shape(rows, inputs, outputs);
        let w = Matrix::new_with_base(
            &s,
            GgufBlockFormat::Q4K,
            inputs,
            outputs,
            3,
            if rows % 2 == 0 { 32 } else { 5 },
        );
        let raw = w.gpu.slice(w.base..w.raw.len() - 7);
        let groups = outputs * inputs / 32;
        let mut cq = s
            .clone_htod(&vec![0xdeadbeefu32; PAD + groups * 8 + PAD])
            .unwrap();
        let mut ca = s.clone_htod(&vec![SENTINEL; PAD + groups + PAD]).unwrap();
        let mut cb = s.clone_htod(&vec![SENTINEL; PAD + groups + PAD]).unwrap();
        {
            let mut qv = cq.slice_mut(PAD..cq.len() - PAD);
            let mut av = ca.slice_mut(PAD..ca.len() - PAD);
            let mut bv = cb.slice_mut(PAD..cb.len() - PAD);
            let count = groups as u32;
            unsafe {
                s.launch_builder(&c.metadata)
                    .arg(&raw)
                    .arg(&mut qv)
                    .arg(&mut av)
                    .arg(&mut bv)
                    .arg(&count)
                    .launch(LaunchConfig::for_num_elems(count))
            }
            .unwrap();
        }
        let codes = s.clone_dtoh(&cq).unwrap();
        let aa = s.clone_dtoh(&ca).unwrap();
        let bb = s.clone_dtoh(&cb).unwrap();
        for group in 0..groups {
            let b = fixture::decode_q4(&w.raw[w.base + (group / 8) * 144..]);
            let g = group % 8;
            assert_eq!(
                aa[PAD + group].to_bits(),
                (b.d.to_f32() * f32::from(b.scales[g])).to_bits()
            );
            assert_eq!(
                bb[PAD + group].to_bits(),
                (b.dmin.to_f32() * f32::from(b.minima[g])).to_bits()
            );
            for word in 0..8 {
                assert_eq!(
                    codes[PAD + group * 8 + word],
                    u32::from_le_bytes(std::array::from_fn(|j| b.quants[g * 32 + word * 4 + j]))
                );
            }
        }
        assert!(codes[..PAD]
            .iter()
            .chain(&codes[codes.len() - PAD..])
            .all(|v| *v == 0xdeadbeef));
        for v in [&aa, &bb] {
            assert!(v[..PAD]
                .iter()
                .chain(&v[v.len() - PAD..])
                .all(|v| *v == SENTINEL));
        }
        let x = fixture::inputs(rows, inputs, 0);
        let (q, d, sum) = pack_fixture(&s, &c, &x[PAD..x.len() - PAD], rows, inputs);
        let stride = outputs + 11;
        let mut y = s
            .clone_htod(&vec![f16::from_f32(SENTINEL); PAD + rows * stride + PAD])
            .unwrap();
        let mut partial = s
            .clone_htod(&vec![
                SENTINEL;
                PAD + scratch_len(rows, inputs, outputs, c.cta_budget)
                    + PAD
            ])
            .unwrap();
        {
            let qv = q.slice(PAD..q.len() - PAD);
            let dv = d.slice(PAD..d.len() - PAD);
            let sv = sum.slice(PAD..sum.len() - PAD);
            let mut yv = y.slice_mut(PAD..y.len() - PAD);
            let mut pv = partial.slice_mut(PAD..partial.len() - PAD);
            let (qp, qg) = qv.device_ptr(&s);
            let (dp, dg) = dv.device_ptr(&s);
            let (sp, sg) = sv.device_ptr(&s);
            let (wp, wg) = raw.device_ptr(&s);
            let (yp, yg) = yv.device_ptr_mut(&s);
            let (pp, pg) = pv.device_ptr_mut(&s);
            c.project(&s, qp, dp, sp, wp, yp, pp, rows, inputs, outputs, stride, 3);
            drop((qg, dg, sg, wg, yg, pg));
        }
        let y = s.clone_dtoh(&y).unwrap();
        fixture::check_projection(
            &x[PAD..x.len() - PAD],
            &y[PAD..y.len() - PAD],
            rows,
            &w,
            stride,
            3,
            true,
        );
        for (i, v) in y.iter().enumerate() {
            let active = i
                .checked_sub(PAD)
                .filter(|j| *j < rows * stride)
                .is_some_and(|j| (3..3 + outputs).contains(&(j % stride)));
            if !active {
                assert_eq!(v.to_f32(), SENTINEL);
            }
        }
        let partial = s.clone_dtoh(&partial).unwrap();
        assert!(partial[..PAD]
            .iter()
            .chain(&partial[partial.len() - PAD..])
            .all(|v| *v == SENTINEL));
        w.check_immutable(&s);
    }
    for rows in [4, 8] {
        for down in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
            let mut f = Case::new(&s, &c, rows, 512, 768, down, 0);
            let a = f.capture(&s, &k, &c, false);
            let b = f.capture(&s, &k, &c, true);
            for generation in 0..2 {
                f.update(&s, generation);
                for (candidate, g) in [(false, &a), (true, &b)] {
                    f.reset(&s);
                    g.launch().unwrap();
                    f.validate(&s, candidate, true);
                }
            }
        }
    }
}
