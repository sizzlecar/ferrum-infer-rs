use super::*;
fn median(mut x: Vec<f64>) -> f64 {
    x.sort_by(f64::total_cmp);
    (x[(x.len() - 1) / 2] + x[x.len() / 2]) * 0.5
}
fn measure(s: &Arc<CudaStream>, g: &CudaGraph) -> (f64, f64) {
    s.synchronize().unwrap();
    let start = std::time::Instant::now();
    let a = s
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..8 {
        g.launch().unwrap();
    }
    let b = s
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    b.synchronize().unwrap();
    (
        f64::from(a.elapsed_ms(&b).unwrap()) * 1e6 / 8.0,
        start.elapsed().as_secs_f64() * 1e9 / 8.0,
    )
}

#[test]
fn residual2_scalar_quantization_bounds_cover_zero_ties_and_finite_dynamic_range() {
    for rows in [1, 5, 8] {
        let mut x = fixture::inputs(rows, 256, 1);
        let v = &mut x[PAD..PAD + rows * 256];
        v[..32].fill(f16::NEG_ZERO);
        for (i, z) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
            .into_iter()
            .enumerate()
        {
            v[32 + i] = f16::from_f32(z);
        }
        fixture::pack_oracle(v, rows, 256, 1);
        fixture::pack_oracle(v, rows, 256, 2);
    }
}
#[test]
#[ignore = "requires actual exclusive CUDA; independent residual quantization oracle"]
fn residual2_pack_and_complete_ffn_qualification_on_cuda() {
    let ctx = CudaContext::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    let c1 = Candidate::load(&ctx, 1);
    let c2 = Candidate::load(&ctx, 2);
    let k = CudaNativeBlockKernels::load(&ctx).unwrap();
    let maxc = if c1.cta_budget > c2.cta_budget {
        &c1
    } else {
        &c2
    };
    // Nonfinite groups are explicitly propagated, and complete pack buffers are guarded.
    let mut x = vec![f16::NEG_ZERO; 256];
    for (i, z) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
        .into_iter()
        .enumerate()
    {
        x[32 + i] = f16::from_f32(z);
    }
    x[64] = f16::NAN;
    x[96] = f16::INFINITY;
    x[128] = f16::NEG_INFINITY;
    let dx = s.clone_htod(&x).unwrap();
    let mut q = s
        .clone_htod(&vec![0xdeadbeefu32; 2 * PAD + 2 * 256 / 4])
        .unwrap();
    let mut d = s
        .clone_htod(&vec![SENTINEL; 2 * PAD + 2 * 256 / 32])
        .unwrap();
    let mut sum = s
        .clone_htod(&vec![i32::MIN; 2 * PAD + 2 * 256 / 32])
        .unwrap();
    {
        let mut qv = q.slice_mut(PAD..q.len() - PAD);
        let mut dv = d.slice_mut(PAD..d.len() - PAD);
        let mut sv = sum.slice_mut(PAD..sum.len() - PAD);
        let (xp, xg) = dx.device_ptr(&s);
        let (qp, qg) = qv.device_ptr_mut(&s);
        let (dp, dg) = dv.device_ptr_mut(&s);
        let (sp, sg) = sv.device_ptr_mut(&s);
        c2.pack(&s, xp, qp, dp, sp, 1, 256);
        drop((xg, qg, dg, sg));
    }
    fixture::check_pack(&s, &x, 1, 256, 2, &q, &d, &sum);
    for rows in [1, 4, 5, 7, 8, 9] {
        for down in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
            let mut f = Case::new(&s, maxc, rows, 512, 768, down, 7);
            // Qualify odd-offset weight views; performance cases use ordinary alignment.
            f.gate = Matrix::new_with_base(&s, GgufBlockFormat::Q4K, 512, 768, 7, 5);
            f.up = Matrix::new_with_base(&s, GgufBlockFormat::Q4K, 512, 768, 8, 5);
            f.down = Matrix::new_with_base(&s, down, 768, 512, 9, 5);
            let gs = [
                f.capture(&s, &k, &c1, false),
                f.capture(&s, &k, &c1, true),
                f.capture(&s, &k, &c2, true),
            ];
            for generation in 0..2 {
                f.update(&s, generation);
                let mut expected = Vec::new();
                for (arm, g) in gs.iter().enumerate() {
                    f.reset(&s);
                    g.launch().unwrap();
                    expected.push(f.validate(&s, arm, true));
                }
                for (arm, g) in gs.iter().enumerate() {
                    f.reset(&s);
                    g.launch().unwrap();
                    assert_eq!(
                        f.validate(&s, arm, false),
                        expected[arm],
                        "changed-input replay"
                    );
                }
            }
        }
    }
}
#[test]
#[ignore = "exclusive CUDA; all three arms include complete FFN pack/project/fixup/SiLU/down"]
fn residual2_three_arm_complete_ffn_screen_on_cuda() {
    let ctx = CudaContext::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    let c1 = Candidate::load(&ctx, 1);
    let c2 = Candidate::load(&ctx, 2);
    let k = CudaNativeBlockKernels::load(&ctx).unwrap();
    let maxc = if c1.cta_budget > c2.cta_budget {
        &c1
    } else {
        &c2
    };
    let mut decisions = Vec::new();
    for down in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let mut cases: Vec<_> = [0, 7, 14, 21]
            .into_iter()
            .map(|salt| Case::new(&s, maxc, 8, 4096, 12288, down, salt))
            .collect();
        let gs: Vec<_> = cases
            .iter_mut()
            .map(|f| {
                [
                    f.capture(&s, &k, &c1, false),
                    f.capture(&s, &k, &c1, true),
                    f.capture(&s, &k, &c2, true),
                ]
            })
            .collect();
        let mut expected = Vec::new();
        for (f, graphs) in cases.iter_mut().zip(&gs) {
            let mut generations = Vec::new();
            for generation in 0..2 {
                f.update(&s, generation);
                let mut arms = Vec::new();
                for (arm, g) in graphs.iter().enumerate() {
                    f.reset(&s);
                    g.launch().unwrap();
                    arms.push(f.validate(&s, arm, true));
                }
                generations.push(arms);
            }
            expected.push(generations);
        }
        let mut strict_ratios = Vec::new();
        let mut single_ratios = Vec::new();
        for round in 0..8 {
            let generation = round % 2;
            for workset in 0..4 {
                let f = &mut cases[workset];
                f.update(&s, generation);
                let order = if (round + workset) % 2 == 0 {
                    [0, 1, 2]
                } else {
                    [2, 1, 0]
                };
                let mut gpu = [0.0; 3];
                let mut host = [0.0; 3];
                for arm in order {
                    f.reset(&s);
                    (gpu[arm], host[arm]) = measure(&s, &gs[workset][arm]);
                    assert_eq!(
                        f.validate(&s, arm, false),
                        expected[workset][generation][arm],
                        "replay after timing"
                    );
                }
                println!(
                    "{}",
                    serde_json::json!({"event":"residual2_paired_sample","rows":8,"down":format!("{down:?}"),"round":round,"workset":workset,"generation":generation,"warmup":round<2,"replays":8,"order":order,"gpu_ns":gpu,"host_ns":host,"dual_vs_strict":gpu[2]/gpu[0],"dual_vs_single":gpu[2]/gpu[1]})
                );
                if round >= 2 {
                    strict_ratios.push(gpu[2] / gpu[0]);
                    single_ratios.push(gpu[2] / gpu[1]);
                }
            }
        }
        let sr = median(strict_ratios);
        let qr = median(single_ratios);
        decisions.push(sr <= 0.70);
        println!(
            "{}",
            serde_json::json!({"event":"residual2_cell_summary","rows":8,"down":format!("{down:?}"),"pairs":24,"dual_vs_strict_paired_median":sr,"dual_vs_single_paired_median":qr,"original_30_percent_gate":sr<=0.70,"model_quality_passed":null,"production_enabled":false})
        );
    }
    println!(
        "{}",
        serde_json::json!({"event":"residual2_summary","all_performance_gates_passed":decisions.iter().all(|x|*x),"model_quality_passed":null,"release_approved":false})
    );
}

#[test]
#[ignore = "actual CUDA; partial N128/M8 tiles and strided destination"]
fn residual2_projection_tail_and_stride_guards_on_cuda() {
    let ctx = CudaContext::new(0).unwrap();
    let s = ctx.new_stream().unwrap();
    let c = Candidate::load(&ctx, 2);
    let (rows, inputs, outputs, stride, offset) = (5, 768, 131, 139, 3);
    let input = fixture::inputs(rows, inputs, 4);
    let dx = s.clone_htod(&input).unwrap();
    let w = Matrix::new_with_base(&s, GgufBlockFormat::Q4K, inputs, outputs, 17, 5);
    let mut q = s
        .clone_htod(&vec![0xdeadbeefu32; 2 * PAD + 2 * rows * inputs / 4])
        .unwrap();
    let mut d = s
        .clone_htod(&vec![SENTINEL; 2 * PAD + 2 * rows * inputs / 32])
        .unwrap();
    let mut sum = s
        .clone_htod(&vec![i32::MIN; 2 * PAD + 2 * rows * inputs / 32])
        .unwrap();
    let mut partial = s
        .clone_htod(&vec![
            SENTINEL;
            2 * PAD
                + scratch_len(rows, inputs, outputs, c.cta_budget)
        ])
        .unwrap();
    let mut y = s
        .clone_htod(&vec![f16::from_f32(SENTINEL); 2 * PAD + rows * stride])
        .unwrap();
    {
        let xv = dx.slice(PAD..dx.len() - PAD);
        let wv = w.gpu.slice(w.base..w.raw.len() - 7);
        let mut qv = q.slice_mut(PAD..q.len() - PAD);
        let mut dv = d.slice_mut(PAD..d.len() - PAD);
        let mut sv = sum.slice_mut(PAD..sum.len() - PAD);
        let mut pv = partial.slice_mut(PAD..partial.len() - PAD);
        let mut yv = y.slice_mut(PAD..y.len() - PAD);
        let (xp, xg) = xv.device_ptr(&s);
        let (wp, wg) = wv.device_ptr(&s);
        let (qp, qg) = qv.device_ptr_mut(&s);
        let (dp, dg) = dv.device_ptr_mut(&s);
        let (sp, sg) = sv.device_ptr_mut(&s);
        let (pp, pg) = pv.device_ptr_mut(&s);
        let (yp, yg) = yv.device_ptr_mut(&s);
        c.pack(&s, xp, qp, dp, sp, rows, inputs);
        c.project(
            &s, qp, dp, sp, wp, yp, pp, rows, inputs, outputs, stride, offset,
        );
        drop((xg, wg, qg, dg, sg, pg, yg));
    }
    fixture::check_pack(
        &s,
        &input[PAD..input.len() - PAD],
        rows,
        inputs,
        2,
        &q,
        &d,
        &sum,
    );
    let result = s.clone_dtoh(&y).unwrap();
    fixture::check_projection(
        &input[PAD..input.len() - PAD],
        &result[PAD..result.len() - PAD],
        rows,
        &w,
        stride,
        offset,
        2,
    );
    for (i, v) in result.iter().enumerate() {
        let used = i
            .checked_sub(PAD)
            .filter(|&n| n < rows * stride && (offset..offset + outputs).contains(&(n % stride)));
        if used.is_none() {
            assert_eq!(v.to_f32(), SENTINEL);
        }
    }
    let p = s.clone_dtoh(&partial).unwrap();
    assert!(p[..PAD]
        .iter()
        .chain(&p[p.len() - PAD..])
        .all(|x| *x == SENTINEL));
    assert_eq!(s.clone_dtoh(&dx).unwrap(), input);
    w.check_immutable(&s);
}
