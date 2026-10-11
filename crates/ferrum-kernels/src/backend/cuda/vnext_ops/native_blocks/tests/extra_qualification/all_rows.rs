//! Qualify the existing production MMQ primitive at every small physical width.
//! This does not widen MMVQ or change an old profile's selected arithmetic.
use super::*;

fn checked_case(s: &Arc<CudaStream>, f: GgufBlockFormat, m: usize, k: usize, n: usize) -> Case {
    let c = Case::new(s, f, m, k, n, true, 0, &[1]);
    let r = &c.routes[0];
    let mut spans = [
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
    .filter(|x| x.bytes != 0)
    .map(|x| (x.address, x.address.checked_add(x.bytes).unwrap()))
    .collect::<Vec<_>>();
    spans.sort_unstable();
    assert!(spans.windows(2).all(|p| p[0].1 <= p[1].0));
    assert_eq!(r.p.algorithm, Algorithm::Mmq as u32);
    assert_eq!(r.p.pack_abi, 1);
    assert_eq!(r.native.row_poison_bytes(), m as u64 * 4);
    println!(
        "{}",
        serde_json::json!({
            "kind":"extra_all_rows_plan", "format":format!("{f:?}"),
            "m":m,"k":k,"n":n,"j":r.p.j,"i":r.p.i,
            "fixup":r.p.fixup,"guard_blocks":r.p.guard_blocks,
            "weight_bytes":r.p.weight_bytes,"packed_bytes":r.p.packed_bytes,
            "fixup_bytes":r.p.fixup_bytes,"operator":r.native.operator()
        })
    );
    c
}

fn repeated(s: &Arc<CudaStream>, c: &mut Case) {
    let mut previous = None;
    for _ in 0..2 {
        c.reset(s);
        c.native(s, 0);
        let bits = c.validate(s, Some(0));
        if let Some(old) = previous {
            assert_eq!(bits, old, "same geometry and arithmetic repeated bits");
        }
        previous = Some(bits);
    }
}

#[test]
#[ignore = "requires locked production extra archive and exclusive CUDA; no performance assertion"]
fn extra_production_mmq_all_small_rows_match_actual_pack_oracle() {
    let (ctx, s, _) = context();
    for f in FORMATS {
        let mut fixup_seen = [false; 2];
        for m in 1..=32 {
            // Full M*N oracle, noncontiguous input/output, K512 pack tail,
            // partial output tile and partial J tile at every applicable width.
            let mut c = checked_case(&s, f, m, 768, 17);
            fixup_seen[c.routes[0].p.fixup as usize] = true;
            repeated(&s, &mut c);
            let ordinary = c.x.clone();
            for row in 0..m {
                c.x[row * c.xstride..row * c.xstride + c.k].fill(f16::from_bits(0x8000));
            }
            c.upload(&s);
            repeated(&s, &mut c); // poisoned pack allocation must become +0
            for row in 0..m {
                for col in 0..c.k {
                    c.x[row * c.xstride + col] = f16::from_bits(1 + ((row + col) % 1023) as u16);
                }
            }
            c.upload(&s);
            repeated(&s, &mut c); // actual finite subnormal pack
            c.x = ordinary.clone();
            for row in [0, m / 2, m - 1] {
                c.x[row * c.xstride] = f16::NAN;
            }
            c.upload(&s);
            repeated(&s, &mut c); // canonical row marker incl. partial last row
            c.x = ordinary;
            let scale_offset = if f == GgufBlockFormat::Q3K { 108 } else { 0 };
            c.w[scale_offset..scale_offset + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
            c.upload(&s);
            repeated(&s, &mut c); // whole-leaf flag distinct from row poison
        }
        // Actual device-derived whole wave covers the no-fixup route without
        // assuming a particular SM count. N>49 uses the existing 9-point oracle.
        let sm = request(&ctx, f, 31, 256, 17).sm_count;
        let mut whole = checked_case(&s, f, 31, 256, sm.checked_mul(128).unwrap() as usize);
        fixup_seen[whole.routes[0].p.fixup as usize] = true;
        repeated(&s, &mut whole);
        assert!(fixup_seen[0]);
        assert!(fixup_seen[1] || sm == 1);
        println!(
            "{}",
            serde_json::json!({
                "kind":"extra_all_rows_fixup_coverage", "format":format!("{f:?}"),
                "multiprocessors":sm,"no_fixup_executed":fixup_seen[0],
                "fixup_executed":fixup_seen[1],"single_sm_fixup_impossible":sm==1
            })
        );
        drop(whole);
        for m in [2, 3, 5, 7, 9, 15, 17, 31] {
            for (k, n) in [(5120, 17408), (17408, 5120)] {
                let mut c = checked_case(&s, f, m, k, n);
                repeated(&s, &mut c); // 9-point F64, all finite/guard/cast/repeat
            }
            if f == GgufBlockFormat::Iq4Nl {
                let mut c = checked_case(&s, f, m, 5120, 10240);
                repeated(&s, &mut c); // actual extra attention projection
            }
        }
    }
    s.synchronize().unwrap();
}
