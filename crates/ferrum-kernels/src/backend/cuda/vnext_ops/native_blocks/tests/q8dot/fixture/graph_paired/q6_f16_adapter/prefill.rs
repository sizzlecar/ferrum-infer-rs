//! Large-row Q6 ABI coverage and isolated projection timing, not serving claims.
use super::*;

#[test]
#[ignore = "requires CUDA and the Q6 F16 prefill native artifact"]
fn q6_f16_prefill_pack_oracle_tail_guards_and_row_isolation() {
    let (_, stream, kernels) = context();
    for (m, k, n) in [
        (33, 768, 129),
        (63, 512, 7),
        (65, 768, 7),
        (513, 768, 7),
        (2048, 256, 7),
    ] {
        let mut c = Case::new(&stream, m, k, n, true, false);
        c.scan(&stream);
        c.inclusive(&stream);
        c.strict(&stream, &kernels);
        stream.synchronize().unwrap();
        assert_eq!(c.p.j, 32);
        correctness::check_pack(&c, &stream);
        let oracle = c.validate(&stream, true);
        let healthy = c.output.read(&stream);
        let original = c.x.clone();
        // Poison the last live row after a complete tile, then clear it with
        // fresh input. Adjacent rows and strided output guards stay untouched.
        c.x[(m - 1) * c.xs] = f16::NAN;
        c.upload_input(&stream);
        c.inclusive(&stream);
        stream.synchronize().unwrap();
        let output = c.output.read(&stream);
        for row in 0..m {
            let range = (row * c.ys + c.output_offset) * 2..(row * c.ys + c.output_offset + n) * 2;
            if row == m - 1 {
                assert!(halves(&output[range]).iter().all(|x| x.to_bits() == 0x7e00));
            } else {
                assert_eq!(&output[range.clone()], &healthy[range]);
            }
        }
        c.guards(&stream);
        c.x = original;
        c.upload_input(&stream);
        let replay = Captured::new(&stream, || c.inclusive(&stream));
        replay.launch();
        stream.synchronize().unwrap();
        assert_eq!(c.output.read(&stream), healthy);
        c.guards(&stream);
        println!(
            "{}",
            serde_json::json!({"kind":"q6_f16_prefill_correctness",
            "m":m,"k":k,"n":n,"j":c.p.j,"fixup":c.p.fixup,"oracle":oracle})
        );
    }
}

#[test]
#[ignore = "exclusive CUDA; synthetic projection diagnostic, not end-to-end throughput"]
fn q6_f16_prefill_projection_pairs() {
    let (_, stream, kernels) = context();
    for m in [128, 512, 2048] {
        let c = Case::new(&stream, m, 9216, 2560, false, false);
        c.scan(&stream);
        c.strict(&stream, &kernels);
        c.inclusive(&stream);
        stream.synchronize().unwrap();
        let oracle = c.validate(&stream, false);
        let graphs = [
            Captured::new(&stream, || c.strict(&stream, &kernels)),
            Captured::new(&stream, || c.inclusive(&stream)),
        ];
        let mut pairs = Vec::new();
        for repetition in 0..6 {
            let mut gpu_ns = [0.0; 2];
            let mut wall_ns = [0.0; 2];
            for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                (wall_ns[route], gpu_ns[route]) = measure(&stream, || graphs[route].launch());
            }
            pairs.push(serde_json::json!({"strict_gpu_ns":gpu_ns[0],
                "mmq_gpu_ns":gpu_ns[1],"strict_wall_ns":wall_ns[0],"mmq_wall_ns":wall_ns[1]}));
        }
        c.guards(&stream);
        println!(
            "{}",
            serde_json::json!({"kind":"q6_f16_prefill_projection_pairs",
            "m":m,"k":c.k,"n":c.n,"scope":"pack+dot+cast; retained weight validation; same weights; hot projection",
            "oracle":oracle,"pairs":pairs})
        );
    }
}
