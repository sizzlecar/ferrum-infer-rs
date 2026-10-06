//! Measures a possible mixed-span LM-head path before changing its resource plan.
//! No production provider selects this experimental gather path.

use super::*;
use cudarc::driver::sys::CUevent_flags;
use std::time::Instant;

#[test]
#[ignore = "requires CUDA and about 1.3 GiB of free device memory; diagnostic microbenchmark"]
fn mixed_f16_last_token_gather_microbench() {
    let context = CudaContext::new(0).expect("LM-head microbenchmark needs a CUDA device");
    let stream = context.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    const HIDDEN: usize = 2560;
    const OUTPUTS: usize = 248320;
    const ITERATIONS: usize = 8;
    let guard = f16::from_f32(-117.0);

    // Dyadic values have an independently computable exact dot product. Every
    // output row has a known signed coefficient; the benchmark validates all
    // outputs outside the timed region, including participant order and guards.
    let coefficient = |column: usize| (column as i32 % 17 - 8) as f32 / 64.0;
    let mut weight = vec![guard; 16 + OUTPUTS * HIDDEN];
    for (column, row) in weight[8..8 + OUTPUTS * HIDDEN]
        .chunks_exact_mut(HIDDEN)
        .enumerate()
    {
        row.fill(f16::from_f32(coefficient(column)));
    }
    let w = stream.clone_htod(&weight).unwrap();
    drop(weight);
    let (wp, _weight_retention) = w.device_ptr(&stream);

    for rows in [2_usize, 4, 8] {
        // Interleave longer prefill spans with unit decode spans. Only the last
        // activation row of each span belongs in the projection.
        let lengths = (0..rows)
            .map(|row| if row % 2 == 0 { 5 + row * 2 } else { 1 })
            .collect::<Vec<_>>();
        let tokens = lengths.iter().sum::<usize>();
        let mut input = vec![guard; 16 + tokens * HIDDEN];
        let mut last_rows = Vec::with_capacity(rows);
        let mut expected_sums = Vec::with_capacity(rows);
        let mut cursor = 0;
        for (row, length) in lengths.into_iter().enumerate() {
            let start = 8 + (cursor + length - 1) * HIDDEN;
            let values = values(HIDDEN, rows - row);
            expected_sums.push(values.iter().map(|value| value.to_f64()).sum::<f64>());
            input[start..start + HIDDEN].copy_from_slice(&values);
            last_rows.push(start);
            cursor += length;
        }
        let x = stream.clone_htod(&input).unwrap();
        let packed_template = vec![guard; 16 + rows * HIDDEN];
        let packed = stream.clone_htod(&packed_template).unwrap();
        let output_template = vec![guard; 16 + rows * OUTPUTS];
        let scalar = stream.clone_htod(&output_template).unwrap();
        let gathered = stream.clone_htod(&output_template).unwrap();
        let (xp, _input_retention) = x.device_ptr(&stream);
        let (pp, _packed_retention) = packed.device_ptr(&stream);
        let (sp, _scalar_retention) = scalar.device_ptr(&stream);
        let (gp, _gathered_retention) = gathered.device_ptr(&stream);

        for round in 0..10 {
            for gather in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                stream.synchronize().unwrap();
                let wall = Instant::now();
                let start = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                for _ in 0..ITERATIONS {
                    if gather {
                        for (row, source) in last_rows.iter().copied().enumerate() {
                            // Same-stream copies and GEMM retain every source,
                            // scratch and output allocation until completion.
                            unsafe {
                                cudarc::driver::result::memcpy_dtod_async(
                                    pp + ((8 + row * HIDDEN) * 2) as u64,
                                    xp + (source * 2) as u64,
                                    HIDDEN * 2,
                                    stream.cu_stream(),
                                )
                            }
                            .unwrap();
                        }
                        transformer::launch_gemm_f16(
                            &blas,
                            pp + 16,
                            wp + 16,
                            gp + 16,
                            rows as i32,
                            OUTPUTS as i32,
                            HIDDEN as i32,
                            "microbench gathered mixed-span LM head",
                        )
                        .unwrap();
                    } else {
                        for (row, source) in last_rows.iter().copied().enumerate() {
                            transformer::launch_gemm_f16(
                                &blas,
                                xp + (source * 2) as u64,
                                wp + 16,
                                sp + ((8 + row * OUTPUTS) * 2) as u64,
                                1,
                                OUTPUTS as i32,
                                HIDDEN as i32,
                                "microbench existing participant LM head",
                            )
                            .unwrap();
                        }
                    }
                }
                let end = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                end.synchronize().unwrap();
                let wall_ns = wall.elapsed().as_nanos() as f64 / ITERATIONS as f64;
                let gpu_ns = f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / ITERATIONS as f64;
                let actual = stream
                    .clone_dtoh(if gather { &gathered } else { &scalar })
                    .unwrap();
                assert!(actual[..8].iter().all(|value| *value == guard));
                assert!(actual[8 + rows * OUTPUTS..]
                    .iter()
                    .all(|value| *value == guard));
                for (row, sum) in expected_sums.iter().copied().enumerate() {
                    for column in 0..OUTPUTS {
                        let expected = f16::from_f64(sum * f64::from(coefficient(column)));
                        assert_eq!(
                            actual[8 + row * OUTPUTS + column],
                            expected,
                            "gather={gather}, row={row}, column={column}"
                        );
                    }
                }
                if round >= 2 {
                    println!(
                        "{}",
                        serde_json::json!({
                            "benchmark": "mixed_f16_last_token_gather",
                            "rows": rows, "hidden": HIDDEN, "outputs": OUTPUTS,
                            "gather": gather, "round": round - 2,
                            "iterations": ITERATIONS, "gpu_ns": gpu_ns,
                            "wall_ns": wall_ns, "validated_outputs": rows * OUTPUTS,
                        })
                    );
                }
            }
        }
        let packed_actual = stream.clone_dtoh(&packed).unwrap();
        assert_eq!(&packed_actual[..8], &packed_template[..8]);
        assert_eq!(
            &packed_actual[8 + rows * HIDDEN..],
            &packed_template[8 + rows * HIDDEN..]
        );
        for (row, source) in last_rows.into_iter().enumerate() {
            assert_eq!(
                &packed_actual[8 + row * HIDDEN..8 + (row + 1) * HIDDEN],
                &input[source..source + HIDDEN]
            );
        }
        assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
    }
}
