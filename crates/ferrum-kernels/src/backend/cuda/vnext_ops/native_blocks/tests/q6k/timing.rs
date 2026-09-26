use super::*;
use fixture::Fixture;

const REPLAYS: usize = 8;
const WARMUP_PAIRS: usize = 2;
const PAIRS: usize = 6;
const MINIMUM_IMPROVEMENT: f64 = 0.15;

fn timed_graph(stream: &Arc<CudaStream>, graph: &CudaGraph) -> (f64, f64) {
    stream.synchronize().unwrap();
    let wall = std::time::Instant::now();
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..REPLAYS {
        graph.launch().unwrap();
    }
    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    let wall_ns = wall.elapsed().as_secs_f64() * 1e9;
    let gpu_ns = f64::from(start.elapsed_ms(&end).unwrap()) * 1e6;
    assert!(gpu_ns.is_finite() && gpu_ns > 0.0 && wall_ns.is_finite());
    (gpu_ns, wall_ns)
}

fn screen<I: Scalar, O: Scalar>(
    context: &Arc<CudaContext>,
    kernels: &CudaNativeBlockKernels,
    rows: usize,
    inputs: usize,
    outputs: usize,
    input_type: ElementType,
    output_type: ElementType,
    label: &str,
) -> serde_json::Value {
    let stream = context.new_stream().unwrap();
    let old = generic_control(kernels);
    let mut case = Fixture::<I, O>::new(&stream, rows, inputs, outputs, input_type, output_type);
    // Full physical shape, all outputs, both input generations. No timing is
    // eligible until real production and generic graphs pass this oracle.
    let reference_bits = fixture::qualify(&mut case, &stream, &old, kernels);
    let old_graph = case.capture(&stream, &old);
    let candidate_graph = case.capture(&stream, kernels);
    let old_name = super::super::shared_dispatch::captured_kernel(&old_graph);
    let candidate_name = super::super::shared_dispatch::captured_kernel(&candidate_graph);
    let suffix = if input_type == ElementType::F32 {
        "f32"
    } else {
        "f16"
    };
    assert_eq!(old_name, format!("vnext_gguf_linear_tiled_{suffix}"));
    assert_eq!(
        candidate_name,
        format!("vnext_gguf_linear_q6k_tiled_{suffix}")
    );
    println!(
        "{}",
        serde_json::json!({
            "benchmark":"q6k_fixed_abi", "event":"qualification", "cell":label,
            "rows":rows,"inputs":inputs,"outputs":outputs,
            "input_type":format!("{input_type:?}"),"output_type":format!("{output_type:?}"),
            "old_kernel":old_name,"candidate_kernel":candidate_name,
            "weight_bytes":(inputs / 256) * 210 * outputs,
            "physical_column_templates":fixture::TEMPLATES,
            "scope":"synthetic repeated columns; projection only, not complete FFN/model/serving",
            "bitwise_qualified":true,"f64_qualified":true,"guards_qualified":true,
            "changed_input_generations":2,"graph_dispatches_per_replay":1,
        })
    );
    let mut samples = Vec::new();
    let mut old_gpu = Vec::new();
    let mut new_gpu = Vec::new();
    let mut old_wall = Vec::new();
    let mut new_wall = Vec::new();
    for round in 0..WARMUP_PAIRS + PAIRS {
        let generation = round % 2;
        case.update(&stream, generation);
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut gpu = [0.0; 2];
        let mut host = [0.0; 2];
        for candidate in order {
            case.reset(&stream);
            let graph = if candidate {
                &candidate_graph
            } else {
                &old_graph
            };
            let (gpu_ns, wall_ns) = timed_graph(&stream, graph);
            // Outside timed events: every output + guard + independent F64
            // oracle after every arm, then exact bits against the original
            // generic graph's preflight output for this actual generation.
            let bits = case.validate(&stream);
            assert_eq!(bits, reference_bits[generation], "timed output mismatch");
            let index = usize::from(candidate);
            gpu[index] = gpu_ns / REPLAYS as f64;
            host[index] = wall_ns / REPLAYS as f64;
            let sample = serde_json::json!({
                "benchmark":"q6k_fixed_abi","event":"sample","cell":label,
                "rows":rows,"inputs":inputs,"outputs":outputs,
                "warmup":round < WARMUP_PAIRS,"round":round,
                "pair":round.checked_sub(WARMUP_PAIRS),"generation":generation,
                "candidate":candidate,"captured_graph_replays":REPLAYS,
                "total_gpu_ns":gpu_ns,"total_host_wall_ns":wall_ns,
                "gpu_ns":gpu[index],"host_wall_ns":host[index],
                "all_output_bits_equal":true,"f64_qualified":true,"guards_qualified":true,
            });
            println!("{sample}");
            samples.push(sample);
        }
        if round >= WARMUP_PAIRS {
            old_gpu.push(gpu[0]);
            new_gpu.push(gpu[1]);
            old_wall.push(host[0]);
            new_wall.push(host[1]);
        }
    }
    // Do not scan 834 MB of immutable weights between timed rounds: that
    // would perturb the next round's cache state. Check once after all pairs.
    case.immutable(&stream);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let paired_ratios = old_gpu
        .iter()
        .zip(&new_gpu)
        .map(|(old, new)| new / old)
        .collect::<Vec<_>>();
    let mut ordered_ratios = paired_ratios.clone();
    ordered_ratios.sort_by(f64::total_cmp);
    let middle = ordered_ratios.len() / 2;
    let paired_median_ratio = (ordered_ratios[middle - 1] + ordered_ratios[middle]) / 2.0;
    let improvement = 1.0 - mean(&new_gpu) / mean(&old_gpu);
    let result = serde_json::json!({
        "benchmark":"q6k_fixed_abi","event":"cell_summary","cell":label,
        "rows":rows,"inputs":inputs,"outputs":outputs,"pairs":PAIRS,"warmup_pairs":WARMUP_PAIRS,
        "old_gpu_ns":old_gpu,"candidate_gpu_ns":new_gpu,
        "old_host_wall_ns":old_wall,"candidate_host_wall_ns":new_wall,
        "old_gpu_mean_ns":mean(&old_gpu),"candidate_gpu_mean_ns":mean(&new_gpu),
        "old_host_wall_mean_ns":mean(&old_wall),"candidate_host_wall_mean_ns":mean(&new_wall),
        "paired_gpu_ratios":paired_ratios,"mean_gpu_improvement":improvement,
        "paired_median_gpu_ratio":paired_median_ratio,
        "minimum_paired_median_gpu_improvement":MINIMUM_IMPROVEMENT,
        "performance_screen_passed":paired_median_ratio <= 1.0 - MINIMUM_IMPROVEMENT,
        "bitwise_qualified":true,"f64_qualified":true,"immutable_qualified":true,
        "model_quality_validated":false,"release_approved":false,"samples":samples,
    });
    println!("{result}");
    result
}

#[test]
#[ignore = "exclusive CUDA diagnostic: four real Q6 shapes, 2 warmups + 6 AB/BA pairs"]
fn q6k_fixed_abi_down_and_head_paired_microbench() {
    let context = CudaContext::new(0).unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let mut cells = Vec::new();
    for rows in [4, 8] {
        cells.push(screen::<f16, f16>(
            &context,
            &kernels,
            rows,
            12288,
            4096,
            ElementType::F16,
            ElementType::F16,
            "down_f16",
        ));
        cells.push(screen::<f32, f32>(
            &context,
            &kernels,
            rows,
            4096,
            248320,
            ElementType::F32,
            ElementType::F32,
            "head_f32",
        ));
    }
    println!(
        "{}",
        serde_json::json!({"benchmark":"q6k_fixed_abi","event":"screen_summary",
        "cells":cells,"release_approved":false,
        "scope":"Each cell requires >=15% paired-median GPU improvement; failed cells do not qualify their selector. No FFN/serving claim."})
    );
}
