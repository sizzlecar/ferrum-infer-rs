//! External MLX operator diagnostic. No Ferrum provider or resource ledger is used.
//! Build the separately supplied CMake bridge against an existing MLX installation.
#[path = "mlx_sdpa/args.rs"]
mod args;
#[path = "mlx_sdpa/bridge.rs"]
mod bridge;
#[path = "mlx_sdpa/fixture.rs"]
mod fixture;

use args::Args;
use bridge::Bridge;
use serde::Serialize;
use std::{error::Error, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Serialize)]
struct Sample {
    iteration: usize,
    warmup: bool,
    evaluate_ms: f64,
    copy_bshd_ms: f64,
    evaluate_and_copy_ms: f64,
    checked_output_elements: usize,
}

#[derive(Serialize)]
struct Statistics {
    count: usize,
    mean_ms: f64,
    median_ms: f64,
    minimum_ms: f64,
    maximum_ms: f64,
}

impl Statistics {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        Self {
            count: n,
            mean_ms: values.iter().sum::<f64>() / n as f64,
            median_ms: (values[(n - 1) / 2] + values[n / 2]) / 2.0,
            minimum_ms: values[0],
            maximum_ms: values[n - 1],
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("MLX SDPA diagnostic failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        println!("{}", args::USAGE);
        return Ok(());
    };
    let bridge = Bridge::load(&args.bridge, &args.expected_mlx_version)?;
    let inherited_mlx_controls: std::collections::BTreeMap<_, _> = [
        "MLX_METAL_GPU_ARCH",
        "MLX_DISABLE_COMPILE",
        "MLX_ENABLE_TF32",
        "MLX_MAX_OPS_PER_BUFFER",
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_SDPA_BLOCKS",
        "MLX_METAL_FAST_SYNCH",
    ]
    .into_iter()
    .map(|key| {
        (
            key,
            std::env::var_os(key).map(|v| v.to_string_lossy().into_owned()),
        )
    })
    .collect();
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "event": "configuration", "schema_version": 1, "args": args,
            "bridge": bridge.identity(),
            "inherited_mlx_controls": inherited_mlx_controls,
            "input_layout": "contiguous BF16 BSHD backing, BHSD transpose view supplied to MLX",
            "mask": "none", "causal": false, "scale": fixture::SCALE,
            "input_recipe": "Q/K=((flat_index%7)-3)/2; V=(((flat_index%(heads*dim))%17)-8)/16",
            "timing_scope": "Rust wall around fresh SDPA graph construction and synchronous mx::eval, then separate CPU copy into Rust BSHD buffer",
            "excluded_from_sample_timing": "input construction/upload, small layout probe, output oracle, serialization and final owner destruction",
            "comparison_scope": "External MLX allocator/dispatch, without Ferrum managed admission/encode/lane/ledger or Ferrum diagnostic logging; evaluate+copy is not pure GPU kernel time",
            "memory_scope": "maximum_host_bytes bounds Rust tensor/oracle vectors only; MLX owns additional input/output arrays, allocator cache and backend workspace outside the Ferrum ledger",
            "actual_selected_shader": null
        }))?
    );

    // Always exercise nonconstant V, different batches/heads and rectangular
    // lengths against an independently constructed FP64 reference first.
    let probe = fixture::layout_probe(&bridge, args.maximum_host_bytes)?;
    println!("{}", serde_json::to_string(&probe)?);

    let shape = args.shape;
    let inputs = fixture::benchmark_inputs(shape)?;
    let mut output = fixture::zeroed_words(shape.query_elements()?)?;
    let start = Instant::now();
    let mut session = bridge.create(shape, &inputs)?;
    let initialization_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(inputs);
    let mut samples = Vec::with_capacity(args.warmups + args.iterations);
    for iteration in 0..args.warmups + args.iterations {
        let start = Instant::now();
        session.evaluate()?;
        let evaluated = Instant::now();
        session.copy_output(&mut output)?;
        let copied = Instant::now();
        // Full exact constant-V oracle, outside both recorded intervals. This
        // cannot by itself validate the softmax probability distribution.
        fixture::check_constant_output(shape, &output)?;
        let sample = Sample {
            iteration,
            warmup: iteration < args.warmups,
            evaluate_ms: (evaluated - start).as_secs_f64() * 1000.0,
            copy_bshd_ms: (copied - evaluated).as_secs_f64() * 1000.0,
            evaluate_and_copy_ms: (copied - start).as_secs_f64() * 1000.0,
            checked_output_elements: output.len(),
        };
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({"event":"sample", "sample":sample}))?
        );
        samples.push(sample);
    }
    let start = Instant::now();
    session.close()?;
    let close_ms = start.elapsed().as_secs_f64() * 1000.0;
    let measured: Vec<_> = samples.iter().filter(|x| !x.warmup).collect();
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "event":"summary", "initialization_ms":initialization_ms, "close_ms":close_ms,
            "evaluate":Statistics::new(measured.iter().map(|x|x.evaluate_ms).collect()),
            "copy_bshd":Statistics::new(measured.iter().map(|x|x.copy_bshd_ms).collect()),
            "evaluate_and_copy":Statistics::new(measured.iter().map(|x|x.evaluate_and_copy_ms).collect()),
            "full_output_checks":samples.len(), "finite_and_exact_constant_v":true,
            "oracle_scope":"constant-V full output check plus separate small nonconstant FP64 layout probe; not a full-domain MLX numerical qualification",
            "sampling_scope":"consecutive calls in one process, not independent process repetitions",
            "array_owner_closed":true,
            "retirement_scope":"MLX session arrays released after stream synchronization; no claim of MLX global allocator cache or native device allocation returning to zero"
        }))?
    );
    Ok(())
}
