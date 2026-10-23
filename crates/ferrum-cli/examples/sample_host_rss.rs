//! Read-only Linux server RSS sampling. This process never signals its target.
//! RSS sums double-count shared pages. Sampled maxima can miss transient peaks;
//! per-process VmHWM is lifetime evidence, never a window peak or a summed peak.
//! Use `--pid <server-pid>` or `--cgroup-directory <absolute-service-cgroup>`,
//! together with `--duration-ms <window>` and `--output <new-file.jsonl>`.
//! An optional fresh `--stop-file <path>` ends the window when created. Target
//! exit/reuse, empty/replaced cgroups and incomplete windows produce a nonzero
//! exit status after writing diagnostics. No target or service is stopped.

use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

#[cfg(all(unix, any(target_os = "linux", test)))]
#[path = "sample_host_rss/sampling.rs"]
mod sampling;

#[derive(Debug, Parser, Serialize)]
#[command(about = "Sample an existing Linux process or cgroup's RSS without controlling it")]
struct Arguments {
    #[arg(
        long,
        conflicts_with = "cgroup_directory",
        required_unless_present = "cgroup_directory"
    )]
    pid: Option<u32>,
    /// Absolute service cgroup directory; child cgroups are included.
    #[arg(long, conflicts_with = "pid", required_unless_present = "pid")]
    cgroup_directory: Option<PathBuf>,
    /// New JSONL file. Existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 200)]
    interval_ms: u64,
    #[arg(long)]
    duration_ms: u64,
    #[arg(long, default_value_t = 256)]
    maximum_pids: usize,
    /// Counts the root plus every directory entry, including ordinary files.
    #[arg(long, default_value_t = 4096)]
    maximum_traversal_nodes: usize,
    /// Aggregate procfs/cgroup content bytes read during one sample.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    maximum_input_bytes: usize,
    /// Optional fresh stop token; its creation stops only this sampler.
    #[arg(long)]
    stop_file: Option<PathBuf>,
}

#[cfg(all(unix, any(target_os = "linux", test)))]
impl Arguments {
    fn validate(&self) -> std::io::Result<()> {
        let invalid = |message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message);
        if self
            .pid
            .is_some_and(|pid| pid <= 1 || pid > i32::MAX as u32)
            || self.pid.is_some() == self.cgroup_directory.is_some()
        {
            return Err(invalid(
                "choose exactly one ordinary PID or cgroup directory",
            ));
        }
        if self
            .cgroup_directory
            .as_ref()
            .is_some_and(|p| !p.is_absolute())
        {
            return Err(invalid("cgroup directory must be absolute"));
        }
        if !(10..=60_000).contains(&self.interval_ms)
            || !(1..=86_400_000).contains(&self.duration_ms)
            || !(1..=65_536).contains(&self.maximum_pids)
            || !(1..=1_000_000).contains(&self.maximum_traversal_nodes)
            || !(1..=64 * 1024 * 1024).contains(&self.maximum_input_bytes)
        {
            return Err(invalid(
                "sampling interval, duration or resource bounds are invalid",
            ));
        }
        if self.stop_file.as_ref().is_some_and(|p| p == &self.output) {
            return Err(invalid("stop token and output must differ"));
        }
        if let Some(path) = &self.stop_file {
            if path.try_exists()? {
                return Err(invalid("stop token already exists"));
            }
        }
        Ok(())
    }
}

#[cfg(all(unix, any(target_os = "linux", test)))]
fn create_output(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

#[cfg(target_os = "linux")]
fn run(args: Arguments) -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        io::{BufWriter, Write},
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };
    args.validate()?;
    let mut sampler = sampling::Sampler::new(&args, PathBuf::from("/proc"))?;
    let file = create_output(&args.output)?;
    let mut output = BufWriter::with_capacity(64 * 1024, file);
    fn line(output: &mut impl Write, value: &impl Serialize) -> std::io::Result<()> {
        serde_json::to_writer(&mut *output, value)?;
        output.write_all(b"\n")?;
        output.flush()
    }
    let start = Instant::now();
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    line(
        &mut output,
        &serde_json::json!({
            "event": "monitor_start", "schema": "ferrum.host_rss_samples.v1",
            "started_unix_ms": started_unix_ms, "configuration": args,
            "semantics": {
                "rss_sum": "sum of observed process VmRSS; shared pages may be counted multiple times",
                "peak": "maximum complete sampled RSS sum; transients between samples can be missed",
                "high_water": "per-process VmHWM is lifetime until observation, not a window peak; never summed",
                "identity": "start_ticks is /proc/PID/stat field 22, boot-relative clock ticks; PID mode pins its first successful observation",
                "cgroup": "recursive cgroup.procs membership, not cgroup memory.current; sequential reads are not atomic",
                "incomplete": "any read/identity/membership error makes rss_sum_bytes null; partial sum is diagnostic only",
                "control": "read-only; no target signals, subprocesses, GPU queries or service changes"
            }
        }),
    )?;
    let interval = Duration::from_millis(args.interval_ms);
    let deadline = start + Duration::from_millis(args.duration_ms);
    let mut next = start;
    let mut samples = 0u64;
    let mut complete = 0u64;
    let mut errors = 0u64;
    let mut peak: Option<u64> = None;
    let mut maximum_gap_us = 0u64;
    let mut previous_sample_us = None;
    let mut maximum_sample_duration_us = 0u64;
    let reason = loop {
        if Instant::now() >= deadline {
            break "duration_elapsed";
        }
        if let Some(path) = &args.stop_file {
            match path.try_exists() {
                Ok(true) => break "stop_token",
                Ok(false) => {}
                Err(error) => {
                    errors += 1;
                    line(
                        &mut output,
                        &serde_json::json!({"event":"control_error","error":error.to_string()}),
                    )?;
                    break "stop_token_read_error";
                }
            }
        }
        let began = Instant::now();
        let elapsed_us = u64::try_from(began.duration_since(start).as_micros()).unwrap_or(u64::MAX);
        if let Some(previous) = previous_sample_us {
            maximum_gap_us = maximum_gap_us.max(elapsed_us.saturating_sub(previous));
        }
        previous_sample_us = Some(elapsed_us);
        let sample = sampler.sample(&args);
        let sample_duration_us = u64::try_from(began.elapsed().as_micros()).unwrap_or(u64::MAX);
        maximum_sample_duration_us = maximum_sample_duration_us.max(sample_duration_us);
        samples += 1;
        errors += sample.errors.len() as u64;
        if let Some(rss) = sample.rss_sum_bytes {
            complete += 1;
            peak = Some(peak.unwrap_or(0).max(rss));
        }
        line(
            &mut output,
            &serde_json::json!({
                "event":"host_sample", "index":samples - 1, "elapsed_us":elapsed_us,
                "sample_duration_us":sample_duration_us, "sample":sample
            }),
        )?;
        if let Some(reason) = sample.stop_reason {
            break reason;
        }
        next += interval;
        let now = Instant::now();
        if next <= now {
            next = now + interval;
        }
        std::thread::sleep(next.min(deadline).saturating_duration_since(now));
    };
    let successful = matches!(reason, "duration_elapsed" | "stop_token")
        && samples > 0
        && complete == samples
        && errors == 0;
    line(
        &mut output,
        &serde_json::json!({
            "event":"monitor_stop", "reason":reason, "complete_window":successful,
            "elapsed_ms":start.elapsed().as_millis(), "samples":samples,
            "complete_samples":complete, "incomplete_samples":samples - complete,
            "read_or_validation_errors":errors, "sampled_max_complete_rss_sum_bytes":peak,
            "maximum_sample_gap_us":maximum_gap_us, "maximum_sample_duration_us":maximum_sample_duration_us
        }),
    )?;
    output.get_ref().sync_all()?;
    if !successful {
        return Err("monitoring window was incomplete; see JSONL diagnostics".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();
    #[cfg(target_os = "linux")]
    {
        run(args)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = args;
        Err("sample_host_rss requires Linux procfs/cgroup sources".into())
    }
}
