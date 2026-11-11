//! Inspect actual image execution evidence with the shared bounded analyzer.
//! A successful command reports facts; it grants no image quality or SLO status.
use anyhow::{Context, Result};
use clap::Parser;
use ferrum_bench_core::images::continuous_trace::{
    analyze_image_continuous_trace, join_image_trace_attempts, ImageContinuousTraceReadLimits,
    ImageTraceBenchAttempt,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "Analyze bounded image step JSONL; does not qualify quality or performance")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    /// Create a new JSON report. Omit to write the report to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value = "67108864")]
    maximum_bytes: NonZeroUsize,
    /// Optional bench-images JSON report. Join opaque identities only; never
    /// subtract its client clock from the trace's server clock.
    #[arg(long)]
    bench_report: Option<PathBuf>,
}

#[derive(Deserialize)]
struct BenchReport {
    endpoints: Vec<BenchEndpoint>,
}
#[derive(Deserialize)]
struct BenchEndpoint {
    endpoint_index: usize,
    requests: Vec<BenchRequest>,
}
#[derive(Deserialize)]
struct BenchRequest {
    sample_index: usize,
    repetition: usize,
    phase: String,
    #[serde(default)]
    client_attempt_id: Option<String>,
    #[serde(default)]
    server_request_id: Option<String>,
    #[serde(default)]
    trace_session_uuid: Option<String>,
}

fn bench_attempts(
    reader: impl Read,
    maximum_bytes: usize,
    maximum_attempts: usize,
) -> Result<Vec<ImageTraceBenchAttempt>> {
    let mut bytes = Vec::new();
    reader
        .take(
            u64::try_from(maximum_bytes)?
                .checked_add(1)
                .context("benchmark report bound overflows")?,
        )
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= maximum_bytes,
        "benchmark report exceeds byte limit"
    );
    // Only the correlation projection is deserialized. Prompt, dataset and
    // image fields are ignored and are never copied into this join report.
    let report: BenchReport =
        serde_json::from_slice(&bytes).context("parse benchmark report correlation")?;
    let count = report
        .endpoints
        .iter()
        .try_fold(0usize, |count, endpoint| {
            count.checked_add(endpoint.requests.len())
        })
        .context("benchmark attempt count overflows")?;
    anyhow::ensure!(
        count <= maximum_attempts,
        "benchmark report exceeds attempt limit"
    );
    Ok(report
        .endpoints
        .into_iter()
        .flat_map(|endpoint| {
            let endpoint_index = endpoint.endpoint_index;
            endpoint
                .requests
                .into_iter()
                .map(move |request| ImageTraceBenchAttempt {
                    endpoint_index,
                    sample_index: request.sample_index,
                    repetition: request.repetition,
                    phase: request.phase,
                    client_attempt_id: request.client_attempt_id,
                    server_request_id: request.server_request_id,
                    trace_session_uuid: request.trace_session_uuid,
                })
        })
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = File::open(&args.input)
        .with_context(|| format!("open image trace {}", args.input.display()))?;
    let mut limits = ImageContinuousTraceReadLimits::default();
    limits.maximum_bytes = args.maximum_bytes.get();
    limits.maximum_record_bytes = limits.maximum_record_bytes.min(limits.maximum_bytes);
    let analysis = analyze_image_continuous_trace(BufReader::new(file), &limits)?;
    let joined = args
        .bench_report
        .as_ref()
        .map(|path| -> Result<_> {
            let attempts = bench_attempts(
                File::open(path).context("open image benchmark report")?,
                args.maximum_bytes.get(),
                limits.maximum_requests,
            )?;
            Ok(join_image_trace_attempts(
                &analysis,
                &attempts,
                limits.maximum_requests,
            )?)
        })
        .transpose()?;
    let mut output: Box<dyn Write> = match args.output {
        Some(path) => Box::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .with_context(|| format!("create image trace report {}", path.display()))?,
        ),
        None => Box::new(std::io::stdout().lock()),
    };
    if let Some(joined) = joined {
        #[derive(Serialize)]
        struct Joined<'a> {
            analysis: &'a ferrum_bench_core::images::continuous_trace::ImageContinuousTraceAnalysis,
            benchmark_join:
                &'a [ferrum_bench_core::images::continuous_trace::ImageTraceBenchJoin],
        }
        serde_json::to_writer_pretty(
            &mut output,
            &Joined {
                analysis: &analysis,
                benchmark_join: &joined,
            },
        )?;
    } else {
        serde_json::to_writer_pretty(&mut output, &analysis)?;
    }
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_bench_report_projection_is_bounded_and_omits_content() {
        let report = br#"{"dataset":{"samples":[{"prompt":"private content"}]},"endpoints":[{"endpoint_index":2,"requests":[{"sample_index":3,"repetition":4,"phase":"measured","client_attempt_id":"00000000-0000-0000-0000-000000000001","e2e_ms":999}]}]}"#;
        let attempts = bench_attempts(&report[..], report.len(), 1).unwrap();
        assert_eq!(
            (attempts[0].endpoint_index, attempts[0].sample_index),
            (2, 3)
        );
        let projected = serde_json::to_string(&attempts).unwrap();
        assert!(!projected.contains("private content") && !projected.contains("e2e_ms"));
        assert!(bench_attempts(&report[..], report.len() - 1, 1).is_err());
        assert!(bench_attempts(&report[..], report.len(), 0).is_err());
    }
}
