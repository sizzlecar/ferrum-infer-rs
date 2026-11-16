//! Read-only paired analysis of independent video_job_bench reports.
//! The manifest predeclares every pair; incomplete pairs are never discarded.
use clap::Parser;
use ferrum_bench_core::ScalarStats;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    maximum_input_bytes: usize,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    experiment: String,
    pairs: Vec<Pair>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    concurrency: usize,
    repetition: usize,
    candidate: PathBuf,
    baseline: PathBuf,
}

#[derive(Deserialize)]
struct InputReport {
    execution_state: String,
    cells: Vec<Cell>,
}
#[derive(Deserialize, Serialize)]
struct Cell {
    phase: String,
    configured_concurrency: usize,
    summary: Summary,
    resources: Option<serde_json::Value>,
    client_errors: Vec<String>,
}
#[derive(Clone, Deserialize, Serialize)]
struct Summary {
    planned: usize,
    offered: usize,
    not_dispatched: usize,
    successes: usize,
    errors: usize,
    completion_fraction: f64,
    p50_seconds: Option<f64>,
    p99_seconds: Option<f64>,
    successful_videos_per_minute: f64,
    successful_generated_frames_per_second: f64,
    successful_video_seconds_per_wall_second: f64,
    client_observation_window_seconds: f64,
    observed_slo_met: bool,
    latency_slo_p99_seconds: f64,
    required_completion_fraction: f64,
    peak_vram_bytes: Option<u64>,
    peak_server_rss_bytes: Option<u64>,
}
impl Summary {
    fn complete(&self) -> bool {
        self.planned > 0
            && self.offered == self.planned
            && self.not_dispatched == 0
            && self.successes == self.planned
            && self.errors == 0
            && self.completion_fraction == 1.0
            && self.required_completion_fraction == 1.0
            && self.p50_seconds.is_some_and(positive)
            && self.p99_seconds.is_some_and(positive)
            && positive(self.successful_videos_per_minute)
            && positive(self.successful_generated_frames_per_second)
            && positive(self.successful_video_seconds_per_wall_second)
            && positive(self.client_observation_window_seconds)
            && positive(self.latency_slo_p99_seconds)
    }
    fn meets_slo(&self) -> bool {
        self.complete()
            && self.observed_slo_met
            && self.p99_seconds.unwrap() <= self.latency_slo_p99_seconds
    }
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

#[derive(Serialize)]
struct Outcome {
    report: PathBuf,
    report_sha256: Option<String>,
    cell: Option<Cell>,
    problem: Option<String>,
}
impl Outcome {
    fn valid(&self) -> Option<&Summary> {
        self.problem.is_none().then_some(())?;
        self.cell.as_ref().map(|c| &c.summary)
    }
}

fn read(path: &std::path::Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    if !file.metadata()?.is_file() || limit == 0 {
        return Err("input must be a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    file.take((limit as u64).checked_add(1).ok_or("byte limit overflow")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input exceeds byte limit".into());
    }
    Ok(bytes)
}

fn load(path: &std::path::Path, concurrency: usize, limit: usize) -> Outcome {
    let mut outcome = Outcome {
        report: path.to_owned(),
        report_sha256: None,
        cell: None,
        problem: None,
    };
    let result = (|| -> Result<()> {
        let bytes = read(path, limit)?;
        outcome.report_sha256 = Some(ferrum_bench_core::images::sha256(&bytes));
        let report: InputReport = serde_json::from_slice(&bytes)?;
        let mut cells = report
            .cells
            .into_iter()
            .filter(|c| c.phase == "measurement");
        outcome.cell = cells.next();
        if report.execution_state != "complete" || cells.next().is_some() {
            return Err("requires one completed independent measurement cell".into());
        }
        let cell = outcome.cell.as_ref().ok_or("measurement cell missing")?;
        if concurrency == 0 || cell.configured_concurrency != concurrency {
            return Err("concurrency differs from manifest".into());
        }
        if !cell.summary.complete() || !cell.client_errors.is_empty() {
            return Err("requests incomplete or invalid; no complete-case subset analysis".into());
        }
        Ok(())
    })();
    outcome.problem = result.err().map(|e| e.to_string());
    outcome
}

#[derive(Debug, Serialize)]
struct Ratio {
    pairs: usize,
    geometric_mean: f64,
    ci95_lower: f64,
    ci95_upper: f64,
}
fn ratio(samples: &[(f64, f64)]) -> Option<Ratio> {
    if samples.len() < 3 || samples.iter().any(|&(c, b)| !positive(c) || !positive(b)) {
        return None;
    }
    let logs: Vec<_> = samples.iter().map(|(c, b)| c.ln() - b.ln()).collect();
    let stats = ScalarStats::from_samples(&logs);
    let result = Ratio {
        pairs: samples.len(),
        geometric_mean: stats.mean.exp(),
        ci95_lower: (stats.mean - stats.ci95_hw).exp(),
        ci95_upper: (stats.mean + stats.ci95_hw).exp(),
    };
    [result.geometric_mean, result.ci95_lower, result.ci95_upper]
        .into_iter()
        .all(positive)
        .then_some(result)
}

#[derive(Serialize)]
struct PairReport {
    repetition: usize,
    candidate: Outcome,
    baseline: Outcome,
}
#[derive(Serialize)]
struct Comparison {
    concurrency: usize,
    pairs: Vec<PairReport>,
    all_pairs_complete: bool,
    all_candidate_slos_met: bool,
    all_baseline_slos_met: bool,
    every_candidate_median_lower: bool,
    p50_seconds_candidate_over_baseline: Option<Ratio>,
    p99_seconds_candidate_over_baseline: Option<Ratio>,
    videos_per_minute_candidate_over_baseline: Option<Ratio>,
}
fn compare(concurrency: usize, pairs: Vec<PairReport>) -> Comparison {
    let complete = !pairs.is_empty()
        && pairs
            .iter()
            .all(|p| p.candidate.valid().is_some() && p.baseline.valid().is_some());
    let candidate_slo = complete
        && pairs
            .iter()
            .all(|p| p.candidate.valid().unwrap().meets_slo());
    let baseline_slo = complete
        && pairs
            .iter()
            .all(|p| p.baseline.valid().unwrap().meets_slo());
    let samples = |metric: fn(&Summary) -> f64| {
        complete.then(|| {
            pairs
                .iter()
                .map(|p| {
                    (
                        metric(p.candidate.valid().unwrap()),
                        metric(p.baseline.valid().unwrap()),
                    )
                })
                .collect::<Vec<_>>()
        })
    };
    let medians = samples(|s| s.p50_seconds.unwrap());
    let every_candidate_median_lower = medians.as_ref().is_some_and(|pairs| {
        pairs
            .iter()
            .all(|(candidate, baseline)| candidate < baseline)
    });
    let median = medians.and_then(|s| ratio(&s));
    let latency = samples(|s| s.p99_seconds.unwrap()).and_then(|s| ratio(&s));
    let throughput = samples(|s| s.successful_videos_per_minute).and_then(|s| ratio(&s));
    Comparison {
        concurrency,
        pairs,
        all_pairs_complete: complete,
        all_candidate_slos_met: candidate_slo,
        all_baseline_slos_met: baseline_slo,
        every_candidate_median_lower,
        p50_seconds_candidate_over_baseline: median,
        p99_seconds_candidate_over_baseline: latency,
        videos_per_minute_candidate_over_baseline: throughput,
    }
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let bytes = read(&args.manifest, args.maximum_input_bytes)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    let mut groups = BTreeMap::<usize, Vec<PairReport>>::new();
    let mut identities = std::collections::BTreeSet::new();
    let mut reports = std::collections::BTreeSet::new();
    for pair in &manifest.pairs {
        if !identities.insert((pair.concurrency, pair.repetition)) {
            return Err("duplicate pair identity".into());
        }
        let mut candidate = load(&pair.candidate, pair.concurrency, args.maximum_input_bytes);
        let mut baseline = load(&pair.baseline, pair.concurrency, args.maximum_input_bytes);
        for outcome in [&mut candidate, &mut baseline] {
            if let Some(hash) = &outcome.report_sha256 {
                if !reports.insert(hash.clone()) {
                    outcome.problem =
                        Some("original report reused as another independent run".into());
                }
            }
        }
        if let (Some(c), Some(b)) = (candidate.valid(), baseline.valid()) {
            if c.planned != b.planned || c.latency_slo_p99_seconds != b.latency_slo_p99_seconds {
                candidate.problem = Some("paired request count or SLO differs".into());
            }
        }
        groups
            .entry(pair.concurrency)
            .or_default()
            .push(PairReport {
                repetition: pair.repetition,
                candidate,
                baseline,
            });
    }
    let comparisons: Vec<_> = groups
        .into_iter()
        .map(|(c, pairs)| compare(c, pairs))
        .collect();
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &serde_json::json!({
            "experiment":manifest.experiment,
            "manifest_sha256":ferrum_bench_core::images::sha256(&bytes),
            "method":"Student-t95CI of mean paired log ratios, exponentiated; n counts independent runs, not requests. Finite-sample engineering evidence, not production-tail certification.",
            "limits":"Independence, model/input equality and quality require external evidence. Resource-query failures remain in each cell; sampled peaks may be incomplete. A speed ratio alone does not establish SLO compliance.",
            "comparisons":comparisons
        }),
    )?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ratios_use_paired_runs_and_reject_missing_or_nonpositive_data() {
        assert!(ratio(&[(1.0, 2.0)]).is_none());
        assert!(ratio(&[(1.0, 2.0), (1.0, 2.0), (0.0, 2.0)]).is_none());
        let r = ratio(&[(1.0, 2.0), (3.0, 6.0), (10.0, 20.0)]).unwrap();
        assert!((r.geometric_mean - 0.5).abs() < 1e-12);
        assert!((r.ci95_lower - 0.5).abs() < 1e-12);
        assert!((r.ci95_upper - 0.5).abs() < 1e-12);
    }
    #[test]
    fn missing_pair_prevents_subset_statistics() {
        let missing = || Outcome {
            report: PathBuf::new(),
            report_sha256: None,
            cell: None,
            problem: Some("missing".into()),
        };
        let result = compare(
            4,
            vec![PairReport {
                repetition: 1,
                candidate: missing(),
                baseline: missing(),
            }],
        );
        assert!(!result.all_pairs_complete);
        assert!(!result.all_candidate_slos_met);
        assert!(!result.every_candidate_median_lower);
        assert!(result.p50_seconds_candidate_over_baseline.is_none());
        assert!(result.p99_seconds_candidate_over_baseline.is_none());
    }
}
