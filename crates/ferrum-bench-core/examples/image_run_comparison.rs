//! Independent whole-run image throughput comparison; never a G3/quality verdict.
//! Run with --plan <predeclared.json>. JSON goes to stdout; inputs are read-only.

use clap::Parser;
use serde::Serialize;
use std::{collections::BTreeSet, path::PathBuf};

#[path = "image_run_comparison/fieller.rs"]
mod fieller;
#[path = "image_run_comparison/input.rs"]
mod input;
#[cfg(test)]
#[path = "image_run_comparison/tests.rs"]
mod tests;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    plan: PathBuf,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    maximum_input_bytes: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    plan_sha256: String,
    plan: input::Plan,
    independence_status: &'static str,
    model_assumption: &'static str,
    quality_capacity_and_g3_status: &'static str,
    workloads: Vec<WorkloadReport>,
}

#[derive(Serialize)]
struct WorkloadReport {
    id: String,
    planned_pairs: usize,
    complete_pairs: usize,
    pairs: Vec<PairReport>,
    analysis: fieller::Analysis,
}

#[derive(Serialize)]
struct PairReport {
    id: String,
    order: input::Order,
    baseline: input::RunOutcome,
    candidate: input::RunOutcome,
}

fn analyze(plan: input::Plan, plan_sha256: String, root: &std::path::Path, limit: usize) -> Report {
    let mut workloads = Vec::new();
    // Reusing the same endpoint from the same original report is not a new run.
    let mut consumed = BTreeSet::new();
    for workload in &plan.workloads {
        let mut pairs = Vec::new();
        let mut samples = Vec::new();
        let mut eligible = true;
        for pair in &workload.pairs {
            let mut baseline =
                input::load_run(root, &pair.baseline, workload, &workload.baseline, limit);
            let mut candidate =
                input::load_run(root, &pair.candidate, workload, &workload.candidate, limit);
            for outcome in [&mut baseline, &mut candidate] {
                if let Some(hash) = &outcome.report_sha256 {
                    if !consumed.insert((hash.clone(), outcome.source.endpoint_index)) {
                        outcome.problem =
                            Some("original endpoint report reused as another run".into());
                    }
                }
            }
            if let (Some(x), Some(y)) =
                (baseline.valid_observation(), candidate.valid_observation())
            {
                samples.push((
                    x.measured.images_per_second.unwrap(),
                    y.measured.images_per_second.unwrap(),
                ));
                eligible &= x.eligible && y.eligible;
            }
            pairs.push(PairReport {
                id: pair.id.clone(),
                order: pair.order,
                baseline,
                candidate,
            });
        }
        let complete = samples.len() == workload.planned_pairs;
        let mut analysis = if complete {
            fieller::analyze(&samples)
        } else {
            fieller::Analysis::insufficient(
                "missing or invalid pair; no complete-case subset analysis",
            )
        };
        if !eligible && complete {
            analysis.status = fieller::Status::Insufficient;
            analysis.reason = Some("request/capture failures or measured latency SLO not met; all observations retained".into());
        }
        workloads.push(WorkloadReport {
            id: workload.id.clone(),
            planned_pairs: workload.planned_pairs,
            complete_pairs: samples.len(),
            pairs,
            analysis,
        });
    }
    Report { schema_version: 1, plan_sha256, plan,
        independence_status: "operator_declared_external_evidence_requires_manual_review; report IDs and restarts do not prove independence",
        model_assumption: "independent identically distributed paired whole-run throughputs; bivariate normal model; n counts pairs, not requests or in-process repetitions",
        quality_capacity_and_g3_status: "not_evaluated", workloads }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();
    let bytes = input::read_bounded(&args.plan, args.maximum_input_bytes)?;
    let plan: input::Plan = serde_json::from_slice(&bytes)?;
    plan.validate().map_err(std::io::Error::other)?;
    let report = analyze(
        plan,
        ferrum_bench_core::images::sha256(&bytes),
        args.plan.parent().unwrap_or(std::path::Path::new(".")),
        args.maximum_input_bytes,
    );
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    // Successful analysis may report insufficient/below-target; never turn a
    // statistical finding into an IO error that discards the JSON evidence.
    Ok(())
}
