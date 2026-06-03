//! Markdown report generation for bench cells (PLAYBOOK § 2.4 + § 7).
//!
//! Renders one or more `BenchReport`s into a human-readable markdown
//! document — the artifact you commit alongside the JSON to
//! `docs/bench/<platform>-<date>/`. The format is deliberately uniform
//! across scenarios so PRs can diff bench reports the same way they
//! diff source.
//!
//! Significance markers:
//!   - rows with `n_repeats ≥ 3` show `mean ± ci95_hw`
//!   - rows with `n_repeats < 3` show `mean` only and the table prints
//!     a header warning that CI fields are absent

use crate::{BenchReport, ScalarStats, Scenario};
use std::fmt::Write as _;

/// Render a single bench cell into markdown.
pub fn render_single(report: &BenchReport) -> String {
    let mut s = String::new();
    write_header(&mut s, report);
    write_env_block(&mut s, report);
    write_metrics_block(&mut s, report);
    write_completion_block(&mut s, report);
    s
}

/// Render a sweep of cells (each from the same `bench-serve` invocation)
/// into one markdown document with one table per scenario.
pub fn render_sweep(reports: &[BenchReport]) -> String {
    let mut s = String::new();
    writeln!(s, "# Bench sweep ({} cells)\n", reports.len()).ok();
    if let Some(first) = reports.first() {
        write_env_block(&mut s, first);
    }
    write_sweep_table(&mut s, reports);
    s
}

fn write_header(s: &mut String, r: &BenchReport) {
    let scenario_str = match r.scenario {
        Scenario::ClosedLoop => format!("closed_loop · c={}", r.concurrency.unwrap_or(0)),
        Scenario::OpenLoop => format!("open_loop · rate={} req/s", r.request_rate.unwrap_or(0.0)),
        Scenario::SharedPrefix => "shared_prefix".to_string(),
        Scenario::Cli => "cli".to_string(),
    };
    writeln!(s, "# {} — {}", r.model, scenario_str).ok();
    writeln!(s).ok();
}

fn write_env_block(s: &mut String, r: &BenchReport) {
    writeln!(s, "## Environment").ok();
    writeln!(s).ok();
    writeln!(s, "| key | value |").ok();
    writeln!(s, "|---|---|").ok();
    writeln!(s, "| backend | {} |", r.backend).ok();
    writeln!(s, "| commit | `{}` |", r.env.commit_sha).ok();
    writeln!(s, "| hw_id | {} |", r.env.hw_id).ok();
    if let Some(d) = r.env.driver.as_ref() {
        writeln!(s, "| driver | {} |", d).ok();
    }
    if let Some(c) = r.env.cuda.as_ref() {
        writeln!(s, "| cuda | {} |", c).ok();
    }
    writeln!(s, "| rust | {} |", r.env.rust).ok();
    if !r.env.ferrum_features.is_empty() {
        writeln!(s, "| features | {} |", r.env.ferrum_features.join(", ")).ok();
    }
    if let Some(mhz) = r.env.gpu_clock_lock_mhz {
        writeln!(s, "| gpu_clock_lock | {} MHz |", mhz).ok();
    }
    if let Some(w) = r.env.gpu_power_limit_w {
        writeln!(s, "| gpu_power_limit | {} W |", w).ok();
    }
    if let Some(pm) = r.env.gpu_persistence_mode {
        writeln!(s, "| gpu_persistence_mode | {} |", pm).ok();
    }
    writeln!(s, "| env_hash | `{}` |", r.env_hash).ok();
    writeln!(s).ok();
}

fn write_metrics_block(s: &mut String, r: &BenchReport) {
    let has_ci = r.n_repeats >= 3;
    writeln!(s, "## Metrics").ok();
    writeln!(s).ok();
    writeln!(
        s,
        "n_repeats = {}{}",
        r.n_repeats,
        if has_ci {
            ""
        } else {
            " · ⚠ < 3 → no CI (PLAYBOOK § 0.4)"
        }
    )
    .ok();
    writeln!(s).ok();
    writeln!(s, "| metric | p50 | p75 | p95 | p99 |").ok();
    writeln!(s, "|---|---|---|---|---|").ok();
    writeln!(
        s,
        "| TTFT (ms) | {} | {} | {} | {} |",
        fmt(&r.ttft_ms.p50, has_ci),
        fmt(&r.ttft_ms.p75, has_ci),
        fmt(&r.ttft_ms.p95, has_ci),
        fmt(&r.ttft_ms.p99, has_ci)
    )
    .ok();
    writeln!(
        s,
        "| TPOT (ms) | {} | {} | {} | {} |",
        fmt(&r.tpot_ms.p50, has_ci),
        fmt(&r.tpot_ms.p75, has_ci),
        fmt(&r.tpot_ms.p95, has_ci),
        fmt(&r.tpot_ms.p99, has_ci)
    )
    .ok();
    writeln!(
        s,
        "| ITL (ms)  | {} | {} | {} | {} |",
        fmt(&r.itl_ms.p50, has_ci),
        fmt(&r.itl_ms.p75, has_ci),
        fmt(&r.itl_ms.p95, has_ci),
        fmt(&r.itl_ms.p99, has_ci)
    )
    .ok();
    writeln!(
        s,
        "| E2E (ms)  | {} | {} | {} | {} |",
        fmt(&r.e2e_ms.p50, has_ci),
        fmt(&r.e2e_ms.p75, has_ci),
        fmt(&r.e2e_ms.p95, has_ci),
        fmt(&r.e2e_ms.p99, has_ci)
    )
    .ok();
    writeln!(s).ok();
    writeln!(s, "| throughput / goodput | value |").ok();
    writeln!(s, "|---|---|").ok();
    writeln!(
        s,
        "| output_throughput (tok/s) | {} |",
        fmt(&r.output_throughput_tps, has_ci)
    )
    .ok();
    writeln!(
        s,
        "| total_throughput (tok/s)  | {} |",
        fmt(&r.total_throughput_tps, has_ci)
    )
    .ok();
    writeln!(
        s,
        "| request_throughput (req/s) | {} |",
        fmt(&r.request_throughput_rps, has_ci)
    )
    .ok();
    let slo_meaningful = r.slo.ttft_p99_ms.is_finite()
        && r.slo.tpot_p99_ms.is_finite()
        && r.slo.e2e_p99_ms.is_finite();
    if slo_meaningful {
        writeln!(
            s,
            "| **goodput (req/s)** | {} (SLO ttft:{}ms tpot:{}ms e2e:{}ms) |",
            fmt(&r.goodput_rps, has_ci),
            r.slo.ttft_p99_ms,
            r.slo.tpot_p99_ms,
            r.slo.e2e_p99_ms
        )
        .ok();
    }
    writeln!(s).ok();
}

fn write_completion_block(s: &mut String, r: &BenchReport) {
    writeln!(s, "## Per-run breakdown").ok();
    writeln!(s).ok();
    writeln!(s, "| run | completed | errored |").ok();
    writeln!(s, "|---|---|---|").ok();
    for (i, (c, e)) in r
        .completed_per_run
        .iter()
        .zip(r.errored_per_run.iter())
        .enumerate()
    {
        writeln!(s, "| {} | {} | {} |", i + 1, c, e).ok();
    }
    writeln!(s).ok();
}

fn write_sweep_table(s: &mut String, reports: &[BenchReport]) {
    if reports.is_empty() {
        return;
    }
    let has_ci = reports.iter().all(|r| r.n_repeats >= 3);
    writeln!(s, "## Sweep").ok();
    writeln!(s).ok();
    if !has_ci {
        writeln!(
            s,
            "⚠ At least one cell has `n_repeats < 3` — CI95 columns omitted for those cells."
        )
        .ok();
        writeln!(s).ok();
    }
    writeln!(
        s,
        "| cell | TTFT_p50 | TTFT_p99 | TPOT_p50 | output_thr | goodput |"
    )
    .ok();
    writeln!(s, "|---|---|---|---|---|---|").ok();
    for r in reports {
        let label = match r.scenario {
            Scenario::ClosedLoop => format!("c={}", r.concurrency.unwrap_or(0)),
            Scenario::OpenLoop => format!("rate={}", r.request_rate.unwrap_or(0.0)),
            Scenario::SharedPrefix => "shared_prefix".to_string(),
            Scenario::Cli => "cli".to_string(),
        };
        let cell_has_ci = r.n_repeats >= 3;
        writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} |",
            label,
            fmt(&r.ttft_ms.p50, cell_has_ci),
            fmt(&r.ttft_ms.p99, cell_has_ci),
            fmt(&r.tpot_ms.p50, cell_has_ci),
            fmt(&r.output_throughput_tps, cell_has_ci),
            fmt(&r.goodput_rps, cell_has_ci)
        )
        .ok();
    }
    writeln!(s).ok();
}

fn fmt(stat: &ScalarStats, has_ci: bool) -> String {
    if has_ci && stat.ci95_hw > 0.0 {
        format!("{:.2} ± {:.2}", stat.mean, stat.ci95_hw)
    } else {
        format!("{:.2}", stat.mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compute_metrics, Env, OutputTokenCountSource, RequestRecord, RunRecord, Scenario, Slo,
    };

    fn make_run(records: Vec<(bool, f64, f64, u32, u32)>, duration_s: f64) -> RunRecord {
        let records = records
            .into_iter()
            .map(|(success, ttft, e2e, in_tok, out_tok)| RequestRecord {
                success,
                ttft_ms: ttft,
                e2e_ms: e2e,
                input_tokens: in_tok,
                output_tokens: out_tok,
                output_token_count_source: OutputTokenCountSource::StreamChunks,
                itl_ms: vec![],
            })
            .collect();
        RunRecord {
            records,
            duration_s,
        }
    }

    fn fixture_report() -> BenchReport {
        compute_metrics(
            "qwen3:0.6b".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(32),
            None,
            256,
            128,
            10,
            Slo::default(),
            vec![
                make_run(vec![(true, 100.0, 200.0, 256, 128)], 1.0),
                make_run(vec![(true, 105.0, 210.0, 256, 128)], 1.05),
                make_run(vec![(true, 95.0, 195.0, 256, 128)], 0.98),
            ],
            Env::default(),
        )
    }

    #[test]
    fn single_report_contains_headline_sections() {
        let report = fixture_report();
        let md = render_single(&report);
        assert!(md.contains("# qwen3:0.6b"));
        assert!(md.contains("closed_loop · c=32"));
        assert!(md.contains("## Environment"));
        assert!(md.contains("## Metrics"));
        assert!(md.contains("env_hash"));
        // n_repeats=3 → CI columns ARE present
        assert!(!md.contains("⚠ < 3"));
        assert!(md.contains("±")); // mean ± ci95 format
    }

    #[test]
    fn single_run_omits_ci() {
        let report = compute_metrics(
            "tiny".into(),
            "cpu".into(),
            Scenario::Cli,
            None,
            None,
            5,
            5,
            0,
            Slo::default(),
            vec![make_run(vec![(true, 100.0, 200.0, 5, 5)], 1.0)],
            Env::default(),
        );
        let md = render_single(&report);
        assert!(md.contains("⚠ < 3"));
        assert!(!md.contains("±"));
    }

    #[test]
    fn sweep_table_one_row_per_cell() {
        let reports = vec![fixture_report(), fixture_report()];
        let md = render_sweep(&reports);
        assert!(md.contains("# Bench sweep (2 cells)"));
        assert!(md.contains("| cell |"));
        // Two data rows.
        let row_count = md.matches("| c=32 |").count();
        assert_eq!(row_count, 2);
    }
}
