//! Markdown report generation for bench cells (PLAYBOOK § 2.4 + § 7).
//!
//! Renders one or more `BenchReport`s into a human-readable markdown
//! document — the artifact you commit alongside the JSON to
//! a caller-selected output directory. The format is deliberately uniform
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
    write_dataset_block(&mut s, report);
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
        Scenario::DecodeIsolation => "decode_isolation".to_string(),
    };
    let dataset_suffix = if r.dataset_evidence.is_some() {
        " · ShareGPT variable"
    } else {
        ""
    };
    writeln!(s, "# {} — {}{}", r.model, scenario_str, dataset_suffix).ok();
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
    if let Some(mode) = r.env.http_connection_mode.as_ref() {
        writeln!(s, "| http_connection_mode | {} |", mode).ok();
    }
    if let Some(sampling) = r.env.http_request_sampling {
        writeln!(
            s,
            "| http_request_sampling | `{}` |",
            serde_json::to_string(&sampling).expect("validated sampling")
        )
        .ok();
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

fn write_dataset_block(s: &mut String, r: &BenchReport) {
    let Some(evidence) = &r.dataset_evidence else {
        return;
    };
    let measured: Vec<_> = evidence
        .repeats
        .iter()
        .flat_map(|repeat| &repeat.samples)
        .filter(|sample| sample.phase == crate::BenchmarkPhase::Measured)
        .collect();
    writeln!(s, "## Dataset\n").ok();
    writeln!(s, "ShareGPT variable lengths; the statistics below exclude warmup. Requested output budgets are not actual generated lengths.\n").ok();
    writeln!(s, "| field | value |\n|---|---|").ok();
    writeln!(s, "| measured samples | {} |", measured.len()).ok();
    writeln!(
        s,
        "| prompt tokens (client tokenizer, before chat template) | {} |",
        fmt_token_lengths(measured.iter().map(|sample| sample.input_tokens))
    )
    .ok();
    writeln!(
        s,
        "| requested output budget (tokens) | {} |",
        fmt_token_lengths(measured.iter().map(|sample| sample.requested_output_tokens))
    )
    .ok();
    writeln!(s, "| source SHA-256 | `{}` |", evidence.source_sha256).ok();
    writeln!(
        s,
        "| selection / filter | prompt seed {}; full filter and per-repeat selection hashes in JSON `dataset_evidence` |",
        evidence.prompt_seed
    )
    .ok();
    writeln!(s).ok();
}

fn fmt_token_lengths(values: impl Iterator<Item = u32>) -> String {
    let values: Vec<_> = values.collect();
    match (values.iter().min(), values.iter().max()) {
        (Some(min), Some(max)) => format!(
            "min {}, max {}, mean {:.2}",
            min,
            max,
            values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
        ),
        _ => "unavailable".to_string(),
    }
}

fn missing_visible_itl(r: &BenchReport) -> &'static str {
    match &r.sse_text_event_gap_evidence {
        None => "not collected",
        Some(evidence) if evidence.contributing_intervals == 0 => "unavailable (no intervals)",
        Some(_) => "unavailable (incomplete event timing)",
    }
}

fn write_itl_diagnostics(s: &mut String, r: &BenchReport) {
    writeln!(
        s,
        "Strict-token ITL diagnostic: {}. This qualification does not gate visible text update intervals.",
        if r.has_complete_itl_evidence() {
            "complete token timing evidence"
        } else {
            "incomplete or unavailable token timing evidence"
        }
    )
    .ok();
    writeln!(s).ok();
    if let Some(evidence) = &r.sse_text_event_gap_evidence {
        writeln!(
            s,
            "Visible text timing: {} successful / {} failed requests; {} observed text events / {} observed intervals; {} contributing intervals. Transport coalescing: {} requests / {} chunks; event/usage count mismatches: {}; missing usage: {}; fewer than two text events: {}; interval-count mismatches: {}; successful requests without SSE evidence: {}; failed requests with observed intervals: {}.",
            evidence.successful_requests,
            evidence.failed_requests,
            evidence.observed_text_events,
            evidence.observed_intervals,
            evidence.contributing_intervals,
            evidence.transport_coalesced_requests,
            evidence.transport_coalesced_output_chunks,
            evidence.event_usage_mismatch_requests,
            evidence.missing_usage_requests,
            evidence.fewer_than_two_events_requests,
            evidence.interval_count_mismatch_requests,
            evidence.successful_requests_without_sse_evidence,
            evidence.failed_requests_with_observed_intervals,
        )
        .ok();
        writeln!(s).ok();
    }
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
        "| TPOT (ms/token) | {} | {} | {} | {} |",
        fmt(&r.tpot_ms.p50, has_ci),
        fmt(&r.tpot_ms.p75, has_ci),
        fmt(&r.tpot_ms.p95, has_ci),
        fmt(&r.tpot_ms.p99, has_ci)
    )
    .ok();
    if let Some(itl) = &r.sse_text_event_gap_ms {
        writeln!(
            s,
            "| ITL (visible text updates, ms) | {} | {} | {} | {} |",
            fmt(&itl.p50, has_ci),
            fmt(&itl.p75, has_ci),
            fmt(&itl.p95, has_ci),
            fmt(&itl.p99, has_ci)
        )
        .ok();
    } else {
        let missing = missing_visible_itl(r);
        writeln!(
            s,
            "| ITL (visible text updates, ms) | {missing} | {missing} | {missing} | {missing} |"
        )
        .ok();
    }
    if r.has_complete_itl_evidence() {
        writeln!(
            s,
            "| ITL (strict token diagnostic, ms) | {} | {} | {} | {} |",
            fmt(&r.itl_ms.p50, has_ci),
            fmt(&r.itl_ms.p75, has_ci),
            fmt(&r.itl_ms.p95, has_ci),
            fmt(&r.itl_ms.p99, has_ci)
        )
        .ok();
    } else {
        writeln!(
            s,
            "| ITL (strict token diagnostic, ms) | unavailable | unavailable | unavailable | unavailable |"
        )
        .ok();
    }
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
    write_itl_diagnostics(s, r);
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
    let slo_meaningful = !r.slo.is_unbounded()
        && r.slo.ttft_p99_ms.is_finite()
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
    if slo_meaningful {
        writeln!(
            s,
            "Legacy goodput counts requests meeting individual TTFT, TPOT and E2E bounds; it does not evaluate aggregate TTFT/TPOT/ITL P99 SLOs."
        )
        .ok();
        writeln!(s).ok();
    }
    writeln!(s, "| server memory metric | value |").ok();
    writeln!(s, "|---|---|").ok();
    writeln!(s, "| Peak GPU allocated (GiB) | not collected |").ok();
    writeln!(s, "| Peak OS footprint (GiB) | not collected |").ok();
    writeln!(s, "| Maximum RSS (GiB) | not collected |").ok();
    writeln!(s).ok();
    write_memory_scope_note(s);
}

fn write_memory_scope_note(s: &mut String) {
    writeln!(
        s,
        "Server memory is not collected by this client report. Attach server-side evidence with the API, process/device identity, measurement window and sampling interval. On Metal, sampled peak allocated bytes, OS footprint and maximum RSS have different scopes; none alone establishes the complete model working set, and overlapping values must not be added. Client RSS, configured budgets and model file size are not server memory peaks."
    )
    .ok();
    writeln!(s).ok();
}

fn write_completion_block(s: &mut String, r: &BenchReport) {
    writeln!(s, "## Per-run breakdown").ok();
    writeln!(s).ok();
    writeln!(
        s,
        "| run | completed | errored | bad_output | malformed_stream | missing_done | duplicate_done | zero_output_tokens | http_500 | panic | stream_bulk_flush |"
    )
    .ok();
    writeln!(s, "|---|---|---|---|---|---|---|---|---|---|---|").ok();
    for i in 0..r.completed_per_run.len() {
        writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            i + 1,
            r.completed_per_run[i],
            r.errored_per_run.get(i).copied().unwrap_or(0),
            r.bad_output_per_run.get(i).copied().unwrap_or(0),
            r.malformed_stream_per_run.get(i).copied().unwrap_or(0),
            r.missing_done_per_run.get(i).copied().unwrap_or(0),
            r.duplicate_done_per_run.get(i).copied().unwrap_or(0),
            r.zero_output_tokens_per_run.get(i).copied().unwrap_or(0),
            r.http_500_per_run.get(i).copied().unwrap_or(0),
            r.panic_per_run.get(i).copied().unwrap_or(0),
            r.stream_bulk_flush_per_run.get(i).copied().unwrap_or(0),
        )
        .ok();
    }
    writeln!(s).ok();
}

fn sweep_cell_label(r: &BenchReport) -> String {
    match r.scenario {
        Scenario::ClosedLoop => format!("c={}", r.concurrency.unwrap_or(0)),
        Scenario::OpenLoop => format!("rate={}", r.request_rate.unwrap_or(0.0)),
        Scenario::SharedPrefix => "shared_prefix".to_string(),
        Scenario::Cli => "cli".to_string(),
        Scenario::DecodeIsolation => "decode_isolation".to_string(),
    }
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
        "| concurrency / cell | TTFT P50 (ms) | TTFT P99 (ms) | TPOT P50 (ms/token) | TPOT P99 (ms/token) | ITL (visible text updates) P50 (ms) | ITL (visible text updates) P99 (ms) | output Throughput (tok/s) | Peak GPU allocated (GiB) | Peak OS footprint (GiB) | Maximum RSS (GiB) |"
    )
    .ok();
    writeln!(s, "|---|---|---|---|---|---|---|---|---|---|---|").ok();
    for r in reports {
        let label = sweep_cell_label(r);
        let cell_has_ci = r.n_repeats >= 3;
        let (itl_p50, itl_p99) = if let Some(itl) = &r.sse_text_event_gap_ms {
            (fmt(&itl.p50, cell_has_ci), fmt(&itl.p99, cell_has_ci))
        } else {
            let missing = missing_visible_itl(r);
            (missing.to_string(), missing.to_string())
        };
        writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} | not collected | not collected | not collected |",
            label,
            fmt(&r.ttft_ms.p50, cell_has_ci),
            fmt(&r.ttft_ms.p99, cell_has_ci),
            fmt(&r.tpot_ms.p50, cell_has_ci),
            fmt(&r.tpot_ms.p99, cell_has_ci),
            itl_p50,
            itl_p99,
            fmt(&r.output_throughput_tps, cell_has_ci),
        )
        .ok();
    }
    writeln!(s).ok();
    writeln!(
        s,
        "ITL measures client-visible intervals between nonempty content/reasoning updates, not proven per-token latency. Role-only, empty-content and finish-only events do not count as updates. Coalescing and event/usage count mismatches are disclosed and do not exclude otherwise complete successful streams. Failed requests remain in diagnostic counts. Reports without this measurement show not collected."
    )
    .ok();
    writeln!(s).ok();
    write_memory_scope_note(s);
    for r in reports {
        writeln!(s, "ITL diagnostics for {}:", sweep_cell_label(r)).ok();
        writeln!(s).ok();
        write_itl_diagnostics(s, r);
    }
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
        let records: Vec<RequestRecord> = records
            .into_iter()
            .map(|(success, ttft, e2e, in_tok, out_tok)| RequestRecord {
                benchmark_correlation: None,
                server_request_id: None,
                success,
                ttft_ms: ttft,
                e2e_ms: e2e,
                input_tokens: in_tok,
                server_input_tokens: None,
                output_tokens: out_tok,
                output_token_count_source: OutputTokenCountSource::StreamChunks,
                itl_evidence: crate::RequestItlEvidence::engine(success, out_tok, 0),
                quality_issues: Default::default(),
                itl_ms: vec![],
            })
            .collect();
        let expected_requests = u32::try_from(records.len()).unwrap();
        RunRecord {
            records,
            expected_requests,
            duration_s,
            warmup: Default::default(),
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
            0,
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
        assert!(md.contains("| ITL (visible text updates, ms) | not collected |"));
        assert!(md.contains("| ITL (strict token diagnostic, ms) | unavailable |"));
        // n_repeats=3 → CI columns ARE present
        assert!(!md.contains("⚠ < 3"));
        assert!(md.contains("±")); // mean ± ci95 format
    }

    #[test]
    fn http_sampling_is_shown_only_when_recorded() {
        let mut report = fixture_report();
        assert!(!render_single(&report).contains("http_request_sampling"));
        let sampling = crate::env::HttpRequestSampling {
            temperature: 0.6,
            top_k: Some(20),
            top_p: Some(0.95),
            repetition_penalty: Some(1.0),
            seed: Some(37),
        };
        report.env.http_request_sampling = Some(sampling);
        report.env_hash = report.env.hash();
        let markdown = render_single(&report);
        assert!(markdown.contains("http_request_sampling"));
        assert!(markdown.contains(&serde_json::to_string(&sampling).unwrap()));
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
        assert!(md.contains("| concurrency / cell |"));
        // Two data rows.
        let row_count = md.matches("| c=32 |").count();
        assert_eq!(row_count, 2);
    }

    fn eligible_itl_report() -> BenchReport {
        let mut run = make_run(
            vec![(true, 100.0, 140.0, 8, 3), (true, 200.0, 320.0, 24, 3)],
            1.0,
        );
        for (record, intervals) in run.records.iter_mut().zip([[10.0, 30.0], [40.0, 80.0]]) {
            record.itl_ms = intervals.to_vec();
            record.itl_evidence = crate::RequestItlEvidence::sse(true, 3, Some(3), 2, 0);
            record.output_token_count_source = OutputTokenCountSource::Usage;
        }
        compute_metrics(
            "tiny".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            8,
            3,
            0,
            Slo::unbounded(),
            vec![run],
            Env::default(),
        )
    }

    #[test]
    fn sweep_renders_measured_tail_latencies_and_missing_memory() {
        let report = eligible_itl_report();
        assert!(report.has_complete_itl_evidence());
        let md = render_sweep(&[report]);
        // These are computed from two requests with distinct TPOT and ITL tails.
        assert!(md.contains(
            "| c=2 | 150.00 | 199.00 | 40.00 | 59.60 | 35.00 | 78.80 | 6.00 | not collected | not collected | not collected |"
        ));
        assert!(md.contains("TPOT P99 (ms/token)"));
        assert!(md.contains("ITL (visible text updates) P99 (ms)"));
        assert!(md.contains("server-side evidence"));
        assert!(!md.contains("goodput"));
    }

    #[test]
    fn sweep_never_substitutes_strict_itl_for_missing_visible_text_measurements() {
        let engine_only = fixture_report();
        let mut missing = eligible_itl_report();
        assert!(missing.has_complete_itl_evidence());
        // Older reports may have valid strict ITL but no new visible-text metric.
        missing.sse_text_event_gap_ms = None;
        missing.sse_text_event_gap_evidence = None;
        for report in [engine_only, missing] {
            let single = render_single(&report);
            assert!(single.contains("| ITL (visible text updates, ms) | not collected |"));
            let md = render_sweep(&[report]);
            let row = md.lines().find(|line| line.starts_with("| c=")).unwrap();
            let fields: Vec<_> = row.split('|').map(str::trim).collect();
            assert_eq!(fields[6], "not collected");
            assert_eq!(fields[7], "not collected");
            assert_eq!(fields[9], "not collected");
            assert_eq!(fields[10], "not collected");
            assert_eq!(fields[11], "not collected");
        }
    }

    #[test]
    fn visible_text_itl_retains_stalls_despite_coalescing_and_usage_mismatch() {
        let mut run = make_run(vec![(true, 100.0, 5140.0, 8, 9)], 6.0);
        run.records[0].itl_ms = vec![40.0, 5000.0];
        run.records[0].itl_evidence = crate::RequestItlEvidence::sse(true, 3, Some(9), 2, 1);
        run.records[0].output_token_count_source = OutputTokenCountSource::Usage;
        let report = compute_metrics(
            "tiny".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            8,
            9,
            0,
            Slo::unbounded(),
            vec![run],
            Env::default(),
        );
        assert!(!report.has_complete_itl_evidence());
        let single = render_single(&report);
        let visible = single
            .lines()
            .find(|line| line.starts_with("| ITL (visible text updates, ms)"))
            .unwrap();
        assert!(visible.ends_with("| 4950.40 |"));
        assert!(single.contains("| ITL (strict token diagnostic, ms) | unavailable |"));
        assert!(single.contains("Transport coalescing: 1 requests / 1 chunks"));
        assert!(single.contains("event/usage count mismatches: 1"));
        let sweep = render_sweep(&[report]);
        let row = sweep
            .lines()
            .find(|line| line.starts_with("| c=1 |"))
            .unwrap();
        let fields: Vec<_> = row.split('|').map(str::trim).collect();
        assert_eq!(fields[6], "2520.00");
        assert_eq!(fields[7], "4950.40");
    }

    #[test]
    fn visible_text_itl_without_two_updates_is_unavailable_instead_of_zero() {
        let mut run = make_run(vec![(true, 100.0, 100.0, 8, 1)], 1.0);
        run.records[0].itl_evidence = crate::RequestItlEvidence::sse(true, 1, Some(1), 0, 0);
        run.records[0].output_token_count_source = OutputTokenCountSource::Usage;
        let report = compute_metrics(
            "tiny".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            8,
            1,
            0,
            Slo::unbounded(),
            vec![run],
            Env::default(),
        );
        assert!(report.sse_text_event_gap_evidence.is_some());
        assert!(report.sse_text_event_gap_ms.is_none());
        let single = render_single(&report);
        assert!(single.contains("| ITL (visible text updates, ms) | unavailable (no intervals) |"));
        let sweep = render_sweep(&[report]);
        let row = sweep
            .lines()
            .find(|line| line.starts_with("| c=1 |"))
            .unwrap();
        let fields: Vec<_> = row.split('|').map(str::trim).collect();
        assert_eq!(fields[6], "unavailable (no intervals)");
        assert_eq!(fields[7], "unavailable (no intervals)");
    }

    #[test]
    fn single_report_distinguishes_legacy_goodput_from_p99_slos() {
        let bounded = fixture_report();
        let md = render_single(&bounded);
        assert!(md.contains("**goodput (req/s)**"));
        assert!(md.contains("does not evaluate aggregate TTFT/TPOT/ITL P99 SLOs"));

        let mut unbounded = bounded;
        unbounded.slo = Slo::unbounded();
        assert!(!render_single(&unbounded).contains("**goodput (req/s)**"));
    }

    #[test]
    fn sharegpt_lengths_exclude_warmup_and_distinguish_output_budgets() {
        use crate::dataset::{
            ShareGptDatasetEvidence, ShareGptFilter, ShareGptSample, ShareGptSelection,
        };
        use crate::BenchmarkPhase;

        let mut report = eligible_itl_report();
        let original = render_single(&report);
        assert!(original.starts_with("# tiny — closed_loop · c=2\n"));
        assert!(!original.contains("## Dataset"));
        report.n_prompt = 0;
        report.n_gen = 0;
        let sample = |phase, request_index, input_tokens, requested_output_tokens| ShareGptSample {
            source_record_index: u64::from(request_index),
            original_id: None,
            phase,
            request_index,
            prompt_sha256: "a".repeat(64),
            assistant_sha256: "b".repeat(64),
            input_tokens,
            reference_output_tokens: requested_output_tokens,
            requested_output_tokens,
        };
        report.dataset_evidence = Some(ShareGptDatasetEvidence {
            dataset: "sharegpt".into(),
            source_path: "conversations.json".into(),
            source_sha256: "c".repeat(64),
            source_format: "json".into(),
            tokenizer_sha256: "d".repeat(64),
            filter: ShareGptFilter {
                min_input_tokens: 1,
                max_input_tokens: None,
                min_output_tokens: 1,
                max_output_tokens: None,
                max_total_tokens: None,
                chat_template_reserve_tokens: 0,
                fixed_output_tokens: None,
            },
            counts: Default::default(),
            prompt_seed: 37,
            sampling: "without_replacement".into(),
            ignore_eos: false,
            enable_thinking: None,
            repeats: vec![ShareGptSelection {
                repeat_index: 0,
                rng_seed: 37,
                selection_sha256: "e".repeat(64),
                samples: vec![
                    sample(BenchmarkPhase::Warmup, 0, 9999, 9999),
                    sample(BenchmarkPhase::Measured, 0, 8, 3),
                    sample(BenchmarkPhase::Measured, 1, 24, 7),
                ],
            }],
        });
        let md = render_single(&report);
        assert!(md.starts_with("# tiny — closed_loop · c=2 · ShareGPT variable\n"));
        assert!(md.contains("| measured samples | 2 |"));
        assert!(md.contains(
            "| prompt tokens (client tokenizer, before chat template) | min 8, max 24, mean 16.00 |"
        ));
        assert!(md.contains("| requested output budget (tokens) | min 3, max 7, mean 5.00 |"));
        assert!(md.contains("Requested output budgets are not actual generated lengths"));
        assert!(md.contains(&format!("| source SHA-256 | `{}` |", "c".repeat(64))));
        assert!(md.contains("prompt seed 37"));
        assert!(!md.contains("9999"));
    }
}
