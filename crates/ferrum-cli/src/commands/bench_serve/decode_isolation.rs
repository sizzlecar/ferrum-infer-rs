use super::{build_env, detect_features, BenchServeCommand, RunContext};
use clap::{Args, ValueEnum};
use ferrum_bench_core::decode_isolation::{
    aggregate_decode_isolation_runs, DecodeIsolationCapabilities, DecodeIsolationConfig,
    DecodeIsolationErrorPolicy, DecodeIsolationReport, DecodeIsolationScenarioContract,
    DecodeIsolationWindowEnd,
};
use ferrum_bench_core::Scenario;
use ferrum_types::Result;
use serde::Deserialize;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::time::Duration;

mod runner;

const KV_HEADROOM_NUMERATOR: u32 = 9;
const KV_HEADROOM_DENOMINATOR: u32 = 10;
const AGGRESSOR_OUTPUT_TOKENS: u64 = 1;
const MINIMUM_AGGRESSOR_SCHEDULED_CHUNKS: u32 = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BenchServeWorkload {
    #[default]
    Standard,
    DecodeIsolation,
}

#[derive(Args, Clone, Debug)]
pub struct DecodeIsolationArgs {
    /// Number of live decoders. Defaults to /health capacity minus two and may
    /// decrease only when required by the advertised aggregate KV budget.
    #[arg(long)]
    pub decode_isolation_incumbents: Option<u32>,

    /// Long-prefill input tokens. Defaults to the smaller of eight scheduler
    /// chunks and 75% of the effective per-request limit, then is bounded by
    /// aggregate KV headroom while retaining one incumbent.
    #[arg(long)]
    pub decode_isolation_prefill_tokens: Option<usize>,

    /// Output events each incumbent must emit before aggressor injection.
    #[arg(long, default_value_t = 4)]
    pub decode_isolation_baseline_events: u32,
}

impl Default for DecodeIsolationArgs {
    fn default() -> Self {
        Self {
            decode_isolation_incumbents: None,
            decode_isolation_prefill_tokens: None,
            decode_isolation_baseline_events: 4,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HealthResponse {
    #[serde(default)]
    admission: HealthAdmission,
    #[serde(default)]
    auto_config: HealthAutoConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HealthAdmission {
    effective_max_concurrent: Option<u32>,
    preflight_effective_max_concurrent: Option<u32>,
    maximum_active_sequences: Option<u32>,
    maximum_scheduled_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HealthAutoConfig {
    selected_max_model_len: Option<u64>,
    selected_kv_capacity: Option<u64>,
    #[serde(default)]
    admission: HealthAutoAdmission,
    #[serde(default)]
    model_capabilities: HealthModelCapabilities,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HealthModelCapabilities {
    max_context_len: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HealthAutoAdmission {
    effective_max_concurrent: Option<u32>,
    maximum_active_sequences: Option<u32>,
    maximum_scheduled_tokens: Option<u64>,
    max_batched_tokens: Option<u64>,
    max_model_length: Option<u64>,
    kv_capacity_tokens: Option<u64>,
    kv_block_size_tokens: Option<u64>,
}

pub(super) async fn execute(cmd: &BenchServeCommand, ctx: &RunContext) -> Result<()> {
    let capabilities = fetch_capabilities(&ctx.client, &ctx.base_url, cmd.timeout).await?;
    let config = resolve_config(cmd, capabilities)?;
    let tokenizer_path = cmd.tokenizer.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        ferrum_types::FerrumError::model(format!(
            "Load tokenizer at {}: {error}",
            tokenizer_path.display()
        ))
    })?;

    // Prompt construction can be expensive at long context. Finish every
    // prompt before any request so generation cannot enter measured time.
    let prepared = runner::prepare_runs(cmd, &config, &tokenizer)?;
    let mut runs = Vec::with_capacity(prepared.len());
    for run in prepared {
        eprintln!(
            "  decode-isolation repeat {}/{}: {} incumbents, {}-token aggressor",
            run.repeat + 1,
            cmd.n_repeats,
            config.incumbents,
            config.aggressor_input_tokens
        );
        runs.push(runner::run_once(cmd, ctx, &config, run).await);
    }

    let env = build_env(cmd, detect_features());
    let backend = cmd
        .target_backend
        .expect("decode-isolation validation requires target backend")
        .as_str()
        .to_string();
    let all_evidence_valid = runs.iter().all(|run| run.validity.all_valid);
    let aggregate = aggregate_decode_isolation_runs(&runs);
    let model = match &cmd.tag {
        Some(tag) => format!("{}#{tag}", cmd.model),
        None => cmd.model.clone(),
    };
    let report = DecodeIsolationReport {
        schema_version: 2,
        scenario: Scenario::DecodeIsolation,
        model,
        backend,
        n_repeats: cmd.n_repeats,
        config,
        aggregate,
        runs,
        all_evidence_valid,
        semantic_correctness_evaluated: false,
        env_hash: env.hash(),
        env,
    };
    emit_report(cmd, &report)?;

    if report.runs.iter().any(|run| !run.validity.warmup_valid) {
        return Err(ferrum_types::FerrumError::model(
            "decode-isolation warmup failed; diagnostics were emitted without numeric evidence",
        ));
    }
    if cmd.fail_on_error && !report.all_evidence_valid {
        return Err(ferrum_types::FerrumError::model(
            "decode-isolation evidence validation failed; diagnostics were emitted without numeric evidence",
        ));
    }
    if let Some(max_error_rate) = cmd.max_error_rate {
        let attempted_per_run = u64::from(report.config.incumbents) + 1;
        let total = attempted_per_run.saturating_mul(report.runs.len() as u64);
        let errored = report.runs.iter().fold(0_u64, |errored, run| {
            let missing_incumbents = usize::try_from(report.config.incumbents)
                .unwrap_or(usize::MAX)
                .saturating_sub(run.incumbents.len());
            errored.saturating_add(
                run.incumbents
                    .iter()
                    .filter(|request| !request.contract_valid)
                    .count() as u64
                    + missing_incumbents as u64
                    + u64::from(!run.aggressor.contract_valid),
            )
        });
        let error_rate = if total == 0 {
            1.0
        } else {
            errored as f64 / total as f64
        };
        if error_rate > max_error_rate {
            return Err(ferrum_types::FerrumError::model(format!(
                "decode-isolation error rate {error_rate:.4} exceeds max {max_error_rate:.4}"
            )));
        }
    }
    Ok(())
}

async fn fetch_capabilities(
    client: &reqwest::Client,
    base_url: &str,
    timeout_seconds: f64,
) -> Result<DecodeIsolationCapabilities> {
    let response = client
        .get(format!("{base_url}/health"))
        .timeout(Duration::from_secs_f64(timeout_seconds))
        .send()
        .await
        .map_err(|error| {
            ferrum_types::FerrumError::model(format!(
                "decode-isolation requires readable /health capabilities: {error}"
            ))
        })?;
    if !response.status().is_success() {
        return Err(ferrum_types::FerrumError::model(format!(
            "decode-isolation /health returned HTTP {}",
            response.status()
        )));
    }
    let health: HealthResponse = response.json().await.map_err(|error| {
        ferrum_types::FerrumError::model(format!("decode-isolation invalid /health JSON: {error}"))
    })?;
    resolve_capabilities(health)
}

fn resolve_capabilities(health: HealthResponse) -> Result<DecodeIsolationCapabilities> {
    let missing = |field: &str| {
        ferrum_types::FerrumError::model(format!(
            "decode-isolation /health omitted effective {field} capability"
        ))
    };
    let auto = health.auto_config.admission;
    let effective_max_concurrent = [
        health.admission.maximum_active_sequences,
        health.admission.effective_max_concurrent,
        health.admission.preflight_effective_max_concurrent,
        auto.maximum_active_sequences,
        auto.effective_max_concurrent,
    ]
    .into_iter()
    .flatten()
    .min()
    .ok_or_else(|| missing("maximum active sequences"))?;
    let maximum_scheduled_tokens = [
        health.admission.maximum_scheduled_tokens,
        auto.maximum_scheduled_tokens,
        auto.max_batched_tokens,
    ]
    .into_iter()
    .flatten()
    .min()
    .ok_or_else(|| missing("maximum scheduled tokens"))?;
    let kv_capacity_tokens = auto
        .kv_capacity_tokens
        .ok_or_else(|| missing("KV capacity tokens"))?;
    let (kv_block_size_tokens, kv_block_size_source) = match auto.kv_block_size_tokens {
        Some(value) => (value, "health_auto_config_admission".to_string()),
        None => (1, "default_unit_block".to_string()),
    };
    // Configured model limits may intentionally be null under auto-config.
    // The effective per-request ceiling is the tightest advertised non-null
    // model/config limit, always bounded by physical KV capacity.
    let max_model_length = [
        health.auto_config.selected_max_model_len,
        health.auto_config.selected_kv_capacity,
        auto.max_model_length,
        health.auto_config.model_capabilities.max_context_len,
        Some(kv_capacity_tokens),
    ]
    .into_iter()
    .flatten()
    .min()
    .ok_or_else(|| missing("maximum model length"))?;
    if effective_max_concurrent == 0
        || maximum_scheduled_tokens == 0
        || max_model_length == 0
        || kv_capacity_tokens == 0
        || kv_block_size_tokens == 0
    {
        return Err(ferrum_types::FerrumError::model(
            "decode-isolation /health capabilities must all be greater than zero",
        ));
    }
    Ok(DecodeIsolationCapabilities {
        effective_max_concurrent,
        maximum_scheduled_tokens,
        max_model_length,
        kv_capacity_tokens,
        selected_kv_capacity_tokens: health.auto_config.selected_kv_capacity,
        kv_block_size_tokens,
        kv_block_size_source,
    })
}

fn resolve_config(
    cmd: &BenchServeCommand,
    capabilities: DecodeIsolationCapabilities,
) -> Result<DecodeIsolationConfig> {
    if cmd.decode_isolation.decode_isolation_baseline_events < 2 {
        return Err(ferrum_types::FerrumError::model(
            "--decode-isolation-baseline-events must be >= 2",
        ));
    }
    let incumbent_input_tokens = u64::try_from(cmd.random_input_len)
        .map_err(|_| ferrum_types::FerrumError::model("--random-input-len is too large"))?;
    let incumbent_output_tokens = u64::try_from(cmd.random_output_len)
        .map_err(|_| ferrum_types::FerrumError::model("--random-output-len is too large"))?;
    if incumbent_output_tokens <= u64::from(cmd.decode_isolation.decode_isolation_baseline_events) {
        return Err(ferrum_types::FerrumError::model(
            "--random-output-len must exceed --decode-isolation-baseline-events",
        ));
    }
    let incumbent_footprint = incumbent_input_tokens
        .checked_add(incumbent_output_tokens)
        .ok_or_else(|| ferrum_types::FerrumError::model("incumbent token budget overflow"))?;
    if incumbent_footprint > capabilities.max_model_length {
        return Err(ferrum_types::FerrumError::model(
            "incumbent input plus output exceeds /health maximum model length",
        ));
    }

    let block = capabilities.kv_block_size_tokens;
    let aggregate_kv_budget_blocks = (capabilities.kv_capacity_tokens / block)
        .checked_mul(u64::from(KV_HEADROOM_NUMERATOR))
        .ok_or_else(|| ferrum_types::FerrumError::model("KV headroom calculation overflow"))?
        / u64::from(KV_HEADROOM_DENOMINATOR);
    let aggregate_kv_budget_tokens = aggregate_kv_budget_blocks
        .checked_mul(block)
        .ok_or_else(|| ferrum_types::FerrumError::model("KV headroom calculation overflow"))?;
    let requested_incumbents = cmd.decode_isolation.decode_isolation_incumbents;
    let concurrency_incumbent_limit = capabilities.effective_max_concurrent.saturating_sub(2);
    if concurrency_incumbent_limit == 0 {
        return Err(ferrum_types::FerrumError::model(
            "decode-isolation needs /health effective concurrency >= 3",
        ));
    }
    if requested_incumbents.is_some_and(|value| value == 0) {
        return Err(ferrum_types::FerrumError::model(
            "--decode-isolation-incumbents must be greater than zero",
        ));
    }
    if requested_incumbents.is_some_and(|value| value > concurrency_incumbent_limit) {
        return Err(ferrum_types::FerrumError::model(
            "--decode-isolation-incumbents exceeds the /health N-2 isolation limit",
        ));
    }

    let requested_aggressor = cmd
        .decode_isolation
        .decode_isolation_prefill_tokens
        .map(u64::try_from)
        .transpose()
        .map_err(|_| {
            ferrum_types::FerrumError::model("--decode-isolation-prefill-tokens is too large")
        })?;
    let per_request_aggressor_max = capabilities
        .max_model_length
        .checked_sub(AGGRESSOR_OUTPUT_TOKENS)
        .ok_or_else(|| ferrum_types::FerrumError::model("maximum model length is too small"))?;
    let preferred_aggressor = capabilities
        .maximum_scheduled_tokens
        .checked_mul(8)
        .ok_or_else(|| ferrum_types::FerrumError::model("aggressor target overflow"))?
        .min(capabilities.max_model_length.saturating_mul(3) / 4);
    let minimum_aggressor = capabilities
        .maximum_scheduled_tokens
        .checked_mul(u64::from(MINIMUM_AGGRESSOR_SCHEDULED_CHUNKS))
        .ok_or_else(|| {
            ferrum_types::FerrumError::model("minimum aggressor chunk calculation overflow")
        })?;

    let incumbent_allocation = round_up_to_block(incumbent_footprint, block)?;
    let incumbent_blocks = incumbent_allocation / block;
    let reserved_incumbent_blocks = u64::from(requested_incumbents.unwrap_or(1))
        .checked_mul(incumbent_blocks)
        .ok_or_else(|| ferrum_types::FerrumError::model("reserved incumbent KV overflow"))?;
    let max_aggressor_blocks = aggregate_kv_budget_blocks
        .checked_sub(reserved_incumbent_blocks)
        .ok_or_else(|| infeasible_shape(&capabilities))?;
    let max_aggressor_input_by_kv = max_aggressor_blocks
        .checked_mul(block)
        .and_then(|tokens| tokens.checked_sub(AGGRESSOR_OUTPUT_TOKENS))
        .ok_or_else(|| infeasible_shape(&capabilities))?;
    let aggressor_input_tokens = requested_aggressor.unwrap_or_else(|| {
        preferred_aggressor
            .min(per_request_aggressor_max)
            .min(max_aggressor_input_by_kv)
    });
    if aggressor_input_tokens < minimum_aggressor
        || aggressor_input_tokens > per_request_aggressor_max
        || aggressor_input_tokens > max_aggressor_input_by_kv
    {
        return Err(infeasible_shape(&capabilities));
    }
    let aggressor_allocation = round_up_to_block(
        aggressor_input_tokens
            .checked_add(AGGRESSOR_OUTPUT_TOKENS)
            .ok_or_else(|| ferrum_types::FerrumError::model("aggressor token budget overflow"))?,
        block,
    )?;
    let aggressor_blocks = aggressor_allocation / block;
    let kv_incumbent_limit = aggregate_kv_budget_blocks
        .checked_sub(aggressor_blocks)
        .map(|remaining| remaining / incumbent_blocks)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    let feasible_incumbents = concurrency_incumbent_limit.min(kv_incumbent_limit);
    let incumbents = requested_incumbents.unwrap_or(feasible_incumbents);
    if incumbents == 0 || incumbents > feasible_incumbents {
        return Err(infeasible_shape(&capabilities));
    }
    let incumbent_unrounded_total = u64::from(incumbents)
        .checked_mul(incumbent_footprint)
        .ok_or_else(|| ferrum_types::FerrumError::model("aggregate incumbent KV overflow"))?;
    let incumbent_allocation_total = u64::from(incumbents)
        .checked_mul(incumbent_allocation)
        .ok_or_else(|| ferrum_types::FerrumError::model("aggregate incumbent KV overflow"))?;
    let estimated_unrounded_aggregate_kv_tokens = incumbent_unrounded_total
        .checked_add(aggressor_input_tokens)
        .and_then(|value| value.checked_add(AGGRESSOR_OUTPUT_TOKENS))
        .ok_or_else(|| ferrum_types::FerrumError::model("aggregate KV estimate overflow"))?;
    let estimated_aggregate_kv_tokens = incumbent_allocation_total
        .checked_add(aggressor_allocation)
        .ok_or_else(|| ferrum_types::FerrumError::model("aggregate KV allocation overflow"))?;
    let estimated_aggregate_kv_blocks = estimated_aggregate_kv_tokens / block;

    Ok(DecodeIsolationConfig {
        incumbents,
        incumbents_source: if requested_incumbents.is_some() {
            "cli".to_string()
        } else if incumbents == concurrency_incumbent_limit {
            "health_effective_max_concurrent_minus_two".to_string()
        } else {
            "health_concurrency_reduced_for_kv_headroom".to_string()
        },
        incumbent_input_tokens,
        incumbent_output_tokens,
        aggressor_input_tokens,
        aggressor_input_source: if requested_aggressor.is_some() {
            "cli".to_string()
        } else {
            "health_model_and_kv_capacity".to_string()
        },
        aggressor_scheduled_chunks: aggressor_input_tokens
            .div_ceil(capabilities.maximum_scheduled_tokens),
        aggressor_output_tokens: AGGRESSOR_OUTPUT_TOKENS,
        aggregate_kv_budget_tokens,
        aggregate_kv_budget_blocks,
        estimated_unrounded_aggregate_kv_tokens,
        estimated_aggregate_kv_tokens,
        estimated_aggregate_kv_blocks,
        capabilities,
        contract: DecodeIsolationScenarioContract {
            baseline_output_events_per_incumbent: cmd
                .decode_isolation
                .decode_isolation_baseline_events,
            fixed_output_budget: true,
            injection_requires_all_incumbents_ready: true,
            interference_window_end: DecodeIsolationWindowEnd::AggressorFirstOutputEvent,
            post_aggressor_observable_progress_required_per_incumbent: true,
            minimum_aggressor_scheduled_chunks: MINIMUM_AGGRESSOR_SCHEDULED_CHUNKS,
            kv_capacity_headroom_numerator: KV_HEADROOM_NUMERATOR,
            kv_capacity_headroom_denominator: KV_HEADROOM_DENOMINATOR,
            invalid_evidence_policy: if cmd.fail_on_error {
                DecodeIsolationErrorPolicy::EmitDiagnosticsAndFail
            } else {
                DecodeIsolationErrorPolicy::EmitDiagnostics
            },
            warmup_failure_always_fatal: true,
            measured_error_rate_limit: cmd.max_error_rate,
        },
    })
}

fn round_up_to_block(tokens: u64, block: u64) -> Result<u64> {
    tokens
        .div_ceil(block)
        .checked_mul(block)
        .ok_or_else(|| ferrum_types::FerrumError::model("KV block rounding overflow"))
}

fn infeasible_shape(capabilities: &DecodeIsolationCapabilities) -> ferrum_types::FerrumError {
    ferrum_types::FerrumError::model(format!(
        "decode-isolation shape is infeasible: aggressor must span at least {} scheduler chunks of at most {} tokens and one incumbent must fit within {}% block-rounded KV headroom",
        MINIMUM_AGGRESSOR_SCHEDULED_CHUNKS,
        capabilities.maximum_scheduled_tokens,
        KV_HEADROOM_NUMERATOR * 100 / KV_HEADROOM_DENOMINATOR
    ))
}

fn emit_report(cmd: &BenchServeCommand, report: &DecodeIsolationReport) -> Result<()> {
    match cmd.output.as_str() {
        "json" => write_output(
            cmd,
            &serde_json::to_string_pretty(report).expect("serialize decode-isolation report"),
            false,
        ),
        "jsonl" => {
            if cmd.out.is_none() {
                return Err(ferrum_types::FerrumError::model(
                    "--output jsonl requires --out PATH (append-mode log)",
                ));
            }
            write_output(
                cmd,
                &serde_json::to_string(report).expect("serialize decode-isolation report"),
                true,
            )
        }
        "md" => write_output(cmd, &render_markdown(report), false),
        other => Err(ferrum_types::FerrumError::model(format!(
            "unknown --output '{other}': allowed values are json, jsonl, md"
        ))),
    }
}

fn write_output(cmd: &BenchServeCommand, content: &str, append: bool) -> Result<()> {
    if let Some(path) = &cmd.out {
        let mut options = OpenOptions::new();
        options.create(true).write(true);
        if append {
            options.append(true);
        } else {
            options.truncate(true);
        }
        let mut file = options.open(path).map_err(|error| {
            ferrum_types::FerrumError::model(format!("write {}: {error}", path.display()))
        })?;
        writeln!(file, "{content}").map_err(|error| {
            ferrum_types::FerrumError::model(format!("write {}: {error}", path.display()))
        })?;
        eprintln!("\n→ wrote {}", path.display());
    } else {
        println!("{content}");
    }
    Ok(())
}

fn render_markdown(report: &DecodeIsolationReport) -> String {
    let mut output = format!(
        "# {} — decode isolation\n\nincumbents: {} · aggressor input: {} tokens\n\n",
        report.model, report.config.incumbents, report.config.aggressor_input_tokens
    );
    output.push_str("The output-event gap is measured between user-visible SSE text events; it is not token-level latency.\n\n");
    output.push_str("| repeat | baseline event-gap p50/p95 ms | interference event-gap p50/p95 ms | max event gap ms | observable output progress | time to first output event ms | valid |\n");
    output.push_str("|---:|---:|---:|---:|---:|---:|:---:|\n");
    for run in &report.runs {
        if let (true, Some(metrics), Some(first_event_ms)) = (
            run.validity.all_valid,
            &run.metrics,
            run.aggressor_time_to_first_output_event_ms,
        ) {
            output.push_str(&format!(
                "| {} | {:.2}/{:.2} | {:.2}/{:.2} | {:.2} | {} | {:.2} | yes |\n",
                run.repeat + 1,
                metrics.baseline_output_event_gap.p50_ms,
                metrics.baseline_output_event_gap.p95_ms,
                metrics.interference_output_event_gap.p50_ms,
                metrics.interference_output_event_gap.p95_ms,
                metrics.maximum_output_event_gap_ms,
                metrics.observable_output_progress_events,
                first_event_ms,
            ));
        } else {
            output.push_str(&format!(
                "| {} | — | — | — | — | — | no: {} |\n",
                run.repeat + 1,
                run.validity.invalid_reasons.join("; ")
            ));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> DecodeIsolationCapabilities {
        DecodeIsolationCapabilities {
            effective_max_concurrent: 10,
            maximum_scheduled_tokens: 512,
            max_model_length: 8192,
            kv_capacity_tokens: 65536,
            selected_kv_capacity_tokens: Some(8192),
            kv_block_size_tokens: 16,
            kv_block_size_source: "test".to_string(),
        }
    }

    fn command() -> BenchServeCommand {
        let mut cmd = super::super::tests::test_command();
        cmd.scenario = BenchServeWorkload::DecodeIsolation;
        cmd.random_input_len = 128;
        cmd.random_output_len = 16;
        cmd
    }

    #[test]
    fn markdown_names_output_event_evidence_without_token_latency_claims() {
        let env = ferrum_bench_core::Env::default();
        let report = DecodeIsolationReport {
            schema_version: 2,
            scenario: Scenario::DecodeIsolation,
            model: "test-model".to_string(),
            backend: "cpu".to_string(),
            n_repeats: 0,
            config: resolve_config(&command(), capabilities()).unwrap(),
            aggregate: None,
            runs: vec![],
            all_evidence_valid: false,
            semantic_correctness_evaluated: false,
            env_hash: env.hash(),
            env,
        };
        let markdown = render_markdown(&report);
        assert!(markdown.contains("output-event gap"));
        assert!(markdown.contains("observable output progress"));
        assert!(markdown.contains("time to first output event"));
        assert!(!markdown.contains("ITL"));
        assert!(!markdown.contains("TTFT"));
    }

    #[test]
    fn health_shape_reads_effective_runtime_caps() {
        let health: HealthResponse = serde_json::from_value(serde_json::json!({
            "admission": {
                "effective_max_concurrent": 12,
                "preflight_effective_max_concurrent": 8,
                "maximum_active_sequences": 10,
                "maximum_scheduled_tokens": 1024
            },
            "auto_config": {
                "selected_max_model_len": 16384,
                "admission": {
                    "max_model_length": 8192,
                    "kv_capacity_tokens": 32768,
                    "kv_block_size_tokens": 16
                },
                "model_capabilities": {"max_context_len": 12288}
            }
        }))
        .unwrap();
        let caps = resolve_capabilities(health).unwrap();
        assert_eq!(caps.effective_max_concurrent, 8);
        assert_eq!(caps.maximum_scheduled_tokens, 1024);
        assert_eq!(caps.max_model_length, 8192);
        assert_eq!(caps.kv_capacity_tokens, 32768);
        assert_eq!(caps.kv_block_size_tokens, 16);
    }

    #[test]
    fn missing_kv_capability_is_rejected() {
        let health: HealthResponse = serde_json::from_value(serde_json::json!({
            "admission": {"effective_max_concurrent": 4, "maximum_scheduled_tokens": 64},
            "auto_config": {"selected_max_model_len": 1024, "admission": {}}
        }))
        .unwrap();
        assert!(resolve_capabilities(health).is_err());
    }

    #[test]
    fn null_configured_model_caps_fall_back_to_model_metadata_and_kv() {
        let health: HealthResponse = serde_json::from_value(serde_json::json!({
            "admission": {"effective_max_concurrent": 4, "maximum_scheduled_tokens": 64},
            "auto_config": {
                "selected_max_model_len": null,
                "admission": {"max_model_length": null, "kv_capacity_tokens": 8192},
                "model_capabilities": {"max_context_len": 32768}
            }
        }))
        .unwrap();
        assert_eq!(resolve_capabilities(health).unwrap().max_model_length, 8192);
        assert_eq!(
            resolve_capabilities(
                serde_json::from_value(serde_json::json!({
                    "admission": {"effective_max_concurrent": 4, "maximum_scheduled_tokens": 64},
                    "auto_config": {
                        "admission": {"kv_capacity_tokens": 8192},
                        "model_capabilities": {"max_context_len": 32768}
                    }
                }))
                .unwrap()
            )
            .unwrap()
            .kv_block_size_source,
            "default_unit_block"
        );
    }

    #[test]
    fn jointly_derives_n_minus_two_and_prefill() {
        let config = resolve_config(&command(), capabilities()).unwrap();
        assert_eq!(config.incumbents, 8);
        assert_eq!(config.aggressor_input_tokens, 4096);
        assert!(config.aggressor_scheduled_chunks >= 4);
        assert!(config.estimated_aggregate_kv_tokens <= config.aggregate_kv_budget_tokens);
    }

    #[test]
    fn default_reduces_incumbents_to_make_prefill_feasible() {
        let mut caps = capabilities();
        caps.kv_capacity_tokens = 2500;
        let config = resolve_config(&command(), caps).unwrap();
        assert!(config.incumbents < 8);
        assert!(config.aggressor_scheduled_chunks >= 4);
        assert!(config.estimated_aggregate_kv_tokens <= config.aggregate_kv_budget_tokens);
    }

    #[test]
    fn derives_expected_4050_fill_first_shape_with_block_rounding() {
        let mut cmd = command();
        cmd.random_input_len = 256;
        cmd.random_output_len = 128;
        let config = resolve_config(
            &cmd,
            DecodeIsolationCapabilities {
                effective_max_concurrent: 16,
                maximum_scheduled_tokens: 192,
                max_model_length: 4096,
                kv_capacity_tokens: 4096,
                selected_kv_capacity_tokens: Some(4096),
                kv_block_size_tokens: 16,
                kv_block_size_source: "health_auto_config_admission".to_string(),
            },
        )
        .unwrap();
        assert_eq!(config.aggressor_input_tokens, 1536);
        assert_eq!(config.aggressor_scheduled_chunks, 8);
        assert_eq!(config.incumbents, 5);
        assert_eq!(config.aggregate_kv_budget_blocks, 230);
        assert_eq!(config.estimated_aggregate_kv_blocks, 217);
        assert_eq!(config.estimated_aggregate_kv_tokens, 3472);
        assert_eq!(4096 - config.estimated_aggregate_kv_tokens, 624);
    }

    #[test]
    fn cli_overrides_are_validated_jointly() {
        let mut cmd = command();
        cmd.decode_isolation.decode_isolation_incumbents = Some(3);
        cmd.decode_isolation.decode_isolation_prefill_tokens = Some(2048);
        let config = resolve_config(&cmd, capabilities()).unwrap();
        assert_eq!(config.incumbents, 3);
        assert_eq!(config.aggressor_input_tokens, 2048);

        cmd.decode_isolation.decode_isolation_prefill_tokens = Some(1536);
        assert!(resolve_config(&cmd, capabilities()).is_err());
        cmd.decode_isolation.decode_isolation_prefill_tokens = Some(2048);
        cmd.decode_isolation.decode_isolation_incumbents = Some(10);
        assert!(resolve_config(&cmd, capabilities()).is_err());
    }
}
