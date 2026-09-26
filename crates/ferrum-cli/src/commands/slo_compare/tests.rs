use super::*;
use clap::{Parser, Subcommand};
use ferrum_bench_core::{
    dataset::{
        ShareGptCounts, ShareGptDatasetEvidence, ShareGptFilter, ShareGptSample, ShareGptSelection,
    },
    env::HttpRequestSampling,
    slo::{SloEvaluationConfig, TpotBoundary},
    slo_comparison::{
        artifact::{ArtifactFileRef, ComparisonArtifactManifest},
        CellScope, ComparisonMetric, FixedServerCapacity, FrozenCell, FrozenComparisonContract,
        FrozenMemoryPolicy, FrozenPair, FrozenServerIdentity, SampledMemoryPolicy,
        SharedExecutionIdentity,
    },
    BenchmarkPhase,
};
use ferrum_types::SloAttainmentTargets;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

mod capacity;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn frozen_contract() -> FrozenComparisonContract {
    let samples = vec![ShareGptSample {
        source_record_index: 0,
        original_id: None,
        phase: BenchmarkPhase::Measured,
        request_index: 0,
        prompt_sha256: hash(b"prompt"),
        assistant_sha256: hash(b"assistant"),
        input_tokens: 1,
        reference_output_tokens: 2,
        requested_output_tokens: 2,
    }];
    let selection = ShareGptSelection {
        repeat_index: 0,
        rng_seed: 5,
        selection_sha256: hash(&serde_json::to_vec(&samples).unwrap()),
        samples,
    };
    let server = |name: &str| FrozenServerIdentity {
        implementation: name.into(),
        backend: "fixture".into(),
        request_model_alias: "model".into(),
        binary_sha256: hash(name.as_bytes()),
        effective_configuration_sha256: hash(b"server-config"),
        numerical_policy: "declared".into(),
        intentional_differences: Vec::new(),
    };
    let memory = SampledMemoryPolicy {
        window: "pre-load through shutdown".into(),
        interval_ms: 250,
        max_sample_gap_ns: 500_000_000,
    };
    FrozenComparisonContract {
        schema_version: 1,
        frozen_unix_ns: 1,
        cells: vec![FrozenCell {
            concurrency: 2,
            scope: CellScope::Primary,
        }],
        pairs: vec![FrozenPair {
            pair_id: "pair".into(),
            selection,
        }],
        shared: SharedExecutionIdentity {
            hardware_fingerprint_sha256: hash(b"hardware"),
            hardware_label: "fixture".into(),
            model_content_sha256: hash(b"model"),
            weight_precision: "F16".into(),
            kv_precision: "F16".into(),
            tokenizer_sha256: hash(b"tokenizer"),
            chat_template_sha256: hash(b"template"),
            client_binary_sha256: hash(b"client"),
            client_slo_config_sha256: hash(b"slo"),
        },
        baseline: server("baseline"),
        candidate: server("candidate"),
        capacity: FixedServerCapacity {
            slots: 2,
            context_tokens_per_request: 32,
            batch_tokens: 16,
        },
        dataset: ShareGptDatasetEvidence {
            dataset: "sharegpt".into(),
            source_path: "fixture.json".into(),
            source_sha256: hash(b"dataset"),
            source_format: "json_array".into(),
            tokenizer_sha256: hash(b"tokenizer"),
            filter: ShareGptFilter {
                min_input_tokens: 1,
                max_input_tokens: None,
                min_output_tokens: 1,
                max_output_tokens: None,
                max_total_tokens: Some(32),
                chat_template_reserve_tokens: 1,
                fixed_output_tokens: None,
            },
            counts: ShareGptCounts {
                records: 1,
                eligible: 1,
                ..Default::default()
            },
            prompt_seed: 5,
            sampling: "without replacement".into(),
            ignore_eos: true,
            enable_thinking: Some(false),
            repeats: Vec::new(),
        },
        http_connection_mode: "fresh".into(),
        sampling: HttpRequestSampling::default(),
        memory: FrozenMemoryPolicy {
            device_allocation: memory.clone(),
            os_footprint: memory.clone().into(),
            maximum_rss_window: memory.window,
        },
        slo: SloEvaluationConfig {
            ttft_ms: 100.0,
            tpot_ms: 10.0,
            visible_itl_ms: 10.0,
            attainment: SloAttainmentTargets::default(),
            tpot_boundary: TpotBoundary::LastVisibleOutput,
        },
        ratio_limits: ComparisonMetric::ALL
            .into_iter()
            .map(|metric| {
                (
                    metric,
                    if metric == ComparisonMetric::SuccessfulUsageOutputTps {
                        0.8
                    } else {
                        0.7
                    },
                )
            })
            .collect(),
        uncertainty: None,
    }
}

fn fixture() -> (TempDir, SloCompareCommand) {
    let dir = tempfile::tempdir().unwrap();
    let contract = serde_json::to_vec(&frozen_contract()).unwrap();
    fs::write(dir.path().join("contract.json"), &contract).unwrap();
    let manifest = ComparisonArtifactManifest {
        schema_version: 1,
        eligibility: None,
        contract: ArtifactFileRef {
            path: "contract.json".into(),
            sha256: hash(&contract),
            bytes: contract.len() as u64,
        },
        cells: Vec::new(),
    };
    let manifest_path = dir.path().join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let cmd = SloCompareCommand {
        manifest: manifest_path,
        limits_config: None,
        out: dir.path().join("comparison.json"),
        markdown_en: dir.path().join("comparison.en.md"),
        markdown_zh: dir.path().join("comparison.zh.md"),
    };
    (dir, cmd)
}

#[derive(Parser)]
struct TestCli {
    #[command(subcommand)]
    command: TestCommand,
}
#[derive(Subcommand)]
enum TestCommand {
    SloCompare(SloCompareCommand),
}

#[test]
fn cli_requires_the_manifest_and_three_explicit_output_files() {
    assert!(TestCli::try_parse_from(["ferrum", "slo-compare", "manifest.json"]).is_err());
    let parsed = TestCli::try_parse_from([
        "ferrum",
        "slo-compare",
        "manifest.json",
        "--out",
        "comparison.json",
        "--markdown-en",
        "en.md",
        "--markdown-zh",
        "zh.md",
    ])
    .unwrap();
    let TestCommand::SloCompare(cmd) = parsed.command;
    assert_eq!(cmd.manifest, PathBuf::from("manifest.json"));
    assert_eq!(cmd.markdown_zh, PathBuf::from("zh.md"));
}

#[test]
fn descriptive_success_and_missing_evidence_have_no_success_exit_code() {
    for status in [
        ComparisonStatus::ObservedPass,
        ComparisonStatus::Unknown,
        ComparisonStatus::Inconclusive,
    ] {
        assert_eq!(SloCompareExit::from_status(status).code(), 4);
    }
    assert_eq!(
        SloCompareExit::from_status(ComparisonStatus::Failed).code(),
        3
    );
    assert_eq!(
        SloCompareExit::from_status(ComparisonStatus::ProofPass).code(),
        0
    );
}

#[test]
fn complete_json_and_both_languages_are_written_before_insufficient_exit() {
    let (_dir, cmd) = fixture();
    let paths = [
        cmd.out.clone(),
        cmd.markdown_en.clone(),
        cmd.markdown_zh.clone(),
    ];
    assert_eq!(execute(cmd).unwrap(), SloCompareExit::InsufficientEvidence);
    let report: ArtifactComparisonReport =
        serde_json::from_slice(&fs::read(&paths[0]).unwrap()).unwrap();
    assert_eq!(report.comparison.status, ComparisonStatus::Unknown);
    assert_eq!(
        fs::read_to_string(&paths[1]).unwrap(),
        report.to_markdown(MarkdownLanguage::English)
    );
    assert_eq!(
        fs::read_to_string(&paths[2]).unwrap(),
        report.to_markdown(MarkdownLanguage::Chinese)
    );
    assert_eq!(report.comparison.cells.len(), 1);
}

#[test]
fn loader_failure_creates_no_output_files() {
    let (_dir, cmd) = fixture();
    fs::write(&cmd.manifest, b"not json").unwrap();
    let paths = [
        cmd.out.clone(),
        cmd.markdown_en.clone(),
        cmd.markdown_zh.clone(),
    ];
    assert!(execute(cmd)
        .unwrap_err()
        .to_string()
        .contains("load SLO comparison evidence"));
    assert!(paths.iter().all(|path| !path.exists()));
}

#[test]
fn output_cannot_overwrite_manifest_or_any_referenced_input() {
    for input in ["manifest.json", "contract.json"] {
        let (dir, mut cmd) = fixture();
        cmd.out = dir.path().join(input);
        let original = fs::read(&cmd.out).unwrap();
        let en = cmd.markdown_en.clone();
        let zh = cmd.markdown_zh.clone();
        let output = cmd.out.clone();
        assert!(execute(cmd)
            .unwrap_err()
            .to_string()
            .contains("aliases an input artifact"));
        assert_eq!(fs::read(output).unwrap(), original);
        assert!(!en.exists() && !zh.exists());
    }
}

#[cfg(unix)]
#[test]
fn output_hardlink_alias_cannot_modify_original_evidence() {
    let (dir, mut cmd) = fixture();
    let input = dir.path().join("contract.json");
    let original = fs::read(&input).unwrap();
    let alias = dir.path().join("hardlink.json");
    fs::hard_link(&input, &alias).unwrap();
    cmd.out = alias;
    assert!(execute(cmd)
        .unwrap_err()
        .to_string()
        .contains("aliases an input artifact"));
    assert_eq!(fs::read(input).unwrap(), original);
}

#[cfg(unix)]
#[test]
fn symlinked_output_parents_cannot_hide_duplicate_destinations() {
    let (dir, mut cmd) = fixture();
    fs::create_dir(dir.path().join("actual")).unwrap();
    std::os::unix::fs::symlink(dir.path().join("actual"), dir.path().join("alias")).unwrap();
    cmd.out = dir.path().join("actual/result.json");
    cmd.markdown_en = dir.path().join("alias/result.json");
    assert!(execute(cmd)
        .unwrap_err()
        .to_string()
        .contains("distinct files"));
    assert!(!dir.path().join("actual/result.json").exists());
}

#[test]
fn existing_output_and_missing_parent_fail_before_creating_siblings() {
    let (_dir, cmd) = fixture();
    fs::write(&cmd.markdown_zh, b"existing report").unwrap();
    let out = cmd.out.clone();
    let en = cmd.markdown_en.clone();
    let zh = cmd.markdown_zh.clone();
    assert!(execute(cmd)
        .unwrap_err()
        .to_string()
        .contains("already exists"));
    assert!(!out.exists() && !en.exists());
    assert_eq!(fs::read(zh).unwrap(), b"existing report");
    let (dir, mut cmd) = fixture();
    cmd.markdown_zh = dir.path().join("missing/report.md");
    let out = cmd.out.clone();
    assert!(execute(cmd).unwrap_err().to_string().contains("parent"));
    assert!(!out.exists());
}

#[test]
fn creation_race_retains_only_named_new_files_and_preserves_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.json");
    let occupied = dir.path().join("occupied.md");
    fs::write(&occupied, b"another owner's output").unwrap();
    // Simulates a file appearing after preflight but before create_new.
    let error = write_outputs(&[
        PreparedOutput {
            path: first.clone(),
            bytes: b"new bytes".to_vec(),
        },
        PreparedOutput {
            path: occupied.clone(),
            bytes: b"must not replace".to_vec(),
        },
    ])
    .unwrap_err()
    .to_string();
    assert!(error.contains("New files retained"));
    assert!(error.contains(&first.display().to_string()));
    assert_eq!(fs::read(&occupied).unwrap(), b"another owner's output");
    assert!(fs::read(first).unwrap().is_empty());
}
