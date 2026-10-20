//! Exercise the linked native artifact in an isolated process, so retired
//! environment fallbacks cannot be hidden by earlier process initialization.

use std::path::PathBuf;
use std::process::Command;

const CHILD_DIRECTORY: &str = "FERRUM_TEST_MOE_PROFILE_RETIREMENT_DIRECTORY";
const CHILD_SELECTOR: &str = "FERRUM_TEST_NATIVE_RETIREMENT_SELECTOR";

const RETIRED_ENVIRONMENT: &[&str] = &[
    "FERRUM_MARLIN_TILE",
    "FERRUM_VLLM_MOE_THREAD_K",
    "FERRUM_VLLM_MOE_THREAD_N",
    "FERRUM_VLLM_MOE_BLOCKS_PER_SM",
    "FERRUM_PROFILE_JSONL",
    "FERRUM_PROFILE_COMMIT_SHA",
    "FERRUM_PROFILE_ENV_HASH",
    "FERRUM_PROFILE_MODEL",
    "FERRUM_PROFILE_CONCURRENCY",
    "FERRUM_PROFILE_RUNTIME_FLAGS_JSON",
    "FERRUM_VLLM_MOE_LOG_CONFIG",
    "FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT",
    "FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS",
    "FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS",
];

pub(super) struct ChildCase<'a> {
    pub name: &'static str,
    pub environment: &'a [(&'static str, &'static str)],
    /// None leaves profiling absent. Some checks native writer retirement,
    /// with either an absent file or an existing file owned by someone else.
    pub profile_canary: Option<bool>,
}

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn run_fresh_children(
    module: &str,
    function: &str,
    cases: &[ChildCase<'_>],
    numerical: impl FnOnce(),
) {
    let test_module = module.split_once("::").unwrap().1;
    let test_name = format!("{test_module}::{function}");
    if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
        assert_eq!(std::env::var(CHILD_SELECTOR).unwrap(), test_name);
        numerical();
        // A handshake ensures an accidentally empty --exact selection cannot
        // make the parent pass without executing these numerical assertions.
        std::fs::remove_file(PathBuf::from(directory).join("pending-child")).unwrap();
        return;
    }

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = TestDirectory(std::env::temp_dir().join(format!(
        "ferrum-moe-profile-retirement-{}-{unique}",
        std::process::id()
    )));
    std::fs::create_dir(&directory.0).unwrap();
    for case in cases {
        let case_directory = directory.0.join(case.name);
        std::fs::create_dir(&case_directory).unwrap();
        let canary = case_directory.join("native-profile.jsonl");
        let handshake = case_directory.join("pending-child");
        if case.profile_canary == Some(true) {
            std::fs::write(&canary, b"original-owner-content\n").unwrap();
        }
        std::fs::write(&handshake, b"pending").unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", &test_name, "--ignored", "--nocapture"])
            .env(CHILD_DIRECTORY, &case_directory)
            .env(CHILD_SELECTOR, &test_name);
        // Every scenario starts before native static initialization. In
        // particular, inherited tuning must not contaminate the baseline.
        for key in RETIRED_ENVIRONMENT {
            command.env_remove(key);
        }
        command.envs(case.environment.iter().copied());
        if case.profile_canary.is_some() {
            command
                .env("FERRUM_PROFILE_JSONL", &canary)
                .env("FERRUM_PROFILE_COMMIT_SHA", "retired-canary")
                .env("FERRUM_PROFILE_ENV_HASH", "retired-canary")
                .env("FERRUM_PROFILE_MODEL", "retired-canary")
                .env("FERRUM_PROFILE_CONCURRENCY", "1")
                .env("FERRUM_PROFILE_RUNTIME_FLAGS_JSON", "{}")
                .env("FERRUM_VLLM_MOE_LOG_CONFIG", "1")
                .env("FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT", "1000")
                .env("FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS", "0")
                .env("FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS", "0");
        }
        let output = command.output().unwrap();
        eprintln!(
            "native retirement test={function} scenario={} status={}\nstdout={}\nstderr={}",
            case.name,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            output.status.success(),
            "native child failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !handshake.exists(),
            "native numerical child did not complete"
        );
        if case.profile_canary == Some(true) {
            assert_eq!(std::fs::read(&canary).unwrap(), b"original-owner-content\n");
        } else {
            assert!(
                !canary.exists(),
                "retired native writer created a profile file"
            );
        }
        assert!(
            !String::from_utf8_lossy(&output.stderr).contains("[vllm-moe-config]"),
            "retired native diagnostic writer remains active"
        );
    }
}

#[cfg(feature = "vllm-moe-marlin")]
#[test]
#[ignore = "requires a CUDA host and the profiler-free FP8/MXFP4 Marlin-MoE artifact"]
fn native_moe_profile_retirement_ignores_environment_after_real_math() {
    run_fresh_children(
        module_path!(),
        "native_moe_profile_retirement_ignores_environment_after_real_math",
        &[
            ChildCase {
                name: "baseline",
                environment: &[],
                profile_canary: None,
            },
            ChildCase {
                name: "absent-canary",
                environment: &[],
                profile_canary: Some(false),
            },
            ChildCase {
                name: "existing-canary",
                environment: &[],
                profile_canary: Some(true),
            },
        ],
        || {
            // Real FFI, checked synchronization, and the original independent
            // CPU/source numerical references; no product config early reject.
            super::tests::qwen36_a3_fp8_marlin_moe_ffi_matches_cpu_reference_for_four_cases();
            super::tests::gpt_oss_mxfp4_marlin_moe_bf16_ffi_matches_source_reference_for_four_cases(
            );
        },
    );
}
