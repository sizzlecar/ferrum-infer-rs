//! Exercise the linked native artifact in an isolated process, so retired
//! environment fallbacks cannot be hidden by earlier process initialization.

use std::path::PathBuf;
use std::process::Command;

const CHILD_DIRECTORY: &str = "FERRUM_TEST_MOE_PROFILE_RETIREMENT_DIRECTORY";

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires a CUDA host and the profiler-free FP8/MXFP4 Marlin-MoE artifact"]
fn native_moe_profile_retirement_ignores_environment_after_real_math() {
    if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
        // Both helpers execute real FFI, synchronize/read back, and compare all
        // outputs with their independent source/CPU numerical references.
        super::tests::qwen36_a3_fp8_marlin_moe_ffi_matches_cpu_reference_for_four_cases();
        super::tests::gpt_oss_mxfp4_marlin_moe_bf16_ffi_matches_source_reference_for_four_cases();
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
    let canary = directory.0.join("native-profile.jsonl");
    let handshake = directory.0.join("pending-child");
    let test_module = module_path!().split_once("::").unwrap().1;
    let test_name =
        format!("{test_module}::native_moe_profile_retirement_ignores_environment_after_real_math");

    for existing in [false, true] {
        if existing {
            std::fs::write(&canary, b"original-owner-content\n").unwrap();
        }
        std::fs::write(&handshake, b"pending").unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &test_name, "--ignored", "--nocapture"])
            .env(CHILD_DIRECTORY, &directory.0)
            .env("FERRUM_PROFILE_JSONL", &canary)
            .env("FERRUM_PROFILE_COMMIT_SHA", "retired-canary")
            .env("FERRUM_PROFILE_ENV_HASH", "retired-canary")
            .env("FERRUM_PROFILE_MODEL", "retired-canary")
            .env("FERRUM_PROFILE_CONCURRENCY", "1")
            .env("FERRUM_PROFILE_RUNTIME_FLAGS_JSON", "{}")
            .env("FERRUM_VLLM_MOE_LOG_CONFIG", "1")
            .env("FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT", "1000")
            .env("FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS", "0")
            .env("FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS", "0")
            .output()
            .unwrap();
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
        if existing {
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
