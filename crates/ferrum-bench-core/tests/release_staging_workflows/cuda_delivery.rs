//! Delivery wiring for a second CUDA architecture. Runtime evidence remains
//! independent even when both archives implement the same model profile IDs.
#[cfg(unix)]
use super::key;
use super::Value;
use std::{fs, path::Path};

fn workflow(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.github/workflows")
        .join(name);
    serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn steps(job: &Value) -> &[Value] {
    job["steps"].as_sequence().expect("job steps")
}

fn script<'a>(job: &'a Value, command: &str) -> &'a str {
    steps(job)
        .iter()
        .filter_map(|step| step["run"].as_str())
        .find(|script| script.contains(command))
        .unwrap_or_else(|| panic!("missing workflow command: {command}"))
}

fn needs(job: &Value, dependency: &str) -> bool {
    job["needs"].as_sequence().is_some_and(|values| {
        values
            .iter()
            .any(|value| value.as_str() == Some(dependency))
    })
}

fn downloads(job: &Value, reference: &str) -> bool {
    steps(job).iter().any(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|action| action.starts_with("actions/download-artifact@"))
            && step["with"]["artifact-ids"]
                .as_str()
                .is_some_and(|ids| ids.contains(reference))
    })
}

#[cfg(unix)]
#[test]
fn native_archive_guard_rejects_unbound_digest_and_wrong_source() {
    use std::process::Command;
    let staging = workflow("release-cuda.yml");
    let job = &staging["jobs"]["linux-x86_64-cuda-sm120"];
    let guard = script(job, "SM120 native archive SHA256 has not been bound");
    let url = job["env"]["NATIVE_OPERATOR_SET_ARCHIVE_URL"]
        .as_str()
        .unwrap();
    // Only execute the committed input validation, never its download/build.
    let run = |url: &str, digest: &str| {
        Command::new("bash")
            .args(["--noprofile", "--norc", "-e", "-o", "pipefail", "-c", guard])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("NATIVE_OPERATOR_SET_ARCHIVE_URL", url)
            .env("NATIVE_OPERATOR_SET_ARCHIVE_SHA256", digest)
            .output()
            .unwrap()
            .status
            .success()
    };
    assert!(run(url, &"a".repeat(64)));
    for digest in ["", "PENDING_REAL_ARCHIVE_SHA256", "not-hex"] {
        assert!(!run(url, digest));
    }
    assert!(!run(
        "https://example.invalid/other-native.tar.zst",
        &"a".repeat(64)
    ));
    assert!(!run(url, &"a".repeat(63)));
}

#[test]
fn cuda_staging_keeps_architecture_inputs_and_package_metadata_separate() {
    let staging = workflow("release-cuda.yml");
    let old = &staging["jobs"]["linux-x86_64-cuda-sm89"];
    let production = &staging["jobs"]["linux-x86_64-cuda-sm120"];
    assert!(script(old, "cargo build --release --locked -p ferrum-cli")
        .contains("--features cuda,vllm-moe-marlin,vllm-paged-attn-v2"));
    assert!(
        script(production, "cargo build --release --locked -p ferrum-cli")
            .contains("--features cuda,cuda-production")
    );
    assert_ne!(old["container"]["image"], production["container"]["image"]);
    assert_ne!(
        staging["env"]["NATIVE_OPERATOR_SET_ARCHIVE_URL"],
        production["env"]["NATIVE_OPERATOR_SET_ARCHIVE_URL"]
    );
    let image = production["container"]["image"].as_str().unwrap();
    let (_, digest) = image.split_once("@sha256:").expect("pinned build image");
    assert!(digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()));
    let package = script(production, "release_candidate manifest");
    assert!(
        package.contains(image),
        "ABI must record the actual build image"
    );
    assert!(package.contains("--cargo-features cuda,cuda-production"));
    assert!(package.contains("--cuda-compute-capability \"$CUDA_COMPUTE_CAP\""));
    assert!(package.contains("${NATIVE_OPERATOR_SET_ARCHIVE_SHA256}"));
    // The automatic installer owns a strict archive member allowlist. Changing
    // the native build input does not add a runtime lock file to the product.
    let archive_members = |job: &Value| {
        script(job, "release_candidate manifest")
            .lines()
            .find_map(|line| line.trim().strip_prefix("tar -C dist -czf \"$asset\" "))
            .unwrap()
            .to_owned()
    };
    assert_eq!(archive_members(production), archive_members(old));
    let materialize = script(production, "zstd --decompress");
    assert!(materialize.contains("lock=\"$root/native-operator-set.lock.json\""));
    let check = materialize.find("sha256sum --check --strict").unwrap();
    let extract = materialize.find("zstd --decompress").unwrap();
    assert!(
        check < extract,
        "unverified native bytes must not be extracted"
    );
    assert!(staging["jobs"]["windows-x86_64-cuda-sm89"]["uses"].is_string());
}

#[test]
fn publishing_keeps_sm120_archive_identity_without_claiming_runtime_validation() {
    let delivery = workflow("release-delivery.yml");
    let tasks = &delivery["jobs"]["tasks"];
    let publish = &delivery["jobs"]["publish"];
    let asset = "needs.stage-cuda-linux.outputs.cuda_sm120_asset_id";
    assert!(downloads(tasks, asset));
    assert!(downloads(publish, asset));
    assert!(needs(publish, "cuda-models"));
    let inspect = script(tasks, "extracted-cuda-sm120");
    assert!(inspect
        .contains("--abi \"$RUNNER_TEMP/staged/ferrum-linux-x86_64-cuda-sm120.tar.gz.abi.json\""));
    assert!(inspect.contains(
        "--candidate-sha \"$GITHUB_SHA\" --extract-dir \"$RUNNER_TEMP/extracted-cuda-sm120\""
    ));
    assert!(
        inspect.contains("--output \"$RUNNER_TEMP/installation/cuda-sm120.json\" --extract-only")
    );
    let accept = script(publish, "gate_args=(");
    assert!(accept.contains("--tasks \"$RUNNER_TEMP/tasks/tasks.json\""));
    assert!(!accept.contains("release-cuda-sm120-evidence"));
    for (name, job) in delivery["jobs"].as_mapping().unwrap() {
        assert!(
            !serde_yaml::to_string(&job["runs-on"])
                .unwrap()
                .contains("ferrum-cuda-sm120"),
            "{name:?}"
        );
    }
    assert!(delivery["jobs"]["cuda-models-sm120"].is_null());
    assert!(delivery["jobs"]["install-bootstrap-sm120"].is_null());
    let preparation = steps(&delivery["jobs"]["prepare"]);
    let notes_step = preparation
        .iter()
        .position(|step| {
            step["run"]
                .as_str()
                .is_some_and(|run| run.contains("SM120 validation scope"))
        })
        .unwrap();
    let archive_step = preparation
        .iter()
        .position(|step| {
            step["with"]["name"].as_str() == Some("release-plan")
                && step["uses"]
                    .as_str()
                    .is_some_and(|action| action.starts_with("actions/upload-artifact@"))
        })
        .unwrap();
    assert!(notes_step < archive_step);
    assert!(preparation[archive_step]["with"]["path"]
        .as_str()
        .unwrap()
        .contains("release-notes.md"));
    assert!(accept.contains("--notes \"$RUNNER_TEMP/plan/release-notes.md\""));
    let notes = preparation[notes_step]["run"].as_str().unwrap();
    assert!(notes.contains("Model execution and public SM120 installation/startup are not run"));
    assert!(script(&delivery["jobs"]["complete"], "test \"$PREPARE\"")
        .contains("SM120 model execution and public installation/startup: not_run"));
}

#[cfg(unix)]
fn read_capability(
    source: Option<&str>,
    current: Option<&str>,
    resume: bool,
    verify: bool,
) -> (std::process::Output, String) {
    use std::process::Command;
    let workflow = workflow(if resume {
        "release-resume.yml"
    } else {
        "release-delivery.yml"
    });
    let command = steps(&workflow["jobs"][if resume { "publish" } else { "prepare" }])
        .iter()
        .find(|step| step["id"].as_str() == Some("cuda-capability"))
        .unwrap()["run"]
        .as_str()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (relative, contents) in [
        (
            if resume {
                "candidate/.github/workflows"
            } else {
                ".github/workflows"
            },
            source,
        ),
        (".github/workflows", if resume { current } else { None }),
    ] {
        if let Some(contents) = contents {
            let path = dir.path().join(relative);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("release-cuda.yml"), contents).unwrap();
        }
    }
    let outputs = dir.path().join("outputs");
    let result = Command::new("bash")
        .args([
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            command,
        ])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GITHUB_WORKSPACE", dir.path())
        .env("GITHUB_OUTPUT", &outputs)
        .env(
            "VERIFY_PUBLISHED_VERSION",
            if verify { "1.2.3" } else { "" },
        )
        .output()
        .unwrap();
    (result, fs::read_to_string(outputs).unwrap_or_default())
}

#[cfg(unix)]
#[test]
fn candidate_policy_preserves_legacy_requirements_and_fails_before_runner_queue() {
    let staging = workflow("release-cuda.yml");
    let current = serde_yaml::to_string(&staging).unwrap();
    let mut required = staging.clone();
    let outputs = required["on"]["workflow_call"]["outputs"]
        .as_mapping_mut()
        .unwrap();
    outputs.remove(key("cuda_sm120_runtime_policy"));
    outputs.remove(key("cuda_sm120_runtime_reason"));
    let mut legacy = required.clone();
    legacy["on"]["workflow_call"]["outputs"]
        .as_mapping_mut()
        .unwrap()
        .remove(key("cuda_sm120_asset_id"));
    let required = serde_yaml::to_string(&required).unwrap();
    let legacy = serde_yaml::to_string(&legacy).unwrap();
    for verify in [false, true] {
        let (result, output) = read_capability(Some(&current), None, false, verify);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(output.contains("sm120_required=true\nsm120_runtime_required=false\nsm120_runtime_policy=not_run_hardware_unavailable\n"));
        assert!(output.contains("sm120_runtime_reason=SM120 hardware is unavailable"));
        let (result, _) = read_capability(Some(&required), None, false, verify);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("no approved SM120 runner"));
    }
    for source in [Some(legacy.as_str()), None] {
        let (result, output) = read_capability(source, None, false, true);
        assert!(result.status.success());
        assert_eq!(output, "sm120_required=false\nsm120_runtime_required=false\nsm120_runtime_policy=required\nsm120_runtime_reason=\n");
        assert!(!read_capability(source, None, false, false)
            .0
            .status
            .success());
    }
    // The source under candidate/, not the executing workflow checkout, owns
    // historical requirements. Missing policy on 0.13 remains Required.
    let (result, output) = read_capability(Some(&required), Some(&current), true, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(output, "sm120_required=true\nsm120_runtime_required=true\nsm120_runtime_policy=required\nsm120_runtime_reason=\n");
    let (result, output) = read_capability(Some(&current), Some(&required), true, false);
    assert!(result.status.success());
    assert!(output.contains("sm120_required=true\nsm120_runtime_required=false\n"));
    let (result, output) = read_capability(Some(&legacy), Some(&required), true, false);
    assert!(result.status.success());
    assert!(output.starts_with("sm120_required=false\nsm120_runtime_required=false\n"));
}

#[cfg(unix)]
#[test]
fn candidate_runtime_policy_rejects_ambiguous_or_injectable_declarations() {
    let valid = "'on': {workflow_call: {outputs: {cuda_sm120_asset_id: {value: '${{ jobs.stage.outputs.id }}'}, cuda_sm120_runtime_policy: {value: not_run_hardware_unavailable}, cuda_sm120_runtime_reason: {value: 'No approved device.'}}}}\n";
    for resume in [false, true] {
        assert!(read_capability(Some(valid), None, resume, true)
            .0
            .status
            .success());
        for (before, after) in [
            ("value: not_run_hardware_unavailable", "value: unknown"),
            ("value: not_run_hardware_unavailable", "value: '${{ inputs.policy }}'"),
            ("value: not_run_hardware_unavailable", "value: true"),
            ("value: not_run_hardware_unavailable", "value: required"),
            ("value: 'No approved device.'", "value: ''"),
            ("value: 'No approved device.'", "value: ' trailing space '"),
            ("value: 'No approved device.'", "value: '\u{a0}No approved device.'"),
            ("value: 'No approved device.'", "value: '${{ inputs.reason }}'"),
            ("value: 'No approved device.'", "value: true"),
            ("value: 'No approved device.'", "value: 123"),
            ("value: 'No approved device.'", "value: null"),
            ("value: 'No approved device.'", "value: [reason]"),
            ("value: 'No approved device.'", "value: \"line\\nsm120_runtime_required=false\""),
            ("value: 'No approved device.'", "value: \"control\\u007f\""),
            ("cuda_sm120_asset_id:", "other_asset:"),
            ("cuda_sm120_runtime_reason:", "unrelated_reason:"),
            ("cuda_sm120_runtime_policy: {value: not_run_hardware_unavailable},", "cuda_sm120_runtime_policy: {value: not_run_hardware_unavailable}, cuda_sm120_runtime_policy: {value: required},"),
        ] {
            let invalid = valid.replace(before, after);
            assert!(!read_capability(Some(&invalid), None, resume, true).0.status.success(), "accepted {after:?}, resume={resume}");
        }
        for invalid in [
            "on: [unfinished",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {}}}}",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: one}, cuda_sm120_asset_id: {value: two}}}}",
        ] {
            assert!(!read_capability(Some(invalid), None, resume, true).0.status.success());
        }
    }
}

#[cfg(unix)]
#[test]
fn resume_keeps_required_sm120_task_and_installation_evidence() {
    use std::process::Command;
    let resume = workflow("release-resume.yml");
    let accept = script(&resume["jobs"]["publish"], "gate_args=(");
    let start = accept.find("case \"$SM120_RUNTIME_REQUIRED\" in").unwrap();
    let end = start + accept[start..].find("esac").unwrap() + "esac".len();
    let condition = &accept[start..end];
    let command = format!("gate_args=(--tasks original/tasks.json)\n{condition}\nprintf '%s\\n' \"${{gate_args[@]}}\"");
    let run = |required: &str| {
        Command::new("bash")
            .args([
                "--noprofile",
                "--norc",
                "-e",
                "-o",
                "pipefail",
                "-c",
                &command,
            ])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("SM120_RUNTIME_REQUIRED", required)
            .env("RUNNER_TEMP", "/release-fixture")
            .output()
            .unwrap()
    };
    let not_run = run("false");
    assert!(not_run.status.success());
    assert_eq!(
        String::from_utf8(not_run.stdout).unwrap(),
        "--tasks\noriginal/tasks.json\n"
    );
    let required = run("true");
    assert!(required.status.success());
    assert_eq!(String::from_utf8(required.stdout).unwrap(), "--tasks\noriginal/tasks.json\n--tasks\n/release-fixture/evidence/release-cuda-sm120-evidence/tasks/tasks.json\n--installation\n/release-fixture/evidence/release-cuda-sm120-evidence/installation-cuda-sm120.json\n");
    assert!(!run("").status.success());
    assert!(!run("unknown").status.success());
}

#[cfg(unix)]
#[test]
fn sm120_distribution_preserves_project_and_native_package_license_texts() {
    use std::process::Command;
    let staging = workflow("release-cuda.yml");
    let production = &staging["jobs"]["linux-x86_64-cuda-sm120"];
    let collect = script(production, "for native_license in");
    let dir = tempfile::tempdir().unwrap();
    let ggml = dir.path().join("crates/ferrum-kernels/src/gguf_blocks");
    fs::create_dir_all(&ggml).unwrap();
    fs::write(dir.path().join("LICENSE"), "project MIT fixture\n").unwrap();
    fs::write(ggml.join("LICENSE.ggml"), "GGML copyright fixture\n").unwrap();
    let native = dir.path().join("portable native set");
    let licenses = native.join("packages/marlin/licenses");
    fs::create_dir_all(&licenses).unwrap();
    let third_party = [
        ("Apache-2.0.txt", "Apache native distribution fixture\n"),
        ("llama-MIT.txt", "llama copyright fixture\n"),
        ("Ferrum-MIT.txt", "native Ferrum MIT fixture\n"),
    ];
    for (name, text) in third_party {
        fs::write(licenses.join(name), text).unwrap();
    }
    let run = || {
        Command::new("bash")
            .args([
                "--noprofile",
                "--norc",
                "-e",
                "-o",
                "pipefail",
                "-c",
                collect,
            ])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env(
                "FERRUM_NATIVE_OPERATOR_SET_LOCK",
                native.join("native-operator-set.lock.json"),
            )
            .current_dir(dir.path())
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let combined = fs::read_to_string(dir.path().join("dist/LICENSE")).unwrap();
    assert!(combined.starts_with("project MIT fixture\nGGML copyright fixture\n"));
    for (name, text) in third_party {
        assert!(combined.contains(&format!("packages/marlin/licenses/{name}")));
        assert!(
            combined.contains(text),
            "native license content must be preserved"
        );
    }
    fs::remove_dir_all(licenses).unwrap();
    assert!(
        !run().status.success(),
        "missing native license inputs must not silently pass packaging"
    );
}
