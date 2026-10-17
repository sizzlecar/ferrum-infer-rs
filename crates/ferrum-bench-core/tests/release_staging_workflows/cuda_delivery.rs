//! Delivery wiring for a second CUDA architecture. Runtime evidence remains
//! independent even when both archives implement the same model profile IDs.
use super::{key, Value};
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
fn publishing_requires_independent_sm120_tasks_runtime_and_final_asset() {
    let delivery = workflow("release-delivery.yml");
    let runtime = &delivery["jobs"]["cuda-models-sm120"];
    let publish = &delivery["jobs"]["publish"];
    let asset = "needs.stage-cuda-linux.outputs.cuda_sm120_asset_id";
    assert!(downloads(runtime, asset));
    assert!(downloads(publish, asset));
    assert!(needs(publish, "cuda-models"));
    assert!(needs(publish, "cuda-models-sm120"));
    assert!(publish["if"]
        .as_str()
        .unwrap()
        .contains("needs.cuda-models-sm120.result == 'success'"));
    let execute = script(runtime, "release_delivery local");
    assert!(execute.contains("model_gate prepare --backend cuda --cuda-lane local"));
    assert!(execute.contains("release_delivery inspect"));
    assert!(!execute.contains("--extract-only"));
    assert!(execute.contains("--candidate-sha \"$GITHUB_SHA\""));
    assert!(execute.contains("--ferrum-bin \"$RUNNER_TEMP/extracted-cuda-sm120/ferrum\""));
    assert!(execute.contains("--arg artifact \"$STAGED_ARTIFACT_ID\""));
    let upload = steps(runtime)
        .iter()
        .find(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|action| action.starts_with("actions/upload-artifact@"))
        })
        .unwrap();
    let artifact = upload["with"]["name"].as_str().unwrap();
    let files = upload["with"]["path"].as_str().unwrap();
    assert!(files.contains("/tasks/tasks.json"));
    assert!(files.contains("/installation-cuda-sm120.json"));
    for (name, consumer) in [
        ("publish", publish.clone()),
        (
            "resume",
            workflow("release-resume.yml")["jobs"]["publish"].clone(),
        ),
    ] {
        let accept = script(&consumer, "gate_args=(");
        assert!(
            accept.contains("--tasks \"$RUNNER_TEMP/tasks/tasks.json\""),
            "{name}: existing task set"
        );
        assert!(
            accept.contains(&format!(
                "--tasks \"$RUNNER_TEMP/evidence/{artifact}/tasks/tasks.json\""
            )),
            "{name}: independent SM120 task set"
        );
        assert!(
            accept.contains(&format!(
                "--installation \"$RUNNER_TEMP/evidence/{artifact}/installation-cuda-sm120.json\""
            )),
            "{name}: actual SM120 installation"
        );
    }
}

#[test]
fn completion_requires_auto_installed_sm120_bytes_and_real_cuda_execution() {
    let delivery = workflow("release-delivery.yml");
    let install = &delivery["jobs"]["install-bootstrap-sm120"];
    let complete = &delivery["jobs"]["complete"];
    assert!(needs(install, "publish"));
    assert!(needs(complete, "install-bootstrap-sm120"));
    let install_script = script(install, "curl -fsSL");
    assert!(install_script.contains("https://ferrum.pandaailabs.com/install.sh | sh\n"));
    assert!(
        !install_script.contains("| sh -s -- --backend"),
        "public auto selection must be exercised"
    );
    assert!(install_script.contains(".binary_sha256 == $installed[0].binary_sha256"));
    assert!(install_script.contains(".release_candidate_sha == $source"));
    assert!(script(install, "--checks basic").contains("--backend cuda"));
    let finish = script(complete, "test \"$PREPARE\"");
    let step = steps(complete).first().unwrap();
    let env = step["env"].as_mapping().unwrap();
    let required = env
        .iter()
        .find_map(|(name, value)| {
            (value.as_str() == Some("${{ needs.install-bootstrap-sm120.result }}"))
                .then(|| name.as_str().unwrap())
        })
        .expect("completion dependency result");
    assert!(finish.contains(&format!("test \"${required}\" = success")));
    assert!(delivery["jobs"]
        .as_mapping()
        .unwrap()
        .contains_key(key("install-bootstrap-windows")));
}

#[cfg(unix)]
#[test]
fn checked_out_yaml_controls_sm120_requirement_without_version_or_asset_guessing() {
    use std::process::Command;
    let delivery = workflow("release-delivery.yml");
    let prepare = &delivery["jobs"]["prepare"];
    let step = steps(prepare)
        .iter()
        .find(|step| step["id"].as_str() == Some("cuda-capability"))
        .unwrap();
    let command = step["run"].as_str().unwrap();
    let staging = workflow("release-cuda.yml");
    let current = serde_yaml::to_string(&staging).unwrap();
    let mut previous = staging.clone();
    previous["on"]["workflow_call"]["outputs"]
        .as_mapping_mut()
        .unwrap()
        .remove(key("cuda_sm120_asset_id"));
    let previous = serde_yaml::to_string(&previous).unwrap();
    let run = |source: Option<&str>, verify_only: bool| {
        let dir = tempfile::tempdir().unwrap();
        if let Some(source) = source {
            let workflows = dir.path().join(".github/workflows");
            fs::create_dir_all(&workflows).unwrap();
            fs::write(workflows.join("release-cuda.yml"), source).unwrap();
        }
        let output_file = dir.path().join("outputs");
        let output = Command::new("bash")
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
            .env("GITHUB_OUTPUT", &output_file)
            .env(
                "VERIFY_PUBLISHED_VERSION",
                if verify_only { "1.2.3" } else { "" },
            )
            .output()
            .unwrap();
        (
            output.status.success(),
            fs::read_to_string(output_file).unwrap_or_default(),
        )
    };
    assert_eq!(
        run(Some(&current), false),
        (true, "sm120_required=true\n".into())
    );
    assert_eq!(
        run(Some(&current), true),
        (true, "sm120_required=true\n".into())
    );
    assert_eq!(
        run(Some(&previous), true),
        (true, "sm120_required=false\n".into())
    );
    assert!(
        !run(Some(&previous), false).0,
        "active release must declare the capability"
    );
    assert_eq!(run(None, true), (true, "sm120_required=false\n".into()));
    assert!(!run(None, false).0);
    // Flow style, a quoted trigger key and reordered mappings exercise actual
    // YAML parsing rather than an indentation or line-count heuristic.
    let compact = "'on': {workflow_call: {outputs: {cuda_sm120_asset_id: {value: '${{ jobs.build.outputs.asset }}'}}}}\n";
    assert_eq!(
        run(Some(compact), true),
        (true, "sm120_required=true\n".into())
    );
    let unrelated =
        "on: {workflow_call: {outputs: {}}}\nnote: 'cuda_sm120_asset_id: present only in text'\n";
    assert_eq!(
        run(Some(unrelated), true),
        (true, "sm120_required=false\n".into())
    );
    for invalid in [
        "on: [unfinished",
        "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {}}}}",
        "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: one}, cuda_sm120_asset_id: {value: two}}}}",
    ] {
        assert!(!run(Some(invalid), true).0, "malformed capability must not become an optional check");
    }
    assert_eq!(
        prepare["outputs"]["sm120_required"].as_str(),
        Some("${{ steps.cuda-capability.outputs.sm120_required }}")
    );
    assert!(delivery["jobs"]["install-bootstrap-sm120"]["if"]
        .as_str()
        .unwrap()
        .contains("needs.prepare.outputs.sm120_required == 'true'"));
}

#[cfg(unix)]
#[test]
fn completion_accepts_legacy_skip_only_when_release_source_does_not_support_sm120() {
    use std::process::Command;
    let delivery = workflow("release-delivery.yml");
    let complete = &delivery["jobs"]["complete"];
    let command = script(complete, "test \"$PREPARE\"");
    let run = |verify_only: bool, required: &str, result: &str| {
        let dir = tempfile::tempdir().unwrap();
        Command::new("bash")
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
            .env("GITHUB_STEP_SUMMARY", dir.path().join("summary"))
            .env("VERSION", "1.2.3")
            .env("PREPARE", "success")
            .env("VERIFY_ONLY", if verify_only { "true" } else { "false" })
            .env("ACTIVE", if verify_only { "false" } else { "true" })
            .env("PUBLISH", if verify_only { "skipped" } else { "success" })
            .env("CARGO_INSTALL", "success")
            .env("BREW_INSTALL", "success")
            .env("UNIX_BOOTSTRAP", "success")
            .env("WINDOWS_BOOTSTRAP", "success")
            .env("SM120_REQUIRED", required)
            .env("SM120_BOOTSTRAP", result)
            .output()
            .unwrap()
            .status
            .success()
    };
    assert!(run(false, "true", "success"));
    assert!(run(true, "true", "success"));
    assert!(run(true, "false", "skipped"));
    assert!(!run(false, "false", "skipped"));
    assert!(!run(true, "false", "success"));
    for verify_only in [false, true] {
        for result in ["skipped", "failure", "cancelled", ""] {
            assert!(!run(verify_only, "true", result));
        }
        assert!(!run(verify_only, "", "skipped"));
        assert!(!run(verify_only, "unknown", "skipped"));
    }
}

#[cfg(unix)]
#[test]
fn resume_reads_candidate_yaml_and_preserves_single_task_legacy_callers() {
    use std::process::Command;
    let resume = workflow("release-resume.yml");
    let publish = &resume["jobs"]["publish"];
    let read = steps(publish)
        .iter()
        .find(|step| step["id"].as_str() == Some("cuda-capability"))
        .unwrap()["run"]
        .as_str()
        .unwrap();
    let declared = "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: '${{ jobs.stage.outputs.asset_id }}'}}}}\n";
    let old = "on: {workflow_call: {outputs: {cuda_asset_id: {value: '${{ jobs.stage.outputs.asset_id }}'}}}}\n";
    let run_reader = |candidate: &str, current: &str| {
        let dir = tempfile::tempdir().unwrap();
        for (relative, contents) in [
            ("candidate/.github/workflows", candidate),
            (".github/workflows", current),
        ] {
            let workflows = dir.path().join(relative);
            fs::create_dir_all(&workflows).unwrap();
            fs::write(workflows.join("release-cuda.yml"), contents).unwrap();
        }
        let outputs = dir.path().join("outputs");
        let result = Command::new("bash")
            .args(["--noprofile", "--norc", "-e", "-o", "pipefail", "-c", read])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("GITHUB_WORKSPACE", dir.path())
            .env("GITHUB_OUTPUT", &outputs)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        fs::read_to_string(outputs).unwrap()
    };
    assert_eq!(run_reader(old, declared), "sm120_required=false\n");
    assert_eq!(run_reader(declared, old), "sm120_required=true\n");

    let accept = script(publish, "gate_args=(");
    let start = accept.find("case \"$SM120_REQUIRED\" in").unwrap();
    let end = start + accept[start..].find("esac").unwrap() + "esac".len();
    let condition = &accept[start..end];
    let command = format!(
        "gate_args=(--tasks original/tasks.json)\n{condition}\nprintf '%s\\n' \"${{gate_args[@]}}\""
    );
    let run_args = |required: &str| {
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
            .env("SM120_REQUIRED", required)
            .env("RUNNER_TEMP", "/release-fixture")
            .output()
            .unwrap()
    };
    let legacy = run_args("false");
    assert!(legacy.status.success());
    assert_eq!(
        String::from_utf8(legacy.stdout).unwrap(),
        "--tasks\noriginal/tasks.json\n"
    );
    let current = run_args("true");
    assert!(current.status.success());
    assert_eq!(
        String::from_utf8(current.stdout).unwrap(),
        "--tasks\noriginal/tasks.json\n--tasks\n/release-fixture/evidence/release-cuda-sm120-evidence/tasks/tasks.json\n--installation\n/release-fixture/evidence/release-cuda-sm120-evidence/installation-cuda-sm120.json\n"
    );
    assert!(!run_args("").status.success());
    assert!(!run_args("unknown").status.success());
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
