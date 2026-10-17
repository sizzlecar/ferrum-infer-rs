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
    // GitHub evaluates expression delimiters before invoking the shell, even
    // inside quoted Ruby strings. This reader receives all inputs via env/ARGV
    // and must be a literal script for the exercised rejection rules to apply.
    assert!(
        !command.contains("${{"),
        "candidate reader accidentally opens an Actions expression"
    );
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
    let start = accept.find("sm120_evidence=\"").unwrap();
    let end = start + accept[start..].find("esac").unwrap() + "esac".len();
    let condition = &accept[start..end];
    let command = format!("gate_args=(--tasks original/tasks.json)\n{condition}\nprintf '%s\\n' \"${{gate_args[@]}}\"");
    let run = |required: &str, manual_asset: &str| {
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
            .env("MANUAL_SM120_ASSET_ID", manual_asset)
            .env("RUNNER_TEMP", "/release-fixture")
            .output()
            .unwrap()
    };
    let not_run = run("false", "");
    assert!(not_run.status.success());
    assert_eq!(
        String::from_utf8(not_run.stdout).unwrap(),
        "--tasks\noriginal/tasks.json\n"
    );
    let required = run("true", "");
    assert!(required.status.success());
    assert_eq!(String::from_utf8(required.stdout).unwrap(), "--tasks\noriginal/tasks.json\n--tasks\n/release-fixture/evidence/release-cuda-sm120-evidence/tasks/tasks.json\n--installation\n/release-fixture/evidence/release-cuda-sm120-evidence/installation-cuda-sm120.json\n");
    let manual = run("true", "123");
    assert!(manual.status.success());
    assert_eq!(String::from_utf8(manual.stdout).unwrap(), "--tasks\noriginal/tasks.json\n--sm120-runtime-source\nmanual\n--tasks\n/release-fixture/manual-sm120-evidence/tasks/tasks.json\n--installation\n/release-fixture/manual-sm120-evidence/installation-cuda-sm120.json\n");
    assert!(!run("false", "123").status.success());
    assert!(!run("", "").status.success());
    assert!(!run("unknown", "").status.success());
}

#[cfg(unix)]
#[test]
fn resume_manual_inputs_are_paired_and_only_publication_checkpoint_is_optional() {
    use std::process::Command;
    let resume = workflow("release-resume.yml");
    let job = &resume["jobs"]["publish"];
    let input = script(job, "if [ -n \"$MANUAL_SM120_ASSET_ID\" ] ||");
    let run_input = |id: &str, sha: &str| {
        Command::new("bash")
            .args(["--noprofile", "--norc", "-e", "-o", "pipefail", "-c", input])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("MANUAL_SM120_ASSET_ID", id)
            .env("MANUAL_SM120_SHA256", sha)
            .output()
            .unwrap()
            .status
            .success()
    };
    assert!(run_input("", ""));
    assert!(run_input("123", &"a".repeat(64)));
    for (id, sha) in [
        ("123", ""),
        ("", "a"),
        ("0", "a"),
        ("1;exit 0", "a"),
        ("123", "not-a-digest"),
    ] {
        assert!(!run_input(id, sha));
    }
    let source = script(job, "artifact_id() {");
    let start = source.find("artifact_id() {").unwrap();
    let end = start + source[start..].find("\nfor name ").unwrap();
    let command = format!(
        "{}\nartifact_id \"$REQUESTED_ARTIFACT\"",
        &source[start..end]
    );
    let dir = tempfile::tempdir().unwrap();
    let run_lookup = |artifacts: &str, name: &str| {
        fs::write(dir.path().join("source-artifacts.json"), artifacts).unwrap();
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
            .env("PATH", std::env::var_os("PATH").unwrap())
            .env("RUNNER_TEMP", dir.path())
            .env("REQUESTED_ARTIFACT", name)
            .output()
            .unwrap()
    };
    let missing = run_lookup("[]", "publication-state");
    assert!(missing.status.success());
    assert!(String::from_utf8(missing.stdout).unwrap().trim().is_empty());
    assert!(!run_lookup("[]", "release-tasks").status.success());
    assert!(!run_lookup(
        r#"[{"name":"publication-state","id":1},{"name":"publication-state","id":2}]"#,
        "publication-state"
    )
    .status
    .success());
    let existing = run_lookup(
        r#"[{"name":"publication-state","id":17}]"#,
        "publication-state",
    );
    assert!(existing.status.success());
    assert_eq!(String::from_utf8(existing.stdout).unwrap().trim(), "17");
    for step in steps(job) {
        if let Some(ids) = step["with"]["artifact-ids"].as_str() {
            if ids.contains("outputs.publication_state") {
                assert_eq!(
                    step["if"].as_str(),
                    Some("steps.source.outputs.publication_state != ''")
                );
            } else {
                assert!(
                    step["if"].is_null(),
                    "required evidence cannot depend on a publication checkpoint"
                );
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn resume_distribution_selection_ignores_logs_but_rejects_ambiguous_packages() {
    use std::{collections::BTreeSet, process::Command};
    let resume = workflow("release-resume.yml");
    let source = script(&resume["jobs"]["publish"], "artifact_id() {");
    let start = source
        .find("jq -e --arg run \"$SOURCE_RUN\" --arg candidate \"$candidate\" '")
        .unwrap();
    let end = start + source[start..].find("\nunix_ids=").unwrap();
    let command = &source[start..end];
    let dir = tempfile::tempdir().unwrap();
    let candidate = "a".repeat(40);
    let artifact = |name: &str, attempt: u32, run: &str, sha: &str| serde_json::json!({"name": format!("{name}-run-{run}-{attempt}-{sha}")});
    let packages = vec![
        artifact("ferrum-linux-x86_64", 2, "123", &candidate),
        artifact("ferrum-linux-x86_64-cuda-sm89", 2, "123", &candidate),
        artifact("ferrum-linux-x86_64-cuda-sm120", 2, "123", &candidate),
        artifact("ferrum-macos-aarch64", 2, "123", &candidate),
        artifact("ferrum-windows-x86_64-cpu", 2, "123", &candidate),
        artifact("ferrum-windows-x86_64-cuda", 3, "123", &candidate),
    ];
    let logs = vec![
        artifact("ferrum-windows-cuda-build-evidence", 2, "123", &candidate),
        artifact("ferrum-windows-cuda-build-evidence", 3, "123", &candidate),
    ];
    let foreign = vec![
        artifact("ferrum-windows-x86_64-cuda", 3, "124", &candidate),
        artifact("ferrum-windows-x86_64-cuda", 3, "123", &"b".repeat(40)),
    ];
    let run = |artifacts: &[serde_json::Value]| {
        fs::write(
            dir.path().join("source-artifacts.json"),
            serde_json::to_vec(artifacts).unwrap(),
        )
        .unwrap();
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
            .env("PATH", std::env::var_os("PATH").unwrap())
            .env("RUNNER_TEMP", dir.path())
            .env("SOURCE_RUN", "123")
            .env("candidate", &candidate)
            .output()
            .unwrap();
        let selected = output.status.success().then(|| {
            serde_json::from_slice::<Vec<serde_json::Value>>(
                &fs::read(dir.path().join("staged-artifacts.json")).unwrap(),
            )
            .unwrap()
        });
        (output, selected)
    };
    let names = |artifacts: &[serde_json::Value]| {
        artifacts
            .iter()
            .map(|artifact| artifact["name"].as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>()
    };
    let mut inventory = packages.clone();
    inventory.extend(logs.iter().cloned());
    inventory.extend(foreign.iter().cloned());
    let (output, selected) = run(&inventory);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(names(&selected.unwrap()), names(&packages));
    inventory.push(artifact("ferrum-windows-x86_64-cuda", 2, "123", &candidate));
    assert!(!run(&inventory).0.status.success());
    for missing_packages in [vec![], logs, foreign] {
        assert!(!run(&missing_packages).0.status.success());
    }
}

#[cfg(unix)]
#[test]
fn resume_manual_archive_preserves_raw_bytes_and_rejects_bad_hash_links_and_missing_evidence() {
    use sha2::{Digest, Sha256};
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let resume = workflow("release-resume.yml");
    let download = script(&resume["jobs"]["publish"], "<<'RUBY_EVIDENCE'");
    let ruby = download
        .split_once("<<'RUBY_EVIDENCE'\n")
        .unwrap()
        .1
        .split_once("\nRUBY_EVIDENCE")
        .unwrap()
        .0;
    assert!(
        !ruby.contains("${{"),
        "Actions must pass the verifier as literal Ruby"
    );
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir_all(source.join("tasks")).unwrap();
    fs::create_dir_all(source.join("model-cuda-sm120/task-0")).unwrap();
    fs::write(
        source.join("tasks/tasks.json"),
        b"{\"original_tasks\":true}\n",
    )
    .unwrap();
    fs::write(
        source.join("installation-cuda-sm120.json"),
        b"{\"original_installation\":true}\n",
    )
    .unwrap();
    let raw = b"{\"original_report\":\"transport does not assert semantic success\"}\n";
    fs::write(source.join("model-cuda-sm120/task-0/report.json"), raw).unwrap();
    let archive = dir.path().join("evidence.tar.gz");
    let pack = || {
        let result = Command::new("tar")
            .args(["--format=ustar", "-czf"])
            .arg(&archive)
            .arg("-C")
            .arg(&source)
            .args(["tasks", "installation-cuda-sm120.json", "model-cuda-sm120"])
            .env("COPYFILE_DISABLE", "1")
            .env("LC_ALL", "C")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        format!("{:x}", Sha256::digest(fs::read(&archive).unwrap()))
    };
    let extract = |sha: &str, name: &str| {
        let mut child = Command::new("ruby")
            .args([
                "-rdigest",
                "-rzlib",
                "-rrubygems/package",
                "-rfileutils",
                "-",
            ])
            .arg(&archive)
            .arg(dir.path().join(name))
            .arg(sha)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(ruby.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };
    let sha = pack();
    let valid = extract(&sha, "accepted");
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    assert_eq!(
        fs::read(
            dir.path()
                .join("accepted/model-cuda-sm120/task-0/report.json")
        )
        .unwrap(),
        raw
    );
    assert!(!extract(&"0".repeat(64), "wrong-hash").status.success());
    assert!(
        !dir.path().join("wrong-hash").exists(),
        "SHA must be checked before extraction"
    );
    std::os::unix::fs::symlink("../../outside", source.join("model-cuda-sm120/link")).unwrap();
    let linked = pack();
    assert!(!extract(&linked, "linked").status.success());
    assert!(!dir.path().join("outside").exists());
    fs::remove_file(source.join("model-cuda-sm120/link")).unwrap();
    fs::remove_file(source.join("tasks/tasks.json")).unwrap();
    let missing = pack();
    assert!(!extract(&missing, "missing").status.success());
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
