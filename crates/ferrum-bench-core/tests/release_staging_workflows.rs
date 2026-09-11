//! Contracts for the actual staging workflows, not a count of build commands.
//! Permissions and secret access are checked structurally. The input guard is
//! executed as committed, with dispatch inputs supplied only through its env.
use serde_yaml::{Mapping, Value};
#[cfg(any(unix, windows))]
use std::{collections::BTreeMap, process::Command};
use std::{fs, path::Path};

fn key(name: &str) -> Value {
    Value::String(name.into())
}

fn field<'a>(map: &'a Mapping, name: &str) -> Option<&'a Value> {
    map.get(key(name))
}

fn map<'a>(value: &'a Value, location: &str) -> Result<&'a Mapping, String> {
    value
        .as_mapping()
        .ok_or_else(|| format!("{location} must be a mapping"))
}

fn workflows() -> Vec<(String, Value)> {
    ["release.yml", "release-cuda.yml", "release-windows.yml"]
        .into_iter()
        .map(|name| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.github/workflows")
                .join(name);
            let text = fs::read_to_string(&path).expect("read actual staging workflow");
            (
                name.into(),
                serde_yaml::from_str(&text).expect("parse actual workflow YAML"),
            )
        })
        .collect()
}

fn read_permissions(value: &Value, location: &str) -> Result<(), String> {
    if value.as_str() == Some("read-all") {
        return Ok(());
    }
    let permissions = map(value, location)?;
    // A job mapping replaces, rather than extends, workflow permissions. Missing
    // contents here therefore means none, even when the workflow grants read.
    if field(permissions, "contents").and_then(Value::as_str) != Some("read") {
        return Err(format!("{location} must grant contents: read"));
    }
    for (scope, access) in permissions {
        let scope = scope.as_str().ok_or("permission scope must be a string")?;
        match access.as_str() {
            Some("read" | "none") if scope != "id-token" || access.as_str() == Some("none") => {}
            _ => {
                return Err(format!(
                    "{location}.{scope} must not grant write or indeterminate access"
                ))
            }
        }
    }
    Ok(())
}

fn references_secrets(expression: &str) -> bool {
    let mut characters = expression.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\'' || character == '"' {
            // Ignore quoted literals such as format('secrets are unavailable').
            while let Some(next) = characters.next() {
                if next == character {
                    if characters.peek() == Some(&character) {
                        characters.next();
                    } else {
                        break;
                    }
                }
            }
        } else if character.is_ascii_alphabetic() || character == '_' {
            let mut identifier = String::from(character);
            while characters
                .peek()
                .is_some_and(|next| next.is_ascii_alphanumeric() || *next == '_')
            {
                identifier.push(characters.next().unwrap());
            }
            if identifier.eq_ignore_ascii_case("secrets") {
                return true;
            }
        }
    }
    false
}

fn inspect_credentials(value: &Value, location: &str) -> Result<(), String> {
    match value {
        Value::Mapping(mapping) => {
            for (name, child) in mapping {
                let name = name
                    .as_str()
                    .ok_or("workflow mapping key must be a string")?;
                if name == "secrets" && !child.as_mapping().is_some_and(Mapping::is_empty) {
                    return Err(format!(
                        "{location}.secrets must not forward or inherit credentials"
                    ));
                }
                inspect_credentials(child, &format!("{location}.{name}"))?;
            }
        }
        Value::Sequence(sequence) => {
            for child in sequence {
                inspect_credentials(child, location)?;
            }
        }
        Value::String(text) => {
            let mut remaining = text.as_str();
            while let Some((_, rest)) = remaining.split_once("${{") {
                let (expression, tail) = rest
                    .split_once("}}")
                    .ok_or_else(|| format!("{location}: incomplete workflow expression"))?;
                // The automatic repository token has the checked read-only
                // permissions. Other secret context access needs separate review.
                let compact: String = expression
                    .chars()
                    .filter(|c| !c.is_ascii_whitespace())
                    .collect::<String>()
                    .to_ascii_lowercase();
                if references_secrets(expression)
                    && compact != "secrets.github_token"
                    && compact != "secrets['github_token']"
                {
                    return Err(format!(
                        "{location}: staging must not access publication secrets"
                    ));
                }
                remaining = tail;
            }
        }
        Value::Tagged(_) => return Err(format!("{location}: tagged YAML is not supported")),
        _ => {}
    }
    Ok(())
}

fn staging_permissions(workflow: &Value) -> Result<(), String> {
    let root = map(workflow, "workflow")?;
    read_permissions(
        field(root, "permissions").ok_or("workflow permissions must be explicit")?,
        "workflow.permissions",
    )?;
    let jobs = map(
        field(root, "jobs").ok_or("workflow must have jobs")?,
        "jobs",
    )?;
    if jobs.is_empty() {
        return Err("workflow must have staging jobs".into());
    }
    for (name, job) in jobs {
        let name = name.as_str().ok_or("job ID must be a string")?;
        let job = map(job, name)?;
        if let Some(permissions) = field(job, "permissions") {
            read_permissions(permissions, &format!("jobs.{name}.permissions"))?;
        }
    }
    inspect_credentials(workflow, "workflow")
}

#[test]
fn actual_staging_workflows_keep_read_only_repository_access() {
    for (name, workflow) in workflows() {
        staging_permissions(&workflow).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn workflow_and_job_publication_privileges_are_rejected() {
    for (name, original) in workflows() {
        for permissions in [
            "contents: write",
            "contents: read\npackages: write",
            "contents: read\nid-token: write",
            "write-all",
            "{}",
            "true",
        ] {
            let permissions: Value = serde_yaml::from_str(permissions).unwrap();
            let mut workflow = original.clone();
            workflow["permissions"] = permissions.clone();
            assert!(
                staging_permissions(&workflow).is_err(),
                "{name}: root {permissions:?}"
            );
            let mut workflow = original.clone();
            let job = workflow["jobs"]
                .as_mapping_mut()
                .unwrap()
                .values_mut()
                .next()
                .unwrap();
            job["permissions"] = permissions.clone();
            assert!(
                staging_permissions(&workflow).is_err(),
                "{name}: job {permissions:?}"
            );
        }
        let mut workflow = original.clone();
        workflow
            .as_mapping_mut()
            .unwrap()
            .remove(key("permissions"));
        assert!(
            staging_permissions(&workflow).is_err(),
            "{name}: implicit repository default"
        );
        // Read-only additions and a restrictive job override remain legitimate.
        let mut workflow = original;
        workflow["permissions"] = serde_yaml::from_str("contents: read\nchecks: read").unwrap();
        let job = workflow["jobs"]
            .as_mapping_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        job["permissions"] =
            serde_yaml::from_str("contents: read\npackages: none\nid-token: none").unwrap();
        staging_permissions(&workflow).unwrap();
    }
}

#[test]
fn inherited_and_explicit_publication_secrets_are_rejected() {
    for (name, original) in workflows() {
        for secrets in ["inherit", "registry: ${{ secrets.CARGO_REGISTRY_TOKEN }}"] {
            let mut workflow = original.clone();
            let mut job: Value =
                serde_yaml::from_str("uses: example/build/.github/workflows/stage.yml@main")
                    .unwrap();
            job["secrets"] = serde_yaml::from_str(secrets).unwrap();
            workflow["jobs"]
                .as_mapping_mut()
                .unwrap()
                .insert(key("delegated"), job);
            assert!(staging_permissions(&workflow).is_err(), "{name}: {secrets}");
        }
        for secret in [
            "${{ secrets.RELEASE_TOKEN }}",
            "${{ secrets['REGISTRY_TOKEN'] }}",
            "${{ toJSON(secrets) }}",
        ] {
            let mut workflow = original.clone();
            workflow["env"]["CREDENTIAL"] = key(secret);
            assert!(staging_permissions(&workflow).is_err(), "{name}: {secret}");
        }
        let mut workflow = original;
        workflow["env"]["READ_TOKEN"] = key("${{ secrets.GITHUB_TOKEN }}");
        workflow["env"]["LABEL"] = key("${{ format('secrets are unavailable') }}");
        staging_permissions(&workflow).unwrap();
    }
}

// Execute only guards whose shell family is native to this test host. The
// structural permissions and secret-access checks above still cover every job.
#[cfg(any(unix, windows))]
fn dispatch_inputs() -> BTreeMap<String, String> {
    [
        ("platform", "all".into()),
        ("backend", "cuda".into()),
        ("version", "12.34.56".into()),
        ("release_candidate_sha", "a".repeat(40)),
        ("release_candidate_tag", "v12.34.56-rc.2".into()),
        ("staging_label", "formal-12.34.56_rc.2".into()),
        ("publish_release", "false".into()),
        ("windows_launcher_url", String::new()),
        ("windows_launcher_sha256", String::new()),
    ]
    .into_iter()
    .map(|(name, value)| (name.into(), value))
    .collect()
}

fn guard_shell<'a>(workflow: &'a Value, job: &'a Value, step: &'a Value) -> &'a str {
    step.get("shell")
        .or_else(|| job.get("defaults")?.get("run")?.get("shell"))
        .or_else(|| workflow.get("defaults")?.get("run")?.get("shell"))
        .and_then(Value::as_str)
        .expect("staging guards must declare a shell")
}

#[cfg(any(unix, windows))]
fn guard_consumes_input(workflow: &Value, job: &Value, step: &Value, input: &str) -> bool {
    let script = step["run"].as_str().expect("guard script");
    let expression = format!("${{{{ inputs.{input} }}}}");
    [workflow, job, step].into_iter().any(|scope| {
        scope
            .get("env")
            .and_then(Value::as_mapping)
            .is_some_and(|env| {
                env.iter().any(|(name, value)| {
                    let name = name.as_str().expect("environment key");
                    value.as_str() == Some(expression.as_str())
                        && [
                            format!("${name}"),
                            format!("${{{name}}}"),
                            format!("$env:{name}"),
                        ]
                        .iter()
                        .any(|reference| script.contains(reference))
                })
            })
    })
}

#[cfg(windows)]
fn windows_guard_shell() -> std::path::PathBuf {
    // Hosted runners have pwsh. A developer Windows installation may only have
    // inbox PowerShell 5.1; these environment-only guards use its common syntax.
    if let Some(paths) = std::env::var_os("PATH") {
        for path in std::env::split_paths(&paths) {
            let program = path.join("pwsh.exe");
            if program.is_file() {
                return program;
            }
        }
    }
    let program =
        std::path::PathBuf::from(std::env::var_os("SystemRoot").expect("Windows SystemRoot"))
            .join("System32/WindowsPowerShell/v1.0/powershell.exe");
    assert!(
        program.is_file(),
        "Windows needs an installed PowerShell to test its guards"
    );
    program
}

#[test]
fn staging_guard_shell_uses_step_then_job_then_workflow_default() {
    let workflow: Value = serde_yaml::from_str("defaults:\n  run:\n    shell: bash").unwrap();
    let job: Value = serde_yaml::from_str("defaults:\n  run:\n    shell: pwsh").unwrap();
    let step: Value = serde_yaml::from_str("shell: powershell").unwrap();
    assert_eq!(guard_shell(&workflow, &job, &step), "powershell");
    assert_eq!(guard_shell(&workflow, &job, &Value::Null), "pwsh");
    assert_eq!(guard_shell(&workflow, &Value::Null, &Value::Null), "bash");
}

#[cfg(any(unix, windows))]
fn add_environment(
    command: &mut Command,
    value: Option<&Value>,
    inputs: &BTreeMap<String, String>,
) {
    let Some(value) = value else {
        return;
    };
    for (name, value) in value.as_mapping().expect("workflow env mapping") {
        let name = name.as_str().expect("environment variable name");
        let value = match value {
            Value::String(value) if value.contains("${{") => {
                let input = value
                    .trim()
                    .strip_prefix("${{")
                    .and_then(|v| v.strip_suffix("}}"))
                    .expect("guard env expression must occupy the value")
                    .trim()
                    .strip_prefix("inputs.")
                    .expect("guard env must use dispatch inputs");
                inputs
                    .get(input)
                    .unwrap_or_else(|| panic!("unknown guard input {input}"))
                    .clone()
            }
            Value::String(value) => value.clone(),
            Value::Number(value) => value.to_string(),
            Value::Bool(value) => value.to_string(),
            _ => panic!("unsupported environment value"),
        };
        command.env(name, value);
    }
}

#[cfg(any(unix, windows))]
fn run_guards(
    workflow: &Value,
    inputs: &BTreeMap<String, String>,
    should_succeed: bool,
    label: &str,
    tested_input: Option<&str>,
) -> usize {
    let temporary = tempfile::tempdir().unwrap();
    let mut executed = 0;
    for (job_name, job) in workflow["jobs"].as_mapping().unwrap() {
        if let Some(callee) = job.get("uses").and_then(Value::as_str) {
            // The callee's guards are executed in its own workflow below.
            assert!(
                workflows()
                    .iter()
                    .any(|(name, _)| callee == format!("./.github/workflows/{name}")),
                "unverified staging callee: {callee}"
            );
            continue;
        }
        // Validation must precede checkout, tool installation and build actions.
        let first = job["steps"]
            .as_sequence()
            .expect("staging steps")
            .first()
            .expect("input guard");
        let script = first["run"]
            .as_str()
            .expect("first staging step must execute the input guard");
        let shell = guard_shell(workflow, job, first);
        assert!(
            matches!(shell, "bash" | "pwsh" | "powershell"),
            "unsupported guard shell: {shell}"
        );
        if (cfg!(unix) && shell != "bash") || (cfg!(windows) && shell == "bash") {
            continue;
        }
        if tested_input.is_some_and(|input| !guard_consumes_input(workflow, job, first, input)) {
            continue;
        }
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("bash");
            command
                .args([
                    "--noprofile",
                    "--norc",
                    "-e",
                    "-o",
                    "pipefail",
                    "-c",
                    script,
                ])
                .env_clear()
                .env("PATH", "/usr/bin:/bin");
            command
        };
        #[cfg(windows)]
        let mut command = {
            let program = windows_guard_shell();
            let mut command = Command::new(&program);
            command
                .args([
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ])
                .env_clear();
            for name in ["SystemRoot", "WINDIR", "TEMP", "TMP"] {
                if let Some(value) = std::env::var_os(name) {
                    command.env(name, value);
                }
            }
            eprintln!("guard {job_name:?}: executing {}", program.display());
            command
        };
        command.current_dir(temporary.path());
        add_environment(&mut command, workflow.get("env"), inputs);
        add_environment(&mut command, job.get("env"), inputs);
        add_environment(&mut command, first.get("env"), inputs);
        let output = command
            .output()
            .expect("execute actual workflow guard with bash");
        assert!(
            !temporary.path().join("injected").exists(),
            "{label}: dispatch input was executed"
        );
        assert_eq!(
            output.status.success(),
            should_succeed,
            "{label} {job_name:?}: status={:?}; stdout={}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        executed += 1;
    }
    executed
}

#[cfg(any(unix, windows))]
#[test]
fn actual_input_guards_accept_formal_versions_and_reject_publication_or_bad_inputs() {
    let mut executed = 0;
    for (name, workflow) in workflows() {
        let inputs = dispatch_inputs();
        executed += run_guards(&workflow, &inputs, true, &name, None);
        let mut sha256 = inputs.clone();
        sha256.insert("release_candidate_sha".into(), "b".repeat(64));
        run_guards(&workflow, &sha256, true, &name, None);
        for (field, invalid) in [
            ("platform", "unsupported"),
            ("backend", "unsupported"),
            ("publish_release", "true"),
            ("publish_release", ""),
            ("version", "01.2.3"),
            ("version", "12.34.56-rc.2"),
            ("release_candidate_tag", "v12.34.57-rc.2"),
            ("release_candidate_tag", "v12.34.56-rc.0"),
            ("release_candidate_sha", "not-a-commit"),
            ("staging_label", ""),
        ] {
            let mut invalid_inputs = inputs.clone();
            invalid_inputs.insert(field.into(), invalid.into());
            run_guards(
                &workflow,
                &invalid_inputs,
                false,
                &format!("{name}: {field}={invalid}"),
                Some(field),
            );
        }
        for field in inputs.keys() {
            let mut injected = inputs.clone();
            injected.insert(
                field.clone(),
                "$(printf injected > injected)\"; printf injected > injected; #".into(),
            );
            run_guards(
                &workflow,
                &injected,
                false,
                &format!("{name}: {field} injection"),
                Some(field),
            );
        }
        let mut pinned = inputs.clone();
        pinned.insert("windows_launcher_url".into(), "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v12.34.55/ferrum-windows-launcher-v1.exe".into());
        pinned.insert("windows_launcher_sha256".into(), "b".repeat(64));
        run_guards(
            &workflow,
            &pinned,
            true,
            &name,
            Some("windows_launcher_url"),
        );
        for field in ["windows_launcher_url", "windows_launcher_sha256"] {
            let mut incomplete = pinned.clone();
            incomplete.insert(field.into(), String::new());
            run_guards(
                &workflow,
                &incomplete,
                false,
                &format!("{name}: missing {field}"),
                Some(field),
            );
        }
    }
    assert!(
        executed > 0,
        "no staging input guard was executed on this host"
    );
}
