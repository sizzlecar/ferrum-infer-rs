//! One real Orchestral headless process per task, on the shared runner clock.
use super::{
    config::{self, Manifest, OrchestralSpec, OrchestralToolResultFormat, Server, Task},
    orchestral_evidence,
    process::{self, ManagedChild, Outcome},
    runner::{self, TaskResult},
    write_json,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{process::Command, sync::Barrier};

pub(crate) struct Prepared {
    pub config_path: PathBuf,
    pub home: PathBuf,
    journal: PathBuf,
    session_id: String,
}

pub(crate) fn prepare(
    manifest: &Manifest,
    spec: &OrchestralSpec,
    task: &Task,
    output: &Path,
    base_url: &str,
) -> Result<Prepared> {
    let home = output.join("orchestral-home");
    let journal = output.join("journals");
    let artifacts = output.join("artifacts");
    let log_file = output.join("orchestral.log");
    for path in [&home, &journal, &artifacts] {
        fs::create_dir(path)?;
    }
    let session_id = format!("regression-{:032x}", rand::random::<u128>());
    let template: Value = serde_json::from_slice(&fs::read(&spec.config_template)?)?;
    let substitutions = BTreeMap::from([
        ("{base_url}", json!(base_url)),
        ("{journal_dir}", json!(journal)),
        ("{artifact_dir}", json!(artifacts)),
        ("{log_file}", json!(log_file)),
        ("{workdir}", json!(task.workdir)),
        ("{model}", json!(manifest.server.model)),
        ("{context_window}", json!(manifest.server.context_window)),
        ("{max_tokens}", json!(manifest.server.max_tokens)),
        ("{tool_result_format}", json!(spec.tool_result_format)),
        (
            "{request_timeout_secs}",
            json!(manifest.server.request_timeout_secs),
        ),
    ]);
    let mut rendered = render(&template, &substitutions)?;
    bind_tool_result_format(&mut rendered, spec.tool_result_format)?;
    validate_config(
        &rendered,
        &manifest.server,
        spec.tool_result_format,
        base_url,
        &journal,
        &artifacts,
        &log_file,
    )?;
    let config_path = output.join("orchestral-config.json");
    write_json(&config_path, &rendered)?;
    write_json(
        output.join("orchestral-setup.json"),
        &json!({
            "session_id":session_id,"home":home,"journal":journal,"config":config_path,
            "config_sha256":config::hash_file(&config_path)?,"template":spec.config_template,
            "endpoint":base_url,"validation_repairs":0,"tool_result_format":spec.tool_result_format,
            "thinking_note":"No HTTP body rewriting. Thinking is the actual product/server configuration, not inferred from the Pi thinking field."
        }),
    )?;
    Ok(Prepared {
        config_path,
        home,
        journal,
        session_id,
    })
}

fn render(value: &Value, substitutions: &BTreeMap<&str, Value>) -> Result<Value> {
    Ok(match value {
        Value::String(text) => {
            if let Some(value) = substitutions.get(text.as_str()) {
                value.clone()
            } else {
                ensure!(
                    !substitutions.keys().any(|key| text.contains(key)),
                    "configuration placeholders must occupy a complete JSON value"
                );
                value.clone()
            }
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|v| render(v, substitutions))
                .collect::<Result<_>>()?,
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(k, v)| Ok((k.clone(), render(v, substitutions)?)))
                .collect::<Result<_>>()?,
        ),
        _ => value.clone(),
    })
}

fn validate_config(
    config: &Value,
    server: &Server,
    tool_result_format: OrchestralToolResultFormat,
    base_url: &str,
    journal: &Path,
    artifacts: &Path,
    log_file: &Path,
) -> Result<()> {
    ensure!(
        config["version"] == 1,
        "Orchestral config version must be 1"
    );
    let backends = config["providers"]["backends"]
        .as_array()
        .context("backend array")?;
    let models = config["providers"]["models"]
        .as_array()
        .context("model array")?;
    ensure!(
        backends.len() == 1 && models.len() == 1,
        "one explicit local OpenAI backend/model required"
    );
    let backend = &backends[0];
    let model = &models[0];
    let actual_format = model["config"]
        .get("tool_result_format")
        .map(|value| serde_json::from_value::<OrchestralToolResultFormat>(value.clone()))
        .transpose()
        .context("invalid Orchestral tool result format")?
        .unwrap_or_default();
    ensure!(
        actual_format == tool_result_format,
        "Orchestral tool result format differs from manifest"
    );
    ensure!(
        backend["kind"] == "openai"
            && backend["endpoint"] == base_url
            && backend["config"]["auth"] == "none",
        "backend must use its assigned unauthenticated task proxy"
    );
    ensure!(
        model["backend"] == backend["name"]
            && config["agent"]["backend"] == backend["name"]
            && config["providers"]["default_backend"] == backend["name"]
            && config["agent"]["model_profile"] == model["name"]
            && config["providers"]["default_model"] == model["name"],
        "contradictory selected backend/model"
    );
    ensure!(
        model["model"] == server.model
            && model["max_tokens"] == server.max_tokens
            && config["agent"]["reserved_output_tokens"] == server.max_tokens
            && config["agent"]["max_context_tokens"] == server.context_window
            && backend["config"]["max_context_tokens"] == server.context_window
            && backend["config"]["stream_idle_timeout_secs"] == server.request_timeout_secs,
        "Orchestral model/context/output/timeout differs from manifest"
    );
    let mut actual = match model["config"].get("sampling") {
        Some(sampling) => sampling
            .as_object()
            .context("native sampling must be an object")?
            .clone(),
        None => serde_json::Map::new(),
    };
    ensure!(
        model.get("temperature").is_some(),
        "explicit temperature required"
    );
    actual.insert("temperature".into(), model["temperature"].clone());
    for key in actual.keys() {
        ensure!(
            [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "seed"
            ]
            .contains(&key.as_str()),
            "unsupported native Orchestral sampling field {key}"
        );
    }
    let declared: serde_json::Map<String, Value> =
        server.sampling_params.clone().into_iter().collect();
    ensure!(actual == declared, "Orchestral sampling must equal the explicit manifest parameters; unsupported fields cannot be injected");
    ensure!(
        matches!(
            config["journal"]["backend"].as_str(),
            Some("filesystem" | "fs")
        ) && config["journal"]["root_dir"] == json!(journal),
        "public journal must use its private run directory"
    );
    ensure!(
        matches!(
            config["artifacts"]["backend"].as_str(),
            Some("filesystem" | "fs")
        ) && config["artifacts"]["root_dir"] == json!(artifacts)
            && config["observability"]["log_file"] == json!(log_file),
        "mutable artifacts/logs must stay outside candidate source"
    );
    ensure!(
        config["mcp"]["enabled"] == false
            && config["skills"]["enabled"] == false
            && config["tools"]["exec"]["allow_host_execution"] == false,
        "headless regression requires MCP/skills disabled and sandboxed execution"
    );
    Ok(())
}

fn bind_tool_result_format(config: &mut Value, format: OrchestralToolResultFormat) -> Result<()> {
    let models = config
        .get_mut("providers")
        .and_then(|providers| providers.get_mut("models"))
        .and_then(Value::as_array_mut)
        .context("model array")?;
    ensure!(
        models.len() == 1,
        "one explicit local OpenAI model required"
    );
    let model = models[0]
        .as_object_mut()
        .context("model profile object required")?;
    let profile = model.entry("config").or_insert_with(|| json!({}));
    if profile.is_null() {
        *profile = json!({});
    }
    let profile = profile
        .as_object_mut()
        .context("model profile config object required")?;
    if let Some(value) = profile.get("tool_result_format") {
        ensure!(
            serde_json::from_value::<OrchestralToolResultFormat>(value.clone())? == format,
            "Orchestral template tool result format conflicts with manifest"
        );
    }
    profile.insert("tool_result_format".into(), json!(format));
    Ok(())
}

fn argv(config: &Path, session: &str, prompt: &str) -> Result<Vec<String>> {
    Ok(vec![
        "--config".into(),
        config.to_str().context("UTF-8 config path")?.into(),
        "--no-mcp".into(),
        "--no-skills".into(),
        "--session-id".into(),
        session.into(),
        "--".into(),
        prompt.into(),
    ])
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_task(
    manifest: &Manifest,
    task: &Task,
    output: PathBuf,
    client: Prepared,
    before: BTreeMap<PathBuf, String>,
    clock: Instant,
    barrier: Option<Arc<Barrier>>,
    protected: &[PathBuf],
    frozen: &BTreeMap<PathBuf, String>,
) -> Result<TaskResult> {
    // Every fallible per-process action happens after all prepared participants
    // cross the barrier, so one failed launch cannot strand its peers at it.
    if let Some(barrier) = barrier {
        barrier.wait().await;
    }
    let prompt = fs::read_to_string(&task.prompt_file)?;
    let args = argv(&client.config_path, &client.session_id, &prompt)?;
    write_json(
        output.join("command.json"),
        &json!({"program":manifest.program(),"args":args,
        "cwd":task.workdir,"ORCHESTRAL_HOME":client.home,"session_id":client.session_id}),
    )?;
    let mut command = Command::new(manifest.program());
    process::isolated_local(&mut command);
    command
        .args(&args)
        .current_dir(&task.workdir)
        .env("ORCHESTRAL_HOME", &client.home)
        .stdin(Stdio::null())
        .stdout(File::create(output.join("orchestral-stdout.txt"))?)
        .stderr(File::create(output.join("orchestral-stderr.txt"))?);
    let started_ns = clock.elapsed().as_nanos() as u64;
    let start = Instant::now();
    let budget = Duration::from_secs(task.timeout_secs);
    let mut child = ManagedChild::spawn(&mut command)?;
    let pid = child.pid;
    if let Err(error) = write_json(
        output.join("process-start.json"),
        &json!({"pid":pid,
        "started_ns":started_ns,"session_id":client.session_id}),
    ) {
        child.interrupt_then_stop().await;
        return Err(error);
    }
    let (exit_code, timed_out) = match tokio::time::timeout(budget, child.child.wait()).await {
        Ok(Ok(status)) => (status.code(), false),
        Ok(Err(error)) => {
            child.interrupt_then_stop().await;
            return Err(error.into());
        }
        Err(_) => {
            child.interrupt_then_stop().await;
            (None, true)
        }
    };
    // Reaping the leader alone does not close its process group. Ensure any
    // remaining owned tools cannot edit the candidate during independent validation.
    if !timed_out {
        child.stop().await;
    }
    let finished_ns = clock.elapsed().as_nanos() as u64;
    let process = Outcome {
        exit_code,
        timed_out,
        elapsed_ms: start.elapsed().as_millis(),
    };
    write_json(output.join("process-result.json"), &process)?;
    let mut evidence = orchestral_evidence::read(&client.journal, &client.session_id);
    if evidence.input.as_deref() != Some(prompt.as_str()) {
        evidence
            .errors
            .push("public committed input differs from the frozen prompt".into());
    }
    let validation_source =
        runner::source_snapshot(&task.workdir, &output.join("validation-source"))?;
    let validation = runner::validate_within(
        task,
        &client.home,
        &output.join("final-validation"),
        budget.saturating_sub(start.elapsed()),
    )
    .await?;
    let validation_within_deadline = start.elapsed() <= budget;
    let after = runner::source_snapshot(&task.workdir, &output.join("source-after"))?;
    if validation_source != after {
        evidence
            .errors
            .push("candidate changed during independent validation".into());
    }
    match config::snapshot(protected) {
        Ok(current) if current == *frozen => {}
        Ok(_) => evidence.errors.push("protected inputs changed".into()),
        Err(error) => evidence
            .errors
            .push(format!("protected inputs could not be read: {error:#}")),
    }
    let source_changed = before != after;
    let completed = !timed_out
        && exit_code == Some(0)
        && source_changed
        && evidence.complete()
        && !evidence.tool_exchanges.is_empty()
        && !validation.timed_out
        && validation.exit_code == Some(0)
        && validation_within_deadline;
    let result = TaskResult {
        id: task.id.clone(),
        pid,
        workdir: task.workdir.clone(),
        started_ns,
        finished_ns,
        process,
        events: None,
        validation: Some(validation),
        source_changed,
        closed_loop: false,
        replay: None,
        orchestral: Some(evidence),
        orchestral_transport: None,
        completed,
        validation_repair: None,
    };
    write_json(output.join("result.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_and_paths_remain_literal_arguments_and_json_values() {
        let prompt = "Fix `code`; $(touch forbidden)\n中文 {base_url}";
        let args = argv(
            Path::new("/config with spaces.json"),
            "fresh-session",
            prompt,
        )
        .unwrap();
        assert_eq!(args.last().unwrap(), prompt);
        assert_eq!(&args[args.len() - 2], "--");
        let map = BTreeMap::from([("{base_url}", json!("http://127.0.0.1:42/_agents/a/v1"))]);
        assert_eq!(
            render(&json!({"endpoint":"{base_url}"}), &map).unwrap()["endpoint"],
            map["{base_url}"]
        );
        assert!(render(&json!("prefix/{base_url}"), &map).is_err());
    }

    fn config_fixture() -> (Value, Server) {
        let server = Server {
            base_url: "http://127.0.0.1:8001/v1".into(),
            model: "local".into(),
            context_window: 4096,
            max_tokens: 512,
            request_timeout_secs: 30,
            reasoning: false,
            thinking: "off".into(),
            sampling_params: BTreeMap::from([
                ("temperature".into(), json!(0.6)),
                ("top_k".into(), json!(20)),
            ]),
        };
        let value = json!({"version":1,"agent":{"backend":"local","model_profile":"local","max_context_tokens":4096,"reserved_output_tokens":512},
            "providers":{"default_backend":"local","default_model":"local","backends":[{"name":"local","kind":"openai","endpoint":"http://127.0.0.1:42/_agents/a/v1","config":{"auth":"none","max_context_tokens":4096,"stream_idle_timeout_secs":30}}],
                "models":[{"name":"local","backend":"local","model":"local","max_tokens":512,"temperature":0.6,"config":{"sampling":{"top_k":20}}}]},
            "journal":{"backend":"filesystem","root_dir":"/journal"},"artifacts":{"backend":"filesystem","root_dir":"/artifacts"},
            "observability":{"log_file":"/log"},"mcp":{"enabled":false},"skills":{"enabled":false},"tools":{"exec":{"allow_host_execution":false}}});
        (value, server)
    }

    #[test]
    fn native_configuration_cannot_bypass_proxy_or_silently_change_budgets_and_sampling() {
        let (value, server) = config_fixture();
        let check = |value: &Value, server: &Server| {
            validate_config(
                value,
                server,
                OrchestralToolResultFormat::Json,
                "http://127.0.0.1:42/_agents/a/v1",
                Path::new("/journal"),
                Path::new("/artifacts"),
                Path::new("/log"),
            )
        };
        check(&value, &server).unwrap();
        let mut changed = value.clone();
        changed["providers"]["backends"][0]["endpoint"] = json!("http://127.0.0.1:8001/v1");
        assert!(check(&changed, &server).is_err());
        let mut changed = value.clone();
        changed["providers"]["models"][0]["max_tokens"] = json!(8192);
        assert!(check(&changed, &server).is_err());
        let mut changed = value.clone();
        changed["providers"]["models"][0]["config"]["sampling"]["top_k"] = json!(1);
        assert!(check(&changed, &server).is_err());
        let mut changed = server;
        changed.sampling_params.insert(
            "chat_template_kwargs".into(),
            json!({"enable_thinking":false}),
        );
        assert!(check(&value, &changed).is_err());
    }

    #[test]
    fn declared_tool_result_format_binds_the_profile_without_overriding_conflicts() {
        let (original, server) = config_fixture();
        for format in [
            OrchestralToolResultFormat::Json,
            OrchestralToolResultFormat::Yaml,
        ] {
            let mut value = original.clone();
            bind_tool_result_format(&mut value, format).unwrap();
            assert_eq!(
                value["providers"]["models"][0]["config"]["tool_result_format"],
                json!(format)
            );
            validate_config(
                &value,
                &server,
                format,
                "http://127.0.0.1:42/_agents/a/v1",
                Path::new("/journal"),
                Path::new("/artifacts"),
                Path::new("/log"),
            )
            .unwrap();
            bind_tool_result_format(&mut value, format).unwrap();
        }
        let mut explicit = original.clone();
        explicit["providers"]["models"][0]["config"]["tool_result_format"] = json!("json");
        assert!(bind_tool_result_format(&mut explicit, OrchestralToolResultFormat::Yaml).is_err());
        assert!(validate_config(
            &explicit,
            &server,
            OrchestralToolResultFormat::Yaml,
            "http://127.0.0.1:42/_agents/a/v1",
            Path::new("/journal"),
            Path::new("/artifacts"),
            Path::new("/log"),
        )
        .is_err());
        let mut template = original;
        template["providers"]["models"][0]["config"]["tool_result_format"] =
            json!("{tool_result_format}");
        let mut rendered = render(
            &template,
            &BTreeMap::from([("{tool_result_format}", json!("yaml"))]),
        )
        .unwrap();
        bind_tool_result_format(&mut rendered, OrchestralToolResultFormat::Yaml).unwrap();
        assert!(bind_tool_result_format(
            &mut json!({"providers":false}),
            OrchestralToolResultFormat::Json
        )
        .is_err());
    }

    #[test]
    fn missing_or_null_profile_config_preserves_empty_sampling_defaults() {
        let (original, mut server) = config_fixture();
        server.sampling_params.remove("top_k");
        for config in [None, Some(Value::Null)] {
            let mut value = original.clone();
            let model = value["providers"]["models"][0].as_object_mut().unwrap();
            model.remove("config");
            if let Some(config) = config {
                model.insert("config".into(), config);
            }
            bind_tool_result_format(&mut value, OrchestralToolResultFormat::Json).unwrap();
            validate_config(
                &value,
                &server,
                OrchestralToolResultFormat::Json,
                "http://127.0.0.1:42/_agents/a/v1",
                Path::new("/journal"),
                Path::new("/artifacts"),
                Path::new("/log"),
            )
            .unwrap();
            assert_eq!(
                value["providers"]["models"][0]["config"],
                json!({"tool_result_format":"json"})
            );
        }
        let mut invalid = original;
        invalid["providers"]["models"][0]["config"] = json!(false);
        assert!(bind_tool_result_format(&mut invalid, OrchestralToolResultFormat::Json).is_err());
    }
}
