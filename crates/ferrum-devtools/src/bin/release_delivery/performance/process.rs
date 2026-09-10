//! Child ownership and CLI adaptation; benchmark requests remain in bench-serve.
#[cfg(unix)]
use super::{remaining, write_json};
use super::{
    runtime::{backend_name, is_plan_runtime, ServerIdentity},
    source::Bundle,
    PerformanceArgs,
};
use ferrum_bench_core::release_regression::ExecutionTarget;
#[cfg(any(unix, test))]
use serde_json::json;
use serde_json::Value;
use std::{ffi::OsString, path::Path};
#[cfg(unix)]
use std::{fs, net::TcpListener, process::Stdio, time::Duration};
use tokio::{process::Command, time::Instant};

pub(super) fn clean_command(binary: &Path, cwd: &Path) -> Command {
    let mut command = Command::new(binary);
    command.current_dir(cwd);
    // Stable public CLI parameters own this cell. Do not inherit product tuning
    // or the controller's provider/GitHub credentials into model processes.
    for (key, _) in std::env::vars_os() {
        let text = key.to_string_lossy();
        if text.starts_with("FERRUM_")
            || text.starts_with("GGML_")
            || ["VAST_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"].contains(&text.as_ref())
        {
            command.env_remove(&key);
        }
    }
    command
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost");
    command
}
pub(super) fn server_arguments(
    args: &PerformanceArgs,
    bundle: &Bundle,
    port: u16,
    target: &ExecutionTarget,
) -> Vec<OsString> {
    let mut command = vec![
        "serve".into(),
        bundle.gguf.as_os_str().into(),
        "--backend".into(),
        backend_name(target.backend).into(),
        "--served-model-name".into(),
        "release-perf".into(),
        "--disable-thinking".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string().into(),
    ];
    if !is_plan_runtime(target) {
        command.extend([
            "--kv-dtype".into(),
            "fp16".into(),
            "--kv-capacity".into(),
            args.max_model_len.to_string().into(),
        ]);
    }
    command.extend([
        "--max-model-len".into(),
        args.max_model_len.to_string().into(),
        "--max-num-seqs".into(),
        args.concurrency.to_string().into(),
        "--max-num-batched-tokens".into(),
        args.workload().batched_tokens().to_string().into(),
        "--runtime-memory-budget-bytes".into(),
        args.runtime_memory_budget_bytes.to_string().into(),
        "--disable-prefix-cache".into(),
        "--session-cache".into(),
        "off".into(),
    ]);
    if is_plan_runtime(target) {
        command.extend([
            "--semantic-source".into(),
            bundle.tokenizer_dir.as_os_str().into(),
            "--tokenizer-source".into(),
            bundle.tokenizer_dir.as_os_str().into(),
        ]);
    }
    command
}
pub(super) fn client_arguments(
    args: &PerformanceArgs,
    bundle: &Bundle,
    port: u16,
    report: &Path,
    target: &ExecutionTarget,
) -> Vec<OsString> {
    vec![
        "bench-serve".into(),
        "--base-url".into(),
        format!("http://127.0.0.1:{port}").into(),
        "--model".into(),
        "release-perf".into(),
        "--tokenizer".into(),
        bundle.tokenizer_dir.as_os_str().into(),
        "--target-backend".into(),
        backend_name(target.backend).into(),
        "--dataset".into(),
        "random".into(),
        "--concurrency".into(),
        args.concurrency.to_string().into(),
        "--http-connection-mode".into(),
        "pooled".into(),
        "--random-input-len".into(),
        args.input_tokens.to_string().into(),
        "--random-output-len".into(),
        args.output_tokens.to_string().into(),
        "--ignore-eos".into(),
        "--enable-thinking".into(),
        "false".into(),
        "--seed".into(),
        args.seed.to_string().into(),
        "--num-prompts".into(),
        args.measured_requests.to_string().into(),
        "--warmup-requests".into(),
        args.warmup_requests.to_string().into(),
        "--n-repeats".into(),
        args.repeats.to_string().into(),
        "--require-ci".into(),
        "--fail-on-error".into(),
        "--timeout".into(),
        args.request_timeout_secs.to_string().into(),
        "--output".into(),
        "json".into(),
        "--out".into(),
        report.as_os_str().into(),
    ]
}
#[cfg(unix)]
async fn ready(
    group: &mut super::super::local::ProcessGroup,
    port: u16,
    identity: &ServerIdentity<'_>,
    phase: &Path,
    deadline: Instant,
    terminate: &mut tokio::signal::unix::Signal,
    interrupt: &mut tokio::signal::unix::Signal,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|e| e.to_string())?;
    loop {
        if let Some(status) = group.child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "performance server exited before readiness: {status}"
            ));
        }
        let operation = async {
            if let Ok(response) = client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
            {
                if response.status().is_success() {
                    let health: Value = response.json().await.map_err(|e| e.to_string())?;
                    write_json(&phase.join("health.json"), &health)?;
                    identity.verify(&health)?;
                    return Ok(true);
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<_, String>(false)
        };
        let completed = tokio::select! {
            result=tokio::time::timeout(remaining(deadline)?,operation)=>result.map_err(|_|"performance server startup deadline exceeded".to_string())??,
            _=terminate.recv()=>return Err("performance server startup interrupted by SIGTERM".into()),
            _=interrupt.recv()=>return Err("performance server startup interrupted by SIGINT".into()),
        };
        if completed {
            return Ok(());
        }
    }
}
#[cfg(unix)]
pub(super) async fn phase(
    args: &PerformanceArgs,
    binary: &Path,
    client: &Path,
    identity: &ServerIdentity<'_>,
    bundle: &Bundle,
    phase: &Path,
    deadline: Instant,
) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    use tokio::signal::unix::{signal, SignalKind};
    // Register before either child can own Metal resources.
    let mut terminate = signal(SignalKind::terminate()).map_err(|e| e.to_string())?;
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|e| e.to_string())?;
    let socket = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    let port = socket.local_addr().map_err(|e| e.to_string())?.port();
    let server_args = server_arguments(args, bundle, port, identity.target);
    let bench_args = client_arguments(
        args,
        bundle,
        port,
        &phase.join("bench.json"),
        identity.target,
    );
    write_json(
        &phase.join("commands.json"),
        &json!({"port":port,"server":{"program":binary,"args":server_args},"client":{"program":client,"args":bench_args}}),
    )?;
    let mut command = clean_command(binary, &bundle.tokenizer_dir);
    command
        .args(&server_args)
        .stdin(Stdio::null())
        .stdout(fs::File::create(phase.join("server.stdout.log")).map_err(|e| e.to_string())?)
        .stderr(fs::File::create(phase.join("server.stderr.log")).map_err(|e| e.to_string())?)
        .kill_on_drop(true);
    command.as_std_mut().process_group(0);
    remaining(deadline)?;
    drop(socket);
    let child = command
        .spawn()
        .map_err(|e| format!("spawn performance server: {e}"))?;
    let id = child
        .id()
        .and_then(|id| i32::try_from(id).ok())
        .filter(|id| *id > 0)
        .ok_or("server omitted process group ID")?;
    let mut group = super::super::local::ProcessGroup {
        child,
        id,
        armed: true,
    };
    let mut client_exit_code = None;
    let mut client_cleanup = None;
    let result = async {
        let startup = Instant::now()
            .checked_add(Duration::from_secs(args.startup_timeout_secs))
            .ok_or("startup deadline overflow")?
            .min(deadline);
        ready(&mut group, port, identity, phase, startup, &mut terminate, &mut interrupt).await?;
        let mut bench=clean_command(client,&bundle.tokenizer_dir);
        bench.args(&bench_args).stdin(Stdio::null())
            .stdout(fs::File::create(phase.join("client.stdout.log")).map_err(|e|e.to_string())?)
            .stderr(fs::File::create(phase.join("client.stderr.log")).map_err(|e|e.to_string())?).kill_on_drop(true);
        bench.as_std_mut().process_group(0);
        remaining(deadline)?;
        let child=bench.spawn().map_err(|e|format!("spawn performance benchmark: {e}"))?;
        let id=child.id().and_then(|id|i32::try_from(id).ok()).filter(|id|*id>0).ok_or("benchmark omitted process group ID")?;
        let mut benchmark=super::super::local::ProcessGroup {child,id,armed:true};
        let measured=tokio::select! {
            value=tokio::time::timeout_at(deadline,benchmark.child.wait())=>match value {
                Ok(Ok(status))=>{client_exit_code=status.code();if status.success(){Ok(())}else{Err(format!("benchmark exited {status}"))}},
                Ok(Err(e))=>Err(format!("wait benchmark: {e}")),
                Err(_)=>Err("performance task deadline exceeded during benchmark".into()),
            },
            _=terminate.recv()=>Err("performance benchmark interrupted by SIGTERM".into()),
            _=interrupt.recv()=>Err("performance benchmark interrupted by SIGINT".into()),
        };
        let reaped=benchmark.cleanup().await;
        client_cleanup=Some(reaped.is_ok());
        match (measured,reaped) {
            (Ok(()),Ok(()))=>{},(Err(e),Ok(()))|(Ok(()),Err(e))=>return Err(e),
            (Err(e),Err(c))=>return Err(format!("{e}; benchmark cleanup: {c}")),
        }
        if let Some(status) = group.child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "performance server exited during benchmark: {status}"
            ));
        }
        let health: Value = reqwest::Client::builder().no_proxy()
            .timeout(remaining(deadline)?.min(Duration::from_secs(5)))
            .build().map_err(|e| e.to_string())?
            .get(format!("http://127.0.0.1:{port}/health")).send().await
            .map_err(|e| e.to_string())?.error_for_status().map_err(|e| e.to_string())?
            .json().await.map_err(|e| e.to_string())?;
        write_json(&phase.join("health-after.json"), &health)?;
        let before: Value = super::read_json(&phase.join("health.json"))?;
        identity.verify_after(&before, &health)?;
        Ok(())
    }
    .await;
    let cleanup = group.cleanup().await;
    write_json(
        &phase.join("execution.json"),
        &json!({"server_pid":id,
        "client_exit_code":client_exit_code,"client_cleanup_completed":client_cleanup,
        "client_completed_successfully":result.is_ok(),"error":result.as_ref().err(),
        "cleanup_completed":cleanup.is_ok(),"cleanup_error":cleanup.as_ref().err()}),
    )?;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
        (Err(e), Err(c)) => Err(format!("{e}; cleanup: {c}")),
    }
}
#[cfg(not(unix))]
pub(super) async fn phase(
    _args: &PerformanceArgs,
    _binary: &Path,
    _client: &Path,
    _identity: &ServerIdentity<'_>,
    _bundle: &Bundle,
    _phase: &Path,
    _deadline: Instant,
) -> Result<(), String> {
    Err("Metal performance execution requires Unix process groups".into())
}
