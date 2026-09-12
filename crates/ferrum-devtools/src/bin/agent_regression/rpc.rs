//! Pi JSONL transport. A command acknowledgement is never a completed agent run.
use super::{
    config::{Manifest, Task},
    events::Events,
    process::{self, ManagedChild, Outcome},
    write_json,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs::File,
    io::Write,
    path::Path,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{ChildStdin, Command},
    sync::mpsc,
    task::JoinHandle,
};

pub(crate) struct Agent {
    process: ManagedChild,
    input: Option<ChildStdin>,
    receiver: mpsc::UnboundedReceiver<Value>,
    reader: JoinHandle<Result<Events>>,
    commands: File,
    clock: Instant,
    started: Instant,
    next_command: u64,
    state: RunState,
    session_id: Option<String>,
    session_file: Option<String>,
}

pub(crate) struct Finished {
    pub process: Outcome,
    pub events: Events,
    pub shutdown_timed_out: bool,
}

#[derive(Default)]
struct RunState {
    starts: u64,
    settlements: u64,
    running: bool,
    last_stop: Option<String>,
}
impl RunState {
    fn observe(&mut self, event: &Value) {
        match event["type"].as_str() {
            Some("agent_start") => {
                self.starts += 1;
                self.running = true;
                self.last_stop = None;
            }
            Some("agent_settled") => {
                self.settlements += 1;
                self.running = false;
            }
            Some("message_end") if event["message"]["role"] == "assistant" => {
                self.last_stop = event["message"]["stopReason"].as_str().map(str::to_owned);
            }
            _ => {}
        }
    }
    fn completed_since(&self, starts: u64, settlements: u64) -> bool {
        self.starts > starts && self.settlements > settlements && !self.running
    }
}

pub(crate) fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .context("total task deadline expired")
}

impl Agent {
    pub(crate) fn spawn(
        manifest: &Manifest,
        task: &Task,
        output: &Path,
        agent_dir: &Path,
        clock: Instant,
        started: Instant,
    ) -> Result<Self> {
        let pi = manifest.pi_program()?;
        let mut args = pi.args.clone();
        args.extend(
            [
                "--mode",
                "rpc",
                "--provider",
                "ferrum-local",
                "--model",
                &manifest.server.model,
                "--thinking",
                &manifest.server.thinking,
                "--session-dir",
                output.join("sessions").to_str().context("session path")?,
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--no-context-files",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        write_json(
            output.join("command.json"),
            &json!({"program":pi.program,"args":args,"cwd":task.workdir,"mode":"rpc"}),
        )?;
        let mut command = Command::new(&pi.program);
        process::isolated(&mut command, agent_dir);
        command
            .args(&args)
            .current_dir(&task.workdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(File::create(output.join("pi-stderr.txt"))?);
        let mut process = ManagedChild::spawn(&mut command)?;
        let input = process.child.stdin.take().context("Pi RPC stdin")?;
        let stdout = process.child.stdout.take().context("Pi RPC stdout")?;
        let mut raw = File::create(output.join("pi-stdout.jsonl"))?;
        let mut journal = File::create(output.join("pi-events.jsonl"))?;
        let (sender, receiver) = mpsc::unbounded_channel();
        let reader = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines(); // LF/CRLF only, not Unicode separators.
            let mut events = Events::default();
            while let Some(line) = lines.next_line().await? {
                writeln!(raw, "{line}")?;
                let at = clock.elapsed().as_nanos() as u64;
                match serde_json::from_str::<Value>(&line) {
                    Ok(event) => {
                        events.observe(&event, at);
                        writeln!(journal, "{}", json!({"elapsed_ns":at,"event":event}))?;
                        // Preserve the full journal even after a consumer error.
                        let _ = sender.send(event);
                    }
                    Err(error) => {
                        events
                            .protocol_errors
                            .push(format!("invalid Pi RPC JSON: {error}"));
                        let _ = sender.send(
                            json!({"type":"ferrum_rpc_protocol_error","error":error.to_string()}),
                        );
                    }
                }
            }
            Ok(events)
        });
        Ok(Self {
            process,
            input: Some(input),
            receiver,
            reader,
            commands: File::create(output.join("rpc-commands.jsonl"))?,
            clock,
            started,
            next_command: 0,
            state: RunState::default(),
            session_id: None,
            session_file: None,
        })
    }

    pub(crate) fn pid(&self) -> u32 {
        self.process.pid
    }

    pub(crate) fn unsuccessful_stop_reason(&self) -> Option<&str> {
        self.state
            .last_stop
            .as_deref()
            .filter(|reason| !self.state.running && self.state.settlements > 0 && *reason != "stop")
    }

    async fn next(&mut self, deadline: Instant) -> Result<Value> {
        let event = tokio::time::timeout(remaining(deadline)?, self.receiver.recv())
            .await
            .context("total task deadline expired while awaiting Pi RPC")?
            .context("Pi RPC stdout closed before idle")?;
        ensure!(
            event["type"] != "ferrum_rpc_protocol_error",
            "invalid Pi RPC JSON: {}",
            event["error"]
        );
        self.state.observe(&event);
        Ok(event)
    }

    async fn command(
        &mut self,
        kind: &str,
        mut command: Value,
        deadline: Instant,
    ) -> Result<Value> {
        let id = format!("validation-rpc-{}", self.next_command);
        self.next_command += 1;
        command["id"] = id.clone().into();
        command["type"] = kind.into();
        writeln!(
            self.commands,
            "{}",
            json!({"elapsed_ns":self.clock.elapsed().as_nanos(),"command":command})
        )?;
        self.commands.flush()?;
        let mut bytes = serde_json::to_vec(&command)?;
        bytes.push(b'\n');
        tokio::time::timeout(remaining(deadline)?, async {
            let input = self.input.as_mut().context("Pi RPC stdin already closed")?;
            input.write_all(&bytes).await?;
            input.flush().await?;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("total task deadline expired sending RPC")??;
        loop {
            let event = self.next(deadline).await?;
            if event["type"] == "response" {
                ensure!(
                    event["id"] == id && event["command"] == kind,
                    "unmatched Pi RPC response: {event}"
                );
                ensure!(event["success"] == true, "Pi RPC command rejected: {event}");
                return Ok(event["data"].clone());
            }
        }
    }

    pub(crate) async fn idle(&mut self, manifest: &Manifest, deadline: Instant) -> Result<Value> {
        let state = self.command("get_state", json!({}), deadline).await?;
        ensure!(
            idle_state(&state) && !self.state.running,
            "Pi did not confirm an idle session: {state}"
        );
        let id = state["sessionId"]
            .as_str()
            .filter(|id| !id.is_empty())
            .context("Pi get_state lacks session id")?;
        ensure!(
            self.session_id
                .as_deref()
                .is_none_or(|previous| previous == id),
            "Pi changed session during validation repair"
        );
        self.session_id = Some(id.to_owned());
        if let Some(file) = state["sessionFile"].as_str() {
            ensure!(
                self.session_file
                    .as_deref()
                    .is_none_or(|previous| previous == file),
                "Pi changed the persisted session file"
            );
            self.session_file = Some(file.to_owned());
        } else {
            ensure!(
                self.session_file.is_none(),
                "Pi lost its persisted session file"
            );
        }
        ensure!(
            state["model"]["id"] == manifest.server.model
                && state["model"]["provider"] == "ferrum-local",
            "Pi get_state differs from the declared local model/provider"
        );
        Ok(state)
    }

    pub(crate) async fn prompt(
        &mut self,
        message: &str,
        manifest: &Manifest,
        deadline: Instant,
    ) -> Result<Value> {
        self.idle(manifest, deadline).await?;
        let starts = self.state.starts;
        let settlements = self.state.settlements;
        self.command("prompt", json!({"message":message}), deadline)
            .await?;
        while !self.state.completed_since(starts, settlements) {
            self.next(deadline).await?;
        }
        ensure!(
            self.state.last_stop.as_deref() == Some("stop"),
            "Pi settled without a successful final assistant: {:?}",
            self.state.last_stop
        );
        self.idle(manifest, deadline).await
    }

    pub(crate) async fn finish(mut self, deadline: Instant, abort: bool) -> Result<Finished> {
        // EOF is Pi's documented same-session RPC shutdown. On an error/timeout
        // stop its process group first; cleanup grace never authorizes more work.
        if abort && self.process.child.try_wait()?.is_none() {
            self.process.stop().await;
        }
        drop(self.input.take());
        let mut shutdown_timed_out = false;
        let status =
            match tokio::time::timeout(Duration::from_secs(2), self.process.child.wait()).await {
                Ok(status) => status?,
                Err(_) => {
                    self.process.stop().await;
                    shutdown_timed_out = true;
                    self.process.child.wait().await?
                }
            };
        drop(self.receiver);
        let events = tokio::time::timeout(Duration::from_secs(10), self.reader)
            .await
            .context("Pi RPC output did not close")???;
        Ok(Finished {
            process: Outcome {
                exit_code: status.code(),
                timed_out: Instant::now() >= deadline,
                elapsed_ms: self.started.elapsed().as_millis(),
            },
            events,
            shutdown_timed_out,
        })
    }
}

fn idle_state(state: &Value) -> bool {
    state["isStreaming"] == false
        && state["isCompacting"] == false
        && state["pendingMessageCount"] == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_ack_and_agent_end_are_not_settled_or_idle() {
        let mut run = RunState::default();
        run.observe(&json!({"type":"response","command":"prompt","success":true}));
        run.observe(&json!({"type":"agent_start"}));
        run.observe(&json!({"type":"agent_end","willRetry":true}));
        assert!(!run.completed_since(0, 0));
        run.observe(&json!({"type":"agent_settled"}));
        assert!(run.completed_since(0, 0));
        assert!(!run.completed_since(run.starts, run.settlements));
        run.last_stop = Some("stop".into());
        run.observe(&json!({"type":"agent_start"}));
        assert!(!run.completed_since(0, 0));
        assert!(run.last_stop.is_none());
        assert!(!idle_state(
            &json!({"isStreaming":false,"isCompacting":true,"pendingMessageCount":0})
        ));
        assert!(!idle_state(
            &json!({"isStreaming":false,"isCompacting":false,"pendingMessageCount":1})
        ));
        assert!(!idle_state(&json!({})));
    }
    #[test]
    fn later_rounds_cannot_replace_an_expired_absolute_deadline() {
        let deadline = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let mut run = RunState::default();
        run.observe(&json!({"type":"agent_start"}));
        run.observe(&json!({"type":"agent_settled"}));
        assert!(run.completed_since(0, 0));
        assert!(remaining(deadline).is_err());
        run.observe(&json!({"type":"agent_start"}));
        assert!(remaining(deadline).is_err());
    }
}
