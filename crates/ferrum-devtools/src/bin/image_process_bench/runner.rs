use super::{
    config::{bounded_read, hash_bytes, Prepared},
    Args,
};
use crate::process_group::ProcessGroup;
use anyhow::{Context, Result};
use ferrum_bench_core::{
    images::{validate_reference_png, ImageSample, ValidatedImage},
    percentile,
};
use futures::{stream::FuturesUnordered, StreamExt};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::process::{CommandExt, ExitStatusExt},
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{process::Command, sync::watch};

#[derive(Clone)]
pub(super) struct Cancellation(watch::Sender<bool>);
impl Cancellation {
    pub fn new() -> Self {
        Self(watch::channel(false).0)
    }
    pub fn cancel(&self) {
        self.0.send_replace(true);
    }
    fn requested(&self) -> bool {
        *self.0.borrow()
    }
    async fn wait(&self) {
        let mut receiver = self.0.subscribe();
        while !*receiver.borrow_and_update() {
            if receiver.changed().await.is_err() {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Warmup,
    Measured,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Warmup => "warmup",
            Self::Measured => "measured",
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Record {
    pub phase: Phase,
    pub repetition: usize,
    pub sample_index: usize,
    pub sample_id: String,
    pub directory: PathBuf,
    pub pid: Option<u32>,
    pub spawn_offset_us: Option<u128>,
    pub exit_observed_offset_us: Option<u128>,
    pub spawn_to_exit_ms: Option<f64>,
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
    pub term_requested: bool,
    pub cleanup_error: Option<String>,
    pub validation_ms: Option<f64>,
    pub image: Option<ValidatedImage>,
    pub error: Option<String>,
}
impl Record {
    fn new(
        phase: Phase,
        repetition: usize,
        index: usize,
        sample: &ImageSample,
        directory: PathBuf,
    ) -> Self {
        Self {
            phase,
            repetition,
            sample_index: index,
            sample_id: sample.id.clone(),
            directory,
            pid: None,
            spawn_offset_us: None,
            exit_observed_offset_us: None,
            spawn_to_exit_ms: None,
            exit_code: None,
            exit_signal: None,
            timed_out: false,
            cancelled: false,
            term_requested: false,
            cleanup_error: None,
            validation_ms: None,
            image: None,
            error: None,
        }
    }
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0)
            && self.image.is_some()
            && self.error.is_none()
            && self.cleanup_error.is_none()
            && !self.timed_out
            && !self.cancelled
    }
    fn fail(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.error = Some(match self.error.take() {
            Some(previous) => format!("{previous}; {message}"),
            None => message,
        });
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Metrics {
    pub planned: usize,
    pub spawned: usize,
    pub succeeded: usize,
    pub errors: usize,
    pub round_wall_seconds: f64,
    pub images_per_second: f64,
    pub process_wall_p50_ms: Option<f64>,
    pub process_wall_p99_ms: Option<f64>,
    pub maximum_observed_live_processes: usize,
}
impl Metrics {
    fn from_records(records: &[&Record], wall: f64) -> Self {
        let values: Vec<_> = records
            .iter()
            .filter(|r| r.succeeded())
            .filter_map(|r| r.spawn_to_exit_ms)
            .collect();
        // These are observed process-lifetime intervals, not GPU overlap. End
        // events sort before start events at equal timestamps.
        let mut events = Vec::new();
        for record in records {
            if let (Some(start), Some(end)) =
                (record.spawn_offset_us, record.exit_observed_offset_us)
            {
                if end > start {
                    events.extend([(start, 1_i32), (end, -1)]);
                }
            }
        }
        events.sort_unstable();
        let (mut live, mut maximum) = (0_i32, 0_i32);
        for (_, delta) in events {
            live += delta;
            maximum = maximum.max(live);
        }
        Self {
            planned: records.len(),
            spawned: records.iter().filter(|r| r.pid.is_some()).count(),
            succeeded: values.len(),
            errors: records.len() - values.len(),
            round_wall_seconds: wall,
            images_per_second: if wall > 0.0 {
                values.len() as f64 / wall
            } else {
                0.0
            },
            process_wall_p50_ms: (!values.is_empty()).then(|| percentile(&values, 0.5)),
            process_wall_p99_ms: (!values.is_empty()).then(|| percentile(&values, 0.99)),
            maximum_observed_live_processes: maximum as usize,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Round {
    pub phase: Phase,
    pub repetition: usize,
    pub metrics: Metrics,
}

#[derive(Debug, Serialize)]
pub(super) struct Report {
    pub schema_version: u32,
    pub measurement: &'static str,
    pub started_unix_ms: u128,
    pub template_sha256: String,
    pub dataset_file_sha256: String,
    pub semantic_dataset_sha256: String,
    pub template: super::config::Template,
    pub reference_image: Option<ValidatedImage>,
    pub concurrency: usize,
    pub repetitions: usize,
    pub warmup_repetitions: usize,
    pub timeout_seconds: u64,
    pub termination_grace_seconds: u64,
    pub planned_requests: usize,
    pub overall_wall_seconds: f64,
    pub measured: Metrics,
    pub rounds: Vec<Round>,
    pub records: Vec<Record>,
    pub cancelled: bool,
    pub integrity_errors: Vec<String>,
    pub scope: serde_json::Value,
}
impl Report {
    pub fn succeeded(&self) -> bool {
        !self.cancelled
            && self.integrity_errors.is_empty()
            && self.records.len() == self.planned_requests
            && self.records.iter().all(Record::succeeded)
    }
}

pub(super) fn write_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn raw_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().create_new(true).write(true).open(path)?)
}

pub(super) async fn execute(args: Args) -> Result<Report> {
    use tokio::signal::unix::{signal, SignalKind};
    // Installed before any model child is spawned.
    let mut term = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let cancellation = Cancellation::new();
    let observed = cancellation.clone();
    struct SignalTask(tokio::task::JoinHandle<()>);
    impl Drop for SignalTask {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _signals = SignalTask(tokio::spawn(async move {
        tokio::select! { _ = term.recv() => {}, _ = interrupt.recv() => {} }
        observed.cancel();
    }));
    run(args, cancellation).await
}

pub(super) async fn run(args: Args, cancellation: Cancellation) -> Result<Report> {
    let prepared = Prepared::load(&args)?;
    fs::create_dir(&args.report_dir).context("report directory must be new")?;
    let directory = args.report_dir.canonicalize()?;
    raw_file(&directory.join("command-template.original.json"))?
        .write_all(&prepared.template_bytes)?;
    raw_file(&directory.join("dataset.original.json"))?.write_all(&prepared.dataset_bytes)?;
    let mut journal = raw_file(&directory.join("requests.jsonl"))?;
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let overall = Instant::now();
    let mut records = Vec::new();
    let mut rounds = Vec::new();
    let mut integrity_errors = Vec::new();
    let mut persistence_error = None;
    for phase in [Phase::Warmup, Phase::Measured] {
        let repetitions = if phase == Phase::Warmup {
            args.warmup_repetitions
        } else {
            args.repetitions
        };
        for repetition in 0..repetitions {
            if let Err(error) = prepared.verify() {
                integrity_errors.push(format!("before {phase:?}/{repetition}: {error:#}"));
                cancellation.cancel();
            }
            let start = Instant::now();
            let first_record = records.len();
            let mut next = 0;
            let mut active = FuturesUnordered::new();
            while next < prepared.dataset.samples.len() || !active.is_empty() {
                while next < prepared.dataset.samples.len() && active.len() < args.concurrency {
                    active.push(run_one(
                        &args,
                        &prepared,
                        &directory,
                        &cancellation,
                        phase,
                        repetition,
                        next,
                        overall,
                    ));
                    next += 1;
                }
                if let Some(record) = active.next().await {
                    // Persistence errors cancel admission but still drain every
                    // owned child; propagating here would abandon async cleanup.
                    let persisted = (|| -> Result<()> {
                        write_new(&record.directory.join("result.json"), &record)?;
                        serde_json::to_writer(&mut journal, &record)?;
                        journal.write_all(b"\n")?;
                        journal.sync_data()?;
                        Ok(())
                    })();
                    if let Err(error) = persisted {
                        persistence_error
                            .get_or_insert(format!("persist request evidence: {error:#}"));
                        cancellation.cancel();
                    }
                    records.push(record);
                }
            }
            let wall = start.elapsed().as_secs_f64();
            let round_records: Vec<_> = records[first_record..].iter().collect();
            rounds.push(Round {
                phase,
                repetition,
                metrics: Metrics::from_records(&round_records, wall),
            });
            if let Err(error) = prepared.verify() {
                integrity_errors.push(format!("after {phase:?}/{repetition}: {error:#}"));
                cancellation.cancel();
            }
        }
    }
    if let Some(error) = persistence_error {
        integrity_errors.push(error);
    }
    let measured_records: Vec<_> = records
        .iter()
        .filter(|r| r.phase == Phase::Measured)
        .collect();
    let measured_wall = rounds
        .iter()
        .filter(|r| r.phase == Phase::Measured)
        .map(|r| r.metrics.round_wall_seconds)
        .sum();
    let report = Report {
        schema_version: 1,
        measurement: "fresh_process_per_image",
        started_unix_ms,
        template_sha256: hash_bytes(&prepared.template_bytes),
        dataset_file_sha256: hash_bytes(&prepared.dataset_bytes),
        semantic_dataset_sha256: prepared.dataset.fingerprint()?,
        template: prepared.template,
        reference_image: prepared.reference.map(|(_, image)| image),
        concurrency: args.concurrency,
        repetitions: args.repetitions,
        warmup_repetitions: args.warmup_repetitions,
        timeout_seconds: args.timeout_seconds,
        termination_grace_seconds: args.termination_grace_seconds,
        planned_requests: prepared.planned_requests,
        overall_wall_seconds: overall.elapsed().as_secs_f64(),
        measured: Metrics::from_records(&measured_records, measured_wall),
        rounds,
        records,
        cancelled: cancellation.requested(),
        integrity_errors,
        scope: serde_json::json!({
            "process_wall": "immediately before spawn through observed child exit; startup, weight load, inference, image save and child shutdown included; controller PNG validation excluded",
            "round_wall": "all admission/preparation, process failures, group cleanup, PNG validation and durable record IO; request slots retained through validation",
            "overall_wall": "all warm and measured rounds plus inter-round integrity checks; initial input verification excluded",
            "overlap": "observed OS process lifetimes, not GPU execution overlap or model batching",
            "warmup": "earlier fresh processes excluded from measured metrics; OS cache may be warm, no resident model process is reused",
            "quality": "PNG checksums/full decode/exact dimensions only; prompt quality and precision equivalence not established",
            "slo_status": "not_evaluated", "ttft_tpot_itl": "not_applicable",
            "model_process_memory": "unavailable; controller memory is not model memory; an explicitly bound external resource tool may produce separate raw records",
            "environment": "inherited except explicit template env/remove_env; inherited values are not captured",
            "termination": "timeout/cancel requests TERM, waits configured grace, then cleans entire owned group with KILL and reaps leader; forced cleanup is not graceful shutdown evidence"
        }),
    };
    write_new(&directory.join("report.json"), &report)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
async fn run_one(
    args: &Args,
    prepared: &Prepared,
    root: &Path,
    cancellation: &Cancellation,
    phase: Phase,
    repetition: usize,
    index: usize,
    overall: Instant,
) -> Record {
    let sample = &prepared.dataset.samples[index];
    let directory = root.join(format!("{}-{repetition:04}-{index:06}", phase.name()));
    let mut record = Record::new(phase, repetition, index, sample, directory.clone());
    let setup = (|| -> Result<Command> {
        fs::create_dir(&directory)?;
        let argv = prepared
            .template
            .args
            .iter()
            .map(|v| prepared.expand(v, sample, &directory))
            .collect::<Result<Vec<_>>>()?;
        let env = prepared
            .template
            .env
            .iter()
            .map(|(k, v)| Ok((k.clone(), prepared.expand(v, sample, &directory)?)))
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        write_new(
            &directory.join("command.json"),
            &serde_json::json!({
                "program": prepared.template.program, "args": argv, "cwd": prepared.template.cwd,
                "env": env, "remove_env": prepared.template.remove_env, "sample": sample
            }),
        )?;
        let mut command = Command::new(&prepared.template.program.path);
        command
            .args(argv)
            .current_dir(&prepared.template.cwd)
            .envs(env);
        for key in &prepared.template.remove_env {
            command.env_remove(key);
        }
        command
            .stdin(Stdio::null())
            .stdout(raw_file(&directory.join("stdout.log"))?)
            .stderr(raw_file(&directory.join("stderr.log"))?)
            .kill_on_drop(true);
        command.as_std_mut().process_group(0);
        Ok(command)
    })();
    let mut command = match setup {
        Ok(command) => command,
        Err(error) => {
            record.fail(format!("prepare: {error:#}"));
            return record;
        }
    };
    if cancellation.requested() {
        record.cancelled = true;
        record.fail("not started: benchmark cancelled");
        return record;
    }
    let started = Instant::now();
    let offset = overall.elapsed().as_micros();
    let child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            record.fail(format!("spawn: {error}"));
            return record;
        }
    };
    // On Unix a just-spawned live child has a positive pid_t. Keep ownership
    // even if that platform invariant fails, so no child escapes an error path.
    let mut child = child;
    let Some(pid) = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok())
        .filter(|pid| *pid > 0)
    else {
        let _ = child.kill().await;
        let _ = child.wait().await;
        record.fail("spawned child omitted a representable PID");
        return record;
    };
    record.pid = Some(pid as u32);
    record.spawn_offset_us = Some(offset);
    let mut group = ProcessGroup {
        child,
        id: pid,
        armed: true,
    };
    let mut status: Option<ExitStatus> = tokio::select! {
        waited = tokio::time::timeout(Duration::from_secs(args.timeout_seconds), group.child.wait()) => match waited {
            Ok(Ok(status)) => Some(status),
            Ok(Err(error)) => { record.fail(format!("wait: {error}")); None },
            Err(_) => { record.timed_out = true; record.fail("process deadline exceeded"); None },
        },
        _ = cancellation.wait() => { record.cancelled = true; record.fail("benchmark cancelled"); None },
    };
    if status.is_none() {
        record.term_requested = true;
        if let Err(error) = group.signal(libc::SIGTERM) {
            record.fail(error);
        }
        if let Ok(Ok(exited)) = tokio::time::timeout(
            Duration::from_secs(args.termination_grace_seconds),
            group.child.wait(),
        )
        .await
        {
            status = Some(exited);
        }
    }
    if status.is_some() {
        record.exit_observed_offset_us = Some(overall.elapsed().as_micros());
        record.spawn_to_exit_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    }
    if let Err(error) = group.cleanup().await {
        record.cleanup_error = Some(error);
        cancellation.cancel();
    }
    if status.is_none() {
        status = group.child.try_wait().ok().flatten();
        if status.is_some() {
            record.exit_observed_offset_us = Some(overall.elapsed().as_micros());
            record.spawn_to_exit_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    if let Some(status) = status {
        record.exit_code = status.code();
        record.exit_signal = status.signal();
        if !status.success() {
            record.fail(format!("child exited {status}"));
        }
    } else {
        record.fail("child exit was not observed");
    }
    if record.exit_code == Some(0) && !record.timed_out && !record.cancelled {
        let validation = Instant::now();
        let output = directory.join("output.png");
        let limits = prepared.limits.clone();
        let dimensions = (sample.width, sample.height);
        let validated = tokio::task::spawn_blocking(move || -> Result<ValidatedImage> {
            let bytes = bounded_read(&output, limits.maximum_png_bytes)?;
            let image = validate_reference_png(&bytes, &limits)?;
            anyhow::ensure!(
                (image.width, image.height) == dimensions,
                "PNG dimensions differ from sample"
            );
            Ok(image)
        })
        .await;
        record.validation_ms = Some(validation.elapsed().as_secs_f64() * 1000.0);
        match validated {
            Ok(Ok(image)) => record.image = Some(image),
            Ok(Err(error)) => record.fail(format!("PNG validation: {error:#}")),
            Err(error) => record.fail(format!("PNG validation worker: {error}")),
        }
    }
    record
}
