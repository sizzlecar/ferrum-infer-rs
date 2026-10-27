use super::*;
use config::{hash_bytes, BoundFile, Prepared, Template};
use ferrum_bench_core::images::{ImageDataset, ImageRequestMode, ImageSample};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

fn fixture(directory: &Path, modes: &[&str]) -> Args {
    let executable = std::env::current_exe().unwrap();
    let template = Template {
        schema_version: 1,
        label: "real Rust subprocess fixture".into(),
        model_and_precision: "no model; deterministic 1x1 PNG".into(),
        program: BoundFile {
            sha256: hash_bytes(&fs::read(&executable).unwrap()),
            path: executable,
        },
        cwd: directory.to_owned(),
        args: vec![
            "--ignored".into(),
            "--exact".into(),
            "tests::process_fixture".into(),
            "--nocapture".into(),
        ],
        env: BTreeMap::from([
            ("FERRUM_IMAGE_FIXTURE_MODE".into(), "{prompt}".into()),
            ("FERRUM_IMAGE_FIXTURE_OUTPUT".into(), "{output}".into()),
            ("FERRUM_IMAGE_FIXTURE_ID".into(), "{sample_id}".into()),
            ("FERRUM_IMAGE_FIXTURE_SEED".into(), "{seed}".into()),
            ("FERRUM_IMAGE_FIXTURE_WIDTH".into(), "{width}".into()),
            ("FERRUM_IMAGE_FIXTURE_HEIGHT".into(), "{height}".into()),
            (
                "FERRUM_IMAGE_FIXTURE_ROOT".into(),
                directory.to_str().unwrap().into(),
            ),
        ]),
        remove_env: vec![],
        inputs: BTreeMap::new(),
    };
    let dataset = ImageDataset {
        schema_version: 1,
        mode: ImageRequestMode::Generate,
        samples: modes
            .iter()
            .enumerate()
            .map(|(index, mode)| ImageSample {
                id: format!("sample-{index}"),
                prompt: (*mode).into(),
                seed: index as u64,
                width: 1,
                height: 1,
                num_inference_steps: None,
            })
            .collect(),
    };
    let command_template = directory.join("template.json");
    let dataset_path = directory.join("dataset.json");
    fs::write(&command_template, serde_json::to_vec(&template).unwrap()).unwrap();
    fs::write(&dataset_path, serde_json::to_vec(&dataset).unwrap()).unwrap();
    Args {
        command_template,
        dataset: dataset_path,
        report_dir: directory.join("report"),
        concurrency: 2,
        repetitions: 1,
        warmup_repetitions: 0,
        timeout_seconds: 15,
        termination_grace_seconds: 1,
        maximum_total_requests: 100,
    }
}

#[test]
fn templates_bind_frozen_samples_without_shell_or_recursive_substitution() {
    let directory = tempfile::tempdir().unwrap();
    let args = fixture(
        directory.path(),
        &["$(touch forbidden); {output}\n'quoted'"],
    );
    let prepared = Prepared::load(&args).unwrap();
    let sample = &prepared.dataset.samples[0];
    assert_eq!(
        prepared
            .expand("{prompt}", sample, directory.path())
            .unwrap(),
        sample.prompt
    );
    assert_eq!(
        prepared
            .expand("{{output}}", sample, directory.path())
            .unwrap(),
        "{output}"
    );
    assert!(prepared
        .expand("{unknown}", sample, directory.path())
        .is_err());
    assert!(prepared
        .expand("{steps}", sample, directory.path())
        .is_err());
    assert!(!directory.path().join("forbidden").exists());
    let mut invalid = args;
    invalid.maximum_total_requests = 0;
    assert!(Prepared::load(&invalid).is_err());
}

#[test]
fn bound_configuration_changes_are_rejected_before_a_process_is_started() {
    let directory = tempfile::tempdir().unwrap();
    let args = fixture(directory.path(), &["success"]);
    let path = directory.path().join("model-config.json");
    fs::write(&path, b"original").unwrap();
    let mut template: Template =
        serde_json::from_slice(&fs::read(&args.command_template).unwrap()).unwrap();
    template.inputs.insert(
        "config".into(),
        BoundFile {
            path,
            sha256: hash_bytes(b"original"),
        },
    );
    fs::write(
        &args.command_template,
        serde_json::to_vec(&template).unwrap(),
    )
    .unwrap();
    let prepared = Prepared::load(&args).unwrap();
    fs::write(directory.path().join("model-config.json"), b"changed").unwrap();
    assert!(prepared
        .verify()
        .unwrap_err()
        .to_string()
        .contains("bound bytes changed"));
    assert!(!args.report_dir.exists());
}

// Real child executable, not a shell or Python stub. The generated 1x1 image
// exercises the normal complete PNG validator; a minimal PNG builder keeps the
// fixture independent of model files and avoids adding a decoder dependency.
fn png() -> Vec<u8> {
    use std::io::Write;
    fn chunk(bytes: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
        bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(payload);
        let mut crc = !0_u32;
        for byte in kind.iter().chain(payload) {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb88320 & (0_u32.wrapping_sub(crc & 1)));
            }
        }
        bytes.extend_from_slice(&(!crc).to_be_bytes());
    }
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(
        &mut bytes,
        b"IHDR",
        &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
    );
    let mut compressed = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    compressed.write_all(&[0, 255, 0, 0, 255]).unwrap();
    chunk(&mut bytes, b"IDAT", &compressed.finish().unwrap());
    chunk(&mut bytes, b"IEND", &[]);
    bytes
}

#[test]
#[ignore = "owned child fixture; invoked only by the process benchmark tests"]
fn process_fixture() {
    use std::{
        net::TcpListener,
        time::{Duration, Instant},
    };
    let mode = std::env::var("FERRUM_IMAGE_FIXTURE_MODE").expect("fixture mode");
    let root = PathBuf::from(std::env::var_os("FERRUM_IMAGE_FIXTURE_ROOT").unwrap());
    let output = PathBuf::from(std::env::var_os("FERRUM_IMAGE_FIXTURE_OUTPUT").unwrap());
    let id = std::env::var("FERRUM_IMAGE_FIXTURE_ID").unwrap();
    fs::write(
        root.join(format!("started-{id}")),
        std::process::id().to_string(),
    )
    .unwrap();
    match mode.as_str() {
        "barrier" => {
            let start = Instant::now();
            while fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("started-"))
                .count()
                < 2
            {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "second real process never started"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            // Both processes have reached the admission barrier before either exits.
            std::thread::sleep(Duration::from_millis(50));
        }
        "descendant" => {
            let _child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "tests::process_fixture",
                    "--nocapture",
                ])
                .env("FERRUM_IMAGE_FIXTURE_MODE", "listener")
                .spawn()
                .unwrap();
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        "listener" => {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            fs::write(
                root.join("listener-address"),
                listener.local_addr().unwrap().to_string(),
            )
            .unwrap();
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        "missing" => return,
        "invalid" => {
            fs::write(output, b"not a PNG").unwrap();
            return;
        }
        "nonzero" => {
            eprintln!("deliberate child failure");
            std::process::exit(7);
        }
        "success" => {}
        other => panic!("unsupported fixture {other}"),
    }
    fs::write(output, png()).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn real_cli_processes_overlap_and_every_png_is_validated() {
    let directory = tempfile::tempdir().unwrap();
    let args = fixture(directory.path(), &["barrier", "barrier"]);
    let report = runner::run(args, runner::Cancellation::new())
        .await
        .unwrap();
    assert!(report.succeeded(), "{report:#?}");
    assert_eq!(report.measured.spawned, 2);
    assert_eq!(report.measured.maximum_observed_live_processes, 2);
    let latest_start = report
        .records
        .iter()
        .map(|r| r.spawn_offset_us.unwrap())
        .max()
        .unwrap();
    let first_exit = report
        .records
        .iter()
        .map(|r| r.exit_observed_offset_us.unwrap())
        .min()
        .unwrap();
    assert!(latest_start < first_exit, "C2 must have two live processes");
    for record in &report.records {
        assert_eq!(record.image.as_ref().unwrap().rgba8_bytes, 4);
        assert!(record.cleanup_error.is_none());
    }
    assert_eq!(
        fs::read_to_string(directory.path().join("report/requests.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2
    );
}

#[cfg(unix)]
#[tokio::test]
async fn zero_exit_with_missing_or_invalid_png_and_nonzero_exit_are_errors() {
    let directory = tempfile::tempdir().unwrap();
    let args = fixture(
        directory.path(),
        &["success", "missing", "invalid", "nonzero"],
    );
    let report = runner::run(args, runner::Cancellation::new())
        .await
        .unwrap();
    assert!(!report.succeeded());
    assert_eq!(report.measured.succeeded, 1);
    assert_eq!(report.measured.errors, 3);
    for index in [1, 2] {
        let record = report
            .records
            .iter()
            .find(|r| r.sample_index == index)
            .unwrap();
        assert_eq!(record.exit_code, Some(0));
        assert!(record.error.as_deref().unwrap().contains("PNG validation"));
    }
    let failure = report.records.iter().find(|r| r.sample_index == 3).unwrap();
    assert_eq!(failure.exit_code, Some(7));
    assert!(fs::read_to_string(failure.directory.join("stderr.log"))
        .unwrap()
        .contains("deliberate child failure"));
}

#[cfg(unix)]
#[tokio::test]
async fn warmup_processes_are_recorded_but_excluded_from_measured_metrics() {
    let directory = tempfile::tempdir().unwrap();
    let mut args = fixture(directory.path(), &["success"]);
    args.warmup_repetitions = 1;
    args.repetitions = 2;
    let report = runner::run(args, runner::Cancellation::new())
        .await
        .unwrap();
    assert!(report.succeeded());
    assert_eq!(report.records.len(), 3);
    assert_eq!(report.measured.planned, 2);
    assert_eq!(
        report
            .records
            .iter()
            .filter(|r| r.phase == runner::Phase::Warmup)
            .count(),
        1
    );
    assert!(report.overall_wall_seconds >= report.measured.round_wall_seconds);
}

#[cfg(unix)]
async fn wait_for_listener(path: &Path) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while !path.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "real child listener did not start"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[cfg(unix)]
fn assert_listener_closed(root: &Path) {
    let address = fs::read_to_string(root.join("listener-address"))
        .unwrap()
        .parse()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_millis(30))
        .is_ok()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "owned descendant survived cleanup"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(unix)]
#[tokio::test]
async fn process_deadline_reaps_the_owned_group_including_descendants() {
    let directory = tempfile::tempdir().unwrap();
    let mut args = fixture(directory.path(), &["descendant"]);
    args.timeout_seconds = 2;
    let report = runner::run(args, runner::Cancellation::new())
        .await
        .unwrap();
    assert!(!report.succeeded());
    let record = &report.records[0];
    assert!(record.timed_out && record.term_requested);
    assert!(record.cleanup_error.is_none());
    assert!(record.exit_signal.is_some());
    assert_listener_closed(directory.path());
}

#[cfg(unix)]
#[tokio::test]
async fn cancellation_drains_live_children_and_does_not_admit_remaining_samples() {
    let directory = tempfile::tempdir().unwrap();
    let mut args = fixture(directory.path(), &["descendant", "success"]);
    args.concurrency = 1;
    let cancellation = runner::Cancellation::new();
    let token = cancellation.clone();
    let running = tokio::spawn(async move { runner::run(args, token).await });
    wait_for_listener(&directory.path().join("listener-address")).await;
    cancellation.cancel();
    let report = running.await.unwrap().unwrap();
    assert!(report.cancelled && !report.succeeded());
    assert_eq!(report.measured.spawned, 1);
    assert_eq!(report.records.len(), 2);
    assert!(report
        .records
        .iter()
        .all(|r| r.cancelled && r.cleanup_error.is_none()));
    assert_listener_closed(directory.path());
}
