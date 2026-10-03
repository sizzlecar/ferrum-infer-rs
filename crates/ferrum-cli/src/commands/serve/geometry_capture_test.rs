//! Explicit ignored backend diagnostic. Stage the typed geometry-capture-case.json
//! beside the copied libtest binary; the guard owns the fresh working directory.
//! No HTTP/client workload.
use super::*;
use clap::Parser;
use ferrum_engine::geometry_capture::{capture, CaptureOptions};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{io::Read, num::NonZeroU64, path::PathBuf};

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    serve: ServeCliCommand,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCase {
    /// Original serve arguments, excluding executable and the `serve` subcommand.
    serve_argv: Vec<String>,
    cli_config: Option<PathBuf>,
    output: PathBuf,
    report: PathBuf,
    maximum_bytes: NonZeroU64,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a pinned real backend/model/config, applicable native operator lock and exclusive device guard"]
async fn automatic_input_geometry_capture_stops_before_source8() {
    let mut bytes = Vec::new();
    let input_path = std::env::current_exe()
        .unwrap()
        .with_file_name("geometry-capture-case.json");
    std::fs::File::open(input_path)
        .expect("explicit typed geometry capture case")
        .take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(
        bytes.len() <= 64 * 1024,
        "capture input exceeds bounded test metadata"
    );
    let case: CaptureCase = serde_json::from_slice(&bytes).unwrap();
    let input_sha256 = Sha256::digest(&bytes).into();
    assert_ne!(case.output, case.report);
    let command =
        TestCli::try_parse_from(std::iter::once("ferrum".to_owned()).chain(case.serve_argv))
            .unwrap();
    let (config, interleaved) = match &case.cli_config {
        Some(path) => (
            CliConfig::load(path).await.unwrap(),
            crate::config::load_interleaved_system_coalescing(path)
                .await
                .unwrap(),
        ),
        None => (CliConfig::default(), true),
    };
    let (result, report) = capture(
        CaptureOptions {
            output: case.output,
            maximum_bytes: case.maximum_bytes,
            input_sha256,
        },
        execute_cli(command.serve, config, interleaved),
    )
    .await
    .unwrap();
    // Preserve the typed result before asserting; neither Err text nor a guard
    // exit alone grants successful collection/cleanup or a qualified model.
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(case.report)
        .unwrap();
    serde_json::to_writer_pretty(file, &report).unwrap();
    assert!(
        result.is_err(),
        "inventory diagnostic must not expose an HTTP engine"
    );
    assert!(
        report.is_complete(),
        "original inventory or cleanup incomplete: {report:?}; product result: {result:?}"
    );
}
