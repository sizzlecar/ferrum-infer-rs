use super::*;
use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub(super) async fn mock(
    replies: Vec<(&'static str, &'static str, Duration)>,
) -> (Client, tokio::task::JoinHandle<Vec<String>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = Url::parse(&format!(
        "http://{}/v1/videos",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    let server = tokio::spawn(async move {
        let mut paths = Vec::new();
        for (status, body, delay) in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut chunk = [0u8; 1024];
                let count = socket.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(offset) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break offset + 4;
                }
            };
            let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
            paths.push(headers.lines().next().unwrap().to_owned());
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while bytes.len() < header_end + content_length {
                let mut chunk = [0u8; 1024];
                let count = socket.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
            }
            tokio::time::sleep(delay).await;
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
        }
        paths
    });
    let client = Client {
        http: reqwest::Client::new(),
        endpoint,
        poll_interval: Duration::from_millis(1),
        timeout: Duration::from_secs(2),
        json_limit: 4096,
        artifact_limit: 1024,
        cancel_on_timeout: false,
    };
    (client, server)
}

pub(super) fn request() -> Request {
    Request {
        path: PathBuf::from("fixture"),
        sha256: String::new(),
        body: serde_json::json!({"size":"832x480", "num_frames":124, "fps":24}),
        expected: super::super::input::Expected {
            width: 832,
            height: 480,
            frames: 124,
            fps: 24.0,
        },
    }
}

#[tokio::test]
async fn follows_one_job_and_times_complete_download_separately() {
    let (client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"job_1","status":"queued"}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"job_1","status":"completed"}"#,
            Duration::ZERO,
        ),
        ("200 OK", "fixture-media", Duration::from_millis(20)),
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let observation = client
        .request(
            &request(),
            directory.path().join("out.mp4"),
            0,
            0,
            Instant::now(),
        )
        .await;
    assert!(observation.error.is_none(), "{:?}", observation.error);
    assert!(!observation.remote_terminal_unknown);
    assert_eq!(observation.artifact_bytes, Some(13));
    assert!(
        observation.artifact_received_seconds.unwrap()
            > observation.terminal_observed_seconds.unwrap()
    );
    assert_eq!(
        server.await.unwrap(),
        vec![
            "POST /v1/videos HTTP/1.1",
            "GET /v1/videos/job_1 HTTP/1.1",
            "GET /v1/videos/job_1/content HTTP/1.1"
        ]
    );
}

#[tokio::test]
async fn failed_job_is_terminal_error_without_download_or_retry() {
    let (client, server) = mock(vec![(
        "200 OK",
        r#"{"id":"job_1","status":"failed","error":{"message":"allocation"}}"#,
        Duration::ZERO,
    )])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let observation = client
        .request(
            &request(),
            directory.path().join("out.mp4"),
            0,
            0,
            Instant::now(),
        )
        .await;
    assert!(observation.error.is_some());
    assert!(!observation.remote_terminal_unknown);
    assert!(observation.artifact.is_none());
    assert_eq!(server.await.unwrap().len(), 1);
}

#[tokio::test]
async fn timeout_delete_does_not_claim_remote_gpu_is_terminal() {
    let (mut client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"job_1","status":"queued"}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"job_1","status":"in_progress"}"#,
            Duration::from_millis(100),
        ),
        ("200 OK", r#"{"id":"job_1","deleted":true}"#, Duration::ZERO),
    ])
    .await;
    client.timeout = Duration::from_millis(30);
    client.cancel_on_timeout = true;
    let directory = tempfile::tempdir().unwrap();
    let observation = client
        .request(
            &request(),
            directory.path().join("out.mp4"),
            0,
            0,
            Instant::now(),
        )
        .await;
    assert!(observation.error.is_some());
    assert!(observation.remote_terminal_unknown);
    assert_eq!(observation.cancellation.unwrap()["deleted"], true);
    assert_eq!(
        server.await.unwrap().last().unwrap(),
        "DELETE /v1/videos/job_1 HTTP/1.1"
    );
}

#[tokio::test]
async fn mismatched_poll_identity_is_not_success() {
    let (client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"job_1","status":"queued"}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"job_2","status":"completed"}"#,
            Duration::ZERO,
        ),
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let observation = client
        .request(
            &request(),
            directory.path().join("out.mp4"),
            0,
            0,
            Instant::now(),
        )
        .await;
    assert!(observation
        .error
        .as_deref()
        .unwrap()
        .contains("does not match"));
    assert!(observation.remote_terminal_unknown);
    assert_eq!(observation.job_id.as_deref(), Some("job_1"));
    server.await.unwrap();
}

#[tokio::test]
async fn content_length_is_checked_before_artifact_allocation() {
    let (mut client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"job_1","status":"completed"}"#,
            Duration::ZERO,
        ),
        ("200 OK", "oversized", Duration::ZERO),
    ])
    .await;
    client.artifact_limit = 1;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("out.mp4");
    let observation = client
        .request(&request(), path.clone(), 0, 0, Instant::now())
        .await;
    assert!(observation
        .error
        .as_deref()
        .unwrap()
        .contains("Content-Length"));
    assert!(!path.exists());
    assert!(!observation.remote_terminal_unknown);
    server.await.unwrap();
}

#[tokio::test]
async fn failed_submission_receipt_retains_job_identity_for_recovery() {
    let (client, server) = mock(vec![(
        "200 OK",
        r#"{"id":"job_1","status":"queued"}"#,
        Duration::ZERO,
    )])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("out.mp4");
    std::fs::create_dir(path.with_extension("submission.json.tmp")).unwrap();
    let observation = client.request(&request(), path, 0, 0, Instant::now()).await;
    assert_eq!(observation.job_id.as_deref(), Some("job_1"));
    assert!(observation.remote_terminal_unknown);
    assert!(observation.evidence_error.is_some());
    assert_eq!(server.await.unwrap().len(), 1);
}

#[tokio::test]
async fn failed_jobs_still_produce_complete_cell_denominator_and_receipts() {
    let (client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"job_1","status":"failed"}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"job_2","status":"failed"}"#,
            Duration::ZERO,
        ),
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut arguments = super::super::Arguments::try_parse_from([
        "test",
        "--base-url",
        "http://localhost:1",
        "--request-json",
        "unused",
        "--provenance",
        "unused",
        "--server-config",
        "unused",
        "--output",
        "unused",
    ])
    .unwrap();
    arguments.output = directory.path().to_owned();
    let cell = run_cell(
        Arc::new(client),
        &arguments,
        &[request()],
        "measurement",
        2,
        2,
        0,
        2,
        None,
    )
    .await
    .unwrap();
    assert_eq!(cell.summary.offered, 2);
    assert_eq!(cell.summary.errors, 2);
    assert!(!cell.summary.observed_slo_met);
    for index in 0..2 {
        let bytes = std::fs::read(directory.path().join(format!(
            "c2-r0-measurement/request-{index}.observation.json"
        )))
        .unwrap();
        let receipt: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(receipt["index"], index);
        assert!(!receipt["error"].is_null());
    }
    server.await.unwrap();
}
