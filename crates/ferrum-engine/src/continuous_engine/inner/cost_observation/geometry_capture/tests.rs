use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::input_geometry_pivots_v1;

fn directory() -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("ferrum-geometry-capture-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&path).unwrap();
    path
}
fn options(path: PathBuf) -> CaptureOptions {
    CaptureOptions {
        output: path,
        maximum_bytes: NonZeroU64::new(1024 * 1024).unwrap(),
        input_sha256: [0; 32],
    }
}
fn retirement() {
    inventory_retired(
        true,
        tokio::time::Instant::now() + std::time::Duration::from_secs(1),
        5,
        7,
        3,
        4,
    );
    series_retired(true);
    shutdown_finished(true);
}
fn original_calls() {
    let settings = StructuredSettingsV2::default();
    let data = [[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]];
    let rows: Vec<_> = data.iter().map(|row| row.as_slice()).collect();
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::MIN);
    begin_selection(2);
    for index in 0..2 {
        population(index, &index);
        matrix(&[10, 20, 30], &rows, &[0, 2], &settings, &work, 1024 * 1024);
        assert!(
            input_geometry_pivots_v1(&rows, &[0, 2], &settings, &mut work, 1024 * 1024).is_err()
        );
        assert!(work.exhausted());
        matrix_result(&false, &"capacity", &[10, 30], &work);
    }
    assert_eq!(
        work.visits(),
        0,
        "the refused original charge is never refunded or manufactured"
    );
    end_selection();
}

#[tokio::test]
async fn geometry_capture_keeps_bits_and_later_exhausted_calls_without_scope_leak() {
    let dir = directory();
    assert!(!armed());
    let (_, report) = capture(options(dir.join("capture.jsonl")), async {
        assert!(armed());
        assert!(!tokio::spawn(async { armed() }).await.unwrap());
        original_calls();
        retirement();
    })
    .await
    .unwrap();
    assert!(report.is_complete());
    assert!(!armed());
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&report.path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let matrices: Vec<_> = rows
        .iter()
        .filter(|row| row["kind"] == "original_matrix")
        .collect();
    assert_eq!(matrices.len(), 2);
    assert_eq!(
        matrices[0]["axis_bits"],
        serde_json::json!([
            [1f64.to_bits(), 0, 0],
            [1f64.to_bits(), 1f64.to_bits(), 0],
            [1f64.to_bits(), 0, 1f64.to_bits()]
        ])
    );
    assert_eq!(matrices[0]["axis_bits"], matrices[1]["axis_bits"]);
    assert_eq!(matrices[0]["mandatory_anchors"], serde_json::json!([0, 2]));
    assert_eq!(matrices[1]["exhausted_before"], true);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn geometry_capture_rejects_missing_inventory_or_failed_cleanup() {
    let dir = directory();
    for (name, inventory, cleanup) in [("missing", false, true), ("failed", true, false)] {
        let (_, report) = capture(options(dir.join(name)), async {
            original_calls();
            if inventory {
                inventory_retired(true, tokio::time::Instant::now(), 1, 1, 1, 1);
            }
            series_retired(cleanup);
            shutdown_finished(true);
        })
        .await
        .unwrap();
        assert!(!report.is_complete());
        assert!(
            !dir.join(name).exists(),
            "partial evidence cannot publish to the completed path"
        );
    }
    assert!(!armed());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn geometry_capture_staged_bytes_exact_limit_and_one_byte_short() {
    fn write(path: &std::path::Path, limit: u64) -> CaptureReport {
        let mut file = StagedFile::create(path, limit).unwrap();
        file.preserve_unpublished();
        let mut state = State {
            writer: Some(BufWriter::with_capacity(BUFFER_BYTES, file)),
            started: Instant::now(),
            phase: Phase::Shutdown,
            failure: None,
            expected: Some(1),
            matrices: 1,
            results: 1,
            selection_ended: true,
            retained_bytes: 0,
        };
        state.emit(&[0u64, 1, u64::MAX]);
        state.finish().unwrap()
    }
    let dir = directory();
    let generous = write(&dir.join("generous"), 4096);
    assert!(generous.is_complete());
    let exact = write(&dir.join("exact"), generous.bytes);
    assert!(exact.is_complete());
    assert_eq!(exact.bytes, generous.bytes);
    let short = write(&dir.join("short"), generous.bytes - 1);
    assert!(!short.is_complete());
    assert!(!dir.join("short").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn geometry_capture_missing_call_cannot_claim_complete_enumeration() {
    let dir = directory();
    let (_, report) = capture(options(dir.join("missing_call")), async {
        // An invalid anchor can return before the matrix hook. A later result
        // record and even all cleanup markers must not conceal that omission.
        begin_selection(1);
        let work = StructuredInputGeometryWorkV1::new(NonZeroU64::MIN);
        matrix_result(&false, &"invalid_anchor", &[], &work);
        end_selection();
        retirement();
    })
    .await
    .unwrap();
    assert!(!report.is_complete());
    std::fs::remove_dir_all(dir).unwrap();
}
