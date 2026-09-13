use super::*;
use OrchestralToolResultFormat as Format;

fn enrich(result: &mut Value) {
    result["path"] = json!("src/λ.rs");
    result["empty"] = json!("");
    result["alive"] = json!(false);
    result["exit_code"] = json!(0);
    result["optional"] = Value::Null;
    result["wall_time_seconds"] = json!(1.1254992910000001_f64);
}

fn fixture(format: Format) -> Fixture {
    let mut fixture = Fixture::new();
    let session_path = fixture.dir.path().join("session-test.json");
    let mut session: Value = serde_json::from_slice(&fs::read(&session_path).unwrap()).unwrap();
    for event in session.as_array_mut().unwrap() {
        if event["payload"]["type"] == "tool_exchange_committed" {
            enrich(&mut event["payload"]["tool"]["content"][0]["result"]);
        }
    }
    fs::write(&session_path, serde_json::to_vec(&session).unwrap()).unwrap();
    fixture.public = orchestral_evidence::read(fixture.dir.path(), "session");
    assert!(fixture.public.complete(), "{:?}", fixture.public.errors);
    for record in &mut fixture.records {
        for message in &mut record.messages {
            if message["role"] == "tool" {
                let mut envelope: Value =
                    serde_json::from_str(message["content"].as_str().unwrap()).unwrap();
                enrich(&mut envelope["result"]);
                message["content"] =
                    tool_result_content(format, &envelope["result"], false).unwrap();
            }
        }
    }
    fixture
}

#[test]
fn exact_text_parts_revision_binds_full_history_and_wrong_revision_fails() {
    for (format, wrong) in [
        (Format::TextParts, Format::TextPartsV2),
        (Format::TextPartsV2, Format::TextParts),
    ] {
        let fixture = fixture(format);
        let bound = fixture.bind_as(format);
        assert!(bound.complete(), "{:?}", bound.unproven);
        assert_eq!(bound.receipts.len(), 2);
        assert_eq!(bound.terminal.unwrap().successful_requests_bound, 3);
        assert!(!fixture.bind_as(wrong).complete());
        assert!(!fixture.bind_as(Format::Json).complete());
    }
}

#[test]
fn v2_metadata_values_source_bytes_order_and_lifecycle_stay_exact() {
    for (field, changed) in [
        ("alive", json!(true)),
        ("exit_code", json!("0")),
        ("optional", json!("null")),
        ("empty", json!(" ")),
        ("path", json!("src/other.rs")),
        (
            "wall_time_seconds",
            json!(f64::from_bits(1.1254992910000001_f64.to_bits() - 1)),
        ),
    ] {
        let mut fixture = fixture(Format::TextPartsV2);
        let part = &mut fixture.records[1].messages[3]["content"][0]["text"];
        let text = part.as_str().unwrap();
        let mut metadata: Value =
            serde_json::from_str(text.strip_prefix("Tool result metadata:\n").unwrap()).unwrap();
        metadata["result"][field] = changed;
        *part = format!("Tool result metadata:\n{metadata}\n").into();
        assert!(
            !fixture.bind_as(Format::TextPartsV2).complete(),
            "accepted changed {field}"
        );
    }

    let mut changed = fixture(Format::TextPartsV2);
    let part = &mut changed.records[1].messages[3]["content"][1]["text"];
    *part = format!("{}\n", part.as_str().unwrap()).into();
    assert!(!changed.bind_as(Format::TextPartsV2).complete());

    let mut reordered = fixture(Format::TextPartsV2);
    reordered.records[1].messages[3]["content"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(!reordered.bind_as(Format::TextPartsV2).complete());
    let mut missing = fixture(Format::TextPartsV2);
    missing.records[1].messages.pop();
    assert!(!missing.bind_as(Format::TextPartsV2).complete());
    let mut cancelled = fixture(Format::TextPartsV2);
    cancelled.public.delivered = false;
    assert!(!cancelled.bind_as(Format::TextPartsV2).complete());
    let mut compacted = fixture(Format::TextPartsV2);
    compacted.public.compaction_events = 1;
    assert!(!compacted.bind_as(Format::TextPartsV2).complete());
}

#[test]
fn saved_text_parts_v2_requires_explicit_revision_and_preserves_original_evidence() {
    let fixture = fixture(Format::TextPartsV2);
    // A historical manifest declared the style before the reader knew v2.
    let report = fixture.save_report(Format::TextParts);
    let result_path = report.path().join("coding/result.json");
    let manifest_path = report.path().join("manifest.json");
    let original_result = fs::read(&result_path).unwrap();
    let original_manifest = fs::read(&manifest_path).unwrap();
    assert_eq!(
        audit_saved(
            report.path(),
            "coding",
            &report.path().join("legacy-audit.json"),
            None
        )
        .unwrap(),
        1
    );

    let output = report.path().join("v2-audit.json");
    assert_eq!(
        audit_saved(report.path(), "coding", &output, Some(Format::TextPartsV2)).unwrap(),
        0
    );
    let evidence: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(evidence["manifest_tool_result_format"], "text_parts");
    assert_eq!(evidence["selected_tool_result_format"], "text_parts_v2");
    assert_eq!(evidence["tool_result_format_override"], "text_parts_v2");
    assert_eq!(evidence["source_manifest_sha256"], sha(&original_manifest));
    assert_eq!(evidence["source_task_result_sha256"], sha(&original_result));
    assert_eq!(evidence["wire"]["terminal"]["successful_requests_bound"], 3);
    for record in &fixture.records {
        let index = record.request_index.to_string();
        let body = fs::read(
            report
                .path()
                .join(format!("requests/coding-{index}.request.json")),
        )
        .unwrap();
        assert_eq!(evidence["request_body_sha256"][index.as_str()], sha(&body));
    }
    assert!(audit_saved(report.path(), "coding", &output, Some(Format::TextPartsV2)).is_err());
    assert!(audit_saved(
        report.path(),
        "coding",
        &result_path,
        Some(Format::TextPartsV2)
    )
    .is_err());
    assert_eq!(fs::read(&result_path).unwrap(), original_result);
    assert_eq!(fs::read(&manifest_path).unwrap(), original_manifest);

    for format in [Format::TextParts, Format::TextPartsV2] {
        let report = self::fixture(format).save_report(format);
        assert_eq!(
            audit_saved(
                report.path(),
                "coding",
                &report.path().join("declared.json"),
                None
            )
            .unwrap(),
            0
        );
    }
    for (declared, selected) in [
        (Format::Json, Format::TextPartsV2),
        (Format::Yaml, Format::TextPartsV2),
        (Format::TextPartsV2, Format::Json),
    ] {
        let report = fixture.save_report(declared);
        let output = report.path().join("wrong-family.json");
        assert!(audit_saved(report.path(), "coding", &output, Some(selected)).is_err());
        assert!(!output.exists());
    }
}
