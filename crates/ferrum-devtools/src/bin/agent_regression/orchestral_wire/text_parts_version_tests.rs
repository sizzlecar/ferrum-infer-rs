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
    fixture_with_error(format, false)
}

fn fixture_with_error(format: Format, is_error: bool) -> Fixture {
    let mut fixture = Fixture::new();
    let session_path = fixture.dir.path().join("session-test.json");
    let mut session: Value = serde_json::from_slice(&fs::read(&session_path).unwrap()).unwrap();
    for event in session.as_array_mut().unwrap() {
        if event["payload"]["type"] == "tool_exchange_committed" {
            enrich(&mut event["payload"]["tool"]["content"][0]["result"]);
            event["payload"]["tool"]["content"][0]["is_error"] = json!(is_error);
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
                    tool_result_content(format, &envelope["result"], is_error).unwrap();
            }
        }
    }
    fixture
}

#[test]
fn exact_text_parts_revision_binds_full_history_and_wrong_revision_fails() {
    let formats = [Format::TextParts, Format::TextPartsV2, Format::TextPartsV3];
    for format in formats {
        let fixture = fixture(format);
        let bound = fixture.bind_as(format);
        assert!(bound.complete(), "{:?}", bound.unproven);
        assert_eq!(bound.receipts.len(), 2);
        assert_eq!(bound.terminal.unwrap().successful_requests_bound, 3);
        for wrong in formats.into_iter().filter(|wrong| *wrong != format) {
            assert!(!fixture.bind_as(wrong).complete());
        }
        assert!(!fixture.bind_as(Format::Json).complete());
    }
}

#[test]
fn multiline_revisions_metadata_values_source_bytes_order_and_lifecycle_stay_exact() {
    for format in [Format::TextPartsV2, Format::TextPartsV3] {
        let heading = if format == Format::TextPartsV2 {
            "Tool result metadata:\n"
        } else {
            ""
        };
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
        let mut fixture = fixture(format);
        let part = &mut fixture.records[1].messages[3]["content"][0]["text"];
        let text = part.as_str().unwrap();
        let mut metadata: Value =
            serde_json::from_str(text.strip_prefix(heading).unwrap()).unwrap();
        metadata["result"][field] = changed;
        *part = format!("{heading}{metadata}\n").into();
        assert!(
            !fixture.bind_as(format).complete(),
            "accepted changed {field}"
        );
    }

    for (original, replacement) in [
        ("fn answer()", "fn altered()"),
        ("\n\t0\n", "\r\n\t0\r\n"),
        ("final newline: yes", "final newline: no"),
        ("\n```\n", "\n```\n\n"),
    ] {
        let mut changed = fixture(format);
        let part = &mut changed.records[1].messages[3]["content"][1]["text"];
        assert!(part.as_str().unwrap().contains(original));
        *part = part.as_str().unwrap().replacen(original, replacement, 1).into();
        assert!(!changed.bind_as(format).complete());
    }

    let mut reordered = fixture(format);
    reordered.records[1].messages[3]["content"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(!reordered.bind_as(format).complete());
    let mut missing = fixture(format);
    missing.records[1].messages.pop();
    assert!(!missing.bind_as(format).complete());
    let mut cancelled = fixture(format);
    cancelled.public.delivered = false;
    assert!(!cancelled.bind_as(format).complete());
    let mut compacted = fixture(format);
    compacted.public.compaction_events = 1;
    assert!(!compacted.bind_as(format).complete());
    }
}

#[test]
fn v3_error_flag_and_request_scoped_native_call_ownership_stay_exact() {
    for is_error in [false, true] {
        let mut fixture = fixture_with_error(Format::TextPartsV3, is_error);
        let evidence = fixture.bind_as(Format::TextPartsV3);
        assert!(evidence.complete(), "{:?}", evidence.unproven);
        let part = &mut fixture.records[1].messages[3]["content"][0]["text"];
        let mut metadata: Value = serde_json::from_str(part.as_str().unwrap()).unwrap();
        if is_error {
            metadata.as_object_mut().unwrap().remove("is_error");
        } else {
            // Even an equivalent false flag is not the declared v3 bytes.
            metadata["is_error"] = json!(false);
        }
        *part = format!("{metadata}\n").into();
        assert!(!fixture.bind_as(Format::TextPartsV3).complete());
    }

    let mut misowned = fixture(Format::TextPartsV3);
    misowned.public.tool_exchanges[1].request_id =
        misowned.public.tool_exchanges[0].request_id.clone();
    assert!(!misowned.bind_as(Format::TextPartsV3).complete());

    // Both responses use native call_0; that does not authorize swapping their
    // different tool results inside the following request's retained history.
    let mut swapped = fixture(Format::TextPartsV3);
    swapped.records[2].messages.swap(3, 5);
    assert!(!swapped.bind_as(Format::TextPartsV3).complete());

    let mut relabeled = fixture(Format::TextPartsV3);
    let part = &mut relabeled.records[1].messages[3]["content"][1]["text"];
    *part = part.as_str().unwrap().replacen("\"text\"", "\"other\"", 1).into();
    assert!(!relabeled.bind_as(Format::TextPartsV3).complete());

    let mut flattened = fixture(Format::TextPartsV3);
    let content = &mut flattened.records[1].messages[3]["content"];
    *content = content.as_array().unwrap().iter()
        .map(|part| part["text"].as_str().unwrap())
        .collect::<Vec<_>>().join("").into();
    assert!(!flattened.bind_as(Format::TextPartsV3).complete());
}

#[test]
fn saved_text_parts_revisions_require_explicit_selection_and_preserve_original_evidence() {
    for (declared, selected) in [
        (Format::TextParts, Format::TextPartsV2),
        (Format::TextParts, Format::TextPartsV3),
        (Format::TextPartsV2, Format::TextPartsV3),
    ] {
    let fixture = fixture(selected);
    // A historical manifest declared the style before the reader knew its revision.
    let report = fixture.save_report(declared);
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

    let output = report.path().join("revision-audit.json");
    assert_eq!(
        audit_saved(report.path(), "coding", &output, Some(selected)).unwrap(),
        0
    );
    let evidence: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(evidence["manifest_tool_result_format"], json!(declared));
    assert_eq!(evidence["selected_tool_result_format"], json!(selected));
    assert_eq!(evidence["tool_result_format_override"], json!(selected));
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
    assert!(audit_saved(report.path(), "coding", &output, Some(selected)).is_err());
    assert!(audit_saved(
        report.path(),
        "coding",
        &result_path,
        Some(selected)
    )
    .is_err());
    assert_eq!(fs::read(&result_path).unwrap(), original_result);
    assert_eq!(fs::read(&manifest_path).unwrap(), original_manifest);
    }

    for format in [Format::TextParts, Format::TextPartsV2, Format::TextPartsV3] {
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
        (Format::Json, Format::TextPartsV3),
        (Format::Yaml, Format::TextPartsV3),
        (Format::TextPartsV3, Format::Yaml),
    ] {
        let report = fixture(selected).save_report(declared);
        let output = report.path().join("wrong-family.json");
        assert!(audit_saved(report.path(), "coding", &output, Some(selected)).is_err());
        assert!(!output.exists());
    }
}
