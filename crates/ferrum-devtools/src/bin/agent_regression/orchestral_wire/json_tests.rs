use super::*;

// Producer JSON-codec ordering, deliberately independent of the auditor's
// serializer and serde_json's feature-dependent map representation.
const SORTED_ENVELOPE: &str = r#"{"is_error":false,"result":{"a":null,"text":"fn answer() {\n\t0\n}\n","z":[{"a":2,"z":true}]}}"#;

fn fixture_with_unsorted_public_result() -> Fixture {
    let mut fixture = Fixture::new();
    let session_path = fixture.dir.path().join("session-test.json");
    let mut session: Value = serde_json::from_slice(&fs::read(&session_path).unwrap()).unwrap();
    // Root and nested keys arrive in a different order from the model-visible
    // text. A preserve_order build must not leak that journal order into it.
    session[1]["payload"]["tool"]["content"][0]["result"] = serde_json::from_str(
        r#"{"z":[{"z":true,"a":2}],"text":"fn answer() {\n\t0\n}\n","a":null}"#,
    )
    .unwrap();
    fs::write(&session_path, serde_json::to_vec(&session).unwrap()).unwrap();
    fixture.public = orchestral_evidence::read(fixture.dir.path(), "session");
    assert!(fixture.public.complete(), "{:?}", fixture.public.errors);
    for record in &mut fixture.records[1..] {
        record.messages[3]["content"] = SORTED_ENVELOPE.into();
    }
    fixture
}

#[test]
fn declared_json_reconstructs_sorted_wire_independently_of_public_object_order() {
    let fixture = fixture_with_unsorted_public_result();
    let result = &fixture.public.tool_exchanges[0].calls[0].result;
    let original = result.clone();
    assert_eq!(
        tool_result_content(OrchestralToolResultFormat::Json, result, false).unwrap(),
        SORTED_ENVELOPE
    );
    assert_eq!(*result, original);
    let evidence = fixture.bind();
    assert!(evidence.complete(), "{:?}", evidence.unproven);
    assert_eq!(evidence.receipts.len(), 2);
    assert_eq!(evidence.terminal.unwrap().successful_requests_bound, 3);
}

#[test]
fn declared_json_still_rejects_changed_values_types_and_model_visible_text() {
    for (description, changed) in [
        (
            "source bytes",
            SORTED_ENVELOPE.replace("fn answer", "fn altered"),
        ),
        ("error flag", SORTED_ENVELOPE.replacen("false", "true", 1)),
        (
            "scalar type",
            SORTED_ENVELOPE.replace("\"a\":2", "\"a\":\"2\""),
        ),
        ("missing field", SORTED_ENVELOPE.replace("\"a\":null,", "")),
        (
            "additional field",
            SORTED_ENVELOPE.replace("\"a\":null", "\"a\":null,\"extra\":1"),
        ),
        (
            "nested array",
            SORTED_ENVELOPE.replace("[{\"a\":2,\"z\":true}]", "[]"),
        ),
        ("whitespace", format!("{SORTED_ENVELOPE}\n")),
        (
            "wire key order",
            format!(
                "{{\"result\":{},\"is_error\":false}}",
                SORTED_ENVELOPE
                    .strip_prefix("{\"is_error\":false,\"result\":")
                    .unwrap()
                    .strip_suffix('}')
                    .unwrap()
            ),
        ),
    ] {
        assert_ne!(changed, SORTED_ENVELOPE, "ineffective {description} change");
        let mut fixture = fixture_with_unsorted_public_result();
        fixture.records[1].messages[3]["content"] = changed.into();
        let evidence = fixture.bind();
        assert!(!evidence.complete(), "accepted changed {description}");
        assert!(evidence.receipts.is_empty());
        assert!(evidence.unproven[0].contains("exact expected history"));
    }
}
