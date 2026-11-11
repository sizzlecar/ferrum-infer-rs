use super::*;

pub(super) fn observation(capture: bool) -> Value {
    let text = "A small red cat sits quietly.";
    json!({"kind":"prompt_expansion_observed","job":19,"request_id":null,
        "policy":if capture {json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":4096})} else {json!({"mode":"summary"})},
        "prompt_tokens":8,"protocol_eos_token_id":2,"initialization_wall_us":12,"retirement_wall_us":3,
        "executor_retired":true,"outcome":"expanded",
        "token_history":{"output_tokens_including_eos":2,"last_token_id":2,"protocol_eos_observed":true,"finish":"EOS"},
        "executor_metrics":{"prefill_operations":4,"decode_operations":1},
        "effective_prompt_utf8_bytes":text.len(),"effective_prompt_sha256":crate::images::sha256(text.as_bytes()),
        "capture":if capture {json!({"status":"captured","text":text})} else {json!({"status":"disabled"})}})
}

#[test]
fn prompt_expansion_diagnostics_read_without_creating_step_or_timing_qualification() {
    for captured in [false, true] {
        let report = analyze(&artifact(&[observation(captured)], |_| {})).unwrap();
        assert_eq!(report.prompt_expansion_observations.len(), 1);
        assert_eq!(report.status, ImageStepEvidenceStatus::NotEvaluated);
        assert_eq!(report.facts.committed_waves, 0);
        assert_eq!(report.timing.status, ImageStepEvidenceStatus::NotEvaluated);
        let mut original = events();
        original.push(observation(captured));
        let report = analyze(&artifact(&original, |_| {})).unwrap();
        assert_eq!(report.status, ImageStepEvidenceStatus::Observed);
        assert_eq!(report.facts.committed_waves, 3);
        assert_eq!(report.prompt_expansion_observations.len(), 1);
    }
}

#[test]
fn prompt_expansion_diagnostics_reject_malformed_capture_eos_duplicates_and_missing_fields() {
    for mutate in [
        |v: &mut Value| {
            v.as_object_mut().unwrap().remove("prompt_tokens");
        },
        |v: &mut Value| {
            v["token_history"]["protocol_eos_observed"] = json!(false);
        },
        |v: &mut Value| {
            v["executor_retired"] = json!(false);
        },
        |v: &mut Value| {
            v["capture"]["text"] = json!("wrong text");
        },
        |v: &mut Value| {
            v["policy"]["maximum_utf8_bytes"] = json!(0);
        },
        |v: &mut Value| {
            v["capture"]["unexpected"] = json!(true);
        },
    ] {
        let mut value = observation(true);
        mutate(&mut value);
        assert!(analyze(&artifact(&[value], |_| {})).is_err());
    }
    assert!(analyze(&artifact(&[observation(false), observation(false)], |_| {})).is_err());
    for (field, key) in [("policy", "maximum_utf8_bytes"), ("capture", "text")] {
        let mut value = observation(false);
        value[field][key] = json!(1);
        assert!(
            analyze(&artifact(&[value], |_| {})).is_err(),
            "unit mode ignored extra field {field}.{key}"
        );
    }
    let mut refused = observation(true);
    let actual = refused["effective_prompt_utf8_bytes"].clone();
    refused["policy"]["maximum_utf8_bytes"] = json!(1);
    refused["capture"] =
        json!({"status":"capacity_refused","actual_utf8_bytes":actual,"maximum_utf8_bytes":1});
    let report = analyze(&artifact(&[refused], |_| {})).unwrap();
    assert!(matches!(
        report.prompt_expansion_observations[0].capture,
        ImageTraceExpansionPromptCapture::CapacityRefused { .. }
    ));
}

#[test]
fn vision_binding_receipts_are_optional_typed_and_do_not_require_a_live_producer() {
    let legacy = analyze(&artifact(&[observation(false)], |_| {})).unwrap();
    assert!(legacy.prompt_expansion_observations[0]
        .vision_binding
        .is_none());
    let producer = uuid::Uuid::new_v4();
    let mut miss = observation(false);
    miss["request_id"] = json!(producer);
    miss["vision_binding"] = json!({
        "reused":false,"producer_job":19,"producer_request_id":producer,"output_bytes":8192,
    });
    let mut hit = observation(false);
    hit["job"] = json!(20);
    hit["request_id"] = json!(uuid::Uuid::new_v4());
    hit["vision_binding"] = json!({
        "reused":true,"producer_job":19,"producer_request_id":producer,"output_bytes":8192,
    });
    let report = analyze(&artifact(&[miss.clone(), hit.clone()], |_| {})).unwrap();
    let receipt = report.prompt_expansion_observations[1]
        .vision_binding
        .as_ref()
        .unwrap();
    assert!(receipt.reused);
    assert_eq!(receipt.producer_job, 19);
    assert_eq!(receipt.producer_request_id, Some(producer));
    assert_eq!(receipt.output_bytes, 8192);
    // Visual completion may precede cancellation. The producer can therefore
    // have no causal observation or request-linked/completed event at all.
    let isolated = analyze(&artifact(&[hit], |_| {})).unwrap();
    assert_eq!(isolated.facts.committed_waves, 0);
    assert_eq!(isolated.status, ImageStepEvidenceStatus::NotEvaluated);
    for mutate in [
        |v: &mut Value| v["vision_binding"]["output_bytes"] = json!(0),
        |v: &mut Value| v["vision_binding"]["producer_job"] = json!(20),
        |v: &mut Value| v["vision_binding"]["producer_request_id"] = json!(null),
        |v: &mut Value| v["vision_binding"]["reused"] = json!("true"),
        |v: &mut Value| v["vision_binding"]["extra"] = json!(1),
        |v: &mut Value| {
            v["vision_binding"]
                .as_object_mut()
                .unwrap()
                .remove("output_bytes");
        },
    ] {
        let mut invalid = miss.clone();
        mutate(&mut invalid);
        assert!(analyze(&artifact(&[invalid], |_| {})).is_err());
    }
}
