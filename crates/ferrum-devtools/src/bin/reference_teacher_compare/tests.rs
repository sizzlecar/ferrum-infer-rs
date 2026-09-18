use super::*;
use ferrum_interfaces::vnext::TokenSpanWork;

fn response(probabilities: &[f64]) -> Value {
    json!({"truncated":false,"tokens_evaluated":3,"tokens_predicted":1,
        "completion_probabilities":[{"top_logprobs":probabilities.iter().enumerate().rev()
            .map(|(id,p)|json!({"id":id,"logprob":p.ln()})).collect::<Vec<_>>()}]})
}

#[test]
fn full_distribution_comparison_is_order_and_logit_shift_invariant() {
    let p = [0.25_f64, 0.75];
    let reference = distribution::reference_log_probabilities(&response(&p), 2, 3).unwrap();
    assert_eq!(reference, p.map(f64::ln));
    let logits = [p[0].ln() as f32 + 5.0, p[1].ln() as f32 + 5.0];
    let metrics = distribution::compare(&reference, &logits, 0).unwrap();
    assert!(metrics["kl_reference_to_candidate_nats"].as_f64().unwrap() < 1e-12);
    assert!((metrics["reference_nll_nats"].as_f64().unwrap() - 4.0_f64.ln()).abs() < 1e-12);
    let changed = distribution::compare(&reference, &[0.0, 0.0], 1).unwrap();
    let expected_kl = 0.25 * 0.5_f64.ln() + 0.75 * 1.5_f64.ln();
    assert!(
        (changed["kl_reference_to_candidate_nats"].as_f64().unwrap() - expected_kl).abs() < 1e-12
    );
    assert!((changed["delta_nll_nats"].as_f64().unwrap() - 1.5_f64.ln()).abs() < 1e-12);
}

#[test]
fn reference_rejects_missing_duplicate_renormalized_or_truncated_evidence() {
    let valid = response(&[0.25, 0.75]);
    for (pointer, value) in [
        ("/tokens_evaluated", json!(2)),
        ("/tokens_predicted", json!(0)),
        ("/truncated", json!(true)),
        ("/completion_probabilities/0/top_logprobs/0/id", json!(0)),
        (
            "/completion_probabilities/0/top_logprobs/0/logprob",
            json!(null),
        ),
        (
            "/completion_probabilities/0/top_logprobs/0/logprob",
            json!(-10.0),
        ),
    ] {
        let mut broken = valid.clone();
        *broken.pointer_mut(pointer).unwrap() = value;
        assert!(
            distribution::reference_log_probabilities(&broken, 2, 3).is_err(),
            "{pointer}"
        );
    }
    assert!(distribution::reference_log_probabilities(&valid, 3, 3).is_err());
    assert!(distribution::compare(&[0.0], &[f32::NAN], 0).is_err());
    assert!(distribution::compare(&[0.0], &[0.0], 1).is_err());
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let prompt = [0, 1, 2];
    let tokens = [1, 0];
    let plan = json!({"schema_version":4,"plan_id":"plan","plan_hash":"hash","model_id":"model",
        "family_fingerprint":"family","program_fingerprint":"program","run_id":"run",
        "maximum_prefill_waves":1,"maximum_decode_waves":1,"capture_product_output":true,
        "teacher_forcing":{"mode":"canonical-history","encoding":"u32-le","prompt_file":"teacher-prompt.json",
            "token_count":tokens.len(),"token_ids_sha256":capture::token_digest(&tokens)}});
    write_new(
        &root.path().join("plan.json"),
        &serde_json::to_vec(&plan).unwrap(),
    )
    .unwrap();
    write_new(
        &root.path().join("teacher-prompt.json"),
        &serde_json::to_vec(&json!({"schema_version":1,
        "encoding":"u32-le","request_id":"request","token_count":prompt.len(),"token_ids":prompt,
        "token_ids_sha256":capture::token_digest(&prompt)}))
        .unwrap(),
    )
    .unwrap();
    let mut history = prompt.to_vec();
    for (index, token) in tokens.into_iter().enumerate() {
        let mut wave = plan.clone();
        wave["capture_index"] = json!(0);
        wave["wave_kind"] = json!(if index == 0 { "prefill" } else { "decode" });
        wave["participant_count"] = json!(1);
        wave["teacher_forced_decision"] = json!({"token_index":index,"token_id":token});
        let logits = [-1.0_f32, 0.0, 1.0];
        let raw: Vec<_> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let name = format!("logits-{index}.raw");
        write_new(&root.path().join(&name), &raw).unwrap();
        let span = TokenSpanWork::from_token_ids(
            &history,
            if index == 0 {
                0..history.len()
            } else {
                history.len() - 1..history.len()
            },
        )
        .unwrap();
        wave["product_outputs"] = json!([{"output_mode":"full-logits","participant_index":0,"request_id":"request",
            "token_span":span,"output_layout":{"element_type":"f32","element_count":logits.len()},
            "raw_file":name,"raw_bytes":raw.len(),"raw_sha256":sha256(&raw)}]);
        let name = if index == 0 {
            "wave-0000.json"
        } else {
            "decode-wave-0000.json"
        };
        write_new(&root.path().join(name), &serde_json::to_vec(&wave).unwrap()).unwrap();
        history.push(token);
    }
    root
}

#[test]
fn canonical_history_capture_preserves_every_decision() {
    let root = fixture();
    let actual = capture::Capture::read(root.path()).unwrap();
    assert_eq!(actual.prompt, [0, 1, 2]);
    assert_eq!(
        actual
            .waves
            .iter()
            .map(|wave| wave.token)
            .collect::<Vec<_>>(),
        [1, 0]
    );
    assert_eq!(actual.teacher_sha256, capture::token_digest(&[1, 0]));
    assert_eq!(
        reference_request(&actual.prompt, 3, false)["prompt"],
        json!([0, 1, 2])
    );
}

#[test]
fn capture_rejects_history_owner_artifact_and_inventory_drift() {
    for (pointer, value) in [
        ("/teacher_forced_decision/token_id", json!(2)),
        ("/product_outputs/0/request_id", json!("other")),
        ("/product_outputs/0/token_span/fingerprint", json!("wrong")),
        ("/product_outputs/0/raw_file", json!("../elsewhere")),
        ("/product_outputs/0/raw_sha256", json!("wrong")),
        ("/product_outputs/0/output_layout/element_count", json!(2)),
        ("/family_fingerprint", json!("changed")),
    ] {
        let root = fixture();
        let path = root.path().join("decode-wave-0000.json");
        let mut wave: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        *wave.pointer_mut(pointer).unwrap() = value;
        fs::write(path, serde_json::to_vec(&wave).unwrap()).unwrap();
        assert!(capture::Capture::read(root.path()).is_err(), "{pointer}");
    }
    let root = fixture();
    fs::remove_file(root.path().join("decode-wave-0000.json")).unwrap();
    assert!(capture::Capture::read(root.path()).is_err());
    let root = fixture();
    fs::write(root.path().join("wave-0001.json"), b"{}").unwrap();
    assert!(capture::Capture::read(root.path()).is_err());
}
