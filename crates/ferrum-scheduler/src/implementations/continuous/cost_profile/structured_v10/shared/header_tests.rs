use super::super::tests::fixture;
use super::*;

fn original() -> Header {
    let (bytes, _) = fixture::source();
    let line: serde_json::Value =
        serde_json::from_slice(bytes.split_inclusive(|b| *b == b'\n').next().unwrap()).unwrap();
    serde_json::from_value(line["record"].clone()).unwrap()
}
fn build(children: &[Header], maximum: u64) -> Result<serde_json::Value, CostProfileError> {
    structured_shared_source_header_v4(
        children
            .iter()
            .map(|h| serde_json::to_value(h).unwrap())
            .collect(),
        maximum,
        128,
        512 * 1024 * 1024,
        16_777_216,
    )
}

#[test]
fn shared_source4_header_common_payload_does_not_multiply_by_child_count() {
    let mut old = original();
    // This remains below the original per-record bound in every legacy source.
    // Its old N-child expansion alone crosses that bound.
    old.cohort_manifest_payload = serde_json::json!({"declaration": "a".repeat(600 * 1024)});
    old.cohort_manifest_sha256 = old
        .cohort_plan
        .signature(&old.cohort_manifest_payload)
        .unwrap();
    let children: Vec<_> = (1..=16u8)
        .map(|index| {
            let mut h = old.clone();
            h.capture_identity = [index; 32];
            h.scope.owner.algorithm_domain = [index; 32];
            h.membership_rule.owner = h.scope.owner.clone();
            h
        })
        .collect();
    assert!(
        serde_json::to_vec(&children).unwrap().len()
            > super::super::replay::MAX_SOURCE_RECORD_BYTES
    );
    let compact = build(&children, old.maximum_file_bytes).unwrap();
    assert!(compact["common"]["cohort_manifest_payload"].is_object());
    for child in compact["children"].as_array().unwrap() {
        for common in [
            "fingerprint",
            "producer",
            "opening",
            "cohort_plan",
            "cohort_manifest_payload",
        ] {
            assert!(child.get(common).is_none(), "{common} must be common only");
        }
    }
    let bytes = checked_header_record_bytes(&compact, old.maximum_file_bytes).unwrap();
    assert!(bytes < 1024 * 1024);
    let parsed: HeaderV4 = serde_json::from_value(compact).unwrap();
    assert_eq!(parsed.children.len(), children.len());
    assert_eq!(
        parsed.common.cohort_manifest_payload,
        old.cohort_manifest_payload
    );
}

#[test]
fn shared_source4_header_exact_record_and_common_file_budgets_include_wrapper_and_lf() {
    let maximum = super::super::replay::MAX_SOURCE_RECORD_BYTES;
    let empty = serde_json::json!({"payload": ""});
    let overhead = checked_header_record_bytes(&empty, u64::MAX).unwrap();
    let exact = serde_json::json!({"payload": "a".repeat(maximum - overhead)});
    assert_eq!(
        checked_header_record_bytes(&exact, maximum as u64).unwrap(),
        maximum
    );
    assert!(checked_header_record_bytes(&exact, maximum as u64 - 1).is_err());
    let oversized = serde_json::json!({"payload": "a".repeat(maximum - overhead + 1)});
    assert!(checked_header_record_bytes(&oversized, u64::MAX).is_err());
    // The public live constructor calls the same guard, before offers exist.
    let mut child = original();
    child.cohort_manifest_payload = serde_json::json!({"payload": "a".repeat(maximum)});
    assert!(matches!(
        build(&[child.clone()], child.maximum_file_bytes),
        Err(CostProfileError::Limit(_))
    ));
    child.cohort_manifest_payload = serde_json::Value::Null;
    child.maximum_file_bytes = 1;
    assert!(matches!(
        build(&[child], 1),
        Err(CostProfileError::Limit(_))
    ));
}

#[test]
fn shared_source4_header_preserves_child_open_times_and_rejects_common_disagreement() {
    let mut first = original();
    first.opened_at_ns = 3;
    first.opening.monotonic_ns = 11;
    let mut second = first.clone();
    second.capture_identity = [71; 32];
    second.scope.owner.algorithm_domain = [72; 32];
    second.membership_rule.owner = second.scope.owner.clone();
    second.opened_at_ns = 7;
    let maximum = first.maximum_file_bytes;
    let good = build(&[first.clone(), second.clone()], maximum).unwrap();
    assert_eq!(good["common"]["opening"]["monotonic_ns"], 11);
    assert_eq!(good["children"][0]["opened_at_ns"], 3);
    assert_eq!(good["children"][1]["opened_at_ns"], 7);
    for case in 0..9 {
        let mut changed = second.clone();
        match case {
            0 => changed.opening.monotonic_ns += 1,
            1 => changed.opening.wall_unix_ns += 1,
            2 => changed.fingerprint.execution_config = [99; 32],
            3 => changed.producer["package_version"] = "other".into(),
            4 => changed.cohort_manifest_payload = serde_json::Value::Null,
            5 => changed.initial_fifo_cutoff += 1,
            6 => changed.maximum_offered_waves += 1,
            7 => changed.maximum_file_bytes += 1,
            8 => changed.opened_at_ns = 12,
            _ => unreachable!(),
        }
        assert!(
            build(&[first.clone(), changed], maximum).is_err(),
            "case {case}"
        );
    }
    // The former repeated-header source4 shape is rejected explicitly; source3
    // remains a different supported wire contract with unchanged bytes.
    let mut former = good.clone();
    former.as_object_mut().unwrap().remove("common");
    former["children"] = serde_json::to_value([first, second]).unwrap();
    assert!(serde_json::from_value::<HeaderV4>(former).is_err());
    let mut injected = good;
    injected["children"][0]["opening"] = serde_json::json!({"wall_unix_ns":1,"monotonic_ns":1});
    assert!(serde_json::from_value::<HeaderV4>(injected).is_err());
}
