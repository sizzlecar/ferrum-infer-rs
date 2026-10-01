use super::*;

fn health() -> serde_json::Value {
    serde_json::json!({"admission": {
        "runtime_snapshot_available":true, "queue_depth":2, "active_prefill":1, "active_decode":3,
        "queue_observation": {
            "schema_version":1, "engine_instance":"original-instance", "observed_at_ns":999,
            "waiting_requests":2, "active_prefill_sequences":1, "active_decode_sequences":3,
            "preempted_requests":1, "oldest_waiting_ingress_age_ns":1234,
            "oldest_unfinished_ingress_age_ns":2345
        }
    }})
}

#[test]
fn capacity_server_queue_decode_preserves_original_values_and_missing_is_not_zero() {
    let raw = health();
    let observed = decode(&serde_json::to_vec(&raw).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(observed).unwrap(),
        raw["admission"]["queue_observation"]
    );
    assert_eq!(
        decode(br#"{"status":"ok","slots_idle":32}"#),
        Err(ServerQueueFailure::Unavailable)
    );
    let mut missing_age = raw.clone();
    missing_age["admission"]["queue_observation"]["oldest_waiting_ingress_age_ns"] =
        serde_json::Value::Null;
    assert_eq!(
        decode(&serde_json::to_vec(&missing_age).unwrap()),
        Err(ServerQueueFailure::Malformed)
    );
    let mut mismatched = raw;
    mismatched["admission"]["queue_depth"] = 1.into();
    assert_eq!(
        decode(&serde_json::to_vec(&mismatched).unwrap()),
        Err(ServerQueueFailure::Malformed)
    );
}

#[test]
fn capacity_server_queue_pre_window_clock_is_signed_and_errors_are_bounded() {
    let origin = Instant::now();
    let before = origin - Duration::from_millis(2);
    let raw = TimedAttempt {
        started: before,
        completed: before + Duration::from_millis(1),
        observation: Err(ServerQueueFailure::Timeout),
    }
    .relative(origin);
    assert!(raw.request_started_seconds < raw.response_completed_seconds);
    assert!(raw.response_completed_seconds < 0.0);
    let raw = serde_json::json!({"admission":{"queue_observation_error":"繁".repeat(300)}});
    let Err(ServerQueueFailure::Runtime(message)) = decode(&serde_json::to_vec(&raw).unwrap())
    else {
        panic!("original unavailable reason expected");
    };
    assert!(message.len() <= 512);
    assert!(!message.is_empty());
}
