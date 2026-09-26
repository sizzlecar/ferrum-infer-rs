use super::*;

fn spec() -> VNextTeacherExecutionSpec {
    VNextTeacherExecutionSpec {
        owners: vec![
            VNextTeacherOwner {
                owner_id: "a".into(),
                prompt_token_ids: vec![1, 2],
                teacher_token_ids: vec![3, 4],
            },
            VNextTeacherOwner {
                owner_id: "b".into(),
                prompt_token_ids: vec![5],
                teacher_token_ids: vec![6, 7],
            },
        ],
        mode: VNextTeacherMode::Batched,
        maximum_sequence_tokens: 4,
        prefill_chunk_tokens: 1,
    }
}

#[test]
fn teacher_history_validates_capacity_vocabulary_and_owner_identity() {
    let valid = spec();
    valid.validate(8, 4).unwrap();
    let mut bad = valid.clone();
    bad.owners[1].owner_id = bad.owners[0].owner_id.clone();
    assert!(bad.validate(8, 4).is_err());
    let mut bad = valid.clone();
    bad.owners[1].teacher_token_ids[1] = 8;
    assert!(bad.validate(8, 4).is_err());
    let mut bad = valid.clone();
    bad.maximum_sequence_tokens = 3;
    assert!(bad.validate(8, 4).is_err());
    let mut bad = valid;
    bad.owners[1].teacher_token_ids.pop();
    assert!(bad.validate(8, 4).is_err());
}

#[test]
fn teacher_admission_covers_each_full_history_without_claiming_the_global_context() {
    let mut spec = spec();
    spec.maximum_sequence_tokens = 4096;
    spec.validate(8, 4096).unwrap();
    assert_eq!(spec.owner_sequence_ceiling(&spec.owners[0]).unwrap(), 4);
    assert_eq!(spec.owner_sequence_ceiling(&spec.owners[1]).unwrap(), 3);
    // Lengthening the continuation consumes real capacity; the request does
    // not silently truncate it to either the earlier budget or context limit.
    spec.owners[0].teacher_token_ids.resize(4095, 3);
    assert!(spec.owner_sequence_ceiling(&spec.owners[0]).is_err());
    assert!(spec.validate(8, 4096).is_err());
    assert_eq!(spec.owners[0].teacher_token_ids.len(), 4095);
    assert_eq!(spec.maximum_sequence_tokens, 4096);
}

#[test]
fn teacher_full_logits_reject_missing_vocabulary_and_nonfinite_values() {
    validate_logits(&[1.0, 2.0], 2).unwrap();
    assert!(validate_logits(&[1.0], 2).is_err());
    assert!(validate_logits(&[1.0, f32::NAN], 2).is_err());
    assert!(validate_logits(&[f32::INFINITY, 2.0], 2).is_err());
    assert_ne!(
        vnext_teacher_token_digest(&[1, 2]),
        vnext_teacher_token_digest(&[2, 1])
    );
}

#[test]
fn teacher_history_binding_rejects_cross_owner_tokens_skips_and_cache_frontier_drift() {
    let owners = spec().owners;
    assert_eq!(
        validate_decision_history(&owners[0], 0, &[1, 2], 2).unwrap(),
        vnext_teacher_token_digest(&[1, 2])
    );
    validate_decision_history(&owners[0], 1, &[1, 2, 3], 3).unwrap();
    assert!(validate_decision_history(&owners[0], 1, &[1, 2, 6], 3).is_err());
    assert!(validate_decision_history(&owners[0], 1, &[1, 2, 3], 2).is_err());
    assert!(validate_decision_history(&owners[0], 1, &[1, 2], 2).is_err());
    assert!(validate_decision_history(&owners[0], 0, &[1, 2, 3], 3).is_err());
    assert!(validate_decision_history(&owners[0], 2, &[1, 2, 3, 4], 4).is_err());
}
