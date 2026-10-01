use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[test]
fn collecting_call_ids_preserve_legacy_order_and_duplicate_rejection() {
    let mut calls = Calls::Collecting(Vec::new());
    let mut legacy = BTreeSet::new();
    for call in [1, 9, 20, 3, 7, 9, 2, 20, 21, 4] {
        assert_eq!(calls.insert(call).unwrap(), legacy.insert(call));
    }
    let mut observed = Vec::new();
    calls.for_each(|call| observed.push(call));
    assert_eq!(observed, legacy.iter().copied().collect::<Vec<_>>());
    let mut state = PhaseState::new();
    state.calls = calls;
    let mut actual = Sha256::new();
    state.bind_parameters(&mut actual);
    let mut old_binding = Sha256::new();
    for value in [0u64, 0, 0, 0, u64::MAX, legacy.len() as u64] {
        old_binding.update(value.to_le_bytes());
    }
    for call in legacy {
        old_binding.update(call.to_le_bytes());
    }
    assert_eq!(actual.finalize(), old_binding.finalize());
}

#[test]
fn collecting_and_frozen_call_payload_keep_the_same_parameter_binding() {
    let mut state = PhaseState::new();
    for call in [11, 2, 5, 1, 30] {
        assert!(state.calls.insert(call).unwrap());
    }
    let before = state.retained_heap_bytes().unwrap();
    assert!(before >= 5 * std::mem::size_of::<u64>());
    let mut collecting = Sha256::new();
    state.bind_parameters(&mut collecting);
    state.freeze_calls().unwrap();
    assert_eq!(
        state.retained_heap_bytes(),
        Some(5 * std::mem::size_of::<u64>())
    );
    let mut frozen = Sha256::new();
    state.bind_parameters(&mut frozen);
    assert_eq!(collecting.finalize(), frozen.finalize());
    assert!(matches!(
        state.calls.insert(40),
        Err(StructuredUnknown::WrongProtocol)
    ));
}
