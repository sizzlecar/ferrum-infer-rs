use super::*;
use serde_json::json;

fn id(value: &str) -> ProgramValueId {
    ProgramValueId::new(value).unwrap()
}

#[test]
fn empty_output_only_role_preserves_legacy_wire() {
    let legacy = json!({
        "contract_version": {"major": 1, "minor": 0},
        "token_input": "input.tokens",
        "conditioning_inputs": ["input.condition"]
    });
    let inputs: ProgramCheckpointInputs = serde_json::from_value(legacy.clone()).unwrap();
    assert!(inputs.output_only_inputs().is_empty());
    assert_eq!(serde_json::to_value(&inputs).unwrap(), legacy);
    let before = serde_json::to_vec(&inputs).unwrap();
    let empty = inputs.with_output_only_inputs(BTreeSet::new()).unwrap();
    assert_eq!(serde_json::to_vec(&empty).unwrap(), before);
}

#[test]
fn explicit_roles_cover_every_input_without_weakening_conditioning() {
    let inputs =
        ProgramCheckpointInputs::new(id("input.tokens"), BTreeSet::from([id("input.condition")]))
            .unwrap()
            .with_output_only_inputs(BTreeSet::from([id("input.selection")]))
            .unwrap();
    assert!(inputs.covers(&[
        id("input.tokens"),
        id("input.selection"),
        id("input.condition")
    ]));
    assert!(!inputs.covers(&[id("input.tokens"), id("input.selection")]));
    assert!(!inputs.covers(&[
        id("input.tokens"),
        id("input.selection"),
        id("input.condition"),
        id("input.other")
    ]));
    assert_eq!(
        inputs.conditioning_inputs(),
        &BTreeSet::from([id("input.condition")])
    );
    let restored: ProgramCheckpointInputs =
        serde_json::from_slice(&serde_json::to_vec(&inputs).unwrap()).unwrap();
    assert_eq!(restored, inputs);
    for overlap in ["input.tokens", "input.condition"] {
        assert!(inputs
            .clone()
            .with_output_only_inputs(BTreeSet::from([id(overlap)]))
            .is_err());
    }
}

#[test]
fn output_only_wire_rejects_noncanonical_overlapping_or_unknown_roles() {
    let inputs =
        ProgramCheckpointInputs::new(id("input.tokens"), BTreeSet::from([id("input.condition")]))
            .unwrap();
    for invalid in [
        json!(null),
        json!(["input.z", "input.a"]),
        json!(["input.a", "input.a"]),
        json!(["input.tokens"]),
        json!(["input.condition"]),
    ] {
        let mut wire = serde_json::to_value(&inputs).unwrap();
        wire["output_only_inputs"] = invalid;
        assert!(serde_json::from_value::<ProgramCheckpointInputs>(wire).is_err());
    }
    let mut wire = serde_json::to_value(&inputs).unwrap();
    wire["unverified_inputs"] = json!(["input.a"]);
    assert!(serde_json::from_value::<ProgramCheckpointInputs>(wire).is_err());
}
