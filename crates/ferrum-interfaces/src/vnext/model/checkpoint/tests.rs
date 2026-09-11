use super::*;
use crate::vnext::{
    ElementType, ModelFamilyId, ModelProgram, NodeId, OperationId, ProgramBlock, ProgramNode,
    ProgramNodeWorkSpec, ProgramTensorSpec, ProgramValueId, ResolvedTensorLayout,
    StateCapacityDemand, StateId, StateInitialization, StateSpec,
};
use serde_json::json;
use std::collections::BTreeMap;

fn state() -> StateSpec {
    StateSpec {
        id: StateId::new("state.fixture").unwrap(),
        value_id: ProgramValueId::new("value.state").unwrap(),
        tensor: ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::FixedPerScope,
        initialization: StateInitialization::Zero,
        checkpoint: StateCheckpointCapability::Unsupported,
    }
}

fn supported(contents: StateCheckpointContents) -> StateCheckpointCapability {
    StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
        contents,
        CheckpointInputDependency::ExactTokenPrefix,
    ))
}

fn program(family: &str, state: StateSpec) -> Result<ModelProgram, VNextError> {
    let input = ProgramValueId::new("value.input").unwrap();
    let output = ProgramValueId::new("value.output").unwrap();
    ModelProgram::new(
        ModelFamilyId::new(family).unwrap(),
        vec![input.clone()],
        vec![ProgramBlock {
            id: "block.main".to_owned(),
            nodes: vec![ProgramNode {
                id: NodeId::new("node.main").unwrap(),
                operation_id: OperationId::new("operation.fixture").unwrap(),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![input, state.value_id.clone()],
                outputs: vec![output.clone()],
                attributes: BTreeMap::new(),
            }],
        }],
        vec![state],
        vec![],
        vec![output],
    )
}

#[test]
fn undeclared_state_preserves_old_wire_and_never_infers_capacity_semantics() {
    for capacity_demand in [
        StateCapacityDemand::FixedPerScope,
        StateCapacityDemand::TokenScaled {
            bytes_per_token: 16,
            maximum_tokens: 1024,
        },
    ] {
        let mut state = state();
        state.capacity_demand = capacity_demand;
        let wire = serde_json::to_value(&state).unwrap();
        assert!(wire.get("checkpoint").is_none());
        let restored: StateSpec = serde_json::from_value(wire.clone()).unwrap();
        assert!(restored.checkpoint.is_unsupported());
        assert_eq!(serde_json::to_value(&restored).unwrap(), wire);
        // The same capacity can also hold a boundary summary; capacity is not
        // the source of the valid-content declaration.
        state.checkpoint = supported(StateCheckpointContents::BoundaryValue);
        let restored: StateSpec =
            serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
        assert_eq!(restored, state);
    }
}

#[test]
fn explicit_state_contents_and_input_dependency_change_program_identity() {
    for family in ["family.first", "family.second"] {
        let plain = state();
        let original = program(family, plain.clone()).unwrap();
        let original_wire = serde_json::to_vec(&original).unwrap();
        let round_trip: ModelProgram = serde_json::from_slice(&original_wire).unwrap();
        assert_eq!(
            original.fingerprint().unwrap(),
            round_trip.fingerprint().unwrap()
        );

        let mut prefix = plain.clone();
        prefix.checkpoint = supported(StateCheckpointContents::PrefixPositions);
        let prefix = program(family, prefix).unwrap();
        let mut boundary = plain.clone();
        boundary.checkpoint = supported(StateCheckpointContents::BoundaryValue);
        let boundary = program(family, boundary).unwrap();
        let mut conditioned = plain;
        conditioned.checkpoint =
            StateCheckpointCapability::CompletedBoundary(StateCheckpointContract::new(
                StateCheckpointContents::BoundaryValue,
                CheckpointInputDependency::EntireTokenInput,
            ));
        let conditioned = program(family, conditioned).unwrap();
        assert_ne!(
            original.fingerprint().unwrap(),
            prefix.fingerprint().unwrap()
        );
        assert_ne!(
            prefix.fingerprint().unwrap(),
            boundary.fingerprint().unwrap()
        );
        assert_ne!(
            boundary.fingerprint().unwrap(),
            conditioned.fingerprint().unwrap()
        );
        let round_trip: ModelProgram =
            serde_json::from_slice(&serde_json::to_vec(&conditioned).unwrap()).unwrap();
        assert_eq!(round_trip, conditioned);
    }
}

#[test]
fn unsupported_lifetimes_cannot_be_opted_in_by_wire_or_struct_literal() {
    for lifetime in [StateLifetime::Request, StateLifetime::Step] {
        let mut state = state();
        state.lifetime = lifetime;
        assert!(program("family.fixture", state.clone()).is_ok());
        state.checkpoint = supported(StateCheckpointContents::BoundaryValue);
        assert!(program("family.fixture", state.clone()).is_err());
        assert!(serde_json::from_value::<StateSpec>(serde_json::to_value(state).unwrap()).is_err());
    }
}

#[test]
fn state_checkpoint_wire_rejects_unknown_version_fields_and_null() {
    let mut state = state();
    state.checkpoint = supported(StateCheckpointContents::BoundaryValue);
    let wire = serde_json::to_value(state).unwrap();
    for version in [ContractVersion::new(1, 1), ContractVersion::new(2, 0)] {
        let mut changed = wire.clone();
        changed["checkpoint"]["completed_boundary"]["contract_version"] =
            serde_json::to_value(version).unwrap();
        assert!(serde_json::from_value::<StateSpec>(changed).is_err());
    }
    let mut changed = wire.clone();
    changed["checkpoint"]["completed_boundary"]["undeclared_dependency"] = json!(true);
    assert!(serde_json::from_value::<StateSpec>(changed).is_err());
    let mut changed = wire;
    changed["checkpoint"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<StateSpec>(changed).is_err());
}

#[test]
fn checkpoint_program_inputs_require_complete_explicit_identity_and_version() {
    use crate::vnext::ProgramCheckpointInputs;
    use std::collections::BTreeSet;
    let original = program("family.fixture", state()).unwrap();
    assert!(serde_json::to_value(&original)
        .unwrap()
        .get("checkpoint_inputs")
        .is_none());
    let declared = original
        .clone()
        .with_checkpoint_inputs(
            ProgramCheckpointInputs::new(
                ProgramValueId::new("value.input").unwrap(),
                BTreeSet::new(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_ne!(
        original.fingerprint().unwrap(),
        declared.fingerprint().unwrap()
    );
    assert!(original
        .with_checkpoint_inputs(
            ProgramCheckpointInputs::new(
                ProgramValueId::new("value.unknown").unwrap(),
                BTreeSet::new(),
            )
            .unwrap()
        )
        .is_err());
    assert!(ProgramCheckpointInputs::new(
        ProgramValueId::new("value.input").unwrap(),
        BTreeSet::from([ProgramValueId::new("value.input").unwrap()])
    )
    .is_err());
    let wire = serde_json::to_value(&declared).unwrap();
    assert_eq!(
        serde_json::from_value::<ModelProgram>(wire.clone()).unwrap(),
        declared
    );
    let mut future = wire;
    future["checkpoint_inputs"]["contract_version"] =
        serde_json::to_value(ContractVersion::new(1, 1)).unwrap();
    assert!(serde_json::from_value::<ModelProgram>(future).is_err());
}
