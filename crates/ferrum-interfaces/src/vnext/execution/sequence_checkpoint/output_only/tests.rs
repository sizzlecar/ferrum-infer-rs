use super::*;
use crate::vnext::{
    AliasPolicy, AllocationLifetime, BufferUsage, ElementType, NodeId, NodeTokenBindingProjection,
    NodeWorkContract, PlanExactAlias, PlanExactAliasKind, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueBinding, ResolvedValueStorage, ResourceId, StateId,
};
use std::collections::BTreeSet;

fn value(name: &str) -> ProgramValueId {
    ProgramValueId::new(name).unwrap()
}

fn binding(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    access: TensorAccess,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        value(name),
        role,
        ordinal,
        ResolvedTensorSpec::new(vec![1], ElementType::F32, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        access,
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        ResolvedValueStorage::single(
            ResourceId::new(format!("resource.{name}")).unwrap(),
            0,
            4,
            ElementType::F32,
        )
        .unwrap(),
    )
    .unwrap()
}

fn node(name: &str, input: &str, output: &str) -> PlanNode {
    let mut node = PlanNode::resource_test_node(NodeId::new(name).unwrap());
    node.values = vec![
        binding(input, ResolvedValueRole::Input, 0, TensorAccess::Read),
        binding(output, ResolvedValueRole::Output, 0, TensorAccess::Write),
    ];
    node
}

fn stateful(name: &str, input: &str) -> PlanNode {
    let mut stateful = node(name, input, "value.end");
    let fixture = PlanNode::resource_test_node_with_state_effect(
        stateful.id().clone(),
        StateId::new("state.fixture").unwrap(),
        value("value.state"),
        AllocationLifetime::Sequence,
        TensorAccess::ReadWrite,
        vec![ResourceId::new("resource.state").unwrap()],
    );
    stateful.state_effects = fixture.state_effects;
    stateful
}

fn declaration(input: &str) -> ProgramCheckpointInputs {
    ProgramCheckpointInputs::new(value("value.tokens"), BTreeSet::new())
        .unwrap()
        .with_output_only_inputs(BTreeSet::from([value(input)]))
        .unwrap()
}

#[test]
fn transitive_values_and_writable_inputs_reach_state_without_name_heuristics() {
    for input in ["value.selection", "value.arbitrary"] {
        let first = node("node.first", input, "value.middle");
        let mut mutation = node("node.mutation", "value.middle", "value.unused");
        mutation.values.push(binding(
            "value.shared",
            ResolvedValueRole::Input,
            1,
            TensorAccess::ReadWrite,
        ));
        let state = stateful("node.state", "value.shared");
        let reasons = state_dependencies(&declaration(input), &[first, mutation, state]).unwrap();
        assert!(
            matches!(&reasons[0], SequenceCheckpointUnsupportedReason::OutputOnlyInputAffectsState { value_id, node_id, .. }
            if value_id == &value(input) && node_id.as_str() == "node.state")
        );
    }
}

#[test]
fn identical_node_names_do_not_override_actual_dependency_edges() {
    let inputs = declaration("value.selection");
    let unrelated = [
        stateful("node.state", "value.tokens"),
        node("node.output", "value.selection", "value.end"),
    ];
    assert!(state_dependencies(&inputs, &unrelated).unwrap().is_empty());
    let dependent = [
        node("node.output", "value.selection", "value.middle"),
        stateful("node.state", "value.middle"),
    ];
    assert!(!state_dependencies(&inputs, &dependent).unwrap().is_empty());
}

#[test]
fn token_work_source_remains_a_dependency_for_a_mutable_input() {
    let mut state = stateful("node.state", "value.selection");
    state.values[0] = binding(
        "value.selection",
        ResolvedValueRole::Input,
        0,
        TensorAccess::ReadWrite,
    );
    state.work = NodeWorkContract::Tokens {
        source: NodeTokenBindingProjection {
            value_id: value("value.selection"),
            role: ResolvedValueRole::Input,
            ordinal: 0,
            axis: 0,
            rank: 1,
            canonical_extent: 1,
        },
        projections: vec![],
    };
    assert!(
        !state_dependencies(&declaration("value.selection"), &[state])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn exact_storage_aliases_propagate_to_the_original_writable_value() {
    let mut alias = node("node.alias", "value.shared", "value.aliased");
    alias.values.insert(
        1,
        binding(
            "value.selection",
            ResolvedValueRole::Input,
            1,
            TensorAccess::Read,
        ),
    );
    let output = ResolvedValueBinding::new(
        value("value.aliased"),
        ResolvedValueRole::Output,
        0,
        alias.values[0].tensor().clone(),
        TensorAccess::Write,
        AliasPolicy::MustAlias { tensor_index: 0 },
        BufferUsage::Activations,
        None,
        alias.values[0].storage().clone(),
    )
    .unwrap();
    alias.values[2] = output;
    alias.exact_aliases = vec![PlanExactAlias {
        output_value_id: value("value.aliased"),
        output_ordinal: 0,
        input_value_id: value("value.shared"),
        input_ordinal: 0,
        kind: PlanExactAliasKind::MustAlias,
    }];
    assert!(!state_dependencies(
        &declaration("value.selection"),
        // Alias closure is intentionally conservative about reuse of an
        // earlier-read value; it cannot turn writes into independent inputs.
        &[stateful("node.state", "value.shared"), alias]
    )
    .unwrap()
    .is_empty());
}
