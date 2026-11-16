use super::*;

fn view(source: &str, output: &str, name: &str, dependencies: &[&str]) -> PlanNode {
    let mut node = plan_node(name, dependencies, &["resource/shared"]);
    let input = token_binding(
        source,
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/shared",
    );
    let output = ResolvedValueBinding::new(
        ProgramValueId::new(output).unwrap(),
        ResolvedValueRole::Output,
        0,
        input.tensor().clone(),
        TensorAccess::Read,
        AliasPolicy::ReadOnlyView { tensor_index: 0 },
        BufferUsage::Activations,
        None,
        input.storage().clone(),
    )
    .unwrap();
    node.exact_aliases = vec![PlanExactAlias {
        input_value_id: input.value_id().clone(),
        input_ordinal: 0,
        output_value_id: output.value_id().clone(),
        output_ordinal: 0,
        kind: PlanExactAliasKind::ReadOnlyView,
    }];
    node.values = vec![input, output];
    node
}

fn validate(nodes: &[PlanNode]) -> Result<(), VNextError> {
    let mut values = BTreeMap::new();
    for binding in nodes.iter().flat_map(|n| &n.values) {
        ExecutionPlan::validate_cross_node_value(binding, &mut values)?;
    }
    ExecutionPlan::validate_global_storage_aliasing(&values, nodes)
}

#[test]
fn read_only_view_does_not_authorize_undeclared_storage_aliases() {
    let mut node = view("value/source", "value/a", "node/a", &[]);
    let unrelated = token_binding(
        "value/unrelated",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/shared",
    );
    node.values.push(unrelated);
    assert!(validate(&[node])
        .unwrap_err()
        .to_string()
        .contains("undeclared"));
}

#[test]
fn read_only_view_forks_and_transitive_views_keep_the_original_readable() {
    let mut nodes = vec![
        view("value/source", "value/a", "node/a", &[]),
        view("value/a", "value/b", "node/b", &["node/a"]),
        view("value/source", "value/c", "node/c", &["node/a"]),
    ];
    let mut late = plan_node("node/late", &["node/b", "node/c"], &["resource/shared"]);
    late.values.push(token_binding(
        "value/source",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/shared",
    ));
    nodes.push(late);
    validate(&nodes).unwrap();
    let classes = ExecutionPlan::alias_classes(&nodes).unwrap();
    assert_eq!(
        classes[&ProgramValueId::new("value/source").unwrap()],
        classes[&ProgramValueId::new("value/b").unwrap()]
    );
    assert_eq!(classes.len(), 4);
}

#[test]
fn read_only_view_rejects_mutation_through_source_and_transitive_aliases() {
    for value in ["value/source", "value/a", "value/b"] {
        let mut nodes = vec![
            view("value/source", "value/a", "node/a", &[]),
            view("value/a", "value/b", "node/b", &["node/a"]),
        ];
        let mut mutation = plan_node("node/mutate", &["node/b"], &["resource/shared"]);
        mutation.values.push(token_binding(
            value,
            ResolvedValueRole::Input,
            TensorAccess::ReadWrite,
            "resource/shared",
        ));
        nodes.push(mutation);
        assert!(validate(&nodes)
            .unwrap_err()
            .to_string()
            .contains("mutates read-only view"));
    }
}

#[test]
fn read_only_view_allows_one_original_producer_but_no_in_place_writer() {
    let mut producer = plan_node("node/producer", &[], &["resource/shared"]);
    producer.values.push(token_binding(
        "value/source",
        ResolvedValueRole::Output,
        TensorAccess::Write,
        "resource/shared",
    ));
    let mut nodes = vec![
        producer,
        view("value/source", "value/a", "node/a", &["node/producer"]),
    ];
    validate(&nodes).unwrap();
    let mut writer = view("value/a", "value/written", "node/write", &["node/a"]);
    let old = &writer.values[1];
    writer.values[1] = ResolvedValueBinding::new(
        old.value_id().clone(),
        old.role(),
        0,
        old.tensor().clone(),
        TensorAccess::Write,
        AliasPolicy::MustAlias { tensor_index: 0 },
        old.usage(),
        None,
        old.storage().clone(),
    )
    .unwrap();
    writer.exact_aliases[0].kind = PlanExactAliasKind::MustAlias;
    nodes.push(writer);
    assert!(validate(&nodes).is_err());
}

#[test]
fn read_only_view_storage_cannot_be_reused_before_a_late_reader() {
    let mut first = plan_node("node/producer", &[], &["resource/shared"]);
    first.values.push(token_binding(
        "value/source",
        ResolvedValueRole::Output,
        TensorAccess::Write,
        "resource/shared",
    ));
    let view = view(
        "value/source",
        "value/view",
        "node/view",
        &["node/producer"],
    );
    let mut middle = plan_node("node/middle", &["node/view"], &["resource/other"]);
    middle.values.push(token_binding(
        "value/middle",
        ResolvedValueRole::Output,
        TensorAccess::Write,
        "resource/other",
    ));
    let mut late = plan_node(
        "node/late",
        &["node/middle"],
        &["resource/shared", "resource/other"],
    );
    late.values.push(token_binding(
        "value/view",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/shared",
    ));
    late.values.push(token_binding(
        "value/middle",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/other",
    ));
    let nodes = [first, view, middle, late];
    validate(&nodes).unwrap();
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "layout").unwrap(),
    )
    .unwrap();
    let descriptors = ["resource/shared", "resource/other"].map(|name| {
        let mut d = step_descriptor(name, 64, BufferUsage::Activations, storage.clone());
        d.demand = DynamicResourceDemand::fixed(64).unwrap();
        d
    });
    let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 1 << 20).unwrap();
    assert_eq!(
        pools.iter().map(|p| p.minimum_step_bytes()).sum::<u64>(),
        128
    );
}

#[test]
fn read_only_view_encoding_has_no_compute_and_cannot_erase_writable_work() {
    let node = view("value/source", "value/a", "node/a", &[]);
    let operation = crate::EncodedDeviceOperation::<()>::read_only_view(&node.values).unwrap();
    assert_eq!(operation.compute_command_count(), 0);
    assert!(operation
        .validate_compute_bindings(&node.values, false)
        .is_err());
    let ordinary = [token_binding(
        "value/result",
        ResolvedValueRole::Output,
        TensorAccess::Write,
        "resource/result",
    )];
    assert!(operation
        .validate_compute_bindings(&ordinary, true)
        .is_err());
    assert!(crate::EncodedDeviceOperation::<()>::read_only_view(&[]).is_err());
    let mut commands = crate::DeviceCommandBatch::with_capacity(0);
    commands.push_operation(0, operation);
    assert_eq!(commands.len(), 0);
}
