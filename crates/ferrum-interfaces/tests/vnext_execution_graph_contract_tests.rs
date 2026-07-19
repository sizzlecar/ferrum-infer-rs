mod vnext_core_contract;

use vnext_core_contract::*;

fn graph_operation(
    operation_id: &str,
    state_access: Option<TensorAccess>,
    output_alias: AliasPolicy,
) -> OperationDescriptor {
    let mut inputs = Vec::new();
    if let Some(access) = state_access {
        inputs.push(tensor_contract(
            ElementType::F32,
            access,
            AliasPolicy::NoAlias,
        ));
    }
    inputs.extend([
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
        tensor_contract(ElementType::F32, TensorAccess::Read, AliasPolicy::NoAlias),
    ]);
    OperationDescriptor {
        id: id(operation_id),
        version: ContractVersion::new(1, 0),
        inputs,
        outputs: vec![tensor_contract(
            ElementType::F32,
            TensorAccess::Write,
            output_alias,
        )],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Required,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Required,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::from([id("capability.compute")]),
        },
        profile_phase: ProfilePhase::Decode,
    }
}

fn graph_catalog(alias_policy: AliasPolicy) -> CapabilityCatalog {
    let operations = vec![
        graph_operation("operation.graph.alias", None, alias_policy),
        graph_operation("operation.graph.consume", None, AliasPolicy::NoAlias),
        graph_operation(
            "operation.graph.state-read",
            Some(TensorAccess::Read),
            AliasPolicy::NoAlias,
        ),
        graph_operation(
            "operation.graph.state-rw",
            Some(TensorAccess::ReadWrite),
            AliasPolicy::NoAlias,
        ),
    ];
    let device_id: DeviceId = id("device.execution-graph.0");
    let capabilities = BTreeSet::from([id("capability.compute")]);
    let providers = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| {
            (
                operation.id.clone(),
                vec![OperationProviderDescriptor::new(
                    id(format!("provider.operation.graph.{index}")),
                    operation.id.clone(),
                    operation.fingerprint().unwrap(),
                    sha(char::from(b'1' + index as u8)),
                    ContractVersion::new(1, 0),
                    device_id.clone(),
                    capabilities.clone(),
                    BTreeSet::from([id("weight-format.execution-graph")]),
                    BTreeSet::new(),
                    contiguous_storage_bindings(operation),
                    format!("resource-estimator.graph.{index}"),
                    ContractVersion::new(1, 0),
                    sha(char::from(b'5' + index as u8)),
                )
                .unwrap()],
            )
        })
        .collect::<BTreeMap<_, _>>();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1 << 20,
            runtime_implementation_fingerprint: sha('a'),
            capabilities: capabilities.clone(),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        operations,
        providers,
        vec![EngineProviderDescriptor::new(
            id("provider.engine.execution-graph"),
            ContractVersion::new(1, 0),
            sha('b'),
            device_id,
            capabilities,
        )
        .unwrap()],
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
enum GraphAliasStorage {
    Distinct,
    ExactTarget,
    PartialTarget,
    ExactWrongInput,
}

fn graph_value_binding(
    value_id: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    resource_id: &str,
    offset_bytes: u64,
) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id(value_id),
        role,
        ordinal,
        resolved_tensor(ElementType::F32),
        access,
        alias,
        usage,
        ResolvedValueStorage::single(id(resource_id), offset_bytes, 16, ElementType::F32).unwrap(),
    )
    .unwrap()
}

fn graph_weight_binding(ordinal: u32) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id("value.weight"),
        ResolvedValueRole::Input,
        ordinal,
        resolved_tensor(ElementType::F32),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
            Some(id("weight.component")),
            id("resource.graph.weight"),
            0,
            16,
            ElementType::F32,
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

fn graph_node_bindings(
    node: &ProgramNode,
    alias_policy: &AliasPolicy,
    alias_storage: GraphAliasStorage,
) -> Vec<ResolvedValueBinding> {
    match node.operation_id.as_str() {
        "operation.graph.alias" => {
            let (resource, offset) = match alias_storage {
                GraphAliasStorage::Distinct => ("resource.graph.alias", 0),
                GraphAliasStorage::ExactTarget => ("resource.graph.input.0", 0),
                GraphAliasStorage::PartialTarget => ("resource.graph.input.0", 8),
                GraphAliasStorage::ExactWrongInput => ("resource.graph.input.1", 0),
            };
            vec![
                graph_value_binding(
                    "value.input.0",
                    ResolvedValueRole::Input,
                    0,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.0",
                    0,
                ),
                graph_value_binding(
                    "value.input.1",
                    ResolvedValueRole::Input,
                    1,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.1",
                    0,
                ),
                graph_weight_binding(2),
                graph_value_binding(
                    "value.alias",
                    ResolvedValueRole::Output,
                    0,
                    TensorAccess::Write,
                    alias_policy.clone(),
                    BufferUsage::Activations,
                    resource,
                    offset,
                ),
            ]
        }
        "operation.graph.consume" => vec![
            graph_value_binding(
                "value.input.0",
                ResolvedValueRole::Input,
                0,
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.input.0",
                0,
            ),
            graph_value_binding(
                "value.input.1",
                ResolvedValueRole::Input,
                1,
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.input.1",
                0,
            ),
            graph_weight_binding(2),
            graph_value_binding(
                "value.late",
                ResolvedValueRole::Output,
                0,
                TensorAccess::Write,
                AliasPolicy::NoAlias,
                BufferUsage::Activations,
                "resource.graph.late",
                0,
            ),
        ],
        "operation.graph.state-read" | "operation.graph.state-rw" => {
            let state_access = if node.operation_id.as_str() == "operation.graph.state-read" {
                TensorAccess::Read
            } else {
                TensorAccess::ReadWrite
            };
            vec![
                graph_value_binding(
                    "value.state",
                    ResolvedValueRole::Input,
                    0,
                    state_access,
                    AliasPolicy::NoAlias,
                    BufferUsage::State,
                    "resource.graph.state",
                    0,
                ),
                graph_value_binding(
                    "value.input.0",
                    ResolvedValueRole::Input,
                    1,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.0",
                    0,
                ),
                graph_value_binding(
                    "value.input.1",
                    ResolvedValueRole::Input,
                    2,
                    TensorAccess::Read,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    "resource.graph.input.1",
                    0,
                ),
                graph_weight_binding(3),
                graph_value_binding(
                    node.outputs[0].as_str(),
                    ResolvedValueRole::Output,
                    0,
                    TensorAccess::Write,
                    AliasPolicy::NoAlias,
                    BufferUsage::Activations,
                    &format!("resource.graph.{}", node.outputs[0].as_str()),
                    0,
                ),
            ]
        }
        _ => unreachable!(),
    }
}

struct GraphPlanFixture {
    family: PreparedModelFamily,
    catalog: CapabilityCatalog,
    policy: ResolvedRuntimePolicy,
    resolutions: Vec<PlanNodeResolution>,
    plan: ExecutionPlan,
}

fn graph_plan_fixture(
    scenario: &str,
    alias_policy: AliasPolicy,
    alias_storage: GraphAliasStorage,
) -> Result<GraphPlanFixture, VNextError> {
    let family =
        TypedFamilyRegistration::new(GraphFamily).prepare(&json!({"scenario": scenario}))?;
    let catalog = graph_catalog(alias_policy.clone());
    let policy = policy(16 * 1024);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut resolutions = Vec::new();
    for node in family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
    {
        let planning_handle = planning.planning();
        resolutions.push(PlanNodeResolution::resolve(
            &family,
            &catalog,
            &policy,
            &planning_handle,
            node.id.clone(),
            graph_node_bindings(node, &alias_policy, alias_storage),
            BTreeSet::new(),
            None,
        )?);
    }
    let plan = ExecutionPlan::build(PlanBuildRequest::new(
        &family,
        &catalog,
        &policy,
        resolutions.clone(),
    )?)?;
    Ok(GraphPlanFixture {
        family,
        catalog,
        policy,
        resolutions,
        plan,
    })
}

#[test]
fn execution_alias_must_alias_builds_exact_equivalence_and_single_allocation() {
    let fixture = graph_plan_fixture(
        "alias",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    let node = &fixture.plan.payload().nodes()[0];
    assert_eq!(node.exact_aliases().len(), 1);
    assert_eq!(
        node.exact_aliases()[0].kind(),
        PlanExactAliasKind::MustAlias
    );
    assert_eq!(
        node.exact_aliases()[0].input_value_id().as_str(),
        "value.input.0"
    );
    assert_eq!(
        node.exact_aliases()[0].output_value_id().as_str(),
        "value.alias"
    );
    let memory = fixture.plan.payload().memory();
    let matching_static = memory
        .static_allocations()
        .iter()
        .filter(|allocation| allocation.resource_id().as_str() == "resource.graph.input.0")
        .count();
    let matching_dynamic = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.base_resource_id().as_str() == "resource.graph.input.0")
        .count();
    assert_eq!(matching_static + matching_dynamic, 1);
}

#[test]
fn execution_alias_may_alias_supports_distinct_or_exact_storage() {
    let distinct = graph_plan_fixture(
        "alias",
        AliasPolicy::MayAlias { tensor_index: 0 },
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    assert!(distinct.plan.payload().nodes()[0]
        .exact_aliases()
        .is_empty());

    let exact = graph_plan_fixture(
        "alias",
        AliasPolicy::MayAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    assert_eq!(exact.plan.payload().nodes()[0].exact_aliases().len(), 1);
    assert_eq!(
        exact.plan.payload().nodes()[0].exact_aliases()[0].kind(),
        PlanExactAliasKind::MayAlias
    );
}

#[test]
fn execution_alias_rejects_partial_and_wrong_input_overlap() {
    for storage in [
        GraphAliasStorage::PartialTarget,
        GraphAliasStorage::ExactWrongInput,
    ] {
        assert!(
            graph_plan_fixture("alias", AliasPolicy::MayAlias { tensor_index: 0 }, storage,)
                .is_err()
        );
    }
}

#[test]
fn execution_alias_rejects_overwrite_before_last_consumer() {
    assert!(graph_plan_fixture(
        "alias_late_consumer",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .is_err());
}

#[test]
fn execution_state_effect_graph_orders_raw_war_waw() {
    let fixture = graph_plan_fixture(
        "state_chain",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    let nodes = fixture.plan.payload().nodes();
    assert_eq!(nodes.len(), 4);
    assert_eq!(nodes[0].state_effects()[0].access(), TensorAccess::Read);
    assert!(nodes[0].dependencies().is_empty());
    assert_eq!(
        nodes[1].state_effects()[0].access(),
        TensorAccess::ReadWrite
    );
    assert_eq!(nodes[1].dependencies(), &[id("node.state-read.0")]);
    assert_eq!(
        nodes[2].state_effects()[0].access(),
        TensorAccess::ReadWrite
    );
    assert_eq!(nodes[2].dependencies(), &[id("node.state-rw.0")]);
    assert_eq!(nodes[3].state_effects()[0].access(), TensorAccess::Read);
    assert_eq!(nodes[3].dependencies(), &[id("node.state-rw.1")]);
}

#[test]
fn execution_state_read_only_nodes_remain_independent() {
    let fixture = graph_plan_fixture(
        "state_read_only",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    assert!(fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .all(
            |node| node.state_effects()[0].access() == TensorAccess::Read
                && node.dependencies().is_empty()
        ));
}

#[test]
fn execution_alias_effect_wire_mutations_are_rejected() {
    let alias = graph_plan_fixture(
        "alias",
        AliasPolicy::MustAlias { tensor_index: 0 },
        GraphAliasStorage::ExactTarget,
    )
    .unwrap();
    let mut value = serde_json::to_value(&alias.plan).unwrap();
    value["payload"]["nodes"][0]["exact_aliases"] = json!([]);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &alias.family,
        &alias.catalog,
        &alias.policy,
        alias.resolutions.clone(),
    )
    .is_err());

    let state = graph_plan_fixture(
        "state_chain",
        AliasPolicy::NoAlias,
        GraphAliasStorage::Distinct,
    )
    .unwrap();
    let mut value = serde_json::to_value(&state.plan).unwrap();
    value["payload"]["nodes"][1]["state_effects"][0]["access"] = json!("read");
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &state.family,
        &state.catalog,
        &state.policy,
        state.resolutions.clone(),
    )
    .is_err());

    let mut value = serde_json::to_value(&state.plan).unwrap();
    value["payload"]["nodes"][2]["dependencies"] = json!([]);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &state.family,
        &state.catalog,
        &state.policy,
        state.resolutions,
    )
    .is_err());
}
