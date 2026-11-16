use super::*;

fn descriptor(name: &str) -> DynamicResourceDescriptor {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let mut descriptor = step_descriptor(name, 64, BufferUsage::Activations, storage);
    descriptor.demand = DynamicResourceDemand::fixed(64).unwrap();
    descriptor
}

fn node(
    name: &str,
    dependencies: &[&str],
    inputs: &[(&str, &str)],
    output: (&str, &str),
) -> PlanNode {
    let resources = inputs
        .iter()
        .map(|(_, resource)| *resource)
        .chain(std::iter::once(output.1))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut node = plan_node(name, dependencies, &resources);
    node.values = inputs
        .iter()
        .map(|(value, resource)| {
            token_binding(
                value,
                ResolvedValueRole::Input,
                TensorAccess::Read,
                resource,
            )
        })
        .chain(std::iter::once(token_binding(
            output.0,
            ResolvedValueRole::Output,
            TensorAccess::Write,
            output.1,
        )))
        .collect();
    node
}

fn chain() -> Vec<PlanNode> {
    vec![
        node(
            "node/a",
            &[],
            &[("value/x", "resource/x")],
            ("value/a", "resource/a"),
        ),
        node(
            "node/b",
            &["node/a"],
            &[("value/a", "resource/a")],
            ("value/b", "resource/b"),
        ),
        node(
            "node/c",
            &["node/b"],
            &[("value/b", "resource/b")],
            ("value/c", "resource/c"),
        ),
        node(
            "node/out",
            &["node/c"],
            &[("value/c", "resource/c"), ("value/late", "resource/late")],
            ("value/out", "resource/out"),
        ),
    ]
}

fn pools(nodes: &[PlanNode], retained: &[&str]) -> Vec<DynamicBackingPoolSpec> {
    let descriptors =
        ["x", "late", "a", "b", "c", "out"].map(|name| descriptor(&format!("resource/{name}")));
    MemoryPlan::derive_dynamic_pools_with_completion_retention(
        &descriptors,
        nodes,
        1 << 20,
        &retained
            .iter()
            .map(|name| ResourceId::new(*name).unwrap())
            .collect(),
    )
    .unwrap()
}

fn token_chain() -> Vec<PlanNode> {
    let mut nodes = chain();
    for node in &mut nodes {
        let mut input_ordinal = 0;
        let mut output_ordinal = 0;
        for value in &mut node.values {
            let ordinal = if value.role() == ResolvedValueRole::Input {
                &mut input_ordinal
            } else {
                &mut output_ordinal
            };
            *value = with_ordinal(value, *ordinal);
            *ordinal += 1;
        }
        let projections = node.values.iter().map(token_projection).collect::<Vec<_>>();
        node.work = NodeWorkContract::Tokens {
            source: projections[0].clone(),
            projections,
        };
    }
    nodes
}

fn with_ordinal(value: &ResolvedValueBinding, ordinal: u32) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        value.value_id().clone(),
        value.role(),
        ordinal,
        value.tensor().clone(),
        value.access(),
        value.alias().clone(),
        value.usage(),
        value.weight().cloned(),
        value.storage().clone(),
    )
    .unwrap()
}

fn token_projection(value: &ResolvedValueBinding) -> NodeTokenBindingProjection {
    NodeTokenBindingProjection {
        value_id: value.value_id().clone(),
        role: value.role(),
        ordinal: value.ordinal(),
        axis: 0,
        rank: 2,
        canonical_extent: 8,
    }
}

fn token_descriptors() -> Vec<DynamicResourceDescriptor> {
    ["x", "late", "a", "b", "c", "out"]
        .map(|name| {
            let mut descriptor = descriptor(&format!("resource/{name}"));
            // Match the actual [8, 4] F16 binding: eight bytes per token,
            // independently aligned in the real dynamic pool.
            descriptor.demand = DynamicResourceDemand::tokens(8, 8).unwrap();
            descriptor
        })
        .to_vec()
}

#[test]
fn token_scaled_late_uploaded_inputs_are_dedicated_while_internal_values_still_reuse() {
    let nodes = token_chain();
    let descriptors = token_descriptors();
    let retained = BTreeSet::from([ResourceId::new("resource/out").unwrap()]);
    let pools = MemoryPlan::derive_dynamic_pools_with_completion_retention(
        &descriptors,
        &nodes,
        1 << 20,
        &retained,
    )
    .unwrap();
    assert_eq!(pools.len(), 1);
    let pool = &pools[0];
    assert_eq!(pool.minimum_step_bytes(), 4 * 16);
    let shared = pool
        .step_resource_slots()
        .iter()
        .filter(|slot| slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave)
        .collect::<Vec<_>>();
    assert_eq!(shared.len(), 2);
    // The completion output begins at node/out. It may take b's dead
    // backing, while its own lifetime still extends through terminal readback.
    assert!(shared.iter().any(|slot| slot.resource_ids()
        == &[
            ResourceId::new("resource/b").unwrap(),
            ResourceId::new("resource/out").unwrap(),
        ]));
    assert_eq!(
        shared[0].resource_ids(),
        &[
            ResourceId::new("resource/a").unwrap(),
            ResourceId::new("resource/c").unwrap()
        ]
    );
    // Every input is already uploaded when node/a starts. In particular late
    // must not occupy b's slot before node/out reads its original bytes.
    for name in ["resource/x", "resource/late"] {
        let resource = ResourceId::new(name).unwrap();
        let slot = pool
            .step_resource_slots()
            .iter()
            .find(|slot| slot.resource_ids().contains(&resource))
            .unwrap();
        assert_eq!(slot.kind(), StepResourceSlotKind::Dedicated);
        assert_eq!(slot.resource_ids(), &[resource]);
    }
    let exact = MemoryPlan::from_core_with_completion_retention(
        64,
        64,
        0,
        1,
        vec![],
        descriptors.clone(),
        &nodes,
        None,
        &retained,
    )
    .unwrap();
    assert_eq!(exact.minimum_runnable_request_bytes(), 64);
    assert!(matches!(
        MemoryPlan::from_core_with_completion_retention(
            63,
            63,
            0,
            1,
            vec![],
            descriptors,
            &nodes,
            None,
            &retained,
        ),
        Err(VNextError::PlanCapacityExceeded {
            required_bytes: 64,
            usable_capacity_bytes: 63
        })
    ));
}

#[test]
fn token_scaled_external_alias_cannot_use_an_internal_producer_lifetime() {
    let mut nodes = token_chain();
    let alias = with_ordinal(
        &token_binding(
            "value/external-alias",
            ResolvedValueRole::Input,
            TensorAccess::Read,
            "resource/a",
        ),
        1,
    );
    if let NodeWorkContract::Tokens { projections, .. } = &mut nodes[0].work {
        projections.push(token_projection(&alias));
    }
    nodes[0].values.push(alias);
    let pools = MemoryPlan::derive_dynamic_pools_with_completion_retention(
        &token_descriptors(),
        &nodes,
        1 << 20,
        &BTreeSet::from([ResourceId::new("resource/out").unwrap()]),
    )
    .unwrap();
    assert_eq!(pools[0].minimum_step_bytes(), 5 * 16);
    for resource in ["resource/a", "resource/late"] {
        let resource = ResourceId::new(resource).unwrap();
        let slot = pools[0]
            .step_resource_slots()
            .iter()
            .find(|slot| slot.resource_ids().contains(&resource))
            .unwrap();
        assert_eq!(slot.kind(), StepResourceSlotKind::Dedicated);
        assert_eq!(slot.resource_ids(), &[resource]);
    }
}

#[test]
fn fixed_uploaded_input_reuses_storage_only_after_its_last_read() {
    let pools = pools(&chain(), &["resource/out"]);
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].minimum_step_bytes(), 3 * 64);
    let shared = pools[0]
        .step_resource_slots()
        .iter()
        .filter(|slot| slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave)
        .collect::<Vec<_>>();
    assert_eq!(shared.len(), 2);
    // x is uploaded before node/a and dies there; b and then the retained
    // output can overwrite its Step copy after the proven completion edges.
    assert!(shared.iter().any(|slot| {
        ["resource/x", "resource/b", "resource/out"]
            .into_iter()
            .all(|id| slot.resource_ids().contains(&ResourceId::new(id).unwrap()))
    }));
    assert_eq!(
        shared[0].resource_ids(),
        &[
            ResourceId::new("resource/a").unwrap(),
            ResourceId::new("resource/c").unwrap()
        ]
    );
    for name in ["resource/late"] {
        let resource = ResourceId::new(name).unwrap();
        let slot = pools[0]
            .step_resource_slots()
            .iter()
            .find(|slot| slot.resource_ids().contains(&resource))
            .unwrap();
        assert_eq!(slot.kind(), StepResourceSlotKind::Dedicated);
    }
}

#[test]
fn fixed_retention_overlap_and_missing_completion_order_prevent_reuse() {
    let retained = pools(&chain(), &["resource/a", "resource/out"]);
    assert_eq!(retained[0].minimum_step_bytes(), 4 * 64);
    // An input which is also a terminal result cannot be overwritten.
    let retained_input = pools(&chain(), &["resource/x", "resource/out"]);
    assert_eq!(retained_input[0].minimum_step_bytes(), 4 * 64);
    assert!(retained_input[0]
        .step_resource_slots()
        .iter()
        .any(|slot| { slot.resource_ids() == &[ResourceId::new("resource/x").unwrap()] }));

    let mut unordered = chain();
    unordered[2].dependencies.clear();
    let unordered = pools(&unordered, &["resource/out"]);
    assert_eq!(unordered[0].minimum_step_bytes(), 5 * 64);

    let mut overlap = chain();
    overlap[2].values.push(token_binding(
        "value/a",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/a",
    ));
    overlap[2]
        .resources
        .push(ResourceId::new("resource/a").unwrap());
    overlap[2].resources.sort();
    let overlap = pools(&overlap, &["resource/out"]);
    assert_eq!(overlap[0].minimum_step_bytes(), 4 * 64);
}

#[test]
fn serial_execution_orders_independent_branches_before_reusing_their_storage() {
    let mut nodes = vec![
        node(
            "node/a",
            &[],
            &[("value/x", "resource/x")],
            ("value/a", "resource/a"),
        ),
        node(
            "node/b",
            &["node/a"],
            &[("value/a", "resource/a")],
            ("value/b", "resource/b"),
        ),
        node(
            "node/c",
            &[],
            &[("value/x", "resource/x")],
            ("value/c", "resource/c"),
        ),
        node(
            "node/out",
            &["node/b", "node/c"],
            &[
                ("value/b", "resource/b"),
                ("value/c", "resource/c"),
                ("value/late", "resource/late"),
            ],
            ("value/out", "resource/out"),
        ),
    ];
    let original = nodes.clone();
    ExecutionPlan::apply_execution_order(&mut nodes, PlanExecutionOrder::DataDependencies);
    assert_eq!(nodes, original);
    assert_eq!(
        pools(&nodes, &["resource/out"])[0].minimum_step_bytes(),
        5 * 64
    );
    let shares = |pool: &DynamicBackingPoolSpec, left: &str, right: &str| {
        pool.step_resource_slots().iter().any(|slot| {
            slot.resource_ids()
                .contains(&ResourceId::new(left).unwrap())
                && slot
                    .resource_ids()
                    .contains(&ResourceId::new(right).unwrap())
        })
    };
    assert!(!shares(
        &pools(&nodes, &["resource/out"])[0],
        "resource/a",
        "resource/c"
    ));

    ExecutionPlan::apply_execution_order(&mut nodes, PlanExecutionOrder::Serial);
    assert_eq!(nodes[2].dependencies, vec![NodeId::new("node/b").unwrap()]);
    assert_eq!(nodes[3].dependencies, original[3].dependencies);
    let serial = pools(&nodes, &["resource/out"]);
    assert_eq!(serial[0].minimum_step_bytes(), 4 * 64);
    assert!(shares(&serial[0], "resource/a", "resource/c"));
    // The serial proof also allows the output to reuse x after node/c reads
    // it. Without that proof all uploaded inputs retain dedicated backing.
    let descriptors = ["x", "late", "a", "b", "c", "out"]
        .map(|name| descriptor(&format!("resource/{name}")))
        .to_vec();
    let retained = BTreeSet::from([ResourceId::new("resource/out").unwrap()]);
    for capacity in [4 * 64, 4 * 64 - 1] {
        let plan = MemoryPlan::from_core_with_completion_retention(
            capacity,
            capacity,
            0,
            1,
            vec![],
            descriptors.clone(),
            &nodes,
            None,
            &retained,
        );
        if capacity == 4 * 64 {
            assert_eq!(plan.unwrap().minimum_runnable_request_bytes(), capacity);
        } else {
            assert!(matches!(
                plan,
                Err(VNextError::PlanCapacityExceeded {
                    required_bytes: 256,
                    usable_capacity_bytes: 255
                })
            ));
        }
    }
    // Serial execution does not shorten a real reader/retention lifetime.
    assert_eq!(
        pools(&nodes, &["resource/a", "resource/out"])[0].minimum_step_bytes(),
        5 * 64
    );
    nodes[3].values.push(token_binding(
        "value/a",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/a",
    ));
    nodes[3]
        .resources
        .push(ResourceId::new("resource/a").unwrap());
    nodes[3].resources.sort();
    assert_eq!(
        pools(&nodes, &["resource/out"])[0].minimum_step_bytes(),
        5 * 64
    );
}

#[test]
fn fixed_external_alias_cannot_borrow_an_internal_producer_proof() {
    let mut nodes = chain();
    // Even if a lower-level declaration aliases an external input with an
    // internal output, the whole physical resource remains non-reusable.
    nodes[0].values.push(token_binding(
        "value/external-alias",
        ResolvedValueRole::Input,
        TensorAccess::Read,
        "resource/a",
    ));
    let pools = pools(&nodes, &["resource/out"]);
    assert_eq!(pools[0].minimum_step_bytes(), 4 * 64);
}

#[test]
fn fixed_resources_without_producer_bindings_fail_closed_to_dedicated() {
    let descriptors = [descriptor("resource/a"), descriptor("resource/b")];
    let nodes = [
        plan_node("node/a", &[], &["resource/a"]),
        plan_node("node/b", &["node/a"], &["resource/b"]),
    ];
    let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 1 << 20).unwrap();
    assert_eq!(pools[0].minimum_step_bytes(), 128);
}
