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

#[test]
fn fixed_internal_liveness_reuses_slots_but_late_uploaded_inputs_stay_dedicated() {
    let pools = pools(&chain(), &["resource/out"]);
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].minimum_step_bytes(), 5 * 64);
    let shared = pools[0]
        .step_resource_slots()
        .iter()
        .filter(|slot| slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave)
        .collect::<Vec<_>>();
    assert_eq!(shared.len(), 1);
    assert_eq!(
        shared[0].resource_ids(),
        &[
            ResourceId::new("resource/a").unwrap(),
            ResourceId::new("resource/c").unwrap()
        ]
    );
    for name in ["resource/x", "resource/late", "resource/out"] {
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
    assert_eq!(retained[0].minimum_step_bytes(), 6 * 64);

    let mut unordered = chain();
    unordered[2].dependencies.clear();
    let unordered = pools(&unordered, &["resource/out"]);
    assert_eq!(unordered[0].minimum_step_bytes(), 6 * 64);

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
    assert_eq!(overlap[0].minimum_step_bytes(), 6 * 64);
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
    assert_eq!(pools[0].minimum_step_bytes(), 6 * 64);
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
