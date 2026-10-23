use super::super::completion_order::CompletionOrderCache;
use super::*;

fn query_result(result: Result<bool, VNextError>) -> Result<bool, String> {
    result.map_err(|error| error.to_string())
}

fn compare_all_queries(nodes: &[PlanNode]) {
    let by_id = nodes
        .iter()
        .map(|node| (node.id.clone(), node))
        .collect::<BTreeMap<_, _>>();
    let mut ids = nodes.iter().map(|node| node.id.clone()).collect::<Vec<_>>();
    ids.push(NodeId::new("node/missing").unwrap());
    ids.push(NodeId::new("node/unrelated").unwrap());
    let mut cache = CompletionOrderCache::new(&by_id);
    // Exercise reuse within one successor and replacement between successors.
    for successor in ids.iter().chain(ids.iter().rev()) {
        for predecessor in ids.iter().chain(ids.iter().rev()) {
            assert_eq!(
                query_result(cache.precedes(predecessor, successor)),
                query_result(node_completion_precedes(&by_id, predecessor, successor)),
                "{predecessor} -> {successor}",
            );
        }
    }
}

#[test]
fn cached_completion_order_matches_dfs_for_branches_cycles_and_non_topological_nodes() {
    let mut nodes = vec![
        plan_node("node/a", &[], &[]),
        plan_node("node/b", &["node/a"], &[]),
        plan_node("node/c", &["node/a"], &[]),
        plan_node("node/d", &["node/b", "node/c"], &[]),
        plan_node("node/e", &[], &[]),
    ];
    compare_all_queries(&nodes);
    nodes.swap(0, 3);
    compare_all_queries(&nodes);
    nodes[3].dependencies.push(NodeId::new("node/d").unwrap());
    compare_all_queries(&nodes);
    nodes[4].dependencies.push(NodeId::new("node/e").unwrap());
    compare_all_queries(&nodes);
}

#[test]
fn cached_completion_order_preserves_missing_dependency_early_return_order() {
    let mut nodes = vec![
        plan_node("node/a", &[], &[]),
        plan_node("node/b", &["node/missing"], &[]),
        plan_node("node/c", &["node/a", "node/b"], &[]),
    ];
    compare_all_queries(&nodes);
    // DFS pops the last dependency first. Reversing it changes whether a
    // query for a is resolved before the missing node is encountered.
    let a = NodeId::new("node/a").unwrap();
    let c = NodeId::new("node/c").unwrap();
    {
        let by_id = nodes.iter().map(|node| (node.id.clone(), node)).collect();
        let mut cache = CompletionOrderCache::new(&by_id);
        assert!(cache.precedes(&a, &c).is_err());
        assert!(cache
            .precedes(&NodeId::new("node/missing").unwrap(), &c)
            .unwrap());
    }
    nodes[2].dependencies.reverse();
    compare_all_queries(&nodes);
    let by_id = nodes.iter().map(|node| (node.id.clone(), node)).collect();
    assert!(CompletionOrderCache::new(&by_id).precedes(&a, &c).unwrap());
}

#[test]
fn repeated_predecessors_share_one_traversal_and_cache_retains_only_current_successor() {
    let nodes = (0..64)
        .map(|index| {
            let name = format!("node/{index}");
            let mut node = plan_node(&name, &[], &[]);
            if index != 0 {
                node.dependencies
                    .push(NodeId::new(format!("node/{}", index - 1)).unwrap());
            }
            node
        })
        .collect::<Vec<_>>();
    let by_id = nodes.iter().map(|node| (node.id.clone(), node)).collect();
    let mut cache = CompletionOrderCache::new(&by_id);
    let last = &nodes.last().unwrap().id;
    for node in &nodes[..nodes.len() - 1] {
        assert!(cache.precedes(&node.id, last).unwrap());
    }
    assert_eq!(cache.traversal_count(), 1);
    assert_eq!(cache.retained_ancestor_count(), nodes.len() - 1);
    assert!(!cache.precedes(last, &nodes[0].id).unwrap());
    assert_eq!(cache.retained_ancestor_count(), 0);
    assert_eq!(cache.traversal_count(), 2);
    assert!(cache.precedes(&nodes[0].id, last).unwrap());
    assert_eq!(cache.traversal_count(), 3);
}

fn descriptor(name: &str, fixed: bool) -> DynamicResourceDescriptor {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let mut result = step_descriptor(name, 64, BufferUsage::Activations, storage);
    if fixed {
        result.demand = DynamicResourceDemand::fixed(64).unwrap();
    }
    result
}

fn value_node(
    name: &str,
    dependencies: &[&str],
    inputs: &[(&str, &str)],
    output: (&str, &str),
) -> PlanNode {
    let resources = inputs
        .iter()
        .map(|(_, resource)| *resource)
        .chain(std::iter::once(output.1))
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

fn compare_slots(
    nodes: &[PlanNode],
    descriptors: &[DynamicResourceDescriptor],
    retained: &BTreeSet<ResourceId>,
) -> Result<Vec<StepResourceSlot>, String> {
    let descriptors = descriptors.iter().collect::<Vec<_>>();
    let by_id = nodes.iter().map(|node| (node.id.clone(), node)).collect();
    let expected = MemoryPlan::derive_pool_step_slots_with_order(
        nodes,
        &descriptors,
        retained,
        |predecessor, successor| node_completion_precedes(&by_id, predecessor, successor),
    )
    .map_err(|error| error.to_string());
    let actual = MemoryPlan::derive_pool_step_slots(nodes, &descriptors, retained)
        .map_err(|error| error.to_string());
    assert_eq!(actual, expected);
    if let (Ok(actual), Ok(expected)) = (&actual, &expected) {
        assert_eq!(
            serde_json::to_vec(actual).unwrap(),
            serde_json::to_vec(expected).unwrap()
        );
        assert_eq!(
            canonical_fingerprint(actual, "cached slots").unwrap(),
            canonical_fingerprint(expected, "legacy slots").unwrap(),
        );
    }
    actual
}

#[test]
fn cached_slots_match_legacy_for_fixed_and_token_demands_retention_and_external_aliases() {
    let mut nodes = vec![
        value_node(
            "node/a",
            &[],
            &[("value/input", "resource/input")],
            ("value/a", "resource/a"),
        ),
        value_node(
            "node/b",
            &["node/a"],
            &[("value/a", "resource/a")],
            ("value/b", "resource/b"),
        ),
        value_node(
            "node/c",
            &["node/b"],
            &[("value/b", "resource/b")],
            ("value/c", "resource/c"),
        ),
        value_node(
            "node/out",
            &["node/c"],
            &[("value/c", "resource/c"), ("value/late", "resource/late")],
            ("value/out", "resource/out"),
        ),
    ];
    for fixed in [false, true] {
        let descriptors = ["input", "late", "a", "b", "c", "out"]
            .map(|name| descriptor(&format!("resource/{name}"), fixed));
        let retained = BTreeSet::from([ResourceId::new("resource/out").unwrap()]);
        compare_slots(&nodes, &descriptors, &retained).unwrap();
        let mut retained_extra = retained.clone();
        retained_extra.insert(ResourceId::new("resource/a").unwrap());
        compare_slots(&nodes, &descriptors, &retained_extra).unwrap();
        let mut alias = nodes.clone();
        alias[0].values.push(token_binding(
            "value/external",
            ResolvedValueRole::Input,
            TensorAccess::Read,
            "resource/a",
        ));
        let slots = compare_slots(&alias, &descriptors, &retained).unwrap();
        if fixed {
            let slot = slots
                .iter()
                .find(|slot| {
                    slot.resource_ids()
                        .contains(&ResourceId::new("resource/a").unwrap())
                })
                .unwrap();
            assert_eq!(slot.kind(), StepResourceSlotKind::Dedicated);
        }
        let mut branch = nodes.clone();
        branch[2].dependencies.clear();
        compare_slots(&branch, &descriptors, &retained).unwrap();
        branch.swap(0, 2);
        compare_slots(&branch, &descriptors, &retained).unwrap();
    }
    let mut mixed = ["input", "late", "a", "b", "c", "out"]
        .map(|name| descriptor(&format!("resource/{name}"), true));
    mixed[3] = descriptor("resource/b", false);
    let projections = nodes[1]
        .values
        .iter()
        .map(|binding| NodeTokenBindingProjection {
            value_id: binding.value_id().clone(),
            role: binding.role(),
            ordinal: binding.ordinal(),
            axis: 0,
            rank: 2,
            canonical_extent: 8,
        })
        .collect::<Vec<_>>();
    nodes[1].work = NodeWorkContract::Tokens {
        source: projections[0].clone(),
        projections,
    };
    compare_slots(&nodes, &mixed, &BTreeSet::new()).unwrap();
}

#[test]
fn cached_slots_still_compare_every_member_when_an_interval_spans_unordered_users() {
    let nodes = vec![
        plan_node("node/a", &[], &["resource/a"]),
        plan_node("node/b-first", &["node/a"], &["resource/b"]),
        plan_node("node/b-last", &[], &["resource/b"]),
        plan_node("node/c", &["node/b-last"], &["resource/c"]),
    ];
    let descriptors = ["a", "b", "c"].map(|name| descriptor(&format!("resource/{name}"), false));
    let slots = compare_slots(&nodes, &descriptors, &BTreeSet::new()).unwrap();
    assert_eq!(slots.len(), 2);
    let c = slots
        .iter()
        .find(|slot| {
            slot.resource_ids()
                .contains(&ResourceId::new("resource/c").unwrap())
        })
        .unwrap();
    assert_eq!(c.kind(), StepResourceSlotKind::Dedicated);
}

#[test]
fn cached_slots_preserve_dependency_errors_and_unreferenced_descriptor_errors() {
    let mut nodes = vec![
        plan_node("node/a", &[], &["resource/a"]),
        plan_node("node/b", &["node/a", "node/missing"], &["resource/b"]),
    ];
    let descriptors = [
        descriptor("resource/a", false),
        descriptor("resource/b", false),
    ];
    assert!(compare_slots(&nodes, &descriptors, &BTreeSet::new()).is_err());
    nodes[1].dependencies.reverse();
    assert!(compare_slots(&nodes, &descriptors, &BTreeSet::new()).is_ok());
    let descriptors = [descriptor("resource/unreferenced", false)];
    assert!(compare_slots(&nodes, &descriptors, &BTreeSet::new()).is_err());
}
