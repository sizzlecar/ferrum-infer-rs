use super::*;

#[path = "alignment_coalescing_tests.rs"]
mod alignment_coalescing;

fn binding(name: &str, role: ResolvedValueRole, ordinal: u32, bytes: u64) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        ProgramValueId::new(format!("value/{name}")).unwrap(),
        role,
        ordinal,
        ResolvedTensorSpec::new(
            vec![bytes / 2],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        if role == ResolvedValueRole::Input {
            TensorAccess::Read
        } else {
            TensorAccess::Write
        },
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        ResolvedValueStorage::single(
            ResourceId::new(format!("resource/{name}")).unwrap(),
            0,
            bytes,
            ElementType::F16,
        )
        .unwrap(),
    )
    .unwrap()
}

/// The first two intermediates overlap. After both are consumed, the next
/// intermediate can reuse either slot; only reusing the large one avoids growth.
fn fixture() -> (Vec<PlanNode>, Vec<DynamicResourceDescriptor>) {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let descriptors = [("small", 64), ("large", 256), ("later", 256), ("out", 16)]
        .into_iter()
        .map(|(name, bytes)| {
            let mut d = step_descriptor(
                &format!("resource/{name}"),
                bytes,
                BufferUsage::Activations,
                storage.clone(),
            );
            d.demand = DynamicResourceDemand::fixed(bytes).unwrap();
            d
        })
        .collect();
    let mut nodes = vec![
        plan_node("node/small", &[], &["resource/small"]),
        plan_node(
            "node/large",
            &["node/small"],
            &["resource/small", "resource/large"],
        ),
        plan_node("node/consume", &["node/large"], &["resource/large"]),
        plan_node("node/later", &["node/consume"], &["resource/later"]),
        plan_node(
            "node/out",
            &["node/later"],
            &["resource/later", "resource/out"],
        ),
    ];
    nodes[0].values = vec![binding("small", ResolvedValueRole::Output, 0, 64)];
    nodes[1].values = vec![
        binding("small", ResolvedValueRole::Input, 0, 64),
        binding("large", ResolvedValueRole::Output, 0, 256),
    ];
    nodes[2].values = vec![binding("large", ResolvedValueRole::Input, 0, 256)];
    nodes[3].values = vec![binding("later", ResolvedValueRole::Output, 0, 256)];
    nodes[4].values = vec![
        binding("later", ResolvedValueRole::Input, 0, 256),
        binding("out", ResolvedValueRole::Output, 0, 16),
    ];
    (nodes, descriptors)
}

fn memory(
    nodes: &[PlanNode],
    descriptors: &[DynamicResourceDescriptor],
    capacity: u64,
    extra_retained: &[&str],
) -> Result<MemoryPlan, VNextError> {
    let retained = std::iter::once("out")
        .chain(extra_retained.iter().copied())
        .map(|name| ResourceId::new(format!("resource/{name}")).unwrap())
        .collect();
    MemoryPlan::from_core_with_completion_retention(
        capacity,
        capacity,
        0,
        1,
        vec![],
        descriptors.to_vec(),
        nodes,
        None,
        &retained,
    )
}

#[test]
fn compatible_larger_slot_avoids_growth_at_the_real_capacity_boundary() {
    let (nodes, descriptors) = fixture();
    // The terminal output can take the completed small buffer. Retention must
    // protect its later readback, without allocating it before its definition.
    let required = 64 + 256;
    let plan = memory(&nodes, &descriptors, required, &[]).unwrap();
    assert_eq!(plan.minimum_step_bytes(), required);
    assert_eq!(
        serde_json::from_slice::<MemoryPlan>(&serde_json::to_vec(&plan).unwrap()).unwrap(),
        plan
    );
    assert!(matches!(
        memory(&nodes, &descriptors, required - 1, &[]),
        Err(VNextError::PlanCapacityExceeded { required_bytes, usable_capacity_bytes })
            if required_bytes == required && usable_capacity_bytes == required - 1
    ));
}

#[test]
fn byte_savings_do_not_replace_completion_order_or_retained_output_proofs() {
    let (mut nodes, descriptors) = fixture();
    // Index order alone does not prove that either earlier allocation is dead.
    nodes[3].dependencies.clear();
    assert_eq!(
        memory(&nodes, &descriptors, 1024, &[])
            .unwrap()
            .minimum_step_bytes(),
        64 + 256 + 256 + 16
    );
    let (nodes, descriptors) = fixture();
    // The larger slot must stay reserved through completion; the small one
    // grows instead, even though retaining the large slot is more expensive.
    assert_eq!(
        memory(&nodes, &descriptors, 1024, &["large"])
            .unwrap()
            .minimum_step_bytes(),
        256 + 256 + 16
    );
    // A later retained diagnostic can take an earlier completed slot. Both
    // it and the terminal output must then survive through the same fence.
    let plan = memory(&nodes, &descriptors, 320, &["later"]).unwrap();
    assert_eq!(plan.minimum_step_bytes(), 320);
    let retained =
        ["later", "out"].map(|name| ResourceId::new(format!("resource/{name}")).unwrap());
    for slot in plan
        .dynamic_pools()
        .iter()
        .flat_map(|p| p.step_resource_slots())
    {
        assert!(
            retained
                .iter()
                .filter(|id| slot.resource_ids().contains(id))
                .count()
                <= 1
        );
    }
    // Retaining the earlier large value too forbids its reuse by `later`.
    assert_eq!(
        memory(&nodes, &descriptors, 1024, &["large", "later"])
            .unwrap()
            .minimum_step_bytes(),
        256 + 256 + 16
    );
}

#[test]
fn a_local_saving_cannot_make_the_complete_fixed_domain_more_expensive() {
    let (mut nodes, mut descriptors) = fixture();
    descriptors[2].demand = DynamicResourceDemand::fixed(128).unwrap();
    let mut overlapping = descriptors[1].clone();
    overlapping.base_resource_id = ResourceId::new("resource/overlapping").unwrap();
    descriptors.push(overlapping);
    nodes[3].values = vec![binding("later", ResolvedValueRole::Output, 0, 128)];
    let mut next = plan_node(
        "node/overlapping",
        &["node/later"],
        &["resource/later", "resource/overlapping"],
    );
    next.values = vec![
        binding("later", ResolvedValueRole::Input, 0, 128),
        binding("overlapping", ResolvedValueRole::Output, 0, 256),
    ];
    nodes.insert(4, next);
    nodes[5].dependencies = vec![NodeId::new("node/overlapping").unwrap()];
    nodes[5].resources = ["resource/later", "resource/overlapping", "resource/out"]
        .map(|name| ResourceId::new(name).unwrap())
        .to_vec();
    nodes[5].values = vec![
        binding("later", ResolvedValueRole::Input, 0, 128),
        binding("overlapping", ResolvedValueRole::Input, 1, 256),
        binding("out", ResolvedValueRole::Output, 0, 16),
    ];
    // Greedily putting `later` in the already-large slot saves 64 bytes now,
    // but makes the overlapping 256-byte value grow the other slot by 192.
    // Keeping the original placement fits both plus the retained output.
    let required = 128 + 256 + 16;
    assert_eq!(
        memory(&nodes, &descriptors, required, &[])
            .unwrap()
            .minimum_step_bytes(),
        required
    );
}

#[test]
fn disjoint_large_values_share_storage_even_when_small_values_need_an_extra_slot() {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let values = [("a", 256), ("b", 64), ("c", 64), ("d", 256), ("out", 16)];
    let mut nodes = Vec::new();
    let mut descriptors = Vec::new();
    for (index, &(name, bytes)) in values.iter().enumerate() {
        let mut descriptor = step_descriptor(
            &format!("resource/{name}"),
            bytes,
            BufferUsage::Activations,
            storage.clone(),
        );
        descriptor.demand = DynamicResourceDemand::fixed(bytes).unwrap();
        descriptors.push(descriptor);
        let mut node = plan_node(&format!("node/{name}"), &[], &[]);
        if index > 0 {
            let (previous, previous_bytes) = values[index - 1];
            node.dependencies = vec![NodeId::new(format!("node/{previous}")).unwrap()];
            node.values.push(binding(
                previous,
                ResolvedValueRole::Input,
                0,
                previous_bytes,
            ));
            node.resources
                .push(ResourceId::new(format!("resource/{previous}")).unwrap());
        }
        node.values
            .push(binding(name, ResolvedValueRole::Output, 0, bytes));
        node.resources
            .push(ResourceId::new(format!("resource/{name}")).unwrap());
        nodes.push(node);
    }
    // The chain has intervals a=[0,1], b=[1,2], c=[2,3], d=[3,4].
    // Either chronological policy uses two 256-byte slots. Sharing a with d
    // and keeping b and c separate instead uses 256 + 64 + 64 bytes.
    // The retained output takes an earlier completed 64-byte buffer.
    let required = 256 + 64 + 64;
    assert_eq!(
        memory(&nodes, &descriptors, required, &[])
            .unwrap()
            .minimum_step_bytes(),
        required
    );
    assert!(matches!(
        memory(&nodes, &descriptors, required - 1, &[]),
        Err(VNextError::PlanCapacityExceeded { required_bytes, .. })
            if required_bytes == required
    ));
}

#[test]
fn offline_slots_preserve_all_pairwise_completion_proofs_on_branching_dags() {
    let values = [("a", 256), ("b", 64), ("c", 128), ("d", 256), ("out", 16)];
    let edges = (0..values.len())
        .flat_map(|to| (0..to).map(move |from| (from, to)))
        .collect::<Vec<_>>();
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let descriptors = values
        .iter()
        .map(|&(name, bytes)| {
            let mut d = step_descriptor(
                &format!("resource/{name}"),
                bytes,
                BufferUsage::Activations,
                storage.clone(),
            );
            d.demand = DynamicResourceDemand::fixed(bytes).unwrap();
            d
        })
        .collect::<Vec<_>>();
    // Enumerate every forward-edge DAG on these values, including disconnected
    // branches. Every edge is a real input use with its producer dependency.
    for edge_mask in 0..(1_u16 << edges.len()) {
        let mut reachable = [[false; 5]; 5];
        let mut last_user = [0, 1, 2, 3, 4];
        let mut nodes = Vec::new();
        for (to, &(name, bytes)) in values.iter().enumerate() {
            let mut node = plan_node(&format!("node/{name}"), &[], &[]);
            for (bit, &(from, target)) in edges.iter().enumerate() {
                if target == to && edge_mask & (1 << bit) != 0 {
                    let (input, input_bytes) = values[from];
                    node.dependencies
                        .push(NodeId::new(format!("node/{input}")).unwrap());
                    node.resources
                        .push(descriptors[from].base_resource_id.clone());
                    node.values.push(binding(
                        input,
                        ResolvedValueRole::Input,
                        node.values.len() as u32,
                        input_bytes,
                    ));
                    reachable[from][to] = true;
                    last_user[from] = to;
                }
            }
            node.resources
                .push(descriptors[to].base_resource_id.clone());
            node.values
                .push(binding(name, ResolvedValueRole::Output, 0, bytes));
            nodes.push(node);
        }
        // Independent transitive closure checks every pair in the final slots,
        // independently of the placement algorithm and its reachability cache.
        for via in 0..values.len() {
            for from in 0..values.len() {
                for to in 0..values.len() {
                    reachable[from][to] |= reachable[from][via] && reachable[via][to];
                }
            }
        }
        let plan = memory(&nodes, &descriptors, 4096, &[]).unwrap();
        for slot in plan
            .dynamic_pools()
            .iter()
            .flat_map(|pool| pool.step_resource_slots())
        {
            let members = slot
                .resource_ids()
                .iter()
                .map(|id| {
                    descriptors
                        .iter()
                        .position(|d| &d.base_resource_id == id)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            for (index, &a) in members.iter().enumerate() {
                for &b in &members[index + 1..] {
                    assert!(
                        (a != 4 && last_user[a] < b && reachable[last_user[a]][b])
                            || (b != 4 && last_user[b] < a && reachable[last_user[b]][a]),
                        "unordered live values share storage: graph={edge_mask}, a={a}, b={b}"
                    );
                }
            }
        }
    }
}

/// CPU-only compiler workload: a dependency chain with skip connections and
/// varying activation sizes. It exercises the shared planner, not a model or
/// device-specific shortcut. Timings are observations, never test thresholds.
#[test]
#[ignore = "CPU profiling workload; run explicitly with --nocapture"]
fn profile_step_slot_placement_on_layered_graph() {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"contiguous_v1", "test layout").unwrap(),
    )
    .unwrap();
    let sizes = (0..4096)
        .map(|i| [16384, 65536, 262144, 4194304][i % 4] * (1 + i as u64 % 5))
        .collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(sizes.len());
    let mut descriptors = Vec::with_capacity(sizes.len());
    for (i, &bytes) in sizes.iter().enumerate() {
        let name = format!("activation/{i:05}");
        let mut descriptor = step_descriptor(
            &format!("resource/{name}"),
            bytes,
            BufferUsage::Activations,
            storage.clone(),
        );
        descriptor.demand = DynamicResourceDemand::fixed(bytes).unwrap();
        let mut node = plan_node(&format!("node/{i:05}"), &[], &[]);
        for predecessor in [i.checked_sub(1), i.checked_sub(8)].into_iter().flatten() {
            let input = format!("activation/{predecessor:05}");
            node.dependencies
                .push(NodeId::new(format!("node/{predecessor:05}")).unwrap());
            node.resources
                .push(ResourceId::new(format!("resource/{input}")).unwrap());
            node.values.push(binding(
                &input,
                ResolvedValueRole::Input,
                node.values.len() as u32,
                sizes[predecessor],
            ));
        }
        node.resources.push(descriptor.base_resource_id.clone());
        node.values
            .push(binding(&name, ResolvedValueRole::Output, 0, bytes));
        nodes.push(node);
        descriptors.push(descriptor);
    }
    let references = descriptors.iter().collect::<Vec<_>>();
    let retained = [descriptors.last().unwrap().base_resource_id.clone()]
        .into_iter()
        .collect();
    let started = std::time::Instant::now();
    let slots =
        MemoryPlan::derive_pool_step_slots(std::hint::black_box(&nodes), &references, &retained)
            .unwrap();
    let elapsed = started.elapsed();
    let bytes = descriptors
        .iter()
        .map(|d| (&d.base_resource_id, d.minimum_request_bytes().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let slot_bytes = slots
        .iter()
        .map(|slot| {
            slot.resource_ids()
                .iter()
                .map(|id| bytes[id])
                .max()
                .unwrap()
        })
        .sum::<u64>();
    println!(
        "step_slot_profile nodes={} elapsed_seconds={:.6} slot_bytes={} slot_fingerprint={}",
        nodes.len(),
        elapsed.as_secs_f64(),
        slot_bytes,
        canonical_fingerprint(&slots, "step slots").unwrap(),
    );
}
