use super::{
    invalid_resource, AllocationKind, AllocationLifetime, BTreeMap, BufferUsage,
    DynamicPoolDomainSpec, DynamicStorageView, ElementType, EvaluatedBackingRequest,
    ExecutionLaneId, InvocationLivenessMode, NodeId, PhysicalBackingClaimIdentity, PlanNode,
    ResourceId, Serialize, SubmissionWaveDomainCapacityLayout, SubmissionWaveDomainLayout,
    VNextError,
};
use crate::vnext::ReusableExecutionBucketId;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramBindingSlot {
    node_index: usize,
    node_id: NodeId,
    resource_id: ResourceId,
    physical_offset_bytes: u64,
    capacity_size_bytes: u64,
    alignment_bytes: u64,
}

impl ProgramBindingSlot {
    pub const fn node_index(&self) -> usize {
        self.node_index
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn physical_offset_bytes(&self) -> u64 {
        self.physical_offset_bytes
    }

    pub const fn capacity_size_bytes(&self) -> u64 {
        self.capacity_size_bytes
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
}

/// Cold-compiled binding arena layout for one immutable reusable-execution
/// bucket. Every slot is a fixed, non-overlapping projection into one
/// contiguous lane-stable physical claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramBindingLayout {
    reusable_execution_bucket_id: ReusableExecutionBucketId,
    claim_identity: PhysicalBackingClaimIdentity,
    physical_size_bytes: u64,
    slots: Vec<ProgramBindingSlot>,
    fingerprint: String,
}

impl ProgramBindingLayout {
    pub fn reusable_execution_bucket_id(&self) -> &ReusableExecutionBucketId {
        &self.reusable_execution_bucket_id
    }

    pub fn claim_identity(&self) -> &PhysicalBackingClaimIdentity {
        &self.claim_identity
    }

    pub const fn physical_size_bytes(&self) -> u64 {
        self.physical_size_bytes
    }

    pub fn slots(&self) -> &[ProgramBindingSlot] {
        &self.slots
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn slot_for_node(&self, node_index: usize) -> Option<&ProgramBindingSlot> {
        self.slots
            .binary_search_by_key(&node_index, ProgramBindingSlot::node_index)
            .ok()
            .and_then(|index| self.slots.get(index))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaneStableArenaSlotIdentity {
    lane_id: ExecutionLaneId,
    lifetime: AllocationLifetime,
    reusable_execution_bucket_id: ReusableExecutionBucketId,
    layout_fingerprint: String,
    slot_id: u64,
}

impl LaneStableArenaSlotIdentity {
    pub(super) fn new(
        lane_id: ExecutionLaneId,
        lifetime: AllocationLifetime,
        reusable_execution_bucket_id: ReusableExecutionBucketId,
        layout_fingerprint: String,
        slot_id: u64,
    ) -> Self {
        Self {
            lane_id,
            lifetime,
            reusable_execution_bucket_id,
            layout_fingerprint,
            slot_id,
        }
    }

    pub const fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn reusable_execution_bucket_id(&self) -> &ReusableExecutionBucketId {
        &self.reusable_execution_bucket_id
    }

    pub fn layout_fingerprint(&self) -> &str {
        &self.layout_fingerprint
    }

    pub const fn slot_id(&self) -> u64 {
        self.slot_id
    }
}

pub(super) fn compile_program_binding_layouts(
    domains: &[DynamicPoolDomainSpec],
    nodes: &[PlanNode],
    layouts: &[Option<SubmissionWaveDomainLayout>],
    reusable_capacity_layouts: &BTreeMap<
        ReusableExecutionBucketId,
        Vec<Option<SubmissionWaveDomainCapacityLayout>>,
    >,
) -> Result<BTreeMap<ReusableExecutionBucketId, ProgramBindingLayout>, VNextError> {
    if domains.len() != layouts.len()
        || reusable_capacity_layouts
            .values()
            .any(|capacities| capacities.len() != domains.len())
    {
        return Err(invalid_resource(
            "program binding compiler received inconsistent domain layouts",
        ));
    }

    let binding_domains = domains
        .iter()
        .enumerate()
        .filter(|(_, domain)| {
            domain
                .descriptors
                .iter()
                .any(|descriptor| matches!(descriptor.kind(), AllocationKind::Binding { .. }))
        })
        .collect::<Vec<_>>();
    if binding_domains.is_empty() {
        return Ok(BTreeMap::new());
    }
    let [(domain_index, domain)] = binding_domains.as_slice() else {
        return Err(invalid_resource(
            "one compiled program cannot span multiple binding domains",
        ));
    };
    if domain.pool.compatibility().usage() != BufferUsage::Binding
        || domain.pool.compatibility().element_type() != ElementType::U8
        || domain.pool.compatibility().profile().view() != DynamicStorageView::Contiguous
        || domain.pool.invocation_liveness_mode() != InvocationLivenessMode::ConservativeConcurrent
        || domain.descriptors.iter().any(|descriptor| {
            descriptor.lifetime() != AllocationLifetime::Invocation
                || descriptor.usage() != BufferUsage::Binding
                || descriptor.element_type() != ElementType::U8
                || !matches!(descriptor.kind(), AllocationKind::Binding { .. })
        })
    {
        return Err(invalid_resource(
            "program bindings require one contiguous conservative U8 binding domain",
        ));
    }
    let base_layout = layouts
        .get(*domain_index)
        .and_then(Option::as_ref)
        .ok_or_else(|| invalid_resource("program binding domain has no submission-wave layout"))?;
    if base_layout.projection_count != domain.descriptors.len()
        || base_layout.claim_identity.resource_ids().len() != domain.descriptors.len()
    {
        return Err(invalid_resource(
            "program binding domain layout does not cover every binding descriptor",
        ));
    }

    reusable_capacity_layouts
        .iter()
        .map(|(bucket_id, capacity_layouts)| {
            let capacity = capacity_layouts
                .get(*domain_index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    invalid_resource(
                        "reusable program binding bucket has no physical capacity layout",
                    )
                })?;
            if capacity.projections.len() != domain.descriptors.len()
                || capacity.physical_size_bytes == 0
            {
                return Err(invalid_resource(
                    "reusable program binding capacity layout is incomplete",
                ));
            }

            let mut slots = domain
                .descriptors
                .iter()
                .enumerate()
                .map(|(projection_index, descriptor)| {
                    let AllocationKind::Binding { node_id } = descriptor.kind() else {
                        unreachable!("binding domain was validated above")
                    };
                    let node_index = nodes
                        .iter()
                        .position(|node| node.id() == node_id)
                        .ok_or_else(|| {
                            invalid_resource(
                                "program binding descriptor references a missing plan node",
                            )
                        })?;
                    let node = &nodes[node_index];
                    if node.binding_resource() != Some(descriptor.base_resource_id())
                        || !node.resources().contains(descriptor.base_resource_id())
                    {
                        return Err(invalid_resource(
                            "program binding node does not own its binding descriptor",
                        ));
                    }
                    let projection = &capacity.projections[projection_index];
                    Ok(ProgramBindingSlot {
                        node_index,
                        node_id: node_id.clone(),
                        resource_id: descriptor.base_resource_id().clone(),
                        physical_offset_bytes: projection.physical_offset_bytes,
                        capacity_size_bytes: projection.capacity_size_bytes,
                        alignment_bytes: descriptor.alignment_bytes(),
                    })
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            slots.sort_by_key(ProgramBindingSlot::physical_offset_bytes);
            let contiguous_end = slots.iter().try_fold(0_u64, |expected_offset, slot| {
                if slot.physical_offset_bytes != expected_offset
                    || slot.capacity_size_bytes == 0
                    || slot.physical_offset_bytes % slot.alignment_bytes != 0
                    || slot.capacity_size_bytes % slot.alignment_bytes != 0
                {
                    return Err(invalid_resource(
                        "program binding slots are empty, misaligned, or non-contiguous",
                    ));
                }
                expected_offset
                    .checked_add(slot.capacity_size_bytes)
                    .ok_or_else(|| invalid_resource("program binding arena size overflows u64"))
            })?;
            if contiguous_end != capacity.physical_size_bytes {
                return Err(invalid_resource(
                    "program binding slots do not cover their physical arena exactly",
                ));
            }
            slots.sort_by_key(ProgramBindingSlot::node_index);
            if slots
                .windows(2)
                .any(|pair| pair[0].node_index >= pair[1].node_index)
            {
                return Err(invalid_resource(
                    "program binding slots do not have unique plan-node owners",
                ));
            }

            #[derive(Serialize)]
            struct FingerprintMaterial<'a> {
                domain: &'static str,
                reusable_execution_bucket_id: &'a ReusableExecutionBucketId,
                claim_identity: &'a PhysicalBackingClaimIdentity,
                physical_size_bytes: u64,
                slots: &'a [ProgramBindingSlot],
            }
            let bytes = serde_json::to_vec(&FingerprintMaterial {
                domain: "ferrum.runtime-vnext.program-binding-layout.v1",
                reusable_execution_bucket_id: bucket_id,
                claim_identity: &base_layout.claim_identity,
                physical_size_bytes: capacity.physical_size_bytes,
                slots: &slots,
            })
            .map_err(|error| {
                invalid_resource(format!(
                    "program binding layout fingerprint encode failed: {error}"
                ))
            })?;
            let layout = ProgramBindingLayout {
                reusable_execution_bucket_id: bucket_id.clone(),
                claim_identity: base_layout.claim_identity.clone(),
                physical_size_bytes: capacity.physical_size_bytes,
                slots,
                fingerprint: format!("sha256/{:x}", Sha256::digest(bytes)),
            };
            Ok((bucket_id.clone(), layout))
        })
        .collect()
}

pub(super) fn lane_stable_layout_fingerprint(
    lifetime: AllocationLifetime,
    bucket_id: &ReusableExecutionBucketId,
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<String, VNextError> {
    #[derive(Serialize)]
    struct Projection<'a> {
        resource_id: &'a ResourceId,
        physical_offset_bytes: u64,
        capacity_size_bytes: u64,
    }

    #[derive(Serialize)]
    struct Request<'a> {
        claim_identity: &'a PhysicalBackingClaimIdentity,
        capacity_size_bytes: u64,
        projections: Vec<Projection<'a>>,
    }

    #[derive(Serialize)]
    struct Material<'a> {
        domain: &'static str,
        lifetime: AllocationLifetime,
        reusable_execution_bucket_id: &'a ReusableExecutionBucketId,
        requests: Vec<Request<'a>>,
    }

    let material = Material {
        domain: "ferrum.runtime-vnext.lane-stable-layout.v1",
        lifetime,
        reusable_execution_bucket_id: bucket_id,
        requests: requests
            .iter()
            .map(|request| Request {
                claim_identity: &request.claim_identity,
                capacity_size_bytes: request.capacity_size_bytes,
                projections: request
                    .projections
                    .iter()
                    .map(|projection| Projection {
                        resource_id: projection.descriptor.base_resource_id(),
                        physical_offset_bytes: projection.physical_offset_bytes,
                        capacity_size_bytes: projection.capacity_size_bytes,
                    })
                    .collect(),
            })
            .collect(),
    };
    serde_json::to_vec(&material)
        .map(|bytes| format!("sha256/{:x}", Sha256::digest(bytes)))
        .map_err(|error| {
            invalid_resource(format!(
                "lane-stable layout fingerprint encode failed: {error}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{
        DynamicResourceDemand, DynamicResourceDescriptor, DynamicStorageAllocator,
        DynamicStorageContract, DynamicStorageProfile, MemoryPlan, ResolvedReusableExecutionBucket,
        ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
        ReusableExecutionMemoryPlan, ReusablePoolWorkspaceBudget,
    };

    fn binding_descriptor(
        resource_id: &str,
        node_id: &str,
        bytes: u64,
        storage: DynamicStorageContract,
    ) -> DynamicResourceDescriptor {
        DynamicResourceDescriptor::resource_test_binding(
            ResourceId::new(resource_id).expect("valid resource id"),
            DynamicResourceDemand::fixed(bytes).expect("valid fixed binding demand"),
            16,
            NodeId::new(node_id).expect("valid node id"),
            storage,
            8,
        )
        .expect("valid binding descriptor")
    }

    fn compile_two_node_layout() -> (
        Vec<PlanNode>,
        Vec<DynamicPoolDomainSpec>,
        Vec<Option<SubmissionWaveDomainLayout>>,
        BTreeMap<ReusableExecutionBucketId, Vec<Option<SubmissionWaveDomainCapacityLayout>>>,
        ReusableExecutionBucketId,
    ) {
        let storage = DynamicStorageContract::resource_test_contract(
            DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .expect("valid binding storage profile"),
            "a".repeat(64),
        )
        .expect("valid binding storage");
        let descriptors = vec![
            binding_descriptor(
                "resource/program-binding-first",
                "node/program-binding-first",
                64,
                storage.clone(),
            ),
            binding_descriptor(
                "resource/program-binding-second",
                "node/program-binding-second",
                128,
                storage,
            ),
        ];
        let nodes = vec![
            PlanNode::resource_test_node_with_binding(
                NodeId::new("node/program-binding-first").unwrap(),
                descriptors[0].base_resource_id().clone(),
            ),
            PlanNode::resource_test_node_with_binding(
                NodeId::new("node/program-binding-second").unwrap(),
                descriptors[1].base_resource_id().clone(),
            ),
        ];
        let pools =
            MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 1 << 20).expect("derive pools");
        let (_, domains) =
            super::super::plan_dynamic_pool_admission(1, &pools, &descriptors).unwrap();
        let layouts = domains
            .iter()
            .map(|domain| {
                super::super::dynamic_pool::compile_submission_wave_domain_layout(domain, &nodes)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("test.program-binding").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let bucket_id = bucket.bucket_id().clone();
        let reusable = ReusableExecutionMemoryPlan::new(
            1,
            1,
            vec![ResolvedReusableExecutionBucket::new(
                bucket,
                vec![
                    ReusablePoolWorkspaceBudget::new(descriptors[0].pool_id().clone(), 0, 192)
                        .unwrap(),
                ],
            )
            .unwrap()],
        )
        .unwrap();
        let capacity_layouts =
            super::super::dynamic_pool::compile_submission_wave_reusable_capacity_layouts(
                &domains,
                &layouts,
                Some(&reusable),
            )
            .unwrap();
        (nodes, domains, layouts, capacity_layouts, bucket_id)
    }

    #[test]
    fn binding_layout_cold_compiles_one_exact_stable_arena() {
        let (nodes, domains, layouts, capacity_layouts, bucket_id) = compile_two_node_layout();
        let compiled =
            compile_program_binding_layouts(&domains, &nodes, &layouts, &capacity_layouts)
                .expect("compile binding layout");
        let layout = compiled.get(&bucket_id).expect("compiled bucket layout");

        assert_eq!(layout.physical_size_bytes(), 192);
        assert_eq!(layout.slots().len(), 2);
        assert_eq!(layout.slot_for_node(0).unwrap().physical_offset_bytes(), 0);
        assert_eq!(layout.slot_for_node(0).unwrap().capacity_size_bytes(), 64);
        assert_eq!(layout.slot_for_node(1).unwrap().physical_offset_bytes(), 64);
        assert_eq!(layout.slot_for_node(1).unwrap().capacity_size_bytes(), 128);
        assert!(layout.fingerprint().starts_with("sha256/"));
        assert_eq!(
            compiled,
            compile_program_binding_layouts(&domains, &nodes, &layouts, &capacity_layouts)
                .expect("recompile stable binding layout")
        );
    }

    #[test]
    fn binding_layout_rejects_a_node_that_does_not_own_its_binding() {
        let (_, domains, layouts, capacity_layouts, _) = compile_two_node_layout();
        let nodes = vec![
            PlanNode::resource_test_node(NodeId::new("node/program-binding-first").unwrap()),
            PlanNode::resource_test_node(NodeId::new("node/program-binding-second").unwrap()),
        ];

        let error = compile_program_binding_layouts(&domains, &nodes, &layouts, &capacity_layouts)
            .expect_err("binding layout must reject missing node ownership");
        assert!(error
            .to_string()
            .contains("does not own its binding descriptor"));
    }
}
