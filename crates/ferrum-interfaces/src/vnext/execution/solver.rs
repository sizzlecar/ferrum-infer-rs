use super::{
    invalid_plan, AllocationLifetime, BTreeMap, BTreeSet, BufferUsage, DynamicBackingPoolSpec,
    DynamicResourceDemand, DynamicResourceDescriptor, DynamicStorageProfile, ElementType,
    InvocationLivenessMode, InvocationResourceLiveness, NodeId, PlanNode, ProgramValueId,
    ProviderId, ProviderResourcePlan, RejectedProvider, ResolvedTensorSpec, ResolvedValueStorage,
    ResourceId, StateId, StateInitialization, VNextError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CanonicalValueBinding {
    pub(super) tensor: ResolvedTensorSpec,
    pub(super) usage: BufferUsage,
    pub(super) storage: ResolvedValueStorage,
}

pub(super) struct GlobalValueRange {
    pub(super) value_id: ProgramValueId,
    pub(super) offset_bytes: u64,
    pub(super) end_bytes: u64,
}

#[derive(Default)]
pub(super) struct StateDependencyTracker {
    pub(super) last_writer: BTreeMap<StateId, NodeId>,
    pub(super) readers_since_write: BTreeMap<StateId, BTreeSet<NodeId>>,
}

#[derive(Clone)]
pub(super) struct JointProviderCandidate {
    pub(super) resources: ProviderResourcePlan,
    pub(super) allowed_profiles: BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
    pub(super) is_preferred: bool,
}

pub(super) struct JointProviderStorageSelection {
    pub(super) node_resources: BTreeMap<NodeId, ProviderResourcePlan>,
    pub(super) resource_profiles: BTreeMap<ResourceId, DynamicStorageProfile>,
    pub(super) storage_rejections: BTreeMap<NodeId, RejectedProvider>,
}

#[derive(Clone, Default)]
pub(super) struct JointPartialSelection {
    pub(super) chosen: Vec<ProviderResourcePlan>,
    pub(super) preferred: Vec<bool>,
}

pub(super) struct JointComponentSolution {
    pub(super) chosen: Vec<ProviderResourcePlan>,
    pub(super) resource_profiles: BTreeMap<ResourceId, DynamicStorageProfile>,
}

pub(super) struct JointSelectionObjective {
    pub(super) preferred: Vec<bool>,
    /// Counts for the least-preferred profile through profile rank 1. Rank 0
    /// has zero penalty, so candidate-specific extra rank-0 resources cannot
    /// bias selection merely by increasing the resource count.
    pub(super) profile_penalty: Vec<usize>,
    pub(super) provider_ids: Vec<ProviderId>,
}

impl JointSelectionObjective {
    pub(super) fn new(
        partial: &JointPartialSelection,
        profile_ranks: impl Iterator<Item = usize>,
        profile_count: usize,
    ) -> Result<Self, VNextError> {
        let mut profile_penalty = vec![0_usize; profile_count.saturating_sub(1)];
        for rank in profile_ranks {
            if rank >= profile_count {
                return Err(invalid_plan(
                    "joint storage objective contains an unknown profile rank",
                ));
            }
            if rank != 0 {
                let penalty_index = profile_count - 1 - rank;
                profile_penalty[penalty_index] = profile_penalty[penalty_index]
                    .checked_add(1)
                    .ok_or_else(|| invalid_plan("joint storage profile penalty overflows usize"))?;
            }
        }
        Ok(Self {
            preferred: partial.preferred.clone(),
            profile_penalty,
            provider_ids: partial
                .chosen
                .iter()
                .map(|resources| resources.provider_id().clone())
                .collect(),
        })
    }

    pub(super) fn precedes(&self, other: &Self) -> bool {
        match compare_preferred(&self.preferred, &other.preferred) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
        match self.profile_penalty.cmp(&other.profile_penalty) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => self.provider_ids < other.provider_ids,
        }
    }
}

pub(super) fn compare_preferred(left: &[bool], right: &[bool]) -> std::cmp::Ordering {
    for (left, right) in left.iter().zip(right) {
        match (*right as u8).cmp(&(*left as u8)) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    left.len().cmp(&right.len())
}

pub(super) fn joint_partial_precedes(
    left: &JointPartialSelection,
    right: &JointPartialSelection,
) -> bool {
    match compare_preferred(&left.preferred, &right.preferred) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => left
            .chosen
            .iter()
            .map(ProviderResourcePlan::provider_id)
            .cmp(right.chosen.iter().map(ProviderResourcePlan::provider_id))
            .is_lt(),
    }
}

pub(super) fn joint_candidate_components(
    candidate_sets: &[Vec<JointProviderCandidate>],
) -> Vec<Vec<usize>> {
    let mut parents = (0..candidate_sets.len()).collect::<Vec<_>>();
    let mut ranks = vec![0_u8; candidate_sets.len()];
    let mut resource_owner = BTreeMap::<ResourceId, usize>::new();
    for (node_index, candidates) in candidate_sets.iter().enumerate() {
        let resource_ids = candidates
            .iter()
            .flat_map(|candidate| candidate.allowed_profiles.keys().cloned())
            .collect::<BTreeSet<_>>();
        for resource_id in resource_ids {
            if let Some(owner) = resource_owner.insert(resource_id, node_index) {
                joint_union(&mut parents, &mut ranks, owner, node_index);
            }
        }
    }
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for node_index in 0..candidate_sets.len() {
        let root = joint_find(&mut parents, node_index);
        components.entry(root).or_default().push(node_index);
    }
    let mut components = components.into_values().collect::<Vec<_>>();
    components.sort_by_key(|component| component[0]);
    components
}

pub(super) fn storage_incompatible_resource_ids(
    candidate: &JointProviderCandidate,
    selected_profiles: &BTreeMap<ResourceId, DynamicStorageProfile>,
) -> Vec<ResourceId> {
    candidate
        .allowed_profiles
        .iter()
        .filter(|(resource_id, accepted)| {
            selected_profiles
                .get(*resource_id)
                .is_some_and(|profile| !accepted.contains(profile))
        })
        .map(|(resource_id, _)| resource_id.clone())
        .collect()
}

pub(super) fn joint_find(parents: &mut [usize], value: usize) -> usize {
    let mut root = value;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = value;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

pub(super) fn joint_union(parents: &mut [usize], ranks: &mut [u8], left: usize, right: usize) {
    let left_root = joint_find(parents, left);
    let right_root = joint_find(parents, right);
    if left_root == right_root {
        return;
    }
    match ranks[left_root].cmp(&ranks[right_root]) {
        std::cmp::Ordering::Less => parents[left_root] = right_root,
        std::cmp::Ordering::Greater => parents[right_root] = left_root,
        std::cmp::Ordering::Equal => {
            parents[right_root] = left_root;
            ranks[left_root] = ranks[left_root].saturating_add(1);
        }
    }
}

#[derive(Default)]
pub(super) struct PoolAggregateEvidence {
    pub(super) minimum_request_bytes: u64,
    pub(super) minimum_sequence_bytes: u64,
    pub(super) minimum_step_bytes: u64,
    pub(super) minimum_invocation_peak_bytes: u64,
    pub(super) theoretical_ceiling_bytes: u128,
}

impl PoolAggregateEvidence {
    pub(super) fn add(&mut self, pool: &DynamicBackingPoolSpec) -> Result<(), VNextError> {
        self.minimum_request_bytes = self
            .minimum_request_bytes
            .checked_add(pool.minimum_request_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool request minimum overflows u64"))?;
        self.minimum_sequence_bytes = self
            .minimum_sequence_bytes
            .checked_add(pool.minimum_sequence_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool sequence minimum overflows u64"))?;
        self.minimum_step_bytes = self
            .minimum_step_bytes
            .checked_add(pool.minimum_step_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool step minimum overflows u64"))?;
        self.minimum_invocation_peak_bytes = self
            .minimum_invocation_peak_bytes
            .checked_add(pool.minimum_invocation_peak_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool invocation minimum overflows u64"))?;
        self.theoretical_ceiling_bytes = self
            .theoretical_ceiling_bytes
            .checked_add(pool.theoretical_ceiling_bytes.get())
            .ok_or_else(|| invalid_plan("aggregate pool theoretical ceiling overflows u128"))?;
        Ok(())
    }
}

pub(super) fn minimum_for_lifetime(
    descriptors: &[&DynamicResourceDescriptor],
    lifetime: AllocationLifetime,
    overflow_context: &'static str,
) -> Result<u64, VNextError> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.lifetime == lifetime)
        .try_fold(0_u64, |total, descriptor| {
            total
                .checked_add(descriptor.minimum_request_bytes()?)
                .ok_or_else(|| invalid_plan(format!("{overflow_context} overflows u64")))
        })
}

pub(super) fn node_completion_precedes(
    nodes_by_id: &BTreeMap<NodeId, &PlanNode>,
    predecessor: &NodeId,
    successor: &NodeId,
) -> Result<bool, VNextError> {
    if predecessor == successor {
        return Ok(false);
    }
    let successor = nodes_by_id
        .get(successor)
        .ok_or_else(|| invalid_plan("completion-order successor node is missing"))?;
    let mut pending = successor.dependencies.clone();
    let mut visited = BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if &node_id == predecessor {
            return Ok(true);
        }
        if !visited.insert(node_id.clone()) {
            continue;
        }
        let node = nodes_by_id
            .get(&node_id)
            .ok_or_else(|| invalid_plan("completion-order dependency node is missing"))?;
        pending.extend(node.dependencies.iter().cloned());
    }
    Ok(false)
}

pub(super) fn validate_pool_liveness_rows(
    mode: InvocationLivenessMode,
    rows: &[InvocationResourceLiveness],
    invocation_ids: &BTreeSet<ResourceId>,
    descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
) -> Result<u64, VNextError> {
    if invocation_ids.is_empty() {
        if mode != InvocationLivenessMode::NoInvocationResources || !rows.is_empty() {
            return Err(invalid_plan(
                "pool without invocation resources has liveness evidence",
            ));
        }
        return Ok(0);
    }
    if mode == InvocationLivenessMode::NoInvocationResources || rows.is_empty() {
        return Err(invalid_plan(
            "invocation pool is missing typed liveness evidence",
        ));
    }
    let mut covered = BTreeSet::new();
    let mut maximum_row = 0_u64;
    let mut concurrent_sum = 0_u64;
    for row in rows {
        if row.resource_ids.is_empty() || row.resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_plan(
                "pool invocation liveness row is empty or non-canonical",
            ));
        }
        let row_bytes = row
            .resource_ids
            .iter()
            .try_fold(0_u64, |total, resource_id| {
                if !invocation_ids.contains(resource_id) {
                    return Err(invalid_plan(
                        "pool liveness references a non-member invocation resource",
                    ));
                }
                covered.insert(resource_id.clone());
                total
                    .checked_add(
                        descriptors
                            .get(resource_id)
                            .ok_or_else(|| invalid_plan("pool liveness descriptor is missing"))?
                            .minimum_request_bytes()?,
                    )
                    .ok_or_else(|| invalid_plan("pool invocation row bytes overflow u64"))
            })?;
        maximum_row = maximum_row.max(row_bytes);
        concurrent_sum = concurrent_sum
            .checked_add(row_bytes)
            .ok_or_else(|| invalid_plan("pool concurrent invocation bytes overflow u64"))?;
    }
    if covered != *invocation_ids {
        return Err(invalid_plan(
            "pool liveness does not cover every invocation member",
        ));
    }
    Ok(match mode {
        InvocationLivenessMode::NoInvocationResources => unreachable!(),
        InvocationLivenessMode::TotalOrderReuse => maximum_row,
        InvocationLivenessMode::ConservativeConcurrent => concurrent_sum,
    })
}

pub(super) struct ValueAllocationAccumulator {
    pub(super) end_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) demand: ValueResourceDemand,
    pub(super) initialization: StateInitialization,
    pub(super) logical_layout_fingerprints: BTreeSet<String>,
    pub(super) merge_result: Option<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueResourceDemand {
    PlanStatic,
    Fixed {
        lifetime: AllocationLifetime,
    },
    TokenScaled {
        lifetime: AllocationLifetime,
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
}

impl ValueResourceDemand {
    pub(super) fn lifetime(self) -> Option<AllocationLifetime> {
        match self {
            Self::PlanStatic => None,
            Self::Fixed { lifetime } | Self::TokenScaled { lifetime, .. } => Some(lifetime),
        }
    }

    pub(super) fn dynamic_demand(
        self,
        fixed_bytes: u64,
    ) -> Result<DynamicResourceDemand, VNextError> {
        match self {
            Self::PlanStatic => Err(invalid_plan(
                "plan-static value cannot produce a dynamic demand",
            )),
            Self::Fixed { .. } => DynamicResourceDemand::fixed(fixed_bytes),
            Self::TokenScaled {
                bytes_per_token,
                maximum_tokens,
                ..
            } => DynamicResourceDemand::tokens(bytes_per_token, maximum_tokens),
        }
    }
}

impl ValueAllocationAccumulator {
    pub(super) fn merge(
        &mut self,
        end_bytes: u64,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        demand: ValueResourceDemand,
        initialization: StateInitialization,
        logical_layout_fingerprint: String,
    ) -> Option<()> {
        if self.usage != usage
            || self.element_type != element_type
            || self.demand != demand
            || self.initialization != initialization
        {
            return None;
        }
        self.end_bytes = self.end_bytes.max(end_bytes);
        self.alignment_bytes = self.alignment_bytes.max(alignment_bytes);
        self.logical_layout_fingerprints
            .insert(logical_layout_fingerprint);
        Some(())
    }
}
