//! Complete selected-state effects, derived from the actual selected providers.
use super::*;
use crate::{
    CheckpointInputDependency, ModelProgram, ProviderCandidateStateEffects as Effects,
    ProviderCandidateStatePort as Port, StateCheckpointCapability, StateCheckpointContents,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CandidateStateLayout {
    pub(crate) copies: Vec<CandidateStateCopy>,
    pub(crate) prefix_resources: Vec<ResourceId>,
    profiles: BTreeMap<ResourceId, DynamicStorageProfile>,
    pub(crate) inputs: crate::ProgramCheckpointInputs,
    pub(crate) input_ports: Vec<CandidateInputPort>,
    pub(crate) output_only_ports: Vec<(NodeId, u32)>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CandidateInputPort {
    pub node: NodeId,
    pub ordinal: u32,
    pub value: ProgramValueId,
    pub tensor: ResolvedTensorSpec,
    pub token_rows: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CandidateStateCopy {
    pub node: NodeId,
    pub candidate_value: ProgramValueId,
    pub candidate_resource: ResourceId,
    pub candidate_offset: u64,
    pub destination: ResourceId,
    pub destination_offset: u64,
    pub bytes: u64,
    pub element_type: ElementType,
}
impl CandidateStateLayout {
    /// Candidate copies remain same-frame effects. On-demand replay may
    /// accelerate other compute nodes, but complete candidate-state producers
    /// must remain eager boundaries and their outputs must retain Step storage.
    pub(crate) fn validate_replay_policy<P: RuntimePolicy>(
        &self,
        nodes: &[PlanNode],
        policy: &P,
    ) -> Result<(), VNextError> {
        let Some(programs) = policy
            .reusable_execution_policy()
            .and_then(|reusable| reusable.program_policy())
        else {
            return Ok(());
        };
        if programs.preparation() != crate::ReusableExecutionProgramPreparation::OnDemand
            || policy.execution_determinism_requirement()
                != crate::ExecutionDeterminismRequirement::BitwiseSameRuntime
        {
            return Err(invalid_plan(
                "candidate replay requires explicit on-demand compute segments with declared eager boundaries",
            ));
        }
        for copy in &self.copies {
            let node = nodes
                .iter()
                .find(|node| node.id() == &copy.node)
                .ok_or_else(|| invalid_plan("candidate copy producer is absent"))?;
            if node.provider_execution_semantics().replay_equivalence()
                != crate::ProviderReplayEquivalence::Ineligible
            {
                return Err(invalid_plan(
                    "candidate state producers must remain outside reusable compute segments",
                ));
            }
        }
        // Dispatch independently requires every ineligible provider to declare
        // EagerBoundary. It rejects Static/Dynamic topology for such providers;
        // this planning permission never grants candidate-state replay.
        Ok(())
    }

    pub(crate) fn retained_values(&self) -> impl Iterator<Item = &ProgramValueId> {
        self.copies.iter().map(|x| &x.candidate_value)
    }
    pub(crate) fn validate_memory(&self, memory: &MemoryPlan) -> Result<(), VNextError> {
        let mut covered: BTreeSet<_> = self.prefix_resources.iter().collect();
        for c in &self.copies {
            covered.insert(&c.destination);
            let d = memory
                .dynamic_descriptors()
                .iter()
                .find(|d| d.base_resource_id() == &c.candidate_resource)
                .ok_or_else(|| invalid_plan("candidate output lacks dynamic storage"))?;
            if d.lifetime() != AllocationLifetime::Step
                || d.kind() != &AllocationKind::Value
                || d.element_type() != c.element_type
            {
                return Err(invalid_plan(
                    "candidate output must be independent Step value storage",
                ));
            }
        }
        for (id, profile) in &self.profiles {
            let d = memory
                .dynamic_descriptors()
                .iter()
                .find(|d| d.base_resource_id() == id)
                .ok_or_else(|| invalid_plan("candidate state descriptor absent"))?;
            if d.storage().profile() != *profile
                || d.lifetime() != AllocationLifetime::Sequence
                || d.kind() != &AllocationKind::Value
            {
                return Err(invalid_plan(
                    "candidate state storage ABI differs from selected provider promise",
                ));
            }
        }
        for d in memory.dynamic_descriptors() {
            if matches!(d.usage(), BufferUsage::State | BufferUsage::Persistent)
                && !covered.contains(d.base_resource_id())
            {
                return Err(invalid_plan(
                    "candidate settlement does not cover every persistent resource",
                ));
            }
        }
        Ok(())
    }
}
impl ExecutionPlan {
    pub fn candidate_state_layout(&self) -> Option<&CandidateStateLayout> {
        self.candidate_state_layout.as_ref()
    }
    pub fn requires_selected_prefix_commit(&self) -> bool {
        self.candidate_state_layout.is_some()
    }
}

pub(super) fn derive_candidate_state(
    program: &ModelProgram,
    nodes: &[PlanNode],
    catalog: &CapabilityCatalog,
) -> Result<Option<CandidateStateLayout>, VNextError> {
    let providers = nodes
        .iter()
        .map(|node| {
            let p = catalog
                .providers_for(node.operation_id())?
                .iter()
                .find(|p| p.provider_id() == node.selection().selected_provider())
                .ok_or_else(|| invalid_plan("candidate provider absent"))?;
            if p.operation_fingerprint() != node.operation_fingerprint()
                || p.provider_implementation_fingerprint()
                    != node.provider_implementation_fingerprint()
            {
                return Err(invalid_plan("candidate provider identity mismatch"));
            }
            Ok(p)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let enabled=!program.states().is_empty() && providers.iter().any(|p|matches!(p.candidate_state_effects(),Effects::PrefixCausal{ports} if ports.iter().any(|x|matches!(x,Port::TailTwo{..}))));
    if !enabled {
        return Ok(None);
    }
    let inputs = program.checkpoint_inputs().ok_or_else(|| {
        invalid_plan("candidate plan requires explicit token input dependency closure")
    })?;
    if !inputs.covers(program.inputs())
        || !super::sequence_checkpoint::output_only_state_dependencies(inputs, nodes)?.is_empty()
    {
        return Err(invalid_plan(
            "candidate input roles do not cover the program or output-only inputs affect state",
        ));
    }
    let mut input_ports = Vec::new();
    for node in nodes {
        for v in node.values().iter().filter(|v| {
            v.role() == ResolvedValueRole::Input
                && (v.value_id() == inputs.token_input()
                    || inputs.conditioning_inputs().contains(v.value_id()))
        }) {
            if v.access() != TensorAccess::Read
                || v.tensor().dimensions().is_empty()
                || !matches!(v.tensor().layout(), ResolvedTensorLayout::Contiguous)
            {
                return Err(invalid_plan(
                    "candidate immutable inputs must be contiguous read-only row ports",
                ));
            }
            let projection = node.work().token_projection(v.role(), v.ordinal());
            if projection.is_some_and(|p| p.axis() != 0) {
                return Err(invalid_plan(
                    "candidate conditioning currently supports a leading token axis",
                ));
            }
            if v.value_id() == inputs.token_input()
                && (v.tensor().element_type() != ElementType::U32
                    || v.tensor().dimensions().len() != 1
                    || projection.is_none())
            {
                return Err(invalid_plan(
                    "candidate token port must be projected U32 token ids",
                ));
            }
            input_ports.push(CandidateInputPort {
                node: node.id().clone(),
                ordinal: v.ordinal(),
                value: v.value_id().clone(),
                tensor: v.tensor().clone(),
                token_rows: projection.is_some(),
            });
        }
    }
    for value in inputs
        .conditioning_inputs()
        .iter()
        .chain([inputs.token_input()])
    {
        if !input_ports.iter().any(|p| &p.value == value) {
            return Err(invalid_plan("candidate input has no actual read port"));
        }
    }
    let output_only_ports = nodes
        .iter()
        .flat_map(|n| {
            n.values()
                .iter()
                .filter(|v| {
                    v.role() == ResolvedValueRole::Input
                        && inputs.output_only_inputs().contains(v.value_id())
                })
                .map(|v| (n.id().clone(), v.ordinal()))
        })
        .collect();
    let mut copies = Vec::new();
    let mut prefixes = BTreeSet::new();
    let mut covered = BTreeSet::new();
    let mut profiles = BTreeMap::new();
    for (node, provider) in nodes.iter().zip(providers) {
        if node.provider_resources().persistent().is_some() {
            return Err(invalid_plan(
                "candidate plan has hidden persistent workspace",
            ));
        }
        let ports = match provider.candidate_state_effects() {
            Effects::PrefixCausal { ports } => ports,
            Effects::TerminalOnly => {
                let values = node.values().iter().filter(|v| {
                    v.role() == ResolvedValueRole::Output
                        || matches!(v.access(), TensorAccess::Write | TensorAccess::ReadWrite)
                }).map(|v| v.value_id().clone());
                if !node.state_effects().is_empty()
                    || !super::sequence_checkpoint::state_dependencies_from_values(values, nodes)?.is_empty()
                {
                    return Err(invalid_plan("terminal candidate output affects persistent state"));
                }
                continue;
            }
            Effects::Unsupported => return Err(invalid_plan(
                "every selected node must promise prefix-causal effects or prove terminal-only outputs",
            )),
        };
        let mut declared = BTreeSet::new();
        for port in ports {
            let (role, ordinal, profile) = match port {
                Port::TailTwo {
                    input_ordinal,
                    storage_profile,
                    ..
                } => (ResolvedValueRole::Input, *input_ordinal, *storage_profile),
                Port::PrefixAppendOnly {
                    role,
                    ordinal,
                    storage_profile,
                } => (*role, *ordinal, *storage_profile),
            };
            if !declared.insert((role, ordinal)) {
                return Err(invalid_plan("duplicate candidate state port"));
            }
            let binding = node
                .values()
                .iter()
                .find(|v| v.role() == role && v.ordinal() == ordinal)
                .ok_or_else(|| invalid_plan("candidate port absent"))?;
            let state = program
                .states()
                .iter()
                .find(|s| &s.value_id == binding.value_id())
                .ok_or_else(|| invalid_plan("candidate port is not semantic state"))?;
            let StateCheckpointCapability::CompletedBoundary(semantics) = state.checkpoint else {
                return Err(invalid_plan("candidate state contents undeclared"));
            };
            if state.lifetime != StateLifetime::Sequence
                || semantics.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
                || !matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
                || binding.tensor().minimum_storage_bytes()? != state.tensor.byte_len()?
            {
                return Err(invalid_plan(
                    "candidate state must be complete prefix-causal Sequence storage",
                ));
            }
            let [dst] = binding.storage().components() else {
                return Err(invalid_plan(
                    "candidate state requires one physical component",
                ));
            };
            if provider
                .dynamic_storage_for(role, ordinal)
                .is_none_or(|r| !r.accepts(profile))
                || dst.element_type() != binding.tensor().element_type()
                || dst.length_bytes() != state.tensor.byte_len()?
                || binding.usage() != BufferUsage::State
            {
                return Err(invalid_plan("candidate state physical port mismatch"));
            }
            if profiles
                .insert(dst.resource_id().clone(), profile)
                .is_some_and(|old| old != profile)
            {
                return Err(invalid_plan("candidate state storage profiles disagree"));
            }
            match port {
                Port::TailTwo { output_ordinal, .. } => {
                    if semantics.contents() != StateCheckpointContents::BoundaryValue
                        || state.capacity_demand != StateCapacityDemand::FixedPerScope
                        || binding.access() != TensorAccess::Read
                        || role != ResolvedValueRole::Input
                        || !covered.insert(state.id.clone())
                    {
                        return Err(invalid_plan(
                            "boundary candidate original must have one read-only producer",
                        ));
                    }
                    let out = node
                        .values()
                        .iter()
                        .find(|v| {
                            v.role() == ResolvedValueRole::Output && v.ordinal() == *output_ordinal
                        })
                        .ok_or_else(|| invalid_plan("candidate output absent"))?;
                    let [src] = out.storage().components() else {
                        return Err(invalid_plan("candidate output requires one component"));
                    };
                    let dims = out.tensor().dimensions();
                    if dims.first() != Some(&2)
                        || dims.get(1..) != Some(binding.tensor().dimensions())
                        || out.tensor().element_type() != binding.tensor().element_type()
                        || !matches!(out.tensor().layout(), ResolvedTensorLayout::Contiguous)
                        || out.access() != TensorAccess::Write
                        || out.usage() != BufferUsage::Activations
                        || src.resource_id() == dst.resource_id()
                        || src.element_type() != dst.element_type()
                        || src.length_bytes()
                            != dst
                                .length_bytes()
                                .checked_mul(2)
                                .ok_or_else(|| invalid_plan("candidate extent overflow"))?
                        || node
                            .work()
                            .token_projection(ResolvedValueRole::Output, *output_ordinal)
                            .is_some()
                    {
                        return Err(invalid_plan("tail candidates must be independent exact two-slot complete state outputs"));
                    }
                    copies.push(CandidateStateCopy {
                        node: node.id().clone(),
                        candidate_value: out.value_id().clone(),
                        candidate_resource: src.resource_id().clone(),
                        candidate_offset: src.offset_bytes(),
                        destination: dst.resource_id().clone(),
                        destination_offset: dst.offset_bytes(),
                        bytes: dst.length_bytes(),
                        element_type: dst.element_type(),
                    });
                }
                Port::PrefixAppendOnly { .. } => {
                    if semantics.contents() != StateCheckpointContents::PrefixPositions
                        || !matches!(
                            binding.access(),
                            TensorAccess::Write | TensorAccess::ReadWrite
                        )
                        || dst.offset_bytes() != 0
                        || !matches!(state.capacity_demand,StateCapacityDemand::TokenScaled{bytes_per_token,..} if bytes_per_token==dst.length_bytes())
                        || !covered.insert(state.id.clone())
                    {
                        return Err(invalid_plan(
                            "prefix writer must cover exact token-major append-only state once",
                        ));
                    }
                    prefixes.insert(dst.resource_id().clone());
                }
            }
        }
        for effect in node.state_effects() {
            if effect.lifetime() != AllocationLifetime::Sequence {
                return Err(invalid_plan(
                    "candidate plan has non-Sequence persistent state",
                ));
            }
            for v in node
                .values()
                .iter()
                .filter(|v| v.value_id() == effect.state_value_id())
            {
                if matches!(v.access(), TensorAccess::Write | TensorAccess::ReadWrite)
                    && !declared.contains(&(v.role(), v.ordinal()))
                {
                    return Err(invalid_plan(
                        "candidate plan has an undeclared persistent writer",
                    ));
                }
            }
        }
    }
    if copies.is_empty() || covered.len() != program.states().len() {
        return Err(invalid_plan(
            "candidate effects do not cover every semantic state",
        ));
    }
    copies.sort_by(|a, b| {
        (&a.destination, a.destination_offset).cmp(&(&b.destination, b.destination_offset))
    });
    for p in copies.windows(2) {
        if p[0].destination == p[1].destination
            && p[0]
                .destination_offset
                .checked_add(p[0].bytes)
                .is_none_or(|e| e > p[1].destination_offset)
        {
            return Err(invalid_plan("candidate state destinations overlap"));
        }
    }
    Ok(Some(CandidateStateLayout {
        copies,
        prefix_resources: prefixes.into_iter().collect(),
        profiles,
        inputs: inputs.clone(),
        input_ports,
        output_only_ports,
    }))
}
