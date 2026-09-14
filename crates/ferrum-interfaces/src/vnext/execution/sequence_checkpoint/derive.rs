use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::vnext::{
    AllocationKind, BufferUsage, CapabilityCatalog, CheckpointBoundaryConstraint,
    CheckpointPartitionNumerics, CheckpointTokenSpanConstraint, DynamicResourceDemand,
    ModelProgram, PlanNode, ProviderCheckpointCapability, ResolvedTensorLayout,
    StateCapacityDemand, StateCheckpointCapability, StateCheckpointContents, StateLifetime,
    TensorAccess,
};

pub(crate) fn derive_sequence_checkpoint(
    program: &ModelProgram,
    nodes: &[PlanNode],
    descriptors: &[DynamicResourceDescriptor],
    catalog: &CapabilityCatalog,
) -> Result<
    (
        Option<SequenceCheckpointLayout>,
        Vec<SequenceCheckpointUnsupportedReason>,
    ),
    VNextError,
> {
    use SequenceCheckpointUnsupportedReason as Reason;
    let mut reasons = Vec::new();
    let inputs = program.checkpoint_inputs();
    match inputs {
        None => reasons.push(Reason::InputsUndeclared),
        Some(inputs) if !inputs.covers(program.inputs()) => reasons.push(Reason::InputCoverage),
        Some(_) => {}
    }
    if let Some(inputs) = inputs {
        reasons.extend(super::output_only::state_dependencies(inputs, nodes)?);
    }
    if program.states().is_empty() {
        reasons.push(Reason::NoSequenceState);
    }
    let descriptor_map = descriptors
        .iter()
        .map(|value| (value.base_resource_id(), value))
        .collect::<BTreeMap<_, _>>();
    let mut providers = Vec::new();
    let mut selected = BTreeMap::new();
    let mut dependency = CheckpointInputDependency::ExactTokenPrefix;
    for node in nodes {
        if node.provider_resources().persistent().is_some() {
            reasons.push(Reason::ProviderPersistentWorkspace {
                node_id: node.id().clone(),
            });
        }
        let provider = catalog
            .providers_for(node.operation_id())?
            .iter()
            .find(|provider| provider.provider_id() == node.selection().selected_provider())
            .ok_or_else(|| invalid_plan("selected checkpoint provider is absent from catalog"))?;
        if provider.operation_fingerprint() != node.operation_fingerprint()
            || provider.provider_implementation_fingerprint()
                != node.provider_implementation_fingerprint()
        {
            return Err(invalid_plan(
                "checkpoint provider identity differs from selected node",
            ));
        }
        let ProviderCheckpointCapability::CompletedBoundary(contract) =
            provider.checkpoint_capability()
        else {
            reasons.push(Reason::ProviderUndeclared {
                node_id: node.id().clone(),
                provider_id: provider.provider_id().clone(),
            });
            continue;
        };
        if contract.input_dependency() == CheckpointInputDependency::EntireTokenInput {
            dependency = CheckpointInputDependency::EntireTokenInput;
        }
        if contract.partition_numerics() == CheckpointPartitionNumerics::OperationOracle
            && !node.state_effects().is_empty()
        {
            // OracleSpec presently has no typed per-state observation/coverage
            // contract. Do not treat an output oracle as state verification.
            reasons.push(Reason::StateOracleCoverage {
                node_id: node.id().clone(),
            });
        }
        for port in contract.state_ports() {
            let binding = node.values().iter().find(|binding| {
                binding.role() == port.role() && binding.ordinal() == port.ordinal()
            });
            if binding.is_none_or(|binding| {
                !program
                    .states()
                    .iter()
                    .any(|state| &state.value_id == binding.value_id())
            }) || provider
                .dynamic_storage_for(port.role(), port.ordinal())
                .is_none_or(|storage| !storage.accepts(port.storage_profile()))
            {
                reasons.push(Reason::InvalidStatePort {
                    node_id: node.id().clone(),
                });
            }
        }
        selected.insert(node.id(), contract);
        providers.push(SequenceCheckpointProvider {
            node_id: node.id().clone(),
            provider_id: provider.provider_id().clone(),
            operation_fingerprint: node.operation_fingerprint().to_owned(),
            implementation_fingerprint: node.provider_implementation_fingerprint().to_owned(),
            contract: contract.clone(),
        });
    }
    let mut states = Vec::new();
    for state in program.states() {
        if state.lifetime != StateLifetime::Sequence {
            reasons.push(Reason::StateLifetime {
                state_id: state.id.clone(),
                lifetime: match state.lifetime {
                    StateLifetime::Request => AllocationLifetime::Request,
                    StateLifetime::Step => AllocationLifetime::Step,
                    StateLifetime::Sequence => AllocationLifetime::Sequence,
                },
            });
            continue;
        }
        let StateCheckpointCapability::CompletedBoundary(semantics) = state.checkpoint else {
            reasons.push(Reason::StateUndeclared {
                state_id: state.id.clone(),
            });
            continue;
        };
        if semantics.input_dependency() == CheckpointInputDependency::EntireTokenInput {
            dependency = CheckpointInputDependency::EntireTokenInput;
        }
        let mut projection: Option<SequenceCheckpointState> = None;
        let mut writers = BTreeSet::new();
        for node in nodes {
            for binding in node
                .values()
                .iter()
                .filter(|binding| binding.value_id() == &state.value_id)
            {
                if matches!(
                    binding.access(),
                    TensorAccess::Write | TensorAccess::ReadWrite
                ) {
                    writers.insert(node.id().clone());
                }
                let Some(contract) = selected.get(node.id()) else {
                    continue;
                };
                let [component] = binding.storage().components() else {
                    reasons.push(Reason::StateLayout {
                        state_id: state.id.clone(),
                        reason: "multi-component state storage has no complete checkpoint mapping"
                            .to_owned(),
                    });
                    continue;
                };
                let Some(descriptor) = descriptor_map.get(component.resource_id()).copied() else {
                    reasons.push(Reason::StateLayout {
                        state_id: state.id.clone(),
                        reason: "state has no dynamic resource descriptor".to_owned(),
                    });
                    continue;
                };
                let Some(port) = contract.state_ports().iter().find(|port| {
                    port.role() == binding.role()
                        && port.ordinal() == binding.ordinal()
                        && port.storage_profile() == descriptor.storage().profile()
                }) else {
                    reasons.push(Reason::StatePortUndeclared {
                        node_id: node.id().clone(),
                        state_id: state.id.clone(),
                    });
                    continue;
                };
                let bytes = binding.tensor().minimum_storage_bytes()?;
                let valid_mapping = match (semantics.contents(), port.layout()) {
                    (
                        StateCheckpointContents::BoundaryValue,
                        ProviderCheckpointStateLayout::ContiguousBoundaryValue,
                    ) => true,
                    (
                        StateCheckpointContents::PrefixPositions,
                        ProviderCheckpointStateLayout::TokenMajorPrefix,
                    ) => {
                        // The capability establishes token-major semantics.
                        // Capacity is checked for consistency only after that
                        // explicit declaration, never used to infer semantics.
                        component.offset_bytes() == 0
                            && matches!((state.capacity_demand, descriptor.demand()),
                            (StateCapacityDemand::TokenScaled { bytes_per_token, maximum_tokens }, DynamicResourceDemand::Tokens { bytes_per_token: actual, maximum_tokens: limit })
                                if bytes_per_token == bytes && *actual == bytes && maximum_tokens == *limit)
                    }
                    _ => false,
                };
                let maximum_bytes = descriptor.evaluate_logical_request_bytes_for_shape(
                    descriptor.demand().theoretical_maximum_shape(),
                )?;
                if !valid_mapping
                    || !matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
                    || !matches!(state.tensor.layout, ResolvedTensorLayout::Contiguous)
                    || bytes != state.tensor.byte_len()?
                    || component.length_bytes() != bytes
                    || component.element_type() != binding.tensor().element_type()
                    || descriptor.element_type() != component.element_type()
                    || descriptor.usage() != BufferUsage::State
                    || binding.usage() != BufferUsage::State
                    || descriptor.lifetime() != AllocationLifetime::Sequence
                    || descriptor.kind() != &AllocationKind::Value
                    || descriptor.initialization() != state.initialization
                    || component
                        .offset_bytes()
                        .checked_add(bytes)
                        .is_none_or(|end| end > maximum_bytes)
                {
                    reasons.push(Reason::StateLayout { state_id: state.id.clone(), reason: "state contents, tensor, selected storage ABI, initialization or byte bound is incompatible".to_owned() });
                    continue;
                }
                let candidate = SequenceCheckpointState {
                    state_id: state.id.clone(),
                    value_id: state.value_id.clone(),
                    writers: Vec::new(),
                    semantics,
                    tensor: binding.tensor().clone(),
                    resource_id: component.resource_id().clone(),
                    offset_bytes: component.offset_bytes(),
                    layout: port.layout(),
                    storage: descriptor.storage().clone(),
                    initialization: descriptor.initialization(),
                    descriptor: descriptor.clone(),
                };
                if projection
                    .as_ref()
                    .is_some_and(|previous| previous != &candidate)
                {
                    reasons.push(Reason::StateLayout {
                        state_id: state.id.clone(),
                        reason:
                            "state readers/writers disagree on the complete checkpoint projection"
                                .to_owned(),
                    });
                } else {
                    projection = Some(candidate);
                }
            }
        }
        if writers.is_empty() {
            reasons.push(Reason::StateWithoutWriter {
                state_id: state.id.clone(),
            });
        }
        if let Some(mut projection) = projection {
            projection.writers = writers.into_iter().collect();
            states.push(projection);
        } else {
            reasons.push(Reason::StateLayout {
                state_id: state.id.clone(),
                reason: "state has no proven checkpoint projection".to_owned(),
            });
        }
    }
    let covered = states
        .iter()
        .map(|state| &state.resource_id)
        .collect::<BTreeSet<_>>();
    for descriptor in descriptors {
        if (descriptor.usage() == BufferUsage::State
            || descriptor.usage() == BufferUsage::Persistent)
            && !covered.contains(descriptor.base_resource_id())
        {
            reasons.push(Reason::UncoveredResource {
                resource_id: descriptor.base_resource_id().clone(),
            });
        }
    }
    let boundaries = intersect_boundaries(&providers);
    if boundaries.is_err() {
        reasons.push(Reason::BoundaryIntersection);
    }
    if !reasons.is_empty() {
        return Ok((None, reasons));
    }
    let layout = SequenceCheckpointLayout {
        data: SequenceCheckpointLayoutData {
            contract_version: SEQUENCE_CHECKPOINT_LAYOUT_VERSION,
            inputs: inputs.expect("input coverage checked").clone(),
            input_dependency: dependency,
            boundaries: boundaries.expect("boundary intersection checked"),
            completed_input_capture: if providers.iter().all(|provider| {
                provider.contract.completed_input_capture()
                    == CheckpointCompletedInputCapture::Supported
            }) {
                CheckpointCompletedInputCapture::Supported
            } else {
                CheckpointCompletedInputCapture::Unsupported
            },
            providers,
            states,
        },
    };
    if let Err(error) = layout.validate_aliases() {
        return Ok((
            None,
            vec![Reason::StateLayout {
                state_id: layout.data.states[0].state_id.clone(),
                reason: error.to_string(),
            }],
        ));
    }
    Ok((Some(layout), Vec::new()))
}

fn intersect_boundaries(
    providers: &[SequenceCheckpointProvider],
) -> Result<CheckpointBoundaryConstraint, VNextError> {
    use std::num::NonZeroU64;
    fn gcd(mut left: u64, mut right: u64) -> u64 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }
    let merge = |left: CheckpointTokenSpanConstraint, right: CheckpointTokenSpanConstraint| {
        let a = left.alignment().get();
        let b = right.alignment().get();
        let alignment = (a / gcd(a, b))
            .checked_mul(b)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| {
                invalid_plan("checkpoint span alignments have no representable intersection")
            })?;
        CheckpointTokenSpanConstraint::new(
            left.minimum_tokens().max(right.minimum_tokens()),
            alignment,
        )
    };
    let mut prefix = CheckpointTokenSpanConstraint::any_positive();
    let mut suffix = CheckpointTokenSpanConstraint::any_positive();
    for provider in providers {
        prefix = merge(prefix, provider.contract.boundaries().prefix())?;
        suffix = merge(suffix, provider.contract.boundaries().suffix())?;
    }
    CheckpointBoundaryConstraint::new(prefix, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    // Pure checked intersection arithmetic; whole-plan closure and selection
    // are exercised through real registered plan builds in integration tests.
    fn provider(prefix: u64, suffix: u64) -> SequenceCheckpointProvider {
        SequenceCheckpointProvider {
            node_id: NodeId::new(format!("node.{prefix}.{suffix}")).unwrap(),
            provider_id: ProviderId::new("provider.arithmetic-fixture").unwrap(),
            operation_fingerprint: "a".repeat(64),
            implementation_fingerprint: "b".repeat(64),
            contract: ProviderCheckpointContract::new(
                CheckpointInputDependency::ExactTokenPrefix,
                CheckpointBoundaryConstraint::new(
                    CheckpointTokenSpanConstraint::new(
                        NonZeroU64::MIN,
                        NonZeroU64::new(prefix).unwrap(),
                    )
                    .unwrap(),
                    CheckpointTokenSpanConstraint::new(
                        NonZeroU64::MIN,
                        NonZeroU64::new(suffix).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap(),
                CheckpointPartitionNumerics::BitwiseEquivalent,
            ),
        }
    }

    #[test]
    fn checkpoint_span_intersection_uses_all_providers_and_actual_restored_start() {
        let combined = intersect_boundaries(&[provider(4, 3), provider(6, 2)]).unwrap();
        assert_eq!(combined.prefix().alignment().get(), 12);
        assert_eq!(combined.suffix().alignment().get(), 6);
        assert!(combined.permits_from(5, 17, 23));
        assert!(!combined.permits_from(5, 17, 20));
        assert!(!combined.permits_from(5, 11, 17));
        assert!(!combined.permits_from(17, 17, 23));
        assert!(intersect_boundaries(&[provider(1 << 63, 1), provider(3, 1)]).is_err());
    }
}
