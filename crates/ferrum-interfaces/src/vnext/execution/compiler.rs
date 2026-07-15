use super::*;

/// Explicit semantic inputs and per-node selection preferences for compiling a
/// model program. Product input capacities are required because they bound
/// request-lifetime backing; the compiler never guesses a one-token capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramPlanCompileOptions {
    tensor_specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    required_capabilities: BTreeMap<NodeId, BTreeSet<CapabilityId>>,
    preferred_providers: BTreeMap<NodeId, ProviderId>,
}

impl ProgramPlanCompileOptions {
    pub fn new(
        input_tensor_specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    ) -> Result<Self, VNextError> {
        for (value_id, tensor) in &input_tensor_specs {
            tensor.validate(&format!("program input `{value_id}`"))?;
        }
        Ok(Self {
            tensor_specs: input_tensor_specs,
            required_capabilities: BTreeMap::new(),
            preferred_providers: BTreeMap::new(),
        })
    }

    /// Adds an explicit tensor for an intermediate/output whose contract is
    /// intentionally not inferable (for example a non-contiguous layout).
    pub fn insert_tensor_spec(
        &mut self,
        value_id: ProgramValueId,
        tensor: ProgramTensorSpec,
    ) -> Result<(), VNextError> {
        tensor.validate(&format!("program value `{value_id}`"))?;
        self.tensor_specs.insert(value_id, tensor);
        Ok(())
    }

    pub fn require_capability(&mut self, node_id: NodeId, capability: CapabilityId) {
        self.required_capabilities
            .entry(node_id)
            .or_default()
            .insert(capability);
    }

    pub fn prefer_provider(&mut self, node_id: NodeId, provider_id: ProviderId) {
        self.preferred_providers.insert(node_id, provider_id);
    }

    pub fn tensor_specs(&self) -> &BTreeMap<ProgramValueId, ProgramTensorSpec> {
        &self.tensor_specs
    }
}

/// Immutable executable plus the trusted node resolutions used to build it.
/// Keeping the resolutions allows a product-level plan wrapper to validate the
/// exact same physical decisions without reconstructing provider evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramPlanCompilation {
    executable: ExecutablePlan,
    node_resolutions: Vec<PlanNodeResolution>,
    value_tensors: BTreeMap<ProgramValueId, ResolvedTensorSpec>,
}

impl ProgramPlanCompilation {
    pub fn executable(&self) -> &ExecutablePlan {
        &self.executable
    }

    pub fn node_resolutions(&self) -> &[PlanNodeResolution] {
        &self.node_resolutions
    }

    pub fn value_tensors(&self) -> &BTreeMap<ProgramValueId, ResolvedTensorSpec> {
        &self.value_tensors
    }

    pub fn into_parts(
        self,
    ) -> (
        ExecutablePlan,
        Vec<PlanNodeResolution>,
        BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    ) {
        (self.executable, self.node_resolutions, self.value_tensors)
    }
}

/// Backend-neutral compiler from semantic model programs to immutable physical
/// execution plans. A metadata-only provider pass discovers the initial value
/// alignment, then aligned dtype arenas are rebuilt until the estimator result
/// reaches a bounded monotonic fixed point. This avoids both an unproved
/// alignment guess and one device allocation per weight component.
pub struct ProgramPlanCompiler;

impl ProgramPlanCompiler {
    pub fn compile<P: RuntimePolicy>(
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        policy: &P,
        planning: &OperationPlanningHandle<'_>,
        options: &ProgramPlanCompileOptions,
    ) -> Result<ProgramPlanCompilation, VNextError> {
        validate_compile_options(family, options)?;
        let value_tensors = infer_value_tensors(family, catalog, options)?;
        let family_fingerprint = family.fingerprint()?;

        let probe_locations = dedicated_weight_locations(family, &family_fingerprint)?;
        let probe_storages = build_value_storages(
            family,
            catalog,
            &value_tensors,
            &probe_locations,
            &family_fingerprint,
        )?;
        let probe_resolutions = resolve_nodes(
            family,
            catalog,
            policy,
            planning,
            options,
            &value_tensors,
            &probe_storages,
        )?;
        let mut value_alignment_bytes = maximum_value_alignment(&probe_resolutions)?;
        if !value_alignment_bytes.is_power_of_two() {
            return Err(invalid_plan(
                "provider value-alignment evidence is not a power of two",
            ));
        }

        let mut final_resolution = None;
        for _ in 0..=u64::BITS {
            let arena_locations =
                arena_weight_locations(family, &family_fingerprint, value_alignment_bytes)?;
            let final_storages = build_value_storages(
                family,
                catalog,
                &value_tensors,
                &arena_locations,
                &family_fingerprint,
            )?;
            let node_resolutions = resolve_nodes(
                family,
                catalog,
                policy,
                planning,
                options,
                &value_tensors,
                &final_storages,
            )?;
            let observed_alignment = maximum_value_alignment(&node_resolutions)?;
            if observed_alignment <= value_alignment_bytes {
                final_resolution = Some(node_resolutions);
                break;
            }
            value_alignment_bytes = observed_alignment;
        }
        let node_resolutions = final_resolution.ok_or_else(|| {
            invalid_plan("provider value alignment did not reach a bounded monotonic fixed point")
        })?;
        let plan = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            catalog,
            policy,
            node_resolutions.clone(),
        )?)?;
        let executable = ExecutablePlan::new(plan, catalog.clone())?;
        Ok(ProgramPlanCompilation {
            executable,
            node_resolutions,
            value_tensors,
        })
    }
}

fn maximum_value_alignment(resolutions: &[PlanNodeResolution]) -> Result<u64, VNextError> {
    let alignment = resolutions
        .iter()
        .flat_map(PlanNodeResolution::provider_resource_candidates)
        .map(ProviderResourcePlan::value_alignment_bytes)
        .max()
        .ok_or_else(|| invalid_plan("program has no provider value-alignment evidence"))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_plan(
            "provider value-alignment evidence is not a power of two",
        ));
    }
    Ok(alignment)
}

#[derive(Debug, Clone)]
struct WeightComponentLocation {
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

fn validate_compile_options(
    family: &PreparedModelFamily,
    options: &ProgramPlanCompileOptions,
) -> Result<(), VNextError> {
    let program = family.program();
    let nodes = program
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .collect::<Vec<_>>();
    let node_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    if options
        .required_capabilities
        .keys()
        .chain(options.preferred_providers.keys())
        .any(|node_id| !node_ids.contains(node_id))
    {
        return Err(invalid_plan(
            "program compile options reference an unknown node",
        ));
    }

    let mut known_values = program.inputs().iter().cloned().collect::<BTreeSet<_>>();
    known_values.extend(program.states().iter().map(|state| state.value_id.clone()));
    known_values.extend(
        program
            .weights()
            .iter()
            .map(|weight| weight.value_id.clone()),
    );
    known_values.extend(nodes.iter().flat_map(|node| node.outputs.iter().cloned()));
    if options
        .tensor_specs
        .keys()
        .any(|value_id| !known_values.contains(value_id))
    {
        return Err(invalid_plan(
            "program compile options contain an unknown semantic value",
        ));
    }
    if program
        .inputs()
        .iter()
        .any(|input| !options.tensor_specs.contains_key(input))
    {
        return Err(invalid_plan(
            "every program input requires an explicit canonical tensor capacity",
        ));
    }
    Ok(())
}

fn infer_value_tensors(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    options: &ProgramPlanCompileOptions,
) -> Result<BTreeMap<ProgramValueId, ResolvedTensorSpec>, VNextError> {
    let program = family.program();
    let mut tensors = options
        .tensor_specs
        .iter()
        .map(|(value_id, tensor)| Ok((value_id.clone(), resolved_tensor(tensor)?)))
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;

    for weight in program.weights() {
        insert_exact_tensor(
            &mut tensors,
            &weight.value_id,
            resolved_tensor(&weight.tensor)?,
            "weight",
        )?;
    }
    for state in program.states() {
        insert_exact_tensor(
            &mut tensors,
            &state.value_id,
            resolved_tensor(&state.tensor)?,
            "state",
        )?;
    }

    for node in program.blocks().iter().flat_map(|block| &block.nodes) {
        let operation = catalog.operation(&node.operation_id)?;
        if node.inputs.len() != operation.inputs.len()
            || node.outputs.len() != operation.outputs.len()
        {
            return Err(invalid_plan(format!(
                "node `{}` arity differs from operation `{}`",
                node.id, operation.id
            )));
        }
        // Attribute validation and tensor-shape validation are separate typed
        // contracts. Equal strings do not create an implicit equation between
        // an AttributeId and a DimensionConstraint::Symbol.
        let mut symbols = TensorSymbols::default();

        for (ordinal, (value_id, contract)) in node.inputs.iter().zip(&operation.inputs).enumerate()
        {
            let tensor = tensors.get(value_id).ok_or_else(|| {
                invalid_plan(format!(
                    "node `{}` input `{value_id}` has no concrete tensor",
                    node.id
                ))
            })?;
            unify_tensor(
                contract,
                tensor,
                &mut symbols,
                &node.id,
                "input",
                ordinal,
                value_id,
            )?;
        }
        for (ordinal, (value_id, contract)) in
            node.outputs.iter().zip(&operation.outputs).enumerate()
        {
            if let Some(tensor) = tensors.get(value_id) {
                unify_tensor(
                    contract,
                    tensor,
                    &mut symbols,
                    &node.id,
                    "output",
                    ordinal,
                    value_id,
                )?;
            }
        }
        for (value_id, contract) in node.outputs.iter().zip(&operation.outputs) {
            if !tensors.contains_key(value_id) {
                let tensor = infer_tensor(contract, &mut symbols, &node.id)?;
                tensors.insert(value_id.clone(), tensor);
            }
        }
    }
    if program
        .outputs()
        .iter()
        .any(|output| !tensors.contains_key(output))
    {
        return Err(invalid_plan(
            "program compilation did not resolve every semantic output",
        ));
    }
    Ok(tensors)
}

fn insert_exact_tensor(
    tensors: &mut BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    value_id: &ProgramValueId,
    tensor: ResolvedTensorSpec,
    kind: &str,
) -> Result<(), VNextError> {
    if tensors
        .insert(value_id.clone(), tensor.clone())
        .is_some_and(|existing| existing != tensor)
    {
        return Err(invalid_plan(format!(
            "explicit {kind} tensor `{value_id}` differs from model semantics"
        )));
    }
    Ok(())
}

fn resolved_tensor(tensor: &ProgramTensorSpec) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(
        tensor.dimensions.clone(),
        tensor.element_type,
        tensor.layout.clone(),
    )
}

#[derive(Debug, Clone, Default)]
struct TensorSymbols {
    dimensions: BTreeMap<String, u64>,
    strides: BTreeMap<String, u64>,
}

fn unify_tensor(
    contract: &TensorContract,
    tensor: &ResolvedTensorSpec,
    symbols: &mut TensorSymbols,
    node_id: &NodeId,
    role: &str,
    ordinal: usize,
    value_id: &ProgramValueId,
) -> Result<(), VNextError> {
    let context = format!("node `{node_id}` {role}[{ordinal}] `{value_id}`");
    if !contract.element_types().contains(&tensor.element_type())
        || contract.dimensions().len() != tensor.dimensions().len()
    {
        return Err(invalid_plan(format!(
            "{context} rank or dtype differs from its operation contract: actual rank={} dtype={:?}, expected rank={} dtypes={:?}",
            tensor.dimensions().len(),
            tensor.element_type(),
            contract.dimensions().len(),
            contract.element_types()
        )));
    }
    unify_dimensions(
        contract.dimensions(),
        tensor.dimensions(),
        &mut symbols.dimensions,
        &context,
    )?;
    if !layout_matches(contract.layouts(), tensor.layout(), &mut symbols.strides)? {
        return Err(invalid_plan(format!(
            "{context} layout {:?} differs from its operation contract {:?}",
            tensor.layout(),
            contract.layouts()
        )));
    }
    Ok(())
}

fn unify_dimensions(
    constraints: &[DimensionConstraint],
    dimensions: &[u64],
    symbols: &mut BTreeMap<String, u64>,
    context: &str,
) -> Result<(), VNextError> {
    for (axis, (constraint, extent)) in constraints.iter().zip(dimensions).enumerate() {
        let valid = match constraint {
            DimensionConstraint::Exact(expected) => expected == extent,
            DimensionConstraint::Range { minimum, maximum } => {
                minimum <= extent && extent <= maximum
            }
            DimensionConstraint::Symbol(symbol) => bind_symbol(symbols, symbol, *extent),
        };
        if !valid {
            return Err(invalid_plan(format!(
                "{context} dimension[{axis}]={extent} violates `{constraint:?}`"
            )));
        }
    }
    Ok(())
}

fn bind_symbol(symbols: &mut BTreeMap<String, u64>, symbol: &str, value: u64) -> bool {
    if value == 0 {
        return false;
    }
    match symbols.get(symbol) {
        Some(existing) => *existing == value,
        None => {
            symbols.insert(symbol.to_owned(), value);
            true
        }
    }
}

fn infer_tensor(
    contract: &TensorContract,
    symbols: &mut TensorSymbols,
    node_id: &NodeId,
) -> Result<ResolvedTensorSpec, VNextError> {
    let element_type = contract
        .element_types()
        .iter()
        .copied()
        .next()
        .filter(|_| contract.element_types().len() == 1)
        .ok_or_else(|| {
            invalid_plan(format!(
                "node `{node_id}` output dtype is ambiguous; provide an explicit tensor"
            ))
        })?;
    let dimensions = contract
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            DimensionConstraint::Exact(value) => Ok(*value),
            DimensionConstraint::Range { minimum, maximum } if minimum == maximum => Ok(*minimum),
            DimensionConstraint::Range { .. } => Err(invalid_plan(format!(
                "node `{node_id}` output range is ambiguous; provide an explicit tensor"
            ))),
            DimensionConstraint::Symbol(symbol) => symbols
                .dimensions
                .get(symbol)
                .copied()
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{node_id}` output symbol `{symbol}` is unresolved; provide an explicit tensor"
                    ))
                }),
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let layout = infer_layout(contract.layouts(), &dimensions, &symbols.strides, node_id)?;
    ResolvedTensorSpec::new(dimensions, element_type, layout)
}

fn infer_layout(
    layouts: &[LayoutConstraint],
    dimensions: &[u64],
    stride_symbols: &BTreeMap<String, u64>,
    node_id: &NodeId,
) -> Result<ResolvedTensorLayout, VNextError> {
    if layouts.len() != 1 {
        return Err(invalid_plan(format!(
            "node `{node_id}` output layout is ambiguous; provide an explicit tensor"
        )));
    }
    match &layouts[0] {
        LayoutConstraint::Contiguous => Ok(ResolvedTensorLayout::Contiguous),
        LayoutConstraint::Strided { strides } => {
            let byte_strides = strides
                .iter()
                .map(|stride| match stride {
                    StrideConstraint::ExactBytes(value) => Ok(*value),
                    StrideConstraint::Symbol(symbol) => {
                        stride_symbols.get(symbol).copied().ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{node_id}` stride symbol `{symbol}` is unresolved"
                            ))
                        })
                    }
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            Ok(ResolvedTensorLayout::Strided { byte_strides })
        }
        LayoutConstraint::Blocked { block, axis_order } => {
            let divisible = dimensions
                .iter()
                .zip(block)
                .all(|(extent, block)| extent % block == 0);
            let padding = if divisible {
                BlockedTensorPadding::Exact
            } else {
                let physical_dimensions = axis_order
                    .iter()
                    .map(|axis| {
                        let extent = dimensions[*axis as usize];
                        let block = block[*axis as usize];
                        extent
                            .checked_add(block - 1)
                            .map(|value| value / block * block)
                            .ok_or_else(|| invalid_plan("blocked tensor padding overflows u64"))
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?;
                BlockedTensorPadding::ZeroFill {
                    physical_dimensions,
                }
            };
            Ok(ResolvedTensorLayout::Blocked {
                block: block.clone(),
                axis_order: axis_order.clone(),
                padding,
            })
        }
    }
}

fn layout_matches(
    constraints: &[LayoutConstraint],
    layout: &ResolvedTensorLayout,
    symbols: &mut BTreeMap<String, u64>,
) -> Result<bool, VNextError> {
    for constraint in constraints {
        let mut candidate_symbols = symbols.clone();
        let matches = match (constraint, layout) {
            (LayoutConstraint::Contiguous, ResolvedTensorLayout::Contiguous) => true,
            (
                LayoutConstraint::Strided { strides },
                ResolvedTensorLayout::Strided { byte_strides },
            ) if strides.len() == byte_strides.len() => {
                strides
                    .iter()
                    .zip(byte_strides)
                    .all(|(constraint, stride)| match constraint {
                        StrideConstraint::ExactBytes(expected) => expected == stride,
                        StrideConstraint::Symbol(symbol) => {
                            bind_symbol(&mut candidate_symbols, symbol, *stride)
                        }
                    })
            }
            (
                LayoutConstraint::Blocked { block, axis_order },
                ResolvedTensorLayout::Blocked {
                    block: actual_block,
                    axis_order: actual_axis_order,
                    ..
                },
            ) => block == actual_block && axis_order == actual_axis_order,
            _ => false,
        };
        if matches {
            *symbols = candidate_symbols;
            return Ok(true);
        }
    }
    Ok(false)
}

fn dedicated_weight_locations(
    family: &PreparedModelFamily,
    family_fingerprint: &str,
) -> Result<BTreeMap<WeightId, WeightComponentLocation>, VNextError> {
    referenced_weight_components(family)?
        .into_iter()
        .map(|component| {
            let length_bytes = component.physical_bytes()?;
            Ok((
                component.id.clone(),
                WeightComponentLocation {
                    resource_id: hashed_resource_id(
                        "weight-probe",
                        family_fingerprint,
                        component.id.as_str(),
                    )?,
                    offset_bytes: 0,
                    length_bytes,
                    element_type: component.physical_element_type(),
                },
            ))
        })
        .collect()
}

fn arena_weight_locations(
    family: &PreparedModelFamily,
    family_fingerprint: &str,
    provider_alignment_bytes: u64,
) -> Result<BTreeMap<WeightId, WeightComponentLocation>, VNextError> {
    let components = referenced_weight_components(family)?;
    let mut arena_ids = BTreeMap::<ElementType, ResourceId>::new();
    let mut next_offsets = BTreeMap::<ElementType, u64>::new();
    let mut locations = BTreeMap::new();
    for component in components {
        let element_type = component.physical_element_type();
        let alignment = provider_alignment_bytes.max(element_type.size_bytes());
        let next_offset = next_offsets.entry(element_type).or_insert(0);
        let offset_bytes = checked_align_up(*next_offset, alignment)?;
        let length_bytes = component.physical_bytes()?;
        *next_offset = offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_plan("weight arena byte range overflows u64"))?;
        let resource_id = arena_ids
            .entry(element_type)
            .or_insert(hashed_resource_id(
                "weight-arena",
                family_fingerprint,
                &serde_json::to_string(&element_type).map_err(|error| {
                    VNextError::Serialization {
                        context: "serialize weight arena element type",
                        message: error.to_string(),
                    }
                })?,
            )?)
            .clone();
        locations.insert(
            component.id.clone(),
            WeightComponentLocation {
                resource_id,
                offset_bytes,
                length_bytes,
                element_type,
            },
        );
    }
    Ok(locations)
}

fn referenced_weight_components(
    family: &PreparedModelFamily,
) -> Result<Vec<&super::super::WeightComponentSpec>, VNextError> {
    let referenced = family
        .program()
        .weights()
        .iter()
        .map(|weight| {
            family
                .weight_schema()
                .physical_component_refs(&weight.weight_id)
        })
        .collect::<Result<Vec<_>, VNextError>>()?
        .into_iter()
        .flatten()
        .map(|component| component.id.clone())
        .collect::<BTreeSet<_>>();
    Ok(family
        .weight_schema()
        .components
        .iter()
        .filter(|component| referenced.contains(&component.id))
        .collect())
}

fn checked_align_up(value: u64, alignment: u64) -> Result<u64, VNextError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_plan("weight arena alignment is invalid"));
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| invalid_plan("weight arena alignment overflows u64"))
}

fn hashed_resource_id(
    kind: &str,
    family_fingerprint: &str,
    semantic_identity: &str,
) -> Result<ResourceId, VNextError> {
    let digest =
        Sha256::digest(format!("{kind}\0{family_fingerprint}\0{semantic_identity}").as_bytes());
    ResourceId::new(format!("resource/{kind}/sha256/{digest:x}"))
}

fn build_value_storages(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    tensors: &BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    weight_locations: &BTreeMap<WeightId, WeightComponentLocation>,
    family_fingerprint: &str,
) -> Result<BTreeMap<ProgramValueId, ResolvedValueStorage>, VNextError> {
    let program = family.program();
    let mut storages = BTreeMap::new();
    for input in program.inputs() {
        storages.insert(
            input.clone(),
            activation_storage(input, tensors, family_fingerprint)?,
        );
    }
    for weight in program.weights() {
        let components = family
            .weight_schema()
            .physical_component_refs(&weight.weight_id)?
            .into_iter()
            .map(|component| {
                let location = weight_locations.get(&component.id).ok_or_else(|| {
                    invalid_plan(format!(
                        "weight component `{}` has no physical location",
                        component.id
                    ))
                })?;
                ResolvedStorageComponent::new(
                    Some(component.id.clone()),
                    location.resource_id.clone(),
                    location.offset_bytes,
                    location.length_bytes,
                    location.element_type,
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        storages.insert(
            weight.value_id.clone(),
            ResolvedValueStorage::composite(components)?,
        );
    }
    for state in program.states() {
        let tensor = tensors
            .get(&state.value_id)
            .ok_or_else(|| invalid_plan(format!("state `{}` has no resolved tensor", state.id)))?;
        let tensor_minimum_bytes = tensor.minimum_storage_bytes()?;
        let state_minimum_bytes = checked_align_up(
            state.capacity_demand.minimum_bytes(tensor_minimum_bytes)?,
            tensor.element_type().size_bytes(),
        )?;
        storages.insert(
            state.value_id.clone(),
            ResolvedValueStorage::single(
                hashed_resource_id("state", family_fingerprint, state.id.as_str())?,
                0,
                state_minimum_bytes,
                tensor.element_type(),
            )?,
        );
    }
    for node in program.blocks().iter().flat_map(|block| &block.nodes) {
        let operation = catalog.operation(&node.operation_id)?;
        for (value_id, contract) in node.outputs.iter().zip(&operation.outputs) {
            let storage = match contract.alias() {
                AliasPolicy::MustAlias { tensor_index } => {
                    let input = node.inputs.get(*tensor_index as usize).ok_or_else(|| {
                        invalid_plan(format!("node `{}` aliases an absent input", node.id))
                    })?;
                    storages.get(input).cloned().ok_or_else(|| {
                        invalid_plan(format!(
                            "node `{}` alias input `{input}` has no storage",
                            node.id
                        ))
                    })?
                }
                AliasPolicy::NoAlias | AliasPolicy::MayAlias { .. } => {
                    activation_storage(value_id, tensors, family_fingerprint)?
                }
            };
            storages.insert(value_id.clone(), storage);
        }
    }
    Ok(storages)
}

fn activation_storage(
    value_id: &ProgramValueId,
    tensors: &BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    family_fingerprint: &str,
) -> Result<ResolvedValueStorage, VNextError> {
    let tensor = tensors
        .get(value_id)
        .ok_or_else(|| invalid_plan(format!("activation `{value_id}` has no resolved tensor")))?;
    ResolvedValueStorage::single(
        hashed_resource_id("activation", family_fingerprint, value_id.as_str())?,
        0,
        tensor.minimum_storage_bytes()?,
        tensor.element_type(),
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_nodes<P: RuntimePolicy>(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    policy: &P,
    planning: &OperationPlanningHandle<'_>,
    options: &ProgramPlanCompileOptions,
    tensors: &BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    storages: &BTreeMap<ProgramValueId, ResolvedValueStorage>,
) -> Result<Vec<PlanNodeResolution>, VNextError> {
    family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .map(|node| {
            let operation = catalog.operation(&node.operation_id)?;
            let values = node
                .inputs
                .iter()
                .zip(&operation.inputs)
                .enumerate()
                .map(|(ordinal, (value_id, contract))| {
                    resolved_binding(
                        family,
                        value_id,
                        ResolvedValueRole::Input,
                        ordinal as u32,
                        contract,
                        tensors,
                        storages,
                    )
                })
                .chain(node.outputs.iter().zip(&operation.outputs).enumerate().map(
                    |(ordinal, (value_id, contract))| {
                        resolved_binding(
                            family,
                            value_id,
                            ResolvedValueRole::Output,
                            ordinal as u32,
                            contract,
                            tensors,
                            storages,
                        )
                    },
                ))
                .collect::<Result<Vec<_>, VNextError>>()?;
            PlanNodeResolution::resolve(
                family,
                catalog,
                policy,
                planning,
                node.id.clone(),
                values,
                options
                    .required_capabilities
                    .get(&node.id)
                    .cloned()
                    .unwrap_or_default(),
                options.preferred_providers.get(&node.id).cloned(),
            )
        })
        .collect()
}

fn resolved_binding(
    family: &PreparedModelFamily,
    value_id: &ProgramValueId,
    role: ResolvedValueRole,
    ordinal: u32,
    contract: &TensorContract,
    tensors: &BTreeMap<ProgramValueId, ResolvedTensorSpec>,
    storages: &BTreeMap<ProgramValueId, ResolvedValueStorage>,
) -> Result<ResolvedValueBinding, VNextError> {
    let usage = if family
        .program()
        .weights()
        .iter()
        .any(|weight| weight.value_id == *value_id)
    {
        BufferUsage::Weights
    } else if family
        .program()
        .states()
        .iter()
        .any(|state| state.value_id == *value_id)
    {
        BufferUsage::State
    } else {
        BufferUsage::Activations
    };
    ResolvedValueBinding::new(
        value_id.clone(),
        role,
        ordinal,
        tensors
            .get(value_id)
            .cloned()
            .ok_or_else(|| invalid_plan(format!("value `{value_id}` has no resolved tensor")))?,
        contract.access(),
        if role == ResolvedValueRole::Input {
            AliasPolicy::NoAlias
        } else {
            contract.alias().clone()
        },
        usage,
        storages
            .get(value_id)
            .cloned()
            .ok_or_else(|| invalid_plan(format!("value `{value_id}` has no resolved storage")))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_and_stride_symbols_are_independent_domains() {
        let contract = TensorContract::new(
            vec![DimensionConstraint::Symbol("shared".to_owned())],
            BTreeSet::from([ElementType::F32]),
            vec![LayoutConstraint::Strided {
                strides: vec![StrideConstraint::Symbol("shared".to_owned())],
            }],
            TensorAccess::Read,
            AliasPolicy::NoAlias,
        )
        .unwrap();
        let tensor = ResolvedTensorSpec::new(
            vec![4],
            ElementType::F32,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![16],
            },
        )
        .unwrap();
        let mut symbols = TensorSymbols::default();
        let value_id = ProgramValueId::new("value.symbol-domains").unwrap();
        unify_tensor(
            &contract,
            &tensor,
            &mut symbols,
            &NodeId::new("node.symbol-domains").unwrap(),
            "input",
            0,
            &value_id,
        )
        .unwrap();
        assert_eq!(symbols.dimensions.get("shared"), Some(&4));
        assert_eq!(symbols.strides.get("shared"), Some(&16));
    }

    #[test]
    fn ambiguous_output_range_or_layout_requires_an_explicit_tensor() {
        let range = TensorContract::new(
            vec![DimensionConstraint::Range {
                minimum: 1,
                maximum: 8,
            }],
            BTreeSet::from([ElementType::F32]),
            vec![LayoutConstraint::Contiguous],
            TensorAccess::Write,
            AliasPolicy::NoAlias,
        )
        .unwrap();
        let mut symbols = TensorSymbols::default();
        let node_id = NodeId::new("node.ambiguous").unwrap();
        assert!(infer_tensor(&range, &mut symbols, &node_id).is_err());

        let layouts = TensorContract::new(
            vec![DimensionConstraint::Exact(4)],
            BTreeSet::from([ElementType::F32]),
            vec![
                LayoutConstraint::Contiguous,
                LayoutConstraint::Strided {
                    strides: vec![StrideConstraint::ExactBytes(4)],
                },
            ],
            TensorAccess::Write,
            AliasPolicy::NoAlias,
        )
        .unwrap();
        assert!(infer_tensor(&layouts, &mut symbols, &node_id).is_err());
    }
}
