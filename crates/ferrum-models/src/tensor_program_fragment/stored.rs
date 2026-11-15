//! One stateless graph and one Stored schema, with physical components shared
//! only when every declaration and interpretation is identical. Identical
//! logical uses converge on one weight value; Core's single-layout ownership
//! rule remains unchanged.
use super::*;

pub(crate) struct StoredCompositionIdentity {
    pub family: ModelFamilyId,
    pub profile: NumericalProfileId,
    pub format: WeightFormatId,
    pub layout: WeightLayoutId,
    pub block: String,
}
#[derive(Clone, Copy)]
pub(crate) struct StoredCompositionLimits {
    pub maximum_nodes: usize,
    pub maximum_stored_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
pub(crate) struct StoredComposition {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
impl StoredComposition {
    pub fn join(
        identity: StoredCompositionIdentity,
        fragments: Vec<Fragment>,
        inputs: Vec<ProgramValueId>,
        outputs: Vec<ProgramValueId>,
        primary: ProgramValueId,
        limits: StoredCompositionLimits,
    ) -> Result<Self, VNextError> {
        if limits.maximum_nodes == 0
            || limits.maximum_stored_bytes == 0
            || limits.maximum_metadata_bytes == 0
        {
            return Err(invalid("composition capacities must be positive"));
        }
        let mut components = BTreeMap::new();
        let mut tensors = vec![];
        let mut nodes = vec![];
        let mut weights = vec![];
        let mut specs = BTreeMap::new();
        let mut boundaries = BTreeMap::new();
        let mut operations = BTreeMap::new();
        for fragment in fragments {
            if let Some(part) = fragment.schema {
                if part.format_id != identity.format || part.version != ContractVersion::new(1, 0) {
                    return Err(invalid("joined source format/version differs"));
                }
                for c in part.components {
                    if components.get(&c.id).is_some_and(|old| old != &c) {
                        return Err(invalid("shared physical component declaration differs"));
                    }
                    components.insert(c.id.clone(), c);
                }
                tensors.extend(part.tensors);
            }
            nodes.extend(fragment.nodes);
            weights.extend(fragment.weights);
            if nodes.len() > limits.maximum_nodes {
                return Err(VNextError::ModelProgramNodeCapacityExceeded {
                    required_nodes: nodes.len() as u64,
                    maximum_nodes: limits.maximum_nodes as u64,
                });
            }
            for (id, spec) in fragment.specs {
                if specs.get(&id).is_some_and(|old| old != &spec) {
                    return Err(invalid("joined activation geometry differs"));
                }
                specs.insert(id, spec);
            }
            for (id, dtype) in fragment.boundaries {
                if boundaries.insert(id, dtype).is_some() {
                    return Err(invalid("joined activation has multiple producers"));
                }
            }
            for op in fragment.operations {
                if operations
                    .get(&op.operation_id)
                    .is_some_and(|old| old != &op)
                {
                    return Err(invalid("joined numerical arithmetic differs"));
                }
                operations.insert(op.operation_id.clone(), op);
            }
        }
        let (tensors, weights) = share_identical_weights(tensors, weights, &mut nodes)?;
        let schema = WeightSchema {
            format_id: identity.format,
            layout_id: identity.layout,
            version: ContractVersion::new(1, 0),
            components: components.into_values().collect(),
            tensors,
        };
        let stored = schema.components.iter().try_fold(0u64, |n, c| {
            n.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("weight extent overflows"))
        })?;
        if stored > limits.maximum_stored_bytes {
            return Err(invalid("composed source exceeds capacity"));
        }
        schema.validate(&identity.family)?;
        let mut input_specs = BTreeMap::new();
        for id in &inputs {
            let spec = specs
                .get(id)
                .ok_or_else(|| invalid("missing composed input geometry"))?;
            if input_specs.insert(id.clone(), spec.clone()).is_some() {
                return Err(invalid("duplicate composed input"));
            }
        }
        let program = ModelProgram::new(
            identity.family.clone(),
            inputs,
            vec![ProgramBlock {
                id: identity.block,
                nodes,
            }],
            vec![],
            weights,
            outputs,
        )?
        .with_input_tensor_specs(input_specs)?;
        schema.validate_program_references(&identity.family, &program)?;
        let profile = NumericalExecutionProfile {
            id: identity.profile,
            family_id: identity.family,
            version: ContractVersion::new(1, 0),
            primary_activation: primary,
            boundaries,
            states: vec![],
            kv_storage: vec![],
            operations: operations.into_values().collect(),
        };
        profile.validate_program(&program)?;
        let required = program
            .inputs()
            .iter()
            .chain(
                program
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .flat_map(|n| &n.outputs),
            )
            .collect::<BTreeSet<_>>();
        if specs.keys().collect::<BTreeSet<_>>() != required {
            return Err(invalid("composed geometry is incomplete"));
        }
        if serde_json::to_vec(&(&schema, &program, &profile, &specs))
            .map_err(invalid)?
            .len()
            > limits.maximum_metadata_bytes
        {
            return Err(invalid("composed graph exceeds metadata capacity"));
        }
        Ok(Self {
            schema,
            program,
            profile,
            options: ProgramPlanCompileOptions::new(specs)?,
        })
    }
}

fn share_identical_weights(
    tensors: Vec<WeightTensorSpec>,
    weights: Vec<WeightReference>,
    nodes: &mut [ProgramNode],
) -> Result<(Vec<WeightTensorSpec>, Vec<WeightReference>), VNextError> {
    let mut logical = BTreeMap::new();
    let mut canonical: Vec<WeightTensorSpec> = vec![];
    for tensor in tensors {
        let identical = canonical.iter().find(|old| {
            old.dimensions == tensor.dimensions
                && old.logical_element_type == tensor.logical_element_type
                && old.physical_layout == tensor.physical_layout
                && old.required == tensor.required
        });
        let target = identical
            .map(|old| old.id.clone())
            .unwrap_or_else(|| tensor.id.clone());
        if logical.insert(tensor.id.clone(), target).is_some() {
            return Err(invalid("joined logical weight is defined twice"));
        }
        if identical.is_none() {
            canonical.push(tensor);
        }
    }
    let mut references = BTreeMap::<WeightId, WeightReference>::new();
    let mut seen_values = BTreeSet::new();
    let mut aliases = BTreeMap::new();
    for mut weight in weights {
        if !seen_values.insert(weight.value_id.clone()) {
            return Err(invalid("joined weight value is defined twice"));
        }
        let target = logical
            .get(&weight.weight_id)
            .ok_or_else(|| invalid("joined weight lacks a declaration"))?;
        if let Some(previous) = references.get(target) {
            if previous.tensor != weight.tensor {
                return Err(invalid("shared weight value geometry differs"));
            }
            aliases.insert(weight.value_id, previous.value_id.clone());
        } else {
            weight.weight_id = target.clone();
            references.insert(target.clone(), weight);
        }
    }
    for node in nodes {
        for input in &mut node.inputs {
            if let Some(target) = aliases.get(input) {
                *input = target.clone();
            }
        }
    }
    Ok((canonical, references.into_values().collect()))
}
