use super::*;

/// Borrowed family contracts become one graph, without a second executor or
/// an intermediate Host transfer. IDs already have disjoint family namespaces.
pub(super) struct Fragment {
    pub schema: WeightSchema,
    pub nodes: Vec<ProgramNode>,
    pub weights: Vec<WeightReference>,
    pub specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    pub operations: Vec<NumericalOperationContract>,
    pub boundaries: BTreeMap<ProgramValueId, ElementType>,
}
impl Fragment {
    pub fn family<F: ModelFamilyProvider>(
        family: &F,
        config: &F::Config,
        profile_id: &str,
        options: ProgramPlanCompileOptions,
        inputs: BTreeMap<ProgramValueId, ProgramValueId>,
    ) -> Result<Self, VNextError> {
        let ModelWeights::Stored(schema) = family.weights(config)? else {
            return Err(invalid("composition requires Stored family weights"));
        };
        let profiles = family.numerical_profiles(config)?;
        let profile = profiles.resolve(&NumericalProfileId::new(profile_id).map_err(invalid)?)?;
        let program = family.semantic_program(config, profile)?;
        if !program.states().is_empty()
            || !profile.states.is_empty()
            || !profile.kv_storage.is_empty()
            || program.inputs().iter().cloned().collect::<BTreeSet<_>>()
                != inputs.keys().cloned().collect()
        {
            return Err(invalid(
                "fragment must be stateless with every input explicitly mapped",
            ));
        }
        let lookup = |id: &ProgramValueId| inputs.get(id).cloned().unwrap_or_else(|| id.clone());
        let nodes = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| {
                if n.work != ProgramNodeWorkSpec::Fixed {
                    return Err(invalid("fragment has non-fixed work"));
                }
                Ok(ProgramNode {
                    id: n.id.clone(),
                    operation_id: n.operation_id.clone(),
                    required_version: n.required_version,
                    work: n.work.clone(),
                    inputs: n.inputs.iter().map(&lookup).collect(),
                    outputs: n.outputs.clone(),
                    attributes: n.attributes.clone(),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let required = program
            .inputs()
            .iter()
            .chain(nodes.iter().flat_map(|n| &n.outputs))
            .collect::<BTreeSet<_>>();
        if required != options.tensor_specs().keys().collect() {
            return Err(invalid("fragment compile geometry is incomplete"));
        }
        let specs = options
            .tensor_specs()
            .iter()
            .map(|(id, s)| (lookup(id), s.clone()))
            .collect();
        Ok(Self {
            schema,
            nodes,
            weights: program.weights().to_vec(),
            specs,
            operations: profile.operations.clone(),
            boundaries: profile.boundaries.clone(),
        })
    }
}
pub(super) fn mapping(
    pairs: &[(&str, &str)],
) -> Result<BTreeMap<ProgramValueId, ProgramValueId>, VNextError> {
    pairs
        .iter()
        .map(|(a, b)| Ok((ProgramValueId::new(*a)?, ProgramValueId::new(*b)?)))
        .collect()
}
