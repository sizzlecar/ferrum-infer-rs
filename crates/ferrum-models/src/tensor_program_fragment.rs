//! Stateless family composition. This only rewrites semantic IDs: physical
//! checkpoint identities, numerical contracts and allocation ownership survive.
//! Profile-dependent schema specialization requires a prepared-family consumer;
//! this subset rejects it instead of silently composing the unspecialized graph.
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

mod prepared;
mod schedule;
mod stored;
pub use schedule::TensorProgramSchedule;
pub(crate) use stored::{StoredComposition, StoredCompositionIdentity, StoredCompositionLimits};

pub(crate) struct Fragment {
    pub schema: Option<WeightSchema>,
    pub nodes: Vec<ProgramNode>,
    pub weights: Vec<WeightReference>,
    pub specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    pub operations: Vec<NumericalOperationContract>,
    pub boundaries: BTreeMap<ProgramValueId, ElementType>,
}
impl Fragment {
    /// Families with disjoint IDs need only explicit input bindings and tensor
    /// geometry. Provider selection, materialization and retention policy belong
    /// to the final composed plan, not to an independently compiled fragment.
    pub fn family<F: ModelFamilyProvider>(
        family: &F,
        config: &F::Config,
        profile_id: &str,
        geometry: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
        inputs: BTreeMap<ProgramValueId, ProgramValueId>,
    ) -> Result<Self, VNextError> {
        Self::family_instance(family, config, profile_id, geometry, inputs, None)
    }

    /// Repeated instances namespace every internal value, node and logical
    /// weight. Input targets and physical component IDs stay unchanged.
    pub fn family_instance<F: ModelFamilyProvider>(
        family: &F,
        config: &F::Config,
        profile_id: &str,
        geometry: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
        inputs: BTreeMap<ProgramValueId, ProgramValueId>,
        instance: Option<&str>,
    ) -> Result<Self, VNextError> {
        if family.semantic_metadata(config)? != ModelSemanticMetadata::TensorComponent {
            return Err(invalid("requires a tensor component"));
        }
        let profiles = family.numerical_profiles(config)?;
        let profile = profiles.resolve(&NumericalProfileId::new(profile_id).map_err(invalid)?)?;
        let program = family.semantic_program(config, profile)?;
        profile.validate_program(&program)?;
        let schema = match family.weights(config)? {
            ModelWeights::Stored(schema) => {
                schema.validate(family.family_id())?;
                if family.specialize_weight_schema(config, &schema, profile)? != schema {
                    return Err(invalid(
                        "schema specialization requires a prepared-family fragment",
                    ));
                }
                schema.validate_program_references(family.family_id(), &program)?;
                Some(schema)
            }
            ModelWeights::None if program.weights().is_empty() => None,
            ModelWeights::None => return Err(invalid("weightless fragment declares weights")),
        };
        Self::mapped(program, profile, schema, geometry, inputs, instance)
    }

    /// Consume an already validated fixed Stored composition. Both entrances
    /// use the same identity, geometry and mapping checks; no graph cache or
    /// execution/ownership authority is introduced.
    pub fn composition(
        built: StoredComposition,
        inputs: BTreeMap<ProgramValueId, ProgramValueId>,
        instance: Option<&str>,
    ) -> Result<Self, VNextError> {
        built.schema.validate(&built.profile.family_id)?;
        built
            .schema
            .validate_program_references(&built.profile.family_id, &built.program)?;
        Self::mapped(
            built.program,
            &built.profile,
            Some(built.schema),
            built.options.tensor_specs(),
            inputs,
            instance,
        )
    }
    fn mapped(
        program: ModelProgram,
        profile: &NumericalExecutionProfile,
        mut schema: Option<WeightSchema>,
        geometry: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
        inputs: BTreeMap<ProgramValueId, ProgramValueId>,
        instance: Option<&str>,
    ) -> Result<Self, VNextError> {
        profile.validate_program(&program)?;
        if program.checkpoint_inputs().is_some() {
            return Err(invalid(
                "checkpoint input roles cannot be erased by composition",
            ));
        }
        if !program.states().is_empty()
            || !profile.states.is_empty()
            || !profile.kv_storage.is_empty()
            || program.inputs().iter().cloned().collect::<BTreeSet<_>>()
                != inputs.keys().cloned().collect()
        {
            return Err(invalid(
                "requires stateless work and every input explicitly bound",
            ));
        }
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
        if required != geometry.keys().collect() {
            return Err(invalid("compile geometry is incomplete"));
        }
        for (id, declared) in program.input_tensor_specs() {
            if geometry.get(id) != Some(declared) {
                return Err(invalid("compile geometry changes a declared input"));
            }
        }

        let lookup = |id: &ProgramValueId| -> Result<ProgramValueId, VNextError> {
            match inputs.get(id) {
                Some(target) => Ok(target.clone()),
                None => scoped_value(id, instance),
            }
        };
        // A binding must not silently replace an internal definition or another
        // input. Source operands may repeat, but definitions remain distinct.
        let mut definitions = BTreeSet::new();
        for id in required
            .iter()
            .copied()
            .chain(program.weights().iter().map(|w| &w.value_id))
        {
            if !definitions.insert(lookup(id)?) {
                return Err(invalid("input binding aliases another value definition"));
            }
        }
        let nodes = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| {
                if n.work != ProgramNodeWorkSpec::Fixed {
                    return Err(invalid("fragment has non-fixed work"));
                }
                Ok(ProgramNode {
                    id: NodeId::new(scoped(n.id.as_str(), instance)?)?,
                    operation_id: n.operation_id.clone(),
                    required_version: n.required_version,
                    work: n.work.clone(),
                    inputs: n.inputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                    outputs: n.outputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                    attributes: n.attributes.clone(),
                })
            })
            .collect::<Result<_, VNextError>>()?;
        let weights = program
            .weights()
            .iter()
            .map(|w| {
                Ok(WeightReference {
                    weight_id: WeightId::new(scoped(w.weight_id.as_str(), instance)?)?,
                    value_id: lookup(&w.value_id)?,
                    tensor: w.tensor.clone(),
                })
            })
            .collect::<Result<_, VNextError>>()?;
        if let Some(schema) = &mut schema {
            for tensor in &mut schema.tensors {
                tensor.id = WeightId::new(scoped(tensor.id.as_str(), instance)?)?;
            }
        }
        Ok(Self {
            schema,
            nodes,
            weights,
            specs: geometry
                .iter()
                .map(|(id, s)| Ok((lookup(id)?, s.clone())))
                .collect::<Result<_, VNextError>>()?,
            operations: profile.operations.clone(),
            boundaries: profile
                .boundaries
                .iter()
                .map(|(id, t)| Ok((lookup(id)?, *t)))
                .collect::<Result<_, VNextError>>()?,
        })
    }
}

pub(crate) fn scoped_value(
    id: &ProgramValueId,
    instance: Option<&str>,
) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(scoped(id.as_str(), instance)?)
}
fn scoped(id: &str, instance: Option<&str>) -> Result<String, VNextError> {
    match instance {
        None => Ok(id.to_owned()),
        Some(label)
            if !label.is_empty()
                && label
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_') =>
        {
            Ok(format!("{id}.{label}"))
        }
        _ => Err(invalid("instance must be a nonempty ASCII identifier")),
    }
}
pub(crate) fn mapping(
    pairs: &[(&str, &str)],
) -> Result<BTreeMap<ProgramValueId, ProgramValueId>, VNextError> {
    let mut inputs = BTreeMap::new();
    for (source, target) in pairs {
        if inputs
            .insert(ProgramValueId::new(*source)?, ProgramValueId::new(*target)?)
            .is_some()
        {
            return Err(invalid("duplicate input binding"));
        }
    }
    Ok(inputs)
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("tensor program fragment: {e}"),
    }
}

#[cfg(test)]
mod tests;
