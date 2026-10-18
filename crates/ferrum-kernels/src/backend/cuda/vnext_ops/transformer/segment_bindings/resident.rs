//! One cold lowering of the same closed binding rules used by ordinary replay.
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaBindingProgramSink, CudaPhysicalOwnerBuilder, ResidentColdParts,
};
use ferrum_runtime::PreparedPhysicalSegmentWave;

pub(crate) fn compile(
    program: &(dyn std::any::Any + Send + Sync),
    wave: &PreparedPhysicalSegmentWave<'_, CudaDeviceBuffer>,
) -> Result<Option<ResidentColdParts>, CudaDeviceRuntimeError> {
    let invalid = |reason: &str| CudaDeviceRuntimeError::contract(reason);
    let Some(program) = program.downcast_ref::<CudaSegmentBindingProgram>() else {
        return Ok(None);
    };
    if program.nodes.len() != wave.node_count() {
        return Err(invalid("resident cold node coverage differs"));
    }
    let mut owners = CudaPhysicalOwnerBuilder::new();
    let mut sink = CudaBindingProgramSink::new();
    let mut controls = Vec::new();
    let mut dependency_identities = Vec::new();
    let mut dependencies = Vec::new();
    for (ordinal, (node_index, recipe)) in program.nodes.iter().enumerate() {
        let node = wave
            .node(ordinal)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        if node.node_index() != *node_index
            || wave.dependency_count(ordinal) != Some(recipe.validations.len())
        {
            return Err(invalid(
                "resident cold declaration differs from physical wave",
            ));
        }
        match &recipe.dynamic {
            DynamicRecipe::PlanDependencies => (),
            DynamicRecipe::GatedDelta(recipe) => {
                super::super::attention::segment_bindings::write_physical_program(
                    &node,
                    recipe,
                    &mut owners,
                    &mut sink,
                )
                .map_err(CudaDeviceRuntimeError::contract)?;
            }
            DynamicRecipe::Causal(recipe) => {
                let Some(rules) =
                    super::super::causal_attention::segment_bindings::resident_control_rules(
                        &node, recipe,
                    )?
                else {
                    return Ok(None);
                };
                controls.extend(rules);
                super::super::causal_attention::segment_bindings::write_resident_physical_program(
                    &node,
                    recipe,
                    &mut owners,
                    &mut sink,
                )
                .map_err(CudaDeviceRuntimeError::contract)?;
            }
        }
        for (index, validation) in recipe.validations.iter().enumerate() {
            let weight = physical::shared_region(&node, recipe.dependency_region_start + index * 2)
                .map_err(CudaDeviceRuntimeError::contract)?;
            let flag =
                physical::shared_region(&node, recipe.dependency_region_start + index * 2 + 1)
                    .map_err(CudaDeviceRuntimeError::contract)?;
            validation.state.validate_segment_owners(&weight, &flag)?;
            let identity = wave
                .dependency_identity(ordinal, index)
                .ok_or_else(|| invalid("resident cold dependency identity is absent"))?;
            if !dependency_identities.contains(identity) {
                dependency_identities.push(identity.clone());
                dependencies.push(validation.state.dynamic_command()?);
            }
        }
    }
    Ok(Some(ResidentColdParts {
        sink,
        owners: owners.finish(),
        controls,
        dependency_identities,
        dependencies,
    }))
}
