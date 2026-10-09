//! One resident upstream GDN address-table recipe. This owns Plan leaves only;
//! all state, scratch and binding regions come from the current core permit.
use super::*;
use ferrum_interfaces::vnext::{
    PreparedSteadyRecipePatch, SteadyRecipeDeclaration, SteadyRecipeRegionRequest,
    SteadyRecipeRegionSelector,
};

struct AttentionSteadyRecipe {
    upstream: super::super::q8act_attention::upstream::PlanOnlyRecipe,
    participants: usize,
    total_tokens: u64,
    leaves: Vec<(usize, Option<usize>)>,
}

const CONV: usize = 0;
const DELTA: usize = 1;
const BINDING: usize = 2;
const SCRATCH: usize = 3;

pub(super) fn declare_from_prepared(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    projections: Option<&PreparedAttentionProjections<'_>>,
    scratch: &CudaBufferRegion,
    _required_scratch_bytes: u64,
    required_binding_bytes: u64,
) -> Result<Option<SteadyRecipeDeclaration>, String> {
    let Some(upstream) = projections
        .map(|projections| projections.plan_only_recipe(invocation, scratch))
        .transpose()?
        .flatten()
    else {
        return Ok(None);
    };
    let first = &invocation.participants()[0];
    let conv = contiguous_region(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 8)?,
        ElementType::F16,
    )?;
    let delta = contiguous_region(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 9)?,
        ElementType::F32,
    )?;
    let destination = super::super::shared_binding_region(invocation, required_binding_bytes)?;
    let request =
        |selector, length_bytes, element_type, alignment_bytes| SteadyRecipeRegionRequest {
            selector,
            offset_bytes: 0,
            length_bytes,
            element_type,
            alignment_bytes,
        };
    let mut regions = vec![
        request(
            SteadyRecipeRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 8,
                component: None,
            },
            conv.length_bytes(),
            ElementType::F16,
            2,
        ),
        request(
            SteadyRecipeRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 9,
                component: None,
            },
            delta.length_bytes(),
            ElementType::F32,
            4,
        ),
        request(
            SteadyRecipeRegionSelector::ProgramBinding,
            destination.length_bytes(),
            ElementType::U8,
            1,
        ),
        request(
            SteadyRecipeRegionSelector::Scratch,
            scratch.length_bytes(),
            ElementType::U8,
            1,
        ),
    ];
    let mut leaves = Vec::new();
    for (ordinal, component, length, element_type, flag_offset) in upstream.leaf_selectors() {
        let weight_index = regions.len();
        regions.push(request(
            SteadyRecipeRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal,
                component: Some(component.clone()),
            },
            length,
            element_type,
            4,
        ));
        let flag_index = flag_offset.map(|offset_bytes| {
            let index = regions.len();
            regions.push(SteadyRecipeRegionRequest {
                selector: SteadyRecipeRegionSelector::Persistent,
                offset_bytes,
                length_bytes: 4,
                element_type: ElementType::U8,
                alignment_bytes: 4,
            });
            index
        });
        leaves.push((weight_index, flag_index));
    }
    let dependencies = upstream.dependencies().cloned().collect();
    SteadyRecipeDeclaration::new(
        regions,
        dependencies,
        Arc::new(AttentionSteadyRecipe {
            upstream,
            participants: invocation.participants().len(),
            total_tokens: invocation.work_shape().immediate_tokens(),
            leaves,
        }),
    )
    .map(Some)
    .map_err(|e| e.to_string())
}

fn region(
    patch: &PreparedSteadyRecipePatch<'_, CudaDeviceBuffer>,
    participant: usize,
    index: usize,
) -> Result<CudaBufferRegion, String> {
    let region = patch
        .region(participant, index)
        .map_err(|e| e.to_string())?
        .physical_region();
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())
}

fn shared_region(
    patch: &PreparedSteadyRecipePatch<'_, CudaDeviceBuffer>,
    index: usize,
) -> Result<CudaBufferRegion, String> {
    let first = region(patch, 0, index)?;
    for participant in 1..patch.participant_count() {
        let next = region(patch, participant, index)?;
        if !super::super::same_physical_region(&first, &next) {
            return Err("steady attention shared region differs across participants".into());
        }
    }
    Ok(first)
}

pub(super) fn encode(
    patch: PreparedSteadyRecipePatch<'_, CudaDeviceBuffer>,
) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, String> {
    let recipe = patch
        .cold_state::<AttentionSteadyRecipe>()
        .map_err(|e| e.to_string())?;
    if patch.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("steady attention participant domain differs from its cold schema".into());
    }
    let destination = shared_region(&patch, BINDING)?;
    let scratch = shared_region(&patch, SCRATCH)?;
    let mut leaf_regions = Vec::with_capacity(recipe.leaves.len());
    for &(weight, flag) in &recipe.leaves {
        leaf_regions.push((
            shared_region(&patch, weight)?,
            flag.map(|index| shared_region(&patch, index)).transpose()?,
        ));
    }
    let validations = recipe.upstream.refresh_services(&scratch, &leaf_regions)?;
    let mut dependencies = Vec::with_capacity(validations.len());
    for (index, validation) in validations {
        let declaration = recipe
            .upstream
            .dependency(index)
            .ok_or("fresh upstream validation lacks a declared Plan dependency")?;
        dependencies.push(
            validation
                .steady_retained_dependency(&patch, declaration)
                .map_err(|e| e.to_string())?,
        );
    }
    let layout = StateBindingLayout::new(recipe.participants)?;
    let mut writes = Vec::with_capacity(recipe.participants);
    let mut fence_dependencies = Vec::with_capacity(recipe.participants.saturating_mul(2));
    for participant in 0..recipe.participants {
        let conv = region(&patch, participant, CONV)?;
        let delta = region(&patch, participant, DELTA)?;
        writes.push(
            super::super::CudaProgramBindingWrite::new(
                layout.offset(participant)?,
                state_binding_payload(&conv, &delta),
            )
            .map_err(|e| e.to_string())?,
        );
        fence_dependencies.push(conv);
        fence_dependencies.push(delta);
    }
    let participant_count = u32::try_from(recipe.participants)
        .map_err(|_| "steady attention participant count exceeds u32")?;
    let command = CudaDeviceCommand::program_binding_patch(
        "vnext_gated_delta_recurrent_attention_bindings",
        patch.program_binding().clone(),
        destination,
        writes,
        fence_dependencies,
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participant_count,
            recipe.total_tokens,
            0,
            u64::from(participant_count),
        )
    })
    .map_err(|e| e.to_string())?;
    Ok(dependencies.into_iter().fold(
        EncodedReusableExecutionBindings::empty().with_program_binding(command),
        |encoded, dependency| encoded.with_retained_plan_dependency(dependency),
    ))
}
