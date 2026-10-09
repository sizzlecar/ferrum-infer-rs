//! Cold upstream facts without an Invocation/Step/Sequence allocation owner.
use super::*;

pub(in crate::backend::cuda::vnext_ops::transformer) struct PlanOnlyRecipe {
    runtime: Arc<Runtime>,
    numerics: PreparedProjectionNumerics,
    leaves: Vec<PlanOnlyLeaf>,
}

struct PlanOnlyLeaf {
    facts: UpstreamProjectionWaveFacts,
    input_offset: u64,
    output_offset: u64,
    weight: CudaBufferRegion,
    flag: Option<CudaBufferRegion>,
    input_ordinal: u32,
    flag_offset: Option<u64>,
    dependency: Option<ferrum_interfaces::vnext::SteadyRecipeDependency>,
}

impl PreparedAttentionProjections<'_> {
    /// Called only after the original upstream preparation has succeeded.
    /// The scratch owner is used to derive offsets and is never retained.
    pub(in crate::backend::cuda::vnext_ops::transformer) fn plan_only_recipe(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        scratch: &CudaBufferRegion,
    ) -> Result<Option<PlanOnlyRecipe>, String> {
        let Some(state) = &self.upstream else {
            return Ok(None);
        };
        // This schema has no transform/sign selector. Legitimate transformed
        // providers retain the original path instead of an incomplete recipe.
        if state
            .launches
            .iter()
            .flat_map(|projection| &projection.leaves)
            .any(|leaf| leaf.signs.is_some() || leaf.part.transform.is_some())
        {
            return Ok(None);
        }
        let persistent = persistent_region(invocation, flag_bytes(&self.numerics)?)?;
        // A cold recipe may keep Plan allocations alive, never dynamic scratch.
        persistent
            .plan_backing_identity()
            .map_err(|e| e.to_string())?;
        let mut leaves = Vec::new();
        for projection in &state.launches {
            let prepared = self.projection(projection.role)?;
            let first_leaf = self
                .numerics
                .projections()
                .iter()
                .take_while(|p| p.role() != projection.role)
                .map(|p| p.leaves().len() as u64)
                .sum();
            let input_offset = projection
                .input
                .backing_byte_offset()
                .checked_sub(scratch.backing_byte_offset())
                .ok_or("cold upstream input precedes its scratch")?;
            let output_offset = projection
                .output
                .backing_byte_offset()
                .checked_sub(scratch.backing_byte_offset())
                .ok_or("cold upstream output precedes its scratch")?;
            for (index, leaf) in projection.leaves.iter().enumerate() {
                leaf.weight
                    .plan_backing_identity()
                    .map_err(|e| e.to_string())?;
                let flag_offset = leaf
                    .stage
                    .native
                    .as_ref()
                    .map(|native| {
                        flag_offset(first_leaf, index as u64, native.geometry().algorithm)
                    })
                    .transpose()?;
                if flag_offset.is_some() != leaf.validation.is_some() {
                    return Err("cold upstream validation differs from its native leaf".into());
                }
                let flag = flag_offset
                    .map(|offset| persistent.subregion(offset, 4).map_err(|e| e.to_string()))
                    .transpose()?;
                leaves.push(PlanOnlyLeaf {
                    facts: UpstreamProjectionWaveFacts {
                        role: projection.role,
                        component_id: leaf.part.component_id.clone(),
                        local_rows: projection.rows,
                        layout: UpstreamProjectionLayout::Columns,
                        input_stride: prepared.input_features(),
                        output_stride: prepared.output_features(),
                        input_byte_offset: projection.input.backing_byte_offset(),
                        output_byte_offset: projection.output.backing_byte_offset(),
                        weight_byte_offset: leaf.weight.backing_byte_offset(),
                        input_available_bytes: projection.input.length_bytes(),
                        output_available_bytes: projection.output.length_bytes(),
                        weight_available_bytes: leaf.weight.length_bytes(),
                        retained_zero_padded_weight_rows: u64::from(leaf.part.rows),
                    },
                    input_offset,
                    output_offset,
                    weight: leaf.weight.clone(),
                    flag,
                    input_ordinal: prepared.weight_input_ordinal(),
                    flag_offset,
                    dependency: leaf.validation.as_ref().zip(flag_offset).map(
                        |(validation, offset)| {
                            validation.steady_dependency_declaration(
                                prepared.weight_input_ordinal(),
                                &leaf.part.component_id,
                                offset,
                            )
                        },
                    ),
                });
            }
        }
        Ok(Some(PlanOnlyRecipe {
            runtime: Arc::clone(&state.runtime),
            numerics: self.numerics.as_ref().clone(),
            leaves,
        }))
    }
}

impl PlanOnlyRecipe {
    /// Static selectors only; the caller resolves them from the fresh checked
    /// table before invoking any native service or issuing dependency authority.
    pub(in crate::backend::cuda::vnext_ops::transformer) fn leaf_selectors(
        &self,
    ) -> impl Iterator<
        Item = (
            u32,
            &ferrum_interfaces::vnext::WeightId,
            u64,
            ElementType,
            Option<u64>,
        ),
    > {
        self.leaves.iter().map(|leaf| {
            (
                leaf.input_ordinal,
                &leaf.facts.component_id,
                leaf.facts.weight_available_bytes,
                leaf.weight.element_type(),
                leaf.flag_offset,
            )
        })
    }

    pub(in crate::backend::cuda::vnext_ops::transformer) fn dependency(
        &self,
        index: usize,
    ) -> Option<&ferrum_interfaces::vnext::SteadyRecipeDependency> {
        self.leaves.get(index)?.dependency.as_ref()
    }

    pub(in crate::backend::cuda::vnext_ops::transformer) fn dependencies(
        &self,
    ) -> impl Iterator<Item = &ferrum_interfaces::vnext::SteadyRecipeDependency> {
        self.leaves
            .iter()
            .filter_map(|leaf| leaf.dependency.as_ref())
    }

    /// All callbacks into mutable native services happen after the core permit
    /// has unlocked. Cached geometry does not authorize skipping their poison
    /// and identity/conflict checks.
    pub(in crate::backend::cuda::vnext_ops::transformer) fn refresh_services(
        &self,
        scratch: &CudaBufferRegion,
        regions: &[(CudaBufferRegion, Option<CudaBufferRegion>)],
    ) -> Result<Vec<(usize, Arc<WeightValidation>)>, String> {
        if regions.len() != self.leaves.len() {
            return Err("fresh upstream leaf coverage differs from its cold schema".into());
        }
        let _estimate = self.runtime.scratch_bytes(&self.numerics)?;
        let mut validations = Vec::new();
        for (index, (leaf, (weight, flag))) in self.leaves.iter().zip(regions).enumerate() {
            if weight.plan_backing_identity().map_err(|e| e.to_string())?
                != leaf
                    .weight
                    .plan_backing_identity()
                    .map_err(|e| e.to_string())?
                || flag
                    .as_ref()
                    .map(CudaBufferRegion::plan_backing_identity)
                    .transpose()
                    .map_err(|e| e.to_string())?
                    != leaf
                        .flag
                        .as_ref()
                        .map(CudaBufferRegion::plan_backing_identity)
                        .transpose()
                        .map_err(|e| e.to_string())?
            {
                return Err("fresh upstream Plan owner differs from the cold recipe".into());
            }
            let input = scratch
                .subregion(leaf.input_offset, leaf.facts.input_available_bytes)
                .map_err(|e| e.to_string())?;
            let output = scratch
                .subregion(leaf.output_offset, leaf.facts.output_available_bytes)
                .map_err(|e| e.to_string())?;
            let mut facts = leaf.facts.clone();
            facts.input_byte_offset = input.backing_byte_offset();
            facts.output_byte_offset = output.backing_byte_offset();
            facts.weight_byte_offset = weight.backing_byte_offset();
            facts.weight_available_bytes = weight.length_bytes();
            if facts != leaf.facts {
                return Err(
                    "fresh upstream physical geometry differs from the resident recipe".into(),
                );
            }
            let stage = self.runtime.plans.prepare_for(
                &self.numerics,
                &facts,
                ProjectionPreparation::BindingsOnly,
            )?;
            match (&stage.native, flag) {
                (Some(native), Some(flag)) => {
                    let validation = self
                        .runtime
                        .validation
                        .prepare(
                            native.clone(),
                            &self.runtime.fingerprint,
                            weight.clone(),
                            flag.clone(),
                        )
                        .map_err(|e| e.to_string())?;
                    validations.push((index, validation));
                }
                (None, None) => {}
                _ => {
                    return Err(
                        "fresh upstream validation bank differs from native selection".into(),
                    )
                }
            }
        }
        Ok(validations)
    }
}
