use super::*;

impl MemoryPlan {
    /// Rebuild the older, larger layout from trusted current descriptors. Only
    /// the choice of a ConservativeConcurrent pool is read from the declared
    /// plan; no untrusted byte count, slot, membership or policy is adopted.
    pub(in crate::execution) fn with_legacy_invocation_layouts(
        mut self,
        declared: &[DynamicBackingPoolSpec],
        node_count: usize,
    ) -> Result<Self, VNextError> {
        if !self.dynamic_pools.iter().any(|pool| {
            pool.invocation_liveness_mode == InvocationLivenessMode::DependencyChainReuse
                && declared.iter().any(|actual| {
                    actual.pool_id == pool.pool_id
                        && actual.invocation_liveness_mode
                            == InvocationLivenessMode::ConservativeConcurrent
                })
        }) {
            return Ok(self);
        }
        let dynamic_capacity = self
            .usable_capacity_bytes
            .checked_sub(self.static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        Self::preserve_legacy_invocation_layouts(
            &mut self.dynamic_pools,
            declared,
            &self.dynamic_descriptors,
            dynamic_capacity,
        )?;
        // A reusable bucket must reserve the old sum too. Its policy remains
        // the one from the trusted build, never the serialized budget values.
        let reusable_policy = self
            .reusable_execution
            .as_ref()
            .map(|p| p.policy())
            .transpose()?;
        self.reusable_execution = reusable_policy
            .as_ref()
            .map(|policy| {
                Self::derive_reusable_execution(
                    policy,
                    node_count,
                    &self.dynamic_descriptors,
                    &self.dynamic_pools,
                )
            })
            .transpose()?;
        let ceilings = self
            .reusable_execution
            .as_ref()
            .map(|p| p.pool_workspace_ceilings())
            .transpose()?
            .unwrap_or_default();
        for pool in &mut self.dynamic_pools {
            *pool = DynamicBackingPoolSpec::from_core(
                pool.compatibility.clone(),
                pool.resource_ids.clone(),
                pool.minimum_request_bytes,
                pool.minimum_sequence_bytes,
                pool.minimum_step_bytes,
                pool.minimum_invocation_peak_bytes,
                pool.step_resource_slots.clone(),
                pool.theoretical_ceiling_bytes.get(),
                ceilings.get(&pool.pool_id).copied().unwrap_or(0),
                pool.checkpoint_growth_ceiling_bytes,
                dynamic_capacity,
                pool.invocation_liveness_mode,
                pool.invocation_liveness.clone(),
            )?;
        }
        (self.invocation_liveness_mode, self.invocation_liveness) =
            Self::summarize_pool_invocation_liveness(&self.dynamic_pools)?;
        self.minimum_invocation_peak_bytes =
            self.dynamic_pools.iter().try_fold(0_u64, |sum, pool| {
                sum.checked_add(pool.minimum_invocation_peak_bytes)
                    .ok_or_else(|| invalid_plan("legacy invocation peak overflows u64"))
            })?;
        self.minimum_runnable_request_bytes = self
            .minimum_request_bytes
            .checked_add(self.minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(self.minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(self.minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("legacy runnable bytes overflow u64"))?;
        self.theoretical_ceiling_bytes = CanonicalU128::new(self.dynamic_pools.iter().try_fold(
            u128::from(self.static_bytes),
            |sum, pool| {
                sum.checked_add(pool.theoretical_ceiling_bytes.get())
                    .and_then(|bytes| {
                        bytes.checked_add(u128::from(pool.reusable_workspace_ceiling_bytes))
                    })
                    .ok_or_else(|| invalid_plan("legacy theoretical bytes overflow u128"))
            },
        )?);
        self.validate()?;
        Ok(self)
    }
}
