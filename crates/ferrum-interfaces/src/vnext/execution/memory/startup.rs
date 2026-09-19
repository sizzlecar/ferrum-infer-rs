//! Metadata-only capacity evaluation over the compiled physical memory plan.

use super::*;

impl MemoryPlan {
    /// Physical bytes needed by equal-length sequences and one execution wave.
    ///
    /// This is startup planning evidence, never a reservation or an admission
    /// permit. Runtime admission still handles live cache retention, contention,
    /// and changing device availability. Each sequence is allocated separately;
    /// physical page padding, provider formulas, and proven workspace reuse are
    /// evaluated by the same plan helpers used to size runtime backing.
    pub fn startup_peak_bytes(
        &self,
        context_tokens: u64,
        active_sequences: u32,
        step_tokens: u64,
    ) -> Result<u64, VNextError> {
        self.startup_workload_peak_bytes(
            context_tokens,
            context_tokens,
            active_sequences,
            step_tokens,
        )
    }

    /// Request-lifetime storage owns the complete requested input/output
    /// ceiling; sequence state owns only its committed frontier. Keep these
    /// dimensions separate when evaluating shallow concurrent decoding.
    pub fn startup_workload_peak_bytes(
        &self,
        request_ceiling_tokens: u64,
        sequence_frontier_tokens: u64,
        active_sequences: u32,
        step_tokens: u64,
    ) -> Result<u64, VNextError> {
        if request_ceiling_tokens == 0
            || sequence_frontier_tokens == 0
            || sequence_frontier_tokens > request_ceiling_tokens
            || active_sequences == 0
            || step_tokens == 0
        {
            return Err(invalid_plan("startup workload dimensions must be non-zero"));
        }
        if active_sequences > self.maximum_active_sequences {
            return Err(invalid_plan(
                "startup workload exceeds the compiled sequence ceiling",
            ));
        }
        // Product language-model state is token-derived. Page-derived providers
        // must supply actual page evidence instead of assuming a token/page ratio.
        let request_shape = DynamicResourceShape::from_validated(1, request_ceiling_tokens, 0);
        let sequence_shape = DynamicResourceShape::from_validated(1, sequence_frontier_tokens, 0);
        let step_shape = DynamicResourceShape::from_validated(active_sequences, step_tokens, 0);
        let descriptors = self
            .dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let sealed_workspace = self
            .reusable_execution
            .as_ref()
            .map(|reusable| reusable.startup_sealed_pool_workspace_bytes())
            .transpose()?
            .unwrap_or_default();

        self.dynamic_pools
            .iter()
            .try_fold(self.static_bytes, |total, pool| {
                let sequence_bytes = pool.resource_ids.iter().try_fold(0_u64, |bytes, id| {
                    let descriptor = descriptors.get(id).ok_or_else(|| {
                        invalid_plan("startup pool references a missing descriptor")
                    })?;
                    let shape = match descriptor.lifetime() {
                        AllocationLifetime::Request => request_shape,
                        AllocationLifetime::Sequence => sequence_shape,
                        _ => return Ok(bytes),
                    };
                    let per_sequence = descriptor.evaluate_request_bytes_for_shape(shape)?;
                    per_sequence
                        .checked_mul(u64::from(active_sequences))
                        .and_then(|amount| bytes.checked_add(amount))
                        .ok_or_else(|| invalid_plan("startup sequence memory overflows u64"))
                })?;
                let step_bytes =
                    Self::reusable_step_bytes_for_shape(pool, &descriptors, step_shape)?;
                let invocation_bytes =
                    Self::reusable_invocation_bytes_for_shape(pool, &descriptors, step_shape)?;
                let mut wave_bytes = step_bytes
                    .checked_add(invocation_bytes)
                    .ok_or_else(|| invalid_plan("startup execution memory overflows u64"))?;

                // Reusable execution can round the wave up to a compiled shape
                // bucket. Preserve that physical capacity without charging every
                // possible cached program as though it executes concurrently.
                if let Some(reusable) = &self.reusable_execution {
                    let mut covered_classes = BTreeSet::new();
                    let mut class_buckets = BTreeMap::new();
                    for resolved in reusable.buckets() {
                        let bucket = resolved.bucket();
                        if !covered_classes.contains(bucket.class_id()) {
                            class_buckets.insert(bucket.class_id(), resolved);
                            if bucket.capacity().covers(active_sequences, step_tokens, 0) {
                                covered_classes.insert(bucket.class_id());
                            }
                        }
                    }
                    // Retaining the largest bucket beyond a class's coverage keeps
                    // this upper bound monotone when execution falls back to eager.
                    for resolved in class_buckets.into_values() {
                        if let Some(budget) = resolved
                            .pool_budgets()
                            .iter()
                            .find(|budget| budget.pool_id() == pool.pool_id())
                        {
                            wave_bytes = wave_bytes.max(budget.total_bytes()?);
                        }
                    }
                    // Workspace buckets apply before exact-program lookup, so
                    // even an eager miss can need a rounded, uncaptured arena.
                    // Sealed captured arenas cannot be trimmed to make room.
                    wave_bytes = wave_bytes
                        .checked_add(sealed_workspace.get(pool.pool_id()).copied().unwrap_or(0))
                        .ok_or_else(|| invalid_plan("sealed startup workspace overflows u64"))?;
                }
                let required = sequence_bytes
                    .checked_add(wave_bytes)
                    .ok_or_else(|| invalid_plan("startup pool memory overflows u64"))?
                    .max(pool.provisioning.minimum_resident_bytes());
                total
                    .checked_add(required)
                    .ok_or_else(|| invalid_plan("startup plan memory overflows u64"))
            })
    }
}
