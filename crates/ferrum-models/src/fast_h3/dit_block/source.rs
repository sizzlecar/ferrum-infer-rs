use super::*;
use ferrum_kernels::native_ops::sage_vsa_fp4::CheckedVsaRegions;
use std::sync::Arc;
#[derive(Clone, Copy)]
enum Route {
    Norm,
    AttentionInput,
    AttentionCore,
    Ffn,
}
pub struct FastH3DitBlockSource {
    package: Arc<FastH3TransformerPackage>,
    attention_input: FastH3AttentionInputSource,
    attention_core: FastH3AttentionCoreSource,
    ffn: FastH3FfnSource,
    routes: BTreeMap<WeightId, (WeightComponentSpec, Route)>,
}
impl FastH3DitBlockSource {
    /// Requests share this exact tile layout and explicitly selected sparse
    /// policy, but each supplies its own evaluation and actual text tags.
    /// This is batch geometry, not a continuous-batching admission decision.
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        requests: &[(usize, &[FastH3TextTagRun])],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        l: FastH3DitBlockLimits,
    ) -> anyhow::Result<(FastH3DitBlockConfig, Self, Arc<CheckedVsaRegions>)> {
        l.validate()?;
        let batches = u64::try_from(requests.len())?;
        if batches == 0
            || batches > l.attention_input.transport.maximum_batches
            || batches > l.attention_core.quantize.maximum_batches
        {
            return Err(invalid("request batch count exceeds explicit geometry capacity").into());
        }
        let modulation = |stage| -> Result<FastH3ModulationConfig, VNextError> {
            let parts = requests
                .iter()
                .map(|(evaluation, tags)| {
                    FastH3ModulationConfig::from_package(
                        &package,
                        block,
                        stage,
                        layout.packed(),
                        schedule,
                        *evaluation,
                        tags,
                        l.modulation,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let c = FastH3ModulationConfig::concatenate_requests(&parts, l.modulation)?;
            FastH3ModulationSource::new(&package, &c, l.modulation)?;
            Ok(c)
        };
        let norm1 = modulation(FastH3ModulationStage::Attention)?;
        let norm2 = modulation(FastH3ModulationStage::FeedForward)?;
        Self::prepare_with_modulation(
            package, block, layout, batches, norm1, norm2, selection, arithmetic, host, l,
        )
    }
    /// One stable plan for a fixed admitted request geometry. Each evaluation
    /// supplies a new compact role table, without changing the stored weights.
    pub fn prepare_conditioning_roles(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        requests: &[&[FastH3TextTagRun]],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        l: FastH3DitBlockLimits,
    ) -> anyhow::Result<(FastH3DitBlockConfig, Self, Arc<CheckedVsaRegions>)> {
        l.validate()?;
        let batches = u64::try_from(requests.len())?;
        if batches == 0
            || batches > l.attention_input.transport.maximum_batches
            || batches > l.attention_core.quantize.maximum_batches
        {
            return Err(invalid("request batch count exceeds explicit geometry capacity").into());
        }
        let modulation = |stage| -> Result<FastH3ModulationConfig, VNextError> {
            let c = FastH3ModulationConfig::from_conditioning_roles(
                &package,
                block,
                stage,
                layout.packed(),
                schedule,
                requests,
                l.modulation,
            )?;
            FastH3ModulationSource::new(&package, &c, l.modulation)?;
            Ok(c)
        };
        let norm1 = modulation(FastH3ModulationStage::Attention)?;
        let norm2 = modulation(FastH3ModulationStage::FeedForward)?;
        Self::prepare_with_modulation(
            package, block, layout, batches, norm1, norm2, selection, arithmetic, host, l,
        )
    }
    fn prepare_with_modulation(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        layout: &FastH3VsaLayout,
        batches: u64,
        norm1: FastH3ModulationConfig,
        norm2: FastH3ModulationConfig,
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        l: FastH3DitBlockLimits,
    ) -> anyhow::Result<(FastH3DitBlockConfig, Self, Arc<CheckedVsaRegions>)> {
        let (attention_input, input_source) = FastH3AttentionInputSource::prepare(
            package.clone(),
            block,
            layout,
            batches,
            host,
            l.attention_input,
        )?;
        let (attention_core, core_source, regions) = FastH3AttentionCoreSource::prepare(
            package.clone(),
            block,
            layout,
            batches,
            selection,
            arithmetic,
            host,
            l.attention_core,
        )?;
        let (ffn, ffn_source) =
            FastH3FfnSource::prepare(package.clone(), block, norm1.rows(), host, l.ffn)?;
        let c = FastH3DitBlockConfig {
            norm1,
            attention_input,
            attention_core,
            norm2,
            ffn,
        };
        let built = program::compose(&c, l)?;
        let norm = FastH3ModulationFamily::new(l.modulation)?;
        let mut routes = BTreeMap::new();
        for (weights, route) in [
            (norm.weights(&c.norm1)?, Route::Norm),
            (norm.weights(&c.norm2)?, Route::Norm),
            (
                FastH3AttentionInputFamily::new(l.attention_input)?.weights(&c.attention_input)?,
                Route::AttentionInput,
            ),
            (
                FastH3AttentionCoreFamily::new(l.attention_core)?.weights(&c.attention_core)?,
                Route::AttentionCore,
            ),
            (FastH3FfnFamily::new(l.ffn)?.weights(&c.ffn)?, Route::Ffn),
        ] {
            let ModelWeights::Stored(schema) = weights else {
                return Err(invalid("fragment source must remain Stored").into());
            };
            for component in schema.components {
                match routes.entry(component.id.clone()) {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert((component, route));
                    }
                    std::collections::btree_map::Entry::Occupied(e) if e.get().0 == component => {}
                    _ => {
                        return Err(
                            invalid("physical source identity conflicts across fragments").into(),
                        )
                    }
                }
            }
        }
        if routes.len() != built.schema.components.len()
            || built
                .schema
                .components
                .iter()
                .any(|c| routes.get(&c.id).is_none_or(|(original, _)| original != c))
        {
            return Err(invalid("composed source inventory differs").into());
        }
        Ok((
            c,
            Self {
                package,
                attention_input: input_source,
                attention_core: core_source,
                ffn: ffn_source,
                routes,
            },
            regions,
        ))
    }
}
impl WeightComponentSource for FastH3DitBlockSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((original, route)) = self.routes.get(&c.id) else {
            return Err(invalid("undeclared block source"));
        };
        if c != original {
            return Err(invalid("declared block source identity differs"));
        }
        match route {
            Route::Norm => self.package.dense_archive().component(c),
            Route::AttentionInput => self.attention_input.component(c),
            Route::AttentionCore => self.attention_core.component(c),
            Route::Ffn => self.ffn.component(c),
        }
    }
}
