use super::*;
pub struct FastH3DitBackboneSource {
    blocks: Vec<FastH3DitBlockSource>,
    routes: BTreeMap<WeightId, (WeightComponentSpec, usize)>,
}
impl FastH3DitBackboneSource {
    /// Every checkpoint layer, in order. Selection is explicitly stable across
    /// evaluations: the fixed baseline uses Trained at all eight forwards.
    /// A future schedule with dense warmup needs a separate declared plan.
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        requests: &[&[FastH3TextTagRun]],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        l: FastH3DitBackboneLimits,
    ) -> anyhow::Result<(FastH3DitBackboneConfig, Self, Arc<CheckedVsaRegions>)> {
        l.validate()?;
        let layers = package.config().layers();
        if layers == 0 || layers > l.maximum_layers {
            return Err(invalid("checkpoint layer count exceeds capacity").into());
        }
        let mut configs = Vec::with_capacity(layers);
        let mut blocks = Vec::with_capacity(layers);
        let mut routes = BTreeMap::new();
        let mut regions: Option<Arc<CheckedVsaRegions>> = None;
        let family = FastH3DitBlockFamily::new(l.block)?;
        for i in 0..layers {
            let (c, source, current_regions) = FastH3DitBlockSource::prepare_conditioning_roles(
                package.clone(),
                i,
                layout,
                schedule,
                requests,
                selection,
                arithmetic,
                host,
                l.block,
            )?;
            if let Some(original) = &regions {
                if original.digest() != current_regions.digest() {
                    return Err(
                        invalid("native region interpretation changes across blocks").into(),
                    );
                }
            } else {
                regions = Some(current_regions);
            }
            let ModelWeights::Stored(schema) = family.weights(&c)? else {
                return Err(invalid("blocks must retain Stored sources").into());
            };
            for component in schema.components {
                match routes.entry(component.id.clone()) {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert((component, i));
                    }
                    std::collections::btree_map::Entry::Occupied(e) if e.get().0 == component => {}
                    _ => {
                        return Err(invalid(
                            "physical component interpretation differs across layers",
                        )
                        .into())
                    }
                }
            }
            configs.push(c);
            blocks.push(source);
        }
        let config = FastH3DitBackboneConfig { blocks: configs };
        let built = compose(&config, l)?;
        if built.schema.components.len() != routes.len()
            || built
                .schema
                .components
                .iter()
                .any(|c| routes.get(&c.id).is_none_or(|(expected, _)| expected != c))
        {
            return Err(invalid("stack source inventory differs").into());
        }
        Ok((
            config,
            Self { blocks, routes },
            regions.ok_or_else(|| invalid("missing native region owner"))?,
        ))
    }
}
impl WeightComponentSource for FastH3DitBackboneSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((expected, block)) = self.routes.get(&c.id) else {
            return Err(invalid("undeclared stack component"));
        };
        if expected != c {
            return Err(invalid("stack source identity differs"));
        }
        self.blocks[*block].component(c)
    }
}
