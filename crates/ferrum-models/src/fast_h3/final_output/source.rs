use super::*;

/// Routes each complete physical declaration to exactly one source. A failed
/// source lookup is never retried against a different owner.
pub struct FastH3FinalOutputSource<'a> {
    norm: FastH3OutputConditioningSource<'a>,
    video: FastH3PddProjectionSource,
    audio: FastH3PddProjectionSource,
    components: BTreeMap<WeightId, (WeightComponentSpec, usize)>,
}
impl<'a> FastH3FinalOutputSource<'a> {
    pub fn prepare(
        package: &'a FastH3TransformerPackage,
        plan: &FastH3PddPlan,
        layout: &FastH3PackedLayout,
        evaluation: usize,
        host: &HostMemoryAccount,
        limits: FastH3FinalOutputLimits,
    ) -> Result<(FastH3FinalOutputConfig, Self), VNextError> {
        limits.validate()?;
        plan.schedule()
            .validate(limits.maximum_pdd_steps)
            .map_err(invalid)?;
        let schedule = FastH3ConditioningSchedule::all_modalities(plan)?;
        let norm = FastH3OutputNormConfig::from_package(
            package,
            layout,
            &schedule,
            evaluation,
            limits.conditioning,
        )?;
        let norm_source =
            FastH3OutputConditioningSource::for_normalization(package, &norm, limits.conditioning)?;
        let (video, video_source) = FastH3PddProjectionSource::prepare(
            package,
            plan,
            evaluation,
            FastH3OutputModality::Video,
            layout.target_video_rows().end.try_into().map_err(invalid)?,
            host,
            limits.projection,
        )
        .map_err(invalid)?;
        let (audio, audio_source) = FastH3PddProjectionSource::prepare(
            package,
            plan,
            evaluation,
            FastH3OutputModality::Audio,
            layout.target_audio_rows().end.try_into().map_err(invalid)?,
            host,
            limits.projection,
        )
        .map_err(invalid)?;
        let config = FastH3FinalOutputConfig {
            input_residency: FastH3PredictionInput::Host,
            norm,
            video,
            audio,
            schedule: plan.schedule().clone(),
            segments: layout
                .segments()
                .iter()
                .map(|s| {
                    Ok(Segment {
                        kind: s.kind,
                        rows: s.packed_rows.len().try_into().map_err(invalid)?,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        };
        // Check the complete graph, aggregate capacities and source identity
        // before returning either immutable derived buffer owner.
        program::compose(&config, limits)?;
        let nf = FastH3OutputNormFamily::new(limits.conditioning)?;
        let pf = FastH3PddProjectionFamily::new(limits.projection)?;
        let mut components = BTreeMap::new();
        for (index, weights) in [
            nf.weights(&config.norm)?,
            pf.weights(&config.video)?,
            pf.weights(&config.audio)?,
        ]
        .into_iter()
        .enumerate()
        {
            let ModelWeights::Stored(schema) = weights else {
                return Err(invalid("output source requires Stored components"));
            };
            for c in schema.components {
                if components.insert(c.id.clone(), (c, index)).is_some() {
                    return Err(invalid("source component identity aliases another owner"));
                }
            }
        }
        Ok((
            config,
            Self {
                norm: norm_source,
                video: video_source,
                audio: audio_source,
                components,
            },
        ))
    }
}
impl WeightComponentSource for FastH3FinalOutputSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let (declared, index) = self
            .components
            .get(&c.id)
            .ok_or_else(|| invalid("undeclared output component"))?;
        if c != declared {
            return Err(invalid("output component declaration differs"));
        }
        match index {
            0 => self.norm.component(c),
            1 => self.video.component(c),
            2 => self.audio.component(c),
            _ => Err(invalid("unknown source owner")),
        }
    }
}

/// Original checkpoint plus already-derived PDD buffers owned by the service.
/// The normalizer's complete declaration remains in the checked route table.
pub struct OwnedFastH3FinalOutputSource {
    package: std::sync::Arc<FastH3TransformerPackage>,
    video: FastH3PddProjectionSource,
    audio: FastH3PddProjectionSource,
    components: BTreeMap<WeightId, (WeightComponentSpec, usize)>,
}
impl OwnedFastH3FinalOutputSource {
    pub fn prepare(
        package: std::sync::Arc<FastH3TransformerPackage>,
        plan: &FastH3PddPlan,
        layout: &FastH3PackedLayout,
        evaluation: usize,
        host: &HostMemoryAccount,
        limits: FastH3FinalOutputLimits,
    ) -> Result<(FastH3FinalOutputConfig, Self), VNextError> {
        let (config, source) =
            FastH3FinalOutputSource::prepare(&package, plan, layout, evaluation, host, limits)?;
        let FastH3FinalOutputSource {
            norm,
            video,
            audio,
            components,
        } = source;
        drop(norm);
        Ok((
            config,
            Self {
                package,
                video,
                audio,
                components,
            },
        ))
    }
}
impl WeightComponentSource for OwnedFastH3FinalOutputSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let (expected, route) = self
            .components
            .get(&c.id)
            .ok_or_else(|| invalid("undeclared owned output component"))?;
        if expected != c {
            return Err(invalid("owned output interpretation differs"));
        }
        match route {
            0 => self.package.dense_archive().component(c),
            1 => self.video.component(c),
            2 => self.audio.component(c),
            _ => Err(invalid("unknown owned output source")),
        }
    }
}
