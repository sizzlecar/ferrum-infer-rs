use super::*;
pub struct FastH3DenoisingSource<'a> {
    media: FastH3InputProjectionSource<'a>,
    backbone: FastH3DitBackboneSource,
    routes: BTreeMap<WeightId, (WeightComponentSpec, bool)>,
}
impl<'a> FastH3DenoisingSource<'a> {
    pub fn prepare(
        package: &'a Arc<FastH3TransformerPackage>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        text_tags: &[FastH3TextTagRun],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &'a HostMemoryAccount,
        l: FastH3DenoisingLimits,
    ) -> anyhow::Result<(FastH3DenoisingConfig, Self, Arc<CheckedVsaRegions>)> {
        Self::prepare_batch(
            package,
            layout,
            schedule,
            &[text_tags],
            selection,
            arithmetic,
            host,
            l,
        )
    }
    /// Same checked document geometry, independent presentation routing per
    /// member. Slot order is transient; it is not stored in any continuation.
    pub fn prepare_batch(
        package: &'a Arc<FastH3TransformerPackage>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        text_tags: &[&[FastH3TextTagRun]],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &'a HostMemoryAccount,
        l: FastH3DenoisingLimits,
    ) -> anyhow::Result<(FastH3DenoisingConfig, Self, Arc<CheckedVsaRegions>)> {
        let batches = u64::try_from(text_tags.len()).map_err(invalid)?;
        let media = FastH3InputProjectionConfig::media_batch_from_package(
            package,
            layout.packed(),
            batches,
            l.input,
        )?;
        let media_source = FastH3InputProjectionSource::new(package, &media, host, l.input)?;
        let packing = FastH3InputPackingConfig::from_batched_layout(
            layout.packed(),
            package.config().hidden_size() as u64,
            batches,
            l.input,
        )?;
        let (backbone, backbone_source, regions) = FastH3DitBackboneSource::prepare(
            package.clone(),
            layout,
            schedule,
            text_tags,
            selection,
            arithmetic,
            host,
            l.backbone,
        )?;
        let c = FastH3DenoisingConfig {
            media,
            packing,
            backbone,
            schedule: crate::TensorProgramSchedule::default(),
        };
        let built = compose(&c, l)?;
        let mut routes = BTreeMap::new();
        for (weights, media) in [
            (
                FastH3InputProjectionFamily::new(l.input)?.weights(&c.media)?,
                true,
            ),
            (
                FastH3DitBackboneFamily::new(l.backbone)?.weights(&c.backbone)?,
                false,
            ),
        ] {
            let ModelWeights::Stored(schema) = weights else {
                return Err(invalid("phase requires Stored source").into());
            };
            for component in schema.components {
                if routes
                    .insert(component.id.clone(), (component, media))
                    .is_some()
                {
                    return Err(invalid("input and block source identities overlap").into());
                }
            }
        }
        if routes.len() != built.schema.components.len()
            || built
                .schema
                .components
                .iter()
                .any(|c| routes.get(&c.id).is_none_or(|(expected, _)| expected != c))
        {
            return Err(invalid("source inventory differs").into());
        }
        Ok((
            c,
            Self {
                media: media_source,
                backbone: backbone_source,
                routes,
            },
            regions,
        ))
    }
}
impl WeightComponentSource for FastH3DenoisingSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((expected, media)) = self.routes.get(&c.id) else {
            return Err(invalid("undeclared source"));
        };
        if expected != c {
            return Err(invalid("source interpretation differs"));
        }
        if *media {
            self.media.component(c)
        } else {
            self.backbone.component(c)
        }
    }
}

/// Service-owned form of the same cold preparation. It retains the original
/// archive and derived backbone owners without a self-reference or a second
/// materialization cache. Restoring an installation does not rebuild the graph.
pub struct OwnedFastH3DenoisingSource {
    package: Arc<FastH3TransformerPackage>,
    host: HostMemoryAccount,
    backbone: FastH3DitBackboneSource,
    routes: BTreeMap<WeightId, (WeightComponentSpec, bool)>,
}
impl OwnedFastH3DenoisingSource {
    pub fn prepare_batch(
        package: Arc<FastH3TransformerPackage>,
        layout: &FastH3VsaLayout,
        schedule: &FastH3ConditioningSchedule,
        text_tags: &[&[FastH3TextTagRun]],
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        limits: FastH3DenoisingLimits,
    ) -> anyhow::Result<(FastH3DenoisingConfig, Self, Arc<CheckedVsaRegions>)> {
        let (config, source, regions) = FastH3DenoisingSource::prepare_batch(
            &package, layout, schedule, text_tags, selection, arithmetic, host, limits,
        )?;
        let FastH3DenoisingSource {
            media,
            backbone,
            routes,
        } = source;
        drop(media);
        Ok((
            config,
            Self {
                package,
                host: host.clone(),
                backbone,
                routes,
            },
            regions,
        ))
    }
}
impl WeightComponentSource for OwnedFastH3DenoisingSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let (expected, media) = self
            .routes
            .get(&c.id)
            .ok_or_else(|| invalid("undeclared owned source"))?;
        if expected != c {
            return Err(invalid("owned source interpretation differs"));
        }
        if *media {
            ferrum_quantization::tensor_archive::ManagedDenseArchiveSource::new(
                self.package.dense_archive(),
                &self.host,
            )
            .into_component(c)
        } else {
            self.backbone.component(c)
        }
    }
}
