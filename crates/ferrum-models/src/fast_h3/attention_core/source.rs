use super::*;
use ferrum_kernels::native_ops::sage_vsa_fp4::CheckedVsaRegions;
use std::sync::Arc;
#[derive(Clone, Copy)]
enum Route {
    Prefix,
    VideoQuery,
    Untile,
    Expand,
    Projection,
}
/// Original projection parts and derived maps retain their existing sources.
/// Native region metadata is returned separately for explicit provider binding.
pub struct FastH3AttentionCoreSource {
    prefix: FastH3PrefixAttentionSource,
    video_query: FastH3VsaRowTransportSource,
    untile: FastH3VsaRowTransportSource,
    expand: FastH3VsaRowTransportSource,
    projection: FastH3AttentionOutputSource,
    routes: BTreeMap<WeightId, (WeightComponentSpec, Route)>,
}
impl FastH3AttentionCoreSource {
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        layout: &FastH3VsaLayout,
        batches: u64,
        selection: FastH3VsaSelection,
        arithmetic: FastH3PrefixAttentionArithmetic,
        host: &HostMemoryAccount,
        l: FastH3AttentionCoreLimits,
    ) -> anyhow::Result<(FastH3AttentionCoreConfig, Self, Arc<CheckedVsaRegions>)> {
        l.validate()?;
        let transformer = package.config();
        let pool = FastH3VsaPoolConfig::from_layout(layout, batches, transformer, l.pooling)?;
        let pooled = FastH3PooledAttentionConfig::from_pool(&pool, l.pooled)?;
        let quantize =
            FastH3VsaQkvQuantizeConfig::from_layout(layout, batches, transformer, l.quantize)?;
        let fine_pack =
            FastH3HeadAxisPackConfig::from_video_layout(layout, batches, transformer, l.head_pack)?;
        let pooled_pack = FastH3HeadAxisPackConfig::from_pool(&pool, l.head_pack)?;
        let regions = regions::prepare(
            layout,
            batches,
            quantize.heads(),
            selection,
            host,
            l.maximum_region_bytes,
        )?;
        let (prefix, prefix_source) =
            FastH3PrefixAttentionSource::prepare(layout, batches, transformer, host, l.prefix)?;
        let prefix = prefix.with_attention_arithmetic(arithmetic, l.prefix)?;
        let columns = fine_pack.output_shape()?[2];
        let (video_query, video_source) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::VideoQueryRows,
            batches,
            columns,
            host,
            l.transport,
        )?;
        let (untile, untile_source) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::Untile,
            batches,
            columns,
            host,
            l.transport,
        )?;
        let (expand, expand_source) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::ExpandTileValues,
            batches,
            columns,
            host,
            l.transport,
        )?;
        let rows = batches
            .checked_mul(layout.packed().sequence_rows().try_into()?)
            .ok_or_else(|| invalid("packed batch rows overflow"))?;
        let (projection, projection_source) =
            FastH3AttentionOutputSource::prepare(package, block, rows, host, l.projection)?;
        let c = FastH3AttentionCoreConfig {
            prefix,
            quantize,
            pooled,
            video_query,
            fine_pack,
            pooled_pack,
            untile,
            expand,
            projection,
            region_digest: regions.digest().into(),
        };
        let built = program::compose(&c, l)?;
        let mut routes = BTreeMap::new();
        for (weights, route) in [
            (
                FastH3PrefixAttentionFamily::new(l.prefix)?.weights(&c.prefix)?,
                Route::Prefix,
            ),
            (
                FastH3VsaRowTransportFamily::new(l.transport)?.weights(&c.video_query)?,
                Route::VideoQuery,
            ),
            (
                FastH3VsaRowTransportFamily::new(l.transport)?.weights(&c.untile)?,
                Route::Untile,
            ),
            (
                FastH3VsaRowTransportFamily::new(l.transport)?.weights(&c.expand)?,
                Route::Expand,
            ),
            (
                FastH3AttentionOutputFamily::new(l.projection)?.weights(&c.projection)?,
                Route::Projection,
            ),
        ] {
            let ModelWeights::Stored(schema) = weights else {
                return Err(invalid("source must remain Stored").into());
            };
            for component in schema.components {
                if routes
                    .insert(component.id.clone(), (component, route))
                    .is_some()
                {
                    return Err(invalid("distinct source routes claim the same component").into());
                }
            }
        }
        if routes.len() != built.schema.components.len()
            || built
                .schema
                .components
                .iter()
                .any(|c| routes.get(&c.id).is_none_or(|(declared, _)| declared != c))
        {
            return Err(invalid("composition source inventory differs").into());
        }
        Ok((
            c,
            Self {
                prefix: prefix_source,
                video_query: video_source,
                untile: untile_source,
                expand: expand_source,
                projection: projection_source,
                routes,
            },
            regions,
        ))
    }
}
impl WeightComponentSource for FastH3AttentionCoreSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((declared, route)) = self.routes.get(&c.id) else {
            return Err(invalid("undeclared core source component"));
        };
        if declared != c {
            return Err(invalid("source identity differs"));
        }
        match route {
            Route::Prefix => self.prefix.component(c),
            Route::VideoQuery => self.video_query.component(c),
            Route::Untile => self.untile.component(c),
            Route::Expand => self.expand.component(c),
            Route::Projection => self.projection.component(c),
        }
    }
}
