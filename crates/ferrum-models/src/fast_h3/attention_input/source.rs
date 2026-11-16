use super::*;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Route {
    Map,
    Projection,
    Norm,
    Counts,
}
/// Every physical map/count/unit component has one original owner. Repeated
/// graph instances borrow the same backing, not copied derived components.
pub struct FastH3AttentionInputSource {
    package: Arc<FastH3TransformerPackage>,
    map: FastH3VsaRowTransportSource,
    projection: FastH3AttentionProjectionSource,
    counts: FastH3VsaPoolSource,
    routes: BTreeMap<WeightId, (WeightComponentSpec, Route)>,
}
impl FastH3AttentionInputSource {
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        layout: &FastH3VsaLayout,
        batches: u64,
        host: &HostMemoryAccount,
        limits: FastH3AttentionInputLimits,
    ) -> anyhow::Result<(FastH3AttentionInputConfig, Self)> {
        limits.validate()?;
        let tiled = u64::try_from(layout.padded_rows())?;
        let packed = u64::try_from(layout.packed().sequence_rows())?;
        let flat_tiled = batches
            .checked_mul(tiled)
            .ok_or_else(|| invalid("tiled batch product overflows"))?;
        let flat_packed = batches
            .checked_mul(packed)
            .ok_or_else(|| invalid("packed batch product overflows"))?;
        let qk = FastH3QkNormRopeConfig::from_package(&package, block, flat_tiled, limits.qk)?;
        // This authoritative check happens before the norm components become routes.
        FastH3QkNormRopeSource::new(&package, &qk, limits.qk)?;
        let pooling =
            FastH3VsaPoolConfig::from_layout(layout, batches, package.config(), limits.pooling)?;
        let (tile_hidden, map) = FastH3VsaRowTransportSource::prepare(
            layout,
            FastH3VsaRowTransport::Tile,
            batches,
            package.config().hidden_size().try_into()?,
            host,
            limits.transport,
        )?;
        let tile_rotary = tile_hidden.with_columns(qk.rotary_dim(), limits.transport)?;
        // One checked inventory, four alpha scalars and a SINGLE unit F32. Both
        // groups retain this same source; they only differ in logical row shape.
        let (all, projection) = FastH3AttentionProjectionSource::prepare(
            package.clone(),
            block,
            flat_tiled,
            host,
            limits.projection,
        )?;
        let c = FastH3AttentionInputConfig {
            qkv: all.select_group(
                FastH3AttentionProjectionGroup::Qkv,
                flat_tiled,
                limits.projection,
            )?,
            gate: all.select_group(
                FastH3AttentionProjectionGroup::Gate,
                flat_packed,
                limits.projection,
            )?,
            tile_hidden,
            tile_rotary,
            qk,
            pooling,
        };
        let built = program::compose(&c, limits)?;
        let counts = FastH3VsaPoolSource::new(&c.pooling, layout, host, limits.pooling)?;
        let mut routes = BTreeMap::new();
        for (weights, route) in [
            (
                FastH3VsaRowTransportFamily::new(limits.transport)?.weights(&c.tile_hidden)?,
                Route::Map,
            ),
            (
                FastH3AttentionProjectionFamily::new(limits.projection)?.weights(&all)?,
                Route::Projection,
            ),
            (
                FastH3QkNormRopeFamily::new(limits.qk)?.weights(&c.qk)?,
                Route::Norm,
            ),
            (
                FastH3VsaPoolFamily::new(limits.pooling)?.weights(&c.pooling)?,
                Route::Counts,
            ),
        ] {
            let ModelWeights::Stored(s) = weights else {
                return Err(invalid("source must remain Stored").into());
            };
            for component in s.components {
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
                .any(|v| routes.get(&v.id).is_none_or(|(c, _)| c != v))
        {
            return Err(invalid("composition source inventory differs").into());
        }
        Ok((
            c,
            Self {
                package,
                map,
                projection,
                counts,
                routes,
            },
        ))
    }
}
impl WeightComponentSource for FastH3AttentionInputSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((declared, route)) = self.routes.get(&c.id) else {
            return Err(invalid("undeclared source component"));
        };
        if declared != c {
            return Err(invalid("source component identity differs"));
        }
        match route {
            Route::Map => self.map.component(c),
            Route::Projection => self.projection.component(c),
            Route::Norm => self.package.dense_archive().component(c),
            Route::Counts => self.counts.component(c),
        }
    }
}
