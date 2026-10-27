//! Additional physical Host claims for one sequential family initialization.
use super::*;

pub(super) fn additional(
    host: &HostMemoryAccount,
    prepared: &plan::Prepared,
    cache: Option<&host_cache::HostCache>,
    source: Option<&dyn WeightComponentSource>,
) -> Result<usize, VNextError> {
    let Some(cache) = cache else {
        return usize::try_from(prepared.maximum_group_accounted_bytes(host)?)
            .map_err(|_| invalid("additional Host backing exceeds usize"));
    };
    let mut forecast = cache.forecast()?;
    let hits = prepared
        .conversion_groups
        .iter()
        .map(|group| {
            let Some(source) = source.filter(|_| cache.fits(group.payload_bytes)) else {
                return Ok(false);
            };
            let key = group_key(cache, source, group)?;
            Ok(forecast.hit_or_insert(key, group.payload_bytes))
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    missing_bytes(host, prepared, cache, &hits)
}

/// Hold exact warm owners and register cold keys atomically with cache lookup.
/// The returned lease must live through every initialization using this grant.
pub(super) fn prepare(
    host: &HostMemoryAccount,
    prepared: &plan::Prepared,
    cache: Option<&host_cache::HostCache>,
    source: Option<&dyn WeightComponentSource>,
) -> Result<HostBackingPreparation, VNextError> {
    let (Some(cache), Some(source)) = (cache, source) else {
        return additional(host, prepared, cache, source).map(HostBackingPreparation::new);
    };
    let keys = prepared
        .conversion_groups
        .iter()
        .map(|group| {
            cache
                .fits(group.payload_bytes)
                .then(|| group_key(cache, source, group))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (hits, retention) = cache.retain(keys)?;
    Ok(
        HostBackingPreparation::new(missing_bytes(host, prepared, cache, &hits)?)
            .with_retention(retention),
    )
}

fn group_key(
    cache: &host_cache::HostCache,
    source: &dyn WeightComponentSource,
    group: &plan::ConversionGroup,
) -> Result<host_cache::Key, VNextError> {
    let sources = group.sources.iter().collect::<Vec<_>>();
    let outputs = group.outputs.iter().collect::<Vec<_>>();
    let payloads = conversion::resolve_sources(source, &sources)?;
    cache.key(&sources, &outputs, &payloads)
}

fn missing_bytes(
    host: &HostMemoryAccount,
    prepared: &plan::Prepared,
    cache: &host_cache::HostCache,
    hits: &[bool],
) -> Result<usize, VNextError> {
    let mut new_retained = 0usize;
    let mut transient_peak = 0usize;
    for (group, &hit) in prepared.conversion_groups.iter().zip(hits) {
        if hit {
            continue;
        }
        let cached = cache.fits(group.payload_bytes);
        let payload = host
            .backing_bytes(group.payload_bytes)
            .map_err(|error| invalid(format!("additional payload backing: {error}")))?;
        let lookup = if group.lookup_bytes == 0 {
            0
        } else {
            host.backing_bytes(group.lookup_bytes)
                .map_err(|error| invalid(format!("additional lookup backing: {error}")))?
        };
        let transient = if cached {
            // Do not subtract evicted entries: an upload alias can retain them,
            // and an entry allocated by an earlier closed scope returns capacity
            // to its parent rather than to this invocation's scope credit.
            new_retained = new_retained
                .checked_add(payload)
                .ok_or_else(|| invalid("additional cached backing overflow"))?;
            lookup
        } else {
            payload
                .checked_add(lookup)
                .ok_or_else(|| invalid("additional uncached backing overflow"))?
        };
        transient_peak = transient_peak.max(transient);
    }
    new_retained
        .checked_add(transient_peak)
        .ok_or_else(|| invalid("additional Host backing overflow"))
}
