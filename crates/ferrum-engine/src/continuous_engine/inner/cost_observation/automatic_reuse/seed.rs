//! Cold, input-only declaration. Restoring coordinates never restores numeric
//! authority; the independently replayed catalog and feedback do that separately.
use super::*;
use ferrum_interfaces::execution_cost::CostWorkloadDomainV1;
pub(super) use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

#[derive(Clone, Copy)]
pub(super) struct SeedLimits {
    pub maximum_axes: usize,
    pub maximum_bytes: usize,
}

pub(super) fn validate(
    seed: &DeclaredAlgorithmUniverseV1,
    workload: &CostWorkloadDomainV1,
    limits: SeedLimits,
) -> Result<()> {
    if seed.workload_domain_signature() != workload.sha256() {
        return Err(CacheMiss::Identity);
    }
    seed.validate_budget(limits.maximum_axes, limits.maximum_bytes)
        .map_err(|_| CacheMiss::Capacity)
}

/// serde's typed UniverseWire clone, bounded encoded bytes and final retained
/// declaration coexist. This preauthorization precedes even counting encoding.
fn memory_bound(retained: usize, encoded_bytes: usize) -> Result<usize> {
    encoded_bytes
        .checked_mul(18)
        .and_then(|n| n.checked_add(retained))
        .ok_or(CacheMiss::Capacity)
}

pub(super) fn persist(
    cache: &CacheSession,
    seed: &DeclaredAlgorithmUniverseV1,
    context: &ReplayContext<'_>,
    transaction: &ColdTransaction,
) -> Result<ArtifactReceipt> {
    transaction.poll()?;
    let limits = context.seed_limits.ok_or(CacheMiss::Identity)?;
    validate(seed, context.workload, limits)?;
    let backing = seed.retained_payload_bytes().ok_or(CacheMiss::Capacity)?;
    let encoding_retained = backing
        .checked_mul(3)
        .and_then(|n| n.checked_add(context.original_retained_bytes))
        .ok_or(CacheMiss::Capacity)?;
    if encoding_retained > cache.0.limits.maximum_retained_bytes {
        return Err(CacheMiss::Capacity);
    }
    let mut counter = Encoding {
        maximum: limits.maximum_bytes,
        written: 0,
        bytes: None,
    };
    serde_json::to_writer(&mut counter, seed).map_err(|_| CacheMiss::Capacity)?;
    transaction.poll()?;
    if memory_bound(encoding_retained, counter.written)? > cache.0.limits.maximum_retained_bytes {
        return Err(CacheMiss::Capacity);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(counter.written)
        .map_err(|_| CacheMiss::Capacity)?;
    if bytes.capacity() != counter.written {
        return Err(CacheMiss::Capacity);
    }
    let mut encoded = Encoding {
        maximum: counter.written,
        written: 0,
        bytes: Some(bytes),
    };
    serde_json::to_writer(&mut encoded, seed).map_err(|_| CacheMiss::Capacity)?;
    let bytes = encoded.bytes.as_ref().ok_or(CacheMiss::Corrupt)?;
    if bytes.len() != counter.written {
        return Err(CacheMiss::Corrupt);
    }
    let file = cache.reserve_external(bytes.len() as u64, transaction)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&file.path)
        .map_err(io)?;
    for chunk in bytes.chunks(8192) {
        transaction.poll()?;
        output.write_all(chunk).map_err(io)?;
    }
    output.sync_all().map_err(io)?;
    drop(output);
    file.finish(transaction)
}

pub(super) fn restore(
    cache: &CacheSession,
    manifest: &CacheManifest,
    context: &ReplayContext<'_>,
    retained: usize,
    transaction: &ColdTransaction,
) -> Result<Option<DeclaredAlgorithmUniverseV1>> {
    let Some(receipt) = manifest.startup_seed else {
        return Ok(None);
    };
    transaction.poll()?;
    let limits = context.seed_limits.ok_or(CacheMiss::Identity)?;
    let length = usize::try_from(receipt.bytes).map_err(|_| CacheMiss::Capacity)?;
    if length == 0
        || length > limits.maximum_bytes
        || memory_bound(retained, length)? > cache.0.limits.maximum_retained_bytes
    {
        return Err(CacheMiss::Capacity);
    }
    let path = cache.artifact_path(manifest.generation, &receipt);
    store::verify_file(&path, &receipt, transaction)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| CacheMiss::Capacity)?;
    if bytes.capacity() != length {
        return Err(CacheMiss::Capacity);
    }
    bytes.resize(length, 0);
    let mut input = File::open(&path).map_err(io)?;
    for chunk in bytes.chunks_mut(8192) {
        transaction.poll()?;
        input.read_exact(chunk).map_err(io)?;
    }
    if input.read(&mut [0]).map_err(io)? != 0 {
        return Err(CacheMiss::Corrupt);
    }
    // Strict typed import recomputes the universe signature and rejects invalid
    // primitive classes/order. No serialized digest grants unchecked authority.
    let seed = serde_json::from_slice(&bytes).map_err(|_| CacheMiss::Corrupt)?;
    validate(&seed, context.workload, limits)?;
    transaction.poll()?;
    Ok(Some(seed))
}

struct Encoding {
    maximum: usize,
    written: usize,
    bytes: Option<Vec<u8>>,
}
impl Write for Encoding {
    fn write(&mut self, value: &[u8]) -> std::io::Result<usize> {
        let next = self
            .written
            .checked_add(value.len())
            .filter(|n| *n <= self.maximum)
            .ok_or_else(|| std::io::Error::other("algorithm declaration encoding capacity"))?;
        if let Some(bytes) = &mut self.bytes {
            if next > bytes.capacity() {
                return Err(std::io::Error::other(
                    "algorithm declaration allocation capacity",
                ));
            }
            bytes.extend_from_slice(value);
        }
        self.written = next;
        Ok(value.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_reuse_algorithm_seed_encoding_never_grows_after_authorization() {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(4).unwrap();
        let capacity = bytes.capacity();
        let mut writer = Encoding {
            maximum: 4,
            written: 0,
            bytes: Some(bytes),
        };
        writer.write_all(b"seed").unwrap();
        assert!(writer.write_all(b"x").is_err());
        assert_eq!(writer.written, 4);
        assert_eq!(writer.bytes.as_ref().unwrap().capacity(), capacity);
        assert_eq!(memory_bound(17, 5).unwrap(), 107);
        assert_eq!(memory_bound(usize::MAX, 1), Err(CacheMiss::Capacity));
    }
}
