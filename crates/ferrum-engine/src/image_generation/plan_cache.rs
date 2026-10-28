//! Compilation metadata cache. Entries also retain their materializer owners;
//! any converted-weight Host backing remains separately charged to that owner.
//! Eviction releases this cache's Arc, never another request's compiled component.
//! Encoded size measures serialization, not heap usage or Host payload bytes.
use super::*;
use std::{
    collections::VecDeque,
    io,
    sync::{Mutex, MutexGuard},
    time::Instant,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImagePlanCachePolicy {
    Disabled,
    /// One compiled plan per (stage, role), without serialized-size accounting.
    #[default]
    LastPerStage,
    /// LRU within each stage and across all stages for the encoded-byte limit.
    /// Both limits are required; neither measures process heap or device memory.
    MultiShape {
        maximum_entries_per_stage: usize,
        maximum_encoded_metadata_bytes: usize,
    },
}
impl ImagePlanCachePolicy {
    pub fn validate(self) -> anyhow::Result<()> {
        if let Self::MultiShape {
            maximum_entries_per_stage,
            maximum_encoded_metadata_bytes,
        } = self
        {
            ensure!(
                maximum_entries_per_stage > 0
                    && maximum_entries_per_stage <= isize::MAX as usize / 6,
                "image multi_shape cache entry limit must be positive and addressable across six stages"
            );
            ensure!(
                maximum_encoded_metadata_bytes > 0
                    && maximum_encoded_metadata_bytes <= isize::MAX as usize,
                "image multi_shape encoded metadata limit must be positive and addressable"
            );
        }
        Ok(())
    }
}

// Preserve the original string variants. The strict nested struct also rejects
// unknown fields and duplicate limits rather than silently accepting a typo.
impl<'de> Deserialize<'de> for ImagePlanCachePolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Limits {
            maximum_entries_per_stage: usize,
            maximum_encoded_metadata_bytes: usize,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Wire {
            Disabled,
            LastPerStage,
            MultiShape(Limits),
        }
        let policy = match Wire::deserialize(deserializer)? {
            Wire::Disabled => Self::Disabled,
            Wire::LastPerStage => Self::LastPerStage,
            Wire::MultiShape(limits) => Self::MultiShape {
                maximum_entries_per_stage: limits.maximum_entries_per_stage,
                maximum_encoded_metadata_bytes: limits.maximum_encoded_metadata_bytes,
            },
        };
        policy.validate().map_err(serde::de::Error::custom)?;
        Ok(policy)
    }
}

/// Snapshot after this acquisition; counters are cumulative for this pipeline
/// policy and saturate at u64::MAX. A cache hit is separate from session reuse.
#[derive(Debug, Clone, Serialize)]
pub struct ImagePlanCacheReceipt {
    pub policy: ImagePlanCachePolicy,
    pub entries: usize,
    pub stage_entries: usize,
    /// None for LastPerStage: that policy does not measure encoded sizes.
    pub encoded_metadata_bytes: Option<usize>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub oversized_entries_skipped: u64,
    /// Whether this acquisition's entry is retained at snapshot time.
    pub retained: bool,
}

struct CachedStage<R: DeviceRuntime> {
    insertion: u64,
    options: ProgramPlanCompileOptions,
    component: Arc<CompiledTensorComponent<R>>,
    receipt: ImageStagePlanReceipt,
    encoded_bytes: usize,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedFamilyKey<'a> {
    pub stage: ImageStage,
    pub role: ImageStagePlanRole,
    pub batch_size: usize,
    pub family: &'a ModelFamilyId,
    pub metadata: &'a ExternalModelMetadataId,
    pub config: &'a serde_json::Value,
    pub profile: &'a NumericalProfileId,
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// Reuse only a family already validated by this pipeline's fixed built-in
    /// provider. Config identity is checked even on hits; compile options,
    /// materializer selection, full fingerprints and capacity checks follow.
    pub(super) fn prepare_family<P: ModelFamilyProvider>(
        &self,
        stage: ImageStage,
        role: ImageStagePlanRole,
        batch_size: usize,
        provider: P,
        config: &P::Config,
        profile: &NumericalProfileId,
    ) -> anyhow::Result<PreparedModelFamily> {
        let started = Instant::now();
        ensure!(batch_size > 0, "stage batch size must be positive");
        let raw_config = serde_json::to_value(config)?;
        let metadata = provider.validated_external_metadata_id(&raw_config, config)?;
        ensure!(
            provider.external_metadata_ids().contains(&metadata),
            "provider selected an undeclared external metadata identity"
        );
        let cached = self.plan_cache.prepared_family(
            self.config.plan_cache,
            PreparedFamilyKey {
                stage,
                role,
                batch_size,
                family: provider.family_id(),
                metadata: &metadata,
                config: &raw_config,
                profile,
            },
        )?;
        let reused = cached.is_some();
        let family = match cached {
            Some(family) => family,
            None => {
                TypedFamilyRegistration::new(provider).prepare_with_profile(&raw_config, profile)?
            }
        };
        tracing::debug!(
            ?stage,
            ?role,
            batch_size,
            prepared_family_reused = reused,
            prepare_family_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            "image family metadata preparation"
        );
        Ok(family)
    }
}
impl<R: DeviceRuntime> CachedStage<R> {
    fn matches(
        &self,
        stage: ImageStage,
        role: ImageStagePlanRole,
        batch_size: usize,
        fingerprint: &str,
        options: &ProgramPlanCompileOptions,
    ) -> bool {
        self.receipt.stage == stage
            && self.receipt.role == role
            && self.receipt.batch_size == batch_size
            && self.receipt.family_fingerprint == fingerprint
            && &self.options == options
    }
}

struct CacheState<R: DeviceRuntime> {
    policy: ImagePlanCachePolicy,
    // Front is least recently acquired. No native/device operation is performed
    // while this lock is held, including destruction of an evicted provider.
    entries: VecDeque<CachedStage<R>>,
    encoded_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    oversized: u64,
    next_insertion: u64,
}
impl<R: DeviceRuntime> CacheState<R> {
    fn new(policy: ImagePlanCachePolicy) -> Self {
        Self {
            policy,
            entries: VecDeque::new(),
            encoded_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            oversized: 0,
            next_insertion: 0,
        }
    }
    fn configure(&mut self, policy: ImagePlanCachePolicy) -> VecDeque<CachedStage<R>> {
        if self.policy == policy {
            VecDeque::new()
        } else {
            // Production pipeline configuration is immutable; this also makes
            // explicit policy changes in injected tests release old metadata.
            let mut next = Self::new(policy);
            next.next_insertion = self.next_insertion;
            std::mem::replace(self, next).entries
        }
    }
    fn snapshot(&self, stage: ImageStage, retained: bool) -> ImagePlanCacheReceipt {
        ImagePlanCacheReceipt {
            policy: self.policy,
            entries: self.entries.len(),
            stage_entries: self
                .entries
                .iter()
                .filter(|e| e.receipt.stage == stage)
                .count(),
            encoded_metadata_bytes: matches!(self.policy, ImagePlanCachePolicy::MultiShape { .. })
                .then_some(self.encoded_bytes),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            oversized_entries_skipped: self.oversized,
            retained,
        }
    }
    fn evict(&mut self, index: usize, removed: &mut VecDeque<CachedStage<R>>) {
        if let Some(entry) = self.entries.remove(index) {
            self.encoded_bytes -= entry.encoded_bytes;
            self.evictions = self.evictions.saturating_add(1);
            removed.push_back(entry);
        }
    }
}

pub(super) struct StagePlanCache<R: DeviceRuntime> {
    state: Arc<Mutex<CacheState<R>>>,
    candidate: Mutex<()>,
}

/// A candidate owns all newly retained stages until every plan and the combined
/// capacity proof succeed. Failure releases these cache references, including
/// their native leases. Existing callers' compiled Arcs remain valid even if
/// normal bounded-cache insertion evicted their old cache entry.
pub(super) struct CandidatePlanCache<'a, R: DeviceRuntime> {
    cache: &'a StagePlanCache<R>,
    first_insertion: u64,
    committed: bool,
    _serial: MutexGuard<'a, ()>,
}
impl<R: DeviceRuntime> CandidatePlanCache<'_, R> {
    /// Finish serialized compilation, retaining rollback ownership until the
    /// prepared request actually starts. Later candidates use disjoint insertion
    /// intervals, so cancelling this request cannot remove their new entries.
    pub(super) fn defer(mut self) -> anyhow::Result<PendingCandidatePlanCache<R>> {
        let end_insertion = self
            .cache
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?
            .next_insertion;
        let pending = PendingCandidatePlanCache {
            state: self.cache.state.clone(),
            first_insertion: self.first_insertion,
            end_insertion,
            committed: false,
        };
        self.committed = true;
        Ok(pending)
    }
}
impl<R: DeviceRuntime> Drop for CandidatePlanCache<'_, R> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        release_candidate_entries(&self.cache.state, self.first_insertion, u64::MAX);
    }
}

pub(super) struct PendingCandidatePlanCache<R: DeviceRuntime> {
    state: Arc<Mutex<CacheState<R>>>,
    first_insertion: u64,
    end_insertion: u64,
    committed: bool,
}
impl<R: DeviceRuntime> PendingCandidatePlanCache<R> {
    pub(super) fn commit(mut self) {
        self.committed = true;
    }
}
impl<R: DeviceRuntime> Drop for PendingCandidatePlanCache<R> {
    fn drop(&mut self) {
        if !self.committed {
            release_candidate_entries(&self.state, self.first_insertion, self.end_insertion);
        }
    }
}

fn release_candidate_entries<R: DeviceRuntime>(cache: &Mutex<CacheState<R>>, first: u64, end: u64) {
    // Cleanup also runs while unwinding. Preserve poison for future callers,
    // but release cache/native owners outside the accounting lock.
    let mut state = cache.lock().unwrap_or_else(|e| e.into_inner());
    let mut removed = VecDeque::new();
    let mut index = 0;
    while index < state.entries.len() {
        if (first..end).contains(&state.entries[index].insertion) {
            state.evict(index, &mut removed);
        } else {
            index += 1;
        }
    }
    drop(state);
    drop(removed);
}
impl<R: DeviceRuntime> StagePlanCache<R> {
    pub(super) fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(CacheState::new(ImagePlanCachePolicy::default()))),
            candidate: Mutex::new(()),
        }
    }
    pub(super) fn begin_candidate(&self) -> anyhow::Result<CandidatePlanCache<'_, R>> {
        // This separate guard serializes candidate preparations without holding
        // the cache state lock across compilation or native destruction.
        let serial = self
            .candidate
            .lock()
            .map_err(|_| anyhow::anyhow!("image candidate preparation poisoned"))?;
        let first_insertion = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?
            .next_insertion;
        Ok(CandidatePlanCache {
            cache: self,
            first_insertion,
            committed: false,
            _serial: serial,
        })
    }
    /// This metadata lookup neither adds an entry nor changes LRU/counters.
    /// Only the later exact plan/options acquisition counts as a cache hit.
    pub(super) fn prepared_family(
        &self,
        policy: ImagePlanCachePolicy,
        key: PreparedFamilyKey<'_>,
    ) -> anyhow::Result<Option<PreparedModelFamily>> {
        policy.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
        let removed = state.configure(policy);
        let component = (policy != ImagePlanCachePolicy::Disabled)
            .then(|| {
                state.entries.iter().rev().find(|entry| {
                    let family = entry.component.family();
                    entry.receipt.stage == key.stage
                        && entry.receipt.role == key.role
                        && entry.receipt.batch_size == key.batch_size
                        && family.family_id() == key.family
                        && family.external_metadata_id() == key.metadata
                        && family.canonical_config() == key.config
                        && &family.numerical_profile().id == key.profile
                })
            })
            .flatten()
            .map(|entry| entry.component.clone());
        drop(state);
        drop(removed);
        // Deep-clone only the selected prepared graph, outside the cache lock.
        // The existing metadata budget/Host payload distinction is unchanged.
        Ok(component.map(|component| component.family().clone()))
    }
    pub(super) fn get(
        &self,
        policy: ImagePlanCachePolicy,
        stage: ImageStage,
        role: ImageStagePlanRole,
        batch_size: usize,
        family_fingerprint: &str,
        options: &ProgramPlanCompileOptions,
    ) -> anyhow::Result<Option<(Arc<CompiledTensorComponent<R>>, ImageStagePlanReceipt)>> {
        policy.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
        let removed = state.configure(policy);
        let found = state
            .entries
            .iter()
            .position(|entry| entry.matches(stage, role, batch_size, family_fingerprint, options));
        let result = if let Some(index) = found {
            let entry = state.entries.remove(index).expect("located cache entry");
            let component = entry.component.clone();
            let mut receipt = entry.receipt.clone();
            state.entries.push_back(entry);
            state.hits = state.hits.saturating_add(1);
            receipt.plan_cache = Some(state.snapshot(stage, true));
            Some((component, receipt))
        } else {
            state.misses = state.misses.saturating_add(1);
            None
        };
        drop(state);
        drop(removed);
        Ok(result)
    }
    pub(super) fn insert(
        &self,
        policy: ImagePlanCachePolicy,
        component: Arc<CompiledTensorComponent<R>>,
        mut receipt: ImageStagePlanReceipt,
        options: ProgramPlanCompileOptions,
    ) -> anyhow::Result<ImagePlanCacheReceipt> {
        policy.validate()?;
        // Normalize receipt: neither timing digits nor an earlier cache snapshot
        // affect the charge. Count only on insertion, never on a cache hit.
        receipt.compile_wall_duration_us = 0;
        receipt.plan_cache_hit = false;
        receipt.plan_cache = None;
        let encoded = match policy {
            ImagePlanCachePolicy::MultiShape {
                maximum_encoded_metadata_bytes,
                ..
            } => encoded_size(
                &component,
                &receipt,
                &options,
                maximum_encoded_metadata_bytes,
            )?,
            _ => Some(0),
        };
        let stage = receipt.stage;
        let role = receipt.role;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
        let mut removed = state.configure(policy);
        let snapshot = if let Some(encoded_bytes) =
            encoded.filter(|_| policy != ImagePlanCachePolicy::Disabled)
        {
            let insertion = state.next_insertion;
            state.next_insertion = insertion
                .checked_add(1)
                .context("image plan cache insertion identity exhausted")?;
            // Concurrent preparations of the same key must not create duplicates;
            // each caller still owns its independently compiled component.
            if let Some(index) = state.entries.iter().position(|e| {
                e.matches(
                    stage,
                    role,
                    receipt.batch_size,
                    &receipt.family_fingerprint,
                    &options,
                )
            }) {
                state.evict(index, &mut removed);
            }
            let (stage_limit, byte_limit) = match policy {
                ImagePlanCachePolicy::MultiShape {
                    maximum_entries_per_stage,
                    maximum_encoded_metadata_bytes,
                } => (maximum_entries_per_stage, maximum_encoded_metadata_bytes),
                _ => (1, usize::MAX),
            };
            // LastPerStage is one slot per typed role. MultiShape retains its
            // original aggregate per-stage bound across all roles.
            let same_slot = |entry: &CachedStage<R>| {
                entry.receipt.stage == stage
                    && (policy != ImagePlanCachePolicy::LastPerStage || entry.receipt.role == role)
            };
            while state
                .entries
                .iter()
                .filter(|entry| same_slot(entry))
                .count()
                >= stage_limit
            {
                let index = state
                    .entries
                    .iter()
                    .position(same_slot)
                    .expect("stage at capacity");
                state.evict(index, &mut removed);
            }
            while state.encoded_bytes > byte_limit - encoded_bytes {
                state.evict(0, &mut removed);
            }
            state.encoded_bytes += encoded_bytes;
            state.entries.push_back(CachedStage {
                insertion,
                options,
                component,
                receipt,
                encoded_bytes,
            });
            state.snapshot(stage, true)
        } else {
            if encoded.is_none() {
                state.oversized = state.oversized.saturating_add(1);
            }
            state.snapshot(stage, false)
        };
        drop(state);
        drop(removed);
        Ok(snapshot)
    }
}

// Stream the serialized public metadata to a counter. No encoded plan Vec is
// constructed. Shared provider/native authorities are not serialized here.
fn encoded_size<R: DeviceRuntime>(
    component: &CompiledTensorComponent<R>,
    receipt: &ImageStagePlanReceipt,
    options: &ProgramPlanCompileOptions,
    limit: usize,
) -> anyhow::Result<Option<usize>> {
    let mut counter = LimitedCounter {
        bytes: 0,
        limit,
        exceeded: false,
    };
    let result = serde_json::to_writer(
        &mut counter,
        &(
            component.family(),
            component.executable().execution_plan(),
            component.executable().capabilities(),
            compile_options_projection(options),
            receipt,
        ),
    );
    if counter.exceeded {
        Ok(None)
    } else {
        result?;
        Ok(Some(counter.bytes))
    }
}
// Compile options deliberately have no runtime wire format. Account their
// public typed view without adding Serialize to runtime authority types. The
// cache key still compares the complete options, including private preferences
// and quality artifact bytes; those private fields are outside this size proxy.
fn compile_options_projection(options: &ProgramPlanCompileOptions) -> impl Serialize + '_ {
    let required_materializer = match options.weight_preparation() {
        WeightPreparation::Auto => None,
        WeightPreparation::Require(selection) => Some((
            selection.materializer_id(),
            selection.has_numeric_quality_artifact(),
        )),
    };
    (
        options.tensor_specs(),
        options.completion_retention().values(),
        required_materializer,
    )
}
struct LimitedCounter {
    bytes: usize,
    limit: usize,
    exceeded: bool,
}
impl io::Write for LimitedCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes {
            self.exceeded = true;
            return Err(io::Error::other(
                "encoded plan metadata exceeds cache limit",
            ));
        }
        self.bytes += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod encoded_options_tests {
    use super::*;

    #[test]
    fn public_projection_counts_retention_and_materializer_without_runtime_wire() {
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::new()).unwrap();
        let initial = serde_json::to_value(compile_options_projection(&options)).unwrap();
        assert!(initial[2].is_null());
        let output = ProgramValueId::new("value.cache.output").unwrap();
        options.retain_completion_value(output.clone());
        options.require_weight_materializer(
            WeightMaterializerId::new("weight-materializer.cache.exact").unwrap(),
        );
        let encoded = serde_json::to_vec(&compile_options_projection(&options)).unwrap();
        let view: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(view[1], serde_json::json!([output]));
        assert_eq!(
            view[2],
            serde_json::json!(["weight-materializer.cache.exact", false])
        );
        let mut exact = LimitedCounter {
            bytes: 0,
            limit: encoded.len(),
            exceeded: false,
        };
        serde_json::to_writer(&mut exact, &compile_options_projection(&options)).unwrap();
        assert_eq!(exact.bytes, encoded.len());
        let mut short = LimitedCounter {
            bytes: 0,
            limit: encoded.len() - 1,
            exceeded: false,
        };
        assert!(serde_json::to_writer(&mut short, &compile_options_projection(&options)).is_err());
        assert!(short.exceeded);
    }
}
