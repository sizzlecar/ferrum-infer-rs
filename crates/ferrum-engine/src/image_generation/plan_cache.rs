//! Compilation metadata only. Eviction releases the cache's Arc, never an
//! initialized session or another request's compiled component. Encoded size
//! is a serialization budget, not measured heap usage or Host payload bytes.
use super::*;
use std::{collections::VecDeque, io, sync::Mutex};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImagePlanCachePolicy {
    Disabled,
    /// One compiled plan per stage, without serialized-size accounting.
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
    options: ProgramPlanCompileOptions,
    component: Arc<CompiledTensorComponent<R>>,
    receipt: ImageStagePlanReceipt,
    encoded_bytes: usize,
}
impl<R: DeviceRuntime> CachedStage<R> {
    fn matches(
        &self,
        stage: ImageStage,
        batch_size: usize,
        fingerprint: &str,
        options: &ProgramPlanCompileOptions,
    ) -> bool {
        self.receipt.stage == stage
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
        }
    }
    fn configure(&mut self, policy: ImagePlanCachePolicy) -> VecDeque<CachedStage<R>> {
        if self.policy == policy {
            VecDeque::new()
        } else {
            // Production pipeline configuration is immutable; this also makes
            // explicit policy changes in injected tests release old metadata.
            std::mem::replace(self, Self::new(policy)).entries
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
    state: Mutex<CacheState<R>>,
}
impl<R: DeviceRuntime> StagePlanCache<R> {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(CacheState::new(ImagePlanCachePolicy::default())),
        }
    }
    pub(super) fn get(
        &self,
        policy: ImagePlanCachePolicy,
        stage: ImageStage,
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
            .position(|entry| entry.matches(stage, batch_size, family_fingerprint, options));
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
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
        let mut removed = state.configure(policy);
        let snapshot = if let Some(encoded_bytes) =
            encoded.filter(|_| policy != ImagePlanCachePolicy::Disabled)
        {
            // Concurrent preparations of the same key must not create duplicates;
            // each caller still owns its independently compiled component.
            if let Some(index) = state.entries.iter().position(|e| {
                e.matches(
                    stage,
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
            while state
                .entries
                .iter()
                .filter(|e| e.receipt.stage == stage)
                .count()
                >= stage_limit
            {
                let index = state
                    .entries
                    .iter()
                    .position(|e| e.receipt.stage == stage)
                    .expect("stage at capacity");
                state.evict(index, &mut removed);
            }
            while state.encoded_bytes > byte_limit - encoded_bytes {
                state.evict(0, &mut removed);
            }
            state.encoded_bytes += encoded_bytes;
            state.entries.push_back(CachedStage {
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
