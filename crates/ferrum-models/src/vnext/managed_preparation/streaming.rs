//! One-use bounded source normalization, armed by the exact selected plan.
use super::*;
use ferrum_quantization::{ManagedDenseArchiveStreamingSlot, PreparedDenseArchiveComponent};
use ferrum_runtime::{
    PlanHash, StaticWeightSourceContext, StaticWeightSourceRetention, WeightComponentRange,
    WeightComponentRangeRequest, WeightComponentStreamSpec, WeightId,
};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::atomic::{AtomicU8, Ordering},
};

const UNPREPARED: u8 = 0;
const PREPARING: u8 = 1;
const ARMED: u8 = 2;
const INITIALIZING: u8 = 3;
const READY: u8 = 4;
const FAILED: u8 = 5;

enum Component<'archive> {
    Stream(PreparedDenseArchiveComponent<'archive>),
    Full(WeightComponentSpec),
}

impl Component<'_> {
    fn spec(&self) -> &WeightComponentSpec {
        match self {
            Self::Stream(prepared) => prepared.component(),
            Self::Full(spec) => spec,
        }
    }
}

struct Recipe<'archive> {
    plan_hash: PlanHash,
    family_fingerprint: String,
    conversion_policy: Bf16F16ConversionPolicy,
    components: BTreeMap<WeightId, Component<'archive>>,
    logical_capacity: Option<NonZeroUsize>,
    maximum_backing_bytes: usize,
}

pub(super) struct StaticStreaming<'archive> {
    state: AtomicU8,
    recipe: OnceLock<Recipe<'archive>>,
    slot: OnceLock<ManagedDenseArchiveStreamingSlot<'archive>>,
}

impl<'archive> StaticStreaming<'archive> {
    pub(super) fn new() -> Self {
        Self {
            state: AtomicU8::new(UNPREPARED),
            recipe: OnceLock::new(),
            slot: OnceLock::new(),
        }
    }

    pub(super) fn prepare(
        &self,
        source: &ManagedProductionWeightSource<'archive>,
        plan: &ExecutionPlan,
        maximum_physical_bytes: usize,
        conversion_policy: Bf16F16ConversionPolicy,
    ) -> Result<(), VNextError> {
        self.state
            .compare_exchange(UNPREPARED, PREPARING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("streaming source must be prepared once before use"))?;
        let result = (|| {
            if source.mapped_cache.is_some() || maximum_physical_bytes == 0 {
                return Err(invalid(
                    "streaming needs a positive physical bound and no mapped cache",
                ));
            }
            let groups = host_backing::selected_groups(source, plan)?;
            let schema = source.prepared.family().stored_weight_schema()?;
            let mut components = BTreeMap::new();
            let mut maximum_full_stream_bytes = 0;
            let mut fallback_peak = 0usize;
            for ids in groups {
                let mut fallback_group = 0usize;
                for id in ids {
                    let component = schema
                        .components
                        .iter()
                        .find(|c| &c.id == id)
                        .ok_or_else(|| invalid("streaming group references an unknown source"))?;
                    let bytes = usize::try_from(source.component_backing_bytes(component)?)
                        .map_err(|_| invalid("source backing exceeds address space"))?;
                    if let Some(prepared) = source.source.prepare_dense_stream(component)? {
                        maximum_full_stream_bytes =
                            maximum_full_stream_bytes.max(prepared.full_bytes());
                        components.insert(id.clone(), Component::Stream(prepared));
                    } else {
                        if bytes != 0 {
                            fallback_group = fallback_group
                                .checked_add(
                                    source
                                        .host
                                        .backing_bytes(bytes)
                                        .map_err(|error| invalid(error.to_string()))?,
                                )
                                .ok_or_else(|| invalid("fallback group backing overflows"))?;
                        }
                        components.insert(id.clone(), Component::Full(component.clone()));
                    }
                }
                fallback_peak = fallback_peak.max(fallback_group);
            }
            let logical_capacity = if maximum_full_stream_bytes == 0 {
                None
            } else {
                Some(logical_capacity(
                    source.host,
                    maximum_full_stream_bytes,
                    maximum_physical_bytes,
                )?)
            };
            // The single persistent slot remains charged while a full-path
            // affine/tiny group is converted. Count both, including padding.
            let slot_backing = logical_capacity
                .map(|n| source.host.backing_bytes(n.get()))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?
                .unwrap_or(0);
            let maximum_backing_bytes = slot_backing
                .checked_add(fallback_peak)
                .ok_or_else(|| invalid("streaming plus fallback backing overflows"))?;
            self.recipe
                .set(Recipe {
                    plan_hash: plan.plan_hash().clone(),
                    family_fingerprint: plan.payload().prepared_family_fingerprint().to_owned(),
                    conversion_policy,
                    components,
                    logical_capacity,
                    maximum_backing_bytes,
                })
                .map_err(|_| invalid("streaming recipe is already bound"))?;
            Ok(())
        })();
        self.state.store(
            if result.is_ok() { ARMED } else { FAILED },
            Ordering::Release,
        );
        result
    }

    fn recipe(&self) -> Result<&Recipe<'archive>, VNextError> {
        if !matches!(self.state.load(Ordering::Acquire), ARMED | READY) {
            return Err(invalid(
                "streaming source has no available prepared installation",
            ));
        }
        self.recipe
            .get()
            .ok_or_else(|| invalid("streaming source has no recipe"))
    }

    pub(super) fn maximum_backing_bytes(&self, plan: &ExecutionPlan) -> Result<usize, VNextError> {
        let recipe = self.recipe()?;
        validate_plan(recipe, plan)?;
        Ok(recipe.maximum_backing_bytes)
    }

    pub(super) fn component_spec(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<Option<WeightComponentStreamSpec>, VNextError> {
        let recipe = self.recipe()?;
        match selected(recipe, component)? {
            Component::Stream(prepared) => stream_spec(recipe, prepared).map(Some),
            Component::Full(_) => Ok(None),
        }
    }

    pub(super) fn validate(
        &self,
        source: &ManagedProductionWeightSource<'archive>,
        context: &StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        self.state
            .compare_exchange(ARMED, INITIALIZING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid("streaming installation is one-use and already claimed or unavailable")
            })?;
        let result = (|| {
            let recipe = self
                .recipe
                .get()
                .ok_or_else(|| invalid("streaming source has no recipe"))?;
            validate_plan(recipe, context.plan())?;
            if !context.matches_host_account(source.host)
                || context.family().family_id() != source.prepared.family().family_id()
                || context.source_retention() != StaticWeightSourceRetention::FencedCopy
                || !context.supports_managed_source_upload()
            {
                return Err(invalid("streaming requires the original Host scope/family and managed FencedCopy upload"));
            }
            if let Some(declared) = recipe.logical_capacity {
                let runtime_maximum = usize::try_from(context.maximum_source_chunk_bytes())
                    .map_err(|_| invalid("streaming upload bound exceeds address space"))?;
                let actual = declared.get().min(runtime_maximum);
                let actual = NonZeroUsize::new(actual - actual % 2)
                    .ok_or_else(|| invalid("streaming upload bound cannot hold one F16 element"))?;
                let slot = source.source.bounded_streaming_slot(actual)?;
                self.slot
                    .set(slot)
                    .map_err(|_| invalid("streaming slot is already installed"))?;
            }
            Ok(())
        })();
        self.state.store(
            if result.is_ok() { READY } else { FAILED },
            Ordering::Release,
        );
        result
    }

    pub(super) fn validate_full_component(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<(), VNextError> {
        if self.state.load(Ordering::Acquire) != READY {
            return Err(invalid("streaming full-path source is not initialized"));
        }
        match selected(self.recipe()?, component)? {
            Component::Full(_) => Ok(()),
            Component::Stream(_) => Err(invalid(
                "streamed component cannot fall back to full allocation",
            )),
        }
    }

    pub(super) fn component_range(
        &self,
        request: &WeightComponentRangeRequest,
    ) -> Result<WeightComponentRange, VNextError> {
        if self.state.load(Ordering::Acquire) != READY {
            return Err(invalid("streaming range source is not initialized"));
        }
        let result = (|| {
            let recipe = self.recipe()?;
            let Component::Stream(prepared) = selected(recipe, request.component())? else {
                return Err(invalid(
                    "component does not declare a streaming range route",
                ));
            };
            if request.spec() != &stream_spec(recipe, prepared)? {
                return Err(invalid(
                    "range request differs from the exact prepared stream metadata",
                ));
            }
            let slot = self
                .slot
                .get()
                .ok_or_else(|| invalid("streaming slot is not installed"))?;
            let bytes = prepared.materialize_range_with_policy(
                slot,
                request.offset_bytes(),
                request.length_bytes(),
                recipe.conversion_policy,
            )?;
            WeightComponentRange::new(request, bytes)
        })();
        if result.is_err() {
            // Busy aliases, conversion failures and contract errors cannot be
            // retried through a fresh allocation in this installation.
            self.state.store(FAILED, Ordering::Release);
        }
        result
    }
}

fn validate_plan(recipe: &Recipe<'_>, plan: &ExecutionPlan) -> Result<(), VNextError> {
    if plan.plan_hash() != &recipe.plan_hash
        || plan.payload().prepared_family_fingerprint() != recipe.family_fingerprint
    {
        return Err(invalid(
            "streaming source differs from the exact selected plan/family",
        ));
    }
    Ok(())
}

fn selected<'a, 'archive>(
    recipe: &'a Recipe<'archive>,
    component: &WeightComponentSpec,
) -> Result<&'a Component<'archive>, VNextError> {
    recipe
        .components
        .get(&component.id)
        .filter(|entry| entry.spec() == component)
        .ok_or_else(|| invalid("component is not the exact selected streaming/fallback source"))
}

fn stream_spec(
    recipe: &Recipe<'_>,
    component: &PreparedDenseArchiveComponent<'_>,
) -> Result<WeightComponentStreamSpec, VNextError> {
    WeightComponentStreamSpec::new(
        component.component(),
        component.source_files().to_vec(),
        recipe
            .logical_capacity
            .ok_or_else(|| invalid("streaming recipe has no slot capacity"))?
            .get(),
    )
}

fn logical_capacity(
    host: &HostMemoryAccount,
    maximum_full_bytes: usize,
    physical_limit: usize,
) -> Result<NonZeroUsize, VNextError> {
    let alignment = host.allocation_alignment();
    let aligned_limit = physical_limit
        .checked_sub(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| invalid("streaming physical ceiling cannot hold Host padding"))?;
    let logical = maximum_full_bytes.min(aligned_limit);
    let logical = NonZeroUsize::new(logical - logical % 2)
        .ok_or_else(|| invalid("streaming physical ceiling cannot hold an F16 element"))?;
    if host
        .backing_bytes(logical.get())
        .map_err(|error| invalid(error.to_string()))?
        > physical_limit
    {
        return Err(invalid(
            "streaming slot exceeds the declared physical ceiling",
        ));
    }
    Ok(logical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

    #[test]
    fn logical_slot_bound_includes_real_alignment_rounding_and_padding() {
        for alignment in [1, 2, 64, 4096] {
            let host = HostMemoryAccount::with_allocation_alignment(
                HostMemoryLimits::new(
                    1 << 20,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 0,
                        transfer_staging_bytes: 0,
                        output_bytes: 0,
                        weight_backing_bytes: 1 << 20,
                    },
                )
                .unwrap(),
                alignment,
            )
            .unwrap();
            let exact = host.backing_bytes(10).unwrap();
            assert_eq!(logical_capacity(&host, 10, exact).unwrap().get(), 10);
            assert!(logical_capacity(&host, 10, host.backing_bytes(2).unwrap() - 1).is_err());
            let bounded = logical_capacity(&host, 1 << 18, 16384).unwrap().get();
            assert!(host.backing_bytes(bounded).unwrap() <= 16384);
            assert_eq!(bounded % 2, 0);
            assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        }
    }
}
