//! An explicitly armed, one-use installation lease. No process-wide cache.
use super::*;
use ferrum_quantization::ManagedDenseArchiveInstallation;
use ferrum_runtime::{PlanHash, StaticWeightSourceContext, StaticWeightSourceRetention, WeightId};
use std::{
    collections::BTreeSet,
    num::NonZeroUsize,
    sync::atomic::{AtomicU8, Ordering},
};

const FRESH: u8 = 0;
const FRESH_USED: u8 = 1;
const ARMING: u8 = 2;
const ARMED: u8 = 3;
const INITIALIZING: u8 = 4;
const READY: u8 = 5;
const FAILED: u8 = 6;

struct Recipe {
    plan_hash: PlanHash,
    family_fingerprint: String,
    selected: BTreeSet<WeightId>,
    maximum_component_bytes: Option<NonZeroUsize>,
}

pub(super) struct StaticReuse<'archive> {
    state: AtomicU8,
    recipe: OnceLock<Recipe>,
    installation: OnceLock<ManagedDenseArchiveInstallation<'archive>>,
}

impl<'archive> StaticReuse<'archive> {
    pub(super) fn new() -> Self {
        Self {
            state: AtomicU8::new(FRESH),
            recipe: OnceLock::new(),
            installation: OnceLock::new(),
        }
    }

    pub(super) fn prepare(
        &self,
        source: &ManagedProductionWeightSource<'archive>,
        plan: &ExecutionPlan,
    ) -> Result<(), VNextError> {
        if source.mapped_cache.is_some() {
            return Err(invalid(
                "reusable backing cannot replace configured process-mapped weights",
            ));
        }
        self.state
            .compare_exchange(FRESH, ARMING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid("reusable backing must be armed once before source materialization")
            })?;
        let result = (|| {
            let groups = host_backing::selected_groups(source, plan)?;
            let schema = source.prepared.family().stored_weight_schema()?;
            let mut selected = BTreeSet::new();
            let mut maximum = 0usize;
            for ids in groups {
                let mut converted_owners = 0usize;
                for id in ids {
                    let component = schema
                        .components
                        .iter()
                        .find(|c| &c.id == id)
                        .ok_or_else(|| invalid("selected group references an unknown source"))?;
                    selected.insert(id.clone());
                    let bytes = usize::try_from(source.component_backing_bytes(component)?)
                        .map_err(|_| invalid("reusable source exceeds Host address space"))?;
                    if bytes != 0 {
                        converted_owners += 1;
                        maximum = maximum.max(bytes);
                    }
                }
                if converted_owners > 1 {
                    return Err(invalid(
                        "single-slot reuse cannot hold multiple converted owners in one group",
                    ));
                }
            }
            self.recipe
                .set(Recipe {
                    plan_hash: plan.plan_hash().clone(),
                    family_fingerprint: plan.payload().prepared_family_fingerprint().to_owned(),
                    selected,
                    maximum_component_bytes: NonZeroUsize::new(maximum),
                })
                .map_err(|_| invalid("reusable source recipe is already bound"))?;
            self.state
                .compare_exchange(ARMING, ARMED, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| invalid("reusable source was used before preparation completed"))?;
            Ok(())
        })();
        if result.is_err() {
            self.state.store(FAILED, Ordering::Release);
        }
        result
    }

    pub(super) fn validate(
        &self,
        source: &ManagedProductionWeightSource<'archive>,
        context: &StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        // Core accepting the fresh route also seals the policy: a concurrent
        // caller cannot arm reuse after validation but before the first write.
        let state = match self.state.compare_exchange(
            FRESH,
            FRESH_USED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(previous) | Err(previous) => previous,
        };
        match state {
            FRESH | FRESH_USED => return Ok(()),
            ARMED => {}
            ARMING => {
                // The armer may already have published ARMED, or another
                // initializer may have claimed it. Never overwrite that newer
                // state based on this earlier observation of ARMING.
                let _ = self.state.compare_exchange(
                    ARMING,
                    FAILED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Err(invalid("reusable source preparation has not completed"));
            }
            _ => {
                return Err(invalid(
                    "reusable source installation is one-use and is not available",
                ))
            }
        }
        self.state
            .compare_exchange(ARMED, INITIALIZING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("reusable source installation is already claimed"))?;
        let result = (|| {
            let recipe = self
                .recipe
                .get()
                .ok_or_else(|| invalid("reusable source has no recipe"))?;
            if context.source_retention() != StaticWeightSourceRetention::FencedCopy {
                return Err(invalid(
                    "reusable source requires a copy route with fence-bounded source release",
                ));
            }
            if !context.matches_host_account(source.host) {
                return Err(invalid(
                    "reusable source belongs to a different Host parent or admission scope",
                ));
            }
            if context.plan().plan_hash() != &recipe.plan_hash
                || context.plan().payload().prepared_family_fingerprint()
                    != recipe.family_fingerprint
                || context.family().family_id() != source.prepared.family().family_id()
            {
                return Err(invalid(
                    "reusable source differs from the actual validated plan or family",
                ));
            }
            if let Some(maximum) = recipe.maximum_component_bytes {
                let installation = source.source.reusable_installation(maximum)?;
                self.installation
                    .set(installation)
                    .map_err(|_| invalid("reusable source already owns an installation"))?;
            }
            Ok(())
        })();
        self.state.store(
            if result.is_ok() { READY } else { FAILED },
            Ordering::Release,
        );
        result
    }

    pub(super) fn uses_reuse(&self) -> Result<bool, VNextError> {
        match self
            .state
            .compare_exchange(FRESH, FRESH_USED, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) | Err(FRESH_USED) => Ok(false),
            Err(READY) => Ok(true),
            Err(_) => Err(invalid(
                "reusable source is not validated for an active installation",
            )),
        }
    }

    pub(super) fn component<'a>(
        &'a self,
        fresh: &'a ManagedDenseArchiveSource<'archive>,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let recipe = self
            .recipe
            .get()
            .ok_or_else(|| invalid("reusable source has no recipe"))?;
        if !recipe.selected.contains(&component.id) {
            return Err(invalid(
                "component has no selected placement in this reusable installation",
            ));
        }
        match self.installation.get() {
            Some(installation) => installation.component(component),
            None => fresh.component(component), // every selected source is borrowed
        }
    }
}
