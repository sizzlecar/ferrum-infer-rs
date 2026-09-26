//! Fixed numeric recipes from actual encoders. No device buffer or lease is
//! retained here. Only successful native capture makes one resident.
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::DeviceReplayCostWork;
use ferrum_types::SloStructuredCostCapture;
use std::sync::Arc;

pub(crate) struct CudaReplayCostRecipe {
    captured_work: DeviceReplayCostWork,
    kind: RecipeKind,
}

enum RecipeKind {
    Primitive {
        primitive: cost_route::Primitive,
        hidden: u64,
    },
    NativeFfn(native_swiglu::replay_cost::Recipe),
    DenseFfn {
        shape: dense_swiglu_api::Shape,
        identity: cublas_api::CublasHandleApiIdentity,
    },
}
impl std::fmt::Debug for CudaReplayCostRecipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaReplayCostRecipe")
            .field("captured_work", &self.captured_work)
            .finish_non_exhaustive()
    }
}
impl CudaReplayCostRecipe {
    pub(super) fn primitive(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        primitive: cost_route::Primitive,
        hidden: u64,
        capture: SloStructuredCostCapture,
    ) -> Option<Arc<Self>> {
        if capture.is_disabled()
            || !matches!(
                primitive,
                cost_route::Primitive::RmsNorm { .. } | cost_route::Primitive::ResidualAdd { .. }
            )
        {
            return None;
        }
        let recipe = Self {
            captured_work: invocation.replay_cost_work()?,
            kind: RecipeKind::Primitive { primitive, hidden },
        };
        recipe.project(&recipe.captured_work)?;
        Some(Arc::new(recipe))
    }

    pub(super) fn native(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        numeric: native_swiglu::replay_cost::Recipe,
    ) -> Option<Arc<Self>> {
        let recipe = Self {
            captured_work: invocation.replay_cost_work()?,
            kind: RecipeKind::NativeFfn(numeric),
        };
        recipe.project(&recipe.captured_work)?;
        Some(Arc::new(recipe))
    }

    pub(super) fn dense(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        shape: dense_swiglu_api::Shape,
        identity: cublas_api::CublasHandleApiIdentity,
    ) -> Option<Arc<Self>> {
        let recipe = Self {
            captured_work: invocation.replay_cost_work()?,
            kind: RecipeKind::DenseFfn { shape, identity },
        };
        recipe.project(&recipe.captured_work)?;
        Some(Arc::new(recipe))
    }

    pub(crate) fn captured_evidence(&self) -> Option<SelectedCommandCostEvidenceV1> {
        self.project(&self.captured_work)
    }

    pub(crate) fn project(
        &self,
        current: &DeviceReplayCostWork,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        // Packed kernels fix total M, not the individual row partition.
        // Token values/source positions and a legal equal-total partition may
        // change; current resource windows were independently revalidated.
        if current.tokens() != self.captured_work.tokens()
            || current.participant_ranges().len() != self.captured_work.participant_ranges().len()
        {
            return None;
        }
        match &self.kind {
            RecipeKind::Primitive { primitive, hidden } => cost_route::selected(
                *primitive,
                current.tokens(),
                *hidden,
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::NativeFfn(recipe) => {
                recipe.project(current.tokens(), current.participant_ranges())
            }
            RecipeKind::DenseFfn { shape, identity } => shape.project(current.tokens(), *identity),
        }
    }
}
