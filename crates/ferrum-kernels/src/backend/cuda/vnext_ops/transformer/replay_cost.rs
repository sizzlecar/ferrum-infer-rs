//! Fixed numeric recipes from actual encoders. No device buffer or lease is
//! retained here. Only successful native capture makes one resident.
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::DeviceReplayCostWork;
use ferrum_types::SloStructuredCostCapture;
use std::sync::Arc;

pub(crate) struct CudaReplayCostRecipe {
    captured_work: DeviceReplayCostWork,
    primitive: cost_route::Primitive,
    hidden: u64,
}
impl std::fmt::Debug for CudaReplayCostRecipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaReplayCostRecipe")
            .field("captured_work", &self.captured_work)
            .field("hidden", &self.hidden)
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
            primitive,
            hidden,
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
        // Packed primitives fix total M, not the individual row partition.
        // Token values/source positions and a legal equal-total partition may
        // change; current resource windows were independently revalidated.
        if current.tokens() != self.captured_work.tokens()
            || current.participant_ranges().len() != self.captured_work.participant_ranges().len()
        {
            return None;
        }
        cost_route::selected(
            self.primitive,
            current.tokens(),
            self.hidden,
            SloStructuredCostCapture::HostSettledV1,
        )
    }
}
