//! Fixed numeric recipes from actual encoders. No device buffer or lease is
//! retained here. Only successful native capture makes one resident.
use super::super::native_blocks::weights;
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::{
    DeviceObservationDiagnostic, DeviceObservationFailureStage, DeviceReplayCostWork,
    FrozenObservationInput,
};
use ferrum_types::SloStructuredCostCapture;
use std::sync::Arc;

pub(crate) struct CudaReplayCostRecipe {
    budget: Arc<ferrum_interfaces::vnext::DeviceObservationTemplateBudget>,
    captured_work: DeviceReplayCostWork,
    captured_input: FrozenObservationInput,
    kind: RecipeKind,
    _construction: ferrum_interfaces::vnext::DeviceObservationTemplateReservation,
}

enum RecipeKind {
    AttentionBindings {
        participants: usize,
    },
    CausalBindings {
        bytes: Box<[u64]>,
    },
    Primitive {
        primitive: cost_route::Primitive,
        hidden: u64,
    },
    NativeFfn(native_swiglu::replay_cost::Recipe),
    Attention(attention::selected::Recipe),
    Causal(causal_attention::selected::Recipe),
    DenseEmbedding {
        rows: Box<[(u64, u64, u64)]>,
    },
    NativeEmbedding {
        part: weights::MatrixPart,
        counts: Box<[u64]>,
        element: ElementType,
        scratch: u64,
    },
    NativeHead {
        parts: Box<[weights::MatrixPart]>,
        rows: Box<[u32]>,
        stride: u32,
        element: ElementType,
        scratch: u64,
    },
    Argmax {
        precision: super::super::ArgmaxPrecision,
        rows: Box<[(i32, i32)]>,
    },
    RnFragmentFfn {
        shape: rn_fragment_swiglu::Shape,
        identity: Option<cublas_api::CublasHandleApiIdentity>,
    },
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
    pub(crate) fn budget(&self) -> &Arc<ferrum_interfaces::vnext::DeviceObservationTemplateBudget> {
        &self.budget
    }
    pub(crate) fn library_identity(&self) -> Option<cublas_api::CublasHandleApiIdentity> {
        match &self.kind {
            RecipeKind::DenseFfn { identity, .. } => Some(*identity),
            RecipeKind::RnFragmentFfn { identity, .. } => *identity,
            RecipeKind::Attention(recipe) => recipe.library_identity(),
            RecipeKind::Causal(recipe) => recipe.library_identity(),
            _ => None,
        }
    }
    pub(super) fn primitive(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        primitive: cost_route::Primitive,
        hidden: u64,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled()
            || !matches!(
                primitive,
                cost_route::Primitive::RmsNorm { .. } | cost_route::Primitive::ResidualAdd { .. }
            )
        {
            return Ok(None);
        }
        Self::build(invocation, 0, || {
            Some(RecipeKind::Primitive { primitive, hidden })
        })
    }

    pub(super) fn native(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        numeric: native_swiglu::replay_cost::Recipe,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        Self::build(invocation, 0, || Some(RecipeKind::NativeFfn(numeric)))
    }
    pub(super) fn dense(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        shape: dense_swiglu_api::Shape,
        identity: cublas_api::CublasHandleApiIdentity,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        Self::build(invocation, 0, || {
            Some(RecipeKind::DenseFfn { shape, identity })
        })
    }
    pub(super) fn rn_fragment(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        shape: rn_fragment_swiglu::Shape,
        identity: Option<cublas_api::CublasHandleApiIdentity>,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        Self::build(invocation, 0, || {
            Some(RecipeKind::RnFragmentFfn { shape, identity })
        })
    }

    pub(crate) fn captured_evidence(&self) -> Option<SelectedCommandCostEvidenceV1> {
        self.project_observation(&self.captured_input)
    }
    pub(crate) fn captured_input(&self) -> &FrozenObservationInput {
        &self.captured_input
    }
    pub(crate) fn retained_payload_bytes(&self) -> Option<usize> {
        let extra = match &self.kind {
            RecipeKind::CausalBindings { bytes } => {
                bytes.len().checked_mul(std::mem::size_of::<u64>())?
            }
            RecipeKind::NativeFfn(recipe) => recipe.retained_payload_bytes()?,
            RecipeKind::Attention(recipe) => recipe.retained_payload_bytes(),
            RecipeKind::Causal(recipe) => recipe.retained_payload_bytes(),
            RecipeKind::DenseEmbedding { rows } => {
                rows.len()
                    .checked_mul(std::mem::size_of::<(u64, u64, u64)>())?
            }
            RecipeKind::NativeEmbedding { part, counts, .. } => {
                weights::retained_payload_bytes(std::slice::from_ref(part))?
                    .checked_add(counts.len().checked_mul(std::mem::size_of::<u64>())?)?
            }
            RecipeKind::NativeHead { parts, rows, .. } => {
                weights::retained_payload_bytes(parts)?
                    .checked_add(rows.len().checked_mul(std::mem::size_of::<u32>())?)?
            }
            RecipeKind::Argmax { rows, .. } => {
                rows.len().checked_mul(std::mem::size_of::<(i32, i32)>())?
            }
            _ => 0,
        };
        std::mem::size_of::<Self>()
            .checked_add(self.captured_work.retained_payload_bytes()?)?
            .checked_add(self.captured_input.retained_payload_bytes()?)?
            .checked_add(extra)
    }
    pub(crate) fn projection_scratch_bytes(&self) -> Option<usize> {
        match &self.kind {
            RecipeKind::Causal(recipe) => recipe.projection_scratch_bytes(),
            _ => Some(0),
        }
    }
    pub(crate) fn project_observation(
        &self,
        input: &FrozenObservationInput,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        if let RecipeKind::Causal(recipe) = &self.kind {
            return recipe.project(input);
        }
        self.project(&input.replay_cost_work()?)
    }
    pub(super) fn attention(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        recipe: attention::selected::Recipe,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        Self::build(invocation, 0, || Some(RecipeKind::Attention(recipe)))
    }
    pub(super) fn causal(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        recipe: causal_attention::selected::Recipe,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        Self::build(invocation, 0, || Some(RecipeKind::Causal(recipe)))
    }
    pub(in crate::backend::cuda::vnext_ops) fn dense_embedding(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        rows: impl ExactSizeIterator<Item = (u64, u64, u64)>,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        let extra = (|| {
            Some(
                rows.len()
                    .checked_mul(std::mem::size_of::<(u64, u64, u64)>())?,
            )
        })()
        .ok_or_else(|| Self::recipe_failure("recipe.copied_payload"))?;
        Self::build(invocation, extra, || {
            Some(RecipeKind::DenseEmbedding {
                rows: rows.collect(),
            })
        })
    }
    pub(in crate::backend::cuda::vnext_ops) fn native_embedding(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        part: &weights::MatrixPart,
        counts: impl ExactSizeIterator<Item = u64>,
        element: ElementType,
        scratch: u64,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        let extra = (|| {
            Some(
                weights::retained_payload_bytes(std::slice::from_ref(part))?
                    .checked_add(counts.len().checked_mul(std::mem::size_of::<u64>())?)?,
            )
        })()
        .ok_or_else(|| Self::recipe_failure("recipe.copied_payload"))?;
        Self::build(invocation, extra, || {
            Some(RecipeKind::NativeEmbedding {
                part: part.clone(),
                counts: counts.collect(),
                element,
                scratch,
            })
        })
    }
    pub(in crate::backend::cuda::vnext_ops) fn native_head(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        parts: &[weights::MatrixPart],
        rows: impl ExactSizeIterator<Item = u32>,
        stride: u32,
        element: ElementType,
        scratch: u64,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        let extra = (|| {
            Some(
                weights::retained_payload_bytes(parts)?
                    .checked_add(rows.len().checked_mul(std::mem::size_of::<u32>())?)?,
            )
        })()
        .ok_or_else(|| Self::recipe_failure("recipe.copied_payload"))?;
        Self::build(invocation, extra, || {
            Some(RecipeKind::NativeHead {
                parts: parts.to_vec().into_boxed_slice(),
                rows: rows.collect(),
                stride,
                element,
                scratch,
            })
        })
    }
    pub(in crate::backend::cuda::vnext_ops) fn argmax(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        precision: super::super::ArgmaxPrecision,
        rows: impl ExactSizeIterator<Item = (i32, i32)>,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        let extra = (|| Some(rows.len().checked_mul(std::mem::size_of::<(i32, i32)>())?))()
            .ok_or_else(|| Self::recipe_failure("recipe.copied_payload"))?;
        Self::build(invocation, extra, || {
            Some(RecipeKind::Argmax {
                precision,
                rows: rows.collect(),
            })
        })
    }
    pub(super) fn attention_bindings(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        Self::build(invocation, 0, || {
            Some(RecipeKind::AttentionBindings {
                participants: invocation.participants().len(),
            })
        })
    }
    pub(super) fn causal_bindings(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        bytes: impl ExactSizeIterator<Item = u64>,
        capture: SloStructuredCostCapture,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        if capture.is_disabled() {
            return Ok(None);
        }
        let extra = (|| Some(bytes.len().checked_mul(std::mem::size_of::<u64>())?))()
            .ok_or_else(|| Self::recipe_failure("recipe.copied_payload"))?;
        Self::build(invocation, extra, || {
            Some(RecipeKind::CausalBindings {
                bytes: bytes.collect(),
            })
        })
    }
    // The local closure runs once while this reservation is held; no closure,
    // runtime, buffer or resource authority is stored in the resulting recipe.
    fn build(
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        copied_payload: usize,
        freeze: impl FnOnce() -> Option<RecipeKind>,
    ) -> Result<Option<Arc<Self>>, DeviceObservationDiagnostic> {
        let budget = Arc::clone(
            invocation
                .observation_template_budget()
                .ok_or_else(|| Self::recipe_failure("recipe.template_budget_absent"))?,
        );
        let upper = (|| {
            let ranges = invocation
                .work_shape()
                .participant_token_ranges()
                .len()
                .checked_mul(3)?
                .checked_mul(std::mem::size_of::<std::ops::Range<u64>>())?;
            std::mem::size_of::<Self>()
                .checked_add(copied_payload.checked_add(ranges)?.checked_mul(2)?)
        })()
        .ok_or_else(|| Self::recipe_failure("recipe.construction_bound"))?;
        let construction = budget.reserve_with_diagnostic(upper, "recipe.build.reserve")?;
        Ok(Some(Arc::new(Self {
            budget,
            _construction: construction,
            captured_work: invocation
                .replay_cost_work()
                .ok_or_else(|| Self::recipe_failure("recipe.captured_work"))?,
            captured_input: FrozenObservationInput::from_work_shape(invocation.work_shape())
                .ok_or_else(|| Self::recipe_failure("recipe.captured_input"))?,
            kind: freeze().ok_or_else(|| Self::recipe_failure("recipe.freeze"))?,
        })))
    }
    fn recipe_failure(site: &'static str) -> DeviceObservationDiagnostic {
        DeviceObservationDiagnostic::new(DeviceObservationFailureStage::Recipe, site, None)
    }
    pub(crate) fn projection_site(&self) -> &'static str {
        match &self.kind {
            RecipeKind::AttentionBindings { .. } => "recipe.attention_bindings.project",
            RecipeKind::CausalBindings { .. } => "recipe.causal_bindings.project",
            RecipeKind::Primitive { .. } => "recipe.primitive.project",
            RecipeKind::NativeFfn(_) => "recipe.native_ffn.project",
            RecipeKind::Attention(_) => "recipe.attention.project",
            RecipeKind::Causal(_) => "recipe.causal.project",
            RecipeKind::DenseEmbedding { .. } => "recipe.dense_embedding.project",
            RecipeKind::NativeEmbedding { .. } => "recipe.native_embedding.project",
            RecipeKind::NativeHead { .. } => "recipe.native_head.project",
            RecipeKind::Argmax { .. } => "recipe.argmax.project",
            RecipeKind::RnFragmentFfn { .. } => "recipe.rn_fragment.project",
            RecipeKind::DenseFfn { .. } => "recipe.dense_ffn.project",
        }
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
            RecipeKind::AttentionBindings { participants } => attention::selected::bindings(
                *participants,
                current.tokens(),
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::CausalBindings { bytes } => causal_attention::selected::binding_evidence(
                bytes.iter().copied(),
                current.tokens(),
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::Primitive { primitive, hidden } => cost_route::selected(
                *primitive,
                current.tokens(),
                *hidden,
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::DenseEmbedding { rows } => {
                if rows.len() != current.participant_ranges().len()
                    || rows
                        .iter()
                        .zip(current.participant_ranges())
                        .any(|(row, range)| range.end.checked_sub(range.start) != Some(row.2))
                {
                    return None;
                }
                super::super::embedding::selected_dense(
                    rows.iter().copied(),
                    current.tokens(),
                    SloStructuredCostCapture::HostSettledV1,
                )
            }
            RecipeKind::NativeEmbedding {
                part,
                counts,
                element,
                scratch,
            } => {
                if counts.len() != current.participant_ranges().len()
                    || counts
                        .iter()
                        .zip(current.participant_ranges())
                        .any(|(count, range)| range.end.checked_sub(range.start) != Some(*count))
                {
                    return None;
                }
                super::super::native_blocks::embedding::selected(
                    part,
                    counts.iter().copied(),
                    current.tokens(),
                    *element,
                    *scratch,
                    SloStructuredCostCapture::HostSettledV1,
                )
            }
            RecipeKind::NativeHead {
                parts,
                rows,
                stride,
                element,
                scratch,
            } => super::super::native_blocks::selected::linear(
                parts,
                rows.iter().copied(),
                current.participant_ranges().len() as u64,
                *stride,
                *element,
                *scratch,
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::Argmax { precision, rows } => super::super::selection::selected::evidence(
                *precision,
                u32::try_from(current.participant_ranges().len()).ok()?,
                rows.iter().copied(),
                SloStructuredCostCapture::HostSettledV1,
            ),
            RecipeKind::Attention(recipe) => recipe.project(current),
            // Legacy direct replay queries cannot supply current source positions.
            RecipeKind::Causal(_) => None,
            RecipeKind::NativeFfn(recipe) => {
                recipe.project(current.tokens(), current.participant_ranges())
            }
            RecipeKind::DenseFfn { shape, identity } => shape.project(current.tokens(), *identity),
            RecipeKind::RnFragmentFfn { shape, identity } => {
                shape.project(current.tokens(), *identity)
            }
        }
    }
}
