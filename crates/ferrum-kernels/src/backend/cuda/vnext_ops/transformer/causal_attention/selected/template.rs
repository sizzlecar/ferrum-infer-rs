//! Plan-node constants and one-query geometry. Neither object owns a device
//! allocation, queue snapshot, current clock, or reusable execution permission.
use super::super::super::cublas_api::{
    BoundGemmF16Cost, GemmCostQueryIdentity, PreparedGemmF16Cost,
};
use super::*;
use std::ops::{Deref, DerefMut, Range};

pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) struct CostTemplate {
    shape: CausalAttentionShape,
    precision: CausalPrecision,
    projection: CausalProjection,
    cuda: CudaCausalAttentionShape,
    classes: Vec<(&'static str, [u32; 3], SelectedAlgorithmClassV1)>,
    gemms: [PreparedGemmF16Cost; 4],
    observation: std::sync::OnceLock<Option<ObservationRetention>>,
}

struct ObservationRetention {
    budget: std::sync::Arc<ferrum_interfaces::vnext::DeviceObservationTemplateBudget>,
    _lease: ferrum_interfaces::vnext::DeviceObservationTemplateReservation,
}

impl CostTemplate {
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn new(
        shape: CausalAttentionShape,
        precision: CausalPrecision,
        projection: CausalProjection,
    ) -> Result<Self, String> {
        let cuda = shape.cuda_shape()?;
        let mut entries = vec![
            (
                precision.rms_kernel(),
                [
                    super::super::super::rms_norm_threads(cuda.hidden_size),
                    1,
                    1,
                ],
            ),
            (PREPARE_FUNCTION, [WARP_THREADS, 1, 1]),
            (ATTENTION_FUNCTION, [WARP_THREADS, 1, 1]),
            (VARLEN_ADDRESSED_FUNCTION, [128, 1, 1]),
            (VARLEN_TILED_ADDRESSED_FUNCTION, [128, 1, 1]),
            (ATTENTION_GATE_FUNCTION, [THREADS_PER_BLOCK, 1, 1]),
            (precision.residual_kernel(), [THREADS_PER_BLOCK, 1, 1]),
        ];
        if let Some(entry) = precision.inplace_residual_kernel() {
            entries.push((entry, [THREADS_PER_BLOCK, 1, 1]));
        }
        if let Some(block) = grouped_fallback_block_threads(cuda) {
            entries.push((GROUPED_ATTENTION_FUNCTION, [block, 1, 1]));
        }
        let block = LaunchConfig::for_num_elems(1).block_dim;
        entries.push((
            "vnext_causal_gather_decode_lengths",
            [block.0, block.1, block.2],
        ));
        #[cfg(feature = "vllm-paged-attn-v2")]
        match shape.head_dim {
            128 => entries.extend([
                (
                    "vllm.paged_attention_v1_kernel.f16.h128.b16.t128",
                    [128, 1, 1],
                ),
                (
                    "vllm.paged_attention_v2_kernel.f16.h128.b16.t128.p512",
                    [128, 1, 1],
                ),
                (
                    "vllm.paged_attention_v2_reduce_kernel.f16.h128.t128.p512",
                    [128, 1, 1],
                ),
            ]),
            256 => entries.extend([
                (
                    "vllm.paged_attention_v1_kernel.f16.h256.b16.t128",
                    [128, 1, 1],
                ),
                (
                    "vllm.paged_attention_v2_kernel.f16.h256.b16.t128.p512",
                    [128, 1, 1],
                ),
                (
                    "vllm.paged_attention_v2_reduce_kernel.f16.h256.t128.p512",
                    [128, 1, 1],
                ),
            ]),
            _ => {}
        }
        let classes = entries
            .into_iter()
            .map(|(entry, block)| {
                Ok((
                    entry,
                    block,
                    class(entry, block).ok_or("causal kernel class is invalid")?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let matrix = |n, k| {
            PreparedGemmF16Cost::new(
                checked_i32(n, "prepared causal GEMM N")?,
                checked_i32(k, "prepared causal GEMM K")?,
            )
            .map_err(|error| error.to_string())
        };
        Ok(Self {
            shape,
            precision,
            projection,
            cuda,
            classes,
            gemms: [
                matrix(shape.query_projection_features, shape.hidden_size)?,
                matrix(shape.kv_features, shape.hidden_size)?,
                matrix(shape.kv_features, shape.hidden_size)?,
                matrix(shape.hidden_size, shape.query_features)?,
            ],
            observation: std::sync::OnceLock::new(),
        })
    }

    /// Share the original plan-node table, charging it once while any plan or
    /// queued actual recipe retains it. A full or different ledger only loses
    /// this optimization; the original current-work projector stays usable.
    pub(super) fn for_observation(
        self: &std::sync::Arc<Self>,
        shape: CausalAttentionShape,
        precision: CausalPrecision,
        projection: CausalProjection,
        budget: &std::sync::Arc<ferrum_interfaces::vnext::DeviceObservationTemplateBudget>,
    ) -> Option<std::sync::Arc<Self>> {
        let projection_matches = match (self.projection, projection) {
            (CausalProjection::F16, CausalProjection::F16) => true,
            (
                CausalProjection::Native {
                    transform_bytes_per_token: a,
                },
                CausalProjection::Native {
                    transform_bytes_per_token: b,
                },
            ) => a == b,
            _ => false,
        };
        if self.shape != shape || self.precision != precision || !projection_matches {
            return None;
        }
        let retention = self
            .observation
            .get_or_init(|| {
                let bytes = self.observation_payload_bytes()?;
                Some(ObservationRetention {
                    _lease: budget.reserve(bytes).ok()?,
                    budget: std::sync::Arc::clone(budget),
                })
            })
            .as_ref()?;
        std::sync::Arc::ptr_eq(&retention.budget, budget).then(|| std::sync::Arc::clone(self))
    }

    pub(super) fn observation_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.classes.capacity().checked_mul(std::mem::size_of::<(
                &'static str,
                [u32; 3],
                SelectedAlgorithmClassV1,
            )>())?)?
            .checked_add(2 * std::mem::size_of::<usize>())
    }

    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn geometry(
        &self,
        policy: AttentionExecutionPolicy,
        tokens: u64,
        participants: usize,
    ) -> Result<Geometry<'_>, String> {
        Geometry::new(
            self.shape,
            self.precision,
            self.projection,
            policy,
            self.cuda,
            tokens,
            participants,
            Some(self),
        )
    }

    fn class(&self, entry: &str, block: [u32; 3]) -> Option<SelectedAlgorithmClassV1> {
        self.classes.iter().find_map(|&(name, geometry, class)| {
            (name == entry && geometry == block).then_some(class)
        })
    }
}

/// Built once per query, before dynamic binding ranges are inspected. The
/// borrowed template is the exact provider/plan-node prepared-data binding.
#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) struct Geometry<'a> {
    pub(super) template: Option<&'a CostTemplate>,
    pub(super) shape: CausalAttentionShape,
    pub(super) precision: CausalPrecision,
    pub(super) projection: CausalProjection,
    pub(super) cuda: CudaCausalAttentionShape,
    pub(super) layout: ScratchLayout,
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) bindings: BindingLayout,
    pub(super) tokens: u64,
    pub(super) participants: usize,
}
impl<'a> Geometry<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        shape: CausalAttentionShape,
        precision: CausalPrecision,
        projection: CausalProjection,
        policy: AttentionExecutionPolicy,
        cuda: CudaCausalAttentionShape,
        tokens: u64,
        participants: usize,
        template: Option<&'a CostTemplate>,
    ) -> Result<Self, String> {
        Ok(Self {
            template,
            shape,
            precision,
            projection,
            cuda,
            tokens,
            participants,
            layout: ScratchLayout::for_participants(
                shape,
                tokens,
                participants,
                projection,
                policy,
            )?,
            bindings: BindingLayout::new(shape, participants)?,
        })
    }

    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn finish(
        self,
        rows: &'a [Row],
        packed: bool,
        projections: ProjectionWork<'a>,
    ) -> Option<Query<'a>> {
        if self.shape.int8_kv
            || rows.is_empty()
            || rows.len() != self.participants
            || (packed && rows.len() < 2)
            || rows
                .iter()
                .try_fold(0_u64, |sum, row| sum.checked_add(row.tokens))?
                != self.tokens
        {
            return None;
        }
        match (self.projection, projections) {
            (CausalProjection::Native { .. }, ProjectionWork::Native(parts))
                if parts.iter().all(|part| !part.is_empty()) => {}
            (CausalProjection::F16, ProjectionWork::DenseF16(_)) => {}
            _ => return None,
        }
        let batch_decode = batch_decode::eligible(rows.iter().map(|row| row.path), packed);
        let groups = if batch_decode {
            let paths = rows.iter().map(|row| row.path).collect::<Vec<_>>();
            let mut groups = Vec::new();
            for range in batch_decode::group_ranges(&paths) {
                let maximum = rows[range.clone()]
                    .iter()
                    .map(|row| row.envelope.sequence_capacity_tokens)
                    .max()?;
                let binding = BindingLayout {
                    slot_bytes: self.bindings.slot_bytes,
                    required_bytes: self.bindings.slot_bytes.checked_mul(range.len() as u64)?,
                };
                batch_decode::BatchDecode::dimensions(
                    range.len(),
                    binding,
                    maximum,
                    self.shape.query_heads,
                    self.shape.head_dim,
                )
                .ok()?;
                groups.push((range, maximum));
            }
            groups
        } else {
            Vec::new()
        };
        let extra = if packed {
            projections.extra_dispatches(self.tokens)?
        } else {
            rows.iter().try_fold(0_u64, |sum, row| {
                sum.checked_add(projections.extra_dispatches(row.tokens)?)
            })?
        };
        let dispatches = physical_dispatch_count(
            rows.iter().map(|row| row.path),
            self.shape.output_gate,
            self.shape.post_attention_norm,
            packed,
        )
        .checked_add(extra)?;
        Some(Query {
            geometry: self,
            rows,
            projections,
            packed,
            batch_decode,
            groups,
            dispatches,
        })
    }
}

/// Query-local numerical selection. Search and fresh replay independently
/// construct this object; it cannot supply an allocation or graph permission.
pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) struct Query<'a> {
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) geometry: Geometry<'a>,
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) rows: &'a [Row],
    pub(super) projections: ProjectionWork<'a>,
    pub(super) packed: bool,
    pub(super) batch_decode: bool,
    pub(super) groups: Vec<(Range<usize>, u64)>,
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) dispatches: u64,
}
impl Query<'_> {
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn compute(
        &self,
        capture: SloStructuredCostCapture,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        super::compute_query(self, capture)
    }
}

/// One evidence builder with immutable kernel classes and short-lived library
/// binding. No library observation is retained in the plan-node template.
pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) struct CostBuilder<'a> {
    inner: SelectedCommandCostBuilderV1,
    template: Option<&'a CostTemplate>,
    gemms: Option<[BoundGemmF16Cost<'a>; 4]>,
}
impl<'a> CostBuilder<'a> {
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn new(
        query: &Query<'a>,
        capture: SloStructuredCostCapture,
    ) -> Option<Self> {
        let inner = selected_cost::builder(capture, query.geometry.tokens)?;
        let template = query.geometry.template;
        let gemms = match (template, query.projections) {
            (Some(template), ProjectionWork::DenseF16(identity)) => {
                let identity = GemmCostQueryIdentity::new(identity);
                Some([
                    template.gemms[0].bind(identity).ok()?,
                    template.gemms[1].bind(identity).ok()?,
                    template.gemms[2].bind(identity).ok()?,
                    template.gemms[3].bind(identity).ok()?,
                ])
            }
            _ => None,
        };
        Some(Self {
            inner,
            template,
            gemms,
        })
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn class(
        &self,
        entry: &str,
        block: [u32; 3],
    ) -> Option<SelectedAlgorithmClassV1> {
        match self.template {
            Some(template) => template.class(entry, block),
            None => class(entry, block),
        }
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn gemm(
        &mut self,
        index: usize,
        rows: i32,
        output: i32,
        input: i32,
        identity: CublasHandleApiIdentity,
    ) -> Option<()> {
        match &self.gemms {
            Some(gemms) => gemms
                .get(index)?
                .append_selected(&mut self.inner, rows, output, input, identity)
                .ok(),
            None => GemmF16ApiPlan::new(rows, output, input)
                .ok()?
                .append_selected(&mut self.inner, identity)
                .ok(),
        }
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn finish(
        self,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        self.inner.finish().ok()
    }
}
impl Deref for CostBuilder<'_> {
    type Target = SelectedCommandCostBuilderV1;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for CostBuilder<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
