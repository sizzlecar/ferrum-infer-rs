//! Immutable GDN numerical constants and one synchronous future query.
//! No allocation authority, lane snapshot or executable permission is retained.
use super::super::super::cublas_api::{
    BoundGemmF16Cost, GemmCostQueryIdentity, PreparedGemmF16Cost,
};
use super::*;
use ferrum_interfaces::vnext::OperationCostWorkRow;

pub(in crate::backend::cuda::vnext_ops::transformer::attention) struct PreparedCostTemplate {
    shape: AttentionShape,
    precision: AttentionPrecision,
    projection: AttentionProjection,
    classes: PreparedKernelClasses,
    cuda: CudaAttentionShape,
    gemms: Option<[PreparedGemmF16Cost; 2]>,
}
impl PreparedCostTemplate {
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn new(
        shape: AttentionShape,
        precision: AttentionPrecision,
        projection: AttentionProjection,
    ) -> Result<Self, String> {
        let gemms = if matches!(precision, AttentionPrecision::F32MasterGgufF16Projections) {
            let matrix = |n, k| {
                PreparedGemmF16Cost::new(
                    checked_i32(n, "prepared GDN GEMM N")?,
                    checked_i32(k, "prepared GDN GEMM K")?,
                )
                .map_err(|error| error.to_string())
            };
            Some([
                matrix(shape.qkvzba_features, shape.hidden_size)?,
                matrix(shape.hidden_size, shape.value_features)?,
            ])
        } else {
            None
        };
        Ok(Self {
            shape,
            precision,
            projection,
            gemms,
            classes: PreparedKernelClasses::new(shape, precision)
                .ok_or("CUDA GDN kernel identity preparation failed")?,
            cuda: shape.cuda_shape()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn query<'a>(
        &'a self,
        rows: &'a [OperationCostWorkRow],
        tokens: u64,
        packed: bool,
        capabilities: GatedDeltaExecutionCapabilities,
        evidence: Option<ProjectionEvidence<'a>>,
        topology_required: bool,
    ) -> Result<Option<CheckedGdnQuery<'a>>, String> {
        let Some(evidence) = evidence else {
            return Ok(None);
        };
        if rows.is_empty() || (packed && rows.len() < 2) {
            return Err("GDN query has invalid participants".into());
        }
        let library = matches!(
            self.precision,
            AttentionPrecision::F32MasterGgufF16Projections
        );
        if library != matches!(evidence, ProjectionEvidence::Library(_))
            || !matches!(
                (self.projection, evidence),
                (AttentionProjection::F16, ProjectionEvidence::Library(_))
                    | (
                        AttentionProjection::Native { .. } | AttentionProjection::NativeQ8 { .. },
                        ProjectionEvidence::Native { .. }
                    )
            )
        {
            return Ok(None);
        }
        self.shape.validate_launch_extents(tokens)?;
        let layout = ScratchLayout::new(self.shape, tokens, rows.len(), self.projection)?;
        let bindings = StateBindingLayout::new(rows.len())?;
        let mut seen_tokens = 0_u64;
        let mut packed_form = None;
        let mut topology = topology_required
            .then(|| GdnTopologyDigest::new(rows.len()))
            .transpose()?;
        for row in rows {
            self.shape.validate_launch_extents(row.count.get())?;
            seen_tokens = seen_tokens
                .checked_add(row.count.get())
                .ok_or("GDN query tokens overflow")?;
            let form = capabilities
                .select(
                    row.count.get(),
                    GatedDeltaExecutionPreference::RecurrentScan,
                )
                .map_err(|error| error.to_string())?;
            if matches!(form, GatedDeltaExecutionForm::ChunkedScan(_)) {
                return Ok(None);
            }
            if packed
                && packed_form
                    .replace(form)
                    .is_some_and(|previous| previous != form)
            {
                return Ok(None);
            }
            if let Some(topology) = &mut topology {
                topology.push(row.count.get(), form)?;
            }
        }
        if seen_tokens != tokens {
            return Err("GDN query token population changed".into());
        }
        let participants =
            u32::try_from(rows.len()).map_err(|_| "GDN query participants exceed u32")?;
        let dispatches_for = |count| match evidence {
            ProjectionEvidence::Native { input, output } => combine_attention_dispatches(
                cost_route::native_projection_dispatches(
                    input,
                    count,
                    self.precision.quantizes_projections(),
                )?,
                cost_route::native_projection_dispatches(
                    output,
                    count,
                    self.precision.quantizes_projections(),
                )?,
            ),
            ProjectionEvidence::Library(_) => combine_attention_dispatches(1, 1),
        };
        let dispatches = if packed {
            dispatches_for(tokens)?
        } else {
            rows.iter().try_fold(0_u64, |total, row| {
                total
                    .checked_add(dispatches_for(row.count.get())?)
                    .ok_or_else(|| "GDN query dispatch count overflows".to_owned())
            })?
        };
        let transfers = (if packed { 1 } else { rows.len() as u64 })
            .checked_mul(combine_attention_transfers(0, 0)?)
            .ok_or("GDN query transfers overflow")?;
        let gemms = match (&self.gemms, evidence) {
            (Some(matrices), ProjectionEvidence::Library(identity)) => {
                let identity = GemmCostQueryIdentity::new(identity);
                Some([
                    matrices[0]
                        .bind(identity)
                        .map_err(|error| format!("{error:?}"))?,
                    matrices[1]
                        .bind(identity)
                        .map_err(|error| format!("{error:?}"))?,
                ])
            }
            (None, ProjectionEvidence::Native { .. }) => None,
            _ => return Ok(None),
        };
        Ok(Some(CheckedGdnQuery {
            template: self,
            rows,
            tokens,
            participants,
            packed,
            evidence,
            layout,
            bindings,
            gemms,
            dispatches,
            transfers,
            topology: topology.map(GdnTopologyDigest::finish).transpose()?,
        }))
    }
}

pub(in crate::backend::cuda::vnext_ops::transformer::attention) struct CheckedGdnQuery<'a> {
    template: &'a PreparedCostTemplate,
    rows: &'a [OperationCostWorkRow],
    tokens: u64,
    participants: u32,
    packed: bool,
    evidence: ProjectionEvidence<'a>,
    layout: ScratchLayout,
    bindings: StateBindingLayout,
    gemms: Option<[BoundGemmF16Cost<'a>; 2]>,
    dispatches: u64,
    transfers: u64,
    topology: Option<DeviceReusableExecutionTopologyFingerprint>,
}
impl CheckedGdnQuery<'_> {
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn participants(&self) -> u32 {
        self.participants
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn dispatches(&self) -> u64 {
        self.dispatches
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn transfers(&self) -> u64 {
        self.transfers
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn binding_offset(
        &self,
        row: usize,
    ) -> Result<u64, String> {
        self.bindings.offset(row)
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn compute(
        &self,
        capture: SloStructuredCostCapture,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        let leaves = std::iter::once((self.tokens, self.participants, true))
            .take(usize::from(self.packed))
            .chain(
                self.rows
                    .iter()
                    .filter(|_| !self.packed)
                    .map(|row| (row.count.get(), 1, false)),
            );
        super::compute_inner(
            self.template.shape,
            self.template.precision,
            self.template.projection,
            self.evidence,
            leaves,
            self.tokens,
            self.participants as usize,
            true,
            capture,
            Some(&self.template.classes),
            Some(self),
        )
    }
    pub(super) fn geometry(&self) -> (ScratchLayout, CudaAttentionShape) {
        (self.layout, self.template.cuda)
    }
    pub(super) fn append_library(
        &self,
        builder: &mut SelectedCommandCostBuilderV1,
        index: usize,
        identity: CublasHandleApiIdentity,
        rows: u64,
        columns: u64,
        reduction: u64,
    ) -> Option<()> {
        self.gemms
            .as_ref()?
            .get(index)?
            .append_selected(
                builder,
                i32::try_from(rows).ok()?,
                i32::try_from(columns).ok()?,
                i32::try_from(reduction).ok()?,
                identity,
            )
            .ok()
    }
    pub(in crate::backend::cuda::vnext_ops::transformer::attention) fn topology(
        &self,
        request: &impl ReusableExecutionTopologyView,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        if request.operation_id().as_str() != self.template.precision.operation()
            || request.participant_count() != self.rows.len()
            || request.immediate_tokens() != self.tokens
            || self
                .rows
                .iter()
                .enumerate()
                .any(|(index, row)| request.token_row(index) != Some(*row))
        {
            return Err(invalid_plan("GDN selected topology query identity changed"));
        }
        if !reusable_attention_address_scope(request)? {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
        self.topology
            .map(ReusableExecutionTopology::Dynamic)
            .ok_or_else(|| invalid_plan("GDN topology was not requested"))
    }
}
