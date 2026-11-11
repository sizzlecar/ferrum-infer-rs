use super::*;
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageCurveBlockRowRun {
    length: u64,
    time_index: u64,
    modality: u32,
}
impl<'de> Deserialize<'de> for PH3ImageCurveBlockRowRun {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            length: u64,
            time_index: u64,
            modality: u32,
        }
        let w = Wire::deserialize(d)?;
        if w.length == 0 || w.modality > 2 {
            return Err(serde::de::Error::custom("invalid block row run"));
        }
        Ok(Self {
            length: w.length,
            time_index: w.time_index,
            modality: w.modality,
        })
    }
}
impl PH3ImageCurveBlockRowRun {
    pub fn new(
        length: u64,
        time_index: u64,
        modality: PH3ImageModality,
    ) -> Result<Self, VNextError> {
        if length == 0 {
            return Err(invalid("row run must be nonempty"));
        }
        Ok(Self {
            length,
            time_index,
            modality: modality as u32,
        })
    }
    pub fn length(&self) -> u64 {
        self.length
    }
    pub fn time_index(&self) -> u64 {
        self.time_index
    }
    pub fn modality(&self) -> PH3ImageModality {
        match self.modality {
            0 => PH3ImageModality::Image,
            1 => PH3ImageModality::Text,
            2 => PH3ImageModality::Audio,
            _ => unreachable!("checked block config"),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageCurveBlockRequest {
    runs: Vec<PH3ImageCurveBlockRowRun>,
}
impl<'de> Deserialize<'de> for PH3ImageCurveBlockRequest {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            runs: Vec<PH3ImageCurveBlockRowRun>,
        }
        let w = Wire::deserialize(d)?;
        if w.runs.is_empty() {
            return Err(serde::de::Error::custom("empty block request"));
        }
        let request = Self { runs: w.runs };
        request.rows().map_err(serde::de::Error::custom)?;
        Ok(request)
    }
}
impl PH3ImageCurveBlockRequest {
    pub fn from_layout(
        layout: PH3ImageRequestLayout<'_>,
        time_indices: &[u64],
        max_metadata_bytes: usize,
    ) -> Result<Self, VNextError> {
        if time_indices.len() != layout.rows() {
            return Err(invalid("time index count differs from request layout"));
        }
        let mut count = 0usize;
        let mut last = None;
        for (row, &time) in time_indices.iter().enumerate() {
            let current = (
                time,
                layout
                    .modality(row)
                    .ok_or_else(|| invalid("layout row missing"))? as u32,
            );
            if last != Some(current) {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| invalid("run count overflows"))?;
                last = Some(current);
            }
        }
        count
            .checked_mul(std::mem::size_of::<PH3ImageCurveBlockRowRun>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .filter(|n| *n <= max_metadata_bytes)
            .ok_or_else(|| invalid("request routing exceeds declared metadata capacity"))?;
        let mut runs: Vec<PH3ImageCurveBlockRowRun> = Vec::new();
        runs.try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        for (row, &time) in time_indices.iter().enumerate() {
            let modality = layout
                .modality(row)
                .ok_or_else(|| invalid("layout row missing"))? as u32;
            if let Some(previous) = runs
                .last_mut()
                .filter(|r| r.time_index == time && r.modality == modality)
            {
                previous.length = previous
                    .length
                    .checked_add(1)
                    .ok_or_else(|| invalid("run length overflows"))?;
            } else {
                runs.push(PH3ImageCurveBlockRowRun {
                    length: 1,
                    time_index: time,
                    modality,
                });
            }
        }
        Ok(Self { runs })
    }
    pub fn runs(&self) -> &[PH3ImageCurveBlockRowRun] {
        &self.runs
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.runs.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.length)
                .ok_or_else(|| invalid("request rows overflow"))
        })
    }
}
/// Canonical block geometry extracted from the checked original model and
/// actual curve-table header. Dense time_embed_dim is deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageCurveBlockConfig {
    pub(super) projection_storage: PH3ImageBlockProjectionStorage,
    pub(super) hidden: u64,
    pub(super) heads: u64,
    pub(super) head_dim: u64,
    pub(super) ffn: u64,
    pub(super) rotary: u64,
    pub(super) basis: u64,
    pub(super) grid: u64,
    pub(super) block_index: u64,
    pub(super) basis_rows: u64,
    pub(super) norm_eps: CanonicalRational,
    pub(super) qk_eps: CanonicalRational,
    pub(super) attention_scale: CanonicalRational,
    pub(super) requests: Vec<PH3ImageCurveBlockRequest>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    #[serde(default)]
    projection_storage: PH3ImageBlockProjectionStorage,
    hidden: u64,
    heads: u64,
    head_dim: u64,
    ffn: u64,
    rotary: u64,
    basis: u64,
    grid: u64,
    block_index: u64,
    basis_rows: u64,
    norm_eps: CanonicalRational,
    qk_eps: CanonicalRational,
    attention_scale: CanonicalRational,
    requests: Vec<PH3ImageCurveBlockRequest>,
}
impl<'de> Deserialize<'de> for PH3ImageCurveBlockConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let c = Self {
            projection_storage: w.projection_storage,
            hidden: w.hidden,
            heads: w.heads,
            head_dim: w.head_dim,
            ffn: w.ffn,
            rotary: w.rotary,
            basis: w.basis,
            grid: w.grid,
            block_index: w.block_index,
            basis_rows: w.basis_rows,
            norm_eps: w.norm_eps,
            qk_eps: w.qk_eps,
            attention_scale: w.attention_scale,
            requests: w.requests,
        };
        c.validate().map_err(serde::de::Error::custom)?;
        Ok(c)
    }
}
impl PH3ImageCurveBlockConfig {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        block_index: u64,
        basis_rows: u64,
        requests: Vec<PH3ImageCurveBlockRequest>,
    ) -> Result<Self, VNextError> {
        if block_index >= t.layers() {
            return Err(invalid("block index exceeds model layers"));
        }
        let c = Self {
            projection_storage: PH3ImageBlockProjectionStorage::default(),
            hidden: t.hidden_size(),
            heads: t.attention_heads(),
            head_dim: t.attention_head_dim(),
            ffn: t.ffn_dim(),
            rotary: 6 * t.rope_freq_dim(),
            basis: curve.basis_dim() as u64,
            grid: curve.grid_points() as u64,
            block_index,
            basis_rows,
            norm_eps: CanonicalRational::from_decimal_str(&t.norm_eps().to_string())?,
            qk_eps: CanonicalRational::from_decimal_str(&t.qk_norm_eps().to_string())?,
            attention_scale: scale(t.attention_head_dim())?,
            requests,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn projection_storage(&self) -> PH3ImageBlockProjectionStorage {
        self.projection_storage
    }
    pub fn with_projection_storage(
        mut self,
        storage: PH3ImageBlockProjectionStorage,
    ) -> Result<Self, VNextError> {
        self.projection_storage = storage;
        self.validate()?;
        Ok(self)
    }
    pub fn from_checkpoint(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        block_index: u64,
        basis_rows: u64,
        requests: Vec<PH3ImageCurveBlockRequest>,
        source: &ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource,
    ) -> anyhow::Result<Self> {
        Self::new(t, curve, block_index, basis_rows, requests)?
            .with_projection_storage(PH3ImageBlockProjectionStorage::from_checkpoint(
                source,
                block_index,
            )?)
            .map_err(Into::into)
    }
    pub fn attention_inner_dim(&self) -> u64 {
        self.heads * self.head_dim
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn ffn_dim(&self) -> u64 {
        self.ffn
    }
    pub fn rotary_dim(&self) -> u64 {
        self.rotary
    }
    pub fn basis_dim(&self) -> u64 {
        self.basis
    }
    pub fn basis_rows(&self) -> u64 {
        self.basis_rows
    }
    pub fn block_index(&self) -> u64 {
        self.block_index
    }
    pub fn requests(&self) -> &[PH3ImageCurveBlockRequest] {
        &self.requests
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.rows()?)
                .ok_or_else(|| invalid("packed rows overflow"))
        })
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let rows = self.rows()?;
        Ok(BTreeMap::from([
            (
                ProgramValueId::new(PH3ImageCurveBlockFamily::INPUT_ID)?,
                tensor(vec![rows, self.hidden], ElementType::Bf16),
            ),
            (
                ProgramValueId::new(PH3ImageCurveBlockFamily::BASIS_ID)?,
                tensor(vec![self.basis_rows, self.basis], ElementType::F32),
            ),
            (
                ProgramValueId::new(PH3ImageCurveBlockFamily::COSINE_ID)?,
                tensor(vec![rows, self.rotary], ElementType::Bf16),
            ),
            (
                ProgramValueId::new(PH3ImageCurveBlockFamily::SINE_ID)?,
                tensor(vec![rows, self.rotary], ElementType::Bf16),
            ),
        ]))
    }
    /// Transport and reshape output symbols are not determined solely by
    /// their input dimensions. Declare all intermediate shapes from this
    /// checked routing plan; providers still validate the actual attributes.
    /// This does not retain intermediate tensors as component outputs.
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate()?;
        program::compile_options(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.hidden,
            self.heads,
            self.head_dim,
            self.ffn,
            self.rotary,
            self.basis,
            self.basis_rows,
        ]
        .contains(&0)
            || self.grid < 2
            || self.grid - 1 > 1 << 24
            || self.rotary % 6 != 0
            || self.rotary > self.head_dim
            || self.head_dim % 2 != 0
            || self.head_dim > 256
            || self.requests.is_empty()
        {
            return Err(invalid(
                "invalid curve block dimensions or request documents",
            ));
        }
        let inner = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("attention width overflows"))?;
        if [self.hidden, inner, self.ffn].iter().any(|v| v % 256 != 0)
            || self.attention_scale != scale(self.head_dim)?
        {
            return Err(invalid(
                "W4 route requires K multiples of256 and the native head scale",
            ));
        }
        for request in &self.requests {
            if request.runs.is_empty() {
                return Err(invalid("empty attention request"));
            }
            for run in &request.runs {
                if run.length == 0
                    || run.length > i64::MAX as u64
                    || run.time_index >= self.basis_rows
                    || run.modality > 2
                {
                    return Err(invalid("invalid timestep/modality row mapping"));
                }
            }
        }
        for v in [self.hidden, inner, self.ffn, self.basis, self.basis_rows] {
            if v > i32::MAX as u64 {
                return Err(invalid("matrix dimension exceeds native i32 domain"));
            }
        }
        for multiplier in [3u64, 6, 18] {
            self.hidden
                .checked_mul(multiplier)
                .filter(|v| *v <= i32::MAX as u64)
                .ok_or_else(|| invalid("modulation extent overflows"))?;
        }
        inner
            .checked_mul(3)
            .filter(|v| *v <= i32::MAX as u64)
            .ok_or_else(|| invalid("QKV width exceeds i32"))?;
        self.ffn
            .checked_mul(2)
            .filter(|v| *v <= i32::MAX as u64)
            .ok_or_else(|| invalid("SwiGLU width exceeds i32"))?;
        if self.rows()? > i64::MAX as u64 {
            return Err(invalid("row metadata exceeds i64"));
        }
        let specs = self.input_tensor_specs()?;
        for (id, spec) in &specs {
            spec.validate(id.as_str())?;
            if spec.byte_len()? > isize::MAX as u64 {
                return Err(invalid("input byte extent exceeds address space"));
            }
        }
        let x = resolved(vec![self.rows()?, self.hidden], ElementType::Bf16)?;
        let gamma = resolved(vec![self.hidden], ElementType::Bf16)?;
        RowRmsNormSingleRoundBf16Geometry::new(&x, &gamma, &x, self.norm_eps)?;
        let q = resolved(vec![self.rows()?, inner], ElementType::Bf16)?;
        let qg = resolved(vec![self.head_dim], ElementType::Bf16)?;
        let c = resolved(vec![self.rows()?, self.rotary], ElementType::Bf16)?;
        FusedHeadRmsPartialRopeBf16Geometry::new(
            &q,
            &qg,
            &c,
            &c,
            &q,
            self.head_dim,
            self.rotary,
            self.qk_eps,
        )?;
        self.metadata_bytes()?;
        Ok(())
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, VNextError> {
        let runs = self.requests.iter().try_fold(0usize, |n, r| {
            n.checked_add(r.runs.len())
                .ok_or_else(|| invalid("metadata count overflows"))
        })?;
        runs.checked_mul(std::mem::size_of::<PH3ImageCurveBlockRowRun>())
            .and_then(|n| {
                n.checked_add(
                    self.requests
                        .len()
                        .checked_mul(std::mem::size_of::<PH3ImageCurveBlockRequest>())?,
                )
            })
            .ok_or_else(|| invalid("metadata bytes overflow"))
    }
    pub(super) fn estimated_plan_nodes(&self) -> Result<usize, VNextError> {
        if program::compact_parameters(self) {
            return Ok(19);
        }
        let runs = self.requests.iter().try_fold(0usize, |n, r| {
            n.checked_add(r.runs.len())
                .ok_or_else(|| invalid("run count overflows"))
        })?;
        let chunks = runs.div_ceil(ROW_GATHER_BF16_MAX_RUNS);
        chunks
            .checked_mul(12)
            .and_then(|n| n.checked_add(15))
            .ok_or_else(|| invalid("plan nodes overflow"))
    }
}
pub(super) fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(
    dimensions: Vec<u64>,
    element_type: ElementType,
) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(dimensions, element_type, ResolvedTensorLayout::Contiguous)
}
fn scale(head: u64) -> Result<CanonicalRational, VNextError> {
    if head == 0 {
        return Err(invalid("zero head width"));
    }
    CanonicalRational::from_decimal_str(&format!("{:.17}", 1.0 / (head as f64).sqrt()))
}
