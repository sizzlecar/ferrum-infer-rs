use super::*;
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageTokenRefinerConfig {
    pub(super) hidden: u64,
    pub(super) text_dim: u64,
    pub(super) heads: u64,
    pub(super) head_dim: u64,
    pub(super) ffn: u64,
    pub(super) layers: u64,
    pub(super) norm_eps: CanonicalRational,
    pub(super) qk_eps: CanonicalRational,
    pub(super) final_eps: CanonicalRational,
    pub(super) attention_scale: CanonicalRational,
    pub(super) requests: Vec<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    hidden: u64,
    text_dim: u64,
    heads: u64,
    head_dim: u64,
    ffn: u64,
    layers: u64,
    norm_eps: CanonicalRational,
    qk_eps: CanonicalRational,
    final_eps: CanonicalRational,
    attention_scale: CanonicalRational,
    requests: Vec<u64>,
}
impl<'de> Deserialize<'de> for PH3ImageTokenRefinerConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let c = Self {
            hidden: w.hidden,
            text_dim: w.text_dim,
            heads: w.heads,
            head_dim: w.head_dim,
            ffn: w.ffn,
            layers: w.layers,
            norm_eps: w.norm_eps,
            qk_eps: w.qk_eps,
            final_eps: w.final_eps,
            attention_scale: w.attention_scale,
            requests: w.requests,
        };
        c.validate().map_err(serde::de::Error::custom)?;
        Ok(c)
    }
}
impl PH3ImageTokenRefinerConfig {
    pub fn new(t: &PH3ImageTransformerConfig, requests: Vec<u64>) -> Result<Self, VNextError> {
        let c = Self {
            hidden: t.hidden_size(),
            text_dim: t.text_dim(),
            heads: t.attention_heads(),
            head_dim: t.attention_head_dim(),
            ffn: t.ffn_dim(),
            layers: t.refiner_layers(),
            norm_eps: CanonicalRational::from_decimal_str(&t.norm_eps().to_string())?,
            qk_eps: CanonicalRational::from_decimal_str(&t.qk_norm_eps().to_string())?,
            final_eps: CanonicalRational::from_decimal_str(&t.final_norm_eps().to_string())?,
            attention_scale: scale(t.attention_head_dim())?,
            requests,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |sum, &rows| {
            sum.checked_add(rows)
                .ok_or_else(|| invalid("rows overflow"))
        })
    }
    pub(crate) fn request_lengths(&self) -> &[u64] {
        &self.requests
    }
    /// Keep one independent document's semantics without retaining unrelated
    /// documents from an earlier continuous batch.
    pub(crate) fn for_request_rows(&self, rows: u64) -> Result<Self, VNextError> {
        let c = Self {
            hidden: self.hidden,
            text_dim: self.text_dim,
            heads: self.heads,
            head_dim: self.head_dim,
            ffn: self.ffn,
            layers: self.layers,
            norm_eps: self.norm_eps,
            qk_eps: self.qk_eps,
            final_eps: self.final_eps,
            attention_scale: self.attention_scale,
            requests: vec![rows],
        };
        c.validate()?;
        Ok(c)
    }
    pub fn inner(&self) -> u64 {
        self.heads * self.head_dim
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([(
            ProgramValueId::new(PH3ImageTokenRefinerFamily::INPUT_ID)?,
            tensor(vec![self.rows()?, self.text_dim], ElementType::Bf16),
        )]))
    }
    /// Every shape-changing output is a production contract, independently
    /// revalidated by the generic providers rather than supplied by a test.
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate()?;
        program::compile_options(self)
    }
    pub(super) fn plan_nodes(&self) -> Result<usize, VNextError> {
        usize::try_from(self.layers)
            .ok()
            .and_then(|v| v.checked_mul(18))
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| invalid("node count overflows"))
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, VNextError> {
        self.requests
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or_else(|| invalid("metadata bytes overflow"))
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.hidden,
            self.text_dim,
            self.heads,
            self.head_dim,
            self.ffn,
            self.layers,
        ]
        .contains(&0)
            || self.requests.is_empty()
            || self.requests.contains(&0)
            || self.head_dim > 256
        {
            return Err(invalid(
                "nonempty dimensions and request documents required",
            ));
        }
        let inner = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("inner width overflows"))?;
        if [self.hidden, inner, self.ffn].iter().any(|v| v % 256 != 0)
            || self.attention_scale != scale(self.head_dim)?
        {
            return Err(invalid(
                "W8 H256 route requires complete rotation groups and native head scale",
            ));
        }
        for extent in [
            self.hidden,
            self.text_dim,
            inner,
            self.ffn,
            self.rows()?,
            inner
                .checked_mul(3)
                .ok_or_else(|| invalid("qkv width overflows"))?,
            self.ffn
                .checked_mul(2)
                .ok_or_else(|| invalid("fc1 width overflows"))?,
            self.rows()?
                .checked_mul(self.heads)
                .ok_or_else(|| invalid("head row count overflows"))?,
        ] {
            if extent > i32::MAX as u64 {
                return Err(invalid("extent exceeds native i32 domain"));
            }
        }
        for (&eps, width) in [
            (&self.norm_eps, self.hidden),
            (&self.qk_eps, self.head_dim),
            (&self.final_eps, self.hidden),
        ] {
            let x = ResolvedTensorSpec::new(
                vec![1, width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let gamma = ResolvedTensorSpec::new(
                vec![width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            RowRmsNormSingleRoundBf16Geometry::new(&x, &gamma, &x, eps)?;
        }
        for (name, spec) in self.input_tensor_specs()? {
            spec.validate(name.as_str())?;
            if spec.byte_len()? > isize::MAX as u64 {
                return Err(invalid("tensor exceeds Host address space"));
            }
        }
        self.plan_nodes()?;
        self.metadata_bytes()?;
        Ok(())
    }
}
pub(super) fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn scale(d: u64) -> Result<CanonicalRational, VNextError> {
    if d == 0 {
        return Err(invalid("zero head width"));
    }
    CanonicalRational::from_decimal_str(&format!("{:.17}", 1.0 / (d as f64).sqrt()))
}
