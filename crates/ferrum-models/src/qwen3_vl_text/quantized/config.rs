use super::*;
use ferrum_quantization::comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizationMetadata};
use serde::Deserializer;

/// One independently scaled projection. The checkpoint metadata, rather than
/// a layer number or exclusion list, selects this execution route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3VLTextComfyProjectionQuantization {
    W4A8,
    W8A8,
}
impl Qwen3VLTextComfyProjectionQuantization {
    pub fn from_metadata(metadata: &ComfyQuantizationMetadata) -> Result<Self, VNextError> {
        match metadata {
            ComfyQuantizationMetadata::CodebookW4A8 {
                group_size: 16,
                convrot: true,
                convrot_groupsize: 256,
            } => Ok(Self::W4A8),
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            } => Ok(Self::W8A8),
            _ => Err(invalid(
                "quantization",
                "requires canonical W4 groups of 16 or INT8 with regular ConvRot groups of 256",
            )),
        }
    }
    pub fn metadata(self) -> ComfyQuantizationMetadata {
        match self {
            Self::W4A8 => ComfyQuantizationMetadata::CodebookW4A8 {
                group_size: 16,
                convrot: true,
                convrot_groupsize: 256,
            },
            Self::W8A8 => ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            },
        }
    }
    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::W4A8 => COMFY_CONVROT_W4A8_F32_OPERATION_ID,
            Self::W8A8 => COMFY_CONVROT_W8A8_F32_OPERATION_ID,
        }
    }
    pub(super) fn components(self) -> &'static [&'static str] {
        match self {
            Self::W4A8 => &[
                "weight",
                "weight_s_rel",
                "weight_s_channel",
                "weight_codebook",
            ],
            Self::W8A8 => &["weight", "weight_scale"],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Qwen3VLTextComfyConfig {
    text: Qwen3VLTextConfig,
    #[serde(
        default,
        skip_serializing_if = "Qwen3VLTextComfyInputMode::is_embeddings"
    )]
    input_mode: Qwen3VLTextComfyInputMode,
    /// Exact canonical external module names, without the `.weight` suffix.
    projections: BTreeMap<String, Qwen3VLTextComfyProjectionQuantization>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    text: Qwen3VLTextConfig,
    #[serde(default)]
    input_mode: Qwen3VLTextComfyInputMode,
    projections: BTreeMap<String, Qwen3VLTextComfyProjectionQuantization>,
}
impl<'de> Deserialize<'de> for Qwen3VLTextComfyConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(d)?;
        let c = Self {
            text: wire.text,
            input_mode: wire.input_mode,
            projections: wire.projections,
        };
        c.validate().map_err(serde::de::Error::custom)?;
        Ok(c)
    }
}
impl Qwen3VLTextComfyConfig {
    pub fn new(
        text: Qwen3VLTextConfig,
        projections: BTreeMap<String, Qwen3VLTextComfyProjectionQuantization>,
    ) -> Result<Self, VNextError> {
        let c = Self {
            text,
            input_mode: Qwen3VLTextComfyInputMode::Embeddings,
            projections,
        };
        c.validate()?;
        Ok(c)
    }
    /// Select every projection from its validated, original `.comfy_quant`.
    /// The checkpoint view subsequently checks physical components and norms.
    pub fn from_checkpoint(
        text: Qwen3VLTextConfig,
        source: &ComfyCanonicalWeightSource,
    ) -> Result<Self, VNextError> {
        let mut projections = BTreeMap::new();
        for (module, _, _) in weights::projection_shapes(&text)? {
            let metadata = source.modules().get(&module).ok_or_else(|| {
                invalid(
                    "checkpoint",
                    format!("missing quantization metadata for {module}"),
                )
            })?;
            projections.insert(
                module,
                Qwen3VLTextComfyProjectionQuantization::from_metadata(metadata)?,
            );
        }
        Self::new(text, projections)
    }
    pub fn text(&self) -> &Qwen3VLTextConfig {
        &self.text
    }
    /// Explicitly consume the four F32 visual outputs. Runs select only the
    /// expanded image feature tokens; image boundary embeddings stay intact.
    pub fn with_visual_inputs(
        mut self,
        runs: super::super::Qwen3VLTextDeepStack,
    ) -> Result<Self, VNextError> {
        self.text = self.text.with_deepstack(runs)?;
        self.input_mode = Qwen3VLTextComfyInputMode::VisualEmbeddings;
        self.validate()?;
        Ok(self)
    }
    pub fn input_mode(&self) -> Qwen3VLTextComfyInputMode {
        self.input_mode
    }
    pub fn visual_inputs(&self) -> Option<&super::super::Qwen3VLTextDeepStack> {
        match self.input_mode {
            Qwen3VLTextComfyInputMode::Embeddings => None,
            Qwen3VLTextComfyInputMode::VisualEmbeddings => self.text.deepstack(),
        }
    }
    pub fn projections(&self) -> &BTreeMap<String, Qwen3VLTextComfyProjectionQuantization> {
        &self.projections
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        self.validate()?;
        let mut specs: BTreeMap<_, _> = self
            .text
            .input_tensor_specs()?
            .into_iter()
            .map(|(id, mut spec)| {
                spec.element_type = ElementType::F32;
                (id, spec)
            })
            .collect();
        if let Some(runs) = self.visual_inputs() {
            specs.insert(
                ProgramValueId::new(Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID)?,
                ProgramTensorSpec {
                    element_type: ElementType::F32,
                    dimensions: vec![runs.rows()?, self.text.architecture().hidden_size()],
                    layout: ResolvedTensorLayout::Contiguous,
                },
            );
        }
        Ok(specs)
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.input_tensor_specs()?)
    }
    pub(super) fn estimated_plan_nodes(&self) -> Result<usize, VNextError> {
        // Seven projections, four norms, two rotations, attention, activation,
        // and two residual additions. An optional final-norm tap adds one.
        usize::try_from(self.text.architecture().layers())
            .ok()
            .and_then(|n| n.checked_mul(17))
            .and_then(|n| {
                n.checked_add(usize::from(
                    self.text.hidden_state_indices().last()
                        == Some(&self.text.architecture().layers()),
                ))
            })
            .and_then(|n| n.checked_add(if self.visual_inputs().is_some() { 4 } else { 0 }))
            .ok_or_else(|| invalid("config", "node inventory overflows"))
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.text.validate()?;
        visual::validate(&self.text, self.input_mode)?;
        let expected = weights::projection_shapes(&self.text)?;
        if expected.len() != self.projections.len() {
            return Err(invalid(
                "projections",
                "requires exactly seven projections per decoder layer",
            ));
        }
        for (module, input, _) in expected {
            if !self.projections.contains_key(&module) {
                return Err(invalid(
                    "projections",
                    format!("missing canonical module {module}"),
                ));
            }
            if input % 256 != 0 {
                return Err(invalid(
                    "projections",
                    "projection input widths must contain complete regular ConvRot groups of 256",
                ));
            }
        }
        self.estimated_plan_nodes()?;
        Ok(())
    }
}
