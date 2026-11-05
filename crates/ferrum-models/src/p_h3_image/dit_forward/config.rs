use super::super::PH3ImageCurveBlockRowRun;
use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageDiTPhase {
    WithTextRefiner,
    RefinedText,
    RefineOnly,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Extent {
    pub height: u64,
    pub width: u64,
}
/// Original Qwen presentation tags, retained separately from main-block time
/// runs. Vision prefix rows use Image; reference rows come from latent patches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextTagRun {
    length: u64,
    image: bool,
}
impl Extent {
    fn from_image(i: PH3ImageReferenceImage) -> Result<Self, VNextError> {
        Ok(Self {
            height: u64::try_from(i.latent_height()).map_err(|_| invalid("height exceeds u64"))?,
            width: u64::try_from(i.latent_width()).map_err(|_| invalid("width exceeds u64"))?,
        })
    }
}
/// Text and image-reference prefix followed by stereo audio4 and one image.
/// Time routing is supplied explicitly, using the same checked recipes as
/// the main stack and final heads; no mutable global timestep index exists.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageDiTRequest {
    pub(super) target: Extent,
    pub(super) reference: Option<Extent>,
    pub(super) text_rows: u64,
    text_tags: Vec<TextTagRun>,
    pub(super) block: PH3ImageCurveBlockRequest,
    pub(super) final_layer: PH3ImageFinalLayerRequest,
}
impl PH3ImageDiTRequest {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        tags: &[PH3ImageModality],
        target: PH3ImageReferenceImage,
        reference: Option<PH3ImageReferenceImage>,
        time_indices: &[u64],
        metadata_capacity: usize,
    ) -> Result<Self, VNextError> {
        let refs = reference.into_iter().collect::<Vec<_>>();
        let layout = PH3ImageRequestLayout::new(t, tags, target, &refs)
            .map_err(|e| invalid(e.to_string()))?;
        let block =
            PH3ImageCurveBlockRequest::from_layout(layout, time_indices, metadata_capacity)?;
        let count = tags
            .iter()
            .enumerate()
            .filter(|(i, t)| *i == 0 || tags[*i - 1] != **t)
            .count();
        let bytes =
            count
                .checked_mul(std::mem::size_of::<TextTagRun>())
                .and_then(|n| {
                    n.checked_add(block.runs().len().checked_mul(std::mem::size_of::<
                        super::super::PH3ImageCurveBlockRowRun,
                    >())?)
                })
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .ok_or_else(|| invalid("request tag/time metadata overflow"))?;
        if bytes > metadata_capacity {
            return Err(invalid("request tag/time metadata exceeds capacity"));
        }
        let mut text_tags: Vec<TextTagRun> = Vec::new();
        text_tags
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        for tag in tags {
            let image = match tag {
                PH3ImageModality::Image => true,
                PH3ImageModality::Text => false,
                PH3ImageModality::Audio => {
                    return Err(invalid("presentation prefix cannot contain audio tags"))
                }
            };
            if let Some(last) = text_tags.last_mut().filter(|last| last.image == image) {
                last.length = last
                    .length
                    .checked_add(1)
                    .ok_or_else(|| invalid("tag run overflow"))?;
            } else {
                text_tags.push(TextTagRun { length: 1, image });
            }
        }
        Ok(Self {
            target: Extent::from_image(target)?,
            reference: reference.map(Extent::from_image).transpose()?,
            text_rows: u64::try_from(tags.len()).map_err(|_| invalid("text rows exceed u64"))?,
            text_tags,
            block,
            final_layer: PH3ImageFinalLayerRequest::from_layout(layout, time_indices)?,
        })
    }
    pub(super) fn image_rows(&self, c: &PH3ImageDiTConfig) -> Result<u64, VNextError> {
        self.target_rows(c)?
            .checked_add(self.reference_rows(c)?)
            .ok_or_else(|| invalid("image rows overflow"))
    }
    pub(super) fn target_rows(&self, c: &PH3ImageDiTConfig) -> Result<u64, VNextError> {
        patch_rows(self.target, c)
    }
    pub(super) fn reference_rows(&self, c: &PH3ImageDiTConfig) -> Result<u64, VNextError> {
        self.reference
            .map(|r| patch_rows(r, c))
            .transpose()
            .map(|n| n.unwrap_or(0))
    }
    fn validate_crosswalk(&self, reference_rows: u64) -> Result<(), VNextError> {
        if self.text_tags.is_empty()
            || self.text_tags.iter().any(|r| r.length == 0)
            || self.text_tags.windows(2).any(|p| p[0].image == p[1].image)
        {
            return Err(invalid("presentation tags must be positive canonical RLE"));
        }
        let tag_rows = self.text_tags.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.length)
                .ok_or_else(|| invalid("tag rows overflow"))
        })?;
        if tag_rows != self.text_rows {
            return Err(invalid("presentation tags do not cover prefix text rows"));
        }
        let runs = self.block.runs();
        let mut index = 0usize;
        let mut consumed = 0u64;
        let mut consume = |mut length: u64,
                           modality: PH3ImageModality,
                           time: Option<u64>|
         -> Result<(), VNextError> {
            while length > 0 {
                let run = runs
                    .get(index)
                    .ok_or_else(|| invalid("main block ends before final/layout routing"))?;
                if run.modality() != modality || time.is_some_and(|t| t != run.time_index()) {
                    return Err(invalid("main block modality/time differs from presentation, reference or final-head routing"));
                }
                let remaining = run
                    .length()
                    .checked_sub(consumed)
                    .ok_or_else(|| invalid("routing cursor exceeds run"))?;
                let take = remaining.min(length);
                if take == 0 {
                    return Err(invalid("routing cursor made no progress"));
                }
                consumed += take;
                length -= take;
                if consumed == run.length() {
                    index += 1;
                    consumed = 0;
                }
            }
            Ok(())
        };
        for tag in &self.text_tags {
            consume(
                tag.length,
                if tag.image {
                    PH3ImageModality::Image
                } else {
                    PH3ImageModality::Text
                },
                None,
            )?;
        }
        consume(reference_rows, PH3ImageModality::Image, None)?;
        for run in self.final_layer.audio_runs() {
            consume(
                run.length(),
                PH3ImageModality::Audio,
                Some(run.time_index()),
            )?;
        }
        for run in self.final_layer.image_runs() {
            consume(
                run.length(),
                PH3ImageModality::Image,
                Some(run.time_index()),
            )?;
        }
        if index != runs.len() || consumed != 0 {
            return Err(invalid(
                "main block has rows outside declared final/layout routing",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageDiTConfig {
    pub(super) phase: PH3ImageDiTPhase,
    #[serde(default)]
    pub(super) attention_policy: PH3ImageDenoiserAttentionPolicy,
    pub(super) channels: u64,
    pub(super) audio_channels: u64,
    pub(super) patch_h: u64,
    pub(super) patch_w: u64,
    pub(super) hidden: u64,
    pub(super) text_dim: u64,
    pub(super) requests: Vec<PH3ImageDiTRequest>,
    pub(super) refiner: PH3ImageTokenRefinerConfig,
    pub(super) stack: PH3ImageCurveStackConfig,
    pub(super) final_layer: PH3ImageFinalLayerConfig,
}
impl PH3ImageDiTConfig {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        basis_rows: u64,
        requests: Vec<PH3ImageDiTRequest>,
        phase: PH3ImageDiTPhase,
        storage: PH3ImageStackProjectionStorage,
    ) -> Result<Self, VNextError> {
        let refiner =
            PH3ImageTokenRefinerConfig::new(t, requests.iter().map(|r| r.text_rows).collect())?;
        let stack = PH3ImageCurveStackConfig::new(
            t,
            curve,
            basis_rows,
            requests.iter().map(|r| r.block.clone()).collect(),
        )?
        .with_projection_storage(storage)?;
        let final_layer = PH3ImageFinalLayerConfig::new(
            t,
            curve,
            basis_rows,
            requests.iter().map(|r| r.final_layer.clone()).collect(),
        )?;
        let [pt, patch_h, patch_w] = t.patch_size();
        if pt != 1 {
            return Err(invalid("one latent frame required"));
        }
        let c = Self {
            phase,
            attention_policy: PH3ImageDenoiserAttentionPolicy::Sequential,
            channels: t.in_channels(),
            audio_channels: t.audio_in_channels(),
            patch_h,
            patch_w,
            hidden: t.hidden_size(),
            text_dim: t.text_dim(),
            requests,
            refiner,
            stack,
            final_layer,
        };
        c.validate()?;
        Ok(c)
    }
    pub(in crate::p_h3_image) fn from_checkpoint(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        basis_rows: u64,
        requests: Vec<PH3ImageDiTRequest>,
        phase: PH3ImageDiTPhase,
        source: &ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource,
        limits: PH3ImageDiTLimits,
    ) -> anyhow::Result<Self> {
        let mut c = Self::new(
            t,
            curve,
            basis_rows,
            requests,
            phase,
            PH3ImageStackProjectionStorage::default(),
        )?;
        c.stack = PH3ImageCurveStackConfig::from_checkpoint(
            t,
            curve,
            basis_rows,
            c.requests.iter().map(|r| r.block.clone()).collect(),
            source,
            limits.stack(),
        )?;
        c.validate()?;
        Ok(c)
    }
    pub fn with_phase(mut self, phase: PH3ImageDiTPhase) -> Self {
        self.phase = phase;
        self
    }
    pub fn phase(&self) -> PH3ImageDiTPhase {
        self.phase
    }
    pub(in crate::p_h3_image) fn with_attention_policy(
        mut self,
        policy: PH3ImageDenoiserAttentionPolicy,
    ) -> Result<Self, VNextError> {
        self.attention_policy = policy;
        self.validate()?;
        Ok(self)
    }
    /// Borrow the checked per-request modality/time runs in packed membership
    /// order. These are the graph's original routing records.
    pub fn request_row_runs(
        &self,
    ) -> impl ExactSizeIterator<Item = &[PH3ImageCurveBlockRowRun]> + '_ {
        self.requests.iter().map(|r| r.block.runs())
    }
    pub(super) fn text_rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.text_rows)
                .ok_or_else(|| invalid("text rows overflow"))
        })
    }
    pub(super) fn image_rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.image_rows(self)?)
                .ok_or_else(|| invalid("image rows overflow"))
        })
    }
    pub(super) fn audio_rows(&self) -> Result<u64, VNextError> {
        u64::try_from(self.requests.len())
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("audio rows overflow"))
    }
    pub(super) fn image_width(&self) -> Result<u64, VNextError> {
        self.channels
            .checked_mul(self.patch_h)
            .and_then(|n| n.checked_mul(self.patch_w))
            .ok_or_else(|| invalid("patch width overflow"))
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        if self.phase == PH3ImageDiTPhase::RefineOnly {
            return Ok(BTreeMap::from([(
                ProgramValueId::new(PH3ImageDiTFamily::RAW_TEXT_ID)?,
                tensor(vec![self.text_rows()?, self.text_dim], ElementType::Bf16),
            )]));
        }
        let mut specs = self.stack.input_tensor_specs()?;
        specs.remove(&ProgramValueId::new(PH3ImageCurveStackFamily::INPUT_ID)?);
        specs.extend([
            (
                ProgramValueId::new(PH3ImageDiTFamily::IMAGE_PATCHES_ID)?,
                tensor(
                    vec![self.image_rows()?, self.image_width()?],
                    ElementType::F32,
                ),
            ),
            (
                ProgramValueId::new(PH3ImageDiTFamily::AUDIO_PATCHES_ID)?,
                tensor(
                    vec![self.audio_rows()?, self.audio_channels],
                    ElementType::F32,
                ),
            ),
            (
                ProgramValueId::new(match self.phase {
                    PH3ImageDiTPhase::WithTextRefiner | PH3ImageDiTPhase::RefineOnly => {
                        PH3ImageDiTFamily::RAW_TEXT_ID
                    }
                    PH3ImageDiTPhase::RefinedText => PH3ImageDiTFamily::REFINED_TEXT_ID,
                })?,
                tensor(
                    vec![
                        self.text_rows()?,
                        if self.phase == PH3ImageDiTPhase::WithTextRefiner {
                            self.text_dim
                        } else {
                            self.hidden
                        },
                    ],
                    ElementType::Bf16,
                ),
            ),
        ]);
        Ok(specs)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.attention_policy == PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1
            && !matches!(self.stack.block_template().head_dim(), 72 | 128)
        {
            return Err(invalid(
                "cuDNN segmented SDPA requires head dimension 72 or 128",
            ));
        }
        if [
            self.channels,
            self.audio_channels,
            self.patch_h,
            self.patch_w,
            self.hidden,
            self.text_dim,
        ]
        .contains(&0)
            || self.requests.is_empty()
        {
            return Err(invalid("nonempty geometry/requests required"));
        }
        if self.stack.block_template().hidden_size() != self.hidden
            || self.final_layer.hidden_size() != self.hidden
            || self.final_layer.image_velocity_width() != self.image_width()?
            || self.final_layer.audio_velocity_width() != self.audio_channels
            || self.refiner.rows()? != self.text_rows()?
        {
            return Err(invalid(
                "subcomponents disagree about hidden, text or patch geometry",
            ));
        }
        if self.stack.block_template().requests()
            != self
                .requests
                .iter()
                .map(|r| r.block.clone())
                .collect::<Vec<_>>()
            || self.final_layer.requests()
                != self
                    .requests
                    .iter()
                    .map(|r| r.final_layer.clone())
                    .collect::<Vec<_>>()
        {
            return Err(invalid("subcomponent request routing differs"));
        }
        if self
            .refiner
            .request_lengths()
            .iter()
            .copied()
            .ne(self.requests.iter().map(|r| r.text_rows))
        {
            return Err(invalid(
                "text-refiner request segments differ from packed documents",
            ));
        }
        if self.stack.block_template().basis_rows() != self.final_layer.basis_rows() {
            return Err(invalid("shared curve basis row count differs"));
        }
        let refiner_inputs = self.refiner.input_tensor_specs()?;
        if refiner_inputs[&ProgramValueId::new(PH3ImageTokenRefinerFamily::INPUT_ID)?].dimensions
            != [self.text_rows()?, self.text_dim]
        {
            return Err(invalid("refiner input width differs"));
        }
        for r in &self.requests {
            let target = r.target_rows(self)?;
            let reference = r.reference_rows(self)?;
            let total = r
                .text_rows
                .checked_add(reference)
                .and_then(|n| n.checked_add(4))
                .and_then(|n| n.checked_add(target))
                .ok_or_else(|| invalid("document rows overflow"))?;
            if r.text_rows == 0
                || r.block.rows()? != total
                || r.final_layer.prefix_rows() != r.text_rows + reference
                || r.final_layer.audio_rows()? != 4
                || r.final_layer.image_rows()? != target
            {
                return Err(invalid("request extent and routing disagree"));
            }
            r.validate_crosswalk(reference)?;
        }
        for (id, spec) in self.input_tensor_specs()? {
            spec.validate(id.as_str())?;
            if spec.byte_len()? > isize::MAX as u64 {
                return Err(invalid("input exceeds Host address space"));
            }
        }
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
fn patch_rows(extent: Extent, c: &PH3ImageDiTConfig) -> Result<u64, VNextError> {
    if extent.height == 0
        || extent.width == 0
        || extent.height % c.patch_h != 0
        || extent.width % c.patch_w != 0
    {
        return Err(invalid(
            "latent extent must contain complete nonempty patches",
        ));
    }
    (extent.height / c.patch_h)
        .checked_mul(extent.width / c.patch_w)
        .ok_or_else(|| invalid("patch rows overflow"))
}
