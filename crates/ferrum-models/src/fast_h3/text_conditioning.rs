//! FastH3's raw ordered media presentation. There is no chat template, BOS,
//! padding, truncation or substitute visual token. Vision execution remains an
//! ordinary separate Core component; this adapter consumes its completed bytes.
use super::{FastH3TextTagRun, FastH3TextTokenTag};
use crate::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextDeepStack,
    Qwen3VLTextEncoderFamily as Text, Qwen3VLTextImageSpan, Qwen3VLTextKeyRun, Qwen3VLTextLayout,
};
use crate::qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionFamily as Vision};
use anyhow::{ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::bf16;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug)]
pub struct FastH3PresentationLimits {
    pub maximum_prompt_bytes: usize,
    pub maximum_tokens: u32,
}

/// Physical Host payloads for the original text inputs and completed vision
/// outputs. This is a sizing calculation, not a reservation or a peak RSS claim.
#[derive(Clone, Copy, Debug)]
pub struct FastH3TextHostFootprint {
    pub input_bytes: usize,
    pub vision_output_bytes: usize,
    pub text_output_bytes: usize,
}

/// Borrowed completed vision output or explicitly admitted file input. This
/// view neither reclassifies its owner nor asserts a device completion fence.
#[derive(Clone, Copy)]
pub struct FastH3VisionFeatureView<'a> {
    id: &'a ProgramValueId,
    tensor: &'a ResolvedTensorSpec,
    bytes: &'a ManagedHostBytes,
}
impl<'a> FastH3VisionFeatureView<'a> {
    pub fn new(
        id: &'a ProgramValueId,
        tensor: &'a ResolvedTensorSpec,
        bytes: &'a ManagedHostBytes,
    ) -> Self {
        Self { id, tensor, bytes }
    }
}

/// Typed completed-reference metadata in the original request order. Audio
/// contributes a text label; its latent sequence belongs to the DiT input, not
/// to a fabricated Qwen visual/audio embedding.
#[derive(Clone, Copy)]
pub enum FastH3PresentationReference<'a> {
    Image(&'a Qwen3VLVisionConfig),
    Video {
        vision: &'a Qwen3VLVisionConfig,
        block_timestamps: &'a [f64],
        has_audio: bool,
    },
    Audio,
}

#[derive(Clone, Debug)]
struct VisualSource {
    first_span: usize,
    span_count: usize,
    rows: u64,
}

#[derive(Clone, Debug)]
pub struct FastH3MediaPresentation {
    tokens: Vec<u32>,
    tags: Vec<FastH3TextTagRun>,
    spans: Vec<Qwen3VLTextImageSpan>,
    sources: Vec<VisualSource>,
    visual_width: u64,
}
mod presentation;
pub type FastH3ImagePresentation = FastH3MediaPresentation;

impl FastH3MediaPresentation {
    pub const PROFILE: &'static str =
        "fasth3.ref2va.ordered_media.raw_label_vision_prompt.bf16_embedding_and_deepstack.v1";
    pub fn single_image(
        tokenizer: &Tokenizer,
        prompt: &str,
        vision: &Qwen3VLVisionConfig,
        limits: FastH3PresentationLimits,
    ) -> Result<Self> {
        Self::ordered_images(tokenizer, prompt, &[vision], limits)
    }

    /// Present labels and complete visual slots in reference order, then the
    /// untemplated prompt. Each image keeps its own spatial MRoPE coordinates.
    pub fn ordered_images(
        tokenizer: &Tokenizer,
        prompt: &str,
        images: &[&Qwen3VLVisionConfig],
        limits: FastH3PresentationLimits,
    ) -> Result<Self> {
        let references = images
            .iter()
            .map(|image| FastH3PresentationReference::Image(image))
            .collect::<Vec<_>>();
        Self::ordered_references(tokenizer, prompt, &references, limits)
    }

    pub fn ordered_references(
        tokenizer: &Tokenizer,
        prompt: &str,
        references: &[FastH3PresentationReference<'_>],
        limits: FastH3PresentationLimits,
    ) -> Result<Self> {
        presentation::build(tokenizer, prompt, references, limits)
    }
    pub fn token_ids(&self) -> &[u32] {
        &self.tokens
    }
    pub fn text_tags(&self) -> &[FastH3TextTagRun] {
        &self.tags
    }
    pub fn image_spans(&self) -> &[Qwen3VLTextImageSpan] {
        &self.spans
    }
    pub fn deepstack(&self) -> Result<Qwen3VLTextDeepStack> {
        Ok(Qwen3VLTextDeepStack::new(
            self.spans
                .iter()
                .map(|span| span.visual_run().clone())
                .collect(),
        )?)
    }
    pub fn layout(&self) -> Result<Qwen3VLTextLayout> {
        let rows = u32::try_from(self.tokens.len())?;
        Ok(Qwen3VLTextLayout::new(
            rows,
            vec![vec![Qwen3VLTextKeyRun::new(rows, true)]],
        )?)
    }

    fn input_payload_sizes(&self, architecture: &Qwen3VLTextArchitecture) -> Result<[usize; 6]> {
        ensure!(
            architecture.hidden_size() == self.visual_width,
            "text hidden width differs from original visual features"
        );
        let width = usize::try_from(self.visual_width)?;
        let rows = self.spans.iter().try_fold(0usize, |sum, span| {
            sum.checked_add(usize::try_from(span.visual_run().length())?)
                .context("visual rows overflow")
        })?;
        let bytes = |rows: usize, columns: usize| {
            rows.checked_mul(columns)
                .and_then(|n| n.checked_mul(2))
                .context("BF16 text payload overflow")
        };
        let hidden = bytes(self.tokens.len(), width)?;
        let updates = bytes(rows, width)?;
        let rotary = bytes(self.tokens.len(), usize::try_from(architecture.head_dim())?)?;
        Ok([hidden, updates, updates, updates, rotary, rotary])
    }

    pub fn host_footprint(
        &self,
        architecture: &Qwen3VLTextArchitecture,
        host: &HostMemoryAccount,
    ) -> Result<FastH3TextHostFootprint> {
        let sizes = self.input_payload_sizes(architecture)?;
        let input_bytes = sizes.into_iter().try_fold(0usize, |sum, bytes| {
            sum.checked_add(host.backing_bytes(bytes)?)
                .context("text Host backing overflow")
        })?;
        let vision_output_bytes = self.sources.iter().try_fold(0usize, |sum, source| {
            let bytes = source
                .rows
                .checked_mul(self.visual_width)
                .and_then(|n| n.checked_mul(2))
                .context("vision output bytes overflow")?;
            let four = host
                .backing_bytes(usize::try_from(bytes)?)?
                .checked_mul(4)
                .context("vision output backing overflow")?;
            sum.checked_add(four)
                .context("ordered vision backing overflow")
        })?;
        Ok(FastH3TextHostFootprint {
            input_bytes,
            vision_output_bytes,
            text_output_bytes: host.backing_bytes(sizes[0])?,
        })
    }
    /// Each ordered visual reference supplies native main merged + three DeepStack IDs.
    /// Video time blocks borrow disjoint rows of the same completed output;
    /// they do not manufacture completion owners or duplicate Host payloads.
    /// The caller establishes execution provenance. Exact shape, dtype and Host
    /// ownership are checked before admitted InputPreprocess copies are made.
    pub fn managed_text_inputs(
        &self,
        text: &Qwen3VLTextConfig,
        archive: &SafetensorsArchive,
        vision_outputs: &[[FastH3VisionFeatureView<'_>; 4]],
        host: &HostMemoryAccount,
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            text.layout() == &self.layout()?
                && text.deepstack() == Some(&self.deepstack()?)
                && text.architecture().hidden_size() == self.visual_width
                && vision_outputs.len() == self.sources.len(),
            "text layout/DeepStack/hidden width differs from presentation"
        );
        let width = usize::try_from(self.visual_width)?;
        let rows = self.spans.iter().try_fold(0usize, |rows, span| {
            rows.checked_add(usize::try_from(span.visual_run().length())?)
                .context("total visual rows overflow")
        })?;
        let names = [
            Vision::OUTPUT_ID,
            Vision::DEEPSTACK_IDS[0],
            Vision::DEEPSTACK_IDS[1],
            Vision::DEEPSTACK_IDS[2],
        ];
        let mut sources = Vec::with_capacity(vision_outputs.len());
        for (source, outputs) in self.sources.iter().zip(vision_outputs) {
            let mut image = Vec::with_capacity(4);
            for name in names {
                let matches = outputs
                    .iter()
                    .filter(|o| o.id.as_str() == name)
                    .collect::<Vec<_>>();
                ensure!(matches.len() == 1, "requires exactly one {name}");
                let output = matches[0];
                ensure!(
                    output.tensor.element_type() == ElementType::Bf16
                        && output.tensor.dimensions() == [source.rows, self.visual_width]
                        && output.tensor.layout() == &ResolvedTensorLayout::Contiguous
                        && host.owns(output.bytes)
                        && matches!(
                            output.bytes.class(),
                            HostMemoryClass::InputPreprocess | HostMemoryClass::Output
                        )
                        && output.bytes.len() as u64 == output.tensor.minimum_storage_bytes()?
                        && finite(output.bytes.bytes()),
                    "vision output requires finite original BF16 rows in the same Host account"
                );
                image.push(output);
            }
            sources.push(image);
        }
        let embedding = archive.tensor(Text::EMBEDDING_EXTERNAL_NAME)?;
        ensure!(
            embedding.element_type() == Some(ElementType::Bf16)
                && embedding.shape() == [text.architecture().vocabulary_size(), self.visual_width]
                && self
                    .tokens
                    .iter()
                    .all(|&id| u64::from(id) < text.architecture().vocabulary_size()),
            "original BF16 embedding/vocabulary differs"
        );
        let [bytes, updates, _, _, _, _] = self.input_payload_sizes(text.architecture())?;
        let mut grants = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, bytes),
                (HostMemoryClass::InputPreprocess, updates),
                (HostMemoryClass::InputPreprocess, updates),
                (HostMemoryClass::InputPreprocess, updates),
            ])?
            .into_iter();
        let mut hidden = grants.next().unwrap().allocate_zeroed()?;
        for (row, &token) in self.tokens.iter().enumerate() {
            let selected =
                &embedding.bytes()[token as usize * width * 2..(token as usize + 1) * width * 2];
            ensure!(
                finite(selected),
                "selected original token embedding is nonfinite"
            );
            hidden.as_mut()[row * width * 2..(row + 1) * width * 2].copy_from_slice(selected);
        }
        for (group, image) in self.sources.iter().zip(&sources) {
            let mut source_offset = 0;
            for span in &self.spans[group.first_span..group.first_span + group.span_count] {
                let offset = usize::try_from(span.visual_run().start())? * width * 2;
                let length = usize::try_from(span.visual_run().length())? * width * 2;
                let source = &image[0].bytes.bytes()[source_offset..source_offset + length];
                hidden.as_mut()[offset..offset + length].copy_from_slice(source);
                source_offset += length;
            }
        }
        let mut result = vec![ManagedHostTensor::new(
            ProgramValueId::new(Text::EMBEDDINGS_ID)?,
            ResolvedTensorSpec::new(
                vec![self.tokens.len() as u64, self.visual_width],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
            hidden.freeze(),
        )?];
        for (layer, (name, grant)) in Text::DEEPSTACK_IDS.iter().zip(grants).enumerate() {
            let mut bytes = grant.allocate_zeroed()?;
            let mut offset = 0;
            for image in &sources {
                let source = image[layer + 1].bytes.bytes();
                bytes.as_mut()[offset..offset + source.len()].copy_from_slice(source);
                offset += source.len();
            }
            result.push(ManagedHostTensor::new(
                ProgramValueId::new(*name)?,
                ResolvedTensorSpec::new(
                    vec![rows as u64, self.visual_width],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )?,
                bytes.freeze(),
            )?);
        }
        result.extend(text.managed_edit_mrope(host, &self.spans)?);
        Ok(result)
    }
}
fn finite(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite())
}

#[cfg(test)]
mod tests;
