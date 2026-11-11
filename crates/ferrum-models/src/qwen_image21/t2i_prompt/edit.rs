//! Ordered reference images per request. DeepStack retains sample-major,
//! reference-major rows and remains three distinct graph inputs.
use super::*;
use crate::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Text,
    Qwen3VLTextImageSpan,
};
use crate::qwen3_vl_vision::Qwen3VLVisionFamily as Vision;
use ferrum_execution::TensorExecutionOutput;
use ferrum_quantization::ArchiveTensor;
use ferrum_runtime::*;

/// One processed reference's pre-merge vision grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImage21EditImageGrid {
    height: u64,
    width: u64,
    merge: u64,
    slots: u32,
}
impl QwenImage21EditImageGrid {
    pub fn new(grid_height: u64, grid_width: u64, merge: u64) -> Result<Self> {
        let span = Qwen3VLTextImageSpan::new(0, grid_height, grid_width, merge)?;
        Ok(Self {
            height: grid_height,
            width: grid_width,
            merge,
            slots: u32::try_from(span.visual_run().length())?,
        })
    }
    pub fn image_slots(&self) -> u32 {
        self.slots
    }
    pub(super) fn span(&self, start: usize) -> Result<Qwen3VLTextImageSpan> {
        Qwen3VLTextImageSpan::new(u64::try_from(start)?, self.height, self.width, self.merge)
    }
}

/// A reference's merged visual rows in the unpadded, system-prefix-cropped TE output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImage21EditImageSlots {
    pub(super) start: u32,
    pub(super) slots: u32,
}
impl QwenImage21EditImageSlots {
    pub fn image_slot_start(&self) -> u32 {
        self.start
    }
    pub fn image_slots(&self) -> u32 {
        self.slots
    }
}

#[derive(Debug, Clone)]
pub struct QwenImage21EditPrompt {
    pub(super) batch: QwenImage21T2iPromptBatch,
    pub(super) spans: Vec<Qwen3VLTextImageSpan>,
    pub(super) image_slot_ranges: Vec<QwenImage21EditImageSlots>,
    pub(super) total_image_slots: u32,
}
impl QwenImage21EditPrompt {
    pub fn prompts(&self) -> &QwenImage21T2iPromptBatch {
        &self.batch
    }
    /// First reference's slot location; retained for single-reference callers.
    pub fn image_slot_start(&self) -> u32 {
        self.image_slot_ranges[0].start
    }
    /// First reference's slot count; use `total_image_slots` for all references.
    pub fn image_slots(&self) -> u32 {
        self.image_slot_ranges[0].slots
    }
    pub fn image_slot_ranges(&self) -> &[QwenImage21EditImageSlots] {
        &self.image_slot_ranges
    }
    pub fn total_image_slots(&self) -> u32 {
        self.total_image_slots
    }
    pub fn text_config(&self, a: &Qwen3VLTextArchitecture) -> Result<Qwen3VLTextConfig> {
        ensure!(
            self.spans.len() == self.batch.batch() * self.image_slot_ranges.len(),
            "edit batch requires one span per request and reference"
        );
        Qwen3VLTextConfig::new(a.clone(), self.batch.layout().clone())?
            .with_image_spans(&self.spans)
    }
    /// All four vision outputs remain alive while separate input backings are
    /// admitted and copied. Output bytes cannot be relabelled InputPreprocess.
    pub fn managed_text_inputs(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &ArchiveTensor<'_>,
        outputs: Vec<TensorExecutionOutput>,
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            self.batch.batch() == 1,
            "single edit input requires batch one"
        );
        self.managed_text_batch_inputs(host, a, embedding, vec![outputs])
    }

    /// Each outer entry contains the four outputs of that request's own vision
    /// execution. Sources stay charged as Output while fresh InputPreprocess
    /// backings concatenate each DeepStack level and replace its image span.
    pub fn managed_text_batch_inputs(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &ArchiveTensor<'_>,
        outputs: Vec<Vec<TensorExecutionOutput>>,
    ) -> Result<Vec<ManagedHostTensor>> {
        self.managed_text_batch_inputs_multi(
            host,
            a,
            embedding,
            outputs.into_iter().map(|sample| vec![sample]).collect(),
        )
    }

    /// Output owners are grouped by request, then ordered reference, then the
    /// four independently identified vision outputs. All stay charged until
    /// their complete embedding/DeepStack copies have been admitted and made.
    pub fn managed_text_batch_inputs_multi(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &ArchiveTensor<'_>,
        outputs: Vec<Vec<Vec<TensorExecutionOutput>>>,
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            outputs.len() == self.batch.batch()
                && outputs
                    .iter()
                    .all(|sample| sample.len() == self.image_slot_ranges.len()
                        && sample.iter().all(|reference| reference.len() == 4)),
            "edit batch requires four vision outputs per request and reference"
        );
        ensure!(
            embedding.external_name() == Text::EMBEDDING_EXTERNAL_NAME
                && embedding.element_type() == Some(ElementType::Bf16)
                && embedding.shape() == [a.vocabulary_size(), a.hidden_size()],
            "edit embedding source differs from declared architecture"
        );
        let parts = outputs
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .map(|reference| {
                        reference
                            .iter()
                            .map(|o| (o.value_id(), o.tensor(), o.bytes()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.managed_text_batch_parts_multi(host, a, embedding.bytes(), &parts)
    }
    #[cfg(test)]
    pub(super) fn managed_text_parts(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &[u8],
        outputs: &[(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)],
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            self.batch.batch() == 1,
            "single edit input requires batch one"
        );
        self.managed_text_batch_parts(host, a, embedding, &[outputs.to_vec()])
    }

    #[cfg(test)]
    pub(super) fn managed_text_batch_parts(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &[u8],
        outputs: &[Vec<(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)>],
    ) -> Result<Vec<ManagedHostTensor>> {
        self.managed_text_batch_parts_multi(
            host,
            a,
            embedding,
            &outputs
                .iter()
                .map(|sample| vec![sample.clone()])
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn managed_text_batch_parts_multi(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &[u8],
        outputs: &[Vec<Vec<(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)>>],
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            outputs.len() == self.batch.batch()
                && outputs
                    .iter()
                    .all(|sample| sample.len() == self.image_slot_ranges.len()),
            "vision output groups must match the edit request and reference counts"
        );
        let config = self.text_config(a)?;
        let sources = outputs
            .iter()
            .flat_map(|sample| sample.iter().zip(&self.image_slot_ranges))
            .map(|(sample, range)| {
                let visual = ResolvedTensorSpec::new(
                    vec![u64::from(range.slots), a.hidden_size()],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )?;
                ensure!(
                    sample.len() == 4,
                    "each reference needs one main and three DeepStack vision outputs"
                );
                for (_, tensor, bytes) in sample {
                    ensure!(
                        host.owns(bytes)
                            && bytes.class() == HostMemoryClass::Output
                            && *tensor == &visual
                            && bytes.len() as u64 == visual.minimum_storage_bytes()?,
                        "vision output geometry or Host owner differs"
                    );
                    ensure!(
                        finite(bytes.bytes()),
                        "vision output contains nonfinite BF16"
                    );
                }
                let find = |id: &str| -> Result<&ManagedHostBytes> {
                    let mut found = sample.iter().filter(|(value, _, _)| value.as_str() == id);
                    let output = found
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("missing vision output {id}"))?;
                    ensure!(found.next().is_none(), "duplicate vision output {id}");
                    Ok(output.2)
                };
                Ok([
                    find(Vision::OUTPUT_ID)?,
                    find(Vision::DEEPSTACK_IDS[0])?,
                    find(Vision::DEEPSTACK_IDS[1])?,
                    find(Vision::DEEPSTACK_IDS[2])?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let batch_visual = ResolvedTensorSpec::new(
            vec![
                u64::try_from(self.batch.batch())?
                    .checked_mul(u64::from(self.total_image_slots()))
                    .ok_or_else(|| anyhow::anyhow!("batched visual rows overflow"))?,
                a.hidden_size(),
            ],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let spec = ResolvedTensorSpec::new(
            vec![self.batch.layout().rows(), a.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let stride = usize::try_from(a.hidden_size())?
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("embedding stride overflows"))?;
        let full = usize::try_from(a.vocabulary_size())?
            .checked_mul(stride)
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| anyhow::anyhow!("embedding table extent overflows"))?;
        ensure!(
            embedding.len() == full,
            "embedding source extent differs from vocabulary"
        );
        ensure!(
            self.batch
                .input_ids()
                .iter()
                .all(|&token| u64::from(token) < a.vocabulary_size()),
            "edit token exceeds vocabulary"
        );
        let requests = [
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(spec.minimum_storage_bytes()?)?,
            ),
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(batch_visual.minimum_storage_bytes()?)?,
            ),
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(batch_visual.minimum_storage_bytes()?)?,
            ),
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(batch_visual.minimum_storage_bytes()?)?,
            ),
        ];
        let mut reservations = host.reserve_many(&requests)?.into_iter();
        let mut rows = reservations.next().unwrap().allocate_zeroed()?;
        let mut span_cursor = 0;
        for (row, &token) in self.batch.input_ids().iter().enumerate() {
            while span_cursor < self.spans.len()
                && row as u64
                    >= self.spans[span_cursor].visual_run().start()
                        + self.spans[span_cursor].visual_run().length()
            {
                span_cursor += 1;
            }
            let start = self
                .spans
                .get(span_cursor)
                .map(|span| span.visual_run().start());
            let source = if let Some(start) = start.filter(|&start| row as u64 >= start) {
                let visual_row = row - usize::try_from(start)?;
                &sources[span_cursor][0].bytes()[visual_row * stride..(visual_row + 1) * stride]
            } else {
                let offset = token as usize * stride;
                &embedding[offset..offset + stride]
            };
            ensure!(finite(source), "nonfinite selected edit embedding");
            rows.as_mut()[row * stride..(row + 1) * stride].copy_from_slice(source);
        }
        let embedding = ManagedHostTensor::new(
            ProgramValueId::new(Text::EMBEDDINGS_ID)?,
            spec,
            rows.freeze(),
        )?;
        let [cos, sin] = config.managed_edit_mrope(host, &self.spans)?;
        let mut inputs = vec![embedding, cos, sin];
        for ((part, id), reservation) in Text::DEEPSTACK_IDS
            .into_iter()
            .enumerate()
            .zip(reservations)
        {
            let mut buffer = reservation.allocate_zeroed()?;
            let mut offset = 0;
            for source in &sources {
                let bytes = source[part + 1].bytes();
                buffer.as_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
                offset += bytes.len();
            }
            inputs.push(ManagedHostTensor::new(
                ProgramValueId::new(id)?,
                batch_visual.clone(),
                buffer.freeze(),
            )?);
        }
        Ok(inputs)
    }
}
fn finite(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80)
}
