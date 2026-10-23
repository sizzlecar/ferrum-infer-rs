//! One reference image per request. Batch rows retain independent image spans;
//! DeepStack remains three distinct graph inputs in the same request order.
use super::*;
use crate::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Text,
    Qwen3VLTextImageSpan,
};
use crate::qwen3_vl_vision::Qwen3VLVisionFamily as Vision;
use ferrum_execution::TensorExecutionOutput;
use ferrum_quantization::SafetensorsTensor;
use ferrum_runtime::*;

#[derive(Debug, Clone)]
pub struct QwenImage21EditPrompt {
    pub(super) batch: QwenImage21T2iPromptBatch,
    pub(super) spans: Vec<Qwen3VLTextImageSpan>,
    pub(super) image_slot_start: u32,
    pub(super) image_slots: u32,
}
impl QwenImage21EditPrompt {
    pub fn prompts(&self) -> &QwenImage21T2iPromptBatch {
        &self.batch
    }
    pub fn image_slot_start(&self) -> u32 {
        self.image_slot_start
    }
    pub fn image_slots(&self) -> u32 {
        self.image_slots
    }
    pub fn text_config(&self, a: &Qwen3VLTextArchitecture) -> Result<Qwen3VLTextConfig> {
        ensure!(
            self.spans.len() == self.batch.batch(),
            "edit batch requires one image span per request"
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
        embedding: &SafetensorsTensor<'_>,
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
        embedding: &SafetensorsTensor<'_>,
        outputs: Vec<Vec<TensorExecutionOutput>>,
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            outputs.len() == self.batch.batch() && outputs.iter().all(|sample| sample.len() == 4),
            "edit batch requires four vision outputs per request"
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
                    .map(|o| (o.value_id(), o.tensor(), o.bytes()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.managed_text_batch_parts(host, a, embedding.bytes(), &parts)
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

    pub(super) fn managed_text_batch_parts(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &[u8],
        outputs: &[Vec<(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)>],
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            outputs.len() == self.batch.batch(),
            "vision output groups must match the edit request count"
        );
        let config = self.text_config(a)?;
        let visual = ResolvedTensorSpec::new(
            vec![u64::from(self.image_slots()), a.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let sources = outputs
            .iter()
            .map(|sample| {
                ensure!(
                    sample.len() == 4,
                    "each request needs one main and three DeepStack vision outputs"
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
                    .checked_mul(u64::from(self.image_slots()))
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
        for (row, &token) in self.batch.input_ids().iter().enumerate() {
            let sample = row / self.batch.sequence_length();
            let start = usize::try_from(self.spans[sample].visual_run().start())?;
            let end = start + self.image_slots() as usize;
            let source = if (start..end).contains(&row) {
                &sources[sample][0].bytes()[(row - start) * stride..(row - start + 1) * stride]
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
        let visual_bytes = usize::try_from(visual.minimum_storage_bytes()?)?;
        for ((part, id), reservation) in Text::DEEPSTACK_IDS
            .into_iter()
            .enumerate()
            .zip(reservations)
        {
            let mut buffer = reservation.allocate_zeroed()?;
            for (sample, source) in sources.iter().enumerate() {
                buffer.as_mut()[sample * visual_bytes..(sample + 1) * visual_bytes]
                    .copy_from_slice(source[part + 1].bytes());
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
