//! Single-reference prompt conditioning. Visual rows replace image placeholders
//! before the language tower; DeepStack remains three distinct graph inputs.
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
    pub(super) span: Qwen3VLTextImageSpan,
}
impl QwenImage21EditPrompt {
    pub fn prompts(&self) -> &QwenImage21T2iPromptBatch {
        &self.batch
    }
    pub fn image_slot_start(&self) -> u32 {
        (self.span.visual_run().start() as usize - self.batch.drop_prefix_tokens()) as u32
    }
    pub fn image_slots(&self) -> u32 {
        self.span.visual_run().length() as u32
    }
    pub fn text_config(&self, a: &Qwen3VLTextArchitecture) -> Result<Qwen3VLTextConfig> {
        Qwen3VLTextConfig::new(a.clone(), self.batch.layout().clone())?
            .with_image_spans(std::slice::from_ref(&self.span))
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
            embedding.external_name() == Text::EMBEDDING_EXTERNAL_NAME
                && embedding.element_type() == Some(ElementType::Bf16)
                && embedding.shape() == [a.vocabulary_size(), a.hidden_size()],
            "edit embedding source differs from declared architecture"
        );
        let parts = outputs
            .iter()
            .map(|o| (o.value_id(), o.tensor(), o.bytes()))
            .collect::<Vec<_>>();
        self.managed_text_parts(host, a, embedding.bytes(), &parts)
    }
    pub(super) fn managed_text_parts(
        &self,
        host: &HostMemoryAccount,
        a: &Qwen3VLTextArchitecture,
        embedding: &[u8],
        outputs: &[(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)],
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            outputs.len() == 4,
            "vision must return one main and three DeepStack outputs"
        );
        let visual = ResolvedTensorSpec::new(
            vec![u64::from(self.image_slots()), a.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        for (_, tensor, bytes) in outputs {
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
            let mut found = outputs.iter().filter(|(value, _, _)| value.as_str() == id);
            let output = found
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing vision output {id}"))?;
            ensure!(found.next().is_none(), "duplicate vision output {id}");
            Ok(output.2)
        };
        let main = find(Vision::OUTPUT_ID)?;
        let deepstack = Vision::DEEPSTACK_IDS
            .map(find)
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let config = self.text_config(a)?;
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
                usize::try_from(visual.minimum_storage_bytes()?)?,
            ),
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(visual.minimum_storage_bytes()?)?,
            ),
            (
                HostMemoryClass::InputPreprocess,
                usize::try_from(visual.minimum_storage_bytes()?)?,
            ),
        ];
        let mut reservations = host.reserve_many(&requests)?.into_iter();
        let mut rows = reservations.next().unwrap().allocate_zeroed()?;
        let start = usize::try_from(self.span.visual_run().start())?;
        let end = start + self.image_slots() as usize;
        for (row, &token) in self.batch.input_ids().iter().enumerate() {
            let source = if (start..end).contains(&row) {
                &main.bytes()[(row - start) * stride..(row - start + 1) * stride]
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
        let [cos, sin] = config.managed_edit_mrope(host, std::slice::from_ref(&self.span))?;
        let mut inputs = vec![embedding, cos, sin];
        for ((id, source), reservation) in Text::DEEPSTACK_IDS
            .into_iter()
            .zip(deepstack)
            .zip(reservations)
        {
            let mut buffer = reservation.allocate_zeroed()?;
            buffer.as_mut().copy_from_slice(source.bytes());
            inputs.push(ManagedHostTensor::new(
                ProgramValueId::new(id)?,
                visual.clone(),
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
