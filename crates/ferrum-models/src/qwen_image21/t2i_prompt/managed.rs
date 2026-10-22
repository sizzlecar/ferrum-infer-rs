use super::*;
use crate::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Family,
};
use crate::qwen_image21::QwenImage21TransformerFamily;
use ferrum_execution::TensorExecutionOutput;
use ferrum_quantization::SafetensorsTensor;
use ferrum_runtime::*;
use half::bf16;

pub struct QwenImage21T2iManagedInputs {
    config: Qwen3VLTextConfig,
    tensors: [ManagedHostTensor; 3],
}
impl QwenImage21T2iManagedInputs {
    pub fn config(&self) -> &Qwen3VLTextConfig {
        &self.config
    }
    pub fn tensors(&self) -> &[ManagedHostTensor; 3] {
        &self.tensors
    }
    pub fn into_parts(self) -> (Qwen3VLTextConfig, [ManagedHostTensor; 3]) {
        (self.config, self.tensors)
    }
}

/// Right-padded, post-system-crop BF16 features for the DiT text input port.
pub struct QwenImage21T2iTextFeatures {
    tensor: ManagedHostTensor,
    sequence_length: u32,
    valid_lengths: Vec<u32>,
}
impl QwenImage21T2iTextFeatures {
    pub fn tensor(&self) -> &ManagedHostTensor {
        &self.tensor
    }
    pub fn sequence_length(&self) -> u32 {
        self.sequence_length
    }
    pub fn valid_lengths(&self) -> &[u32] {
        &self.valid_lengths
    }
    pub fn into_tensor(self) -> ManagedHostTensor {
        self.tensor
    }
}

impl QwenImage21T2iPromptBatch {
    /// The source is an already validated borrowed archive mapping, not a new
    /// owned copy. Only selected rows are read, including actual PAD embeddings.
    /// Metadata and token ranges are checked before reserving all three inputs
    /// atomically; a failure never publishes partially initialized tensors.
    pub fn managed_inputs(
        &self,
        account: &HostMemoryAccount,
        architecture: &Qwen3VLTextArchitecture,
        embedding: &SafetensorsTensor<'_>,
    ) -> Result<QwenImage21T2iManagedInputs> {
        ensure!(
            embedding.external_name() == Family::EMBEDDING_EXTERNAL_NAME,
            "unexpected embedding source tensor"
        );
        ensure!(
            embedding.element_type() == Some(ElementType::Bf16),
            "embedding source must be BF16"
        );
        ensure!(
            embedding.shape() == [architecture.vocabulary_size(), architecture.hidden_size()],
            "embedding source shape differs from text architecture"
        );
        self.prepare_bytes(account, architecture, embedding.bytes())
    }

    pub(super) fn prepare_bytes(
        &self,
        account: &HostMemoryAccount,
        architecture: &Qwen3VLTextArchitecture,
        embedding: &[u8],
    ) -> Result<QwenImage21T2iManagedInputs> {
        let config = Qwen3VLTextConfig::new(architecture.clone(), self.layout.clone())?;
        let h = usize::try_from(architecture.hidden_size())?;
        let d = usize::try_from(architecture.head_dim())?;
        let row_bytes = h
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("embedding row overflows"))?;
        let full_bytes = usize::try_from(architecture.vocabulary_size())?
            .checked_mul(row_bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| anyhow::anyhow!("embedding source size overflows"))?;
        ensure!(
            embedding.len() == full_bytes,
            "embedding source byte length differs from metadata"
        );
        ensure!(
            self.ids
                .iter()
                .all(|&id| u64::from(id) < architecture.vocabulary_size()),
            "token ID exceeds embedding vocabulary"
        );
        ensure!(
            self.sequence_length() <= 1 << 24,
            "positions must be exactly representable in FP32"
        );
        let specs = config.input_tensor_specs()?;
        let ids = [Family::EMBEDDINGS_ID, Family::COSINE_ID, Family::SINE_ID];
        let tensors = ids
            .map(|id| {
                let spec = &specs[&ProgramValueId::new(id)?];
                Ok(ResolvedTensorSpec::new(
                    spec.dimensions.clone(),
                    spec.element_type,
                    spec.layout.clone(),
                )?)
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let requests = tensors
            .iter()
            .map(|t| {
                Ok((
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(t.minimum_storage_bytes()?)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reservations = account.reserve_many(&requests)?.into_iter();
        let mut input = reservations.next().unwrap().allocate_zeroed()?;
        let mut cosine = reservations.next().unwrap().allocate_zeroed()?;
        let mut sine = reservations.next().unwrap().allocate_zeroed()?;
        for (row, &token) in self.ids.iter().enumerate() {
            let start = token as usize * row_bytes;
            let selected = &embedding[start..start + row_bytes];
            ensure!(
                finite_bf16(selected),
                "selected embedding row contains nonfinite BF16"
            );
            input.as_mut()[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(selected);
        }
        // All three axes have the same padded-column position in pure T2I, so
        // interleaved mRoPE selection leaves this frequency vector unchanged.
        // Recompute at most D/2 frequencies, with no input-sized scratch buffer.
        for pair in 0..d / 2 {
            let exponent = (2 * pair) as f32 / d as f32;
            let inverse = 1.0_f32 / libm::powf(architecture.rope_theta() as f32, exponent);
            ensure!(
                inverse.is_finite() && inverse > 0.0,
                "nonfinite/zero text rotary frequency"
            );
            for column in 0..self.sequence_length() {
                let phase = column as f32 * inverse;
                let c = bf16::from_f32(libm::cosf(phase)).to_bits().to_le_bytes();
                let s = bf16::from_f32(libm::sinf(phase)).to_bits().to_le_bytes();
                for batch in 0..self.batch() {
                    let row = batch * self.sequence_length() + column;
                    for channel in [pair, pair + d / 2] {
                        let offset = (row * d + channel) * 2;
                        cosine.as_mut()[offset..offset + 2].copy_from_slice(&c);
                        sine.as_mut()[offset..offset + 2].copy_from_slice(&s);
                    }
                }
            }
        }
        let mut specs = tensors.into_iter();
        Ok(QwenImage21T2iManagedInputs {
            config,
            tensors: [
                ManagedHostTensor::new(
                    ProgramValueId::new(ids[0])?,
                    specs.next().unwrap(),
                    input.freeze(),
                )?,
                ManagedHostTensor::new(
                    ProgramValueId::new(ids[1])?,
                    specs.next().unwrap(),
                    cosine.freeze(),
                )?,
                ManagedHostTensor::new(
                    ProgramValueId::new(ids[2])?,
                    specs.next().unwrap(),
                    sine.freeze(),
                )?,
            ],
        })
    }

    pub fn crop_hidden(
        &self,
        account: &HostMemoryAccount,
        architecture: &Qwen3VLTextArchitecture,
        hidden: &TensorExecutionOutput,
    ) -> Result<QwenImage21T2iTextFeatures> {
        ensure!(
            hidden.value_id().as_str() == Family::OUTPUT_ID,
            "expected the pre-final-norm text output"
        );
        self.crop_bytes(
            account,
            architecture.hidden_size(),
            hidden.tensor(),
            hidden.bytes(),
        )
    }

    pub(super) fn crop_bytes(
        &self,
        account: &HostMemoryAccount,
        hidden_size: u64,
        spec: &ResolvedTensorSpec,
        hidden: &ManagedHostBytes,
    ) -> Result<QwenImage21T2iTextFeatures> {
        ensure!(
            account.owns(hidden),
            "text output belongs to a different Host account"
        );
        let expected = ResolvedTensorSpec::new(
            vec![self.layout.rows(), hidden_size],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        ensure!(
            spec == &expected
                && hidden.len() == usize::try_from(expected.minimum_storage_bytes()?)?,
            "pre-final-norm output shape/dtype/length mismatch"
        );
        let sequence_length = u32::try_from(self.feature_sequence_length())?;
        let valid_lengths = self
            .valid_feature_lengths()
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tensor = ResolvedTensorSpec::new(
            vec![
                self.batch() as u64 * u64::from(sequence_length),
                hidden_size,
            ],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let mut output = account.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(tensor.minimum_storage_bytes()?)?,
        )?;
        let row_bytes = usize::try_from(hidden_size)?
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("hidden row byte size overflows"))?;
        for (batch, &length) in self.lengths.iter().enumerate() {
            let count = length - self.drop_prefix;
            let source_row =
                batch * self.sequence_length() + self.sequence_length() - length + self.drop_prefix;
            let start = source_row * row_bytes;
            let bytes = &hidden.bytes()[start..start + count * row_bytes];
            ensure!(
                finite_bf16(bytes),
                "cropped text hidden state contains nonfinite BF16"
            );
            let out = batch * sequence_length as usize * row_bytes;
            output.as_mut()[out..out + bytes.len()].copy_from_slice(bytes);
        }
        Ok(QwenImage21T2iTextFeatures {
            tensor: ManagedHostTensor::new(
                ProgramValueId::new(QwenImage21TransformerFamily::TEXT_ID)?,
                tensor,
                output.freeze(),
            )?,
            sequence_length,
            valid_lengths,
        })
    }
}
fn finite_bf16(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80)
}
