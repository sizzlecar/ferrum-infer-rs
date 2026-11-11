use super::*;
use crate::qwen3_vl_vision::comfy::Qwen3VLVisionComfyFamily;
use ferrum_execution::TensorExecutionOutput;

impl PH3ImageEditPrompt {
    /// Copy the four *actual* F32 visual readbacks into their declared text
    /// inputs, together with IDs and interleaved mRoPE. All seven allocations
    /// are admitted together from the caller's existing Host Input account.
    /// Readbacks stay Output owners; none is reclassified or rounded to BF16.
    pub fn managed_inputs(
        &self,
        config: &PH3ImageTextConditioningConfig,
        host: &HostMemoryAccount,
        visual_outputs: &[TensorExecutionOutput],
    ) -> Result<Vec<ManagedHostTensor>> {
        self.managed_image_inputs(config, host, &[visual_outputs])
    }

    /// Preserve each Picture's feature rows and DeepStack outputs in input order.
    /// Every source remains an admitted Output owner until all seven text inputs
    /// have been allocated atomically and copied into their real Input owners.
    pub fn managed_image_inputs(
        &self,
        config: &PH3ImageTextConditioningConfig,
        host: &HostMemoryAccount,
        visual_outputs: &[&[TensorExecutionOutput]],
    ) -> Result<Vec<ManagedHostTensor>> {
        let text = config.encoder().text();
        ensure!(
            text.layout() == self.text.layout()
                && text.architecture().vocabulary_size() == self.text.vocabulary
                && config.encoder().visual_inputs() == Some(&self.visual),
            "image presentation differs from the declared text/visual configuration"
        );
        ensure!(
            visual_outputs.len() == self.images.len()
                && visual_outputs.iter().all(|outputs| outputs.len() == 4),
            "image text requires four actual visual outputs per ordered Picture"
        );
        let shape = ResolvedTensorSpec::new(
            vec![self.visual.rows()?, text.architecture().hidden_size()],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let source_ids = std::iter::once(Qwen3VLVisionComfyFamily::OUTPUT_ID)
            .chain(Qwen3VLVisionComfyFamily::DEEPSTACK_IDS);
        let target_ids = std::iter::once(Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID)
            .chain(Qwen3VLTextComfyFamily::DEEPSTACK_IDS);
        let mut features = BTreeMap::new();
        for (source, target) in source_ids.zip(target_ids) {
            let mut chunks = Vec::with_capacity(self.images.len());
            for (image, outputs) in self.images.iter().zip(visual_outputs) {
                let mut matches = outputs.iter().filter(|o| o.value_id().as_str() == source);
                let output = matches.next().context("missing original visual output")?;
                let expected = ResolvedTensorSpec::new(
                    vec![
                        (image.merged_height * image.merged_width) as u64,
                        text.architecture().hidden_size(),
                    ],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )?;
                ensure!(matches.next().is_none()
                    && output.tensor() == &expected
                    && output.bytes().len() as u64 == expected.minimum_storage_bytes()?
                    && host.owns(output.bytes())
                    && output.bytes().class() == HostMemoryClass::Output,
                    "visual readback has foreign ownership, duplicate identity or incompatible Picture shape");
                ensure!(
                    output
                        .bytes()
                        .bytes()
                        .chunks_exact(4)
                        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                    "visual readback contains nonfinite features"
                );
                chunks.push(output.bytes().bytes());
            }
            features.insert(ProgramValueId::new(target)?, chunks);
        }
        let specs = config.input_tensor_specs()?;
        ensure!(
            specs.len() == 7,
            "image text must declare seven original inputs"
        );
        for id in features.keys() {
            ensure!(
                specs
                    .get(id)
                    .is_some_and(|s| s.element_type == ElementType::F32
                        && s.layout == ResolvedTensorLayout::Contiguous
                        && s.dimensions == shape.dimensions()),
                "visual text port differs from the readback geometry"
            );
        }
        let sizes = specs
            .values()
            .map(|s| {
                Ok((
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(s.byte_len()?)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let reservations = host.reserve_many(&sizes)?;
        let mut inputs = Vec::with_capacity(specs.len());
        for ((id, spec), reservation) in specs.into_iter().zip(reservations) {
            let mut buffer = reservation.allocate_zeroed()?;
            match id.as_str() {
                PH3ImageTextConditioningFamily::TOKEN_IDS_ID => {
                    ensure!(
                        buffer.as_mut().len() == self.text.ids.len() * 4,
                        "image ID extent differs from text ports"
                    );
                    for (out, token) in buffer.as_mut().chunks_exact_mut(4).zip(&self.text.ids) {
                        out.copy_from_slice(&token.to_le_bytes());
                    }
                }
                Qwen3VLTextComfyFamily::COSINE_ID | Qwen3VLTextComfyFamily::SINE_ID => {
                    self.write_rotary(
                        text.architecture(),
                        id.as_str() == Qwen3VLTextComfyFamily::COSINE_ID,
                        buffer.as_mut(),
                    )?;
                }
                _ => {
                    let mut offset = 0;
                    for chunk in features.get(&id).context("undeclared image text input")? {
                        buffer.as_mut()[offset..offset + chunk.len()].copy_from_slice(chunk);
                        offset += chunk.len();
                    }
                    ensure!(
                        offset == buffer.len(),
                        "ordered Picture features do not cover text input"
                    );
                }
            }
            inputs.push(ManagedHostTensor::new(
                id,
                ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?,
                buffer.freeze(),
            )?);
        }
        Ok(inputs)
    }

    pub(super) fn write_rotary(
        &self,
        architecture: &Qwen3VLTextArchitecture,
        cosine: bool,
        output: &mut [u8],
    ) -> Result<()> {
        let width = usize::try_from(architecture.head_dim())?;
        let expected = self
            .text
            .ids
            .len()
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .context("image rotary extent overflows")?;
        ensure!(
            output.len() == expected && width <= 256,
            "image rotary output extent or head width differs"
        );
        let sections = architecture.mrope_sections();
        let mut frequencies = [0f32; 128];
        for (pair, frequency) in frequencies[..width / 2].iter_mut().enumerate() {
            *frequency = 1.
                / libm::powf(
                    architecture.rope_theta() as f32,
                    (2 * pair) as f32 / width as f32,
                );
        }
        for row in 0..self.text.ids.len() {
            let positions = self.position_ids(row)?;
            for (pair, &frequency) in frequencies[..width / 2].iter().enumerate() {
                // Pinned interleaved Comfy: start with all T frequencies,
                // replace 1::3 by H and 2::3 by W, each to its own section end.
                let axis = if pair % 3 == 1 && (pair as u64) < sections[1] * 3 {
                    1
                } else if pair % 3 == 2 && (pair as u64) < sections[2] * 3 {
                    2
                } else {
                    0
                };
                let phase = positions[axis] as f32 * frequency;
                let bytes = if cosine {
                    libm::cosf(phase)
                } else {
                    libm::sinf(phase)
                }
                .to_le_bytes();
                for column in [pair, pair + width / 2] {
                    let start = (row * width + column) * 4;
                    output[start..start + 4].copy_from_slice(&bytes);
                }
            }
        }
        Ok(())
    }
}
