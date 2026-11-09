//! Bridge actual main visual output into immutable causal request inputs. The
//! original Output remains charged while separately admitted Input is written.
//! Per-wave zero-filled slices are accounted by the causal executor, separately
//! from these retained feature, mask and complete continuation-position tables.
use super::*;
use ferrum_execution::{
    CausalInputSchema, CausalInputSegment, CausalRequestInputs, CausalTokenInput,
    CausalTokenInputSpec, CausalTokenInputStorage, TensorExecutionOutput,
};
use ferrum_models::vnext::qwen35::Qwen35DensePrecision;
use ferrum_models::{
    p_h3_image::{PH3ImagePromptExpansionImageTokens, PH3ImagePromptExpansionRotary},
    qwen35_config::Qwen35TextConfig,
    qwen3_vl_vision::hf::Qwen3VLVisionHfFamily,
    vnext::qwen35::{
        ROPE_COS_INPUT_ID, ROPE_SIN_INPUT_ID, VISUAL_REPLACE_MASK_INPUT_ID, VISUAL_ROWS_INPUT_ID,
    },
};
use ferrum_runtime::{
    ElementType, HostMemoryClass, ManagedHostBytes, ProgramValueId, ResolvedTensorLayout,
};
use half::{bf16, f16};

pub(super) struct VisualExpansionInputs {
    tokens: PH3ImagePromptExpansionImageTokens,
    hidden: usize,
    rotary: PH3ImagePromptExpansionRotary,
    schema: CausalInputSchema,
    precision: Qwen35DensePrecision,
}

impl VisualExpansionInputs {
    pub(super) fn new(
        tokens: PH3ImagePromptExpansionImageTokens,
        text: &Qwen35TextConfig,
    ) -> Result<Self> {
        Self::with_precision(tokens, text, Qwen35DensePrecision::F16)
    }
    pub(super) fn with_precision(
        tokens: PH3ImagePromptExpansionImageTokens,
        text: &Qwen35TextConfig,
        precision: Qwen35DensePrecision,
    ) -> Result<Self> {
        let element_type = match precision {
            Qwen35DensePrecision::F16 => ElementType::F16,
            Qwen35DensePrecision::Bf16 => ElementType::Bf16,
        };
        let rotary = PH3ImagePromptExpansionRotary::new(text).map_err(model_error)?;
        if text.hidden_size == 0 || tokens.image_runs().is_empty() {
            return Err(FerrumError::request_validation(
                "visual expansion requires actual image rows and a positive hidden width",
            ));
        }
        let specs = [
            (
                VISUAL_ROWS_INPUT_ID,
                element_type,
                vec![text.hidden_size as u64],
                CausalTokenInputStorage::ZeroOutsideSegments,
            ),
            (
                VISUAL_REPLACE_MASK_INPUT_ID,
                ElementType::U8,
                vec![],
                CausalTokenInputStorage::ZeroOutsideSegments,
            ),
            (
                ROPE_COS_INPUT_ID,
                element_type,
                vec![rotary.pairs() as u64],
                CausalTokenInputStorage::Exact,
            ),
            (
                ROPE_SIN_INPUT_ID,
                element_type,
                vec![rotary.pairs() as u64],
                CausalTokenInputStorage::Exact,
            ),
        ]
        .into_iter()
        .map(|(id, element_type, row_dimensions, storage)| {
            Ok(CausalTokenInputSpec {
                value_id: ProgramValueId::new(id).map_err(model_error)?,
                element_type,
                row_dimensions,
                storage,
            })
        })
        .collect::<Result<Vec<_>>>()?;
        let result = Self {
            tokens,
            precision,
            hidden: text.hidden_size,
            rotary,
            schema: CausalInputSchema::new(specs)?,
        };
        result.demands()?;
        Ok(result)
    }

    pub(super) fn tokens(&self) -> &PH3ImagePromptExpansionImageTokens {
        &self.tokens
    }
    pub(super) fn schema(&self) -> &CausalInputSchema {
        &self.schema
    }

    fn extent(rows: usize, width: usize, element: usize) -> Result<usize> {
        rows.checked_mul(width)
            .and_then(|n| n.checked_mul(element))
            .filter(|&n| n > 0 && n <= isize::MAX as usize)
            .ok_or_else(|| FerrumError::request_validation("visual causal input extent overflows"))
    }

    fn demands(&self) -> Result<Vec<(HostMemoryClass, usize)>> {
        let mut demands = Vec::with_capacity(self.tokens.image_runs().len() * 2 + 2);
        for run in self.tokens.image_runs() {
            demands.push((
                HostMemoryClass::InputPreprocess,
                Self::extent(run.rows(), self.hidden, 2)?,
            ));
            demands.push((HostMemoryClass::InputPreprocess, run.rows()));
        }
        let table = Self::extent(self.tokens.maximum_tokens(), self.rotary.pairs(), 2)?;
        demands.extend([(HostMemoryClass::InputPreprocess, table); 2]);
        Ok(demands)
    }

    /// Persistent Input only. Include the original visual Output in transition
    /// peak accounting, and add the compiled executor's independent wave demand.
    pub(super) fn retained_input_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        self.demands()?
            .into_iter()
            .try_fold(0usize, |sum, (_, bytes)| {
                sum.checked_add(host.backing_bytes(bytes).map_err(resource_error)?)
                    .ok_or_else(|| FerrumError::config("visual persistent Input backing overflows"))
            })
    }

    pub(super) fn bind_outputs(
        &self,
        host: &HostMemoryAccount,
        outputs: &[TensorExecutionOutput],
    ) -> Result<Arc<CausalRequestInputs>> {
        if outputs.len() != self.tokens.image_runs().len() {
            return Err(FerrumError::request_validation(
                "visual output count differs from the ordered actual images",
            ));
        }
        for (output, run) in outputs.iter().zip(self.tokens.image_runs()) {
            if output.value_id().as_str() != Qwen3VLVisionHfFamily::OUTPUT_ID
                || output.tensor().element_type() != ElementType::F32
                || output.tensor().layout() != &ResolvedTensorLayout::Contiguous
                || output.tensor().dimensions() != [run.rows() as u64, self.hidden as u64]
            {
                return Err(FerrumError::request_validation("visual main output identity, dtype or shape differs from the original image token run"));
            }
        }
        self.bind_sources(host, &outputs.iter().map(|o| o.bytes()).collect::<Vec<_>>())
    }

    fn bind_sources(
        &self,
        host: &HostMemoryAccount,
        outputs: &[&ManagedHostBytes],
    ) -> Result<Arc<CausalRequestInputs>> {
        if outputs.len() != self.tokens.image_runs().len() {
            return Err(FerrumError::request_validation(
                "visual image count changed",
            ));
        }
        for (output, run) in outputs.iter().zip(self.tokens.image_runs()) {
            if output.class() != HostMemoryClass::Output
                || !host.owns(output)
                || output.len() != Self::extent(run.rows(), self.hidden, 4)?
            {
                return Err(FerrumError::request_validation(
                    "visual features lack the original Host Output owner or exact F32 extent",
                ));
            }
        }
        // Reserve the entire bridge before allocating any of its payloads.
        let mut grants = host
            .reserve_many(&self.demands()?)
            .map_err(resource_error)?
            .into_iter();
        let mut visual = Vec::with_capacity(outputs.len());
        let mut mask = Vec::with_capacity(outputs.len());
        for (output, run) in outputs.iter().zip(self.tokens.image_runs()) {
            let mut rows = grants
                .next()
                .expect("feature demand")
                .allocate_zeroed()
                .map_err(resource_error)?;
            for (src, dst) in output
                .bytes()
                .chunks_exact(4)
                .zip(rows.as_mut().chunks_exact_mut(2))
            {
                let value = f32::from_le_bytes(src.try_into().expect("F32 chunk"));
                let (bytes, finite) = match self.precision {
                    Qwen35DensePrecision::F16 => {
                        let half = f16::from_f32(value);
                        (half.to_le_bytes(), half.is_finite())
                    }
                    Qwen35DensePrecision::Bf16 => {
                        let half = bf16::from_f32(value);
                        (half.to_le_bytes(), half.is_finite())
                    }
                };
                if !value.is_finite() || !finite {
                    return Err(FerrumError::backend(
                        "visual F32 output cannot be represented as finite selected causal input",
                    ));
                }
                dst.copy_from_slice(&bytes);
            }
            let mut ones = grants
                .next()
                .expect("mask demand")
                .allocate_zeroed()
                .map_err(resource_error)?;
            ones.as_mut().fill(1);
            visual.push(CausalInputSegment::new(
                run.token_start(),
                run.rows(),
                rows.freeze(),
            )?);
            mask.push(CausalInputSegment::new(
                run.token_start(),
                run.rows(),
                ones.freeze(),
            )?);
        }
        let mut cosine = grants
            .next()
            .expect("cosine demand")
            .allocate_zeroed()
            .map_err(resource_error)?;
        let mut sine = grants
            .next()
            .expect("sine demand")
            .allocate_zeroed()
            .map_err(resource_error)?;
        let row_bytes = self.rotary.pairs() * 2;
        for (physical, (cos, sin)) in cosine
            .as_mut()
            .chunks_exact_mut(row_bytes)
            .zip(sine.as_mut().chunks_exact_mut(row_bytes))
            .enumerate()
        {
            self.rotary
                .write_row_with_precision(
                    self.tokens.position(physical).map_err(model_error)?,
                    self.precision,
                    cos,
                    sin,
                )
                .map_err(model_error)?;
        }
        let n = self.tokens.maximum_tokens();
        let values = [
            (VISUAL_ROWS_INPUT_ID, visual),
            (VISUAL_REPLACE_MASK_INPUT_ID, mask),
            (
                ROPE_COS_INPUT_ID,
                vec![CausalInputSegment::new(0, n, cosine.freeze())?],
            ),
            (
                ROPE_SIN_INPUT_ID,
                vec![CausalInputSegment::new(0, n, sine.freeze())?],
            ),
        ]
        .into_iter()
        .map(|(id, segments)| {
            Ok(CausalTokenInput::new(
                ProgramValueId::new(id).map_err(model_error)?,
                segments,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(CausalRequestInputs::new(
            host,
            &self.schema,
            n,
            values,
        )?))
    }
}

fn resource_error(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::resource_exhausted(format!("visual expansion input: {error}"))
}

#[cfg(test)]
mod tests;
