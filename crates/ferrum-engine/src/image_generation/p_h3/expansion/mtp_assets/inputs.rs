//! Main conditioning retains the original visual owner. Text-only requests
//! use zero visual rows/mask and ordinary positions through the same schema.
//! Draft hidden is deliberately absent here: it is known only after the main
//! selected commit and must be supplied by that wave's accepted-output owner.
use super::*;
use ferrum_execution::{
    CausalInputSchema, CausalInputSegment, CausalRequestInputs, CausalTokenInput,
    CausalTokenInputSpec, CausalTokenInputStorage,
};
use ferrum_models::{
    p_h3_image::PH3ImagePromptExpansionRotary,
    vnext::qwen35::{
        ROPE_COS_INPUT_ID, ROPE_SIN_INPUT_ID, VISUAL_REPLACE_MASK_INPUT_ID, VISUAL_ROWS_INPUT_ID,
    },
};

impl MtpExpansionAssets {
    pub(crate) fn main_schema(&self) -> Result<CausalInputSchema> {
        schema(&self.text)
    }

    /// Persistent inputs only. Original visual outputs and per-wave scratch
    /// must be included separately in the actual transition peak.
    pub(crate) fn main_input_backing_bytes(
        &self,
        native: &NativeTextExpansion,
        prepared: &PreparedTextExpansion,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        self.validate_request(native, prepared)?;
        if let Some(visual) = &prepared.visual {
            return visual.retained_input_bytes(host);
        }
        let (_, bytes) = text_extent(&self.text, context(prepared)?)?;
        let padded = host.backing_bytes(bytes).map_err(model_error)?;
        add(padded, padded)
    }

    pub(crate) fn main_inputs(
        &self,
        native: &NativeTextExpansion,
        prepared: &PreparedTextExpansion,
        host: &HostMemoryAccount,
    ) -> Result<Arc<CausalRequestInputs>> {
        self.validate_request(native, prepared)?;
        native.validate_bound_inputs(prepared)?;
        if let Some(inputs) = &prepared.causal_inputs {
            if inputs.schema() != &self.main_schema()? {
                return Err(FerrumError::request_validation(
                    "MTP visual input schema changed",
                ));
            }
            // Row selection checks all original Host owners. The original
            // exact RoPE table returns only a slice, without payload copying.
            inputs.materialize_rows(host, &value(ROPE_COS_INPUT_ID)?, 0..1)?;
            return Ok(Arc::clone(inputs));
        }
        text_inputs(&self.text, context(prepared)?, host)
    }

    fn validate_request(
        &self,
        native: &NativeTextExpansion,
        prepared: &PreparedTextExpansion,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.domain, &native.domain) {
            return Err(FerrumError::request_validation(
                "MTP assets belong to another source owner",
            ));
        }
        native.validate_preparation(prepared)
    }
}

fn context(prepared: &PreparedTextExpansion) -> Result<usize> {
    let rows = prepared
        .tokens
        .len()
        .checked_add(prepared.generation.max_new_tokens as usize)
        .filter(|&n| n > 0)
        .ok_or_else(|| FerrumError::request_validation("MTP context overflows"))?;
    if prepared
        .engine
        .runtime
        .max_model_len
        .is_some_and(|n| rows > n)
    {
        return Err(FerrumError::request_validation(
            "MTP inputs exceed the prepared context",
        ));
    }
    Ok(rows)
}

fn schema(text: &Qwen35TextConfig) -> Result<CausalInputSchema> {
    let rotary = PH3ImagePromptExpansionRotary::new(text).map_err(model_error)?;
    let specs = [
        (
            VISUAL_ROWS_INPUT_ID,
            ElementType::F16,
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
            ElementType::F16,
            vec![rotary.pairs() as u64],
            CausalTokenInputStorage::Exact,
        ),
        (
            ROPE_SIN_INPUT_ID,
            ElementType::F16,
            vec![rotary.pairs() as u64],
            CausalTokenInputStorage::Exact,
        ),
    ]
    .into_iter()
    .map(|(id, element_type, row_dimensions, storage)| {
        Ok(CausalTokenInputSpec {
            value_id: value(id)?,
            element_type,
            row_dimensions,
            storage,
        })
    })
    .collect::<Result<Vec<_>>>()?;
    CausalInputSchema::new(specs)
}

fn text_extent(
    text: &Qwen35TextConfig,
    rows: usize,
) -> Result<(PH3ImagePromptExpansionRotary, usize)> {
    let rotary = PH3ImagePromptExpansionRotary::new(text).map_err(model_error)?;
    let bytes = rows
        .checked_mul(rotary.pairs())
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n > 0 && n <= isize::MAX as usize)
        .ok_or_else(|| FerrumError::request_validation("MTP rotary table overflows"))?;
    Ok((rotary, bytes))
}

fn text_inputs(
    text: &Qwen35TextConfig,
    rows: usize,
    host: &HostMemoryAccount,
) -> Result<Arc<CausalRequestInputs>> {
    let (rotary, bytes) = text_extent(text, rows)?;
    let schema = schema(text)?;
    let mut grants = host
        .reserve_many(&[(HostMemoryClass::InputPreprocess, bytes); 2])
        .map_err(model_error)?
        .into_iter();
    let mut cosine = grants
        .next()
        .expect("cosine demand")
        .allocate_zeroed()
        .map_err(model_error)?;
    let mut sine = grants
        .next()
        .expect("sine demand")
        .allocate_zeroed()
        .map_err(model_error)?;
    for (position, (cos, sin)) in cosine
        .as_mut()
        .chunks_exact_mut(rotary.pairs() * 2)
        .zip(sine.as_mut().chunks_exact_mut(rotary.pairs() * 2))
        .enumerate()
    {
        rotary
            .write_row([position as u64; 3], cos, sin)
            .map_err(model_error)?;
    }
    let values = vec![
        CausalTokenInput::new(value(VISUAL_ROWS_INPUT_ID)?, vec![]),
        CausalTokenInput::new(value(VISUAL_REPLACE_MASK_INPUT_ID)?, vec![]),
        CausalTokenInput::new(
            value(ROPE_COS_INPUT_ID)?,
            vec![CausalInputSegment::new(0, rows, cosine.freeze())?],
        ),
        CausalTokenInput::new(
            value(ROPE_SIN_INPUT_ID)?,
            vec![CausalInputSegment::new(0, rows, sine.freeze())?],
        ),
    ];
    Ok(Arc::new(CausalRequestInputs::new(
        host, &schema, rows, values,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_generation::p_h3::expansion::tests::fixture::{host, Fixture};

    #[test]
    fn mtp_text_inputs_account_tables_zero_visual_and_actual_main_positions() {
        let fixture = Fixture::new();
        let native = fixture.expansion();
        let raw = serde_json::from_slice(native.defined.sources().config_json()).unwrap();
        let mut text = Qwen35TextConfig::from_hf_config_value(&raw).unwrap();
        text.rope_parameters.partial_rotary_factor = 1.;
        text.rope_parameters.mrope_interleaved = true;
        text.rope_parameters.mrope_section = Some(vec![1, 0, 0]);
        let host = host();
        let bytes = host.backing_bytes(8).unwrap() * 2;
        let short = host
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes - 1)])
            .unwrap();
        assert!(text_inputs(&text, 4, &short.account()).is_err());
        drop(short);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let inputs = text_inputs(&text, 4, &host).unwrap();
        assert_eq!(host.usage().unwrap().allocated_bytes(), bytes);
        let (_, cos) = inputs
            .materialize_rows(&host, &value(ROPE_COS_INPUT_ID).unwrap(), 2..3)
            .unwrap();
        let (_, sin) = inputs
            .materialize_rows(&host, &value(ROPE_SIN_INPUT_ID).unwrap(), 2..3)
            .unwrap();
        // One frequency at main position 2; independent rounded constants,
        // not a repetition of the implementation's trig calculation.
        assert_eq!(cos.bytes(), &half::f16::from_f32(-0.41614684).to_le_bytes());
        assert_eq!(sin.bytes(), &half::f16::from_f32(0.9092974).to_le_bytes());
        let (_, visual) = inputs
            .materialize_rows(&host, &value(VISUAL_ROWS_INPUT_ID).unwrap(), 1..3)
            .unwrap();
        assert!(visual.bytes().iter().all(|b| *b == 0));
        let (_, mask) = inputs
            .materialize_rows(&host, &value(VISUAL_REPLACE_MASK_INPUT_ID).unwrap(), 1..3)
            .unwrap();
        assert_eq!(mask.bytes(), &[0, 0]);
        drop((inputs, cos, sin, visual, mask));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
