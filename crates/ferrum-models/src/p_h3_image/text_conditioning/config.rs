use super::*;

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageTextConditioningLimits {
    pub max_metadata_bytes: usize,
    /// Encoder projections + promoted norms + original INT8 embedding/scales.
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
impl PH3ImageTextConditioningLimits {
    pub(super) fn encoder(self) -> Qwen3VLTextComfyLimits {
        Qwen3VLTextComfyLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_stored_bytes: self.max_stored_bytes,
            max_plan_nodes: self.max_plan_nodes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageTextConditioningConfig {
    encoder: Qwen3VLTextComfyConfig,
    connector_output_dim: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    encoder: Qwen3VLTextComfyConfig,
    connector_output_dim: u64,
}
impl<'de> Deserialize<'de> for PH3ImageTextConditioningConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        Self::new(w.encoder, w.connector_output_dim).map_err(serde::de::Error::custom)
    }
}
impl PH3ImageTextConditioningConfig {
    /// Checked composition, also usable with scaled deterministic fixtures.
    /// Five taps are mandatory; the last must be the normalized final layer.
    pub fn new(
        encoder: Qwen3VLTextComfyConfig,
        connector_output_dim: u64,
    ) -> std::result::Result<Self, VNextError> {
        let text = encoder.text();
        if text.hidden_state_indices().len() != 5
            || text.hidden_state_indices().last() != Some(&text.architecture().layers())
            || connector_output_dim == 0
        {
            return Err(invalid("requires exactly five taps ending at the final normalized layer and positive connector width"));
        }
        // Reserve the sink at the first *real* token. Left/interior padding is
        // incompatible with this component's raw-text valid-prefix contract.
        for runs in text.layout().key_runs() {
            if !runs[0].is_valid() || runs.len() > 2 || (runs.len() == 2 && runs[1].is_valid()) {
                return Err(invalid(
                    "conditioning accepts only nonempty valid prefixes with optional right padding",
                ));
            }
        }
        let c = Self {
            encoder,
            connector_output_dim,
        };
        c.connector_config().map_err(|e| invalid(e.to_string()))?;
        c.input_tensor_specs()?;
        Ok(c)
    }
    /// Explicit visual mode adds the main row replacement and the original
    /// three DeepStack injections, preserving the complete Stored schema.
    pub fn with_visual_inputs(
        self,
        visual: crate::qwen3_vl_text::Qwen3VLTextDeepStack,
    ) -> std::result::Result<Self, VNextError> {
        Self::new(
            self.encoder.with_visual_inputs(visual)?,
            self.connector_output_dim,
        )
    }
    /// The actual P-H3 target identity, independent of client names. Source
    /// metadata chooses each W4/W8 route; the embedding dtype is explicitly
    /// fixed by the pinned Comfy orig_dtype chain, not inferred from I8 bytes.
    pub fn from_comfy_checkpoint(
        architecture: Qwen3VLTextArchitecture,
        layout: Qwen3VLTextLayout,
        source: &ComfyCanonicalWeightSource,
    ) -> Result<Self> {
        ensure!(
            architecture.layers() == 36
                && architecture.hidden_size() == 2560
                && architecture.intermediate_size() == 9728
                && architecture.query_heads() == 32
                && architecture.key_value_heads() == 8
                && architecture.head_dim() == 128
                && architecture.vocabulary_size() == 151936,
            "checkpoint architecture differs from P-H3's Qwen3-VL-4B text encoder"
        );
        let connector =
            PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({
                "_class_name":"PH3ImageTextConnector",
                "hidden_state_indices":PH3_IMAGE_TEXT_HIDDEN_STATES,
                "text_hidden_size":architecture.hidden_size(),
                "out_channels":5120,
            }))?)?;
        Self::from_comfy_checkpoint_with_connector(architecture, layout, source, &connector)
    }
    /// Follow the actual declared five-tap connector, including the encoder's
    /// final normalized hidden state. Every projection's W4/W8 route still
    /// comes from its original source metadata. Physical checkpoint/statistics
    /// validation remains with the existing checked source adapters.
    pub fn from_comfy_checkpoint_with_connector(
        architecture: Qwen3VLTextArchitecture,
        layout: Qwen3VLTextLayout,
        source: &ComfyCanonicalWeightSource,
        connector: &PH3ImageTextConnectorConfig,
    ) -> Result<Self> {
        let indices = connector
            .hidden_state_indices()
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>();
        ensure!(
            connector.text_hidden_size() == architecture.hidden_size(),
            "declared connector input width differs from its text encoder"
        );
        ensure!(
            indices.len() == 5 && indices.last() == Some(&architecture.layers()),
            "declared connector requires five taps ending at the final normalized encoder layer"
        );
        // The existing typed text config rejects zero, out-of-range,
        // duplicate and non-increasing indices before selecting source routes.
        let text =
            Qwen3VLTextConfig::new(architecture, layout)?.with_hidden_state_indices(indices)?;
        Ok(Self::new(
            Qwen3VLTextComfyConfig::from_checkpoint(text, source)?,
            connector.out_channels(),
        )?)
    }
    pub fn encoder(&self) -> &Qwen3VLTextComfyConfig {
        &self.encoder
    }
    pub fn connector_output_dim(&self) -> u64 {
        self.connector_output_dim
    }
    pub fn connector_config(&self) -> Result<PH3ImageTextConnectorConfig> {
        Ok(PH3ImageTextConnectorConfig::from_hf_json(
            &serde_json::to_vec(&serde_json::json!({
                "_class_name":"PH3ImageTextConnector",
                "hidden_state_indices":self.encoder.text().hidden_state_indices(),
                "text_hidden_size":self.encoder.text().architecture().hidden_size(),
                "out_channels":self.connector_output_dim
            }))?,
        )?)
    }
    pub fn projection_config(&self) -> Result<PH3ImageTextConnectorProjectionConfig> {
        Ok(
            PH3ImageTextConnectorProjectionConfig::from_connector(&self.connector_config()?)?
                .with_checkpoint_layout(
                    PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
                ),
        )
    }
    pub fn input_tensor_specs(
        &self,
    ) -> std::result::Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let mut specs = self.encoder.input_tensor_specs()?;
        specs.remove(&ProgramValueId::new(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)?);
        specs.insert(
            ProgramValueId::new(PH3ImageTextConditioningFamily::TOKEN_IDS_ID)?,
            ProgramTensorSpec {
                dimensions: vec![self.encoder.text().layout().rows()],
                element_type: ElementType::U32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        );
        for spec in specs.values() {
            spec.validate("H3 text input")?;
        }
        Ok(specs)
    }
    pub fn compile_options(&self) -> std::result::Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.input_tensor_specs()?)
    }
    /// Connector-stage peaks, in addition to existing static mmap/device
    /// weights and runtime workspace. The Host account enforces actual sums.
    pub fn host_stage_bytes(&self, precision: PH3ImageConditioningPrecision) -> Result<[u64; 3]> {
        let rows = self.encoder.text().layout().rows();
        let taps = rows
            .checked_mul(self.encoder.text().architecture().hidden_size())
            .and_then(|n| n.checked_mul(5 * 4))
            .context("five-tap bytes overflow")?;
        let projected = rows
            .checked_mul(self.connector_output_dim)
            .and_then(|n| n.checked_mul(4))
            .context("projection bytes overflow")?;
        let final_bytes = rows
            .checked_mul(self.connector_output_dim)
            .and_then(|n| {
                n.checked_mul(if precision == PH3ImageConditioningPrecision::F32 {
                    4
                } else {
                    2
                })
            })
            .context("conditioning bytes overflow")?;
        Ok([
            taps.checked_mul(2).context("tap staging peak overflow")?,
            taps.checked_add(projected)
                .context("projection staging peak overflow")?,
            projected
                .checked_add(final_bytes)
                .context("final staging peak overflow")?,
        ])
    }
}
