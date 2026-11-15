use super::*;
use sha2::{Digest, Sha256};

/// Explicit source arithmetic; Python scalar division in the pinned loader is
/// binary64 followed by the destination F32 tensor construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Projection {
    pub module: String,
    pub rows: u64,
    pub columns: u64,
    pub global_bits: u32,
    pub source_file: String,
}
impl Projection {
    pub fn alpha(&self) -> Result<f32, VNextError> {
        let global = f32::from_bits(self.global_bits);
        let alpha = (1.0_f64 / f64::from(global)) as f32;
        if ![global, alpha]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        {
            return Err(invalid(
                "global SF and derived alpha must be finite positive F32",
            ));
        }
        Ok(alpha)
    }
    pub fn packed(&self) -> Result<WeightComponentSpec, VNextError> {
        component(
            &format!("{}.weight_packed", self.module),
            WeightComponentRole::PackedValues,
            vec![self.rows, self.columns / 2],
            WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4()),
        )
    }
    pub fn scales(&self) -> Result<WeightComponentSpec, VNextError> {
        component(
            &format!("{}.weight_scale", self.module),
            WeightComponentRole::Scales,
            Nvfp4MatrixLayout::new(self.rows, self.columns)?
                .scale_shape()
                .to_vec(),
            WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
        )
    }
    pub fn alpha_component(&self) -> Result<WeightComponentSpec, VNextError> {
        let mut c = component(
            &format!("{}.weight_global_scale", self.module),
            WeightComponentRole::Values,
            vec![1],
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
        )?;
        // Bind the derived source to the exact scalar bits, provenance and
        // arithmetic recipe carried by the prepared family configuration.
        let mut hash = Sha256::new();
        hash.update(SCALAR_PROFILE.as_bytes());
        hash.update([0]);
        hash.update(serde_json::to_vec(self).map_err(|e| invalid(e.to_string()))?);
        c.id = WeightId::new(format!(
            "derived.qwen3_vl.nvfp4.{}.alpha.{:x}",
            self.module,
            hash.finalize()
        ))?;
        Ok(c)
    }
}

pub(super) fn component(
    name: &str,
    role: WeightComponentRole,
    dimensions: Vec<u64>,
    encoding: WeightEncoding,
) -> Result<WeightComponentSpec, VNextError> {
    Ok(WeightComponentSpec {
        id: WeightId::new(format!("component.qwen3_vl.nvfp4.{name}"))?,
        external_names: vec![name.into()],
        role,
        dimensions,
        encoding,
        required: true,
    })
}
pub(super) fn unit_component() -> Result<WeightComponentSpec, VNextError> {
    component(
        UNIT_NAME,
        WeightComponentRole::Values,
        vec![1],
        WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
    )
}
pub(super) fn tensor(
    c: &WeightComponentSpec,
    physical: bool,
) -> Result<WeightTensorSpec, VNextError> {
    Ok(WeightTensorSpec {
        id: WeightId::new(format!("tensor.{}", c.id))?,
        dimensions: c.dimensions.clone(),
        logical_element_type: c.physical_element_type(),
        required: true,
        physical_layout: if physical {
            PhysicalWeightLayout::PhysicalComponent(PhysicalWeightComponentView::new(c.id.clone()))
        } else {
            PhysicalWeightLayout::Dense {
                component_id: c.id.clone(),
            }
        },
    })
}
pub(super) fn value(c: &WeightComponentSpec) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.{}", c.id))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3VLTextNvfp4Config {
    pub(super) text: Qwen3VLTextConfig,
    pub(super) declared_layers: u64,
    pub(super) exported_layers: u64,
    pub(super) scalar_profile: String,
    pub(super) projections: Vec<Projection>,
    #[serde(
        default,
        skip_serializing_if = "Qwen3VLTextNvfp4Attention::is_sequential"
    )]
    pub(super) attention: Qwen3VLTextNvfp4Attention,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3VLTextNvfp4Attention {
    #[default]
    Sequential,
    VendorFusedCausal,
}
impl Qwen3VLTextNvfp4Attention {
    fn is_sequential(&self) -> bool {
        matches!(self, Self::Sequential)
    }
}
impl Qwen3VLTextNvfp4Config {
    pub fn attention(&self) -> Qwen3VLTextNvfp4Attention {
        self.attention
    }
    pub fn numerical_profile_id(&self) -> &'static str {
        match self.attention {
            Qwen3VLTextNvfp4Attention::Sequential => Qwen3VLTextNvfp4Family::CUDA_PROFILE_ID,
            Qwen3VLTextNvfp4Attention::VendorFusedCausal => {
                Qwen3VLTextNvfp4Family::CUDA_FUSED_PROFILE_ID
            }
        }
    }
    pub fn text(&self) -> &Qwen3VLTextConfig {
        &self.text
    }
    pub fn declared_layers(&self) -> u64 {
        self.declared_layers
    }
    pub fn exported_layers(&self) -> u64 {
        self.exported_layers
    }
    pub fn executed_layers(&self) -> u64 {
        self.text.architecture().layers()
    }
    pub fn projection_count(&self) -> usize {
        self.projections.len()
    }
    pub fn scalar_arithmetic_profile(&self) -> &str {
        &self.scalar_profile
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        self.text.input_tensor_specs()
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        Ok(super::program::build_with_options(
            self,
            &ModelFamilyId::new(Qwen3VLTextNvfp4Family::FAMILY_ID)?,
        )?
        .1)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        self.text.validate()?;
        if self.attention == Qwen3VLTextNvfp4Attention::VendorFusedCausal
            && (self.text.architecture().head_dim() != 128
                || self.text.layout().batch() > 16
                || self
                    .text
                    .layout()
                    .key_runs()
                    .iter()
                    .any(|runs| runs.len() != 1 || !runs[0].is_valid()))
        {
            return Err(invalid(
                "fused causal GQA requires D128, B1..16 and all-valid unpadded sequences",
            ));
        }
        if self.executed_layers() > self.exported_layers
            || self.exported_layers > self.declared_layers
            || self.declared_layers > 65536 / 11
            || self.scalar_profile != SCALAR_PROFILE
            || !self.text.hidden_state_indices().is_empty()
        {
            return Err(invalid("requires explicit bounded decoder prefix, unit-SF profile, and pre-final-norm text output"));
        }
        let expected: Vec<_> = self
            .text
            .expected_weights()?
            .into_iter()
            .filter(|w| w.dimensions().len() == 2)
            .collect();
        if expected.len() != self.projections.len() {
            return Err(invalid(
                "projection inventory differs from selected decoder prefix",
            ));
        }
        for (w, p) in expected.iter().zip(&self.projections) {
            if w.external_name().strip_suffix(".weight") != Some(p.module.as_str())
                || w.dimensions() != [p.rows, p.columns]
                || p.rows % 128 != 0
                || p.columns % 64 != 0
                || p.source_file.is_empty()
                || p.source_file.starts_with('/')
                || p.source_file.contains('\\')
                || p.source_file
                    .split('/')
                    .any(|part| matches!(part, "" | "." | ".."))
            {
                return Err(invalid("projection geometry, source identity or tile layout differs from serialized encoder"));
            }
            Nvfp4MatrixLayout::new(p.rows, p.columns)?;
            p.alpha()?;
        }
        Ok(())
    }
    pub(super) fn schema(&self) -> Result<WeightSchema, VNextError> {
        let family = ModelFamilyId::new(Qwen3VLTextNvfp4Family::FAMILY_ID)?;
        let mut schema = super::super::weights::schema(&self.text, &family)?;
        // Preserve the existing BF16 norm schema. Packed projections acquire
        // explicit physical-component views, never numerical Dense-U8 aliases.
        schema.tensors.retain(|t| t.dimensions.len() == 1);
        schema.components.retain(|c| c.dimensions.len() == 1);
        for p in &self.projections {
            for (c, physical) in [
                (p.packed()?, true),
                (p.scales()?, true),
                (p.alpha_component()?, false),
            ] {
                schema.tensors.push(tensor(&c, physical)?);
                schema.components.push(c);
            }
        }
        let unit = unit_component()?;
        schema.tensors.push(tensor(&unit, false)?);
        schema.components.push(unit);
        schema.layout_id = WeightLayoutId::new("weight-layout.qwen3_vl.text.nvfp4_unit.v1")?;
        schema.validate(&family)?;
        Ok(schema)
    }
}
