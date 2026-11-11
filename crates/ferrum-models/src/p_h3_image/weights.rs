//! Dense Diffusers MiniMax-H3 transformer checkpoint structure. These are
//! real tensor names, shapes and precision boundaries, not loaded tensors or
//! an execution family. Comfy quantized storage requires its own adapter.

use anyhow::{ensure, Context, Result};
use ferrum_runtime::ElementType;

use super::{layout::checked_elements, PH3ImageCurveTimeEmbedding, PH3ImageTransformerConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageTransformerCheckpointLayout {
    DiffusersDense,
    ComfyCurve,
}

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageWeightLimits {
    pub max_tensors: usize,
    pub max_payload_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PH3ImageTransformerWeightSpec {
    name: String,
    dimensions: Vec<u64>,
    element_type: ElementType,
    payload_bytes: u64,
}

impl PH3ImageTransformerWeightSpec {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub fn validate_header(&self, dimensions: &[u64], element_type: ElementType) -> Result<()> {
        ensure!(
            dimensions == self.dimensions && element_type == self.element_type,
            "P-H3 checkpoint tensor {} has incompatible shape or precision",
            self.name
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PH3ImageTransformerWeights {
    tensors: Vec<PH3ImageTransformerWeightSpec>,
    payload_bytes: u64,
    layout: PH3ImageTransformerCheckpointLayout,
}

impl PH3ImageTransformerWeights {
    pub fn from_config(
        config: &PH3ImageTransformerConfig,
        limits: PH3ImageWeightLimits,
    ) -> Result<Self> {
        let count = config
            .layers()
            .checked_mul(12)
            .and_then(|count| {
                config
                    .refiner_layers()
                    .checked_mul(10)
                    .and_then(|refiners| count.checked_add(refiners))
            })
            .and_then(|count| count.checked_add(18))
            .context("P-H3 checkpoint tensor count overflows")?;
        let count = usize::try_from(count)?;
        ensure!(
            count <= limits.max_tensors,
            "P-H3 checkpoint exceeds declared tensor capacity"
        );
        checked_elements::<PH3ImageTransformerWeightSpec>(&[count])?;
        let mut inventory = Self {
            tensors: Vec::with_capacity(count),
            payload_bytes: 0,
            layout: PH3ImageTransformerCheckpointLayout::DiffusersDense,
        };
        let h = config.hidden_size();
        let time = config.time_embed_dim();
        let patch = config.patch_input_dim();
        let twice_h = h
            .checked_mul(2)
            .context("P-H3 final modulation width overflows")?;
        // Input patch/audio projections, time MLP and both output heads retain
        // the checkpoint's FP32 storage. The other matrices/norms are BF16.
        for (name, input, output, dtype) in [
            ("proj_in", patch, h, ElementType::F32),
            (
                "audio_proj_in",
                config.audio_in_channels(),
                h,
                ElementType::F32,
            ),
            ("context_embedder", config.text_dim(), h, ElementType::Bf16),
            (
                "time_embedder.linear_1",
                config.freq_dim(),
                config.time_embed_hidden_dim(),
                ElementType::F32,
            ),
            (
                "time_embedder.linear_2",
                config.time_embed_hidden_dim(),
                time,
                ElementType::F32,
            ),
            ("norm_out.linear", time, twice_h, ElementType::Bf16),
            ("proj_out", h, patch, ElementType::F32),
            (
                "audio_proj_out",
                h,
                config.audio_in_channels(),
                ElementType::F32,
            ),
        ] {
            inventory.push(format!("{name}.weight"), &[output, input], dtype, limits)?;
            inventory.push(format!("{name}.bias"), &[output], dtype, limits)?;
        }
        inventory.push(
            "norm_out.norm.weight".into(),
            &[h],
            ElementType::Bf16,
            limits,
        )?;
        inventory.push(
            "token_refiner.final_norm.weight".into(),
            &[h],
            ElementType::Bf16,
            limits,
        )?;
        for layer in 0..config.refiner_layers() {
            inventory.block(
                &format!("token_refiner.refiner_blocks.{layer}"),
                config,
                false,
                limits,
            )?;
        }
        for layer in 0..config.layers() {
            inventory.block(&format!("transformer_blocks.{layer}"), config, true, limits)?;
        }
        Ok(inventory)
    }

    /// Actual curve-form Comfy BF16 base structure. A Comfy quantized matrix
    /// adapter expands selected BF16 logical weights into separate physical
    /// components. `validate_header` remains strict for dense tensors and must
    /// not silently relabel I8/codebook storage as BF16.
    pub fn from_comfy_curve_config(
        config: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        limits: PH3ImageWeightLimits,
    ) -> Result<Self> {
        let count = config
            .layers()
            .checked_mul(10)
            .and_then(|count| {
                config
                    .refiner_layers()
                    .checked_mul(8)
                    .and_then(|refs| count.checked_add(refs))
            })
            .and_then(|count| count.checked_add(16))
            .context("Comfy curve tensor count overflows")?;
        let count = usize::try_from(count)?;
        ensure!(
            count <= limits.max_tensors,
            "Comfy curve checkpoint exceeds tensor capacity"
        );
        checked_elements::<PH3ImageTransformerWeightSpec>(&[count])?;
        let mut inventory = Self {
            tensors: Vec::with_capacity(count),
            payload_bytes: 0,
            layout: PH3ImageTransformerCheckpointLayout::ComfyCurve,
        };
        let h = config.hidden_size();
        let basis = u64::try_from(curve.basis_dim())?;
        let twice_h = h
            .checked_mul(2)
            .context("Comfy final modulation width overflows")?;
        for (name, input, output, dtype) in [
            (
                "video_patch_proj",
                config.patch_input_dim(),
                h,
                ElementType::F32,
            ),
            (
                "audio_patch_proj",
                config.audio_in_channels(),
                h,
                ElementType::F32,
            ),
            ("condition_proj", config.text_dim(), h, ElementType::Bf16),
            (
                "final_layer.adaln_proj.linear",
                basis,
                twice_h,
                ElementType::F32,
            ),
            (
                "final_layer.video_out",
                h,
                config.patch_input_dim(),
                ElementType::F32,
            ),
            (
                "final_layer.audio_out",
                h,
                config.audio_in_channels(),
                ElementType::F32,
            ),
        ] {
            inventory.push(format!("{name}.weight"), &[output, input], dtype, limits)?;
            inventory.push(format!("{name}.bias"), &[output], dtype, limits)?;
        }
        inventory.push(
            "adaln_t_table".into(),
            &[u64::try_from(curve.grid_points())?, basis],
            ElementType::F32,
            limits,
        )?;
        inventory.push(
            "rope.inv_freq".into(),
            &[config.rope_freq_dim()],
            ElementType::F32,
            limits,
        )?;
        for name in ["final_layer.norm.weight", "token_refiner.final_norm.weight"] {
            inventory.push(name.into(), &[h], ElementType::Bf16, limits)?;
        }
        for layer in 0..config.refiner_layers() {
            inventory.comfy_block(
                &format!("token_refiner.blocks.{layer}"),
                config,
                basis,
                false,
                limits,
            )?;
        }
        for layer in 0..config.layers() {
            inventory.comfy_block(&format!("blocks.{layer}"), config, basis, true, limits)?;
        }
        Ok(inventory)
    }

    fn comfy_block(
        &mut self,
        prefix: &str,
        config: &PH3ImageTransformerConfig,
        basis: u64,
        modulated: bool,
        limits: PH3ImageWeightLimits,
    ) -> Result<()> {
        let h = config.hidden_size();
        let inner = config.attention_inner_dim();
        let qkv = inner
            .checked_mul(3)
            .context("Comfy fused QKV width overflows")?;
        let twice_ffn = config
            .ffn_dim()
            .checked_mul(2)
            .context("Comfy SwiGLU width overflows")?;
        for (name, dimensions) in [
            ("norm1.weight", vec![h]),
            ("norm2.weight", vec![h]),
            ("attn.qkv_proj.weight", vec![qkv, h]),
            ("attn.out_proj.weight", vec![h, inner]),
            ("attn.q_norm.weight", vec![config.attention_head_dim()]),
            ("attn.k_norm.weight", vec![config.attention_head_dim()]),
            ("mlp.fc1.weight", vec![twice_ffn, h]),
            ("mlp.fc2.weight", vec![h, config.ffn_dim()]),
        ] {
            self.push(
                format!("{prefix}.{name}"),
                &dimensions,
                ElementType::Bf16,
                limits,
            )?;
        }
        if modulated {
            let output = h
                .checked_mul(18)
                .context("Comfy modality table width overflows")?;
            self.push(
                format!("{prefix}.adaln_proj.linear.weight"),
                &[output, basis],
                ElementType::F32,
                limits,
            )?;
            self.push(
                format!("{prefix}.adaln_proj.linear.bias"),
                &[output],
                ElementType::F32,
                limits,
            )?;
        }
        Ok(())
    }

    fn block(
        &mut self,
        prefix: &str,
        config: &PH3ImageTransformerConfig,
        modulated: bool,
        limits: PH3ImageWeightLimits,
    ) -> Result<()> {
        let h = config.hidden_size();
        let inner = config.attention_inner_dim();
        let ffn = config.ffn_dim();
        let twice_ffn = ffn.checked_mul(2).context("P-H3 SwiGLU width overflows")?;
        for (name, dimensions) in [
            ("norm1.weight", vec![h]),
            ("norm2.weight", vec![h]),
            ("attn.to_q.weight", vec![inner, h]),
            ("attn.to_k.weight", vec![inner, h]),
            ("attn.to_v.weight", vec![inner, h]),
            ("attn.to_out.0.weight", vec![h, inner]),
            ("attn.norm_q.weight", vec![config.attention_head_dim()]),
            ("attn.norm_k.weight", vec![config.attention_head_dim()]),
            ("ff.net.0.proj.weight", vec![twice_ffn, h]),
            ("ff.net.2.weight", vec![h, ffn]),
        ] {
            self.push(
                format!("{prefix}.{name}"),
                &dimensions,
                ElementType::Bf16,
                limits,
            )?;
        }
        if modulated {
            let width = h
                .checked_mul(18)
                .context("P-H3 AdaLN table width overflows")?;
            self.push(
                format!("{prefix}.adaln_proj.linear.weight"),
                &[width, config.time_embed_dim()],
                ElementType::Bf16,
                limits,
            )?;
            self.push(
                format!("{prefix}.adaln_proj.linear.bias"),
                &[width],
                ElementType::Bf16,
                limits,
            )?;
        }
        Ok(())
    }

    fn push(
        &mut self,
        name: String,
        dimensions: &[u64],
        dtype: ElementType,
        limits: PH3ImageWeightLimits,
    ) -> Result<()> {
        let elements = dimensions.iter().try_fold(1_u64, |product, dimension| {
            product
                .checked_mul(*dimension)
                .context("P-H3 checkpoint tensor extent overflows")
        })?;
        let bytes_per_element = match dtype {
            ElementType::F32 => 4,
            ElementType::Bf16 => 2,
            _ => unreachable!("dense P-H3 inventory has only F32/BF16 weights"),
        };
        let payload_bytes = elements
            .checked_mul(bytes_per_element)
            .context("P-H3 checkpoint tensor bytes overflow")?;
        self.payload_bytes = self
            .payload_bytes
            .checked_add(payload_bytes)
            .context("P-H3 checkpoint payload overflows")?;
        ensure!(
            self.payload_bytes <= limits.max_payload_bytes,
            "P-H3 checkpoint exceeds declared payload capacity"
        );
        self.tensors.push(PH3ImageTransformerWeightSpec {
            name,
            dimensions: dimensions.to_vec(),
            element_type: dtype,
            payload_bytes,
        });
        Ok(())
    }

    pub fn tensors(&self) -> &[PH3ImageTransformerWeightSpec] {
        &self.tensors
    }
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub fn checkpoint_layout(&self) -> PH3ImageTransformerCheckpointLayout {
        self.layout
    }
    pub fn tensor(&self, name: &str) -> Option<&PH3ImageTransformerWeightSpec> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

#[cfg(test)]
mod tests;
