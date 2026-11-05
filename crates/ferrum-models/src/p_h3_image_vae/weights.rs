use super::{invalid, PH3ImageVaeDecoderConfig};
use ferrum_runtime::{ElementType, ResolvedTensorLayout, ResolvedTensorSpec, VNextError};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageVaeWeightEncoding {
    /// Source BF16 remains source BF16; execution casts require a separate profile.
    Bf16,
    /// Signed I8 matrix, F32 per-output scale and explicit `.comfy_quant` metadata.
    /// Metadata must be parsed before decoding; the filename grants no format proof.
    ConvRotInt8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
/// Source-oriented mapping, without reordering or numerical conversion.
/// Comfy's `attn.to_qkv` rows group Q/K/V *within each head*. They cannot be
/// split into three contiguous hidden-width matrices without a permutation.
pub struct PH3ImageVaeDecoderWeight {
    external_name: String,
    dimensions: Vec<u64>,
    encoding: PH3ImageVaeWeightEncoding,
}
impl PH3ImageVaeDecoderWeight {
    pub fn external_name(&self) -> &str {
        &self.external_name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub const fn encoding(&self) -> PH3ImageVaeWeightEncoding {
        self.encoding
    }
    pub const fn source_element_type(&self) -> ElementType {
        match self.encoding {
            PH3ImageVaeWeightEncoding::Bf16 => ElementType::Bf16,
            PH3ImageVaeWeightEncoding::ConvRotInt8 => ElementType::I8,
        }
    }
    pub fn scale_external_name(&self) -> Option<String> {
        (self.encoding == PH3ImageVaeWeightEncoding::ConvRotInt8)
            .then(|| format!("{}_scale", self.external_name))
    }
    pub fn quantization_external_name(&self) -> Option<String> {
        (self.encoding == PH3ImageVaeWeightEncoding::ConvRotInt8).then(|| {
            format!(
                "{}.comfy_quant",
                self.external_name
                    .strip_suffix(".weight")
                    .expect("checked matrix weight suffix")
            )
        })
    }
    /// Signed matrix bytes only; scales and variable-length JSON are separate source components.
    pub fn value_bytes(&self) -> Result<u64, VNextError> {
        Ok(ResolvedTensorSpec::new(
            self.dimensions.clone(),
            self.source_element_type(),
            ResolvedTensorLayout::Contiguous,
        )?
        .minimum_storage_bytes()?)
    }
    pub fn scale_tensor(&self) -> Result<Option<ResolvedTensorSpec>, VNextError> {
        if self.encoding == PH3ImageVaeWeightEncoding::ConvRotInt8 {
            Ok(Some(ResolvedTensorSpec::new(
                vec![self.dimensions[0], 1],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?))
        } else {
            Ok(None)
        }
    }
}

pub(super) fn expected(
    config: &PH3ImageVaeDecoderConfig,
) -> Result<Vec<PH3ImageVaeDecoderWeight>, VNextError> {
    config.validate()?;
    let h = config.hidden_size()?;
    let m = config.intermediate_size()?;
    let patch = config.patch_output_channels()?;
    let mut output = Vec::new();
    let mut push = |name: String, dimensions: Vec<u64>, encoding| -> Result<(), VNextError> {
        let w = PH3ImageVaeDecoderWeight {
            external_name: name,
            dimensions,
            encoding,
        };
        if encoding == PH3ImageVaeWeightEncoding::ConvRotInt8
            && (w.dimensions.len() != 2 || !w.external_name.ends_with(".weight"))
        {
            return Err(invalid(
                "weight",
                "ConvRot INT8 mapping requires a named 2D matrix weight",
            ));
        }
        w.value_bytes()?;
        output.push(w);
        Ok(())
    };
    use PH3ImageVaeWeightEncoding::{Bf16, ConvRotInt8};
    for (name, dimensions) in [
        (
            "post_quant_conv.weight",
            vec![config.latent_channels(), config.latent_channels(), 1, 1, 1],
        ),
        ("post_quant_conv.bias", vec![config.latent_channels()]),
        (
            "decoder.x_embedder.weight",
            vec![h, config.latent_channels()],
        ),
        ("decoder.x_embedder.bias", vec![h]),
        (
            "decoder.register_tokens",
            vec![1, config.register_tokens(), h],
        ),
        ("decoder.norm_out.weight", vec![h]),
        ("decoder.norm_out.bias", vec![h]),
        ("decoder.proj_out.weight", vec![patch, h]),
        ("decoder.proj_out.bias", vec![patch]),
    ] {
        push(name.into(), dimensions, Bf16)?;
    }
    for layer in 0..config.layers() {
        let prefix = format!("decoder.transformer_blocks.{layer}");
        for (suffix, shape) in [
            ("norm1.weight", vec![h]),
            ("norm2.weight", vec![h]),
            ("scale1", vec![h]),
            ("scale2", vec![h]),
        ] {
            push(format!("{prefix}.{suffix}"), shape, Bf16)?;
        }
        for (name, rows, columns) in [
            ("attn.to_qkv", 3 * h, h),
            ("attn.to_out", h, h),
            ("ff.w1", 2 * m, h),
            ("ff.w2", h, m),
        ] {
            push(
                format!("{prefix}.{name}.weight"),
                vec![rows, columns],
                ConvRotInt8,
            )?;
            push(format!("{prefix}.{name}.bias"), vec![rows], Bf16)?;
        }
    }
    Ok(output)
}
