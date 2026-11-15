use super::FastH3VideoVaeConfig;
use anyhow::{ensure, Context, Result};
use ferrum_quantization::{
    nvfp4::{
        Nvfp4Geometry, Nvfp4MatrixNames, Nvfp4MatrixSpec, Nvfp4SafetensorsSource, Nvfp4SourceLimits,
    },
    SafetensorsArchive,
};
use ferrum_runtime::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const FORMAT: &str = "fastvideo_h3_decoder_nvfp4_packed_v1";

/// Import bounds, not a Host/device reservation. The caller supplies an archive
/// whose mmap and metadata were already admitted by its archive owner.
#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoDecoderWeightLimits {
    pub maximum_tensors: usize,
    pub maximum_source_bytes: usize,
    pub maximum_metadata_bytes: usize,
}

/// Complete original decoder export: six NVFP4 projections in every declared
/// block, their BF16 biases, and the original F32 outer/norm/scale tensors.
/// No student-depth cut or fallback to a retained master matrix is permitted.
pub struct FastH3VideoDecoderPackage {
    config: FastH3VideoVaeConfig,
    source: Nvfp4SafetensorsSource,
    matrices: BTreeMap<String, Nvfp4MatrixSpec>,
    components: BTreeMap<WeightId, (bool, WeightComponentSpec)>,
    names: BTreeMap<String, WeightId>,
    source_bytes: usize,
}

#[derive(Deserialize)]
struct Metadata {
    block_size: usize,
    rotation_group: serde_json::Value,
    skip_blocks: Vec<usize>,
    act_scale: String,
    num_linears: usize,
    student_layers: usize,
}

impl FastH3VideoDecoderPackage {
    pub(super) fn archive(&self) -> &SafetensorsArchive {
        self.source.archive()
    }
    pub fn new(
        config: &FastH3VideoVaeConfig,
        archive: SafetensorsArchive,
        limits: FastH3VideoDecoderWeightLimits,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            limits.maximum_tensors > 0
                && limits.maximum_source_bytes > 0
                && limits.maximum_metadata_bytes > 0,
            "video decoder bounds must be positive"
        );
        let expected_count = config
            .decoder_layers()
            .checked_mul(34)
            .and_then(|n| n.checked_add(9))
            .context("decoder component count overflow")?;
        ensure!(
            expected_count <= limits.maximum_tensors && archive.tensor_count() == expected_count,
            "video decoder component inventory differs or exceeds capacity"
        );
        let metadata = archive.header_metadata()?;
        ensure!(
            metadata.len() == 1,
            "packed decoder must be one explicitly selected export"
        );
        let header = metadata.values().next().context("missing decoder header")?;
        let bytes = header
            .iter()
            .try_fold(0usize, |n, (k, v)| {
                n.checked_add(k.len()).and_then(|n| n.checked_add(v.len()))
            })
            .context("decoder metadata size overflow")?;
        ensure!(
            bytes <= limits.maximum_metadata_bytes,
            "decoder metadata exceeds capacity"
        );
        ensure!(
            header.get("format").map(String::as_str) == Some(FORMAT),
            "unsupported packed video decoder format"
        );
        let m: Metadata = serde_json::from_str(
            header
                .get("nvfp4_metadata")
                .context("missing packed decoder metadata")?,
        )?;
        ensure!(
            m.block_size == 16
                && m.rotation_group.is_null()
                && m.skip_blocks.is_empty()
                && m.act_scale == "unit",
            "unsupported decoder quantization/activation recipe"
        );
        ensure!(
            m.student_layers == config.decoder_layers()
                && m.num_linears
                    == config
                        .decoder_layers()
                        .checked_mul(6)
                        .context("matrix count overflow")?,
            "decoder metadata must cover every declared layer; depth cuts are not allowed"
        );

        let mut source_bytes = 0usize;
        for name in archive.tensor_names() {
            source_bytes = source_bytes
                .checked_add(archive.tensor(name)?.bytes().len())
                .filter(|n| *n <= limits.maximum_source_bytes)
                .context("video decoder source bytes exceed capacity")?;
        }
        let (entries, matrices) = inventory(config)?;
        let mut components = BTreeMap::new();
        let mut names = BTreeMap::new();
        // Validate all physical shapes/types before scanning any payload.
        for (name, shape, dtype, role, packed) in entries {
            let tensor = archive.tensor(&name)?;
            ensure!(
                tensor
                    .shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(|x| *x as u64))
                    && tensor.element_type() == Some(dtype),
                "decoder physical declaration differs for {name}"
            );
            let encoding = if role == WeightComponentRole::PackedValues {
                WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
            } else {
                WeightEncoding::Dense {
                    element_type: dtype,
                }
            };
            let c = WeightComponentSpec {
                id: WeightId::new(format!("fast_h3.video_vae.{name}"))?,
                role,
                external_names: vec![name.clone()],
                dimensions: tensor.shape().to_vec(),
                encoding,
                required: true,
            };
            ensure!(
                names.insert(name, c.id.clone()).is_none()
                    && components.insert(c.id.clone(), (packed, c)).is_none(),
                "duplicate decoder component"
            );
        }
        let actual: BTreeSet<_> = archive.tensor_names().collect();
        ensure!(
            actual == names.keys().map(String::as_str).collect(),
            "unknown decoder components"
        );
        for (_, c) in components.values() {
            let t = archive.tensor(&c.external_names[0])?;
            match t.element_type() {
                Some(ElementType::F32) => {
                    ensure!(
                        t.bytes()
                            .chunks_exact(4)
                            .all(|v| f32::from_le_bytes(v.try_into().expect("four bytes"))
                                .is_finite()),
                        "decoder F32 parameter must be finite: {}",
                        c.external_names[0]
                    );
                    if c.external_names[0].ends_with(".input_amax") {
                        ensure!(
                            f32::from_le_bytes(t.bytes().try_into()?) >= 0.,
                            "unused unit-profile input_amax must be nonnegative"
                        );
                    }
                }
                Some(ElementType::Bf16) => ensure!(
                    t.bytes()
                        .chunks_exact(2)
                        .all(|v| (u16::from_le_bytes([v[0], v[1]]) & 0x7f80) != 0x7f80),
                    "decoder BF16 bias must be finite"
                ),
                _ => {}
            }
        }
        let source = Nvfp4SafetensorsSource::new(
            archive,
            matrices.values().cloned().collect(),
            Nvfp4SourceLimits {
                maximum_matrices: limits.maximum_tensors,
                maximum_source_bytes: limits.maximum_source_bytes,
            },
        )?;
        // Do not expose the generic source's F64 reciprocal as this model's
        // invocation alpha. The actual decoder uses an explicit F32 recipe.
        Ok(Self {
            config: config.clone(),
            source,
            matrices,
            components,
            names,
            source_bytes,
        })
    }

    pub fn config(&self) -> &FastH3VideoVaeConfig {
        &self.config
    }
    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }
    pub fn matrices(&self) -> impl ExactSizeIterator<Item = (&str, &Nvfp4MatrixSpec)> {
        self.matrices.iter().map(|(n, m)| (n.as_str(), m))
    }
    pub fn matrix(&self, canonical_prefix: &str) -> Option<&Nvfp4MatrixSpec> {
        self.matrices.get(canonical_prefix)
    }
    pub fn components(&self) -> impl ExactSizeIterator<Item = &WeightComponentSpec> {
        self.components.values().map(|(_, c)| c)
    }
    pub fn original_component(&self, original_name: &str) -> Option<&WeightComponentSpec> {
        self.names
            .get(original_name)
            .and_then(|id| self.components.get(id))
            .map(|(_, c)| c)
    }
    pub fn alpha_derivation(
        &self,
        canonical_prefix: &str,
    ) -> Result<super::FastH3DecoderAlphaDerivation> {
        let matrix = self
            .matrix(canonical_prefix)
            .context("undeclared decoder projection")?;
        let component = self
            .original_component(&matrix.names.global_scale)
            .context("missing original decoder global scale")?
            .clone();
        Ok(super::FastH3DecoderAlphaDerivation::from_source(
            self, component,
        )?)
    }
    /// Exact physical B/scale_B views. Invocation alpha is intentionally a
    /// separate, source-bound F32 reciprocal derivation.
    pub fn nvfp4_weight_operands(
        &self,
        canonical_prefix: &str,
        packed_weight_id: WeightId,
        scale_weight_id: WeightId,
    ) -> Result<crate::fast_h3::FastH3Nvfp4WeightOperands> {
        let matrix = self
            .matrix(canonical_prefix)
            .context("undeclared decoder projection")?;
        let parts = [&matrix.names.packed, &matrix.names.scales]
            .into_iter()
            .map(|n| {
                self.original_component(n)
                    .cloned()
                    .context("missing original decoder component")
            })
            .collect::<Result<Vec<_>>>()?;
        crate::fast_h3::FastH3Nvfp4WeightOperands::from_components(
            matrix,
            &parts,
            packed_weight_id,
            scale_weight_id,
        )
    }
}

impl WeightComponentSource for FastH3VideoDecoderPackage {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let (packed, declared) = self
            .components
            .get(&c.id)
            .ok_or_else(|| invalid("undeclared component"))?;
        if c != declared {
            return Err(invalid("component differs from original declaration"));
        }
        if *packed {
            self.source.component(c)
        } else {
            self.source.archive().component(c)
        }
    }
}

type Entry = (String, Vec<usize>, ElementType, WeightComponentRole, bool);
fn inventory(c: &FastH3VideoVaeConfig) -> Result<(Vec<Entry>, BTreeMap<String, Nvfp4MatrixSpec>)> {
    let h = c.hidden_size()?;
    let f = c.ffn_size()?;
    let f2 = f.checked_mul(2).context("FFN packed width overflow")?;
    let mut out = Vec::new();
    let mut matrices = BTreeMap::new();
    for (name, shape) in [
        (
            "post_quant_conv.weight",
            vec![c.latent_channels(), c.latent_channels(), 1, 1, 1],
        ),
        ("post_quant_conv.bias", vec![c.latent_channels()]),
        ("decoder.proj_in.weight", vec![h, c.latent_channels()]),
        ("decoder.proj_in.bias", vec![h]),
        ("decoder.register_tokens", vec![1, c.register_tokens(), h]),
        ("decoder.norm_out.weight", vec![h]),
        ("decoder.norm_out.bias", vec![h]),
        ("decoder.proj_out.weight", vec![c.output_patch_width()?, h]),
        ("decoder.proj_out.bias", vec![c.output_patch_width()?]),
    ] {
        out.push((
            name.into(),
            shape,
            ElementType::F32,
            WeightComponentRole::Values,
            false,
        ));
    }
    for i in 0..c.decoder_layers() {
        let prefix = format!("decoder.transformer_blocks.{i}");
        for name in ["norm1.weight", "norm2.weight", "scale1", "scale2"] {
            out.push((
                format!("{prefix}.{name}"),
                vec![h],
                ElementType::F32,
                WeightComponentRole::Values,
                false,
            ));
        }
        for (suffix, rows, cols) in [
            ("attn.to_q", h, h),
            ("attn.to_k", h, h),
            ("attn.to_v", h, h),
            ("attn.to_out.0", h, h),
            ("ff.net.0.proj", f2, h),
            ("ff.net.2", h, f),
        ] {
            let prefix = format!("{prefix}.{suffix}");
            let spec = Nvfp4MatrixSpec {
                names: Nvfp4MatrixNames::packed_decoder(&prefix),
                geometry: Nvfp4Geometry::new(rows, cols)?,
            };
            out.push((
                spec.names.packed.clone(),
                spec.geometry.packed_shape().to_vec(),
                ElementType::U8,
                WeightComponentRole::PackedValues,
                true,
            ));
            out.push((
                spec.names.scales.clone(),
                spec.geometry.scale_shape().to_vec(),
                ElementType::U8,
                WeightComponentRole::Scales,
                true,
            ));
            out.push((
                spec.names.global_scale.clone(),
                vec![],
                ElementType::F32,
                WeightComponentRole::Metadata,
                true,
            ));
            out.push((
                format!("{prefix}.input_amax"),
                vec![],
                ElementType::F32,
                WeightComponentRole::Metadata,
                false,
            ));
            out.push((
                format!("{prefix}.bias"),
                vec![rows],
                ElementType::Bf16,
                WeightComponentRole::Values,
                false,
            ));
            matrices.insert(prefix, spec);
        }
    }
    for (_, shape, _, _, _) in &out {
        shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= isize::MAX as usize)
            .context("decoder weight dimensions exceed address space")?;
    }
    Ok((out, matrices))
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 video decoder: {reason}"),
    }
}
