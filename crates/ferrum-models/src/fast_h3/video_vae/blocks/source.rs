use super::*;
use crate::fast_h3::nvfp4_scalars::managed_scalar;
use std::{ops::Range, sync::Arc};

pub(super) fn unit_component() -> Result<WeightComponentSpec, VNextError> {
    constant("unit_f32", vec![1], ElementType::F32)
}
pub(super) fn gamma_component(width: u64) -> Result<WeightComponentSpec, VNextError> {
    constant(
        &format!("unit_gamma_bf16_{width}"),
        vec![width],
        ElementType::Bf16,
    )
}
fn constant(
    name: &str,
    shape: Vec<u64>,
    dtype: ElementType,
) -> Result<WeightComponentSpec, VNextError> {
    let name = format!("derived.fast_h3.video_decoder.{name}.v1");
    Ok(WeightComponentSpec {
        id: WeightId::new(&name)?,
        role: WeightComponentRole::Values,
        external_names: vec![name],
        dimensions: shape,
        encoding: WeightEncoding::Dense {
            element_type: dtype,
        },
        required: true,
    })
}

/// One original archive owner plus caller-admitted unit/gamma/alpha constants.
/// All selected blocks share one unit scalar and one non-affine RMS gamma.
pub struct FastH3VideoDecoderBlocksSource {
    package: Arc<FastH3VideoDecoderPackage>,
    originals: BTreeMap<WeightId, WeightComponentSpec>,
    derived: BTreeMap<WeightId, (WeightComponentSpec, ManagedHostBytes, String)>,
}
impl FastH3VideoDecoderBlocksSource {
    /// `blocks=0..package.config().decoder_layers()` selects the full decoder.
    /// A smaller explicit range is a block qualification component, not a
    /// decoded-video pipeline or an implicit quality-reduction preset.
    pub fn prepare(
        package: Arc<FastH3VideoDecoderPackage>,
        blocks: Range<usize>,
        sequences: Vec<u64>,
        host: &HostMemoryAccount,
        limits: FastH3VideoDecoderBlocksLimits,
    ) -> anyhow::Result<(FastH3VideoDecoderBlocksConfig, Self)> {
        anyhow::ensure!(
            blocks.start < blocks.end
                && blocks.end <= package.config().decoder_layers()
                && blocks.len() <= limits.maximum_blocks,
            "decoder block selection exceeds declared depth or capacity"
        );
        let c = package.config();
        let epsilon =
            CanonicalRational::new((c.norm_epsilon() * 1e12).round() as i64, 1_000_000_000_000)?;
        anyhow::ensure!(
            (epsilon.numerator() as f64 / epsilon.denominator() as f64) as f32
                == c.norm_epsilon() as f32,
            "decoder epsilon is not represented by declared canonical precision"
        );
        let original = |name: &str| {
            package
                .original_component(name)
                .cloned()
                .ok_or_else(|| invalid(format!("missing original {name}")))
        };
        let mut selected = Vec::with_capacity(blocks.len());
        for index in blocks {
            let prefix = format!("decoder.transformer_blocks.{index}");
            let norms_and_scales = NORMS
                .iter()
                .map(|s| original(&format!("{prefix}.{s}")))
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| invalid("norm inventory"))?;
            let projections = PROJECTIONS
                .iter()
                .map(|s| -> anyhow::Result<Projection> {
                    let prefix = format!("{prefix}.{s}");
                    Ok(Projection {
                        packed: original(&format!("{prefix}.packed_weight"))?,
                        scales: original(&format!("{prefix}.weight_inv_scale"))?,
                        bias: original(&format!("{prefix}.bias"))?,
                        alpha: package.alpha_derivation(&prefix)?,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| invalid("projection inventory"))?;
            selected.push(Block {
                index,
                norms_and_scales,
                projections,
            });
        }
        let config = FastH3VideoDecoderBlocksConfig {
            sequences,
            hidden: c.hidden_size()?.try_into()?,
            ffn: c.ffn_size()?.try_into()?,
            head_dim: c.attention_head_dim().try_into()?,
            rotary_dim: c.rotary_dimension().try_into()?,
            epsilon,
            declared_layers: c.decoder_layers(),
            blocks: selected,
        };
        config.validate(limits)?;
        let mut originals = BTreeMap::new();
        let mut derived = BTreeMap::new();
        for b in &config.blocks {
            for c in &b.norms_and_scales {
                originals.insert(c.id.clone(), c.clone());
            }
            for p in &b.projections {
                for c in [&p.packed, &p.scales, &p.bias] {
                    originals.insert(c.id.clone(), c.clone());
                }
                let c = p.alpha.output_component()?;
                let bytes = p.alpha.managed_backing(package.as_ref(), host)?;
                derived.insert(c.id.clone(), (c, bytes, p.alpha.source_file().into()));
            }
        }
        let unit = unit_component()?;
        derived.insert(
            unit.id.clone(),
            (
                unit,
                managed_scalar(host, 1f32.to_bits())?,
                "derived/constant-unit-f32".into(),
            ),
        );
        let gamma = gamma_component(config.head_dim)?;
        let bytes: usize = gamma.physical_bytes()?.try_into()?;
        let mut backing = host.allocate_zeroed(HostMemoryClass::WeightBacking, bytes)?;
        for pair in backing.as_mut().chunks_exact_mut(2) {
            pair.copy_from_slice(&0x3f80u16.to_le_bytes());
        }
        derived.insert(
            gamma.id.clone(),
            (gamma, backing.freeze(), "derived/constant-unit-bf16".into()),
        );
        Ok((
            config,
            Self {
                package,
                originals,
                derived,
            },
        ))
    }
}
impl WeightComponentSource for FastH3VideoDecoderBlocksSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if let Some(original) = self.originals.get(&c.id) {
            if c != original {
                return Err(invalid("original block component differs"));
            }
            return self.package.component(c);
        }
        let (declared, bytes, file) = self
            .derived
            .get(&c.id)
            .ok_or_else(|| invalid("undeclared block component"))?;
        if c != declared {
            return Err(invalid("derived block component differs"));
        }
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec![file.clone()],
            c.dimensions.clone(),
            c.physical_element_type(),
            bytes.clone(),
        )
    }
}
