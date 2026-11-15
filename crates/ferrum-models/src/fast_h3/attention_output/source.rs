use super::*;
use crate::fast_h3::nvfp4_scalars;
use std::sync::Arc;

pub(super) fn unit_component() -> Result<WeightComponentSpec, VNextError> {
    let name = "derived.fast_h3.attention_output.unit_f32.v1";
    Ok(WeightComponentSpec {
        id: id(name)?,
        role: WeightComponentRole::Values,
        external_names: vec![name.into()],
        dimensions: vec![1],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    })
}
pub struct FastH3AttentionOutputSource {
    package: Arc<FastH3TransformerPackage>,
    originals: [WeightComponentSpec; 2],
    scalars: [(WeightComponentSpec, ManagedHostBytes, String); 2],
}
impl FastH3AttentionOutputSource {
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        rows: u64,
        host: &HostMemoryAccount,
        limits: FastH3AttentionOutputLimits,
    ) -> anyhow::Result<(FastH3AttentionOutputConfig, Self)> {
        anyhow::ensure!(
            block < package.config().layers(),
            "attention output block outside checkpoint"
        );
        let name = format!("transformer_blocks.{block}.attn.to_out.weight");
        let w = package
            .weight(&name)
            .ok_or_else(|| invalid("missing original to_out projection"))?;
        let FastH3WeightStorage::Nvfp4 { matrix, components } = &w.storage else {
            anyhow::bail!("attention output requires NVFP4");
        };
        anyhow::ensure!(
            w.compute_element_type == ElementType::Bf16
                && matrix.names.input_global_scale.is_none(),
            "unit output projection requires BF16 compute and no static QAD activation scale"
        );
        let operands =
            package.nvfp4_weight_operands(&name, weight("packed")?, weight("scales")?)?;
        let alpha_name = matrix
            .names
            .output_multiplier
            .as_deref()
            .ok_or_else(|| invalid("stored F32 alpha required"))?;
        let alpha = components
            .iter()
            .find(|c| c.external_names == [alpha_name])
            .cloned()
            .ok_or_else(|| invalid("missing original alpha"))?;
        let c = FastH3AttentionOutputConfig {
            rows,
            block,
            hidden: package.config().hidden_size().try_into()?,
            heads: package.config().attention_heads().try_into()?,
            head_dim: package.config().attention_head_dim().try_into()?,
            merge: FastH3AttentionMerge::ActiveCompressionFma,
            packed: operands.components[0].clone(),
            scales: operands.components[1].clone(),
            alpha: FastH3UnitAlphaDerivation::from_source(&package, alpha)?,
        };
        c.validate(limits)?;
        let alpha = c.alpha.output_component()?;
        let alpha_bytes = nvfp4_scalars::managed_scalar(host, c.alpha.result_bits()?)?;
        let unit = unit_component()?;
        let unit_bytes = nvfp4_scalars::managed_scalar(host, 1.0f32.to_bits())?;
        let source = Self {
            package,
            originals: [c.packed.clone(), c.scales.clone()],
            scalars: [
                (alpha, alpha_bytes, c.alpha.source_file().to_owned()),
                (unit, unit_bytes, "derived/constant-unit-f32".into()),
            ],
        };
        Ok((c, source))
    }
}
impl WeightComponentSource for FastH3AttentionOutputSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if let Some(original) = self.originals.iter().find(|p| p.id == c.id) {
            if original != c {
                return Err(invalid("original output projection component differs"));
            }
            return self.package.component(c);
        }
        let (declared, bytes, file) = self
            .scalars
            .iter()
            .find(|(p, _, _)| p.id == c.id)
            .ok_or_else(|| invalid("undeclared output source"))?;
        if declared != c {
            return Err(invalid("derived output scalar differs"));
        }
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec![file.clone()],
            vec![1],
            ElementType::F32,
            bytes.clone(),
        )
    }
}
