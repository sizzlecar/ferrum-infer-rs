use super::*;
use crate::fast_h3::nvfp4_scalars;
use std::sync::Arc;

const UNIT_COMPONENT: &str = "derived.fast_h3.attention_projection.unit_f32.v1";

pub use crate::fast_h3::nvfp4_scalars::FastH3UnitAlphaDerivation;

pub(super) fn unit_component() -> Result<WeightComponentSpec, VNextError> {
    Ok(WeightComponentSpec {
        id: WeightId::new(UNIT_COMPONENT)?,
        role: WeightComponentRole::Values,
        external_names: vec![UNIT_COMPONENT.into()],
        dimensions: vec![1],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    })
}

/// Original packed components borrow the immutable checkpoint mmap. Only four
/// numerical alpha values and one explicit unit scale allocate Host backing.
/// This source does not decide whether VSA's compression branch is active.
pub struct FastH3AttentionProjectionSource {
    package: Arc<FastH3TransformerPackage>,
    originals: BTreeMap<WeightId, WeightComponentSpec>,
    scalars: BTreeMap<WeightId, (WeightComponentSpec, ManagedHostBytes, String)>,
}
impl FastH3AttentionProjectionSource {
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        rows: u64,
        host: &HostMemoryAccount,
        limits: FastH3AttentionProjectionLimits,
    ) -> anyhow::Result<(FastH3AttentionProjectionConfig, Self)> {
        Self::prepare_group(
            package,
            block,
            rows,
            FastH3AttentionProjectionGroup::AllFour,
            host,
            limits,
        )
    }
    pub fn prepare_group(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        rows: u64,
        group: FastH3AttentionProjectionGroup,
        host: &HostMemoryAccount,
        limits: FastH3AttentionProjectionLimits,
    ) -> anyhow::Result<(FastH3AttentionProjectionConfig, Self)> {
        anyhow::ensure!(
            block < package.config().layers(),
            "attention projection block outside checkpoint"
        );
        let mut projections = Vec::with_capacity(group.projections().len());
        for &kind in group.projections() {
            let name = format!("transformer_blocks.{block}.attn.{}.weight", kind.module());
            let w = package
                .weight(&name)
                .ok_or_else(|| invalid("missing attention projection"))?;
            let FastH3WeightStorage::Nvfp4 { matrix, components } = &w.storage else {
                anyhow::bail!("attention projection requires NVFP4 {name}");
            };
            anyhow::ensure!(
                matrix.names.input_global_scale.is_none(),
                "unit attention profile cannot share a static-QAD projection: {name}"
            );
            anyhow::ensure!(
                w.compute_element_type == ElementType::Bf16,
                "attention projection compute precision differs"
            );
            let operands = package.nvfp4_weight_operands(
                &name,
                weight(&format!("{}.packed", kind.module()))?,
                weight(&format!("{}.scales", kind.module()))?,
            )?;
            let alpha_name = matrix
                .names
                .output_multiplier
                .as_deref()
                .ok_or_else(|| invalid("authoritative stored F32 alpha required"))?;
            let alpha = components
                .iter()
                .find(|c| c.external_names == [alpha_name])
                .cloned()
                .ok_or_else(|| invalid("missing original F32 alpha component"))?;
            projections.push(Projection {
                kind,
                packed: operands.components[0].clone(),
                scales: operands.components[1].clone(),
                alpha: FastH3UnitAlphaDerivation::from_source(&package, alpha)?,
            });
        }
        let config = FastH3AttentionProjectionConfig {
            rows,
            block,
            hidden: package.config().hidden_size().try_into()?,
            heads: package.config().attention_heads().try_into()?,
            head_dim: package.config().attention_head_dim().try_into()?,
            group,
            projections,
        };
        // Check every source/recipe/capacity before admitting derived backing.
        config.validate(limits)?;
        let mut originals = BTreeMap::new();
        let mut scalars = BTreeMap::new();
        for p in &config.projections {
            for c in [&p.packed, &p.scales] {
                originals.insert(c.id.clone(), c.clone());
            }
            let c = p.alpha.output_component()?;
            let bytes = nvfp4_scalars::managed_scalar(host, p.alpha.result_bits()?)?;
            scalars.insert(c.id.clone(), (c, bytes, p.alpha.source_file().to_owned()));
        }
        let unit = unit_component()?;
        let bytes = nvfp4_scalars::managed_scalar(host, 1.0f32.to_bits())?;
        scalars.insert(
            unit.id.clone(),
            (unit, bytes, "derived/constant-unit-f32".into()),
        );
        Ok((
            config,
            Self {
                package,
                originals,
                scalars,
            },
        ))
    }
}
impl WeightComponentSource for FastH3AttentionProjectionSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if let Some(original) = self.originals.get(&c.id) {
            if c != original {
                return Err(invalid("original projection component differs"));
            }
            return self.package.component(c);
        }
        let Some((declared, bytes, file)) = self.scalars.get(&c.id) else {
            return Err(invalid("undeclared projection source component"));
        };
        if c != declared {
            return Err(invalid("derived projection scalar differs"));
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
