use super::*;
use crate::fast_h3::nvfp4_scalars;
use crate::fast_h3::{FastH3TransformerPackage, FastH3WeightStorage};
use std::sync::Arc;

/// A scalar recipe is part of the family configuration and thus the plan's
/// identity. It records both original F32 sources; source labels alone never
/// establish that a derived scalar equals a checkpoint payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3ScalarDerivation {
    pub(super) source_alpha: WeightComponentSpec,
    pub(super) source_global_sf: WeightComponentSpec,
    source_files: [String; 2],
    alpha_bits: u32,
    global_sf_bits: u32,
    result_bits: u32,
    arithmetic_profile: String,
}

impl FastH3ScalarDerivation {
    pub fn source_alpha(&self) -> &WeightComponentSpec {
        &self.source_alpha
    }
    pub fn source_global_sf(&self) -> &WeightComponentSpec {
        &self.source_global_sf
    }
    pub fn source_files(&self) -> &[String; 2] {
        &self.source_files
    }
    pub fn input_bits(&self) -> [u32; 2] {
        [self.alpha_bits, self.global_sf_bits]
    }
    pub fn result_bits(&self) -> u32 {
        self.result_bits
    }
    pub fn arithmetic_profile(&self) -> &str {
        &self.arithmetic_profile
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        for c in [&self.source_alpha, &self.source_global_sf] {
            nvfp4_scalars::validate_original_scalar(c)?;
        }
        if self.source_alpha.id == self.source_global_sf.id
            || self.source_alpha.external_names == self.source_global_sf.external_names
            || self.arithmetic_profile != SCALAR_PROFILE
            || self
                .source_files
                .iter()
                .any(|s| !nvfp4_scalars::valid_source_file(s))
        {
            return Err(invalid(
                "invalid scalar source provenance or arithmetic profile",
            ));
        }
        if nvfp4_scalars::quotient(self.alpha_bits, self.global_sf_bits)? != self.result_bits {
            return Err(invalid(
                "alpha/SF must produce the declared finite positive RN32 scalar",
            ));
        }
        Ok(())
    }

    pub(super) fn output_component(&self) -> Result<WeightComponentSpec, VNextError> {
        Ok(WeightComponentSpec {
            id: WeightId::new(format!(
                "derived.{}.rn32.{:08x}.{:08x}",
                self.source_alpha.id, self.alpha_bits, self.global_sf_bits
            ))?,
            role: WeightComponentRole::Values,
            // This names the authoritative alpha source. The other source and
            // the conversion are explicitly recorded by this typed recipe.
            external_names: self.source_alpha.external_names.clone(),
            dimensions: vec![1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        })
    }
}

/// A checked projection subset plus two explicitly derived invocation scalars.
/// Original packed/scales/SF payloads retain the package's mmap owners. Only
/// the two F32 quotients allocate, through the composition's Host account.
pub struct FastH3FfnSource {
    package: Arc<FastH3TransformerPackage>,
    originals: BTreeMap<WeightId, WeightComponentSpec>,
    scalars: BTreeMap<
        WeightId,
        (
            WeightComponentSpec,
            FastH3ScalarDerivation,
            ManagedHostBytes,
        ),
    >,
}

impl FastH3FfnSource {
    pub fn prepare(
        package: Arc<FastH3TransformerPackage>,
        block: usize,
        rows: u64,
        host: &HostMemoryAccount,
        limits: FastH3FfnLimits,
    ) -> anyhow::Result<(FastH3FfnConfig, Self)> {
        anyhow::ensure!(
            block < package.config().layers(),
            "FastH3 FFN block out of range"
        );
        let hidden = u64::try_from(package.config().hidden_size())?;
        let intermediate = u64::try_from(package.config().ffn_dim())?;
        let mut projections = Vec::new();
        for (index, suffix) in ["fc_in", "fc_out"].into_iter().enumerate() {
            let name = format!("transformer_blocks.{block}.ff.{suffix}.weight");
            let Some(weight) = package.weight(&name) else {
                anyhow::bail!("missing FastH3 FFN {name}");
            };
            let FastH3WeightStorage::Nvfp4 { matrix, components } = &weight.storage else {
                anyhow::bail!("FastH3 FFN requires NVFP4 {name}");
            };
            let operands = package.nvfp4_weight_operands(
                &name,
                super::weight(index, "packed")?,
                super::weight(index, "scales")?,
            )?;
            let select = |name: &str| {
                components
                    .iter()
                    .find(|c| c.external_names == [name])
                    .cloned()
                    .ok_or_else(|| invalid("FFN source component absent"))
            };
            let alpha = select(
                matrix
                    .names
                    .output_multiplier
                    .as_deref()
                    .ok_or_else(|| invalid("FFN requires authoritative stored F32 alpha"))?,
            )?;
            let sf = select(
                matrix
                    .names
                    .input_global_scale
                    .as_deref()
                    .ok_or_else(|| invalid("FFN requires static QAD input scale"))?,
            )?;
            let (alpha_bits, alpha_file) = nvfp4_scalars::original_scalar(&package, &alpha)?;
            let (global_sf_bits, sf_file) = nvfp4_scalars::original_scalar(&package, &sf)?;
            let derivation = FastH3ScalarDerivation {
                source_alpha: alpha,
                source_global_sf: sf.clone(),
                source_files: [alpha_file, sf_file],
                alpha_bits,
                global_sf_bits,
                result_bits: nvfp4_scalars::quotient(alpha_bits, global_sf_bits)?,
                arithmetic_profile: SCALAR_PROFILE.into(),
            };
            derivation.validate()?;
            projections.push(Projection {
                rows: matrix.geometry.rows().try_into()?,
                columns: matrix.geometry.columns().try_into()?,
                packed: operands.components[0].clone(),
                scales: operands.components[1].clone(),
                global_sf: sf,
                alpha: derivation.output_component()?,
                derivation,
            });
        }
        let config = FastH3FfnConfig {
            rows,
            chunk_rows: None,
            block,
            hidden,
            intermediate,
            projections: projections
                .try_into()
                .map_err(|_| invalid("FFN requires two projections"))?,
        };
        // All geometry, metadata, source and scalar checks precede allocation.
        config.validate(limits)?;
        let mut originals = BTreeMap::new();
        let mut scalars = BTreeMap::new();
        for p in &config.projections {
            for c in [&p.packed, &p.scales, &p.global_sf] {
                originals.insert(c.id.clone(), c.clone());
            }
            let bytes = nvfp4_scalars::managed_scalar(host, p.derivation.result_bits)?;
            scalars.insert(
                p.alpha.id.clone(),
                (p.alpha.clone(), p.derivation.clone(), bytes),
            );
        }
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

impl WeightComponentSource for FastH3FfnSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if let Some(original) = self.originals.get(&c.id) {
            if c != original {
                return Err(invalid("FFN original source declaration changed"));
            }
            return self.package.component(c);
        }
        let Some((declared, recipe, bytes)) = self.scalars.get(&c.id) else {
            return Err(invalid("undeclared FFN source component"));
        };
        if c != declared {
            return Err(invalid("FFN derived scalar declaration changed"));
        }
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec![recipe.source_files[0].clone()],
            vec![1],
            ElementType::F32,
            bytes.clone(),
        )
    }
}
