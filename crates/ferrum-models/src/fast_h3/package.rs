//! A checked transformer weight package, not a registered inference pipeline.
//! The caller selects archives and an explicit deployment profile. Quantized
//! coverage is never inferred from which matrices happen to be present.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use ferrum_quantization::{
    nvfp4::{
        Nvfp4Geometry, Nvfp4Matrix, Nvfp4MatrixNames, Nvfp4MatrixSpec, Nvfp4SafetensorsSource,
        Nvfp4SourceLimits,
    },
    SafetensorsArchive,
};
use ferrum_runtime::{
    ElementType, QuantizationSpec, VNextError, WeightComponentPayload, WeightComponentRole,
    WeightComponentSource, WeightComponentSpec, WeightEncoding, WeightId,
};

use super::FastH3TransformerConfig;

mod inventory;
mod operands;
pub use operands::FastH3Nvfp4WeightOperands;
#[cfg(test)]
mod tests;

/// FastVideo's three explicit H3 deployment sets. Every set keeps embeddings,
/// refiner, AdaLN and widened PDD heads dense; skip_blocks overrides the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastH3Nvfp4Profile {
    FeedForward,
    DiT,
    DiTWithVsa,
}

/// Import validation bounds, not an execution memory grant. Archives are
/// already mapped by the caller; no device weights are allocated here.
#[derive(Clone, Copy, Debug)]
pub struct FastH3WeightLimits {
    pub maximum_tensors: usize,
    pub maximum_source_bytes: usize,
    pub maximum_skip_metadata_bytes: usize,
}

#[derive(Clone, Debug)]
pub enum FastH3WeightStorage {
    Dense(WeightComponentSpec),
    Nvfp4 {
        matrix: Nvfp4MatrixSpec,
        components: Vec<WeightComponentSpec>,
    },
}

#[derive(Clone, Debug)]
pub struct FastH3Weight {
    /// Canonical FastVideo module parameter name, after import alias resolution.
    pub name: String,
    pub dimensions: Vec<usize>,
    /// Upstream model execution dtype, distinct from stored checkpoint dtype.
    /// A later execution program must explicitly implement any required cast.
    pub compute_element_type: ElementType,
    pub storage: FastH3WeightStorage,
}

/// Complete checked transformer inventory sharing immutable mmap payloads.
/// It does not supply a transformer forward, a device readiness proof, or a
/// memory reservation. Frozen references and PDD progress live elsewhere.
pub struct FastH3TransformerPackage {
    config: FastH3TransformerConfig,
    profile: FastH3Nvfp4Profile,
    skip_blocks: Vec<usize>,
    weights: BTreeMap<String, FastH3Weight>,
    components: BTreeMap<WeightId, (bool, WeightComponentSpec)>,
    dense: SafetensorsArchive,
    packed: Nvfp4SafetensorsSource,
    retained_dense_originals: Vec<String>,
    source_bytes: usize,
}

impl FastH3TransformerPackage {
    pub fn new(
        config: &FastH3TransformerConfig,
        dense: SafetensorsArchive,
        packed: SafetensorsArchive,
        profile: FastH3Nvfp4Profile,
        limits: FastH3WeightLimits,
    ) -> Result<Self> {
        ensure!(
            limits.maximum_tensors > 0
                && limits.maximum_source_bytes > 0
                && limits.maximum_skip_metadata_bytes > 0,
            "FastH3 weight limits must be positive"
        );
        let count = dense
            .tensor_count()
            .checked_add(packed.tensor_count())
            .context("source tensor count overflow")?;
        ensure!(
            count <= limits.maximum_tensors,
            "FastH3 source tensor count exceeds capacity"
        );
        // Budget every source, including retained originals, before the NVFP4
        // adapter validates scale payloads. Metadata queries do not copy weights.
        let mut source_bytes = 0usize;
        for archive in [&dense, &packed] {
            for name in archive.tensor_names() {
                source_bytes = source_bytes
                    .checked_add(archive.tensor(name)?.bytes().len())
                    .context("source byte count overflow")?;
                ensure!(
                    source_bytes <= limits.maximum_source_bytes,
                    "FastH3 source bytes exceed capacity"
                );
            }
        }
        let skip_blocks =
            skip_blocks(&packed, config.layers(), limits.maximum_skip_metadata_bytes)?;
        let entries = inventory::entries(config, profile, &skip_blocks, limits.maximum_tensors)?;
        let mut used_dense = BTreeSet::new();
        let mut used_packed = BTreeSet::new();
        let mut retained_dense_originals = Vec::new();
        let mut matrices = Vec::new();
        let mut weights = BTreeMap::new();
        let mut components = BTreeMap::new();
        for entry in entries {
            let external = resolve_dense(&dense, &entry)?;
            let storage = if entry.packed {
                if let Some(name) = external {
                    validate_dense(&dense, &name, &entry.shape)?;
                    used_dense.insert(name.clone());
                    retained_dense_originals.push(name);
                }
                let prefix = entry
                    .name
                    .strip_suffix(".weight")
                    .context("packed parameter is not a matrix")?;
                let names = resolve_packed(&packed, prefix)?;
                let matrix = Nvfp4MatrixSpec {
                    names,
                    geometry: Nvfp4Geometry::new(entry.shape[0], entry.shape[1])?,
                };
                let parts = packed_components(&packed, &matrix)?;
                for part in &parts {
                    used_packed.insert(part.external_names[0].clone());
                    ensure!(
                        components
                            .insert(part.id.clone(), (true, part.clone()))
                            .is_none(),
                        "duplicate weight component"
                    );
                }
                matrices.push(matrix.clone());
                FastH3WeightStorage::Nvfp4 {
                    matrix,
                    components: parts,
                }
            } else {
                let name = external
                    .with_context(|| format!("missing dense FastH3 parameter {}", entry.name))?;
                let dtype = validate_dense(&dense, &name, &entry.shape)?;
                let part = component(
                    &name,
                    dense.tensor(&name)?.shape().to_vec(),
                    WeightComponentRole::Values,
                    WeightEncoding::Dense {
                        element_type: dtype,
                    },
                )?;
                used_dense.insert(name);
                ensure!(
                    components
                        .insert(part.id.clone(), (false, part.clone()))
                        .is_none(),
                    "duplicate weight component"
                );
                FastH3WeightStorage::Dense(part)
            };
            weights.insert(
                entry.name.clone(),
                FastH3Weight {
                    name: entry.name,
                    dimensions: entry.shape,
                    compute_element_type: entry.compute_type,
                    storage,
                },
            );
        }
        for (archive, used, label) in [
            (&dense, &used_dense, "dense"),
            (&packed, &used_packed, "NVFP4"),
        ] {
            ensure!(
                used.len() == archive.tensor_count(),
                "unrecognized FastH3 {label} parameters: {:?}",
                archive
                    .tensor_names()
                    .filter(|name| !used.contains(*name))
                    .take(8)
                    .collect::<Vec<_>>()
            );
        }
        // This is the actual format consumer: packed shapes, swizzled scale
        // geometry and finite positive authoritative alpha are checked here.
        let packed = Nvfp4SafetensorsSource::new(
            packed,
            matrices,
            Nvfp4SourceLimits {
                maximum_matrices: limits.maximum_tensors,
                maximum_source_bytes: limits.maximum_source_bytes,
            },
        )?;
        Ok(Self {
            config: config.clone(),
            profile,
            skip_blocks,
            weights,
            components,
            dense,
            packed,
            retained_dense_originals,
            source_bytes,
        })
    }

    pub fn config(&self) -> &FastH3TransformerConfig {
        &self.config
    }
    /// Model consumers must still select and verify the canonical inventory
    /// before passing a subset to a managed conversion source.
    pub(super) fn dense_archive(&self) -> &SafetensorsArchive {
        &self.dense
    }
    pub fn profile(&self) -> FastH3Nvfp4Profile {
        self.profile
    }
    pub fn skip_blocks(&self) -> &[usize] {
        &self.skip_blocks
    }
    pub fn weights(&self) -> impl ExactSizeIterator<Item = &FastH3Weight> {
        self.weights.values()
    }
    pub fn weight(&self, canonical_name: &str) -> Option<&FastH3Weight> {
        self.weights.get(canonical_name)
    }
    pub fn components(&self) -> impl ExactSizeIterator<Item = &WeightComponentSpec> {
        self.components.values().map(|(_, c)| c)
    }
    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }
    /// Validated redundant originals are not executable components and never
    /// silently substitute for a missing quantized projection.
    pub fn retained_dense_originals(&self) -> &[String] {
        &self.retained_dense_originals
    }
    pub fn nvfp4_matrix(&self, canonical_name: &str) -> Result<Nvfp4Matrix<'_>> {
        let Some(FastH3Weight {
            storage: FastH3WeightStorage::Nvfp4 { matrix, .. },
            ..
        }) = self.weight(canonical_name)
        else {
            anyhow::bail!("FastH3 parameter {canonical_name} is not an NVFP4 matrix");
        };
        Ok(self.packed.matrix(&matrix.names.packed)?)
    }
}

impl WeightComponentSource for FastH3TransformerPackage {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((packed, declared)) = self.components.get(&c.id) else {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("undeclared FastH3 component {}", c.id),
            });
        };
        if c != declared {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "FastH3 component {} differs from validated declaration",
                    c.id
                ),
            });
        }
        if *packed {
            self.packed.component(c)
        } else {
            self.dense.component(c)
        }
    }
}

fn skip_blocks(
    archive: &SafetensorsArchive,
    layers: usize,
    max_bytes: usize,
) -> Result<Vec<usize>> {
    let metadata = archive.header_metadata()?;
    ensure!(
        metadata.len() == 1,
        "FastH3 packed export must be one explicitly selected file"
    );
    let Some(raw) = metadata.values().next().and_then(|m| m.get("skip_blocks")) else {
        return Ok(Vec::new());
    };
    ensure!(
        raw.len() <= max_bytes,
        "FastH3 skip_blocks metadata exceeds capacity"
    );
    let mut blocks: Vec<usize> = serde_json::from_str(raw)
        .context("skip_blocks must be a JSON array of nonnegative integers")?;
    ensure!(
        blocks.iter().all(|i| *i < layers),
        "skip_blocks index exceeds transformer layers"
    );
    blocks.sort_unstable();
    blocks.dedup(); // Same set semantics as the pinned upstream reader.
    Ok(blocks)
}

fn resolve_dense(archive: &SafetensorsArchive, entry: &inventory::Entry) -> Result<Option<String>> {
    let found: Vec<_> = entry
        .aliases
        .iter()
        .filter(|n| archive.contains(n))
        .collect();
    ensure!(
        found.len() <= 1,
        "ambiguous dense aliases for {}",
        entry.name
    );
    Ok(found.first().map(|name| (*name).clone()))
}

fn validate_dense(
    archive: &SafetensorsArchive,
    name: &str,
    shape: &[usize],
) -> Result<ElementType> {
    let tensor = archive.tensor(name)?;
    ensure!(
        tensor
            .shape()
            .iter()
            .copied()
            .eq(shape.iter().map(|n| *n as u64)),
        "FastH3 dense shape differs for {name}"
    );
    let dtype = tensor
        .element_type()
        .context("unsupported dense storage dtype")?;
    ensure!(
        matches!(
            dtype,
            ElementType::F32 | ElementType::Bf16 | ElementType::F16
        ),
        "FastH3 dense {name} requires floating-point storage"
    );
    Ok(dtype)
}

fn resolve_packed(archive: &SafetensorsArchive, prefix: &str) -> Result<Nvfp4MatrixNames> {
    let plain = Nvfp4MatrixNames::fastvideo_dit(prefix, false);
    let qualified_prefix = format!("minimax_h3.{prefix}");
    let qualified = Nvfp4MatrixNames::fastvideo_dit(&qualified_prefix, false);
    let has_plain = archive.contains(&plain.packed);
    let has_qualified = archive.contains(&qualified.packed);
    ensure!(
        has_plain != has_qualified,
        "missing or ambiguous NVFP4 matrix {prefix}"
    );
    let prefix = if has_plain { prefix } else { &qualified_prefix };
    Ok(Nvfp4MatrixNames::fastvideo_dit(
        prefix,
        archive.contains(&format!("{prefix}::_nvfp4_input_global_sf")),
    ))
}

fn packed_components(
    archive: &SafetensorsArchive,
    matrix: &Nvfp4MatrixSpec,
) -> Result<Vec<WeightComponentSpec>> {
    let n = &matrix.names;
    let mut out = Vec::new();
    for (name, role) in [
        (&n.packed, WeightComponentRole::PackedValues),
        (&n.scales, WeightComponentRole::Scales),
        (&n.global_scale, WeightComponentRole::Metadata),
        (
            n.output_multiplier
                .as_ref()
                .context("FastVideo alpha is required")?,
            WeightComponentRole::Metadata,
        ),
    ]
    .into_iter()
    .chain(
        n.input_global_scale
            .as_ref()
            .map(|s| (s, WeightComponentRole::Metadata)),
    ) {
        let tensor = archive.tensor(name)?;
        let encoding = match role {
            WeightComponentRole::PackedValues => {
                WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
            }
            WeightComponentRole::Scales => WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
            _ => WeightEncoding::Dense {
                element_type: tensor
                    .element_type()
                    .context("unsupported NVFP4 scalar storage")?,
            },
        };
        out.push(component(name, tensor.shape().to_vec(), role, encoding)?);
    }
    Ok(out)
}

fn component(
    name: &str,
    dimensions: Vec<u64>,
    role: WeightComponentRole,
    encoding: WeightEncoding,
) -> Result<WeightComponentSpec> {
    Ok(WeightComponentSpec {
        id: WeightId::new(format!("fast_h3.{name}"))?,
        role,
        external_names: vec![name.to_owned()],
        dimensions,
        encoding,
        required: true,
    })
}
