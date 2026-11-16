use super::super::{Qwen3VLTextArchitecture, Qwen3VLTextLayout};
use super::*;
use config::Projection;
use ferrum_quantization::nvfp4::{
    Nvfp4Geometry, Nvfp4MatrixNames, Nvfp4MatrixSpec, Nvfp4SafetensorsSource, Nvfp4SourceLimits,
};
use ferrum_quantization::{ManagedDenseArchiveSource, SafetensorsArchive};
use safetensors::Dtype;

/// Bounds selected checkpoint inventory and payload validation. Actual Host
/// backing and GPU admission remain the caller's composition accounts.
#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLTextNvfp4Limits {
    pub maximum_projections: usize,
    pub maximum_source_bytes: usize,
}

pub struct Qwen3VLTextNvfp4Checkpoint {
    config: Qwen3VLTextNvfp4Config,
    source: Nvfp4SafetensorsSource,
    schema: WeightSchema,
}
impl Qwen3VLTextNvfp4Checkpoint {
    /// Select an explicit prefix of the checkpoint's explicitly retained
    /// language layers. Visual weights and token embedding lookup are separate
    /// consumers; no dense matrix is reconstructed during this operation.
    pub fn new(
        metadata: &Value,
        archive: SafetensorsArchive,
        layout: Qwen3VLTextLayout,
        execute_layers: u64,
        limits: Qwen3VLTextNvfp4Limits,
    ) -> anyhow::Result<Self> {
        Self::from_shared_archive(
            metadata,
            std::sync::Arc::new(archive),
            layout,
            execute_layers,
            limits,
        )
    }

    /// Different request layouts share the original checkpoint mapping.
    pub fn from_shared_archive(
        metadata: &Value,
        archive: std::sync::Arc<SafetensorsArchive>,
        layout: Qwen3VLTextLayout,
        execute_layers: u64,
        limits: Qwen3VLTextNvfp4Limits,
    ) -> anyhow::Result<Self> {
        validate_metadata(metadata)?;
        let architecture = Qwen3VLTextArchitecture::from_hf_value(metadata)?;
        let exported = metadata
            .get("num_hidden_layers_override")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid("missing explicit num_hidden_layers_override"))?;
        anyhow::ensure!(
            exported > 0
                && exported <= architecture.layers()
                && execute_layers > 0
                && execute_layers <= exported,
            "executed/exported layer counts must form a nonempty architecture prefix"
        );
        let count = usize::try_from(execute_layers)?
            .checked_mul(7)
            .ok_or_else(|| invalid("projection count overflow"))?;
        anyhow::ensure!(
            count <= limits.maximum_projections && limits.maximum_source_bytes > 0,
            "selected inventory exceeds source limits"
        );
        let text = Qwen3VLTextConfig::new(architecture.prefix_layers(execute_layers)?, layout)?;
        let mut specs = Vec::with_capacity(count);
        let mut projections = Vec::with_capacity(count);
        let mut bytes = 0usize;
        for w in text.expected_weights()? {
            if w.dimensions().len() == 1 {
                let t = archive.tensor(w.external_name())?;
                anyhow::ensure!(
                    t.shape() == w.dimensions()
                        && matches!(t.dtype(), Dtype::BF16 | Dtype::F16 | Dtype::F32),
                    "norm shape/dtype mismatch: {}",
                    w.external_name()
                );
                bytes = bounded_bytes(bytes, t.bytes().len(), limits.maximum_source_bytes)?;
                continue;
            }
            let module = w
                .external_name()
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("projection external name"))?;
            anyhow::ensure!(
                !archive.contains(w.external_name())
                    && !archive.contains(&format!("{module}.pre_quant_scale")),
                "unexpected dense or AWQ-smoothed projection: {module}"
            );
            let names = Nvfp4MatrixNames::serialized_encoder(module);
            let rows = w.dimensions()[0];
            let columns = w.dimensions()[1];
            anyhow::ensure!(
                rows % 128 == 0 && columns % 64 == 0,
                "serialized encoder requires whole 128x4 scale tiles"
            );
            let geometry = Nvfp4Geometry::new(usize::try_from(rows)?, usize::try_from(columns)?)?;
            // This checkpoint dialect stores exact U8 2D scale storage and
            // F32[1] global SF. More general source dialects stay explicit.
            let scales = archive.tensor(&names.scales)?;
            anyhow::ensure!(
                scales.dtype() == Dtype::U8 && scales.shape() == [rows, columns / 16],
                "encoder scale storage must be U8 whole tiles"
            );
            let global = archive.tensor(&names.global_scale)?;
            anyhow::ensure!(
                global.dtype() == Dtype::F32 && global.shape() == [1],
                "encoder global scale must preserve F32[1]"
            );
            let bits = u32::from_le_bytes(global.bytes().try_into()?);
            let p = Projection {
                module: module.into(),
                rows,
                columns,
                global_bits: bits,
                source_file: global.source_file().into(),
            };
            p.alpha()?;
            for name in [&names.packed, &names.scales, &names.global_scale] {
                bytes = bounded_bytes(
                    bytes,
                    archive.tensor(name)?.bytes().len(),
                    limits.maximum_source_bytes,
                )?;
            }
            projections.push(p);
            specs.push(Nvfp4MatrixSpec { names, geometry });
        }
        let config = Qwen3VLTextNvfp4Config {
            text,
            declared_layers: architecture.layers(),
            exported_layers: exported,
            scalar_profile: SCALAR_PROFILE.into(),
            projections,
            attention: config::Qwen3VLTextNvfp4Attention::Sequential,
        };
        config.validate()?;
        let schema = config.schema()?;
        let source = Nvfp4SafetensorsSource::from_shared_archive(
            archive,
            specs,
            Nvfp4SourceLimits {
                maximum_matrices: limits.maximum_projections,
                maximum_source_bytes: limits.maximum_source_bytes,
            },
        )?;
        // Both implementations of the pinned scalar recipe must agree before
        // allocating invocation constants or admitting any execution resources.
        for p in &config.projections {
            anyhow::ensure!(
                source
                    .matrix(&format!("{}.weight_packed", p.module))?
                    .output_multiplier()
                    .to_bits()
                    == p.alpha()?.to_bits(),
                "source reciprocal arithmetic differs from declared encoder profile"
            );
        }
        Ok(Self {
            config,
            source,
            schema,
        })
    }
    pub fn config(&self) -> &Qwen3VLTextNvfp4Config {
        &self.config
    }
    pub fn with_attention(
        mut self,
        attention: config::Qwen3VLTextNvfp4Attention,
    ) -> Result<Self, VNextError> {
        let mut config = self.config.clone();
        config.attention = attention;
        config.validate()?;
        if config.schema()? != self.schema {
            return Err(invalid("attention profile must preserve original weights"));
        }
        self.config = config;
        Ok(self)
    }
    /// Bind the same visual-row additions as the dense text family. Visual
    /// inputs are invocation data; no original weight or quantization mapping
    /// changes when enabling the three post-layer DeepStack updates.
    pub fn with_deepstack(
        mut self,
        deepstack: super::super::Qwen3VLTextDeepStack,
    ) -> Result<Self, VNextError> {
        let mut config = self.config.clone();
        config.text = config.text.with_deepstack(deepstack)?;
        config.validate()?;
        if config.schema()? != self.schema {
            return Err(invalid(
                "visual routing must preserve the original stored schema",
            ));
        }
        self.config = config;
        Ok(self)
    }
    pub fn archive(&self) -> &SafetensorsArchive {
        self.source.archive()
    }
    pub fn managed_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> Result<Qwen3VLTextNvfp4Source<'a>, VNextError> {
        let mut scalars = BTreeMap::new();
        let mut insert =
            |c: WeightComponentSpec, bits: u32, file: String| -> Result<(), VNextError> {
                let mut bytes = host
                    .allocate_zeroed(HostMemoryClass::WeightBacking, 4)
                    .map_err(|e| invalid(e.to_string()))?;
                bytes.as_mut().copy_from_slice(&bits.to_le_bytes());
                scalars.insert(c.id.clone(), (c, bytes.freeze(), file));
                Ok(())
            };
        for p in &self.config.projections {
            insert(
                p.alpha_component()?,
                p.alpha()?.to_bits(),
                p.source_file.clone(),
            )?;
        }
        insert(
            config::unit_component()?,
            1.0f32.to_bits(),
            "derived/constant-unit-f32".into(),
        )?;
        Ok(Qwen3VLTextNvfp4Source {
            checkpoint: self,
            dense: ManagedDenseArchiveSource::new(self.archive(), host),
            scalars,
        })
    }
}

fn bounded_bytes(old: usize, added: usize, limit: usize) -> Result<usize, VNextError> {
    old.checked_add(added)
        .filter(|n| *n <= limit)
        .ok_or_else(|| invalid("selected source bytes exceed limit"))
}
fn validate_metadata(raw: &Value) -> Result<(), VNextError> {
    let q = raw
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("explicit NVFP4 metadata required"))?;
    for (key, value) in [
        ("quant_method", "nvfp4"),
        ("activation_scheme", "dynamic"),
        ("fmt", "e2m1"),
        ("scale_fmt", "e4m3"),
        ("scale_layout", "128x4"),
    ] {
        if q.get(key).and_then(Value::as_str) != Some(value) {
            return Err(invalid(format!("unsupported explicit {key}")));
        }
    }
    if q.get("group_size").and_then(Value::as_u64) != Some(16)
        || q.get("modules_to_not_convert") != Some(&serde_json::json!(["model.visual", "lm_head"]))
        || q.get("pre_quant_scale")
            .is_some_and(|v| v != &Value::Bool(false))
        || q.keys().any(|k| {
            !matches!(
                k.as_str(),
                "quant_method"
                    | "activation_scheme"
                    | "fmt"
                    | "group_size"
                    | "scale_fmt"
                    | "scale_layout"
                    | "modules_to_not_convert"
                    | "pre_quant_scale"
                    | "producer"
            )
        })
    {
        return Err(invalid("requires group16, explicit visual/head exclusions and all seven unsmoothed NVFP4 projections"));
    }
    Ok(())
}

/// Source allocation is limited to one F32 alpha per projection and one unit
/// SF. Packed data retains the archive's mmap owner. Norm conversion uses the
/// caller's Host account through the existing managed dense source.
pub struct Qwen3VLTextNvfp4Source<'a> {
    checkpoint: &'a Qwen3VLTextNvfp4Checkpoint,
    dense: ManagedDenseArchiveSource<'a>,
    scalars: BTreeMap<WeightId, (WeightComponentSpec, ManagedHostBytes, String)>,
}
impl WeightComponentSource for Qwen3VLTextNvfp4Source<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if !self
            .checkpoint
            .schema
            .components
            .iter()
            .any(|expected| expected == c)
        {
            return Err(invalid("component differs from checked encoder schema"));
        }
        if let Some((_, bytes, file)) = self.scalars.get(&c.id) {
            return WeightComponentPayload::from_managed_host(
                c,
                c.external_names.clone(),
                vec![file.clone()],
                vec![1],
                ElementType::F32,
                bytes.clone(),
            );
        }
        if matches!(
            c.role,
            WeightComponentRole::PackedValues | WeightComponentRole::Scales
        ) {
            self.checkpoint.source.component(c)
        } else {
            self.dense.component(c)
        }
    }
}
