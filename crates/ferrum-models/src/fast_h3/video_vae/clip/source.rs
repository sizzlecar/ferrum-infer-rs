use super::*;
use ferrum_quantization::ManagedDenseArchiveSource;
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(super) struct Entry {
    pub name: &'static str,
    pub original: Vec<u64>,
    pub shape: Vec<u64>,
    pub dtype: ElementType,
}
pub(super) fn entries(a: &FastH3VideoVaeConfig) -> Result<Vec<Entry>, VNextError> {
    let z = a.latent_channels() as u64;
    let h = a.hidden_size().map_err(invalid)? as u64;
    let p = a.output_patch_width().map_err(invalid)? as u64;
    let r = a.register_tokens() as u64;
    Ok([
        (
            "post_quant_conv.weight",
            vec![z, z, 1, 1, 1],
            vec![z, z],
            ElementType::Bf16,
        ),
        ("post_quant_conv.bias", vec![z], vec![z], ElementType::Bf16),
        (
            "decoder.proj_in.weight",
            vec![h, z],
            vec![h, z],
            ElementType::Bf16,
        ),
        ("decoder.proj_in.bias", vec![h], vec![h], ElementType::Bf16),
        (
            "decoder.register_tokens",
            vec![1, r, h],
            vec![r, h],
            ElementType::F32,
        ),
        (
            "decoder.norm_out.weight",
            vec![h],
            vec![h],
            ElementType::F32,
        ),
        ("decoder.norm_out.bias", vec![h], vec![h], ElementType::F32),
        (
            "decoder.proj_out.weight",
            vec![p, h],
            vec![p, h],
            ElementType::Bf16,
        ),
        ("decoder.proj_out.bias", vec![p], vec![p], ElementType::Bf16),
    ]
    .into_iter()
    .map(|(name, original, shape, dtype)| Entry {
        name,
        original,
        shape,
        dtype,
    })
    .collect())
}
pub(super) fn component(
    original: &WeightComponentSpec,
    shape: Vec<u64>,
    dtype: ElementType,
) -> Result<WeightComponentSpec, VNextError> {
    // Recipe covers source schema, exact singleton reshape and destination
    // rounding. Original external names and source file attribution survive.
    let recipe = serde_json::to_vec(&(
        "fast_h3.video_clip.singleton_view.f32_to_bf16_rne.v1",
        original,
        &shape,
        dtype,
    ))
    .map_err(invalid)?;
    Ok(WeightComponentSpec {
        id: WeightId::new(format!(
            "derived.fast_h3.video_clip.{:x}",
            Sha256::digest(recipe)
        ))?,
        role: WeightComponentRole::Values,
        external_names: original.external_names.clone(),
        dimensions: shape,
        encoding: WeightEncoding::Dense {
            element_type: dtype,
        },
        required: true,
    })
}

/// Original package and block source are retained. The six autocast weights
/// are converted once on the caller's WeightBacking account. Singleton views
/// share precisely that original allocation; no unmanaged conversion mirror.
pub struct FastH3VideoClipSource {
    _package: Arc<FastH3VideoDecoderPackage>,
    blocks: FastH3VideoDecoderBlocksSource,
    outer: BTreeMap<WeightId, (Outer, RetainedHostMemoryRegion, Vec<String>)>,
}
impl FastH3VideoClipSource {
    pub fn prepare(
        package: Arc<FastH3VideoDecoderPackage>,
        grid: [u64; 4],
        host: &HostMemoryAccount,
        limits: FastH3VideoClipLimits,
    ) -> anyhow::Result<(FastH3VideoClipConfig, Self)> {
        anyhow::ensure!(
            !grid.contains(&0) && grid[0] <= limits.blocks.maximum_rows,
            "clip batch exceeds capacity"
        );
        anyhow::ensure!(
            grid[0]
                .checked_mul(8)
                .is_some_and(|n| n <= limits.blocks.maximum_metadata_bytes as u64),
            "clip sequence metadata exceeds capacity"
        );
        let a = package.config().clone();
        let sequence = product(&grid[1..])?
            .checked_add(a.register_tokens() as u64 + 1)
            .ok_or_else(|| invalid("sequence overflow"))?;
        anyhow::ensure!(
            product(&[grid[0], sequence])? <= limits.blocks.maximum_rows,
            "clip rows exceed capacity"
        );
        let outer = entries(&a)?
            .into_iter()
            .map(|e| -> anyhow::Result<Outer> {
                let original = package
                    .original_component(e.name)
                    .cloned()
                    .ok_or_else(|| invalid("missing original outer tensor"))?;
                Ok(Outer {
                    component: component(&original, e.shape, e.dtype)?,
                    original,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let (blocks, block_source) = FastH3VideoDecoderBlocksSource::prepare(
            package.clone(),
            0..a.decoder_layers(),
            vec![sequence; usize::try_from(grid[0])?],
            host,
            limits.blocks,
        )?;
        let config = FastH3VideoClipConfig {
            architecture: a,
            grid,
            blocks,
            outer,
        };
        config.validate(limits)?;
        // All graph/metadata/capacity checks precede the large outer conversions.
        program::compose(&config, limits)?;
        let converter = ManagedDenseArchiveSource::new(package.archive(), host);
        let mut retained = BTreeMap::new();
        for o in &config.outer {
            let original = package.component(&o.original)?;
            let files = original.source_files().to_vec();
            let memory = if o.component.physical_element_type() == ElementType::Bf16 {
                let mut original_shape = o.component.clone();
                original_shape.dimensions = o.original.dimensions.clone();
                let payload = converter.component(&original_shape)?;
                anyhow::ensure!(
                    payload
                        .bytes()
                        .chunks_exact(2)
                        .all(
                            |p| half::bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).is_finite()
                        ),
                    "outer BF16 conversion overflow"
                );
                payload
                    .retained_host_memory()
                    .cloned()
                    .ok_or_else(|| invalid("converted weight lacks admitted backing"))?
            } else {
                original
                    .retained_host_memory()
                    .cloned()
                    .ok_or_else(|| invalid("original archive lacks retained owner"))?
            };
            retained.insert(o.component.id.clone(), (o.clone(), memory, files));
        }
        Ok((
            config,
            Self {
                _package: package,
                blocks: block_source,
                outer: retained,
            },
        ))
    }
}
impl WeightComponentSource for FastH3VideoClipSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some((outer, memory, files)) = self.outer.get(&c.id) else {
            return self.blocks.component(c);
        };
        if c != &outer.component {
            return Err(invalid("outer component differs from immutable source"));
        }
        WeightComponentPayload::from_ordered_sources(
            c,
            c.external_names.clone(),
            files.clone(),
            c.dimensions.clone(),
            c.physical_element_type(),
            memory.bytes(),
        )?
        .with_retained_host_memory(memory.clone())
    }
}
