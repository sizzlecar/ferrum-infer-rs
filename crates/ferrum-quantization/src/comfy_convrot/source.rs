//! Raw canonical component source. In particular, E4M3FN group scale bytes
//! are U8 carriers to the typed W4A8 operation, never integer-valued scales.
use super::*;
use ferrum_runtime::{
    ElementType, VNextError, WeightComponentPayload, WeightComponentRole, WeightComponentSource,
    WeightComponentSpec, WeightEncoding,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID: &str =
    "weight-format.comfy.convrot.canonical.v1";

/// Owns the immutable archive and validates every declared quantized module.
/// Dense exclusions retain their original source loading/transcoding rules.
/// No large weight is expanded, decoded or copied when constructing this source.
pub struct ComfyCanonicalWeightSource {
    archive: Arc<SafetensorsArchive>,
    modules: BTreeMap<String, ComfyQuantizationMetadata>,
}
impl ComfyCanonicalWeightSource {
    pub fn new(archive: SafetensorsArchive) -> Result<Self> {
        Self::from_shared_archive(Arc::new(archive))
    }
    /// Preserve an existing immutable mmap lease (including an original
    /// production source's verified identity), without reopening a path.
    pub fn from_shared_archive(archive: Arc<SafetensorsArchive>) -> Result<Self> {
        let names: Vec<_> = archive
            .tensor_names()
            .filter_map(|name| name.strip_suffix(".comfy_quant").map(str::to_owned))
            .collect();
        let mut modules = BTreeMap::new();
        for name in names {
            let matrix = ComfyQuantizedMatrix::from_archive(&archive, &name)?;
            match matrix.metadata() {
                ComfyQuantizationMetadata::Int8Tensorwise {
                    convrot,
                    convrot_groupsize,
                } => {
                    let ComfyQuantizationComponents::Int8 { scale } = matrix.components() else {
                        unreachable!("validated INT8 matrix components")
                    };
                    if !convrot
                        || *convrot_groupsize != 256
                        || scale.shape() != [matrix.rows() as u64, 1]
                    {
                        return Err(invalid("canonical execution source requires INT8 regular ConvRot groups of 256 and exact F32 per-output scale [N,1]"));
                    }
                }
                ComfyQuantizationMetadata::CodebookW4A8 {
                    group_size,
                    convrot_groupsize,
                    ..
                } => {
                    if *group_size != 16 || *convrot_groupsize != 256 {
                        return Err(invalid("canonical execution source requires W4 scale groups of 16 and regular ConvRot groups of 256"));
                    }
                }
            }
            modules.insert(name, matrix.metadata().clone());
        }
        Ok(Self { archive, modules })
    }
    pub fn archive(&self) -> &SafetensorsArchive {
        &self.archive
    }
    pub fn modules(&self) -> &BTreeMap<String, ComfyQuantizationMetadata> {
        &self.modules
    }
}
impl WeightComponentSource for ComfyCanonicalWeightSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [external] = component.external_names.as_slice() else {
            return self.archive.component(component);
        };
        let Some(module) = external.strip_suffix(".weight_s_rel") else {
            return self.archive.component(component);
        };
        if !matches!(
            self.modules.get(module),
            Some(ComfyQuantizationMetadata::CodebookW4A8 { .. })
        ) || !matches!(
            component.role,
            WeightComponentRole::Values | WeightComponentRole::Scales
        ) || component.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::U8,
            })
        {
            return Err(failure("E4M3 relative scales require their validated W4A8 module and explicit U8 scale-carrier schema"));
        }
        let tensor = self
            .archive
            .tensor(external)
            .map_err(|error| failure(error.to_string()))?;
        if tensor.dtype() != Dtype::F8_E4M3 || tensor.shape() != component.dimensions {
            return Err(failure(
                "relative scale source dtype/shape differs from its canonical schema",
            ));
        }
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            ElementType::U8,
            tensor.bytes(),
        )?
        .with_retained_host_memory(tensor.retained_host_memory().clone())
    }
}
fn failure(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("Comfy canonical source: {}", reason.into()),
    }
}
