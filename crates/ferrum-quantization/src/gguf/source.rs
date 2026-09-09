use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ElementType, RetainedHostMemoryRegion, StableHostMemory, VNextError, WeightComponentPayload,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_types::{FerrumError, Result};

use super::{GgmlDType, GgufFile, NativeGgufFile};
use crate::safetensors_archive::transcode_dense_bytes;

/// Schema-addressed, mmap-backed GGUF source for vNext static weights.
/// Fixed-block payloads borrow the immutable file mapping without
/// dequantization or repacking. Dense floating-point payloads are borrowed
/// when their type matches the schema and materialized on a cold-path source
/// request when the typed execution plan requires another floating-point type.
pub struct GgufWeightComponentSource {
    file: Arc<NativeGgufFile>,
    source_file: String,
}

impl GgufWeightComponentSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source_file = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| is_portable_source_file(name))
            .ok_or_else(|| {
                FerrumError::model(format!(
                    "GGUF path must end in one portable UTF-8 file name: {}",
                    path.display()
                ))
            })?
            .to_owned();
        let file = NativeGgufFile::open(path).map_err(|error| {
            FerrumError::model(format!(
                "open vNext GGUF source {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            file: Arc::new(file),
            source_file,
        })
    }

    pub fn file(&self) -> &NativeGgufFile {
        &self.file
    }

    pub fn source_file(&self) -> &str {
        &self.source_file
    }
}

impl WeightComponentSource for GgufWeightComponentSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let [external_name] = component.external_names.as_slice() else {
            return Err(invalid_component(
                component,
                "GGUF components must bind exactly one physical tensor; combine tensors in PhysicalWeightLayout instead",
            ));
        };
        let info = self.file.tensor_info(external_name).ok_or_else(|| {
            invalid_component(
                component,
                format!("GGUF tensor {external_name:?} is absent"),
            )
        })?;
        let bytes = self.file.tensor_byte_slice(external_name).ok_or_else(|| {
            invalid_component(
                component,
                format!("GGUF tensor {external_name:?} has an invalid block or byte range"),
            )
        })?;

        let (element_type, payload_bytes) = match &component.encoding {
            WeightEncoding::Dense { element_type } => {
                let WeightEncoding::Dense {
                    element_type: actual,
                } = &info.encoding
                else {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF type {} is quantized but the schema declares dense bytes",
                            info.ggml_type
                        ),
                    ));
                };
                let dimensions = &info.dimensions;
                if dimensions != &component.dimensions {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF dense tensor dimensions differ: source_dtype={:?} dimensions={dimensions:?}",
                            info.ggml_type,
                        ),
                    ));
                }
                let materialized =
                    transcode_dense_bytes(bytes, *actual, *element_type, external_name, None)?;
                (*element_type, materialized)
            }
            WeightEncoding::BlockQuantized(spec) => {
                spec.validate()?;
                let WeightEncoding::BlockQuantized(actual) = &info.encoding else {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF type {} is not a fixed-block quantization format",
                            info.ggml_type
                        ),
                    ));
                };
                let logical_elements = info.elements;
                let block_width = u64::from(spec.logical_values_per_block);
                let mut physical_dimensions = info.dimensions.clone();
                let innermost = physical_dimensions.last_mut().ok_or_else(|| {
                    invalid_component(
                        component,
                        "GGUF quantized tensor must have at least one axis",
                    )
                })?;
                if !innermost.is_multiple_of(block_width) {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF innermost dimension {innermost} is not divisible by block width {block_width}"
                        ),
                    ));
                }
                *innermost /= block_width;
                if actual != spec
                    || !logical_elements.is_multiple_of(block_width)
                    || physical_dimensions != component.dimensions
                {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF block ABI differs: file_type={} encoding={actual:?} logical_elements={logical_elements} physical_dimensions={physical_dimensions:?}",
                            info.ggml_type,
                        ),
                    ));
                }
                (ElementType::U8, bytes.into())
            }
            WeightEncoding::DenseAffine { .. } => {
                return Err(invalid_component(
                    component,
                    "GGUF source values are already transformed and cannot apply a dense affine source transform",
                ));
            }
            WeightEncoding::Quantized(_) => {
                return Err(invalid_component(
                    component,
                    "GGUF fixed-block bytes cannot satisfy a separate-component quantization encoding",
                ));
            }
        };

        let retained_host_memory =
            if payload_bytes.as_ptr() == bytes.as_ptr() && payload_bytes.len() == bytes.len() {
                let (offset_bytes, length_bytes) =
                    self.file.tensor_byte_range(external_name).ok_or_else(|| {
                        invalid_component(component, "GGUF tensor byte range is invalid")
                    })?;
                Some(RetainedHostMemoryRegion::new(
                    Arc::clone(&self.file),
                    offset_bytes,
                    length_bytes,
                )?)
            } else {
                None
            };
        let payload = WeightComponentPayload::new(
            component,
            external_name.clone(),
            self.source_file.clone(),
            component.dimensions.clone(),
            element_type,
            payload_bytes,
        )?;
        match retained_host_memory {
            Some(retained) => payload.with_retained_host_memory(retained),
            None => Ok(payload),
        }
    }
}

// SAFETY: GgufFile owns an immutable Mmap whose address and length do not
// change during its lifetime.
unsafe impl StableHostMemory for GgufFile {
    fn stable_bytes(&self) -> &[u8] {
        self.mmap_bytes()
    }
}

// SAFETY: NativeGgufFile owns the same immutable mapping lifetime contract.
unsafe impl StableHostMemory for NativeGgufFile {
    fn stable_bytes(&self) -> &[u8] {
        self.mmap_bytes()
    }
}

pub fn block_quantization_format(dtype: GgmlDType) -> Option<&'static str> {
    match dtype {
        GgmlDType::Q4_0 => Some("quantization.gguf.q4-0"),
        GgmlDType::Q4_1 => Some("quantization.gguf.q4-1"),
        GgmlDType::Q5_0 => Some("quantization.gguf.q5-0"),
        GgmlDType::Q5_1 => Some("quantization.gguf.q5-1"),
        GgmlDType::Q8_0 => Some("quantization.gguf.q8-0"),
        GgmlDType::Q8_1 => Some("quantization.gguf.q8-1"),
        GgmlDType::Q2K => Some("quantization.gguf.q2-k"),
        GgmlDType::Q3K => Some("quantization.gguf.q3-k"),
        GgmlDType::Q4K => Some("quantization.gguf.q4-k"),
        GgmlDType::Q5K => Some("quantization.gguf.q5-k"),
        GgmlDType::Q6K => Some("quantization.gguf.q6-k"),
        GgmlDType::Q8K => Some("quantization.gguf.q8-k"),
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32 => None,
    }
}

fn is_portable_source_file(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(['/', '\\'])
        && !matches!(name, "." | "..")
        && !name.bytes().any(|byte| byte.is_ascii_control())
}

fn invalid_component(component: &WeightComponentSpec, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!(
            "GGUF component `{}` does not match its typed source: {}",
            component.id,
            reason.into()
        ),
    }
}
