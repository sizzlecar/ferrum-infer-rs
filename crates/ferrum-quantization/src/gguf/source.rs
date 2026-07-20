use std::path::Path;

use ferrum_interfaces::vnext::{
    ElementType, VNextError, WeightComponentPayload, WeightComponentSource, WeightComponentSpec,
    WeightEncoding,
};
use ferrum_types::{FerrumError, Result};

use super::{GgmlDType, GgufFile};

/// Schema-addressed, mmap-backed GGUF source for vNext static weights.
/// Tensor payloads borrow the immutable file mapping and are never
/// dequantized or repacked at this boundary.
pub struct GgufWeightComponentSource {
    file: GgufFile,
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
        let file = GgufFile::open(path).map_err(|error| {
            FerrumError::model(format!(
                "open vNext GGUF source {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self { file, source_file })
    }

    pub fn file(&self) -> &GgufFile {
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

        let element_type = match &component.encoding {
            WeightEncoding::Dense { element_type } => {
                let actual = dense_element_type(info.ggml_dtype).ok_or_else(|| {
                    invalid_component(
                        component,
                        format!(
                            "GGUF dtype {:?} is quantized but the schema declares dense bytes",
                            info.ggml_dtype
                        ),
                    )
                })?;
                let dimensions = info
                    .shape
                    .dims()
                    .iter()
                    .map(|dimension| *dimension as u64)
                    .collect::<Vec<_>>();
                if actual != *element_type || dimensions != component.dimensions {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF dense tensor identity differs: dtype={:?} dimensions={dimensions:?}",
                            info.ggml_dtype
                        ),
                    ));
                }
                actual
            }
            WeightEncoding::BlockQuantized(spec) => {
                spec.validate()?;
                let actual_format =
                    block_quantization_format(info.ggml_dtype).ok_or_else(|| {
                        invalid_component(
                            component,
                            format!(
                            "GGUF dtype {:?} is not a supported fixed-block quantization format",
                            info.ggml_dtype
                        ),
                        )
                    })?;
                let logical_elements = u64::try_from(info.shape.elem_count()).map_err(|_| {
                    invalid_component(component, "GGUF tensor element count exceeds u64")
                })?;
                let block_width = u64::from(spec.logical_values_per_block);
                let mut physical_dimensions = info
                    .shape
                    .dims()
                    .iter()
                    .map(|dimension| *dimension as u64)
                    .collect::<Vec<_>>();
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
                if actual_format != spec.format_id.as_str()
                    || info.ggml_dtype.block_size() != spec.logical_values_per_block as usize
                    || info.ggml_dtype.type_size() != spec.bytes_per_block as usize
                    || !logical_elements.is_multiple_of(block_width)
                    || physical_dimensions != component.dimensions
                {
                    return Err(invalid_component(
                        component,
                        format!(
                            "GGUF block ABI differs: dtype={:?} format={actual_format} values_per_block={} bytes_per_block={} logical_elements={logical_elements} physical_dimensions={physical_dimensions:?}",
                            info.ggml_dtype,
                            info.ggml_dtype.block_size(),
                            info.ggml_dtype.type_size(),
                        ),
                    ));
                }
                ElementType::U8
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

        WeightComponentPayload::new(
            component,
            external_name.clone(),
            self.source_file.clone(),
            component.dimensions.clone(),
            element_type,
            bytes,
        )
    }
}

fn dense_element_type(dtype: GgmlDType) -> Option<ElementType> {
    match dtype {
        GgmlDType::F16 => Some(ElementType::F16),
        GgmlDType::BF16 => Some(ElementType::Bf16),
        GgmlDType::F32 => Some(ElementType::F32),
        _ => None,
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
