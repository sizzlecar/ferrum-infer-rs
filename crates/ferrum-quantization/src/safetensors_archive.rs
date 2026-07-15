use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use ferrum_interfaces::vnext::{
    CanonicalRational, ElementType, VNextError, WeightComponentPayload, WeightComponentSource,
    WeightComponentSpec, WeightEncoding,
};
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

#[derive(Debug, Clone)]
struct TensorMeta {
    shard: usize,
    dtype: Dtype,
    shape: Vec<u64>,
    data_start: usize,
    data_end: usize,
}

struct Shard {
    relative_path: String,
    mmap: Mmap,
}

/// Mmap-backed, once-indexed safetensors archive shared by vNext model
/// packages. Tensor payload access does not reparse shard headers.
pub struct SafetensorsArchive {
    shards: Vec<Shard>,
    tensors: BTreeMap<String, TensorMeta>,
}

pub struct SafetensorsTensor<'archive> {
    external_name: &'archive str,
    source_file: &'archive str,
    dtype: Dtype,
    shape: &'archive [u64],
    bytes: &'archive [u8],
}

impl<'archive> SafetensorsTensor<'archive> {
    pub fn external_name(&self) -> &'archive str {
        self.external_name
    }

    pub fn source_file(&self) -> &'archive str {
        self.source_file
    }

    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn element_type(&self) -> Option<ElementType> {
        element_type(self.dtype)
    }

    pub fn shape(&self) -> &'archive [u64] {
        self.shape
    }

    pub fn bytes(&self) -> &'archive [u8] {
        self.bytes
    }
}

impl SafetensorsArchive {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let shard_paths = discover_shards(model_dir)?;
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut tensors = BTreeMap::new();
        for (shard_index, (relative_path, path)) in shard_paths.into_iter().enumerate() {
            let file = File::open(&path)
                .map_err(|error| FerrumError::io(format!("open {path:?}: {error}")))?;
            let mmap = unsafe {
                Mmap::map(&file)
                    .map_err(|error| FerrumError::io(format!("mmap {path:?}: {error}")))?
            };
            let parsed = SafeTensors::deserialize(&mmap)
                .map_err(|error| FerrumError::model(format!("parse {path:?}: {error}")))?;
            for name in parsed.names() {
                let view = parsed.tensor(name).map_err(|error| {
                    FerrumError::model(format!("read tensor {name:?} in {path:?}: {error}"))
                })?;
                let data_start = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                let data_end = data_start.checked_add(view.data().len()).ok_or_else(|| {
                    FerrumError::model(format!("tensor {name:?} byte range overflows"))
                })?;
                let metadata = TensorMeta {
                    shard: shard_index,
                    dtype: view.dtype(),
                    shape: view.shape().iter().map(|extent| *extent as u64).collect(),
                    data_start,
                    data_end,
                };
                if tensors.insert(name.to_owned(), metadata).is_some() {
                    return Err(FerrumError::model(format!(
                        "tensor {name:?} appears in multiple safetensors shards"
                    )));
                }
            }
            shards.push(Shard {
                relative_path,
                mmap,
            });
        }
        if tensors.is_empty() {
            return Err(FerrumError::model(
                "safetensors archive contains no tensors",
            ));
        }
        Ok(Self { shards, tensors })
    }

    pub fn tensor(&self, external_name: &str) -> Result<SafetensorsTensor<'_>> {
        let (external_name, metadata) =
            self.tensors.get_key_value(external_name).ok_or_else(|| {
                FerrumError::model(format!(
                    "tensor {external_name:?} is absent from safetensors"
                ))
            })?;
        let shard = &self.shards[metadata.shard];
        let bytes = shard
            .mmap
            .get(metadata.data_start..metadata.data_end)
            .ok_or_else(|| {
                FerrumError::model(format!(
                    "tensor {external_name:?} has an invalid safetensors byte range"
                ))
            })?;
        Ok(SafetensorsTensor {
            external_name,
            source_file: &shard.relative_path,
            dtype: metadata.dtype,
            shape: &metadata.shape,
            bytes,
        })
    }

    pub fn contains(&self, external_name: &str) -> bool {
        self.tensors.contains_key(external_name)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

impl WeightComponentSource for SafetensorsArchive {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let (expected_element_type, affine) = match component.encoding {
            WeightEncoding::Dense { element_type } => (element_type, None),
            WeightEncoding::DenseAffine {
                element_type,
                scale,
                bias,
            } => (element_type, Some((scale, bias))),
            WeightEncoding::Quantized(_) => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "raw safetensors source cannot decode quantized component `{}` without a format adapter",
                        component.id
                    ),
                })
            }
        };

        let [external_name] = component.external_names.as_slice() else {
            if component.external_names.len() < 2
                || component.dimensions.first().copied()
                    != u64::try_from(component.external_names.len()).ok()
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "stacked component `{}` source count differs from its leading dimension",
                        component.id
                    ),
                });
            }
            let source_dimensions = &component.dimensions[1..];
            let mut source_files = Vec::with_capacity(component.external_names.len());
            let mut bytes =
                Vec::with_capacity(usize::try_from(component.physical_bytes()?).map_err(|_| {
                    VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "stacked component `{}` exceeds host address space",
                            component.id
                        ),
                    }
                })?);
            for external_name in &component.external_names {
                let tensor = self.tensor(external_name).map_err(|error| {
                    VNextError::InvalidExecutionPlan {
                        reason: error.to_string(),
                    }
                })?;
                let actual_element_type = element_type(tensor.dtype()).ok_or_else(|| {
                    VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "tensor {external_name:?} has unsupported safetensors dtype {:?}",
                            tensor.dtype()
                        ),
                    }
                })?;
                if tensor.shape() != source_dimensions {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "tensor {external_name:?} shape differs from stacked component `{}` partition shape",
                            component.id
                        ),
                    });
                }
                let materialized = transcode_dense_bytes(
                    tensor.bytes(),
                    actual_element_type,
                    expected_element_type,
                    external_name,
                    affine,
                )?;
                source_files.push(tensor.source_file().to_owned());
                bytes.extend_from_slice(&materialized);
            }
            return WeightComponentPayload::from_ordered_sources(
                component,
                component.external_names.clone(),
                source_files,
                component.dimensions.clone(),
                expected_element_type,
                bytes,
            );
        };

        let tensor =
            self.tensor(external_name)
                .map_err(|error| VNextError::InvalidExecutionPlan {
                    reason: error.to_string(),
                })?;
        let actual_element_type =
            element_type(tensor.dtype()).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} has unsupported safetensors dtype {:?}",
                    tensor.dtype()
                ),
            })?;
        if tensor.shape() != component.dimensions {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} shape differs from component `{}`",
                    component.id
                ),
            });
        }
        let bytes = transcode_dense_bytes(
            tensor.bytes(),
            actual_element_type,
            expected_element_type,
            external_name,
            affine,
        )?;
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            expected_element_type,
            bytes,
        )
    }
}

fn transcode_dense_bytes<'source>(
    bytes: &'source [u8],
    source: ElementType,
    destination: ElementType,
    external_name: &str,
    affine: Option<(CanonicalRational, CanonicalRational)>,
) -> std::result::Result<Cow<'source, [u8]>, VNextError> {
    if source == destination && affine.is_none() {
        return Ok(Cow::Borrowed(bytes));
    }
    let source_float = matches!(
        source,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    );
    let destination_float = matches!(
        destination,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    );
    if !source_float || !destination_float || bytes.len() % source.size_bytes() as usize != 0 {
        return Err(VNextError::InvalidExecutionPlan {
            reason: format!(
                "tensor {external_name:?} cannot be materialized from {source:?} as {destination:?}"
            ),
        });
    }

    let element_count = bytes.len() / source.size_bytes() as usize;
    let mut materialized = Vec::with_capacity(element_count * destination.size_bytes() as usize);
    let affine = affine.map(|(scale, bias)| {
        (
            scale.numerator() as f64 / scale.denominator() as f64,
            bias.numerator() as f64 / bias.denominator() as f64,
        )
    });
    for index in 0..element_count {
        let mut value = read_float(bytes, source, index);
        if let Some((scale, bias)) = affine {
            value = (f64::from(value) * scale + bias) as f32;
            if !value.is_finite() {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "tensor {external_name:?} affine materialization produced a non-finite value at element {index}"
                    ),
                });
            }
        }
        match destination {
            ElementType::F16 => {
                materialized.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
            }
            ElementType::Bf16 => {
                materialized.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            ElementType::F32 => materialized.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!("non-floating destination was rejected above"),
        }
    }
    Ok(Cow::Owned(materialized))
}

fn read_float(bytes: &[u8], element_type: ElementType, index: usize) -> f32 {
    match element_type {
        ElementType::F16 => {
            let offset = index * 2;
            f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
        }
        ElementType::Bf16 => {
            let offset = index * 2;
            bf16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
        }
        ElementType::F32 => {
            let offset = index * 4;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }
        _ => unreachable!("non-floating source was rejected before decoding"),
    }
}

fn discover_shards(model_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let single_name = "model.safetensors";
    let single = model_dir.join(single_name);
    if single.is_file() {
        return Ok(vec![(single_name.to_owned(), single)]);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    let raw = std::fs::read_to_string(&index_path)
        .map_err(|error| FerrumError::io(format!("read {index_path:?}: {error}")))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| FerrumError::serialization(format!("parse {index_path:?}: {error}")))?;
    let weight_map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| FerrumError::model(format!("{index_path:?} missing weight_map")))?;
    let names = weight_map
        .values()
        .map(|value| {
            value
                .as_str()
                .filter(|name| valid_relative_safetensors_path(name))
                .map(str::to_owned)
                .ok_or_else(|| {
                    FerrumError::model(format!(
                        "{index_path:?} contains an invalid safetensors shard path"
                    ))
                })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if names.is_empty() {
        return Err(FerrumError::model(format!(
            "{index_path:?} contains no safetensors shards"
        )));
    }
    names
        .into_iter()
        .map(|name| {
            let path = model_dir.join(&name);
            if !path.is_file() {
                return Err(FerrumError::model(format!(
                    "safetensors shard {path:?} is missing"
                )));
            }
            Ok((name, path))
        })
        .collect()
}

fn valid_relative_safetensors_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.ends_with(".safetensors")
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."))
}

fn element_type(dtype: Dtype) -> Option<ElementType> {
    match dtype {
        Dtype::BOOL => Some(ElementType::Bool),
        Dtype::U8 => Some(ElementType::U8),
        Dtype::U32 => Some(ElementType::U32),
        Dtype::I8 => Some(ElementType::I8),
        Dtype::I32 => Some(ElementType::I32),
        Dtype::F16 => Some(ElementType::F16),
        Dtype::BF16 => Some(ElementType::Bf16),
        Dtype::F32 => Some(ElementType::F32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_materialization_converts_bf16_and_f32_to_f16() {
        let bf16_bytes = [
            bf16::from_f32(1.5).to_bits().to_le_bytes(),
            bf16::from_f32(-2.25).to_bits().to_le_bytes(),
        ]
        .concat();
        let converted = transcode_dense_bytes(
            &bf16_bytes,
            ElementType::Bf16,
            ElementType::F16,
            "weight",
            None,
        )
        .unwrap();
        let expected = [
            f16::from_f32(1.5).to_bits().to_le_bytes(),
            f16::from_f32(-2.25).to_bits().to_le_bytes(),
        ]
        .concat();
        assert_eq!(converted.as_ref(), expected);

        let f32_bytes = [1.5_f32.to_le_bytes(), (-2.25_f32).to_le_bytes()].concat();
        let converted = transcode_dense_bytes(
            &f32_bytes,
            ElementType::F32,
            ElementType::F16,
            "weight",
            None,
        )
        .unwrap();
        assert_eq!(converted.as_ref(), expected);
    }

    #[test]
    fn dense_materialization_borrows_matching_storage() {
        let bytes = f16::from_f32(1.0).to_bits().to_le_bytes();
        let converted =
            transcode_dense_bytes(&bytes, ElementType::F16, ElementType::F16, "weight", None)
                .unwrap();
        assert!(matches!(converted, Cow::Borrowed(_)));
    }

    #[test]
    fn affine_dense_materialization_applies_logical_transform() {
        let bytes = [(-0.5_f32).to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
        let converted = transcode_dense_bytes(
            &bytes,
            ElementType::F32,
            ElementType::F16,
            "norm.weight",
            Some((
                CanonicalRational::new(1, 1).unwrap(),
                CanonicalRational::new(1, 1).unwrap(),
            )),
        )
        .unwrap();
        let actual = converted
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(actual, [0.5, 3.0]);
        assert!(matches!(converted, Cow::Owned(_)));
    }

    #[test]
    fn affine_dense_materialization_rejects_non_finite_results() {
        let bytes = f32::MAX.to_le_bytes();
        let error = transcode_dense_bytes(
            &bytes,
            ElementType::F32,
            ElementType::F32,
            "weight",
            Some((
                CanonicalRational::new(i64::MAX, 1).unwrap(),
                CanonicalRational::new(0, 1).unwrap(),
            )),
        )
        .expect_err("overflowing affine values must not enter device storage");
        assert!(error.to_string().contains("non-finite"), "{error}");
    }
}
