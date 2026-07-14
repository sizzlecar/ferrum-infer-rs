use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use ferrum_interfaces::vnext::{
    ElementType, VNextError, WeightComponentPayload, WeightComponentSource, WeightComponentSpec,
    WeightEncoding,
};
use ferrum_types::{FerrumError, Result};
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
}

impl WeightComponentSource for SafetensorsArchive {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let matches = component
            .external_names
            .iter()
            .filter(|name| self.contains(name))
            .collect::<Vec<_>>();
        let external_name = match matches.as_slice() {
            [name] => name.as_str(),
            [] => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "weight component `{}` has no matching safetensors source",
                        component.id
                    ),
                })
            }
            _ => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "weight component `{}` matches multiple safetensors aliases",
                        component.id
                    ),
                })
            }
        };
        let tensor =
            self.tensor(external_name)
                .map_err(|error| VNextError::InvalidExecutionPlan {
                    reason: error.to_string(),
                })?;
        let expected_element_type = match component.encoding {
            WeightEncoding::Dense { element_type } => element_type,
            WeightEncoding::Quantized(_) => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "raw safetensors source cannot decode quantized component `{}` without a format adapter",
                        component.id
                    ),
                })
            }
        };
        let actual_element_type =
            element_type(tensor.dtype()).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} has unsupported safetensors dtype {:?}",
                    tensor.dtype()
                ),
            })?;
        if actual_element_type != expected_element_type || tensor.shape() != component.dimensions {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} dtype or shape differs from component `{}`",
                    component.id
                ),
            });
        }
        WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            expected_element_type,
            tensor.bytes(),
        )
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
