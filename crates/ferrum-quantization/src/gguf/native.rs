//! vNext physical GGUF source. The validated native descriptor table retains
//! file type IDs, including IQ types absent from Candle's QTensor enum. Model
//! semantics and tokenizer metadata remain in their independently bound sources.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use candle_core::{Error, Result};
use ferrum_interfaces::vnext::{BlockQuantizationSpec, ElementType, WeightEncoding};
use memmap2::Mmap;

use super::inventory::{block_abi, GgufInventory};

#[derive(Debug, Clone)]
pub struct NativeGgufTensor {
    pub ggml_type: u32,
    pub dimensions: Vec<u64>,
    pub encoding: WeightEncoding,
    pub elements: u64,
    offset: usize,
    bytes: usize,
}

/// Describes physical bytes only. An accepted encoding does not imply that an
/// operation provider exists for any particular device or model tensor role.
pub fn gguf_weight_encoding(ggml_type: u32) -> Result<WeightEncoding> {
    let abi = block_abi(ggml_type)?;
    if let Some(format) = abi.format {
        let spec = BlockQuantizationSpec {
            format_id: format.to_owned().try_into().map_err(Error::wrap)?,
            logical_values_per_block: u32::try_from(abi.values).map_err(Error::wrap)?,
            bytes_per_block: u32::try_from(abi.bytes).map_err(Error::wrap)?,
        };
        spec.validate().map_err(Error::wrap)?;
        Ok(WeightEncoding::BlockQuantized(spec))
    } else {
        let element_type = match ggml_type {
            0 => ElementType::F32,
            1 => ElementType::F16,
            30 => ElementType::Bf16,
            _ => {
                return Err(Error::Msg(format!(
                    "GGUF type {ggml_type} has no declared dense encoding"
                )))
            }
        };
        Ok(WeightEncoding::Dense { element_type })
    }
}

#[derive(Debug)]
pub struct NativeGgufFile {
    mmap: Mmap,
    architecture: String,
    quantization_version: Option<u64>,
    tensors: BTreeMap<String, NativeGgufTensor>,
}

impl NativeGgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(Error::wrap)?;
        // SAFETY: The caller keeps the model artifact immutable while it is
        // loaded. This handle exposes only a read-only file mapping.
        let mmap = unsafe { Mmap::map(&file) }.map_err(Error::wrap)?;
        let inventory = GgufInventory::read(&mut Cursor::new(&mmap[..]), mmap.len() as u64)?;
        if inventory
            .split
            .as_ref()
            .is_some_and(|split| split.count != 1)
        {
            return Err(Error::Msg("a GGUF shard cannot be loaded as a complete weight artifact; all declared shards are required".into()));
        }
        let tensors = inventory
            .tensors
            .into_iter()
            .map(|tensor| {
                let offset = usize::try_from(tensor.absolute_offset).map_err(Error::wrap)?;
                let bytes = usize::try_from(tensor.bytes).map_err(Error::wrap)?;
                let encoding = gguf_weight_encoding(tensor.ggml_type)?;
                Ok((
                    tensor.name,
                    NativeGgufTensor {
                        ggml_type: tensor.ggml_type,
                        dimensions: tensor.dimensions,
                        encoding,
                        elements: tensor.elements,
                        offset,
                        bytes,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            mmap,
            architecture: inventory.architecture,
            quantization_version: inventory.quantization_version,
            tensors,
        })
    }

    pub fn architecture(&self) -> Result<&str> {
        Ok(&self.architecture)
    }
    pub fn quantization_version(&self) -> Option<u64> {
        self.quantization_version
    }
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn tensor_info(&self, name: &str) -> Option<&NativeGgufTensor> {
        self.tensors.get(name)
    }
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn mmap_bytes(&self) -> &[u8] {
        &self.mmap
    }

    pub fn tensor_byte_slice(&self, name: &str) -> Option<&[u8]> {
        let (offset, bytes) = self.tensor_byte_range(name)?;
        self.mmap.get(offset..offset.checked_add(bytes)?)
    }

    pub fn tensor_byte_range(&self, name: &str) -> Option<(usize, usize)> {
        let info = self.tensor_info(name)?;
        Some((info.offset, info.bytes))
    }
}
