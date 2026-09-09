//! Deterministic GGUF tensor inventory without materializing weight payloads.
//!
//! The declared file length may come from an immutable remote artifact when
//! only its header has been fetched. Bounds checked against that length do not
//! prove that the payload was downloaded, hashed, or executed.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek};

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{Error, Result};
use serde::Serialize;

use super::{block_quantization_format, GgmlDType};

mod header;
mod metadata;
use header::{Header, TensorHeader};
pub use metadata::GgufModelMetadata;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufTensorInventory {
    pub name: String,
    pub dtype: String,
    pub ggml_type: u32,
    /// Descriptor recognition does not imply a production provider exists.
    pub candle_dtype_available: bool,
    pub quantization_format: Option<String>,
    /// Logical row-major dimensions, as consumed by the Ferrum GGUF reader.
    pub dimensions: Vec<u64>,
    pub block_axis: usize,
    pub logical_values_per_block: u64,
    pub bytes_per_block: u64,
    pub elements: u64,
    pub absolute_offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufSplitInventory {
    pub index: u64,
    pub count: u64,
    pub total_tensors: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufInventory {
    pub schema_version: u32,
    pub architecture: String,
    pub quantization_version: Option<u64>,
    pub declared_file_bytes: u64,
    pub tensor_data_offset: u64,
    pub tensor_payload_bytes: u64,
    pub split: Option<GgufSplitInventory>,
    pub tensor_counts_by_dtype: BTreeMap<String, usize>,
    /// Sorted by external tensor name, independently of parser map order.
    pub tensors: Vec<GgufTensorInventory>,
}

impl GgufInventory {
    /// Read a complete descriptor table from a full file or a header prefix.
    /// Tensor payload bytes are never read by this method.
    pub fn read<R: Read + Seek>(reader: &mut R, declared_file_bytes: u64) -> Result<Self> {
        Self::from_header(Header::read(reader)?, declared_file_bytes)
    }

    pub fn from_content(content: &Content, declared_file_bytes: u64) -> Result<Self> {
        let header = Header {
            metadata: content
                .metadata
                .iter()
                .filter(|(key, _)| {
                    matches!(
                        key.as_str(),
                        "general.architecture"
                            | "general.alignment"
                            | "general.quantization_version"
                            | "split.no"
                            | "split.count"
                            | "split.tensors.count"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            tensors: content
                .tensor_infos
                .iter()
                .map(|(name, info)| {
                    Ok(TensorHeader {
                        name: name.clone(),
                        dimensions: info.shape.dims().iter().map(|&n| n as u64).collect(),
                        dtype: CANDLE_DTYPES
                            .iter()
                            .find(|(_, dtype)| *dtype == info.ggml_dtype)
                            .ok_or_else(|| {
                                Error::Msg(format!(
                                    "missing GGML file ID for {:?}",
                                    info.ggml_dtype
                                ))
                            })?
                            .0,
                        offset: info.offset,
                    })
                })
                .collect::<Result<_>>()?,
            data_offset: content.tensor_data_offset,
        };
        Self::from_header(header, declared_file_bytes)
    }

    fn from_header(content: Header, declared_file_bytes: u64) -> Result<Self> {
        let invalid = |reason: String| Error::Msg(format!("invalid GGUF inventory: {reason}"));
        let unsupported: BTreeSet<_> = content
            .tensors
            .iter()
            .filter(|tensor| block_abi(tensor.dtype).is_err())
            .map(|tensor| tensor.dtype)
            .collect();
        if !unsupported.is_empty() {
            return Err(invalid(format!(
                "missing block ABIs for GGML types {unsupported:?}"
            )));
        }
        let architecture = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| invalid("missing general.architecture".into()))?
            .to_string()?
            .to_owned();
        if architecture.is_empty() || content.tensors.is_empty() {
            return Err(invalid(
                "architecture and tensor table must be nonempty".into(),
            ));
        }
        if content.data_offset > declared_file_bytes {
            return Err(invalid(
                "tensor data starts beyond the declared file length".into(),
            ));
        }
        let alignment = match content.metadata.get("general.alignment") {
            Some(value) => metadata_integer(value)?,
            None => 32,
        };
        if alignment == 0 || !alignment.is_multiple_of(8) {
            return Err(invalid(
                "alignment must be a nonzero multiple of eight".into(),
            ));
        }
        if !content.data_offset.is_multiple_of(alignment) {
            return Err(invalid("tensor data offset is misaligned".into()));
        }
        let mut tensors = Vec::with_capacity(content.tensors.len());
        let mut tensor_counts_by_dtype = BTreeMap::new();
        let mut tensor_payload_bytes = 0_u64;
        for info in &content.tensors {
            let name = &info.name;
            let dimensions = info.dimensions.clone();
            if name.is_empty() || !(1..=4).contains(&dimensions.len()) || dimensions.contains(&0) {
                return Err(invalid(format!(
                    "{name:?} needs a nonempty name and one to four nonzero dimensions"
                )));
            }
            let block_axis = dimensions.len() - 1;
            let abi = block_abi(info.dtype)?;
            let logical_values_per_block = abi.values;
            let bytes_per_block = abi.bytes;
            if !dimensions[block_axis].is_multiple_of(logical_values_per_block) {
                return Err(invalid(format!(
                    "{name:?} has an incomplete row quantization block"
                )));
            }
            let elements = dimensions
                .iter()
                .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| invalid(format!("{name:?} element count overflows u64")))?;
            let bytes = (elements / logical_values_per_block)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| invalid(format!("{name:?} payload size overflows u64")))?;
            let absolute_offset = content
                .data_offset
                .checked_add(info.offset)
                .ok_or_else(|| invalid(format!("{name:?} absolute offset overflows u64")))?;
            let end = absolute_offset
                .checked_add(bytes)
                .ok_or_else(|| invalid(format!("{name:?} payload end overflows u64")))?;
            if !info.offset.is_multiple_of(alignment) || end > declared_file_bytes {
                return Err(invalid(format!(
                    "{name:?} is misaligned or extends beyond the declared file length"
                )));
            }
            tensor_payload_bytes = tensor_payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("total tensor bytes overflow u64".into()))?;
            let dtype = abi.name;
            *tensor_counts_by_dtype.entry(dtype.clone()).or_insert(0) += 1;
            tensors.push(GgufTensorInventory {
                name: name.clone(),
                dtype,
                ggml_type: info.dtype,
                candle_dtype_available: abi.candle_dtype_available,
                quantization_format: abi.format.map(str::to_owned),
                dimensions,
                block_axis,
                logical_values_per_block,
                bytes_per_block,
                elements,
                absolute_offset,
                bytes,
            });
        }
        tensors.sort_by_key(|tensor| tensor.absolute_offset);
        for pair in tensors.windows(2) {
            if pair[0].absolute_offset + pair[0].bytes > pair[1].absolute_offset {
                return Err(invalid(format!(
                    "tensor payloads {:?} and {:?} overlap",
                    pair[0].name, pair[1].name
                )));
            }
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let split = match (
            content.metadata.get("split.no"),
            content.metadata.get("split.count"),
            content.metadata.get("split.tensors.count"),
        ) {
            (None, None, None) => None,
            (Some(index), Some(count), Some(total)) => {
                let split = GgufSplitInventory {
                    index: metadata_integer(index)?,
                    count: metadata_integer(count)?,
                    total_tensors: metadata_integer(total)?,
                };
                if split.count == 0
                    || split.index >= split.count
                    || split.total_tensors < tensors.len() as u64
                {
                    return Err(invalid(
                        "inconsistent split index, count, or tensor total".into(),
                    ));
                }
                Some(split)
            }
            _ => return Err(invalid("incomplete split metadata".into())),
        };
        Ok(Self {
            schema_version: 1,
            architecture,
            quantization_version: content
                .metadata
                .get("general.quantization_version")
                .map(metadata_integer)
                .transpose()?,
            declared_file_bytes,
            tensor_data_offset: content.data_offset,
            tensor_payload_bytes,
            split,
            tensor_counts_by_dtype,
            tensors,
        })
    }
}

const CANDLE_DTYPES: &[(u32, GgmlDType)] = &[
    (0, GgmlDType::F32),
    (1, GgmlDType::F16),
    (2, GgmlDType::Q4_0),
    (3, GgmlDType::Q4_1),
    (6, GgmlDType::Q5_0),
    (7, GgmlDType::Q5_1),
    (8, GgmlDType::Q8_0),
    (9, GgmlDType::Q8_1),
    (10, GgmlDType::Q2K),
    (11, GgmlDType::Q3K),
    (12, GgmlDType::Q4K),
    (13, GgmlDType::Q5K),
    (14, GgmlDType::Q6K),
    (15, GgmlDType::Q8K),
    (30, GgmlDType::BF16),
];

pub(super) struct BlockAbi {
    name: String,
    pub(super) format: Option<&'static str>,
    pub(super) values: u64,
    pub(super) bytes: u64,
    candle_dtype_available: bool,
}

pub(super) fn block_abi(code: u32) -> Result<BlockAbi> {
    if let Some((_, dtype)) = CANDLE_DTYPES.iter().find(|(id, _)| *id == code) {
        return Ok(BlockAbi {
            name: format!("{dtype:?}"),
            format: block_quantization_format(*dtype),
            values: dtype.block_size() as u64,
            bytes: dtype.type_size() as u64,
            candle_dtype_available: true,
        });
    }
    // GGML_TYPE_IQ4_XS = 23. block_iq4_xs encodes 256 weights in
    // 2-byte d + 2-byte scales_h + 4-byte scales_l + 128-byte quants.
    // https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
    match code {
        23 => {
            return Ok(BlockAbi {
                name: "IQ4_XS".into(),
                format: Some("quantization.gguf.iq4-xs"),
                values: 256,
                bytes: 136,
                candle_dtype_available: false,
            })
        }
        // block_iq4_nl stores one F16 scale and 16 packed bytes for 32 weights.
        20 => {
            return Ok(BlockAbi {
                name: "IQ4_NL".into(),
                format: Some("quantization.gguf.iq4-nl"),
                values: 32,
                bytes: 18,
                candle_dtype_available: false,
            })
        }
        // block_iq3_s: F16 d, 64-byte qs, 8-byte qh, 32-byte signs,
        // and four sub-block scales for each 256-weight superblock.
        21 => {
            return Ok(BlockAbi {
                name: "IQ3_S".into(),
                format: Some("quantization.gguf.iq3-s"),
                values: 256,
                bytes: 110,
                candle_dtype_available: false,
            })
        }
        _ => {}
    }
    Err(Error::Msg(format!(
        "GGUF inventory has no block ABI for GGML type {code}"
    )))
}

fn metadata_integer(value: &Value) -> Result<u64> {
    match value {
        Value::U8(value) => Ok((*value).into()),
        Value::U16(value) => Ok((*value).into()),
        Value::U32(value) => Ok((*value).into()),
        Value::U64(value) => Ok(*value),
        Value::I8(value) => u64::try_from(*value).map_err(Error::wrap),
        Value::I16(value) => u64::try_from(*value).map_err(Error::wrap),
        Value::I32(value) => u64::try_from(*value).map_err(Error::wrap),
        Value::I64(value) => u64::try_from(*value).map_err(Error::wrap),
        _ => Err(Error::Msg(
            "GGUF inventory metadata must be an integer".into(),
        )),
    }
}
