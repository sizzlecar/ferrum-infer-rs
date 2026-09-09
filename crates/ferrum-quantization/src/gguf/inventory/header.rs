//! Descriptor-only parsing preserves GGML type IDs unsupported by Candle.
//! Tokenizer arrays are skipped without allocating their entries. The runtime
//! reader remains responsible for decoding and executing supported tensors.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file::Value;
use candle_core::{Error, Result};

pub(super) struct Header {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorHeader>,
    pub data_offset: u64,
}

pub(super) struct TensorHeader {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

struct Reader<'a, R> {
    input: &'a mut R,
    end: u64,
    version: u32,
}

fn invalid(reason: &str) -> Error {
    Error::Msg(format!("invalid GGUF header: {reason}"))
}

impl Header {
    pub fn read<R: Read + Seek>(input: &mut R) -> Result<Self> {
        let start = input.stream_position()?;
        let end = input.seek(SeekFrom::End(0))?;
        input.seek(SeekFrom::Start(start))?;
        let mut reader = Reader {
            input,
            end,
            version: 0,
        };
        if &reader.bytes::<4>()? != b"GGUF" {
            return Err(invalid("expected a little-endian GGUF file"));
        }
        reader.version = reader.u32()?;
        if !(1..=3).contains(&reader.version) {
            return Err(invalid("unsupported file version"));
        }
        let tensor_count = reader.length()?;
        let metadata_count = reader.length()?;
        if tensor_count > reader.remaining()? / 24 || metadata_count > reader.remaining()? / 8 {
            return Err(invalid("entry counts exceed the available header"));
        }
        let mut metadata = BTreeMap::new();
        let mut keys = BTreeSet::new();
        for _ in 0..metadata_count {
            let key = reader.string()?;
            if !keys.insert(key.clone()) {
                return Err(invalid("duplicate metadata key"));
            }
            let kind = reader.u32()?;
            if matches!(
                key.as_str(),
                "general.architecture"
                    | "general.alignment"
                    | "general.quantization_version"
                    | "split.no"
                    | "split.count"
                    | "split.tensors.count"
            ) {
                metadata.insert(key, reader.scalar(kind)?);
            } else {
                reader.skip_value(kind, 0)?;
            }
        }
        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = reader.string()?;
            if name.is_empty() || !names.insert(name.clone()) {
                return Err(invalid("empty or duplicate tensor name"));
            }
            let rank = reader.u32()?;
            // GGML_MAX_DIMS is part of the GGUF tensor descriptor contract.
            if !(1..=4).contains(&rank) {
                return Err(invalid("tensor rank must be between one and four"));
            }
            let mut dimensions = (0..rank)
                .map(|_| reader.length())
                .collect::<Result<Vec<_>>>()?;
            dimensions.reverse();
            let dtype = reader.u32()?;
            let offset = reader.u64()?;
            tensors.push(TensorHeader {
                name,
                dimensions,
                dtype,
                offset,
            });
        }
        let alignment = metadata
            .get("general.alignment")
            .map(super::metadata_integer)
            .transpose()?
            .unwrap_or(32);
        // The GGUF format permits any nonzero multiple of eight, not only
        // power-of-two alignments (docs/gguf.md, general.alignment).
        if alignment == 0 || !alignment.is_multiple_of(8) {
            return Err(invalid("alignment must be a nonzero multiple of eight"));
        }
        let position = reader.input.stream_position()?;
        let padded = position
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("aligned data offset overflows u64"))?;
        Ok(Self {
            metadata,
            tensors,
            data_offset: padded / alignment * alignment,
        })
    }
}

impl<R: Read + Seek> Reader<'_, R> {
    fn remaining(&mut self) -> Result<u64> {
        self.end
            .checked_sub(self.input.stream_position()?)
            .ok_or_else(|| invalid("read position exceeds the input length"))
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        if N as u64 > self.remaining()? {
            return Err(invalid("truncated descriptor or metadata value"));
        }
        let mut bytes = [0; N];
        self.input.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn length(&mut self) -> Result<u64> {
        if self.version == 1 {
            Ok(self.u32()?.into())
        } else {
            self.u64()
        }
    }

    fn string(&mut self) -> Result<String> {
        let len = self.length()?;
        // This is a memory bound for retained keys/names/scalars. Ignored
        // metadata strings use skip_value and allocate no text buffer.
        if len > self.remaining()? || len > 1024 * 1024 {
            return Err(invalid("retained string is truncated or exceeds 1 MiB"));
        }
        let mut bytes = vec![0; len as usize];
        self.input.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(Error::wrap)
    }

    fn scalar(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(i8::from_le_bytes(self.bytes()?)),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(invalid("invalid boolean")),
            },
            8 => Value::String(self.string()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(invalid("selected metadata must be a known scalar")),
        })
    }

    fn skip(&mut self, count: u64) -> Result<()> {
        if count > self.remaining()? {
            return Err(invalid("truncated metadata"));
        }
        let end = self
            .input
            .stream_position()?
            .checked_add(count)
            .ok_or_else(|| invalid("metadata end overflows u64"))?;
        self.input.seek(SeekFrom::Start(end))?;
        Ok(())
    }

    fn skip_value(&mut self, kind: u32, depth: usize) -> Result<()> {
        if depth > 16 {
            return Err(invalid(
                "metadata array nesting exceeds the reader stack budget",
            ));
        }
        match kind {
            8 => {
                let length = self.length()?;
                self.skip(length)
            }
            9 => {
                let element_type = self.u32()?;
                let count = self.length()?;
                if let Some(size) = scalar_bytes(element_type) {
                    let bytes = count
                        .checked_mul(size)
                        .ok_or_else(|| invalid("metadata array bytes overflow u64"))?;
                    self.skip(bytes)
                } else if matches!(element_type, 8 | 9) {
                    if count > self.remaining()? / if self.version == 1 { 4 } else { 8 } {
                        return Err(invalid("metadata array count exceeds available bytes"));
                    }
                    for _ in 0..count {
                        self.skip_value(element_type, depth + 1)?;
                    }
                    Ok(())
                } else {
                    Err(invalid("unknown metadata array element type"))
                }
            }
            _ => self.skip(scalar_bytes(kind).ok_or_else(|| invalid("unknown metadata type"))?),
        }
    }
}

fn scalar_bytes(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}
