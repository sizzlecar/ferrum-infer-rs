//! Bounded metadata planning for the streamed anonymous safetensors file.
use super::*;
use ferrum_runtime::{ElementType, WeightEncoding};
use safetensors::{Dtype, SafeTensors};
use serde::ser::SerializeMap;
use std::{io, ops::Range};

pub(super) struct Entry {
    pub spec: WeightComponentSpec,
    pub relative: Range<usize>,
}
pub(super) struct Layout {
    pub entries: Vec<Entry>,
    pub lookup_order: Vec<usize>,
    pub write_order: Vec<usize>,
    pub header: Vec<u8>,
    pub requirements: ProcessMappedDenseRequirements,
}

fn dtype(spec: &WeightComponentSpec) -> Result<Dtype, VNextError> {
    let element = match spec.encoding {
        WeightEncoding::Dense { element_type }
        | WeightEncoding::DenseAffine { element_type, .. } => element_type,
        _ => {
            return Err(invalid(
                "only declared dense floating components may be normalized",
            ))
        }
    };
    match element {
        ElementType::F16 => Ok(Dtype::F16),
        ElementType::Bf16 => Ok(Dtype::BF16),
        ElementType::F32 => Ok(Dtype::F32),
        _ => Err(invalid("normalized target must be F16/BF16/F32")),
    }
}

impl Layout {
    pub fn new(
        components: &[WeightComponentSpec],
        limits: ProcessMappedDenseLimits,
    ) -> Result<Self, VNextError> {
        limits.validate()?;
        if components.is_empty() {
            return Err(invalid("normalization needs at least one component"));
        }
        let fixed = components
            .len()
            .checked_mul(
                std::mem::size_of::<Entry>()
                    + 2 * std::mem::size_of::<usize>()
                    + std::mem::size_of::<Vec<String>>(),
            )
            .ok_or_else(|| invalid("component index size overflows"))?;
        let mut metadata_bytes = fixed
            .checked_add(encoded_size(components, limits.maximum_metadata_bytes)?)
            .ok_or_else(|| invalid("component metadata size overflows"))?;
        if metadata_bytes > limits.maximum_metadata_bytes {
            return Err(invalid("component metadata exceeds limit"));
        }
        let mut entries = Vec::with_capacity(components.len());
        let mut payload_bytes = 0u64;
        let mut maximum_component_bytes = 0u64;
        for spec in components {
            dtype(spec)?;
            if spec.id.as_str() == "__metadata__"
                || spec.external_names.is_empty()
                || spec.dimensions.is_empty()
                || spec.dimensions.contains(&0)
            {
                return Err(invalid(
                    "component needs exact source names and positive dimensions",
                ));
            }
            let bytes = spec.physical_bytes()?;
            payload_bytes = payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("file payload size overflows"))?;
            maximum_component_bytes = maximum_component_bytes.max(bytes);
            entries.push(Entry {
                spec: spec.clone(),
                relative: 0..0,
            });
        }
        let mut lookup_order: Vec<_> = (0..entries.len()).collect();
        lookup_order.sort_by(|&a, &b| entries[a].spec.id.cmp(&entries[b].spec.id));
        if lookup_order
            .windows(2)
            .any(|pair| entries[pair[0]].spec.id == entries[pair[1]].spec.id)
        {
            return Err(invalid("duplicate normalized component ID"));
        }
        // Match safetensors' dtype alignment ordering, then deterministic IDs.
        let mut write_order = lookup_order.clone();
        write_order.sort_by(|&a, &b| {
            dtype(&entries[b].spec)
                .unwrap()
                .cmp(&dtype(&entries[a].spec).unwrap())
                .then(entries[a].spec.id.cmp(&entries[b].spec.id))
        });
        let mut offset = 0usize;
        for &index in &write_order {
            let bytes = usize::try_from(entries[index].spec.physical_bytes()?)
                .map_err(|_| invalid("component exceeds address space"))?;
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| invalid("file offset overflows"))?;
            entries[index].relative = offset..end;
            offset = end;
        }
        let remaining = limits.maximum_metadata_bytes - metadata_bytes;
        let mut writer = BoundedWriter {
            bytes: Vec::new(),
            limit: remaining,
        };
        serde_json::to_writer(&mut writer, &Header(&entries)).map_err(invalid)?;
        let padded = writer
            .bytes
            .len()
            .checked_add(7)
            .map(|n| n / 8 * 8)
            .ok_or_else(|| invalid("header padding overflows"))?;
        if padded > remaining {
            return Err(invalid("padded header exceeds metadata limit"));
        }
        writer.bytes.resize(padded, b' ');
        metadata_bytes += padded;
        let file_bytes = payload_bytes
            .checked_add(8)
            .and_then(|n| n.checked_add(padded as u64))
            .ok_or_else(|| invalid("file size overflows"))?;
        if file_bytes > limits.maximum_file_bytes || file_bytes > isize::MAX as u64 {
            return Err(invalid("normalized file exceeds file limit/address space"));
        }
        Ok(Self {
            entries,
            lookup_order,
            write_order,
            header: writer.bytes,
            requirements: ProcessMappedDenseRequirements {
                file_bytes,
                payload_bytes,
                metadata_bytes,
                maximum_component_bytes,
            },
        })
    }

    pub fn data_range(&self, entry: &Entry) -> Result<Range<usize>, VNextError> {
        let start = self
            .header
            .len()
            .checked_add(8)
            .and_then(|n| n.checked_add(entry.relative.start))
            .ok_or_else(|| invalid("mapped offset overflows"))?;
        let end = start
            .checked_add(entry.relative.len())
            .ok_or_else(|| invalid("mapped extent overflows"))?;
        Ok(start..end)
    }

    pub fn validate_mapping(&self, bytes: &[u8]) -> Result<(), VNextError> {
        let tensors = SafeTensors::deserialize(bytes).map_err(invalid)?;
        if tensors.len() != self.entries.len() {
            return Err(invalid("mapped tensor set differs"));
        }
        for entry in &self.entries {
            let tensor = tensors.tensor(entry.spec.id.as_str()).map_err(invalid)?;
            if tensor.dtype() != dtype(&entry.spec)?
                || tensor.shape().iter().map(|&n| n as u64).ne(entry
                    .spec
                    .dimensions
                    .iter()
                    .copied())
                || tensor.data().len() != entry.relative.len()
                || !std::ptr::eq(tensor.data(), &bytes[self.data_range(entry)?])
            {
                return Err(invalid(
                    "mapped tensor differs from its exact normalized layout",
                ));
            }
        }
        Ok(())
    }
}

struct Header<'a>(&'a [Entry]);
impl Serialize for Header<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct TensorInfo<'a> {
            dtype: Dtype,
            shape: &'a [u64],
            data_offsets: [usize; 2],
        }
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for entry in self.0 {
            let info = TensorInfo {
                dtype: dtype(&entry.spec).map_err(serde::ser::Error::custom)?,
                shape: &entry.spec.dimensions,
                data_offsets: [entry.relative.start, entry.relative.end],
            };
            map.serialize_entry(entry.spec.id.as_str(), &info)?;
        }
        map.end()
    }
}

pub(super) fn encoded_size<T: Serialize + ?Sized>(
    value: &T,
    limit: usize,
) -> Result<usize, VNextError> {
    struct Counter {
        bytes: usize,
        limit: usize,
    }
    impl Write for Counter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.bytes = self
                .bytes
                .checked_add(data.len())
                .filter(|&n| n <= self.limit)
                .ok_or_else(|| io::Error::other("metadata exceeds limit"))?;
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter { bytes: 0, limit };
    serde_json::to_writer(&mut counter, value).map_err(invalid)?;
    Ok(counter.bytes)
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(data.len())
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| io::Error::other("header exceeds metadata limit"))?;
        self.bytes
            .try_reserve_exact(length - self.bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
