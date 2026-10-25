//! Immutable, model-independent tensor inventory over stored weight payloads.
//!
//! Archive metadata describes storage, not a device's numerical capabilities.
//! Logical dimensions remain in element units even for packed blocks; runtime
//! component schemas separately describe the physical block dimensions.

use ferrum_runtime::{
    ElementType, RetainedHostMemoryRegion, WeightComponentSource, WeightEncoding,
};
use ferrum_types::{FerrumError, Result};
mod managed;
pub use managed::ManagedDenseArchiveSource;

/// An indexed archive whose payloads can also satisfy an explicit runtime
/// component schema. Implementations retain immutable storage; inspecting a
/// tensor must not dequantize it or allocate a copy of its payload.
pub trait TensorArchive: WeightComponentSource {
    fn tensor(&self, external_name: &str) -> Result<ArchiveTensor<'_>>;
    fn tensor_count(&self) -> usize;
    fn tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_>;
}

/// A checked view of one stored tensor. The retained region can outlive both
/// this metadata view and the archive, including for asynchronous uploads.
pub struct ArchiveTensor<'archive> {
    external_name: &'archive str,
    source_file: &'archive str,
    shape: &'archive [u64],
    encoding: WeightEncoding,
    retained_host_memory: RetainedHostMemoryRegion,
}

impl<'archive> ArchiveTensor<'archive> {
    pub fn new(
        external_name: &'archive str,
        source_file: &'archive str,
        shape: &'archive [u64],
        encoding: WeightEncoding,
        retained_host_memory: RetainedHostMemoryRegion,
    ) -> Result<Self> {
        let invalid =
            |reason: &str| FerrumError::model(format!("stored tensor {external_name:?}: {reason}"));
        let elements = shape
            .iter()
            .try_fold(1_u64, |n, &d| (d > 0).then(|| n.checked_mul(d)).flatten())
            .ok_or_else(|| invalid("logical dimensions contain zero or overflow"))?;
        let bytes = match &encoding {
            WeightEncoding::Dense { element_type } => {
                elements.checked_mul(element_type.size_bytes())
            }
            WeightEncoding::BlockQuantized(spec) => {
                spec.validate().map_err(|e| invalid(&e.to_string()))?;
                let block = u64::from(spec.logical_values_per_block);
                if !shape.last().is_some_and(|d| d.is_multiple_of(block)) {
                    return Err(invalid(
                        "innermost dimension does not contain complete blocks",
                    ));
                }
                (elements / block).checked_mul(u64::from(spec.bytes_per_block))
            }
            WeightEncoding::DenseAffine { .. } | WeightEncoding::Quantized(_) => {
                return Err(invalid(
                    "encoding requires a schema-specific format adapter",
                ));
            }
        }
        .ok_or_else(|| invalid("payload byte size overflows"))?;
        if u64::try_from(retained_host_memory.length_bytes()).ok() != Some(bytes) {
            return Err(invalid(
                "payload length differs from shape and storage encoding",
            ));
        }
        Ok(Self {
            external_name,
            source_file,
            shape,
            encoding,
            retained_host_memory,
        })
    }

    pub fn external_name(&self) -> &'archive str {
        self.external_name
    }
    pub fn source_file(&self) -> &'archive str {
        self.source_file
    }
    pub fn shape(&self) -> &'archive [u64] {
        self.shape
    }
    pub fn encoding(&self) -> &WeightEncoding {
        &self.encoding
    }
    pub fn element_type(&self) -> Option<ElementType> {
        self.encoding.dense_element_type()
    }
    pub fn bytes(&self) -> &[u8] {
        self.retained_host_memory.bytes()
    }
    pub fn retained_host_memory(&self) -> &RetainedHostMemoryRegion {
        &self.retained_host_memory
    }
}
