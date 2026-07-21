use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use super::{ElementType, VNextError, WeightComponentSpec, WeightId};

/// Owner for host bytes whose address, length, and contents remain stable for
/// the lifetime of the owner.
///
/// # Safety
///
/// Implementations must return the same readable allocation from
/// [`Self::stable_bytes`] for their entire lifetime. The allocation must not be
/// mutated while a retained region exists. Device backends may keep a native
/// no-copy view after the source object that produced a payload has been
/// dropped.
pub unsafe trait StableHostMemory: Send + Sync + 'static {
    fn stable_bytes(&self) -> &[u8];
}

/// An owned, bounds-checked subregion of stable host memory.
#[derive(Clone)]
pub struct RetainedHostMemoryRegion {
    owner: Arc<dyn StableHostMemory>,
    offset_bytes: usize,
    length_bytes: usize,
}

impl RetainedHostMemoryRegion {
    pub fn new<T>(
        owner: Arc<T>,
        offset_bytes: usize,
        length_bytes: usize,
    ) -> Result<Self, VNextError>
    where
        T: StableHostMemory,
    {
        let end = offset_bytes.checked_add(length_bytes).ok_or_else(|| {
            VNextError::InvalidExecutionPlan {
                reason: "retained host-memory range overflows the host address space".to_owned(),
            }
        })?;
        if length_bytes == 0 || end > owner.stable_bytes().len() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "retained host-memory range is empty or exceeds its owner".to_owned(),
            });
        }
        Ok(Self {
            owner,
            offset_bytes,
            length_bytes,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.owner.stable_bytes()[self.offset_bytes..self.offset_bytes + self.length_bytes]
    }

    pub const fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> usize {
        self.length_bytes
    }
}

impl fmt::Debug for RetainedHostMemoryRegion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedHostMemoryRegion")
            .field("offset_bytes", &self.offset_bytes)
            .field("length_bytes", &self.length_bytes)
            .finish_non_exhaustive()
    }
}

/// Validated physical bytes for one model weight component.
///
/// Borrowed bytes allow mmap-backed dense weights to avoid a host copy.
/// Owned bytes cover format adapters that must repack before device upload.
pub struct WeightComponentPayload<'source> {
    component_id: WeightId,
    external_names: Vec<String>,
    source_files: Vec<String>,
    dimensions: Vec<u64>,
    element_type: ElementType,
    bytes: Cow<'source, [u8]>,
    retained_host_memory: Option<RetainedHostMemoryRegion>,
}

impl<'source> WeightComponentPayload<'source> {
    pub fn new(
        component: &WeightComponentSpec,
        external_name: impl Into<String>,
        source_file: impl Into<String>,
        dimensions: Vec<u64>,
        element_type: ElementType,
        bytes: impl Into<Cow<'source, [u8]>>,
    ) -> Result<Self, VNextError> {
        Self::from_ordered_sources(
            component,
            vec![external_name.into()],
            vec![source_file.into()],
            dimensions,
            element_type,
            bytes,
        )
    }

    /// Construct a payload materialized from multiple ordered checkpoint
    /// tensors. Multi-source order is part of the component schema, so packed
    /// projections cannot silently swap their logical partitions.
    pub fn from_ordered_sources(
        component: &WeightComponentSpec,
        external_names: Vec<String>,
        source_files: Vec<String>,
        dimensions: Vec<u64>,
        element_type: ElementType,
        bytes: impl Into<Cow<'source, [u8]>>,
    ) -> Result<Self, VNextError> {
        let bytes = bytes.into();
        let valid_source_file = |source_file: &str| {
            !source_file.is_empty()
                && !source_file.starts_with('/')
                && !source_file.contains('\\')
                && source_file
                    .split('/')
                    .all(|component| !matches!(component, "" | "." | ".."))
        };
        let sources_match = !external_names.is_empty()
            && external_names == component.external_names
            && external_names.len() == source_files.len()
            && source_files.iter().all(|file| valid_source_file(file));
        let expected_bytes = component.physical_bytes()?;
        if !sources_match
            || dimensions != component.dimensions
            || element_type != component.physical_element_type()
            || u64::try_from(bytes.len()).ok() != Some(expected_bytes)
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` payload differs from its schema identity, source, shape, type, or byte length",
                    component.id
                ),
            });
        }
        Ok(Self {
            component_id: component.id.clone(),
            external_names,
            source_files,
            dimensions,
            element_type,
            bytes,
            retained_host_memory: None,
        })
    }

    /// Attach the stable owner for an otherwise borrowed payload. Pointer and
    /// length identity are checked here so a backend cannot accidentally retain
    /// a different mmap range than the bytes validated against the schema.
    pub fn with_retained_host_memory(
        mut self,
        retained_host_memory: RetainedHostMemoryRegion,
    ) -> Result<Self, VNextError> {
        let retained_bytes = retained_host_memory.bytes();
        if retained_bytes.len() != self.bytes.len()
            || !std::ptr::eq(retained_bytes.as_ptr(), self.bytes.as_ptr())
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` retained host-memory region differs from its validated payload",
                    self.component_id
                ),
            });
        }
        self.retained_host_memory = Some(retained_host_memory);
        Ok(self)
    }

    pub fn component_id(&self) -> &WeightId {
        &self.component_id
    }

    pub fn external_name(&self) -> &str {
        &self.external_names[0]
    }

    pub fn source_file(&self) -> &str {
        &self.source_files[0]
    }

    pub fn external_names(&self) -> &[String] {
        &self.external_names
    }

    pub fn source_files(&self) -> &[String] {
        &self.source_files
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn retained_host_memory(&self) -> Option<&RetainedHostMemoryRegion> {
        self.retained_host_memory.as_ref()
    }
}

/// Backend-neutral source of schema-addressed physical weight components.
/// Implementations own file-format discovery, validation, and optional
/// repacking; resource initialization owns placement and device submission.
pub trait WeightComponentSource: Send + Sync {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{BlockQuantizationSpec, WeightComponentRole, WeightEncoding};

    struct StableBytes(Vec<u8>);

    // SAFETY: the Vec is never mutated and owns one fixed allocation until it
    // is dropped.
    unsafe impl StableHostMemory for StableBytes {
        fn stable_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    fn packed_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.test.gate_up").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec!["gate.weight".to_owned(), "up.weight".to_owned()],
            dimensions: vec![2, 2, 2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }
    }

    #[test]
    fn packed_payload_preserves_ordered_source_identity() {
        let component = packed_component();
        let payload = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![0_u8; 16],
        )
        .unwrap();
        assert_eq!(payload.external_names(), component.external_names);
        assert_eq!(payload.source_files().len(), 2);

        let error = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.iter().rev().cloned().collect(),
            vec![
                "model-2.safetensors".to_owned(),
                "model-1.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![0_u8; 16],
        )
        .err()
        .expect("source order is part of the packed component identity");
        assert!(error.to_string().contains("differs from its schema"));
    }

    #[test]
    fn block_quantized_payload_validates_block_abi_byte_size() {
        let component = WeightComponentSpec {
            id: WeightId::new("component.test.q4-k").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["weight.q4_k".to_owned()],
            dimensions: vec![2],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: "quantization.gguf.q4-k".to_owned().try_into().unwrap(),
                logical_values_per_block: 256,
                bytes_per_block: 144,
            }),
            required: true,
        };
        let payload = WeightComponentPayload::new(
            &component,
            "weight.q4_k",
            "model.gguf",
            vec![2],
            ElementType::U8,
            vec![0_u8; 288],
        )
        .unwrap();
        assert_eq!(payload.bytes().len(), 288);

        let error = WeightComponentPayload::new(
            &component,
            "weight.q4_k",
            "model.gguf",
            vec![2],
            ElementType::U8,
            vec![0_u8; 2],
        )
        .err()
        .expect("block-grid element count must not be mistaken for byte length");
        assert!(error.to_string().contains("differs from its schema"));
    }

    #[test]
    fn retained_region_must_be_the_validated_payload_and_keeps_its_owner_alive() {
        let component = packed_component();
        let owner = Arc::new(StableBytes(vec![7_u8; 32]));
        let retained = RetainedHostMemoryRegion::new(Arc::clone(&owner), 8, 16).unwrap();
        let payload = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            retained.bytes(),
        )
        .unwrap()
        .with_retained_host_memory(retained.clone())
        .unwrap();
        let retained = payload.retained_host_memory().unwrap().clone();
        drop(payload);
        drop(owner);
        assert_eq!(retained.bytes(), &[7_u8; 16]);

        let other = Arc::new(StableBytes(vec![7_u8; 16]));
        let wrong = RetainedHostMemoryRegion::new(other, 0, 16).unwrap();
        let result = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![7_u8; 16],
        )
        .unwrap()
        .with_retained_host_memory(wrong);
        let error = match result {
            Ok(_) => panic!("a different allocation must not satisfy retained payload identity"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("retained host-memory region differs"));
    }
}
