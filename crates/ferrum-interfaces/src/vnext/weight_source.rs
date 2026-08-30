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

    /// Entire stable allocation. Backends use this only to prove that a
    /// page-aligned native view enclosing [`Self::bytes`] remains within the
    /// retained owner.
    pub fn owner_bytes(&self) -> &[u8] {
        self.owner.stable_bytes()
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

/// One ordered byte segment of a physical weight component.
///
/// A segment may borrow directly from immutable checkpoint storage. When a
/// backend needs to retain that storage beyond the source request, the
/// attached [`RetainedHostMemoryRegion`] proves and owns the exact borrowed
/// range.
pub struct WeightComponentSegment<'source> {
    bytes: Cow<'source, [u8]>,
    retained_host_memory: Option<RetainedHostMemoryRegion>,
}

impl<'source> WeightComponentSegment<'source> {
    pub fn new(bytes: impl Into<Cow<'source, [u8]>>) -> Self {
        Self {
            bytes: bytes.into(),
            retained_host_memory: None,
        }
    }

    /// Attach the stable owner for this segment. Pointer and length identity
    /// are checked so a backend cannot retain a different mmap range than the
    /// segment validated against the component schema.
    pub fn with_retained_host_memory(
        mut self,
        retained_host_memory: RetainedHostMemoryRegion,
    ) -> Result<Self, VNextError> {
        let retained_bytes = retained_host_memory.bytes();
        if retained_bytes.len() != self.bytes.len()
            || !std::ptr::eq(retained_bytes.as_ptr(), self.bytes.as_ptr())
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "weight component segment retained host-memory region differs from its validated bytes"
                    .to_owned(),
            });
        }
        self.retained_host_memory = Some(retained_host_memory);
        Ok(self)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn retained_host_memory(&self) -> Option<&RetainedHostMemoryRegion> {
        self.retained_host_memory.as_ref()
    }
}

/// Validated physical bytes for one model weight component split into
/// deterministic source-order segments.
///
/// Segment boundaries are transport metadata, not part of the component ABI:
/// their concatenated bytes must exactly match the component's physical byte
/// length. This lets existing contiguous sources use one segment while
/// adapters for packed multi-tensor components expose one mmap-backed segment
/// per ordered checkpoint tensor without allocating an aggregate `Vec`.
pub struct WeightComponentSegments<'source> {
    component_id: WeightId,
    external_names: Vec<String>,
    source_files: Vec<String>,
    dimensions: Vec<u64>,
    element_type: ElementType,
    segments: Vec<WeightComponentSegment<'source>>,
}

impl<'source> WeightComponentSegments<'source> {
    pub fn from_ordered_segments(
        component: &WeightComponentSpec,
        external_names: Vec<String>,
        source_files: Vec<String>,
        dimensions: Vec<u64>,
        element_type: ElementType,
        segments: Vec<WeightComponentSegment<'source>>,
    ) -> Result<Self, VNextError> {
        let total_bytes = segments.iter().try_fold(0_u64, |total, segment| {
            u64::try_from(segment.bytes().len())
                .ok()
                .and_then(|length| total.checked_add(length))
        });
        let sources_match = valid_ordered_sources(component, &external_names, &source_files);
        let expected_bytes = component.physical_bytes()?;
        if segments.is_empty()
            || segments.iter().any(|segment| segment.bytes().is_empty())
            || !sources_match
            || dimensions != component.dimensions
            || element_type != component.physical_element_type()
            || total_bytes != Some(expected_bytes)
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` segments differ from its schema identity, source, shape, type, or byte length",
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
            segments,
        })
    }

    /// Wrap an already validated contiguous payload as one segment. This is
    /// the compatibility path used by the default
    /// [`WeightComponentSource::component_segments`] implementation.
    pub fn from_payload(payload: WeightComponentPayload<'source>) -> Self {
        let WeightComponentPayload {
            component_id,
            external_names,
            source_files,
            dimensions,
            element_type,
            bytes,
            retained_host_memory,
        } = payload;
        Self {
            component_id,
            external_names,
            source_files,
            dimensions,
            element_type,
            segments: vec![WeightComponentSegment {
                bytes,
                retained_host_memory,
            }],
        }
    }

    pub fn component_id(&self) -> &WeightId {
        &self.component_id
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

    pub fn segments(&self) -> &[WeightComponentSegment<'source>] {
        &self.segments
    }

    pub fn total_bytes(&self) -> u64 {
        self.segments
            .iter()
            .map(|segment| segment.bytes().len() as u64)
            .sum()
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
        let sources_match = valid_ordered_sources(component, &external_names, &source_files);
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
/// Implementations own checkpoint file-format discovery and source-payload
/// validation. The execution plan's trusted [`super::WeightMaterializer`] owns
/// any repacking or quantization before resource initialization performs
/// placement and device submission.
pub trait WeightComponentSource: Send + Sync {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError>;

    /// Return deterministic source-order segments for one component.
    ///
    /// Existing sources remain compatible through a one-segment wrapper.
    /// Format adapters may override this method to expose independently
    /// retained mmap ranges without changing the contiguous `component()`
    /// contract used by existing materializers and upload paths.
    fn component_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentSegments<'source>, VNextError> {
        self.component(component)
            .map(WeightComponentSegments::from_payload)
    }
}

fn valid_ordered_sources(
    component: &WeightComponentSpec,
    external_names: &[String],
    source_files: &[String],
) -> bool {
    let valid_source_file = |source_file: &str| {
        !source_file.is_empty()
            && !source_file.starts_with('/')
            && !source_file.contains('\\')
            && source_file
                .split('/')
                .all(|component| !matches!(component, "" | "." | ".."))
    };
    !external_names.is_empty()
        && external_names == component.external_names
        && external_names.len() == source_files.len()
        && source_files.iter().all(|file| valid_source_file(file))
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

    struct ContiguousSource(Vec<u8>);

    impl WeightComponentSource for ContiguousSource {
        fn component<'source>(
            &'source self,
            component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'source>, VNextError> {
            WeightComponentPayload::from_ordered_sources(
                component,
                component.external_names.clone(),
                vec![
                    "model-1.safetensors".to_owned(),
                    "model-2.safetensors".to_owned(),
                ],
                component.dimensions.clone(),
                ElementType::F16,
                self.0.as_slice(),
            )
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

    #[test]
    fn ordered_segments_preserve_source_order_and_retain_each_exact_range() {
        let component = packed_component();
        let owner_a = Arc::new(StableBytes(vec![1_u8; 24]));
        let owner_b = Arc::new(StableBytes(vec![2_u8; 24]));
        let retained_a = RetainedHostMemoryRegion::new(Arc::clone(&owner_a), 4, 8).unwrap();
        let retained_b = RetainedHostMemoryRegion::new(Arc::clone(&owner_b), 12, 8).unwrap();
        let segments = WeightComponentSegments::from_ordered_segments(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![
                WeightComponentSegment::new(retained_a.bytes())
                    .with_retained_host_memory(retained_a.clone())
                    .unwrap(),
                WeightComponentSegment::new(retained_b.bytes())
                    .with_retained_host_memory(retained_b.clone())
                    .unwrap(),
            ],
        )
        .unwrap();

        assert_eq!(segments.external_names(), component.external_names);
        assert_eq!(segments.source_files().len(), 2);
        assert_eq!(segments.total_bytes(), 16);
        assert_eq!(segments.segments().len(), 2);
        assert_eq!(segments.segments()[0].bytes(), &[1_u8; 8]);
        assert_eq!(segments.segments()[1].bytes(), &[2_u8; 8]);
        let retained_segments = segments
            .segments()
            .iter()
            .map(|segment| segment.retained_host_memory().unwrap().clone())
            .collect::<Vec<_>>();
        drop(segments);
        drop(retained_a);
        drop(retained_b);
        drop(owner_a);
        drop(owner_b);
        assert_eq!(retained_segments[0].bytes(), &[1_u8; 8]);
        assert_eq!(retained_segments[1].bytes(), &[2_u8; 8]);
    }

    #[test]
    fn component_segments_default_wraps_the_contiguous_payload_once() {
        let component = packed_component();
        let source = ContiguousSource(vec![3_u8; 16]);
        let segments = source.component_segments(&component).unwrap();

        assert_eq!(segments.component_id(), &component.id);
        assert_eq!(segments.dimensions(), component.dimensions);
        assert_eq!(segments.element_type(), ElementType::F16);
        assert_eq!(segments.segments().len(), 1);
        assert_eq!(segments.segments()[0].bytes(), &[3_u8; 16]);
    }

    #[test]
    fn ordered_segments_reject_wrong_total_length_or_retained_range() {
        let component = packed_component();
        let error = WeightComponentSegments::from_ordered_segments(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![WeightComponentSegment::new(vec![0_u8; 15])],
        )
        .err()
        .expect("the segment total must match the physical component size");
        assert!(error
            .to_string()
            .contains("segments differ from its schema"));

        let bytes = vec![5_u8; 8];
        let owner = Arc::new(StableBytes(vec![5_u8; 8]));
        let retained = RetainedHostMemoryRegion::new(owner, 0, 8).unwrap();
        let error = WeightComponentSegment::new(bytes)
            .with_retained_host_memory(retained)
            .err()
            .expect("a different allocation must not satisfy segment retention");
        assert!(error
            .to_string()
            .contains("retained host-memory region differs"));
    }
}
