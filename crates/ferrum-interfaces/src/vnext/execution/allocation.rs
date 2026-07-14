use super::{
    invalid_plan, quantize_storage_bytes, AllocationKind, AllocationLifetime, BufferRequest,
    BufferUsage, Deserialize, DynamicStorageContract, ElementType, ResourceId, Serialize,
    VNextError,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub(super) resource_id: ResourceId,
    pub(super) per_instance_bytes: u64,
    pub(super) instance_stride_bytes: u64,
    pub(super) instance_count: u32,
    pub(super) size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) lifetime: AllocationLifetime,
    pub(super) kind: AllocationKind,
    pub(super) storage: DynamicStorageContract,
}

impl ResourceAllocation {
    pub(super) fn new(
        resource_id: ResourceId,
        per_instance_bytes: u64,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        kind: AllocationKind,
        storage: DynamicStorageContract,
    ) -> Result<Self, VNextError> {
        if per_instance_bytes == 0 || alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(invalid_plan(format!(
                "resource `{resource_id}` has invalid size or alignment"
            )));
        }
        let instance_stride_bytes =
            quantize_storage_bytes(per_instance_bytes, alignment_bytes, storage.profile())?;
        Ok(Self {
            resource_id,
            per_instance_bytes,
            instance_stride_bytes,
            instance_count: 1,
            size_bytes: instance_stride_bytes,
            alignment_bytes,
            usage,
            element_type,
            lifetime: AllocationLifetime::Plan,
            kind,
            storage,
        })
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub const fn per_instance_bytes(&self) -> u64 {
        self.per_instance_bytes
    }

    pub const fn instance_stride_bytes(&self) -> u64 {
        self.instance_stride_bytes
    }

    pub const fn instance_count(&self) -> u32 {
        self.instance_count
    }

    pub fn scoped_offset_bytes(
        &self,
        base_offset_bytes: u64,
        _active_sequence_slot: u32,
    ) -> Result<u64, VNextError> {
        if base_offset_bytes >= self.per_instance_bytes {
            return Err(invalid_plan(format!(
                "resource `{}` base offset is outside its per-instance span",
                self.resource_id
            )));
        }
        Ok(base_offset_bytes)
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn kind(&self) -> &AllocationKind {
        &self.kind
    }

    pub fn storage(&self) -> &DynamicStorageContract {
        &self.storage
    }

    pub fn buffer_request(&self) -> Result<BufferRequest, VNextError> {
        BufferRequest::new(
            self.resource_id.clone(),
            self.size_bytes,
            self.alignment_bytes,
            self.usage,
            self.element_type,
        )
    }
}
