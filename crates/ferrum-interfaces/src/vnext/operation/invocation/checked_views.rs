//! Two outputs from the one prepared-invocation validation body. The numeric
//! output owns descriptors only, never buffers, retentions or execution rights.
use super::super::buffer_view::{
    validate_dynamic_coverage, OperationBufferCoverage, OperationBufferStorageKind,
};
use super::*;
use crate::vnext::resource::ValidatedLogicalBacking;
use crate::vnext::{
    DeviceBufferRetention, DynamicStorageView, LeasedBufferView, ResourceTransactionIdentity,
};

pub(super) enum BackingInspection<'a, B> {
    Physical(LogicalBackingBufferView<'a, B>),
    Numeric(ValidatedLogicalBacking<'a>),
}

impl<'a, B> BackingInspection<'a, B> {
    pub(super) fn participant<R: DeviceRuntime<Buffer = B>>(
        resources: OperationInvocationResources<'a, R>,
        index: usize,
        resource: &ResourceId,
        numeric: bool,
    ) -> Result<Self, VNextError> {
        if numeric {
            resources
                .validate_participant_backing(index, resource)
                .map(Self::Numeric)
        } else {
            resources
                .participant_backing_view(index, resource)
                .map(Self::Physical)
        }
    }
    pub(super) fn shared<R: DeviceRuntime<Buffer = B>>(
        resources: OperationInvocationResources<'a, R>,
        resource: &ResourceId,
        numeric: bool,
    ) -> Result<Self, VNextError> {
        if numeric {
            resources.validate_backing(resource).map(Self::Numeric)
        } else {
            resources.backing_view(resource).map(Self::Physical)
        }
    }
    pub(super) fn size_bytes(&self) -> u64 {
        match self {
            Self::Physical(v) => v.size_bytes(),
            Self::Numeric(v) => v.size_bytes(),
        }
    }
    pub(super) fn capacity_size_bytes(&self) -> u64 {
        match self {
            Self::Physical(v) => v.capacity_size_bytes(),
            Self::Numeric(v) => v.capacity_size_bytes(),
        }
    }
    pub(super) fn alignment_bytes(&self) -> u64 {
        match self {
            Self::Physical(v) => v.alignment_bytes(),
            Self::Numeric(v) => v.alignment_bytes(),
        }
    }
    pub(super) fn usage(&self) -> BufferUsage {
        match self {
            Self::Physical(v) => v.usage(),
            Self::Numeric(v) => v.usage(),
        }
    }
    pub(super) fn element_type(&self) -> ElementType {
        match self {
            Self::Physical(v) => v.element_type(),
            Self::Numeric(v) => v.element_type(),
        }
    }
    pub(super) fn storage_profile(&self) -> crate::vnext::DynamicStorageProfile {
        match self {
            Self::Physical(v) => v.storage_profile(),
            Self::Numeric(v) => v.storage_profile(),
        }
    }
}

pub(super) enum InvocationViews<'a, B> {
    Physical(Vec<OperationBufferView<'a, B>>),
    Numeric(Vec<BufferDescriptor>),
}

impl<'a, B> InvocationViews<'a, B> {
    pub(super) fn new(numeric: bool, capacity: usize) -> Self {
        if numeric {
            Self::Numeric(Vec::with_capacity(capacity))
        } else {
            Self::Physical(Vec::with_capacity(capacity))
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Physical(v) => v.len(),
            Self::Numeric(v) => v.len(),
        }
    }
    pub(super) fn descriptor(&self, index: usize) -> Option<&BufferDescriptor> {
        match self {
            Self::Physical(v) => v.get(index).map(OperationBufferView::descriptor),
            Self::Numeric(v) => v.get(index),
        }
    }
    pub(super) fn push_static<R: DeviceRuntime<Buffer = B>>(
        &mut self,
        runtime: &R,
        expected_identity: Option<&ResourceTransactionIdentity>,
        leased: LeasedBufferView<'a, B>,
        retention: impl FnOnce() -> DeviceBufferRetention,
    ) -> Result<(), VNextError> {
        match self {
            Self::Physical(views) => {
                views.push(OperationBufferView::from_static(leased, retention()))
            }
            Self::Numeric(views) => {
                let actual = super::super::buffer_view::validate_static_runtime(
                    runtime,
                    &leased,
                    expected_identity,
                )?;
                if actual.size_bytes == 0 {
                    return Err(invalid_operation(
                        "operation logical buffer range is empty or outside its resource",
                    ));
                }
                views.push(actual);
            }
        }
        Ok(())
    }
    pub(super) fn push_dynamic(
        &mut self,
        descriptor: BufferDescriptor,
        backing: BackingInspection<'a, B>,
        coverage: OperationBufferCoverage,
        lifetime: AllocationLifetime,
        packed: bool,
    ) -> Result<(), VNextError> {
        match (self, backing) {
            (Self::Physical(views), BackingInspection::Physical(backing)) => {
                let view = match coverage {
                    OperationBufferCoverage::Exact => {
                        OperationBufferView::from_backing_exact(descriptor, backing, lifetime)
                    }
                    OperationBufferCoverage::BackingWindow { offset_bytes } => {
                        OperationBufferView::from_backing_window(
                            descriptor,
                            backing,
                            offset_bytes,
                            lifetime,
                        )
                    }
                };
                views.push(view.with_packed_batch_coordinates(packed));
            }
            (Self::Numeric(views), BackingInspection::Numeric(backing)) => {
                if descriptor.size_bytes == 0 {
                    return Err(invalid_operation(
                        "operation logical buffer range is empty or outside its resource",
                    ));
                }
                let storage = match backing.storage_profile().view() {
                    DynamicStorageView::Contiguous => OperationBufferStorageKind::DynamicContiguous,
                    DynamicStorageView::PagedRegions { .. } => {
                        OperationBufferStorageKind::DynamicPaged
                    }
                };
                validate_dynamic_coverage(
                    storage,
                    descriptor.size_bytes,
                    backing.segment_count(),
                    backing.physical_coverage_bytes()?,
                    coverage,
                )?;
                views.push(descriptor);
            }
            _ => {
                return Err(invalid_operation(
                    "invocation validation output mode differs from backing inspection",
                ))
            }
        }
        Ok(())
    }
    pub(super) fn validate<'views, R: DeviceRuntime<Buffer = B>>(
        &'views self,
        runtime: &R,
        lease_identity: Option<&ResourceTransactionIdentity>,
    ) -> Result<InvocationViewCoverage<'views, 'a, B>, VNextError> {
        match self {
            Self::Physical(views) => {
                FullyCoveredOperationViews::validate(views, runtime, lease_identity)
                    .map(InvocationViewCoverage::Physical)
            }
            // Numeric descriptors were admitted only after fresh authority,
            // runtime descriptor, physical coverage and logical window checks.
            Self::Numeric(views) => Ok(InvocationViewCoverage::Numeric(views)),
        }
    }
    pub(super) fn into_physical(self) -> Result<Vec<OperationBufferView<'a, B>>, VNextError> {
        match self {
            Self::Physical(views) => Ok(views),
            Self::Numeric(_) => Err(invalid_operation(
                "numeric validation does not expose executable views",
            )),
        }
    }
}

pub(super) enum InvocationViewCoverage<'views, 'lease, B> {
    Physical(FullyCoveredOperationViews<'views, 'lease, B>),
    Numeric(&'views [BufferDescriptor]),
}

impl<B> InvocationViewCoverage<'_, '_, B> {
    pub(super) fn descriptor(&self, index: usize) -> Result<&BufferDescriptor, VNextError> {
        match self {
            Self::Physical(proof) => proof.descriptor(index),
            Self::Numeric(views) => views
                .get(index)
                .ok_or_else(|| invalid_operation("validated operation view index is out of range")),
        }
    }
    pub(super) fn validate_subrange(
        &self,
        index: usize,
        offset: u64,
        length: u64,
    ) -> Result<(), VNextError> {
        match self {
            Self::Physical(proof) => proof.validate_subrange(index, offset, length),
            Self::Numeric(_) => super::view_coverage::validate_subrange_bounds(
                self.descriptor(index)?.size_bytes,
                offset,
                length,
            ),
        }
    }
}
