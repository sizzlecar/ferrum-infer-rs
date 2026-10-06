use super::super::foundation::invalid_operation;
use super::{BufferDescriptor, DeviceRuntime, OperationBufferView, VNextError};
use crate::vnext::ResourceTransactionIdentity;

/// A proof over the exact immutable views borrowed by one invocation. Creating
/// it still checks live runtime descriptors, authority evidence, layout and
/// every physical region. It does not survive the invocation or own a cache.
pub(super) struct FullyCoveredOperationViews<'views, 'lease, B> {
    views: &'views [OperationBufferView<'lease, B>],
}

impl<'views, 'lease, B> FullyCoveredOperationViews<'views, 'lease, B> {
    pub(super) fn validate<R: DeviceRuntime<Buffer = B>>(
        views: &'views [OperationBufferView<'lease, B>],
        runtime: &R,
        lease_identity: Option<&ResourceTransactionIdentity>,
    ) -> Result<Self, VNextError> {
        for view in views {
            view.validate_runtime(runtime, lease_identity)?;
            let translated = view.translate(0, view.descriptor().size_bytes)?;
            let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                total
                    .checked_add(region.length_bytes())
                    .ok_or_else(|| invalid_operation("translated operation regions overflow u64"))
            })?;
            if translated_bytes != view.descriptor().size_bytes {
                return Err(invalid_operation(format!(
                    "operation resource `{}` is not fully backed by physical regions",
                    view.resource_id()
                )));
            }
        }
        Ok(Self { views })
    }

    pub(super) fn descriptor(&self, index: usize) -> Result<&BufferDescriptor, VNextError> {
        self.views
            .get(index)
            .map(OperationBufferView::descriptor)
            .ok_or_else(|| invalid_operation("validated operation view index is out of range"))
    }

    /// Full logical coverage includes every nonempty bounded subrange, also
    /// for a paged view or a window into a larger backing extent. The view is
    /// immutably borrowed, so the physical mapping checked above cannot change
    /// between this check and value/workspace validation.
    pub(super) fn validate_subrange(
        &self,
        index: usize,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<(), VNextError> {
        validate_subrange_bounds(
            self.descriptor(index)?.size_bytes,
            offset_bytes,
            length_bytes,
        )
    }
}

fn validate_subrange_bounds(
    covered_bytes: u64,
    offset_bytes: u64,
    length_bytes: u64,
) -> Result<(), VNextError> {
    let end_bytes = offset_bytes
        .checked_add(length_bytes)
        .ok_or_else(|| invalid_operation("operation logical buffer range overflows u64"))?;
    if length_bytes == 0 || end_bytes > covered_bytes {
        return Err(invalid_operation(
            "operation logical buffer range is empty or outside its resource",
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_only_backing_window_coverage<R: DeviceRuntime>(
    runtime: &R,
    backing: crate::vnext::LogicalBackingBufferView<'_, R::Buffer>,
    logical_bytes: u64,
    window_offset_bytes: u64,
    ranges: &[(u64, u64)],
) -> Result<Vec<Result<Vec<(u64, u64, u64)>, VNextError>>, VNextError> {
    let descriptor = BufferDescriptor {
        resource_id: backing.slice().resource_id().clone(),
        size_bytes: logical_bytes,
        alignment_bytes: backing.alignment_bytes(),
        usage: backing.usage(),
        element_type: backing.element_type(),
    };
    let views = [OperationBufferView::from_backing_window(
        descriptor,
        backing,
        window_offset_bytes,
        crate::vnext::AllocationLifetime::Request,
    )];
    let proof = FullyCoveredOperationViews::validate(&views, runtime, None)?;
    Ok(ranges
        .iter()
        .map(|&(offset, length)| {
            let checked = proof.validate_subrange(0, offset, length);
            let translated = views[0].translate(offset, length);
            assert_eq!(checked.is_ok(), translated.is_ok());
            checked.map(|()| {
                translated
                    .unwrap()
                    .iter()
                    .map(|region| {
                        (
                            region.logical_offset_bytes(),
                            region.buffer_and_physical_range().1.start,
                            region.length_bytes(),
                        )
                    })
                    .collect()
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::validate_subrange_bounds;

    #[test]
    fn covered_subranges_include_exact_extent_and_cross_page_boundaries() {
        for (offset, length) in [(0, 192), (0, 1), (191, 1), (63, 66), (64, 128)] {
            validate_subrange_bounds(192, offset, length).unwrap();
        }
        validate_subrange_bounds(u64::MAX, u64::MAX - 1, 1).unwrap();
    }

    #[test]
    fn coverage_never_extends_to_backing_capacity_or_accepts_empty_overflowing_ranges() {
        for (offset, length) in [(0, 0), (96, 0), (96, 1), (95, 2), (0, 128)] {
            assert!(validate_subrange_bounds(96, offset, length).is_err());
        }
        assert!(validate_subrange_bounds(0, 0, 1).is_err());
        assert!(validate_subrange_bounds(u64::MAX, u64::MAX, 1).is_err());
    }
}
