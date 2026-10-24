//! Retain only core-admitted contiguous component and invocation ranges.
use super::*;
use ferrum_runtime::{InvocationWorkProjection, OperationInvocation, ResolvedStorageComponent};

pub(super) fn component<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    stored: &ResolvedStorageComponent,
    alignment: u64,
) -> Result<CudaBufferRegion, String> {
    let mut views = participant
        .views()
        .iter()
        .filter(|view| view.resource_id() == stored.resource_id());
    let view = views
        .next()
        .ok_or("tensor binding lacks committed resource view")?;
    if views.next().is_some() {
        return Err("tensor binding has ambiguous resource views".into());
    }
    let translated = view
        .translate(stored.offset_bytes(), stored.length_bytes())
        .map_err(|e| e.to_string())?;
    let mut pieces = translated.iter();
    let piece = pieces
        .next()
        .ok_or("tensor binding resolved to no physical range")?;
    if pieces.next().is_some() {
        return Err("tensor binding must be physically contiguous".into());
    }
    let (buffer, range, retention) = piece.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    if alignment == 0
        || region.element_type() != stored.element_type()
        || region.length_bytes() != stored.length_bytes()
        || region.device_ptr() % alignment != 0
    {
        return Err("tensor binding physical dtype/extent/alignment mismatch".into());
    }
    Ok(region)
}

pub(super) fn scratch<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    bytes: u64,
    alignment: u64,
) -> Result<CudaBufferRegion, String> {
    let view = participant
        .scratch_view()
        .ok_or("tensor invocation scratch was not admitted")?;
    let translated = view.translate(0, bytes).map_err(|e| e.to_string())?;
    let mut pieces = translated.iter();
    let piece = pieces
        .next()
        .ok_or("tensor scratch resolved to no physical range")?;
    if pieces.next().is_some() {
        return Err("tensor scratch must be physically contiguous".into());
    }
    let (buffer, range, retention) = piece.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    if alignment == 0
        || region.element_type() != ElementType::U8
        || region.length_bytes() != bytes
        || region.device_ptr() % alignment != 0
    {
        return Err("tensor scratch physical dtype/extent/alignment mismatch".into());
    }
    Ok(region)
}
