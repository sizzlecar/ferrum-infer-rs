use ferrum_runtime::SpatialGroupMeanBf16Geometry;

/// Initialized, fixed-width by-value ABI shared with the CUDA kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct NativeGroupMean {
    input_channels: u64,
    input_height: u64,
    input_width: u64,
    output_channels: u64,
    output_height: u64,
    output_width: u64,
    scale_h: u64,
    scale_w: u64,
    frame_factor: u64,
    group_size: u64,
}
// SAFETY: initialized u64 fields, in exactly the CUDA struct's order.
unsafe impl cudarc::driver::DeviceRepr for NativeGroupMean {}
const _: [(); 80] = [(); std::mem::size_of::<NativeGroupMean>()];
impl NativeGroupMean {
    pub(super) fn new(g: &SpatialGroupMeanBf16Geometry) -> Self {
        Self {
            input_channels: g.input_channels(),
            input_height: g.input_height(),
            input_width: g.input_width(),
            output_channels: g.output_channels(),
            output_height: g.output_height(),
            output_width: g.output_width(),
            scale_h: g.scale_h(),
            scale_w: g.scale_w(),
            frame_factor: g.frame_factor(),
            group_size: g.group_size(),
        }
    }
}
