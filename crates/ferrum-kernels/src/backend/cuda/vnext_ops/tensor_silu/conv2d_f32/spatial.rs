//! Checked spatial metadata and explicit padding for the shared F32 provider.
use super::*;
use cudarc::driver::CudaStream;

impl Shape {
    pub(super) fn new3d(g: Conv3dF32Geometry, workspace: u64) -> Result<Self, String> {
        let dim = |v| {
            i32::try_from(v)
                .ok()
                .filter(|&v| v > 0)
                .ok_or("cuDNN F32 axis exceeds positive i32")
        };
        let tensor = |d: [u64; 5]| -> Result<[i32; 5], String> {
            d.iter()
                .try_fold(1u64, |n, &v| n.checked_mul(v))
                .filter(|&v| v <= i32::MAX as u64)
                .ok_or("cuDNN F32 tensor numel exceeds i32")?;
            Ok([dim(d[0])?, dim(d[1])?, dim(d[2])?, dim(d[3])?, dim(d[4])?])
        };
        let [n, c, _, _, _] = g.input();
        let [d, h, w] = g.padded_spatial();
        let s = g.stride();
        let dilation = g.dilation();
        let native = CudnnConv3dDescriptor {
            scalar: CudnnConvScalar::F32Fma,
            input: tensor([n, c, d, h, w])?,
            filter: tensor(g.weight())?,
            output: tensor(g.output())?,
            padding: [0; 3],
            stride: [dim(s[0])?, dim(s[1])?, dim(s[2])?],
            dilation: [dim(dilation[0])?, dim(dilation[1])?, dim(dilation[2])?],
            groups: dim(g.groups())?,
        };
        Self {
            native: native.into(),
            geometry: F32CnnGeometry::Conv3d(g),
            padding_bytes: if g.padding() == [0; 6] {
                0
            } else {
                g.padded_input_bytes()
            },
            padded_height: h,
            padded_width: w,
            workspace_offset: 0,
            workspace_bytes: 0,
            scratch_bytes: 0,
        }
        .with_workspace(workspace)
    }
    pub(super) fn pad_input(
        self,
        stream: &CudaStream,
        kernels: &RetainedFunctions,
        input: u64,
        scratch: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let elements = self.padding_bytes / 4;
        match self.geometry {
            F32CnnGeometry::Conv2d(g) => {
                let height = g.input_height();
                let width = g.input_width();
                let [top, _, left, _] = g.padding();
                let reflect = u32::from(g.padding_mode() == Conv2dF32PaddingMode::Reflect);
                let mut launch = stream.launch_builder(kernels.cnn_f32_function(0));
                launch
                    .arg(&input)
                    .arg(&scratch)
                    .arg(&elements)
                    .arg(&height)
                    .arg(&width)
                    .arg(&self.padded_height)
                    .arg(&self.padded_width)
                    .arg(&top)
                    .arg(&left)
                    .arg(&reflect);
                unsafe { launch.launch(flat(elements)) }
            }
            F32CnnGeometry::Conv3d(g) => {
                let [_, _, depth, height, width] = g.input();
                let [pd, ph, pw] = g.padded_spatial();
                let [front, _, top, _, left, _] = g.padding();
                let reflect = g
                    .padding_modes()
                    .iter()
                    .enumerate()
                    .fold(0u32, |mask, (a, m)| {
                        mask | ((m.attribute_value() as u32) << a)
                    });
                let mut launch = stream.launch_builder(kernels.cnn_f32_function(4));
                launch
                    .arg(&input)
                    .arg(&scratch)
                    .arg(&elements)
                    .arg(&depth)
                    .arg(&height)
                    .arg(&width)
                    .arg(&pd)
                    .arg(&ph)
                    .arg(&pw)
                    .arg(&front)
                    .arg(&top)
                    .arg(&left)
                    .arg(&reflect);
                unsafe { launch.launch(flat(elements)) }
            }
            _ => unreachable!("convolution shape"),
        }
        .map(|_| ())
        .map_err(|e| CudaDeviceRuntimeError::driver("F32 convolution explicit per-axis padding", e))
    }
}
