//! The same native owner and prepared workspace cover two and three spatial
//! axes. Existing 2D descriptors and fingerprints retain their original ABI.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CudnnConv3dDescriptor {
    pub scalar: CudnnConvScalar,
    pub input: [i32; 5],
    pub filter: [i32; 5],
    pub output: [i32; 5],
    pub padding: [i32; 3],
    pub stride: [i32; 3],
    pub dilation: [i32; 3],
    pub groups: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvolutionDimensions {
    Spatial2(CudnnConv2dDescriptor),
    Spatial3(CudnnConv3dDescriptor),
}
impl From<CudnnConv2dDescriptor> for ConvolutionDimensions {
    fn from(s: CudnnConv2dDescriptor) -> Self {
        Self::Spatial2(s)
    }
}
impl From<CudnnConv3dDescriptor> for ConvolutionDimensions {
    fn from(s: CudnnConv3dDescriptor) -> Self {
        Self::Spatial3(s)
    }
}
impl ConvolutionDimensions {
    pub(super) fn scalar(self) -> CudnnConvScalar {
        match self {
            Self::Spatial2(s) => s.scalar,
            Self::Spatial3(s) => s.scalar,
        }
    }
}
impl CudnnConv3dDescriptor {
    pub(super) fn validate(self) -> Result<(), String> {
        contiguous_strides(self.input)?;
        contiguous_strides(self.output)?;
        contiguous_strides(self.filter)?;
        if self.groups <= 0
            || self.padding.iter().any(|&x| x < 0)
            || self.stride.iter().chain(&self.dilation).any(|&x| x <= 0)
            || self.input[0] != self.output[0]
            || self.filter[0] != self.output[1]
            || self.input[1] % self.groups != 0
            || self.output[1] % self.groups != 0
            || i64::from(self.filter[1]) * i64::from(self.groups) != i64::from(self.input[1])
        {
            return Err("cuDNN 3D convolution has inconsistent batch, groups or axes".into());
        }
        for axis in 0..3 {
            let input = i64::from(self.input[axis + 2]) + 2 * i64::from(self.padding[axis]);
            let kernel =
                (i64::from(self.filter[axis + 2]) - 1) * i64::from(self.dilation[axis]) + 1;
            if kernel > input
                || (input - kernel) / i64::from(self.stride[axis]) + 1
                    != i64::from(self.output[axis + 2])
            {
                return Err("cuDNN 3D output differs from convolution geometry".into());
            }
        }
        Ok(())
    }
    pub(super) fn fingerprint(
        self,
        owner: &str,
        cap: u64,
        selected: ConvolutionAlgorithm,
        direction: ConvolutionDirection,
    ) -> String {
        let mut d = Sha256::new();
        d.update(b"cuda.cudnn.conv3d.prepared-direction-actual-workspace.v1");
        d.update([direction.tag()]);
        d.update(owner.as_bytes());
        d.update([self.scalar.tag()]);
        for dimensions in [self.input, self.filter, self.output] {
            for v in dimensions {
                d.update(v.to_le_bytes());
            }
        }
        for dimensions in [self.padding, self.stride, self.dilation] {
            for v in dimensions {
                d.update(v.to_le_bytes());
            }
        }
        d.update(self.groups.to_le_bytes());
        d.update(cap.to_le_bytes());
        d.update(selected.algo.to_le_bytes());
        d.update(selected.math_type.to_le_bytes());
        d.update((selected.required_bytes as u64).to_le_bytes());
        format!("{:x}", d.finalize())
    }
}

/// cuDNN tensor strides are signed 32-bit quantities even on a 64-bit host.
/// Check this native boundary separately from the runtime's byte capacities.
fn contiguous_strides(shape: [i32; 5]) -> Result<[i32; 5], String> {
    if shape.iter().any(|&v| v <= 0) {
        return Err("cuDNN 3D requires positive dimensions".into());
    }
    let mut strides = [1i32; 5];
    for axis in (0..4).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or("cuDNN 3D contiguous stride exceeds i32")?;
    }
    Ok(strides)
}
impl ConvolutionDescriptors {
    pub(super) unsafe fn configure3d(
        self,
        a: &Api,
        s: CudnnConv3dDescriptor,
    ) -> Result<(), String> {
        let input_strides = contiguous_strides(s.input)?;
        let output_strides = contiguous_strides(s.output)?;
        check(
            (a.set_tensor_nd)(
                self.input,
                s.scalar.data_type(),
                5,
                s.input.as_ptr(),
                input_strides.as_ptr(),
            ),
            "set 3D input",
        )?;
        check(
            (a.set_tensor_nd)(
                self.output,
                s.scalar.data_type(),
                5,
                s.output.as_ptr(),
                output_strides.as_ptr(),
            ),
            "set 3D output",
        )?;
        check(
            (a.set_filter_nd)(
                self.filter,
                s.scalar.data_type(),
                ffi::NCHW,
                5,
                s.filter.as_ptr(),
            ),
            "set 3D filter",
        )?;
        check(
            (a.set_convolution_nd)(
                self.convolution,
                3,
                s.padding.as_ptr(),
                s.stride.as_ptr(),
                s.dilation.as_ptr(),
                ffi::CROSS_CORRELATION,
                ffi::FLOAT,
            ),
            "set 3D convolution",
        )?;
        check((a.set_groups)(self.convolution, s.groups), "set 3D groups")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape() -> CudnnConv3dDescriptor {
        CudnnConv3dDescriptor {
            scalar: CudnnConvScalar::F32Fma,
            input: [2, 4, 7, 6, 8],
            filter: [6, 2, 3, 3, 3],
            output: [2, 6, 3, 2, 3],
            padding: [0; 3],
            stride: [2; 3],
            dilation: [1; 3],
            groups: 2,
        }
    }
    #[test]
    fn native_axes_and_contiguous_strides_are_checked_before_descriptor_mutation() {
        let s = shape();
        s.validate().unwrap();
        assert_eq!(contiguous_strides(s.input).unwrap(), [1344, 336, 48, 8, 1]);
        assert!(CudnnConv3dDescriptor { groups: 3, ..s }.validate().is_err());
        assert!(CudnnConv3dDescriptor {
            output: [2, 6, 4, 2, 3],
            ..s
        }
        .validate()
        .is_err());
        assert!(contiguous_strides([1, 2, 65536, 65536, 1]).is_err());
        assert!(contiguous_strides([1, 2, 0, 4, 8]).is_err());
    }
    #[test]
    fn prepared_depth_and_owner_are_part_of_the_execution_authority() {
        let owner = Arc::new(CudaCudnnOwner {
            inner: Mutex::new(None),
            fingerprint: "fixture-3d-owner".into(),
            workspace_limit_bytes: 128,
        });
        let selected = ConvolutionAlgorithm {
            algo: 0,
            math_type: 3,
            required_bytes: 64,
        };
        let s = shape();
        let direction = ConvolutionDirection::Forward;
        let fingerprint = s.fingerprint(owner.fingerprint(), 128, selected, direction);
        let p = PreparedConv {
            owner: owner.clone(),
            shape: s.into(),
            direction,
            selected,
            fingerprint: fingerprint.clone(),
        };
        p.validate_invocation(&owner, s, Some(&fingerprint))
            .unwrap();
        let altered = CudnnConv3dDescriptor {
            input: [2, 4, 8, 6, 8],
            ..s
        };
        assert!(p
            .validate_invocation(&owner, altered, Some(&fingerprint))
            .is_err());
        assert_ne!(
            fingerprint,
            altered.fingerprint(owner.fingerprint(), 128, selected, direction)
        );
        let other = Arc::new(CudaCudnnOwner {
            inner: Mutex::new(None),
            fingerprint: owner.fingerprint.clone(),
            workspace_limit_bytes: 128,
        });
        assert!(p
            .validate_invocation(&other, s, Some(&fingerprint))
            .is_err());
        assert!(selected.validate_workspace(0, 64).is_err());
        selected.validate_workspace(256, 64).unwrap();
    }
}
