//! A convolution's physical dtype and math policy travel with prepared state.
//! BF16 retains the existing DEFAULT/TENSOR_OP policy. F32 selects FMA only;
//! DEFAULT is not sufficient because it can permit TF32 on Ampere and later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CudnnConvScalar {
    Bf16,
    F32Fma,
}
impl CudnnConvScalar {
    pub(super) const fn data_type(self) -> i32 {
        match self {
            Self::Bf16 => 9,
            Self::F32Fma => 0,
        }
    }
    pub(super) const fn requested_math(self) -> i32 {
        match self {
            Self::Bf16 => 1,
            Self::F32Fma => 3,
        }
    }
    pub(super) const fn permits_math(self, math: i32) -> bool {
        match self {
            Self::Bf16 => matches!(math, 0 | 1),
            Self::F32Fma => math == 3,
        }
    }
    pub(super) const fn tag(self) -> u8 {
        match self {
            Self::Bf16 => 0,
            Self::F32Fma => 1,
        }
    }
}
