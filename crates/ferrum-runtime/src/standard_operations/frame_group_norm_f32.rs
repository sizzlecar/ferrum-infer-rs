//! Independent per-frame channel/spatial GroupNorm+SiLU in NCTHW storage.
//! Statistics and affine arithmetic are the ordinary F32 NCHW contract;
//! changing temporal layout does not pool samples from different frames.
use super::*;
use crate::{ResolvedTensorLayout, ResolvedTensorSpec};

pub const FRAME_GROUP_NORM_SILU_NCTHW_F32_OPERATION_ID: &str =
    "operation.frame_group_norm_silu.ncthw.f32";
pub const FRAME_GROUP_NORM_SILU_NCTHW_F32_CAPABILITY_ID: &str =
    "capability.operation.frame_group_norm_silu.ncthw.f32";
pub fn frame_group_norm_silu_ncthw_f32_contract() -> Result<StandardOperationContract, VNextError> {
    let mut d = group_norm_silu_nchw_f32_contract()?.descriptor;
    d.id = OperationId::new(FRAME_GROUP_NORM_SILU_NCTHW_F32_OPERATION_ID)?;
    d.provider = provider_requirement(
        FRAME_GROUP_NORM_SILU_NCTHW_F32_CAPABILITY_ID,
        ContractVersion::new(1, 0),
    )?;
    let dims = vec![
        symbol("batch"),
        symbol("channels"),
        symbol("frames"),
        symbol("height"),
        symbol("width"),
    ];
    d.inputs[0] = contiguous_tensor(dims.clone(), [ElementType::F32], TensorAccess::Read)?;
    d.outputs[0] = contiguous_tensor(dims, [ElementType::F32], TensorAccess::Write)?;
    d.validate()?;
    Ok(StandardOperationContract { descriptor: d })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGroupNormSiluNcthwF32Geometry {
    shape: [u64; 5],
    statistics: GroupNormSiluNchwF32Geometry,
}
impl FrameGroupNormSiluNcthwF32Geometry {
    pub fn new(
        x: &ResolvedTensorSpec,
        gamma: &ResolvedTensorSpec,
        beta: &ResolvedTensorSpec,
        y: &ResolvedTensorSpec,
        groups: u64,
        epsilon: CanonicalRational,
    ) -> Result<Self, VNextError> {
        let shape: [u64; 5] = x
            .dimensions()
            .try_into()
            .map_err(|_| invalid("requires rank-five NCTHW"))?;
        if x != y
            || shape.contains(&0)
            || x.element_type() != ElementType::F32
            || x.layout() != &ResolvedTensorLayout::Contiguous
            || x.minimum_storage_bytes()? > isize::MAX as u64
        {
            return Err(invalid(
                "requires identical addressable F32 contiguous input/output",
            ));
        }
        let frames = shape[0]
            .checked_mul(shape[2])
            .ok_or_else(|| invalid("batch/frame product overflows"))?;
        let flat = ResolvedTensorSpec::new(
            vec![frames, shape[1], shape[3], shape[4]],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let statistics =
            GroupNormSiluNchwF32Geometry::new(&flat, gamma, beta, &flat, groups, epsilon)?;
        Ok(Self { shape, statistics })
    }
    pub const fn shape(self) -> [u64; 5] {
        self.shape
    }
    pub const fn groups(self) -> u64 {
        self.statistics.groups()
    }
    pub const fn elements(self) -> u64 {
        self.statistics.elements()
    }
    pub const fn group_elements(self) -> u64 {
        self.statistics.group_elements()
    }
    pub fn effective_epsilon(self) -> f32 {
        self.statistics.effective_epsilon()
    }
    pub const fn logical_arithmetic_operations(self) -> u64 {
        self.statistics.logical_arithmetic_operations()
    }
    /// Group ordinal is N,T,G; each local index is C-per-group,H,W.
    pub fn element_offset(self, group: u64, local: u64) -> Option<u64> {
        let [n, c, t, h, w] = self.shape;
        let g = self.groups();
        let spatial = h * w;
        if group >= n * t * g || local >= self.group_elements() {
            return None;
        }
        let batch = group / (t * g);
        let frame = (group / g) % t;
        let channel = (group % g) * (c / g) + local / spatial;
        Some(((batch * c + channel) * t + frame) * spatial + local % spatial)
    }
}
fn invalid(s: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("framewise F32 normalization: {s}"),
    }
}
