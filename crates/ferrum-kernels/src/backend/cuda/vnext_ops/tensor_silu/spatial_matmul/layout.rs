//! Checked single-head geometry and one reusable, invocation-owned workspace.
use super::*;

pub(super) const QUERY_TILE: u64 = 256;
pub(super) const SCRATCH_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub batch: u64,
    pub channels: u32,
    pub spatial: u32,
    pub tile: u32,
    pub scale: f32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub elements_per_batch: u64,
    pub score_offset: u64,
    pub result_offset: u64,
    pub scratch_bytes: u64,
    pub operations: u64,
}

impl Shape {
    pub fn new(g: SpatialAttentionBf16Geometry) -> Result<Self, String> {
        let spatial = u32::try_from(g.spatial())
            .ok()
            .filter(|&s| s <= i32::MAX as u32)
            .ok_or("spatial GEMM length must fit positive BLAS i32")?;
        let channels = g.channels() as u32;
        let tile = g.spatial().min(QUERY_TILE) as u32;
        let elements_per_batch = g
            .channels()
            .checked_mul(g.spatial())
            .ok_or("spatial GEMM elements overflow")?;
        let output_bytes = g
            .element_count()
            .checked_mul(2)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or("spatial GEMM output exceeds address space")?;
        let input_bytes = output_bytes
            .checked_mul(3)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or("spatial GEMM input exceeds address space")?;
        let align = |n: u64| {
            n.checked_add(SCRATCH_ALIGNMENT - 1)
                .map(|n| n / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT)
                .ok_or_else(|| "spatial GEMM workspace alignment overflows".to_owned())
        };
        let f32_bytes = |a: u64, b: u64| {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| "spatial GEMM workspace extent overflows".to_owned())
        };
        let score_offset = align(f32_bytes(g.channels(), g.spatial())?)?;
        let result_offset = align(
            score_offset
                .checked_add(f32_bytes(g.spatial(), u64::from(tile))?)
                .ok_or("spatial GEMM score offset overflows")?,
        )?;
        let scratch_bytes = align(
            result_offset
                .checked_add(f32_bytes(g.channels(), u64::from(tile))?)
                .ok_or("spatial GEMM result offset overflows")?,
        )?;
        if scratch_bytes > isize::MAX as u64 {
            return Err("spatial GEMM workspace exceeds address space".into());
        }
        // Nominal arithmetic: two GEMMs plus scale/max/exp/sum/div work.
        // It is not a claim about vendor instruction count or FLOPs.
        let operations = u128::from(g.batch())
            * u128::from(g.spatial()).pow(2)
            * (4 * u128::from(g.channels()) + 6);
        let operations =
            u64::try_from(operations).map_err(|_| "spatial GEMM work count overflows")?;
        Ok(Self {
            batch: g.batch(),
            channels,
            spatial,
            tile,
            scale: g.effective_scale(),
            input_bytes,
            output_bytes,
            elements_per_batch,
            score_offset,
            result_offset,
            scratch_bytes,
            operations,
        })
    }

    pub fn launches(self) -> u64 {
        self.batch * (1 + 4 * u64::from(self.spatial).div_ceil(u64::from(self.tile)))
    }
}

pub(super) fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 2 {
        return Err("spatial GEMM requires one QKV input and one output".into());
    }
    for value in bindings {
        if value.weight().is_some_and(|weight| {
            !matches!(
                weight.physical_layout(),
                ferrum_runtime::PhysicalWeightLayout::Dense { .. }
            )
        }) {
            return Err("spatial GEMM requires dense physical tensors".into());
        }
    }
    Shape::new(
        SpatialAttentionBf16Geometry::new(
            binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
            binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
            attributes,
        )
        .map_err(|e| e.to_string())?,
    )
}
