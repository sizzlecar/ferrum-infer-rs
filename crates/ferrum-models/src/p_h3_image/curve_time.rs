//! Comfy curve-form time conditioning, from ComfyUI
//! `56f179f6c50dad0c4c860ae4da427415c8e86415`,
//! `comfy/ldm/minimax/model.py` lines 744–749.
//!
//! The checkpoint declares an F32 `[grid,basis]` table. This route replaces
//! the dense time MLP and its SiLU, rather than synthesizing missing weights.
//! A declared numerical reference is not proof of equivalence to the dense
//! Diffusers checkpoint, nor to a particular installed Torch CUDA kernel.

use anyhow::{ensure, Context, Result};
use ferrum_runtime::{ElementType, ResolvedTensorLayout, ResolvedTensorSpec};
use serde::Deserialize;

use super::{config::ensure_object, layout::checked_elements};

/// Explicit interpolation arithmetic. ATen's stable branch formula allows
/// compiler FMA contraction; neither source text alone nor a table's shape
/// proves which arithmetic an installed Torch binary actually executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageCurveLerpArithmetic {
    SeparateF32,
    FusedF32,
}

impl PH3ImageCurveLerpArithmetic {
    pub const fn numerical_profile_id(self) -> &'static str {
        match self {
            Self::SeparateF32 => "p_h3_image.comfy_curve.stable_lerp.separate_f32.v1",
            Self::FusedF32 => "p_h3_image.comfy_curve.stable_lerp.fused_f32.v1",
        }
    }
    fn interpolate(self, start: f32, end: f32, weight: f32) -> f32 {
        let difference = end - start;
        if weight < 0.5 {
            match self {
                Self::SeparateF32 => start + weight * difference,
                Self::FusedF32 => weight.mul_add(difference, start),
            }
        } else {
            let complement = 1.0_f32 - weight;
            match self {
                Self::SeparateF32 => end - difference * complement,
                Self::FusedF32 => (-complement).mul_add(difference, end),
            }
        }
    }
}

/// Table geometry declared by the actual checkpoint header. No basis width or
/// grid count is inferred from a model name or from dense `time_embed_dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImageCurveTimeEmbedding {
    grid_points: usize,
    basis_dim: usize,
    table_elements: usize,
}

#[derive(Deserialize)]
struct Header {
    adaln_t_table: TensorHeader,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorHeader {
    dtype: String,
    shape: [u64; 2],
    data_offsets: [u64; 2],
}

impl PH3ImageCurveTimeEmbedding {
    pub fn new(grid_points: usize, basis_dim: usize) -> Result<Self> {
        ensure!(
            grid_points >= 2 && basis_dim > 0,
            "curve time table needs at least two grid points and a positive basis width"
        );
        // pos and the integer index converted back for weight subtraction are
        // FP32. Keep every integer endpoint exact in that declared arithmetic.
        ensure!(
            grid_points - 1 <= 1 << 24,
            "curve table index exceeds exact FP32 integer coordinates"
        );
        let table_elements = checked_elements::<f32>(&[grid_points, basis_dim])?;
        Ok(Self {
            grid_points,
            basis_dim,
            table_elements,
        })
    }

    pub fn from_tensor_spec(spec: &ResolvedTensorSpec) -> Result<Self> {
        let [grid, basis] = spec.dimensions() else {
            anyhow::bail!("curve time table must have rank two");
        };
        ensure!(
            spec.element_type() == ElementType::F32
                && spec.layout() == &ResolvedTensorLayout::Contiguous,
            "curve time table must be contiguous F32"
        );
        Self::new(usize::try_from(*grid)?, usize::try_from(*basis)?)
    }

    /// Consume only a bounded safetensors JSON header, not the full weight
    /// file. Duplicate table declarations and malformed physical extents are
    /// rejected. The file source must separately check header length, complete
    /// file bounds, all other tensors and payload authenticity before reading.
    pub fn from_safetensors_header(bytes: &[u8], max_metadata_bytes: usize) -> Result<Self> {
        ensure!(
            bytes.len() <= max_metadata_bytes,
            "curve checkpoint header exceeds metadata capacity"
        );
        ensure_object(bytes)?;
        let header: Header =
            serde_json::from_slice(bytes).context("invalid Comfy curve table header")?;
        ensure!(
            header.adaln_t_table.dtype == "F32",
            "Comfy curve table must declare F32 storage"
        );
        let [grid, basis] = header.adaln_t_table.shape;
        let descriptor = Self::new(usize::try_from(grid)?, usize::try_from(basis)?)?;
        let [start, end] = header.adaln_t_table.data_offsets;
        ensure!(
            end.checked_sub(start)
                == Some(
                    u64::try_from(descriptor.table_elements)?
                        .checked_mul(4)
                        .context("curve table byte extent overflows")?
                ),
            "Comfy curve table physical extent differs from shape"
        );
        Ok(descriptor)
    }

    pub fn grid_points(self) -> usize {
        self.grid_points
    }
    pub fn basis_dim(self) -> usize {
        self.basis_dim
    }
    pub fn table_elements(self) -> usize {
        self.table_elements
    }
    pub fn output_elements(self, times: usize) -> Result<usize> {
        checked_elements::<f32>(&[times, self.basis_dim])
    }

    /// F32 clamp -> multiplication -> floor -> upper-interval clamp -> F32
    /// subtraction. At t=1 the lower index is grid-2 and the weight is 1.
    /// Finite out-of-range times clamp like the Comfy implementation.
    pub fn interval(self, time: f32) -> Result<(usize, f32)> {
        ensure!(time.is_finite(), "curve time must be finite");
        let position = time.clamp(0.0, 1.0) * (self.grid_points - 1) as f32;
        let lower = (position.floor() as usize).min(self.grid_points - 2);
        Ok((lower, position - lower as f32))
    }

    /// Interpolate independent request times into caller-owned F32
    /// `[times,basis]` storage. No dense time MLP, sinusoidal projection, SiLU,
    /// quantization, scratch tensor or runtime allocation is implicit here.
    /// Validate all table values, intervals and results before any output
    /// write, then repeat the same arithmetic to publish atomically.
    pub fn write_f32(
        self,
        table: &[f32],
        times: &[f32],
        arithmetic: PH3ImageCurveLerpArithmetic,
        output: &mut [f32],
    ) -> Result<()> {
        ensure!(
            !times.is_empty(),
            "curve time conditioning requires a time row"
        );
        ensure!(
            table.len() == self.table_elements
                && output.len() == self.output_elements(times.len())?,
            "curve time table or output length differs from checked geometry"
        );
        ensure!(
            table.iter().all(|value| value.is_finite()),
            "curve time table must be finite F32"
        );
        for &time in times {
            let (lower, weight) = self.interval(time)?;
            for column in 0..self.basis_dim {
                let start = table[lower * self.basis_dim + column];
                let end = table[(lower + 1) * self.basis_dim + column];
                ensure!(
                    arithmetic.interpolate(start, end, weight).is_finite(),
                    "curve interpolation result is not finite"
                );
            }
        }
        for (&time, row) in times.iter().zip(output.chunks_exact_mut(self.basis_dim)) {
            let (lower, weight) = self.interval(time)?;
            for (column, value) in row.iter_mut().enumerate() {
                *value = arithmetic.interpolate(
                    table[lower * self.basis_dim + column],
                    table[(lower + 1) * self.basis_dim + column],
                    weight,
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
