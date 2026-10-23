//! Governed input adapter for the pinned align-corners bilinear learned grid
//! and axial rotary formulas. CPU libm and the four-tap FP32 left fold are an
//! explicit implementation profile, not a Torch/CUDA bitwise-parity claim.
use super::{Qwen3VLVisionConfig, Qwen3VLVisionFamily as Family};
use anyhow::{ensure, Result};
use ferrum_quantization::SafetensorsTensor;
use ferrum_runtime::*;
use half::bf16;

pub const QWEN3_VL_VISION_POSITION_PROFILE: &str =
    "qwen3vl.vision.positions.bilinear_align_corners.f32_4tap_leftfold.libm-0.2.15.v1";
impl Qwen3VLVisionConfig {
    /// Returns positions, cosine and sine in that order. The source is borrowed
    /// mmap storage; all produced bytes are reserved atomically from InputPreprocess.
    pub fn managed_position_inputs(
        &self,
        host: &HostMemoryAccount,
        position_table: &SafetensorsTensor<'_>,
    ) -> Result<[ManagedHostTensor; 3]> {
        ensure!(
            position_table.external_name() == Family::POSITION_EXTERNAL_NAME,
            "unexpected learned vision position source"
        );
        ensure!(
            position_table.element_type() == Some(ElementType::Bf16),
            "learned vision positions must be BF16"
        );
        ensure!(
            position_table.shape()
                == [
                    self.architecture().position_embeddings(),
                    self.architecture().hidden_size()
                ],
            "learned position shape differs from architecture"
        );
        self.position_inputs_bytes(host, position_table.bytes())
    }
    pub(super) fn position_inputs_bytes(
        &self,
        host: &HostMemoryAccount,
        table: &[u8],
    ) -> Result<[ManagedHostTensor; 3]> {
        self.validate()?;
        let a = self.architecture();
        let h = usize::try_from(a.hidden_size())?;
        let d = usize::try_from(a.head_dim())?;
        let side = usize::try_from(a.position_embeddings().isqrt())?;
        let expected = usize::try_from(a.position_embeddings())?
            .checked_mul(h)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("position table bytes overflow"))?;
        ensure!(
            table.len() == expected,
            "position table byte extent differs"
        );
        ensure!(
            table
                .chunks_exact(2)
                .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 != 0x7f80),
            "position table contains nonfinite BF16"
        );
        ensure!(
            side <= 1 << 24
                && self
                    .grids()
                    .iter()
                    .all(|g| g.height() <= 1 << 24 && g.width() <= 1 << 24),
            "position coordinates exceed exact FP32 integer range"
        );
        let ids = [Family::POSITIONS_ID, Family::COSINE_ID, Family::SINE_ID];
        let specs = self.input_tensor_specs()?;
        let tensors = ids
            .iter()
            .map(|id| {
                let s = &specs[&ProgramValueId::new(*id)?];
                ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let requests = tensors
            .iter()
            .map(|t| {
                Ok((
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(t.minimum_storage_bytes()?)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reserved = host.reserve_many(&requests)?.into_iter();
        let mut pos = reserved.next().unwrap().allocate_zeroed()?;
        let mut cos = reserved.next().unwrap().allocate_zeroed()?;
        let mut sin = reserved.next().unwrap().allocate_zeroed()?;
        let merge = usize::try_from(a.merge_size())?;
        // Validated architectures bound head_dim to 256. Reuse pair parameters
        // within this call without retaining any image or position-table payload.
        let mut rotary_pairs = [(0_usize, 0.0_f32); 128];
        for (pair, parameters) in rotary_pairs[..d / 2].iter_mut().enumerate() {
            let axis = pair / (d / 4);
            let frequency = pair % (d / 4);
            let inverse = 1.0_f32 / libm::powf(10000.0, (2 * frequency) as f32 / (d / 2) as f32);
            *parameters = (axis, inverse);
        }
        let mut global = 0;
        for grid in self.grids() {
            let height = usize::try_from(grid.height())?;
            let width = usize::try_from(grid.width())?;
            for local in 0..usize::try_from(grid.rows()?)? {
                let row =
                    local / (merge * merge * (width / merge)) * merge + (local / merge) % merge;
                let col = (local / (merge * merge)) % (width / merge) * merge + local % merge;
                let (yt, yw) = taps(row, height, side);
                let (xt, xw) = taps(col, width, side);
                for channel in 0..h {
                    let mut sum = 0.0_f32;
                    for iy in 0..2 {
                        for ix in 0..2 {
                            let offset = ((yt[iy] * side + xt[ix]) * h + channel) * 2;
                            let value = bf16::from_bits(u16::from_le_bytes([
                                table[offset],
                                table[offset + 1],
                            ]))
                            .to_f32();
                            let weight = yw[iy] * xw[ix];
                            let term = value * weight;
                            sum += term;
                        }
                    }
                    let bits = bf16::from_f32(sum).to_bits();
                    ensure!(
                        bits & 0x7f80 != 0x7f80,
                        "interpolated position overflows BF16"
                    );
                    let offset = (global * h + channel) * 2;
                    pos.as_mut()[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                }
                for (pair, &(axis, inv)) in rotary_pairs[..d / 2].iter().enumerate() {
                    let phase = if axis == 0 { row } else { col } as f32 * inv;
                    let cv = libm::cosf(phase).to_le_bytes();
                    let sv = libm::sinf(phase).to_le_bytes();
                    for channel in [pair, pair + d / 2] {
                        let offset = (global * d + channel) * 4;
                        cos.as_mut()[offset..offset + 4].copy_from_slice(&cv);
                        sin.as_mut()[offset..offset + 4].copy_from_slice(&sv);
                    }
                }
                global += 1;
            }
        }
        let mut tensors = tensors.into_iter();
        Ok([
            ManagedHostTensor::new(
                ProgramValueId::new(ids[0])?,
                tensors.next().unwrap(),
                pos.freeze(),
            )?,
            ManagedHostTensor::new(
                ProgramValueId::new(ids[1])?,
                tensors.next().unwrap(),
                cos.freeze(),
            )?,
            ManagedHostTensor::new(
                ProgramValueId::new(ids[2])?,
                tensors.next().unwrap(),
                sin.freeze(),
            )?,
        ])
    }
}
fn taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = (index as f32 * (side - 1) as f32) / (size - 1).max(1) as f32;
    let floor = libm::floorf(source);
    let lower = floor as usize;
    let fraction = source - floor;
    (
        [lower.min(side - 1), (lower + 1).min(side - 1)],
        [
            (1.0 - fraction.abs()).max(0.0),
            (1.0 - (fraction - 1.0).abs()).max(0.0),
        ],
    )
}
