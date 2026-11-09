//! The Comfy BF16 learned-table arithmetic topology, then explicit F32 inputs.
//! Coordinate generation is separately named: F64 endpoint interpolation
//! rounded to F32, not an unverified Torch linspace binary-parity assertion.
use super::*;
use ferrum_quantization::ArchiveTensor;
use half::bf16;
pub const QWEN3_VL_COMFY_VISION_POSITION_PROFILE:&str="qwen3vl.vision_comfy.positions.f64_endpoint_to_f32.bf16_weight_product_leftfold.f32.libm-0.2.15.v1";
fn b(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn taps(i: usize, length: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let p = if length == 1 {
        0.
    } else {
        ((i as f64 * (side - 1) as f64) / (length - 1) as f64) as f32
    };
    let lo = libm::floorf(p) as usize;
    let fraction = p - lo as f32;
    (
        [lo.min(side - 1), (lo + 1).min(side - 1)],
        [1.0 - fraction, fraction],
    )
}
impl Qwen3VLVisionComfyConfig {
    /// The table remains original borrowed BF16 storage. All three F32 inputs
    /// are admitted atomically from this existing account's InputPreprocess.
    pub fn managed_position_inputs(
        &self,
        host: &HostMemoryAccount,
        table: &ArchiveTensor<'_>,
    ) -> Result<[ManagedHostTensor; 3]> {
        let a = self.architecture();
        ensure!(
            table.external_name() == Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME
                && table.element_type() == Some(ElementType::Bf16)
                && table.shape() == [a.position_embeddings(), a.hidden_size()],
            "original visual position table identity differs"
        );
        let h = usize::try_from(a.hidden_size())?;
        let d = usize::try_from(a.head_dim())?;
        let r = usize::try_from(self.rows()?)?;
        let side = usize::try_from(a.position_embeddings().isqrt())?;
        ensure!(
            side <= 1 << 24 && self.grid().height() <= 1 << 24 && self.grid().width() <= 1 << 24,
            "visual position coordinates exceed exact F32 integer range"
        );
        ensure!(
            table
                .bytes()
                .chunks_exact(2)
                .all(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7f80 != 0x7f80),
            "nonfinite learned visual positions"
        );
        let widths = [h, d, d];
        let names = [
            Qwen3VLVisionComfyFamily::POSITIONS_ID,
            Qwen3VLVisionComfyFamily::COSINE_ID,
            Qwen3VLVisionComfyFamily::SINE_ID,
        ];
        let lengths = widths.map(|w| {
            r.checked_mul(w)
                .and_then(|n| n.checked_mul(4))
                .context("position input extent overflow")
        });
        let [p, c, s] = lengths;
        let lengths = [p?, c?, s?];
        let mut reservations = host
            .reserve_many(&lengths.map(|n| (HostMemoryClass::InputPreprocess, n)))?
            .into_iter();
        let mut pos = reservations.next().unwrap().allocate_zeroed()?;
        let mut cos = reservations.next().unwrap().allocate_zeroed()?;
        let mut sin = reservations.next().unwrap().allocate_zeroed()?;
        let merge = usize::try_from(a.merge_size())?;
        let height = usize::try_from(self.grid().height())?;
        let width = usize::try_from(self.grid().width())?;
        let mut pairs = [(0usize, 0f32); 128];
        for (pair, p) in pairs[..d / 2].iter_mut().enumerate() {
            let axis = pair / (d / 4);
            let frequency = pair % (d / 4);
            *p = (
                axis,
                1.0 / libm::powf(10000., (2 * frequency) as f32 / (d / 2) as f32),
            );
        }
        for local in 0..r {
            let y = local / (merge * merge * (width / merge)) * merge + (local / merge) % merge;
            let x = (local / (merge * merge)) % (width / merge) * merge + local % merge;
            let (ys, yw) = taps(y, height, side);
            let (xs, xw) = taps(x, width, side);
            let weights = [
                b(yw[0] * xw[0]),
                b(yw[0] * xw[1]),
                b(yw[1] * xw[0]),
                b(yw[1] * xw[1]),
            ];
            for channel in 0..h {
                let read = |yi: usize, xi: usize| {
                    let o = ((ys[yi] * side + xs[xi]) * h + channel) * 2;
                    bf16::from_bits(u16::from_le_bytes(
                        table.bytes()[o..o + 2].try_into().unwrap(),
                    ))
                    .to_f32()
                };
                let terms = [
                    b(read(0, 0) * weights[0]),
                    b(read(0, 1) * weights[1]),
                    b(read(1, 0) * weights[2]),
                    b(read(1, 1) * weights[3]),
                ];
                let value = b(b(b(terms[0] + terms[1]) + terms[2]) + terms[3]);
                ensure!(
                    value.is_finite(),
                    "BF16 visual position interpolation overflow"
                );
                let o = (local * h + channel) * 4;
                pos.as_mut()[o..o + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (pair, &(axis, inverse)) in pairs[..d / 2].iter().enumerate() {
                let phase = if axis == 0 { y } else { x } as f32 * inverse;
                let cv = libm::cosf(phase).to_le_bytes();
                let sv = libm::sinf(phase).to_le_bytes();
                for column in [pair, pair + d / 2] {
                    let o = (local * d + column) * 4;
                    cos.as_mut()[o..o + 4].copy_from_slice(&cv);
                    sin.as_mut()[o..o + 4].copy_from_slice(&sv);
                }
            }
        }
        let mut specs = names
            .into_iter()
            .zip(widths)
            .map(|(id, w)| {
                Ok((
                    ProgramValueId::new(id)?,
                    ResolvedTensorSpec::new(
                        vec![r as u64, w as u64],
                        ElementType::F32,
                        ResolvedTensorLayout::Contiguous,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let (id, t) = specs.next().unwrap();
        let p = ManagedHostTensor::new(id, t, pos.freeze())?;
        let (id, t) = specs.next().unwrap();
        let c = ManagedHostTensor::new(id, t, cos.freeze())?;
        let (id, t) = specs.next().unwrap();
        let s = ManagedHostTensor::new(id, t, sin.freeze())?;
        Ok([p, c, s])
    }
}
