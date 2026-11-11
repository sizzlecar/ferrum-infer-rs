//! HF learned-table gather uses F32 interpolation weights/products/sum, then
//! the actual native F32 visual dtype. No intermediate BF16 rounding occurs.
use super::*;
use ferrum_quantization::ArchiveTensor;
use half::bf16;
pub const QWEN3_VL_HF_POSITION_PROFILE: &str =
    "qwen3_5.hf_input.positions.f32_coords_weights_products_leftfold.libm0215_axial.f32.v1";
impl Qwen3VLVisionHfConfig {
    fn position_extents(&self) -> Result<[usize; 3]> {
        let rows = self.rows()?;
        let extent = |width: u64| -> Result<usize> {
            Ok(usize::try_from(
                rows.checked_mul(width)
                    .and_then(|n| n.checked_mul(4))
                    .context("HF visual position extent overflow")?,
            )?)
        };
        Ok([
            extent(self.architecture().hidden_size())?,
            extent(self.architecture().head_dim())?,
            extent(self.architecture().head_dim())?,
        ])
    }
    /// All three independent physical grants, excluding the borrowed original
    /// BF16 table and already-live patch/decoded-image owners.
    pub fn position_input_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        self.position_extents()?
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(host.backing_bytes(n)?)
                    .context("HF physical position capacity overflow")
            })
    }
    pub fn managed_position_inputs(
        &self,
        host: &HostMemoryAccount,
        table: &ArchiveTensor<'_>,
    ) -> Result<[ManagedHostTensor; 3]> {
        self.check()?;
        let a = self.architecture();
        ensure!(
            table.external_name() == super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME
                && table.element_type() == Some(ElementType::Bf16)
                && table.shape() == [a.position_embeddings(), a.hidden_size()],
            "original HF visual position identity/dtype/shape differs"
        );
        let d = usize::try_from(a.head_dim())?;
        let h = usize::try_from(a.hidden_size())?;
        let rows = usize::try_from(self.rows()?)?;
        let side = usize::try_from(a.position_embeddings().isqrt())?;
        ensure!(
            d <= 256
                && d % 4 == 0
                && side <= 1 << 24
                && self.grid().height() <= 1 << 24
                && self.grid().width() <= 1 << 24,
            "HF position coordinates exceed declared numerical profile"
        );
        ensure!(
            table
                .bytes()
                .chunks_exact(2)
                .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80),
            "nonfinite original learned visual positions"
        );
        let lengths = self.position_extents()?;
        let mut grants = host
            .reserve_many(&lengths.map(|n| (HostMemoryClass::InputPreprocess, n)))?
            .into_iter();
        let mut pos = grants.next().unwrap().allocate_zeroed()?;
        let mut cos = grants.next().unwrap().allocate_zeroed()?;
        let mut sin = grants.next().unwrap().allocate_zeroed()?;
        let merge = a.merge_size() as usize;
        let width = self.grid().width() as usize;
        let height = self.grid().height() as usize;
        for row in 0..rows {
            let y = row / (merge * merge * (width / merge)) * merge + (row / merge) % merge;
            let x = (row / (merge * merge)) % (width / merge) * merge + row % merge;
            let (ys, yw) = taps(y, height, side);
            let (xs, xw) = taps(x, width, side);
            let weights = [yw[0] * xw[0], yw[0] * xw[1], yw[1] * xw[0], yw[1] * xw[1]];
            let locations = [
                ys[0] * side + xs[0],
                ys[0] * side + xs[1],
                ys[1] * side + xs[0],
                ys[1] * side + xs[1],
            ];
            for channel in 0..h {
                let mut value = 0f32;
                for tap in 0..4 {
                    let o = (locations[tap] * h + channel) * 2;
                    let v = bf16::from_bits(u16::from_le_bytes(
                        table.bytes()[o..o + 2].try_into().unwrap(),
                    ))
                    .to_f32();
                    value += v * weights[tap];
                }
                ensure!(value.is_finite(), "HF F32 position interpolation overflow");
                let o = (row * h + channel) * 4;
                pos.as_mut()[o..o + 4].copy_from_slice(&value.to_le_bytes());
            }
            for pair in 0..d / 2 {
                let frequency = pair % (d / 4);
                let coordinate = if pair < d / 4 { y } else { x };
                let inv = 1. / libm::powf(10000., (2 * frequency) as f32 / (d / 2) as f32);
                let phase = coordinate as f32 * inv;
                for column in [pair, pair + d / 2] {
                    let o = (row * d + column) * 4;
                    cos.as_mut()[o..o + 4].copy_from_slice(&libm::cosf(phase).to_le_bytes());
                    sin.as_mut()[o..o + 4].copy_from_slice(&libm::sinf(phase).to_le_bytes());
                }
            }
        }
        let input =
            |name: &str, width: usize, bytes: ManagedHostBytes| -> Result<ManagedHostTensor> {
                Ok(ManagedHostTensor::new(
                    ProgramValueId::new(name)?,
                    ResolvedTensorSpec::new(
                        vec![rows as u64, width as u64],
                        ElementType::F32,
                        ResolvedTensorLayout::Contiguous,
                    )?,
                    bytes,
                )?)
            };
        Ok([
            input(Qwen3VLVisionHfFamily::POSITIONS_ID, h, pos.freeze())?,
            input(Qwen3VLVisionHfFamily::COSINE_ID, d, cos.freeze())?,
            input(Qwen3VLVisionHfFamily::SINE_ID, d, sin.freeze())?,
        ])
    }
}
fn taps(index: usize, length: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    // The fixed HF helper computes its closed form directly in F32, rather
    // than evaluating a F64 endpoint interpolation and converting once.
    let coordinate = index as f32 * (side - 1) as f32 / length.saturating_sub(1).max(1) as f32;
    let low = libm::floorf(coordinate) as usize;
    let fraction = coordinate - low as f32;
    (
        [low.min(side - 1), (low + 1).min(side - 1)],
        [1. - fraction, fraction],
    )
}
