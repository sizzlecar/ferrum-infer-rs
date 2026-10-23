//! Image mRoPE input construction: padding contributes no position, image rows
//! advance H/W axes on the merged grid, and subsequent text starts after the
//! largest spatial axis. This does not infer image slots from token IDs.
use super::{
    Qwen3VLTextConfig, Qwen3VLTextDeepStack, Qwen3VLTextEncoderFamily as Family,
    Qwen3VLTextVisualRun,
};
use anyhow::{ensure, Result};
use ferrum_runtime::*;
use half::bf16;

pub const QWEN3_VL_EDIT_MROPE_PROFILE: &str =
    "qwen3vl.edit.mrope.interleaved.f32_libm-0.2.15.to_bf16.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3VLTextImageSpan {
    run: Qwen3VLTextVisualRun,
    height: u64,
    width: u64,
}
impl Qwen3VLTextImageSpan {
    /// `start` is an absolute flattened text row; grid sizes are before merge.
    pub fn new(start: u64, grid_height: u64, grid_width: u64, merge: u64) -> Result<Self> {
        ensure!(
            merge > 0
                && grid_height > 0
                && grid_width > 0
                && grid_height % merge == 0
                && grid_width % merge == 0,
            "image token grid must be positive and merge aligned"
        );
        let height = grid_height / merge;
        let width = grid_width / merge;
        let length = height
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("merged image rows overflow"))?;
        Ok(Self {
            run: Qwen3VLTextVisualRun::new(start, length)?,
            height,
            width,
        })
    }
    pub fn visual_run(&self) -> &Qwen3VLTextVisualRun {
        &self.run
    }
    pub const fn merged_height(&self) -> u64 {
        self.height
    }
    pub const fn merged_width(&self) -> u64 {
        self.width
    }
}
impl Qwen3VLTextConfig {
    pub fn with_image_spans(self, spans: &[Qwen3VLTextImageSpan]) -> Result<Self> {
        Ok(self.with_deepstack(Qwen3VLTextDeepStack::new(
            spans.iter().map(|s| s.run.clone()).collect(),
        )?)?)
    }
    /// Caller supplies the image token runs used for embeddings and DeepStack.
    /// Both BF16 output tables are reserved atomically; there is no payload-sized
    /// temporary position tensor. The caller's base embeddings remain separate.
    pub fn managed_edit_mrope(
        &self,
        host: &HostMemoryAccount,
        spans: &[Qwen3VLTextImageSpan],
    ) -> Result<[ManagedHostTensor; 2]> {
        self.validate()?;
        let expected = Qwen3VLTextDeepStack::new(spans.iter().map(|s| s.run.clone()).collect())?;
        ensure!(
            self.deepstack() == Some(&expected),
            "mRoPE image spans differ from the configured DeepStack rows"
        );
        ensure!(
            self.layout().sequence_length() <= 1 << 24,
            "mRoPE positions exceed exact FP32 integer range"
        );
        let d = usize::try_from(self.architecture().head_dim())?;
        let tensor = ResolvedTensorSpec::new(
            vec![self.layout().rows(), d as u64],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let length = usize::try_from(tensor.minimum_storage_bytes()?)?;
        let mut reservations = host
            .reserve_many(&[(HostMemoryClass::InputPreprocess, length); 2])?
            .into_iter();
        let mut cos = reservations.next().unwrap().allocate_zeroed()?;
        let mut sin = reservations.next().unwrap().allocate_zeroed()?;
        let sections = self.architecture().mrope_sections();
        let sequence = u64::from(self.layout().sequence_length());
        for (batch, runs) in self.layout().key_runs().iter().enumerate() {
            let mut local = 0_u64;
            let mut next_position = 0_u64;
            let mut image_origin = 0_u64;
            for run in runs {
                for _ in 0..run.length() {
                    let row = batch as u64 * sequence + local;
                    let position = if !run.is_valid() {
                        [0; 3]
                    } else if let Some(image) = spans
                        .iter()
                        .find(|s| s.run.start() <= row && row < s.run.start() + s.run.length())
                    {
                        let j = row - image.run.start();
                        if j == 0 {
                            image_origin = next_position;
                        }
                        let p = [
                            image_origin,
                            image_origin + j / image.width,
                            image_origin + j % image.width,
                        ];
                        if j + 1 == image.run.length() {
                            next_position += image.height.max(image.width);
                        }
                        p
                    } else {
                        let p = [next_position; 3];
                        next_position += 1;
                        p
                    };
                    for pair in 0..d / 2 {
                        let axis = if pair % 3 == 1 && (pair as u64) < sections[1] * 3 {
                            1
                        } else if pair % 3 == 2 && (pair as u64) < sections[2] * 3 {
                            2
                        } else {
                            0
                        };
                        let inverse = 1.0_f32
                            / libm::powf(
                                self.architecture().rope_theta() as f32,
                                (2 * pair) as f32 / d as f32,
                            );
                        let phase = position[axis] as f32 * inverse;
                        let c = bf16::from_f32(libm::cosf(phase)).to_bits().to_le_bytes();
                        let s = bf16::from_f32(libm::sinf(phase)).to_bits().to_le_bytes();
                        for column in [pair, pair + d / 2] {
                            let offset = (row as usize * d + column) * 2;
                            cos.as_mut()[offset..offset + 2].copy_from_slice(&c);
                            sin.as_mut()[offset..offset + 2].copy_from_slice(&s);
                        }
                    }
                    local += 1;
                }
            }
        }
        Ok([
            ManagedHostTensor::new(
                ProgramValueId::new(Family::COSINE_ID)?,
                tensor.clone(),
                cos.freeze(),
            )?,
            ManagedHostTensor::new(ProgramValueId::new(Family::SINE_ID)?, tensor, sin.freeze())?,
        ])
    }
}
