//! Model-independent pixel streaming with an explicit stored numeric domain.
//! A view borrows the existing immutable Output allocation; it creates no
//! runtime, memory account, image copy or encoder scratch.
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::SingleFrameLatentShape;
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBytes, ResolvedTensorLayout,
    ResolvedTensorSpec,
};
use half::bf16;
use std::io::Write;

mod bf16_staged;
pub(crate) use bf16_staged::rgba8_finite;

/// The producer declares storage, channels, domain and byte-conversion math.
/// Neither a model name nor the consumer's client selects pixel interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImagePixelEncoding {
    /// BF16 RGBA; clamp signed values, half and shift with BF16 boundaries,
    /// then F32 scale and ties-to-even. Alpha follows the same conversion.
    Bf16SignedRgbaStaged,
    /// F32 RGB already in [0,1]; F32 scale by255 and ties-to-even. The emitted
    /// alpha is opaque. No rescaling from a signed/raw decoder domain occurs.
    F32UnitRgbRne,
}

/// An immutable, validated image artifact with its original Host Output charge.
/// It may originate in a component readback or managed postprocessing. Storage
/// remains in the producer's declared domain until the shared RGBA8 writer runs.
#[derive(Clone)]
pub struct OwnedImagePixels {
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
    encoding: ImagePixelEncoding,
}
impl OwnedImagePixels {
    pub fn new(
        host: &HostMemoryAccount,
        tensor: ResolvedTensorSpec,
        bytes: ManagedHostBytes,
        encoding: ImagePixelEncoding,
    ) -> Result<Self> {
        ManagedImagePixels::new(host, &tensor, &bytes, encoding)?;
        Ok(Self {
            tensor,
            bytes,
            encoding,
        })
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    pub fn encoding(&self) -> ImagePixelEncoding {
        self.encoding
    }
    pub fn view(&self, host: &HostMemoryAccount) -> Result<ManagedImagePixels<'_>> {
        ManagedImagePixels::new(host, &self.tensor, &self.bytes, self.encoding)
    }
}
impl ImagePixelEncoding {
    fn element_type(self) -> ElementType {
        match self {
            Self::Bf16SignedRgbaStaged => ElementType::Bf16,
            Self::F32UnitRgbRne => ElementType::F32,
        }
    }
    fn channels(self) -> usize {
        match self {
            Self::Bf16SignedRgbaStaged => 4,
            Self::F32UnitRgbRne => 3,
        }
    }
    fn element_bytes(self) -> usize {
        match self {
            Self::Bf16SignedRgbaStaged => 2,
            Self::F32UnitRgbRne => 4,
        }
    }
    fn decode(self, bytes: &[u8]) -> f32 {
        match self {
            Self::Bf16SignedRgbaStaged => {
                bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
            }
            Self::F32UnitRgbRne => f32::from_le_bytes(bytes.try_into().unwrap()),
        }
    }
    fn rgba8(self, bytes: &[u8]) -> u8 {
        match self {
            Self::Bf16SignedRgbaStaged => rgba8_finite(bf16::from_bits(u16::from_le_bytes(
                bytes.try_into().unwrap(),
            ))),
            Self::F32UnitRgbRne => (self.decode(bytes) * 255.0).round_ties_even() as u8,
        }
    }
}

pub struct ManagedImagePixels<'a> {
    bytes: &'a ManagedHostBytes,
    encoding: ImagePixelEncoding,
    batch: usize,
    height: usize,
    width: usize,
    positions: usize,
    frame_bytes: usize,
}
impl<'a> ManagedImagePixels<'a> {
    /// Complete shape, owner, finite-value and declared-domain validation
    /// precedes exposure of any writer. Input backing is never reclassified.
    pub fn new(
        host: &HostMemoryAccount,
        tensor: &ResolvedTensorSpec,
        bytes: &'a ManagedHostBytes,
        encoding: ImagePixelEncoding,
    ) -> Result<Self> {
        ensure!(
            host.owns(bytes) && bytes.class() == HostMemoryClass::Output,
            "image pixels require the original Host Output owner"
        );
        ensure!(
            tensor.element_type() == encoding.element_type()
                && tensor.layout() == &ResolvedTensorLayout::Contiguous,
            "image storage differs from the declared pixel encoding"
        );
        let [batch, channels, height, width] = tensor.dimensions() else {
            anyhow::bail!("image pixels require exact NCHW dimensions");
        };
        ensure!(
            *channels == encoding.channels() as u64,
            "image channels differ from the declared pixel encoding"
        );
        let shape = SingleFrameLatentShape::new(
            usize::try_from(*batch)?,
            encoding.channels(),
            usize::try_from(*height)?,
            usize::try_from(*width)?,
        )?;
        ensure!(
            tensor.minimum_storage_bytes()? == bytes.len() as u64,
            "image backing differs from its exact tensor extent"
        );
        for raw in bytes.bytes().chunks_exact(encoding.element_bytes()) {
            let value = encoding.decode(raw);
            ensure!(value.is_finite(), "image pixels contain a nonfinite value");
            if encoding == ImagePixelEncoding::F32UnitRgbRne {
                ensure!(
                    (0.0..=1.0).contains(&value),
                    "image RGB values are outside the declared unit domain"
                );
            }
        }
        let [batch, _, height, width] = shape.nchw_dims();
        let positions = shape.sequence_dims()[1];
        let frame_bytes = positions
            .checked_mul(4)
            .filter(|&n| n <= isize::MAX as usize)
            .context("RGBA frame extent exceeds Host address space")?;
        Ok(Self {
            bytes,
            encoding,
            batch,
            height,
            width,
            positions,
            frame_bytes,
        })
    }
    /// Dimensions of the RGBA8 representation consumed by the shared PNG sink.
    /// RGB storage gains an opaque alpha channel only during streaming.
    pub fn rgba_dimensions(&self) -> [usize; 4] {
        [self.batch, 4, self.height, self.width]
    }
    pub fn encoding(&self) -> ImagePixelEncoding {
        self.encoding
    }
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Caller-owned scratch is bounded externally. Invalid indices/scratch
    /// fail before writes; a writer I/O failure may leave a partial sink.
    pub fn write_frame_rgba8<W: Write + ?Sized>(
        &self,
        frame: usize,
        scratch: &mut [u8],
        writer: &mut W,
    ) -> Result<()> {
        ensure!(
            frame < self.batch && !scratch.is_empty(),
            "image writer needs a valid frame and nonempty scratch"
        );
        // Select the declared conversion once per frame, keeping the existing
        // BF16 lookup loop free of per-pixel format dispatch.
        match self.encoding {
            ImagePixelEncoding::Bf16SignedRgbaStaged => {
                self.write_encoded::<W, 4, 2>(frame, scratch, writer, |b| {
                    ImagePixelEncoding::Bf16SignedRgbaStaged.rgba8(b)
                })
            }
            ImagePixelEncoding::F32UnitRgbRne => {
                self.write_encoded::<W, 3, 4>(frame, scratch, writer, |b| {
                    ImagePixelEncoding::F32UnitRgbRne.rgba8(b)
                })
            }
        }
    }

    fn write_encoded<W: Write + ?Sized, const CHANNELS: usize, const ELEMENT_BYTES: usize>(
        &self,
        frame: usize,
        scratch: &mut [u8],
        writer: &mut W,
        convert: impl Fn(&[u8]) -> u8,
    ) -> Result<()> {
        let frame_start = frame * self.positions * CHANNELS;
        let mut used = 0;
        for position in 0..self.positions {
            for channel in 0..4 {
                scratch[used] = if channel == CHANNELS {
                    255
                } else {
                    let index = (frame_start + channel * self.positions + position) * ELEMENT_BYTES;
                    convert(&self.bytes.bytes()[index..index + ELEMENT_BYTES])
                };
                used += 1;
                if used == scratch.len() {
                    writer.write_all(&scratch[..used])?;
                    used = 0;
                }
            }
        }
        if used != 0 {
            writer.write_all(&scratch[..used])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
