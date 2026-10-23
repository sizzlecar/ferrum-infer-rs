//! File durability and independent decode checks around the shared PNG writer.
//! The codec's bounded workspace remains outside the managed tensor-backing
//! account; see ferrum_output::png for the exact allocation scope.
use anyhow::{ensure, Context, Result};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, BufReader, Write},
    path::Path,
};

pub use ferrum_output::png::MAX_RGBA_ROW_BYTES as MAX_ROW_BYTES;
const DECODE_LIMIT: usize = 8 * 1024 * 1024;

pub fn dimensions(width: u64, height: u64) -> Result<(u32, u32)> {
    let dimensions = PngDimensions::new(width, height)?;
    PngLimits::default().validate(dimensions)?;
    Ok((dimensions.width(), dimensions.height()))
}

struct DigestWriter<W> {
    inner: W,
    hash: Sha256,
    bytes: u64,
}
impl<W: Write> Write for DigestWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("PNG byte count overflow"))?;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Debug, Serialize)]
pub struct PngArtifact {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub decoded_shape: [u32; 3],
    pub decoded_rgba8_sha256: String,
    pub decoded_bytes: u64,
    pub color: &'static str,
    pub bit_depth: u8,
    pub alpha: &'static str,
    pub encoding: &'static str,
}

/// The caller emits exact interleaved RGBA8 pixels in arbitrary chunks. Both
/// stream and container finish errors are checked before decoding the file.
pub fn write(
    path: &Path,
    width: u64,
    height: u64,
    emit: impl FnOnce(&mut dyn Write) -> Result<()>,
) -> Result<PngArtifact> {
    let (width, height) = dimensions(width, height)?;
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut sink = DigestWriter {
        inner: file,
        hash: Sha256::new(),
        bytes: 0,
    };
    let receipt = write_rgba8_png(
        &mut sink,
        PngDimensions::new(u64::from(width), u64::from(height))?,
        PngLimits::default(),
        |stream| emit(stream).map_err(io::Error::other),
    )?;
    ensure!(
        receipt.encoded_bytes() == sink.bytes,
        "PNG sink byte count differs"
    );
    sink.inner.sync_all()?;
    let decoded = decode(path, width, height)?;
    Ok(PngArtifact {
        file: path
            .file_name()
            .and_then(|n| n.to_str())
            .context("PNG artifact name requires UTF-8")?
            .into(),
        bytes: sink.bytes,
        sha256: format!("{:x}", sink.hash.finalize()),
        decoded_shape: [height, width, 4],
        decoded_rgba8_sha256: decoded.0,
        decoded_bytes: decoded.1,
        color: "RGBA",
        bit_depth: 8,
        alpha: "preserved straight alpha; no background blend or premultiplication",
        encoding: "png-0.18.1; noninterlaced; filter none; stored DEFLATE; 16 KiB IDAT buffer",
    })
}

fn decode(path: &Path, width: u32, height: u32) -> Result<(String, u64)> {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    decoder.set_limits(png::Limits {
        bytes: DECODE_LIMIT,
    });
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    ensure!(
        info.width == width
            && info.height == height
            && info.color_type == png::ColorType::Rgba
            && info.bit_depth == png::BitDepth::Eight
            && !info.interlaced,
        "encoded PNG has unexpected dimensions/color/layout"
    );
    let row_bytes = usize::try_from(width)?
        .checked_mul(4)
        .context("decoded row overflows")?;
    let mut rows = 0_u32;
    let mut bytes = 0_u64;
    let mut digest = Sha256::new();
    while let Some(row) = reader.next_row()? {
        ensure!(
            row.data().len() == row_bytes && rows < height,
            "decoded PNG row differs from declared extent"
        );
        digest.update(row.data());
        bytes = bytes
            .checked_add(u64::try_from(row_bytes)?)
            .context("decoded byte count overflows")?;
        rows += 1;
    }
    ensure!(rows == height, "PNG decoded fewer rows than declared");
    reader.finish()?;
    Ok((format!("{:x}", digest.finalize()), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_artifact_is_verified_hashed_and_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rgba.png");
        let pixels = [255, 0, 0, 0, 0, 255, 0, 128, 0, 0, 255, 255, 11, 22, 33, 17];
        let artifact = write(&path, 2, 2, |stream| {
            // Splits both a pixel and a scanline; no assumption of row writes.
            for chunk in pixels.chunks(3) {
                stream.write_all(chunk)?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(artifact.decoded_shape, [2, 2, 4]);
        assert_eq!(artifact.decoded_bytes, 16);
        assert_eq!(
            artifact.decoded_rgba8_sha256,
            format!("{:x}", Sha256::digest(pixels))
        );
        let encoded = std::fs::read(&path).unwrap();
        assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(&encoded)));
        assert_eq!(artifact.bytes, encoded.len() as u64);
        assert!(write(&path, 2, 2, |_| Ok(())).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), encoded);
    }
    #[test]
    fn oversized_rows_do_not_create_files_and_incomplete_files_have_no_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let oversized = dir.path().join("oversized.png");
        assert!(write(&oversized, (MAX_ROW_BYTES / 4 + 1) as u64, 1, |_| Ok(())).is_err());
        assert!(!oversized.exists());
        assert!(write(&dir.path().join("short.png"), 2, 2, |stream| {
            stream.write_all(&[0; 15])?;
            Ok(())
        })
        .is_err());
    }
}
