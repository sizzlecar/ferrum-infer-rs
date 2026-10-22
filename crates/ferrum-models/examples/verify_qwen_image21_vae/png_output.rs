//! A bounded, streaming PNG artifact path using the already cached png crate.
//! PNG 0.18.1's stream writer holds three scanlines and a fixed IDAT buffer.
//! We cap RGBA rows at 64 KiB, use 16 KiB IDAT chunks, and never materialize an
//! image or encoded file Vec. The library's row/deflate/decoder state is outside
//! the managed tensor-backing account; this is not a whole-process RSS claim.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, BufReader, Write},
    path::Path,
};

pub const MAX_ROW_BYTES: usize = 64 * 1024;
const CHUNK_BYTES: usize = 16 * 1024;
const DECODE_LIMIT: usize = 8 * 1024 * 1024;

pub fn dimensions(width: u64, height: u64) -> Result<(u32, u32)> {
    let width = u32::try_from(width).context("PNG width exceeds u32")?;
    let height = u32::try_from(height).context("PNG height exceeds u32")?;
    let row = usize::try_from(width)?
        .checked_mul(4)
        .context("PNG row overflows")?;
    ensure!(
        width != 0 && height != 0 && row <= MAX_ROW_BYTES,
        "PNG requires positive dimensions and an RGBA row <= {MAX_ROW_BYTES} bytes"
    );
    row.checked_mul(usize::try_from(height)?)
        .context("PNG extent exceeds address space")?;
    Ok((width, height))
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
    {
        let mut encoder = png::Encoder::new(&mut sink, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::NoCompression);
        encoder.set_filter(png::Filter::NoFilter);
        let mut writer = encoder.write_header()?;
        {
            let mut stream = writer.stream_writer_with_size(CHUNK_BYTES)?;
            emit(&mut stream)?;
            stream.finish()?;
        }
        writer.finish()?;
    }
    sink.flush()?;
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
    fn streaming_png_roundtrip_keeps_channels_alpha_dimensions_and_digest() {
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
        let mut reader = png::Decoder::new(std::io::Cursor::new(encoded))
            .read_info()
            .unwrap();
        let mut decoded = [0_u8; 16];
        let info = reader.next_frame(&mut decoded).unwrap();
        assert_eq!(
            (info.width, info.height, info.color_type),
            (2, 2, png::ColorType::Rgba)
        );
        assert_eq!(decoded, pixels);
        assert!(write(&path, 2, 2, |_| Ok(())).is_err());
    }
    #[test]
    fn oversized_rows_and_incomplete_stream_are_errors_not_artifacts() {
        assert!(dimensions(0, 1).is_err());
        assert!(dimensions((MAX_ROW_BYTES / 4 + 1) as u64, 1).is_err());
        assert!(dimensions(1, u32::MAX as u64 + 1).is_err());
        let dir = tempfile::tempdir().unwrap();
        assert!(write(&dir.path().join("short.png"), 2, 2, |stream| {
            stream.write_all(&[0; 15])?;
            Ok(())
        })
        .is_err());
    }
}
