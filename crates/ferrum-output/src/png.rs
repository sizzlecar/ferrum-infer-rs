//! Streaming RGBA8 PNG encoding into a caller-owned writer.
//!
//! No complete image or encoded PNG buffer is allocated here. In png 0.18.1,
//! the stream writer allocates three RGBA scanlines and one IDAT buffer. These
//! are capped by [`PngLimits`]; DEFLATE state and allocator overhead are additional
//! library allocations. These limits are not a complete workspace/RSS bound or
//! a Host-memory reservation. The caller retains and accounts for source pixels,
//! conversion scratch and destination backing, including any HTTP body lifetime.
//!
//! The encoding is noninterlaced RGBA8, with no filter and stored DEFLATE. Alpha
//! bytes are preserved without blending, premultiplication or color conversion.

use std::{
    fmt,
    io::{self, Write},
};

/// Maximum supported RGBA scanline storage; callers can select a smaller limit.
pub const MAX_RGBA_ROW_BYTES: usize = 64 * 1024;
/// Maximum supported IDAT buffer; this is also the default chunk size.
pub const MAX_IDAT_BUFFER_BYTES: usize = 16 * 1024;

/// Validated positive dimensions with a host-addressable RGBA8 byte extent.
/// Constructing this value does not reserve pixels or admit an output buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngDimensions {
    width: u32,
    height: u32,
    row_bytes: usize,
    rgba_bytes: usize,
}

impl PngDimensions {
    pub fn new(width: u64, height: u64) -> Result<Self, PngError> {
        let width =
            u32::try_from(width).map_err(|_| PngError::InvalidDimensions("width exceeds u32"))?;
        let height =
            u32::try_from(height).map_err(|_| PngError::InvalidDimensions("height exceeds u32"))?;
        if width == 0 || height == 0 {
            return Err(PngError::InvalidDimensions("dimensions must be positive"));
        }
        let row_bytes = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or(PngError::InvalidDimensions(
                "RGBA row exceeds address space",
            ))?;
        let rgba_bytes = usize::try_from(height)
            .ok()
            .and_then(|height| row_bytes.checked_mul(height))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(PngError::InvalidDimensions(
                "RGBA extent exceeds address space",
            ))?;
        Ok(Self {
            width,
            height,
            row_bytes,
            rgba_bytes,
        })
    }

    pub const fn width(self) -> u32 {
        self.width
    }
    pub const fn height(self) -> u32 {
        self.height
    }
    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub const fn rgba_bytes(self) -> usize {
        self.rgba_bytes
    }
}

/// Explicit bounds for the codec's scanline and IDAT buffers, independent of
/// the caller's limit on total encoded response bytes. Smaller limits are valid;
/// exceeding the implementation ceilings is rejected before constructing a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngLimits {
    maximum_row_bytes: usize,
    idat_buffer_bytes: usize,
}

impl Default for PngLimits {
    fn default() -> Self {
        Self {
            maximum_row_bytes: MAX_RGBA_ROW_BYTES,
            idat_buffer_bytes: MAX_IDAT_BUFFER_BYTES,
        }
    }
}

impl PngLimits {
    pub fn new(maximum_row_bytes: usize, idat_buffer_bytes: usize) -> Result<Self, PngError> {
        if maximum_row_bytes == 0 || maximum_row_bytes > MAX_RGBA_ROW_BYTES {
            return Err(PngError::InvalidLimits(
                "RGBA row limit must be within 1..=65536",
            ));
        }
        if idat_buffer_bytes == 0 || idat_buffer_bytes > MAX_IDAT_BUFFER_BYTES {
            return Err(PngError::InvalidLimits(
                "IDAT buffer must be within 1..=16384",
            ));
        }
        Ok(Self {
            maximum_row_bytes,
            idat_buffer_bytes,
        })
    }

    pub const fn maximum_row_bytes(self) -> usize {
        self.maximum_row_bytes
    }
    pub const fn idat_buffer_bytes(self) -> usize {
        self.idat_buffer_bytes
    }

    pub fn validate(self, dimensions: PngDimensions) -> Result<(), PngError> {
        if dimensions.row_bytes > self.maximum_row_bytes {
            return Err(PngError::RowLimitExceeded {
                row_bytes: dimensions.row_bytes,
                limit: self.maximum_row_bytes,
            });
        }
        Ok(())
    }
}

/// Returned only after the pixel stream, PNG container and sink flush succeed.
/// This does not claim file durability, independent decoding or network delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngReceipt {
    dimensions: PngDimensions,
    encoded_bytes: u64,
}

impl PngReceipt {
    pub const fn dimensions(self) -> PngDimensions {
        self.dimensions
    }
    pub const fn encoded_bytes(self) -> u64 {
        self.encoded_bytes
    }
}

#[derive(Debug)]
pub enum PngError {
    InvalidDimensions(&'static str),
    InvalidLimits(&'static str),
    EncodedLengthOverflow,
    RowLimitExceeded { row_bytes: usize, limit: usize },
    Encoding(::png::EncodingError),
    Io(io::Error),
}

impl fmt::Display for PngError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(reason) => {
                write!(formatter, "invalid PNG dimensions: {reason}")
            }
            Self::InvalidLimits(reason) => write!(formatter, "invalid PNG limits: {reason}"),
            Self::EncodedLengthOverflow => write!(formatter, "PNG encoded length exceeds u64"),
            Self::RowLimitExceeded { row_bytes, limit } => {
                write!(formatter, "PNG RGBA row {row_bytes} exceeds limit {limit}")
            }
            Self::Encoding(error) => write!(formatter, "PNG encoding: {error}"),
            Self::Io(error) => write!(formatter, "PNG I/O: {error}"),
        }
    }
}

impl std::error::Error for PngError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoding(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<::png::EncodingError> for PngError {
    fn from(error: ::png::EncodingError) -> Self {
        Self::Encoding(error)
    }
}

impl From<io::Error> for PngError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Exact byte length of [`write_rgba8_png`] when the emitter writes the complete
/// image **without calling `flush`**. This computes metadata only: it neither
/// reads pixels nor reserves output memory. Arbitrary emitter flushes can add
/// DEFLATE blocks and IDAT chunks and are outside this length contract.
///
/// This describes the current encoding profile: png 0.18.1's streaming
/// NoCompression path, with the workspace-locked flate2 1.1.5 / miniz_oxide 0.8.9
/// Rust backend. It is not a generic PNG-size formula. Dependency/backend or
/// finish-policy changes must revalidate it against the real writer. Callers
/// admitting a fixed output buffer must also check the actual encoded length;
/// a size prediction never authorizes publishing a partial output.
pub fn encoded_rgba8_png_bytes(
    dimensions: PngDimensions,
    limits: PngLimits,
) -> Result<u64, PngError> {
    limits.validate(dimensions)?;
    // One filter-type byte precedes each uncompressed RGBA scanline.
    let filtered_bytes = u64::from(dimensions.width)
        .checked_mul(4)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(u64::from(dimensions.height)))
        .ok_or(PngError::EncodedLengthOverflow)?;
    stored_png_bytes(filtered_bytes, limits.idat_buffer_bytes as u64)
}

fn stored_png_bytes(filtered_bytes: u64, idat_bytes: u64) -> Result<u64, PngError> {
    // miniz_oxide::deflate::stored flushes when bytes_written > 31 * 1024.
    // Every full 31745-byte block adds five bytes. Sync also writes a tail
    // block (even when empty) and an empty sync block, after the zlib header.
    let before_finish = (filtered_bytes / (31 * 1024 + 1))
        .checked_mul(5)
        .and_then(|n| n.checked_add(filtered_bytes))
        .and_then(|n| n.checked_add(2 + 5 + 5))
        .ok_or(PngError::EncodedLengthOverflow)?;
    // StreamWriter::finish first Sync-flushes the ChunkWriter, ending the
    // current IDAT. Finish then emits an empty final block + Adler32 (9 bytes)
    // into a fresh IDAT buffer. These two chunk counts cannot be combined.
    let chunks = before_finish
        .div_ceil(idat_bytes)
        .checked_add(9_u64.div_ceil(idat_bytes))
        .ok_or(PngError::EncodedLengthOverflow)?;
    chunks
        .checked_mul(12) // IDAT length, type and CRC.
        .and_then(|n| n.checked_add(before_finish))
        .and_then(|n| n.checked_add(9 + 8 + 25 + 12)) // Finish, signature, IHDR, IEND.
        .ok_or(PngError::EncodedLengthOverflow)
}

struct CountWriter<'a, W: Write + ?Sized> {
    inner: &'a mut W,
    bytes: u64,
}

impl<W: Write + ?Sized> Write for CountWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
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

/// Emit one exact interleaved RGBA8 image in arbitrary chunks into `sink`.
///
/// Dimension/limit checks happen before any sink writes or emitter calls. The
/// emitter must propagate its write errors; missing or excess pixels, sink
/// errors and finish errors are rejected. On failure the sink may contain a
/// partial PNG (including codec destructor writes): rollback/publication is the
/// caller's responsibility. This function only borrows the sink and retains no
/// source, scratch or backing after it returns. A caller using a bounded memory
/// writer must admit that backing before calling this function.
pub fn write_rgba8_png<W: Write + ?Sized>(
    sink: &mut W,
    dimensions: PngDimensions,
    limits: PngLimits,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<PngReceipt, PngError> {
    limits.validate(dimensions)?;
    let mut sink = CountWriter {
        inner: sink,
        bytes: 0,
    };
    {
        let mut encoder = ::png::Encoder::new(&mut sink, dimensions.width, dimensions.height);
        encoder.set_color(::png::ColorType::Rgba);
        encoder.set_depth(::png::BitDepth::Eight);
        encoder.set_compression(::png::Compression::NoCompression);
        encoder.set_filter(::png::Filter::NoFilter);
        let mut writer = encoder.write_header()?;
        {
            let mut stream = writer.stream_writer_with_size(limits.idat_buffer_bytes)?;
            emit(&mut stream)?;
            stream.finish()?;
        }
        writer.finish()?;
    }
    sink.flush()?;
    Ok(PngReceipt {
        dimensions,
        encoded_bytes: sink.bytes,
    })
}

#[cfg(test)]
mod tests;
