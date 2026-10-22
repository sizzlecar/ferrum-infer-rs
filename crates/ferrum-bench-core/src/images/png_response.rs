use super::{invalid, sha256, ImageBenchError, ImageResponseLimits};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedImage {
    pub width: u32,
    pub height: u32,
    pub png_bytes: usize,
    pub rgba8_bytes: usize,
    pub png_sha256: String,
    /// Raw expanded RGBA8 bytes, without color-profile or gamma conversion.
    pub rgba8_sha256: String,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Datum>,
}
#[derive(Deserialize)]
struct Datum {
    b64_json: String,
    #[serde(default)]
    url: Option<String>,
}

/// Strictly decode one PNG and hash its actual pixels. Integer RGBA8 is finite
/// by construction: this does NOT establish pre-encoding model float finiteness,
/// prompt faithfulness, or equality with another implementation's output.
pub fn validate_image_response(
    body: &[u8],
    width: u32,
    height: u32,
    limits: &ImageResponseLimits,
) -> Result<ValidatedImage, ImageBenchError> {
    limits.validate()?;
    let rgba_bytes = limits.rgba_bytes(width, height)?;
    if body.len() > limits.maximum_response_bytes {
        return Err(invalid("JSON response exceeds byte limit"));
    }
    let response: Response =
        serde_json::from_slice(body).map_err(|_| invalid("invalid image JSON response"))?;
    if response.data.len() != 1 {
        return Err(invalid("response must contain exactly one image"));
    }
    let item = &response.data[0];
    if item.url.is_some() {
        return Err(invalid("URL image responses are not supported"));
    }
    let encoded = item.b64_json.as_bytes();
    // Bound before base64 allocation. STANDARD also rejects noncanonical tails,
    // whitespace, missing padding and invalid characters.
    let maximum_encoded = limits
        .maximum_png_bytes
        .checked_add(2)
        .and_then(|n| n.checked_div(3))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("base64 size limit overflow"))?;
    if encoded.len() > maximum_encoded {
        return Err(invalid("encoded PNG exceeds byte limit"));
    }
    let png_bytes = STANDARD
        .decode(encoded)
        .map_err(|_| invalid("invalid image base64"))?;
    if png_bytes.len() > limits.maximum_png_bytes {
        return Err(invalid("PNG exceeds byte limit"));
    }
    validate_container(&png_bytes)?;
    let mut options = png::DecodeOptions::default();
    options.set_ignore_checksums(false);
    options.set_skip_ancillary_crc_failures(false);
    // Metadata decompression is unnecessary for pixel verification.
    options.set_ignore_text_chunk(true);
    options.set_ignore_iccp_chunk(true);
    let mut decoder = png::Decoder::new_with_options(Cursor::new(&png_bytes), options);
    decoder.set_limits(png::Limits {
        bytes: limits.decoder_workspace_bytes,
    });
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|_| invalid("invalid PNG header or checksum"))?;
    let info = reader.info();
    if info.width != width || info.height != height {
        return Err(invalid("PNG dimensions differ from request"));
    }
    if info.animation_control.is_some() || info.frame_control.is_some() {
        return Err(invalid("animated PNG is not supported"));
    }
    if info.bit_depth == png::BitDepth::Sixteen {
        return Err(invalid("16-bit PNG cannot be silently reduced to RGBA8"));
    }
    let size = reader
        .output_buffer_size()
        .filter(|n| *n <= rgba_bytes)
        .ok_or_else(|| invalid("PNG decoded extent exceeds RGBA8 limit"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(size)
        .map_err(|_| invalid("cannot reserve PNG decode buffer"))?;
    pixels.resize(size, 0);
    let decoded = reader
        .next_frame(&mut pixels)
        .map_err(|_| invalid("invalid PNG pixel stream or checksum"))?;
    reader
        .finish()
        .map_err(|_| invalid("invalid PNG ending or checksum"))?;
    if decoded.bit_depth != png::BitDepth::Eight
        || decoded.width != width
        || decoded.height != height
    {
        return Err(invalid("PNG did not decode to requested 8-bit image"));
    }
    let channels = match decoded.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(invalid("PNG palette was not expanded")),
    };
    let pixel_count = rgba_bytes / 4;
    if decoded.buffer_size() != pixel_count * channels {
        return Err(invalid("PNG pixel count mismatch"));
    }
    let mut digest = Sha256::new();
    for p in pixels[..decoded.buffer_size()].chunks_exact(channels) {
        let rgba = match channels {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        };
        digest.update(rgba);
    }
    Ok(ValidatedImage {
        width,
        height,
        png_bytes: png_bytes.len(),
        rgba8_bytes: rgba_bytes,
        png_sha256: sha256(&png_bytes),
        rgba8_sha256: format!("{:x}", digest.finalize()),
    })
}

// Validate exact framing before decompression, including rejecting bytes after
// IEND. The png decoder independently checks all chunk and DEFLATE checksums.
fn validate_container(bytes: &[u8]) -> Result<(), ImageBenchError> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(invalid("image is not PNG"));
    }
    let mut offset = 8usize;
    let mut first = true;
    loop {
        let header = bytes
            .get(offset..offset.saturating_add(8))
            .ok_or_else(|| invalid("truncated PNG chunk"))?;
        let length = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let kind = &header[4..8];
        if first && (kind != b"IHDR" || length != 13) {
            return Err(invalid("PNG must begin with IHDR"));
        }
        first = false;
        if kind == b"acTL" || kind == b"fcTL" || kind == b"fdAT" {
            return Err(invalid("animated PNG is not supported"));
        }
        offset = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(length))
            .filter(|n| *n <= bytes.len())
            .ok_or_else(|| invalid("truncated PNG chunk"))?;
        if kind == b"IEND" {
            if length != 0 || offset != bytes.len() {
                return Err(invalid("invalid PNG IEND or trailing data"));
            }
            return Ok(());
        }
    }
}
