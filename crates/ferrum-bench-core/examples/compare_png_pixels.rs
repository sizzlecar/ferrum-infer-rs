//! Bounded raw RGB/RGBA8 PNG comparison. No quality threshold or PASS verdict.
//! Samples are compared without color-profile/gamma conversion, alpha
//! compositing or premultiplication. RGB inputs have implicit opaque alpha.
//! Memory limits apply separately to each encoded file, each decoded image and
//! decoder workspace; while decoding the second file, the first pixels remain.

use clap::{Args, Parser};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{self, Cursor, Read, Write},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(about = "Compare two RGB/RGBA8 PNGs; pixel diagnostics, not a quality verdict")]
struct Arguments {
    left: PathBuf,
    right: PathBuf,
    #[command(flatten)]
    limits: Limits,
}

#[derive(Args, Clone, Debug, Serialize)]
struct Limits {
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    maximum_input_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    maximum_decoded_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    decoder_workspace_bytes: usize,
    #[arg(long, default_value_t = 8192)]
    maximum_dimension: u32,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    maximum_pixels: u64,
}

fn invalid(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

impl Limits {
    fn validate(&self) -> Result<()> {
        if self.maximum_dimension == 0
            || self.maximum_pixels == 0
            || self.maximum_pixels > (isize::MAX as u64) / 4
            || [
                self.maximum_input_bytes,
                self.maximum_decoded_bytes,
                self.decoder_workspace_bytes,
            ]
            .iter()
            .any(|&n| n == 0 || n > isize::MAX as usize)
        {
            return Err(invalid(
                "limits must be positive and fit the host address space",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct DecodedImage {
    width: u32,
    height: u32,
    channels: usize,
    pixels: Vec<u8>,
    source_bytes: usize,
    source_sha256: String,
    rgba8_sha256: String,
}

fn load(path: &Path, limits: &Limits) -> Result<DecodedImage> {
    limits.validate()?;
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > limits.maximum_input_bytes as u64
    {
        return Err(invalid(
            "PNG must be a nonempty regular file within the input limit",
        ));
    }
    let length = usize::try_from(metadata.len())?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length)?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0])? != 0 {
        return Err(invalid("PNG grew while reading"));
    }
    decode(&bytes, limits)
}

// Check exact framing before decompression. png independently checks CRC and
// Adler32, including the final chunks when Reader::finish is called.
fn validate_container(bytes: &[u8]) -> Result<()> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(invalid("input is not PNG"));
    }
    let mut offset = 8usize;
    loop {
        let header = bytes
            .get(offset..offset.saturating_add(8))
            .ok_or_else(|| invalid("truncated PNG chunk header"))?;
        let length = u32::from_be_bytes(header[..4].try_into()?) as usize;
        let kind = &header[4..8];
        if (offset == 8 && (kind != b"IHDR" || length != 13)) || (offset != 8 && kind == b"IHDR") {
            return Err(invalid("PNG requires exactly one leading IHDR"));
        }
        if matches!(kind, b"acTL" | b"fcTL" | b"fdAT" | b"tRNS") {
            return Err(invalid(
                "animated PNG and tRNS transparency are unsupported",
            ));
        }
        offset = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(length))
            .filter(|&n| n <= bytes.len())
            .ok_or_else(|| invalid("truncated PNG chunk"))?;
        if kind == b"IEND" {
            if length != 0 || offset != bytes.len() {
                return Err(invalid("invalid IEND or trailing bytes"));
            }
            return Ok(());
        }
    }
}

fn decode(bytes: &[u8], limits: &Limits) -> Result<DecodedImage> {
    limits.validate()?;
    if bytes.len() > limits.maximum_input_bytes {
        return Err(invalid("PNG exceeds input byte limit"));
    }
    validate_container(bytes)?;
    let mut options = png::DecodeOptions::default();
    options.set_ignore_checksums(false);
    options.set_skip_ancillary_crc_failures(false);
    options.set_ignore_text_chunk(true);
    options.set_ignore_iccp_chunk(true);
    let mut decoder = png::Decoder::new_with_options(Cursor::new(bytes), options);
    decoder.set_limits(png::Limits {
        bytes: limits.decoder_workspace_bytes,
    });
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let (width, height) = (info.width, info.height);
    let count = u64::from(width) * u64::from(height);
    if width == 0
        || height == 0
        || width > limits.maximum_dimension
        || height > limits.maximum_dimension
        || count > limits.maximum_pixels
    {
        return Err(invalid("PNG dimensions exceed configured limits"));
    }
    if info.bit_depth != png::BitDepth::Eight {
        return Err(invalid("only 8-bit PNG is supported"));
    }
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err(invalid("only RGB/RGBA PNG is supported")),
    };
    let expected = usize::try_from(count)?
        .checked_mul(channels)
        .filter(|&n| n <= limits.maximum_decoded_bytes)
        .ok_or_else(|| invalid("decoded image exceeds byte limit"))?;
    if reader.output_buffer_size() != Some(expected) {
        return Err(invalid("PNG decoded extent differs from its dimensions"));
    }
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(expected)?;
    pixels.resize(expected, 0);
    let decoded = reader.next_frame(&mut pixels)?;
    if decoded.width != width
        || decoded.height != height
        || decoded.bit_depth != png::BitDepth::Eight
        || decoded.buffer_size() != expected
        || decoded.color_type
            != if channels == 3 {
                png::ColorType::Rgb
            } else {
                png::ColorType::Rgba
            }
    {
        return Err(invalid("PNG decoded format changed"));
    }
    reader.finish()?;
    let mut rgba = Sha256::new();
    for pixel in pixels.chunks_exact(channels) {
        rgba.update([pixel[0], pixel[1], pixel[2], alpha(pixel)]);
    }
    Ok(DecodedImage {
        width,
        height,
        channels,
        pixels,
        source_bytes: bytes.len(),
        source_sha256: format!("{:x}", Sha256::digest(bytes)),
        rgba8_sha256: format!("{:x}", rgba.finalize()),
    })
}

fn alpha(pixel: &[u8]) -> u8 {
    pixel.get(3).copied().unwrap_or(255)
}

#[derive(Debug, Serialize)]
struct Metrics {
    mae: f64,
    rmse: f64,
    max_error: u8,
    unequal_pixels: u64,
    unequal_pixel_ratio: f64,
}

#[derive(Default)]
struct Accumulator {
    absolute: u128,
    squared: u128,
    maximum: u8,
    unequal_pixels: u64,
}
impl Accumulator {
    fn add(&mut self, difference: u8) {
        self.absolute += u128::from(difference);
        self.squared += u128::from(difference) * u128::from(difference);
        self.maximum = self.maximum.max(difference);
    }
    fn finish(self, pixels: u64, channels: u64) -> Metrics {
        let samples = (pixels * channels) as f64;
        Metrics {
            mae: self.absolute as f64 / samples,
            rmse: (self.squared as f64 / samples).sqrt(),
            max_error: self.maximum,
            unequal_pixels: self.unequal_pixels,
            unequal_pixel_ratio: self.unequal_pixels as f64 / pixels as f64,
        }
    }
}

fn compare(left: &DecodedImage, right: &DecodedImage) -> Result<(Metrics, Metrics)> {
    if (left.width, left.height) != (right.width, right.height) {
        return Err(invalid("images must have identical dimensions"));
    }
    let mut rgb = Accumulator::default();
    let mut alpha_stats = Accumulator::default();
    for (a, b) in left
        .pixels
        .chunks_exact(left.channels)
        .zip(right.pixels.chunks_exact(right.channels))
    {
        let mut changed = false;
        for channel in 0..3 {
            let difference = a[channel].abs_diff(b[channel]);
            rgb.add(difference);
            changed |= difference != 0;
        }
        rgb.unequal_pixels += u64::from(changed);
        let difference = alpha(a).abs_diff(alpha(b));
        alpha_stats.add(difference);
        alpha_stats.unequal_pixels += u64::from(difference != 0);
    }
    let pixels = u64::from(left.width) * u64::from(left.height);
    Ok((rgb.finish(pixels, 3), alpha_stats.finish(pixels, 1)))
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let left = load(&args.left, &args.limits)?;
    let right = load(&args.right, &args.limits)?;
    let (rgb, alpha) = compare(&left, &right)?;
    let source = |path: &Path, image: &DecodedImage| {
        serde_json::json!({
            "path": path, "png_bytes": image.source_bytes, "png_sha256": image.source_sha256,
            "color_type": if image.channels == 3 { "rgb8" } else { "rgba8" },
            "decoded_bytes": image.pixels.len(), "rgba8_sha256": image.rgba8_sha256
        })
    };
    let report = serde_json::json!({
        "schema": "ferrum.png_pixel_comparison.v1",
        "scope": "raw decoded sample diagnostics only; no quality PASS threshold",
        "units": "0..255 sample values; no ICC/gamma conversion or alpha compositing; RGB alpha=255",
        "rgb_unequal_pixel_definition": "at least one of the three RGB channels differs",
        "width": left.width, "height": left.height,
        "pixel_count": u64::from(left.width) * u64::from(left.height),
        "left": source(&args.left, &left), "right": source(&args.right, &right),
        "rgb": rgb, "alpha": alpha, "limits": args.limits
    });
    let mut output = io::stdout().lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Arguments::try_parse_from(["compare", "a.png", "b.png"])
            .unwrap()
            .limits
    }
    fn png(color: png::ColorType, depth: png::BitDepth, pixels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(color);
            encoder.set_depth(depth);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            writer.finish().unwrap();
        }
        bytes
    }

    #[test]
    fn actual_rgb_and_rgba_decode_have_same_pixels_with_opaque_alpha() {
        let rgb = decode(
            &png(
                png::ColorType::Rgb,
                png::BitDepth::Eight,
                &[0, 127, 255, 1, 2, 3],
            ),
            &limits(),
        )
        .unwrap();
        let rgba = decode(
            &png(
                png::ColorType::Rgba,
                png::BitDepth::Eight,
                &[0, 127, 255, 255, 1, 2, 3, 255],
            ),
            &limits(),
        )
        .unwrap();
        let (rgb_stats, alpha_stats) = compare(&rgb, &rgba).unwrap();
        assert_eq!(rgb_stats.mae, 0.0);
        assert_eq!(rgb_stats.unequal_pixels, 0);
        assert_eq!(alpha_stats.mae, 0.0);
        assert_eq!(rgb.rgba8_sha256, rgba.rgba8_sha256);
        assert_ne!(rgb.source_sha256, rgba.source_sha256);
    }

    #[test]
    fn actual_decode_rgb_and_alpha_differences_match_independent_arithmetic() {
        let left = decode(
            &png(
                png::ColorType::Rgb,
                png::BitDepth::Eight,
                &[0, 10, 20, 30, 40, 50],
            ),
            &limits(),
        )
        .unwrap();
        let right = decode(
            &png(
                png::ColorType::Rgba,
                png::BitDepth::Eight,
                &[3, 6, 20, 253, 30, 40, 50, 255],
            ),
            &limits(),
        )
        .unwrap();
        let (rgb, alpha) = compare(&left, &right).unwrap();
        assert_eq!(rgb.mae, 7.0 / 6.0);
        assert_eq!(rgb.rmse, (25.0_f64 / 6.0).sqrt());
        assert_eq!(rgb.max_error, 4);
        assert_eq!(rgb.unequal_pixel_ratio, 0.5);
        assert_eq!(alpha.mae, 1.0);
        assert_eq!(alpha.rmse, 2.0_f64.sqrt());
        assert_eq!(alpha.max_error, 2);
        assert_eq!(alpha.unequal_pixel_ratio, 0.5);
    }

    #[test]
    fn rejects_formats_corruption_trailing_data_and_memory_dimension_limits() {
        let bytes = png(png::ColorType::Rgba, png::BitDepth::Eight, &[0; 8]);
        for forbidden in [
            png(png::ColorType::Grayscale, png::BitDepth::Eight, &[0; 2]),
            png(png::ColorType::Rgb, png::BitDepth::Sixteen, &[0; 12]),
        ] {
            assert!(decode(&forbidden, &limits()).is_err());
        }
        for animated in [false, true] {
            let mut unsupported = Vec::new();
            {
                let mut encoder = png::Encoder::new(&mut unsupported, 2, 1);
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                if animated {
                    encoder.set_animated(1, 0).unwrap();
                } else {
                    encoder.set_trns(vec![0; 6]);
                }
                let mut writer = encoder.write_header().unwrap();
                writer.write_image_data(&[0; 6]).unwrap();
                writer.finish().unwrap();
            }
            assert!(decode(&unsupported, &limits()).is_err());
        }
        let mut corrupt = bytes.clone();
        corrupt[29] ^= 1;
        assert!(decode(&corrupt, &limits()).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode(&trailing, &limits()).is_err());
        assert!(decode(&bytes[..bytes.len() - 1], &limits()).is_err());
        for restricted in [
            Limits {
                maximum_input_bytes: bytes.len() - 1,
                ..limits()
            },
            Limits {
                maximum_decoded_bytes: 7,
                ..limits()
            },
            Limits {
                maximum_dimension: 1,
                ..limits()
            },
            Limits {
                maximum_pixels: 1,
                ..limits()
            },
            Limits {
                decoder_workspace_bytes: 1,
                ..limits()
            },
        ] {
            assert!(decode(&bytes, &restricted).is_err());
        }
        let mut exact = limits();
        exact.maximum_input_bytes = bytes.len();
        exact.maximum_decoded_bytes = 8;
        exact.maximum_pixels = 2;
        assert!(decode(&bytes, &exact).is_ok());
        let mut transposed = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut transposed, 1, 2);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0; 8]).unwrap();
            writer.finish().unwrap();
        }
        let other = decode(&transposed, &limits()).unwrap();
        let original = decode(&bytes, &limits()).unwrap();
        assert!(compare(&original, &other).is_err());
    }

    #[test]
    fn file_reader_enforces_input_limit_before_decode() {
        let bytes = png(png::ColorType::Rgb, png::BitDepth::Eight, &[0; 6]);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        assert_eq!(load(file.path(), &limits()).unwrap().pixels.len(), 6);
        assert!(load(
            file.path(),
            &Limits {
                maximum_input_bytes: bytes.len() - 1,
                ..limits()
            }
        )
        .is_err());
    }
}
