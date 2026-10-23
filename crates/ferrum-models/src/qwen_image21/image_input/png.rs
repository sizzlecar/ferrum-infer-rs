use super::*;
use ::png::{BitDepth, ColorType, DecodeOptions, Decoder};
use std::io::Cursor;

pub(super) fn decode(
    host: &HostMemoryAccount,
    encoded: &ManagedHostBytes,
    limits: ImageInputLimits,
) -> ImageInputResult<ManagedRgba8Image> {
    check_input_owner(host, encoded)?;
    if encoded.len() > limits.maximum_encoded_bytes {
        return Err(ImageInputError::Limit("encoded PNG bytes"));
    }
    // Validate framing and IHDR limits before the decoder allocates its codec
    // storage. Every CRC/DEFLATE checksum is independently checked by png.
    let (width, height, color) = inspect(encoded.bytes(), limits)?;
    let mut options = DecodeOptions::default();
    options.set_ignore_text_chunk(true);
    options.set_ignore_iccp_chunk(true);
    let mut decoder = Decoder::new_with_options(Cursor::new(encoded.bytes()), options);
    decoder.set_limits(::png::Limits {
        bytes: limits.decoder_workspace_bytes,
    });
    let mut reader = decoder.read_info()?;
    if reader.info().width != width
        || reader.info().height != height
        || reader.info().bit_depth != BitDepth::Eight
        || reader.info().color_type != color
        || reader.info().trns.is_some()
    {
        return Err(ImageInputError::Unsupported(
            "PNG header/transparency differs from supported RGB/RGBA8",
        ));
    }
    let rgba_len = limits.rgba_bytes(width, height)?;
    let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, rgba_len)?;
    let frame = reader.next_frame(output.as_mut())?;
    if frame.width != width
        || frame.height != height
        || frame.color_type != color
        || frame.bit_depth != BitDepth::Eight
    {
        return Err(ImageInputError::Invalid(
            "decoded frame differs from admitted PNG header",
        ));
    }
    reader.finish()?;
    if color == ColorType::Rgb {
        // Decode RGB into the prefix of the admitted RGBA backing and expand
        // backwards, so no second complete pixel buffer is necessary.
        for pixel in (0..rgba_len / 4).rev() {
            let rgb = [
                output.as_ref()[pixel * 3],
                output.as_ref()[pixel * 3 + 1],
                output.as_ref()[pixel * 3 + 2],
            ];
            output.as_mut()[pixel * 4..pixel * 4 + 4]
                .copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Ok(ManagedRgba8Image {
        bytes: output.freeze(),
        width,
        height,
        source_has_alpha: color == ColorType::Rgba,
    })
}

fn inspect(bytes: &[u8], limits: ImageInputLimits) -> ImageInputResult<(u32, u32, ColorType)> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(ImageInputError::Unsupported("expected PNG signature"));
    }
    let mut offset = 8usize;
    let mut dimensions = None;
    loop {
        let header = bytes
            .get(offset..offset.saturating_add(8))
            .ok_or(ImageInputError::Invalid("truncated PNG chunk"))?;
        let length = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let kind = &header[4..8];
        let end = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(length))
            .filter(|&n| n <= bytes.len())
            .ok_or(ImageInputError::Invalid("truncated PNG payload"))?;
        if dimensions.is_none() {
            if kind != b"IHDR" || length != 13 {
                return Err(ImageInputError::Invalid("PNG must begin with one IHDR"));
            }
            let data = &bytes[offset + 8..offset + 21];
            let width = u32::from_be_bytes(data[..4].try_into().unwrap());
            let height = u32::from_be_bytes(data[4..8].try_into().unwrap());
            limits.rgba_bytes(width, height)?;
            let color = match (data[8], data[9]) {
                (8, 2) => ColorType::Rgb,
                (8, 6) => ColorType::Rgba,
                _ => {
                    return Err(ImageInputError::Unsupported(
                        "only 8-bit RGB/RGBA PNG is supported",
                    ))
                }
            };
            dimensions = Some((width, height, color));
        } else if kind == b"IHDR" {
            return Err(ImageInputError::Invalid("duplicate PNG IHDR"));
        }
        if matches!(kind, b"acTL" | b"fcTL" | b"fdAT") {
            return Err(ImageInputError::Unsupported("animated PNG"));
        }
        if kind == b"tRNS" || kind == b"eXIf" {
            return Err(ImageInputError::Unsupported(
                "PNG tRNS/EXIF requires an explicit conversion policy",
            ));
        }
        if kind == b"IEND" {
            if length != 0 || end != bytes.len() {
                return Err(ImageInputError::Invalid(
                    "invalid PNG IEND or trailing bytes",
                ));
            }
            return dimensions.ok_or(ImageInputError::Invalid("missing PNG dimensions"));
        }
        offset = end;
    }
}
