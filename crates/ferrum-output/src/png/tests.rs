use super::*;
use std::{cell::Cell, io::Cursor};

const PIXELS: [u8; 16] = [255, 0, 0, 0, 0, 255, 0, 128, 0, 0, 255, 255, 11, 22, 33, 17];

fn emit_pixels(writer: &mut dyn Write) -> io::Result<()> {
    // Splits both pixels and scanlines; the encoder must not require row writes.
    for chunk in PIXELS.chunks(3) {
        writer.write_all(chunk)?;
    }
    Ok(())
}

#[test]
fn streaming_png_roundtrip_keeps_channels_alpha_dimensions_and_byte_count() {
    let dimensions = PngDimensions::new(2, 2).unwrap();
    let mut encoded = Vec::new();
    let receipt =
        write_rgba8_png(&mut encoded, dimensions, PngLimits::default(), emit_pixels).unwrap();
    assert_eq!(receipt.dimensions(), dimensions);
    assert_eq!(receipt.encoded_bytes(), encoded.len() as u64);
    assert_eq!(dimensions.rgba_bytes(), PIXELS.len());

    let mut reader = ::png::Decoder::new(Cursor::new(&encoded))
        .read_info()
        .unwrap();
    assert!(!reader.info().interlaced);
    let mut decoded = [0_u8; 16];
    let info = reader.next_frame(&mut decoded).unwrap();
    assert_eq!(
        (info.width, info.height, info.color_type, info.bit_depth),
        (2, 2, ::png::ColorType::Rgba, ::png::BitDepth::Eight)
    );
    assert_eq!(decoded, PIXELS);
    reader.finish().unwrap();

    // The same encoder can target a pre-admitted fixed buffer without ownership
    // transfer or a second growable allocation inside the encoder.
    let mut backing = vec![0xa5; encoded.len() + 2];
    let mut cursor = Cursor::new(&mut backing[1..encoded.len() + 1]);
    let fixed =
        write_rgba8_png(&mut cursor, dimensions, PngLimits::default(), emit_pixels).unwrap();
    assert_eq!(fixed, receipt);
    assert_eq!(cursor.position(), receipt.encoded_bytes());
    assert_eq!(&backing[1..encoded.len() + 1], encoded);
    assert_eq!(backing[0], 0xa5);
    assert_eq!(backing[encoded.len() + 1], 0xa5);
}

#[test]
fn dimensions_and_workspace_limits_reject_before_sink_or_emitter() {
    assert!(PngDimensions::new(0, 1).is_err());
    assert!(PngDimensions::new(1, 0).is_err());
    assert!(PngDimensions::new(u32::MAX as u64 + 1, 1).is_err());
    assert!(PngDimensions::new(1, u32::MAX as u64 + 1).is_err());
    assert!(PngDimensions::new(u32::MAX as u64, u32::MAX as u64).is_err());
    assert!(PngLimits::new(0, 1).is_err());
    assert!(PngLimits::new(1, 0).is_err());
    assert!(PngLimits::new(MAX_RGBA_ROW_BYTES + 1, 1).is_err());
    assert!(PngLimits::new(1, MAX_IDAT_BUFFER_BYTES + 1).is_err());

    struct Untouched;
    impl Write for Untouched {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            panic!("invalid layout wrote to sink")
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("invalid layout flushed sink")
        }
    }
    let oversized = PngDimensions::new((MAX_RGBA_ROW_BYTES / 4 + 1) as u64, 1).unwrap();
    assert!(matches!(
        write_rgba8_png(&mut Untouched, oversized, PngLimits::default(), |_| panic!(
            "emitter ran"
        )),
        Err(PngError::RowLimitExceeded { .. })
    ));
    let dimensions = PngDimensions::new(2, 2).unwrap();
    assert!(matches!(
        write_rgba8_png(
            &mut Untouched,
            dimensions,
            PngLimits::new(7, 16).unwrap(),
            |_| panic!("emitter ran")
        ),
        Err(PngError::RowLimitExceeded { .. })
    ));
}

#[test]
fn short_and_excess_pixel_streams_cannot_return_receipts() {
    for length in [15, 17] {
        let mut sink = Vec::new();
        assert!(write_rgba8_png(
            &mut sink,
            PngDimensions::new(2, 2).unwrap(),
            PngLimits::default(),
            |stream| stream.write_all(&[0_u8; 17][..length])
        )
        .is_err());
    }
}

#[test]
fn caller_scratch_policy_can_reduce_idat_and_row_buffers() {
    let limits = PngLimits::new(8, 1).unwrap();
    assert_eq!(limits.maximum_row_bytes(), 8);
    assert_eq!(limits.idat_buffer_bytes(), 1);
    let mut encoded = Vec::new();
    let receipt = write_rgba8_png(
        &mut encoded,
        PngDimensions::new(2, 2).unwrap(),
        limits,
        emit_pixels,
    )
    .unwrap();
    let mut reader = ::png::Decoder::new(Cursor::new(&encoded))
        .read_info()
        .unwrap();
    let mut decoded = [0; 16];
    reader.next_frame(&mut decoded).unwrap();
    reader.finish().unwrap();
    assert_eq!(decoded, PIXELS);
    assert_eq!(receipt.encoded_bytes(), encoded.len() as u64);
}

#[test]
fn insufficient_sink_capacity_and_emitter_failure_are_propagated() {
    let dimensions = PngDimensions::new(2, 2).unwrap();
    let mut backing = [0xa5; 19];
    let mut bounded = Cursor::new(&mut backing[1..18]);
    assert!(write_rgba8_png(&mut bounded, dimensions, PngLimits::default(), emit_pixels).is_err());
    assert!(bounded.position() <= 17);
    assert_eq!(backing[0], 0xa5);
    assert_eq!(backing[18], 0xa5);

    let result = write_rgba8_png(
        &mut Vec::new(),
        dimensions,
        PngLimits::default(),
        |stream| {
            stream.write_all(&PIXELS[..4])?;
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "pixel source failed",
            ))
        },
    );
    assert!(
        matches!(result, Err(PngError::Io(error)) if error.kind() == io::ErrorKind::Interrupted)
    );
}

#[test]
fn sink_flush_failure_cannot_publish_a_success_receipt() {
    struct FailingFlush<'a> {
        bytes: usize,
        calls: &'a Cell<usize>,
    }
    impl Write for FailingFlush<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.calls.set(self.calls.get() + 1);
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "destination closed",
            ))
        }
    }
    let calls = Cell::new(0);
    let mut sink = FailingFlush {
        bytes: 0,
        calls: &calls,
    };
    assert!(write_rgba8_png(
        &mut sink,
        PngDimensions::new(2, 2).unwrap(),
        PngLimits::default(),
        emit_pixels
    )
    .is_err());
    assert!(calls.get() > 0);
    assert!(sink.bytes > 0);
}

fn assert_predicted_length_and_pixels(width: u64, height: u64, idat: usize, chunk: usize) {
    let dimensions = PngDimensions::new(width, height).unwrap();
    let limits = PngLimits::new(MAX_RGBA_ROW_BYTES, idat).unwrap();
    // Different contents have the same stored-stream length. Chunks deliberately
    // cross both RGBA pixels and scanline boundaries; the emitter never flushes.
    for patterned in [false, true] {
        let pixels = (0..dimensions.rgba_bytes())
            .map(|index| {
                if patterned {
                    (index.wrapping_mul(37) >> 3) as u8
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let expected = encoded_rgba8_png_bytes(dimensions, limits).unwrap();
        let mut encoded = Vec::new();
        let receipt = write_rgba8_png(&mut encoded, dimensions, limits, |writer| {
            for bytes in pixels.chunks(chunk) {
                writer.write_all(bytes)?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(
            receipt.encoded_bytes(),
            expected,
            "{width}x{height}, IDAT {idat}"
        );
        assert_eq!(encoded.len() as u64, expected);
        let mut reader = ::png::Decoder::new(Cursor::new(&encoded))
            .read_info()
            .unwrap();
        let mut decoded = vec![0; dimensions.rgba_bytes()];
        reader.next_frame(&mut decoded).unwrap();
        reader.finish().unwrap();
        assert_eq!(decoded, pixels);
    }
}

#[test]
fn exact_length_matches_stored_block_boundaries_and_row_chunking() {
    // Filtered lengths include 31743, 31745, 31746, 63489 and 63490. The exact
    // multiples must account for the empty tail written by Sync, not ceil(L/B).
    for (width, height) in [
        (1, 1),
        (2645, 3),
        (7936, 1),
        (3968, 2),
        (1, 6349),
        (15872, 1),
        (1, 12698),
        ((MAX_RGBA_ROW_BYTES / 4) as u64, 2),
    ] {
        for chunk in [13, 8192] {
            assert_predicted_length_and_pixels(width, height, MAX_IDAT_BUFFER_BYTES, chunk);
        }
    }
}

#[test]
fn exact_length_keeps_sync_and_finish_idat_chunks_separate() {
    // 2x2 has 18 filtered bytes: 30 bytes before Finish and a separate 9-byte
    // tail. Cover exact divisions of either portion and IDAT boundaries ±1.
    for idat in [1, 3, 8, 9, 10, 15, 16, 29, 30, 31] {
        assert_predicted_length_and_pixels(2, 2, idat, 3);
    }
}

#[test]
fn exact_length_checks_limits_and_arithmetic_without_allocating_pixels() {
    let too_wide = PngDimensions::new((MAX_RGBA_ROW_BYTES / 4 + 1) as u64, 1).unwrap();
    assert!(matches!(
        encoded_rgba8_png_bytes(too_wide, PngLimits::default()),
        Err(PngError::RowLimitExceeded { .. })
    ));
    assert!(matches!(
        encoded_rgba8_png_bytes(
            PngDimensions::new(2, 2).unwrap(),
            PngLimits::new(7, 1).unwrap()
        ),
        Err(PngError::RowLimitExceeded { .. })
    ));
    // Exercise both DEFLATE-length and IDAT-overhead overflow in the checked
    // arithmetic helper; public dimensions reject impossible extents earlier.
    for filtered_bytes in [u64::MAX, u64::MAX / 2] {
        assert!(matches!(
            stored_png_bytes(filtered_bytes, 1),
            Err(PngError::EncodedLengthOverflow)
        ));
    }
    if usize::BITS == 64 {
        let dimensions =
            PngDimensions::new((MAX_RGBA_ROW_BYTES / 4) as u64, u32::MAX as u64).unwrap();
        assert!(
            encoded_rgba8_png_bytes(dimensions, PngLimits::default()).unwrap()
                > dimensions.rgba_bytes() as u64
        );
    }
}

#[test]
fn explicit_emitter_flush_is_outside_the_exact_length_contract() {
    let dimensions = PngDimensions::new(2, 2).unwrap();
    let limits = PngLimits::default();
    let mut encoded = Vec::new();
    write_rgba8_png(&mut encoded, dimensions, limits, |writer| {
        writer.write_all(&PIXELS[..8])?;
        writer.flush()?;
        writer.write_all(&PIXELS[8..])
    })
    .unwrap();
    assert!(encoded.len() as u64 > encoded_rgba8_png_bytes(dimensions, limits).unwrap());
    let mut reader = ::png::Decoder::new(Cursor::new(&encoded))
        .read_info()
        .unwrap();
    let mut decoded = [0; 16];
    reader.next_frame(&mut decoded).unwrap();
    reader.finish().unwrap();
    assert_eq!(decoded, PIXELS);
}
