use super::*;

// Invert physical coordinates independently of the production logical indexer.
fn active_offsets(rows: usize, columns: usize, bytes: usize) -> Vec<usize> {
    let column_tiles = (columns / 16).div_ceil(4);
    (0..bytes)
        .filter(|&offset| {
            let tile = offset / 512;
            let local = offset % 512;
            let row = (tile / column_tiles) * 128 + local / 16 + ((local / 4) % 4) * 32;
            let group = (tile % column_tiles) * 4 + local % 4;
            row < rows && group < columns / 16
        })
        .collect()
}

#[test]
fn all_scale_encodings_preserve_the_domain_and_decoded_values() {
    for (rows, columns) in [(1, 16), (128, 64), (129, 80), (256, 96)] {
        let geometry = Nvfp4Geometry::new(rows, columns).unwrap();
        let active = active_offsets(rows, columns, geometry.scale_bytes());
        assert_eq!(active.len(), rows * columns / 16);
        let mut scales = vec![0xff; geometry.scale_bytes()];
        for code in 0..=u8::MAX {
            for &offset in &active {
                scales[offset] = code;
            }
            let exponent = (code >> 3) & 15;
            let fraction = code & 7;
            let finite = !(exponent == 15 && fraction == 7);
            let allowed = code & 128 == 0 && finite;
            assert_eq!(
                geometry.validate_scales(&scales).is_ok(),
                allowed,
                "shape {rows}x{columns}, code {code:#04x}"
            );
            let decoded = decode_e4m3_scale(code);
            assert_eq!(decoded.is_ok(), allowed);
            if allowed {
                let significand = if exponent == 0 {
                    fraction
                } else {
                    8 + fraction
                };
                let power = if exponent == 0 {
                    -9
                } else {
                    i32::from(exponent) - 10
                };
                let expected = (f64::from(significand) * 2.0_f64.powi(power)) as f32;
                assert_eq!(decoded.unwrap().to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn invalid_active_scales_are_rejected_at_every_physical_position() {
    for (rows, columns) in [(1, 16), (128, 64), (129, 80), (256, 96)] {
        let geometry = Nvfp4Geometry::new(rows, columns).unwrap();
        let active = active_offsets(rows, columns, geometry.scale_bytes());
        let mut scales = vec![0xff; geometry.scale_bytes()];
        for &offset in &active {
            scales[offset] = 0x38;
        }
        geometry.validate_scales(&scales).unwrap();
        for offset in active {
            scales[offset] = 0x7f;
            assert!(
                geometry.validate_scales(&scales).is_err(),
                "offset {offset}"
            );
            scales[offset] = 0x38;
        }
        assert!(geometry
            .validate_scales(&scales[..scales.len() - 1])
            .is_err());
        scales.push(0x38);
        assert!(geometry.validate_scales(&scales).is_err());
    }
}

#[test]
#[ignore = "diagnostic timing; run with an optimized profile on the target host"]
fn benchmark_scale_validation_without_dequantization() {
    use std::{hint::black_box, time::Instant};

    fn prior_validation(geometry: Nvfp4Geometry, source: &[u8]) -> ferrum_types::Result<()> {
        for row in 0..geometry.rows() {
            for group in 0..geometry.columns() / 16 {
                decode_e4m3_scale(source[geometry.scale_index(row, group)?])?;
            }
        }
        Ok(())
    }

    let geometry = Nvfp4Geometry::new(8192, 32768).unwrap();
    let scales: Vec<u8> = (0..geometry.scale_bytes())
        .map(|index| (index % 127) as u8)
        .collect();
    for repetition in 0..4 {
        let methods = if repetition % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for storage_scan in methods {
            let source = black_box(scales.as_slice());
            let start = Instant::now();
            if storage_scan {
                black_box(geometry.validate_scales(source)).unwrap();
            } else {
                black_box(prior_validation(geometry, source)).unwrap();
            }
            eprintln!("scale_validation repetition={repetition} storage_scan={storage_scan} bytes={} seconds={:.9}", scales.len(), start.elapsed().as_secs_f64());
        }
    }
}
