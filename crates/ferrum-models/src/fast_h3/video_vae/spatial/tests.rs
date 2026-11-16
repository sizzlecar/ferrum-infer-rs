use super::*;
mod benchmark;
mod groups;
fn config() -> FastH3VideoVaeConfig {
    let mut raw = serde_json::to_value(crate::fast_h3::video_vae::tests::config()).unwrap();
    raw["block_out_channels"] = serde_json::json!([32, 32, 32, 32, 32, 32]);
    raw["spatial_downsample_factors"] = serde_json::json!([2, 2, 2, 2, 1, 1]);
    raw["temporal_downsample_factors"] = serde_json::json!([1, 2, 2, 1, 1, 1]);
    FastH3VideoVaeConfig::parse(&serde_json::to_vec(&raw).unwrap(), 4096).unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 32 << 20,
                transfer_staging_bytes: 0,
                output_bytes: 32 << 20,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

#[test]
fn overlap_credit_covers_the_largest_real_readback_including_a_short_group() {
    let host = host();
    let plan = FastH3VideoSpatialPlan::new(
        &config(),
        [1, 2, 37, 30, 52],
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    let batch = SpatialTileBatch::new(7).unwrap();
    let bound = plan
        .maximum_grouped_readback_backing_bytes(1, batch, &host)
        .unwrap();
    let mut retained = Vec::new();
    for first in (0..plan.tile_count()).step_by(batch.maximum_tiles()) {
        let group = plan.tile_group(first, batch).unwrap();
        let tile = plan.tiles.tile(first).unwrap();
        let bytes = extent([1, group.tile_count(), 3, 1, tile.height, tile.width], 2).unwrap();
        assert!(host.backing_bytes(bytes).unwrap() <= bound);
        retained.push(
            host.allocate_zeroed(HostMemoryClass::Output, bytes)
                .unwrap(),
        );
    }
    assert_eq!(retained.len(), 2);
    assert_eq!(host.backing_bytes(retained[0].len()).unwrap(), bound);
    assert!(retained[1].len() < retained[0].len());
    assert!(host.usage().unwrap().allocated_bytes() <= 2 * bound);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(plan
        .maximum_grouped_readback_backing_bytes(0, batch, &host)
        .is_err());
}
#[test]
fn original_480p_spatial_grid_crops_exact_temporal_channels_and_denormalizes_once() {
    let config = config();
    let h = host();
    let shape = [1, 2, 37, 30, 52];
    let plan = FastH3VideoSpatialPlan::new(
        &config,
        shape,
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    assert_eq!(plan.tile_count(), 12);
    assert_eq!(plan.clip_grid(7).unwrap(), [1, 7, 16, 16]);
    for i in 0..12 {
        let t = plan.tiles.tile(i).unwrap();
        assert_eq!([t.x, t.y], [i % 4 * 192, i / 4 * 112]);
        assert_eq!([t.width, t.height], [256, 256]);
        if i >= 4 {
            assert_eq!(t.overlap_top, 144);
        }
    }
    let temporal = FastH3VideoDecodePlan::new(&config, 37, 124).unwrap();
    let mut source = h
        .allocate_zeroed(HostMemoryClass::InputPreprocess, extent(shape, 4).unwrap())
        .unwrap();
    for (i, w) in source.as_mut().chunks_exact_mut(4).enumerate() {
        w.copy_from_slice(&(i as f32 / 1024.).to_le_bytes());
    }
    let source = source.freeze();
    let got = plan
        .managed_tile_input(
            &temporal,
            6,
            11,
            FastH3VideoLatentUnits::Normalized,
            &source,
            &h,
        )
        .unwrap();
    assert_eq!(got.tensor().dimensions(), [7 * 16 * 16, 2]);
    for (i, word) in got.bytes().bytes().chunks_exact(4).enumerate() {
        let c = i % 2;
        let token = i / 2;
        let x = token % 16;
        let y = token / 16 % 16;
        let t = (30 + token / (16 * 16)).min(36);
        let original = ((((c * 37 + t) * 30 + 14 + y) * 52 + 36 + x) as f32) / 1024.;
        let expected = if c == 0 {
            original + 0.25
        } else {
            original * 2. - 0.5
        };
        assert_eq!(f32::from_le_bytes(word.try_into().unwrap()), expected);
    }
    assert!(plan
        .managed_tile_input(
            &temporal,
            0,
            0,
            FastH3VideoLatentUnits::Normalized,
            &source,
            &host()
        )
        .is_err());
    drop(got);
    drop(source);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn spatial_output_admission_covers_real_readback_unpack_and_trimmed_staging() {
    let make_host = |bytes| {
        HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    transfer_staging_bytes: 0,
                    output_bytes: bytes,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            256,
        )
        .unwrap()
    };
    let metadata_host = make_host(32 << 20);
    let plan = FastH3VideoSpatialPlan::new(
        &config(),
        [1, 2, 7, 30, 52],
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    // At the final tile there are four original neighbours, a Core readback,
    // an unpacked tile and full-size staging, alongside the whole canvas.
    let bound = metadata_host.backing_bytes(3 * 480 * 832 * 2).unwrap()
        + 7 * metadata_host.backing_bytes(3 * 256 * 256 * 2).unwrap();
    assert_eq!(plan.output_backing_bytes(1, &metadata_host).unwrap(), bound);
    assert_eq!(metadata_host.usage().unwrap().committed_bytes(), 0);
    for deficit in [0, 1] {
        let host = make_host(bound - deficit);
        let mut canvas = plan.begin_canvas(1, &host).unwrap();
        for index in 0..plan.tile_count() {
            let shape = [1, 3, 1, 256, 256];
            let bytes = extent(shape, 2).unwrap();
            let readback = host
                .allocate_zeroed(HostMemoryClass::Output, bytes)
                .unwrap();
            let raw = FastH3VideoRawClip {
                shape,
                bytes: host
                    .allocate_zeroed(HostMemoryClass::Output, bytes)
                    .unwrap()
                    .freeze(),
            };
            let result = canvas.push(index, &raw);
            if deficit == 1 && index + 1 == plan.tile_count() {
                assert!(result.is_err());
                assert_eq!(canvas.completed_tiles(), index);
            } else {
                result.unwrap();
            }
            drop(readback);
        }
        drop(canvas);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
fn value(tile: usize, plane: usize, y: usize, x: usize) -> f64 {
    (tile * 11 + plane * 3) as f64 / 8. + (y % 7) as f64 / 16. - (x % 5) as f64 / 32.
}
fn r(x: f64) -> f64 {
    f64::from(bf16::from_f64(x).to_f32())
}
fn mix(a: f64, b: f64, position: usize, extent: usize) -> f64 {
    let wb = r(position as f64 / extent as f64);
    let wa = r(1. - wb);
    r(r(a * wa) + r(b * wb))
}
#[test]
fn bf16_spatial_canvas_uses_raw_neighbours_vertical_then_horizontal_and_retains_host() {
    let h = host();
    let plan = FastH3VideoSpatialPlan::new(
        &config(),
        [2, 2, 7, 20, 20],
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    assert_eq!(plan.tile_count(), 4);
    let mut canvas = plan.begin_canvas(2, &h).unwrap();
    let raw: Vec<_> = (0..4)
        .map(|tile| {
            let shape = [2, 3, 2, 256, 256];
            let mut bytes = h
                .allocate_zeroed(HostMemoryClass::Output, extent(shape, 2).unwrap())
                .unwrap();
            for (i, w) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let p = i / (256 * 256);
                let y = i / 256 % 256;
                let x = i % 256;
                w.copy_from_slice(&bf16::from_f64(value(tile, p, y, x)).to_bits().to_le_bytes());
            }
            FastH3VideoRawClip {
                shape,
                bytes: bytes.freeze(),
            }
        })
        .collect();
    assert!(canvas.push(1, &raw[1]).is_err());
    assert_eq!(canvas.completed_tiles(), 0);
    // Exhaust admission before staging: failed push must retain exact progress.
    let available = (32 << 20) - h.usage().unwrap().committed_bytes();
    let blocker = h
        .allocate_zeroed(HostMemoryClass::Output, available)
        .unwrap();
    assert!(canvas.push(0, &raw[0]).is_err());
    assert_eq!(canvas.completed_tiles(), 0);
    drop(blocker);
    for (i, tile) in raw.iter().enumerate() {
        canvas.push(i, tile).unwrap();
    }
    let output = canvas.finish().unwrap();
    assert_eq!(output.ncthw_shape(), [2, 3, 2, 320, 320]);
    for (i, w) in output.bytes().bytes().chunks_exact(2).enumerate() {
        let p = i / (320 * 320);
        let y = i / 320 % 320;
        let x = i % 320;
        let row = usize::from(y >= 64);
        let col = usize::from(x >= 64);
        let ly = y - row * 64;
        let lx = x - col * 64;
        let tile = row * 2 + col;
        let mut expected = r(value(tile, p, ly, lx));
        if row > 0 && ly < 192 {
            expected = mix(r(value(tile - 2, p, 64 + ly, lx)), expected, ly, 192);
        }
        if col > 0 && lx < 192 {
            expected = mix(r(value(tile - 1, p, ly, 64 + lx)), expected, lx, 192);
        }
        assert_eq!(
            u16::from_le_bytes(w.try_into().unwrap()),
            bf16::from_f64(expected).to_bits(),
            "plane{p} y{y} x{x}"
        );
    }
    // The crossing must differ from recursively blending the upper-left tile.
    let actual = u16::from_le_bytes(
        output.bytes().bytes()[2 * (100 * 320 + 100)..2 * (100 * 320 + 100) + 2]
            .try_into()
            .unwrap(),
    );
    let recursive_left = mix(r(value(0, 0, 100, 100)), r(value(2, 0, 36, 100)), 36, 192);
    let recursive = mix(
        recursive_left,
        mix(r(value(1, 0, 100, 36)), r(value(3, 0, 36, 36)), 36, 192),
        36,
        192,
    );
    assert_ne!(actual, bf16::from_f64(recursive).to_bits());
    drop(raw);
    assert!(h.owns(output.bytes()));
    drop(output);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
