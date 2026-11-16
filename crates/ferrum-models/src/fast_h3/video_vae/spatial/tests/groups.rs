use super::*;

#[test]
fn uniform_groups_choose_exact_divisors_and_always_keep_scalar_fallback() {
    for (shape, expected) in [
        ([1, 2, 7, 30, 52], vec![4, 3, 2, 1]),
        ([1, 2, 7, 40, 40], vec![3, 1]),
        ([1, 2, 7, 16, 52], vec![4, 2, 1]),
    ] {
        let plan = FastH3VideoSpatialPlan::new(
            &config(),
            shape,
            FastH3VideoSpatialPolicy::default(),
            32 << 20,
        )
        .unwrap();
        let candidates = plan.uniform_tile_group_candidates(SpatialTileBatch::new(4).unwrap());
        assert_eq!(
            candidates
                .iter()
                .map(|b| b.maximum_tiles())
                .collect::<Vec<_>>(),
            expected
        );
        for batch in candidates {
            assert_eq!(plan.tile_group_shapes(batch), [batch.maximum_tiles()]);
            assert_eq!(plan.tile_count() % batch.maximum_tiles(), 0);
        }
    }
}

#[test]
fn large_group_preference_is_bounded_by_the_real_canvas() {
    let plan = FastH3VideoSpatialPlan::new(
        &config(),
        [1, 2, 7, 30, 52],
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    let candidates = plan.uniform_tile_group_candidates(SpatialTileBatch::new(usize::MAX).unwrap());
    assert_eq!(
        candidates
            .iter()
            .map(|b| b.maximum_tiles())
            .collect::<Vec<_>>(),
        [12, 6, 4, 3, 2, 1]
    );
    for batch in candidates {
        assert_eq!(plan.tile_group_shapes(batch), [batch.maximum_tiles()]);
        assert_eq!(
            plan.grouped_clip_grid(7, batch.maximum_tiles()).unwrap(),
            [batch.maximum_tiles() as u64, 7, 16, 16]
        );
    }
}

#[test]
fn spatial_groups_keep_requests_and_time_distinct_and_pack_only_real_tiles() {
    let config = config();
    let host = host();
    let shape = [2, 2, 37, 30, 52];
    let plan = FastH3VideoSpatialPlan::new(
        &config,
        shape,
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    let temporal = FastH3VideoDecodePlan::new(&config, 37, 124).unwrap();
    let mut source = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, extent(shape, 4).unwrap())
        .unwrap();
    for (i, word) in source.as_mut().chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&(i as f32 / 1024.).to_le_bytes());
    }
    let source = source.freeze();
    for maximum in [1, 2, 3, 4, 6, 7, 12, usize::MAX] {
        let batch = SpatialTileBatch::new(maximum).unwrap();
        let mut first = 0;
        while first < plan.tile_count() {
            let group = plan.tile_group(first, batch).unwrap();
            let grid = plan.grouped_clip_grid(7, group.tile_count()).unwrap();
            assert_eq!(grid, [2 * group.tile_count() as u64, 7, 16, 16]);
            let grouped = plan
                .managed_tile_group_input(
                    &temporal,
                    6,
                    group,
                    FastH3VideoLatentUnits::Normalized,
                    &source,
                    &host,
                )
                .unwrap();
            let mut offset = 0;
            for tile in first..first + group.tile_count() {
                let single = plan
                    .managed_tile_input(
                        &temporal,
                        6,
                        tile,
                        FastH3VideoLatentUnits::Normalized,
                        &source,
                        &host,
                    )
                    .unwrap();
                let end = offset + single.bytes().len();
                assert_eq!(
                    &grouped.bytes().bytes()[offset..end],
                    single.bytes().bytes()
                );
                offset = end;
            }
            assert_eq!(offset, grouped.bytes().len());
            first += group.tile_count();
        }
    }
    assert!(plan.grouped_clip_grid(7, 0).is_err());
    assert!(plan.grouped_clip_grid(7, plan.tile_count() + 1).is_err());
    drop(source);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn grouped_readback_admission_covers_exact_live_backing_and_one_byte_short() {
    let make_host = |bytes| {
        HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    output_bytes: bytes,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            256,
        )
        .unwrap()
    };
    let plan = FastH3VideoSpatialPlan::new(
        &config(),
        [1, 2, 7, 30, 52],
        FastH3VideoSpatialPolicy::default(),
        32 << 20,
    )
    .unwrap();
    for maximum in [1, 4, 12] {
        let batch = SpatialTileBatch::new(maximum).unwrap();
        let bound = plan
            .grouped_output_backing_bytes(1, batch, &make_host(32 << 20))
            .unwrap();
        for deficit in [0, 1] {
            let host = make_host(bound - deficit);
            let mut canvas = plan.begin_canvas(1, &host).unwrap();
            for first in (0..plan.tile_count()).step_by(maximum) {
                let readback = host
                    .allocate_zeroed(HostMemoryClass::Output, maximum * 3 * 256 * 256 * 2)
                    .unwrap();
                for tile in first..first + maximum {
                    let raw = FastH3VideoRawClip {
                        shape: [1, 3, 1, 256, 256],
                        bytes: host
                            .allocate_zeroed(HostMemoryClass::Output, 3 * 256 * 256 * 2)
                            .unwrap()
                            .freeze(),
                    };
                    let result = canvas.push(tile, &raw);
                    if deficit == 1 && tile + 1 == plan.tile_count() {
                        assert!(result.is_err());
                        assert_eq!(canvas.completed_tiles(), tile);
                    } else {
                        result.unwrap();
                    }
                }
                drop(readback);
            }
            drop(canvas);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}
