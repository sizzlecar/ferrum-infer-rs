use super::*;
use crate::fast_h3::FastH3MetadataLimits;

fn checkpoint() -> FastH3Checkpoint {
    checkpoint_from(include_bytes!("../fixtures/inference.json"))
}
fn checkpoint_from(inference: &[u8]) -> FastH3Checkpoint {
    FastH3Checkpoint::parse(
        inference,
        include_bytes!("../fixtures/transformer.json"),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap()
}
fn limits() -> FastH3VsaLayoutLimits {
    FastH3VsaLayoutLimits {
        maximum_packed_rows: 10_000,
        maximum_tiles: 1000,
        maximum_metadata_bytes: 1 << 20,
    }
}
fn references() -> [FastH3Reference; 2] {
    [
        FastH3Reference::Video {
            latent_frames: 5,
            latent_height: 10,
            latent_width: 18,
            audio_latents_per_channel: Some(2),
        },
        FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        },
    ]
}
pub(super) fn layout() -> FastH3VsaLayout {
    FastH3VsaLayout::ref2va(
        &checkpoint(),
        3,
        &references(),
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        limits(),
    )
    .unwrap()
}
fn mask_limits() -> FastH3VsaMaskLimits {
    FastH3VsaMaskLimits {
        maximum_mask_entries: 1 << 20,
        maximum_descriptor_bytes: 1 << 22,
    }
}

#[test]
fn tile128_matches_source_segment_order_and_spatial_border_fixture() {
    let layout = layout();
    // Original packed order: text3, ref-audio4, ref-video225, image2,
    // target-audio16, target-video8. Dense segments move first, separately.
    assert_eq!(layout.packed().sequence_rows(), 258);
    assert_eq!(layout.prefix_tiles(), 4);
    assert_eq!(
        layout.valid_rows_per_tile(),
        &[3, 4, 2, 16, 128, 16, 32, 4, 32, 4, 8, 1, 8]
    );
    assert_eq!(layout.regions()[0].tiles(), 4..12);
    assert_eq!(
        layout.regions()[0].kind(),
        FastH3VsaRegionKind::ReferenceVideo
    );
    assert_eq!(layout.regions()[1].tiles(), 12..13);
    let map = layout.tile_to_packed_rows();
    assert_eq!(&map[..3], &[0, 1, 2]);
    assert_eq!(&map[128..132], &[3, 4, 5, 6]);
    assert_eq!(&map[256..258], &[232, 233]);
    assert_eq!(&map[384..400], &(234..250).collect::<Vec<_>>());
    // The narrow W-border tile contains contiguous valid slots, despite
    // noncontiguous original raster rows. These values are hand-enumerated.
    assert_eq!(
        &map[640..656],
        &[15, 24, 33, 42, 60, 69, 78, 87, 105, 114, 123, 132, 150, 159, 168, 177]
    );
    assert_eq!(map[4 * 128 + 127], 176);
    assert_eq!(map[11 * 128], 231);
    assert_eq!(
        &map[12 * 128..12 * 128 + 8],
        &(250..258).collect::<Vec<_>>()
    );
    for (tile, &valid) in layout.valid_rows_per_tile().iter().enumerate() {
        assert!(map[tile * 128 + valid as usize..(tile + 1) * 128]
            .iter()
            .all(|&v| v == FAST_H3_VSA_PADDING_ROW));
    }
    let mut sorted = map
        .iter()
        .copied()
        .filter(|&v| v != FAST_H3_VSA_PADDING_ROW)
        .collect::<Vec<_>>();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..258).collect::<Vec<_>>());
    for (packed, &tile) in layout.packed_to_tile_rows().iter().enumerate() {
        assert_eq!(map[tile as usize] as usize, packed);
    }
}

#[test]
fn image_reference_stays_dense_and_does_not_create_a_sparse_region() {
    let layout = FastH3VsaLayout::ref2va(
        &checkpoint(),
        2,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        }],
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(layout.valid_rows_per_tile(), &[2, 2, 16, 8]);
    assert_eq!(layout.prefix_tiles(), 3);
    assert_eq!(layout.regions().len(), 1);
    assert_eq!(layout.regions()[0].kind(), FastH3VsaRegionKind::TargetVideo);
}

fn selected_mask(layout: &FastH3VsaLayout, batch: usize, heads: usize) -> Vec<bool> {
    let tiles = layout.tile_count();
    let mut mask = vec![false; batch * heads * tiles * tiles];
    for b in 0..batch {
        for h in 0..heads {
            for q in 0..tiles {
                let row = &mut mask[((b * heads + h) * tiles + q) * tiles..][..tiles];
                if q < layout.prefix_tiles() {
                    row.fill(true);
                } else {
                    row[..layout.prefix_tiles()].fill(true);
                    for region in layout.regions() {
                        let span = region.tiles();
                        assert_eq!(region.trained_key_tiles(), 1);
                        row[span.start + (b + h + q) % span.len()] = true;
                    }
                }
            }
        }
    }
    mask
}

#[test]
fn dynamic_mask_lowers_descending_lists_and_valid_halves_per_batch_and_head() {
    let layout = layout();
    let mask = selected_mask(&layout, 2, 2);
    let lists = layout
        .lower_mask(2, 2, &mask, FastH3VsaSelection::Trained, mask_limits())
        .unwrap();
    assert_eq!(
        (
            lists.batch(),
            lists.heads(),
            lists.query_blocks(),
            lists.key_blocks()
        ),
        (2, 2, 13, 13)
    );
    assert_eq!(lists.video_query_blocks(), 4..13);
    assert_eq!(
        lists.half_valid(),
        &[3, 0, 4, 0, 2, 0, 16, 0, 64, 64, 16, 0, 32, 0, 4, 0, 32, 0, 4, 0, 8, 0, 1, 0, 8, 0]
    );
    assert_eq!(
        lists.selected_blocks(0, 0, 0).unwrap(),
        &(0..13).rev().collect::<Vec<_>>()
    );
    assert_eq!(
        lists.selected_blocks(0, 0, 4).unwrap(),
        &[12, 8, 3, 2, 1, 0]
    );
    assert_eq!(
        lists.selected_blocks(1, 1, 4).unwrap(),
        &[12, 10, 3, 2, 1, 0]
    );
    assert!(lists.selected_blocks(2, 0, 0).is_none());
    assert!(lists.selected_blocks(0, 2, 0).is_none());
    assert!(lists.selected_blocks(0, 0, 13).is_none());
    // Interpret the descriptor back into visibility independently of the
    // lowering loop: no entry may cross this document's key dimension.
    for row in 0..4 * 13 {
        let n = lists.counts()[row] as usize;
        let keys = &lists.indices()[row * 13..row * 13 + n];
        for k in 0..13 {
            assert_eq!(keys.contains(&(k as i32)), mask[row * 13 + k]);
        }
        assert_eq!(keys.last(), Some(&0));
        assert!(lists.indices()[row * 13 + n..(row + 1) * 13]
            .iter()
            .all(|&v| v == 0));
    }
}

#[test]
fn lowering_rejects_prefix_leaks_region_budget_changes_and_concatenated_documents() {
    let layout = layout();
    let valid = selected_mask(&layout, 1, 1);
    let reject = |mask: &[bool]| {
        layout
            .lower_mask(1, 1, mask, FastH3VsaSelection::Trained, mask_limits())
            .is_err()
    };
    let mut missing_prefix = valid.clone();
    missing_prefix[4 * 13] = false;
    assert!(reject(&missing_prefix));
    let mut sparse_prefix_query = valid.clone();
    sparse_prefix_query[12] = false;
    assert!(reject(&sparse_prefix_query));
    // Same global number of selected video keys, wrong per-region budget.
    let mut stolen_target_budget = valid.clone();
    stolen_target_budget[4 * 13 + 12] = false;
    stolen_target_budget[4 * 13 + 9] = true;
    assert!(reject(&stolen_target_budget));
    assert!(reject(&vec![true; 26 * 26]));
    let dense = vec![true; 13 * 13];
    assert!(reject(&dense));
    let dense = layout
        .lower_mask(1, 1, &dense, FastH3VsaSelection::Dense, mask_limits())
        .unwrap();
    assert!(dense.counts().iter().all(|&count| count == 13));
    assert!(layout
        .lower_mask(1, 1, &valid, FastH3VsaSelection::Dense, mask_limits())
        .is_err());
}

#[test]
fn declared_capacities_and_integer_overflow_fail_before_map_or_list_allocation() {
    let layout = layout();
    let build = |limits| {
        FastH3VsaLayout::ref2va(
            &checkpoint(),
            3,
            &references(),
            &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
            limits,
        )
    };
    let exact = FastH3VsaLayoutLimits {
        maximum_packed_rows: layout.packed().sequence_rows(),
        maximum_tiles: layout.tile_count(),
        maximum_metadata_bytes: layout.metadata_bytes(),
    };
    assert!(build(exact).is_ok());
    assert!(build(FastH3VsaLayoutLimits {
        maximum_packed_rows: 257,
        ..exact
    })
    .is_err());
    assert!(build(FastH3VsaLayoutLimits {
        maximum_tiles: 12,
        ..exact
    })
    .is_err());
    assert!(build(FastH3VsaLayoutLimits {
        maximum_metadata_bytes: exact.maximum_metadata_bytes - 1,
        ..exact
    })
    .is_err());
    let mask = selected_mask(&layout, 1, 1);
    let lists = layout
        .lower_mask(1, 1, &mask, FastH3VsaSelection::Trained, mask_limits())
        .unwrap();
    let exact = FastH3VsaMaskLimits {
        maximum_mask_entries: mask.len(),
        maximum_descriptor_bytes: lists.descriptor_bytes(),
    };
    assert!(layout
        .lower_mask(1, 1, &mask, FastH3VsaSelection::Trained, exact)
        .is_ok());
    for limited in [
        FastH3VsaMaskLimits {
            maximum_mask_entries: mask.len() - 1,
            ..exact
        },
        FastH3VsaMaskLimits {
            maximum_descriptor_bytes: lists.descriptor_bytes() - 1,
            ..exact
        },
    ] {
        assert!(layout
            .lower_mask(1, 1, &mask, FastH3VsaSelection::Trained, limited)
            .is_err());
    }
    assert!(layout
        .lower_mask(usize::MAX, 2, &[], FastH3VsaSelection::Trained, exact)
        .is_err());
    assert!(layout
        .lower_mask(0, 1, &[], FastH3VsaSelection::Trained, exact)
        .is_err());
    // No giant mask allocation is needed to reject individually valid int
    // dimensions whose product overflows the kernel's scheduler/index math.
    let error = layout
        .lower_mask(
            i32::MAX as usize / layout.tile_count() + 1,
            1,
            &[],
            FastH3VsaSelection::Trained,
            exact,
        )
        .unwrap_err();
    assert!(error.to_string().contains("native work-grid extent"));
}

#[test]
fn checkpoint_tile64_does_not_silently_select_a_different_native_shape() {
    let mut inference: serde_json::Value =
        serde_json::from_slice(include_bytes!("../fixtures/inference.json")).unwrap();
    inference["vsa_tile_size"] = 64.into();
    let checkpoint = checkpoint_from(&serde_json::to_vec(&inference).unwrap());
    assert!(FastH3VsaLayout::ref2va(
        &checkpoint,
        3,
        &references(),
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        limits(),
    )
    .is_err());
}

#[test]
fn zero_target_sparsity_makes_reference_video_regions_dense_as_well() {
    let mut inference: serde_json::Value =
        serde_json::from_slice(include_bytes!("../fixtures/inference.json")).unwrap();
    inference["vsa_sparsity"] = 0.0.into();
    let layout = FastH3VsaLayout::ref2va(
        &checkpoint_from(&serde_json::to_vec(&inference).unwrap()),
        3,
        &references(),
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(layout.regions()[0].trained_key_tiles(), 8);
    assert!(layout
        .lower_mask(
            1,
            1,
            &vec![true; 13 * 13],
            FastH3VsaSelection::Trained,
            mask_limits(),
        )
        .is_ok());
}

#[test]
fn each_reference_video_retains_its_own_sparse_budget_and_row_span() {
    let first = references();
    let refs = [first[0].clone(), first[1].clone(), first[0].clone()];
    let layout = FastH3VsaLayout::ref2va(
        &checkpoint(),
        3,
        &refs,
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        limits(),
    )
    .unwrap();
    // Prefix is text, first ref audio, image, second ref audio, target audio.
    assert_eq!(layout.prefix_tiles(), 5);
    assert_eq!(layout.regions()[0].tiles(), 5..13);
    assert_eq!(layout.regions()[1].tiles(), 13..21);
    assert_eq!(layout.regions()[2].tiles(), 21..22);
    let mask = selected_mask(&layout, 1, 1);
    assert!(layout
        .lower_mask(1, 1, &mask, FastH3VsaSelection::Trained, mask_limits())
        .is_ok());
    let mut wrong = mask;
    let query = 5;
    let row = &mut wrong[query * 22..(query + 1) * 22];
    row[13 + query % 8] = false;
    row[5 + (query + 1) % 8] = true;
    assert!(layout
        .lower_mask(1, 1, &wrong, FastH3VsaSelection::Trained, mask_limits())
        .is_err());
}
