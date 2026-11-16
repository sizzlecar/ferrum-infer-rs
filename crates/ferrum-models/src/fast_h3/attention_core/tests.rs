use super::*;
use ferrum_kernels::native_ops::sage_vsa_fp4::ListShape;
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
#[test]
fn region_owners_preserve_layout_budgets_and_rollback_partial_admission() {
    let cp = FastH3Checkpoint::parse(
        include_bytes!("../fixtures/inference.json"),
        include_bytes!("../fixtures/transformer.json"),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let layout = FastH3VsaLayout::ref2va(
        &cp,
        3,
        &[FastH3Reference::Video {
            latent_frames: 5,
            latent_height: 10,
            latent_width: 18,
            audio_latents_per_channel: Some(2),
        }],
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 10000,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap();
    let bytes = layout.regions().len() * 12 + layout.tile_count() * 8;
    let h = account(bytes);
    let mut digests = vec![];
    for selection in [FastH3VsaSelection::Trained, FastH3VsaSelection::Dense] {
        let p = regions::prepare(&layout, 2, 3, selection, &h, bytes as u64).unwrap();
        assert_eq!(
            p.shape(),
            ListShape::new(
                2,
                3,
                (layout.tile_count() - layout.prefix_tiles()) as u32,
                layout.tile_count() as u32
            )
            .unwrap()
        );
        assert_eq!(h.usage().unwrap().committed_bytes(), bytes);
        assert!(p.checked().is_ok());
        digests.push(p.digest().to_owned());
        let other = Arc::clone(&p);
        drop(p);
        assert_eq!(h.usage().unwrap().committed_bytes(), bytes);
        drop(other);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    assert_ne!(digests[0], digests[1]);
    assert!(regions::prepare(
        &layout,
        2,
        3,
        FastH3VsaSelection::Trained,
        &h,
        bytes as u64 - 1
    )
    .is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let first_only = account(layout.regions().len() * 12);
    assert!(regions::prepare(
        &layout,
        2,
        3,
        FastH3VsaSelection::Trained,
        &first_only,
        bytes as u64
    )
    .is_err());
    assert_eq!(first_only.usage().unwrap().committed_bytes(), 0);
}
use std::sync::Arc;
