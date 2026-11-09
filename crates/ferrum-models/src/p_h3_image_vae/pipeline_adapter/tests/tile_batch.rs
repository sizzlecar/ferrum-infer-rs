use super::*;

#[test]
fn spatial_tile_recipes_keep_semantic_request_batch_and_original_numerical_weights() {
    let (_dir, source, _, _) = fixture();
    let adapter = PH3ImageVaeAdapter::new(source, &metadata(), limits()).unwrap();
    let prepared = adapter
        .prepare_decoder_with_tile_batch(
            2,
            96,
            64,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 32,
            },
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
            PH3ImageVaeTileBatch::new(4).unwrap(),
        )
        .unwrap();
    assert_eq!(
        prepared
            .shapes()
            .iter()
            .map(|s| (s.tile_count(), s.physical_batch()))
            .collect::<Vec<_>>(),
        [(4, 8), (3, 6)]
    );
    assert_eq!(
        prepared.normalized_latent_spec().unwrap().dimensions(),
        [2, 24, 4, 6]
    );
    assert_eq!(
        prepared.pixel_output_spec().unwrap().dimensions(),
        [2, 3, 64, 96]
    );
    let source_schema = prepared.shapes()[0]
        .family()
        .stored_weight_schema()
        .unwrap();
    for shape in prepared.shapes() {
        assert_eq!(shape.request_batch(), 2);
        assert_eq!(
            shape.normalized_latent_spec().unwrap(),
            prepared.normalized_latent_spec().unwrap()
        );
        assert_eq!(
            shape.pixel_output_spec().unwrap(),
            prepared.pixel_output_spec().unwrap()
        );
        assert_eq!(
            shape.family().stored_weight_schema().unwrap(),
            source_schema
        );
        assert_eq!(
            shape.family().numerical_profile(),
            prepared.shapes()[0].family().numerical_profile()
        );
        assert_eq!(
            shape.tile_layout_specs().unwrap().0.dimensions(),
            [shape.physical_batch() * 4, 24]
        );
    }
    assert!(prepared.shape(2).is_err());
    assert!(prepared.shape(1).is_err());
    let whole = adapter
        .prepare_decoder_with_tile_batch(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
            PH3ImageVaeTileBatch::new(4).unwrap(),
        )
        .unwrap();
    assert_eq!(whole.shapes().len(), 1);
    assert_eq!(whole.shapes()[0].physical_batch(), 1);
    let legacy = adapter
        .prepare_decoder(
            1,
            32,
            32,
            PH3ImageVaeSpatialPolicy::WholeImage,
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
        )
        .unwrap();
    assert_eq!(
        legacy.family().fingerprint().unwrap(),
        whole.shapes()[0].family().fingerprint().unwrap()
    );
}

#[cfg(feature = "cuda")]
mod cuda;
