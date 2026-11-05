use super::*;
use crate::p_h3_image::test_fixture::tiny_transformer;
use PH3ImageModality::{Audio, Image, Text};

#[test]
fn text_to_image_positions_keep_text_time_audio_channels_and_image_patch_order() {
    let config = tiny_transformer(8, 1);
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let layout = PH3ImageRequestLayout::new(&config, &[Text, Text], target, &[]).unwrap();
    assert_eq!(layout.rows(), 10);
    assert_eq!(layout.text_rows(), 0..2);
    assert_eq!(layout.audio_rows(), 2..6);
    assert_eq!(layout.target_rows(), 6..10);
    let mut positions = [[-1.; 3]; 10];
    layout.write_positions(&mut positions).unwrap();
    assert_eq!(
        positions,
        [
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0., 0.],
            [3., 0., 0.],
            [2., 0., 16.],
            [3., 0., 16.],
            [2., 0., 0.],
            [2., 0., 16.],
            [2., 16., 0.],
            [2., 16., 16.]
        ]
    );
    assert_eq!(layout.modality(0), Some(Text));
    assert_eq!(layout.modality(2), Some(Audio));
    assert_eq!(layout.modality(6), Some(Image));
    assert_eq!(layout.image_row(0), Some(6));
    assert_eq!(layout.image_row(3), Some(9));
    assert_eq!(layout.image_row(4), None);
    assert_eq!(layout.position(10), None);
}

#[test]
fn ordered_reference_images_use_their_own_aspect_grid_and_rotary_time() {
    let config = tiny_transformer(8, 1);
    let square = PH3ImageReferenceImage::new(4, 4).unwrap();
    let wide = PH3ImageReferenceImage::new(4, 16).unwrap();
    let references = [square, wide];
    let layout =
        PH3ImageRequestLayout::new(&config, &[Text, Image, Text], square, &references).unwrap();
    assert_eq!(layout.reference_rows(), 3..23);
    assert_eq!(layout.audio_rows(), 23..27);
    assert_eq!(layout.target_rows(), 27..31);
    assert_eq!(layout.image_row_count(), 24);
    assert_eq!(layout.image_row(19), Some(22));
    assert_eq!(layout.image_row(20), Some(27));
    assert_eq!(layout.position(3), Some([3., 0., 0.]));
    // sqrt(4*16)=8; height spans [.25,.75), width [-.5,1.5), scaled by32.
    assert_eq!(layout.position(7), Some([4., 8., -16.]));
    assert_eq!(layout.position(22), Some([4., 16., 40.]));
    assert_eq!(layout.position(23), Some([5., 0., 0.]));
    assert_eq!(layout.position(26), Some([6., 0., 16.]));
    assert_eq!(layout.position(27), Some([5., 0., 0.]));
    assert_eq!(
        layout.row_kind(7),
        Some(PH3ImageRowKind::ReferenceImage { image: 1, patch: 0 })
    );
}

#[test]
fn prefix_vision_tags_do_not_change_prefix_time_or_reference_noise_augmentation() {
    let config = tiny_transformer(8, 1);
    let image = PH3ImageReferenceImage::new(2, 2).unwrap();
    let references = [image];
    let layout =
        PH3ImageRequestLayout::new(&config, &[Text, Image, Text], image, &references).unwrap();
    let mut times = [0.; 9];
    layout
        .write_row_timesteps(0.25, 0.5, 0.99, &mut times)
        .unwrap();
    assert_eq!(times, [0.25, 0.25, 0.25, 0.99, 0.5, 0.5, 0.5, 0.5, 0.25]);
    assert_eq!(layout.position(1), Some([1., 0., 0.]));
    assert_eq!(layout.modality(1), Some(Image));
    layout
        .write_row_timesteps(1.0, 1.0, 0.99, &mut times)
        .unwrap();
    assert_eq!(times, [1.; 9]);
}

#[test]
fn packed_requests_reset_positions_and_have_mutually_isolated_full_attention() {
    let config = tiny_transformer(8, 1);
    let square = PH3ImageReferenceImage::new(4, 4).unwrap();
    let wide = PH3ImageReferenceImage::new(4, 16).unwrap();
    let requests = [
        PH3ImageRequestLayout::new(&config, &[Text, Text], square, &[]).unwrap(),
        PH3ImageRequestLayout::new(&config, &[Text, Text, Text], wide, &[]).unwrap(),
    ];
    let packed = PH3ImagePackedBatchLayout::new(&requests).unwrap();
    assert_eq!(packed.request_range(0), Some(0..10));
    assert_eq!(packed.request_range(1), Some(10..33));
    assert_eq!(packed.request_count(), 2);
    let mut lengths = [0; 2];
    packed.write_segment_lengths(&mut lengths).unwrap();
    assert_eq!(lengths, [10, 23]);
    for query in 0..33 {
        for key in 0..33 {
            assert_eq!(
                packed.allows_attention(query, key),
                (query < 10) == (key < 10)
            );
        }
    }
    assert!(!packed.allows_attention(33, 0));
    let mut positions = [[-1.; 3]; 33];
    packed.write_positions(&mut positions).unwrap();
    assert_eq!(positions[0], [0., 0., 0.]);
    assert_eq!(positions[10], [0., 0., 0.]);
    assert_eq!(positions[17], [3., 8., -16.]);
    assert_eq!(packed.request_at_row(17), Some((1, 7)));
}

#[test]
fn invalid_geometry_modalities_reference_capacity_and_address_space_fail_closed() {
    let config = tiny_transformer(8, 1);
    let image = PH3ImageReferenceImage::new(4, 4).unwrap();
    assert!(PH3ImageReferenceImage::new(0, 4).is_err());
    assert!(PH3ImageRequestLayout::new(&config, &[], image, &[]).is_err());
    assert!(PH3ImageRequestLayout::new(&config, &[Audio], image, &[]).is_err());
    let partial = PH3ImageReferenceImage::new(3, 4).unwrap();
    assert!(PH3ImageRequestLayout::new(&config, &[Text], partial, &[]).is_err());
    assert!(PH3ImageRequestLayout::new(&config, &[Text], image, &[partial]).is_err());
    assert!(PH3ImageRequestLayout::new(&config, &[Text], image, &[image; 9]).is_err());
    let overflow = PH3ImageReferenceImage::new(usize::MAX - 1, 2).unwrap();
    assert!(PH3ImageRequestLayout::new(&config, &[Text], overflow, &[]).is_err());
    assert!(PH3ImagePackedBatchLayout::new(&[]).is_err());
}

#[test]
fn invalid_destinations_or_times_preserve_outputs() {
    let config = tiny_transformer(8, 1);
    let image = PH3ImageReferenceImage::new(2, 2).unwrap();
    let layout = PH3ImageRequestLayout::new(&config, &[Text], image, &[]).unwrap();
    let mut positions = [[9.; 3]; 5];
    assert!(layout.write_positions(&mut positions).is_err());
    assert_eq!(positions, [[9.; 3]; 5]);
    let mut times = [9.; 6];
    assert!(layout
        .write_row_timesteps(f32::NAN, 0.5, 0.99, &mut times)
        .is_err());
    assert!(layout
        .write_row_timesteps(0.25, 1.1, 0.99, &mut times)
        .is_err());
    assert_eq!(times, [9.; 6]);
    let requests = [layout];
    let packed = PH3ImagePackedBatchLayout::new(&requests).unwrap();
    let mut lengths = [7; 2];
    assert!(packed.write_segment_lengths(&mut lengths).is_err());
    assert_eq!(lengths, [7; 2]);
}
