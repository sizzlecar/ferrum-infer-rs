//! Independent patch-order oracle with differently shaped native references.
use super::*;

#[test]
fn ordered_references_keep_their_own_geometry_and_precede_the_target() {
    let t = transformer();
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let extents = [
        PH3ImageReferenceImage::new(2, 4).unwrap(),
        PH3ImageReferenceImage::new(4, 2).unwrap(),
    ];
    let layout =
        PH3ImageRequestLayout::new(&t, &[PH3ImageModality::Text], target, &extents).unwrap();
    let mut time = vec![0; layout.rows()];
    time[layout.reference_rows()].fill(2);
    let request = PH3ImageDiTRequest::with_references(
        &t,
        &[PH3ImageModality::Text],
        target,
        &extents,
        &time,
        1 << 20,
    )
    .unwrap();
    let c = PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        vec![request],
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::default(),
    )
    .unwrap();
    let host = account(1 << 20);
    let image = f32_tensor(
        &host,
        "value.multi.target",
        vec![1, 2, 1, 4, 4],
        (0..32).map(|_| 0.),
    );
    let audio = f32_tensor(
        &host,
        "value.multi.audio",
        vec![1, 2, 2, 2],
        (0..8).map(|_| 0.),
    );
    let first = f32_tensor(
        &host,
        "value.multi.first",
        vec![1, 2, 1, 2, 4],
        (0..16).map(|i| i as f32),
    );
    let second = f32_tensor(
        &host,
        "value.multi.second",
        vec![1, 2, 1, 4, 2],
        (0..16).map(|i| 100. + i as f32),
    );
    let connector = f32_tensor(
        &host,
        "value.multi.text",
        vec![1, 4],
        (0..4).map(|i| i as f32 / 32.),
    );
    let samples = [PH3ImageDiTSample {
        image: &image,
        audio: &audio,
    }];
    let references = vec![vec![
        PH3ImageDiTReference {
            latent: &first,
            patch_noise: None,
            augmentation: 1.,
        },
        PH3ImageDiTReference {
            latent: &second,
            patch_noise: None,
            augmentation: 1.,
        },
    ]];
    let keys = cache::fixture_keys(&c, &connector);
    let (basis, cos, sin) = supplemental(&c, &host);
    let prepared = PH3ImageDiTInputAdapter::new(&c, &host)
        .unwrap()
        .prepare_references(
            &samples,
            &references,
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys,
            },
            basis,
            cos,
            sin,
        )
        .unwrap();
    let patches = prepared
        .inputs
        .iter()
        .find(|v| v.value_id().as_str() == PH3ImageDiTFamily::IMAGE_PATCHES_ID)
        .unwrap();
    let actual = patches
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = [
        0., 1., 4., 5., 8., 9., 12., 13., 2., 3., 6., 7., 10., 11., 14., 15., 100., 101., 102.,
        103., 108., 109., 110., 111., 104., 105., 106., 107., 112., 113., 114., 115.,
    ];
    assert_eq!(&actual[..32], &expected);
    assert!(actual[32..].iter().all(|v| *v == 0.));
    let wrong_order = vec![vec![references[0][1], references[0][0]]];
    let (basis, cos, sin) = supplemental(&c, &host);
    assert!(PH3ImageDiTInputAdapter::new(&c, &host)
        .unwrap()
        .prepare_references(
            &samples,
            &wrong_order,
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys
            },
            basis,
            cos,
            sin
        )
        .is_err());
}
