//! One actual DiT forward with independent scheduler times, compared with
//! standalone requests. This checks model batching, not service scheduling.
use super::*;

fn run_rows(fixture: &Fixture, rows: &[(usize, f32)]) -> Vec<u16> {
    let lengths = [2, 1];
    let config = QwenImage21TransformerConfig::new(
        fixture.config.transformer().clone(),
        QwenImage21T2iLayout::new(
            TEXT as u32,
            2,
            2,
            rows.iter().map(|&(row, _)| lengths[row]).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let time_bytes = (rows.len() + 1) * 256 * 2;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            time_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: time_bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let time = config
        .managed_time_features_per_sample(
            &host,
            &rows.iter().map(|&(_, time)| time).collect::<Vec<_>>(),
        )
        .unwrap();
    let mut inputs = BTreeMap::from([
        (id(F::TIME_FEATURES_ID), time.bytes().bytes().to_vec()),
        (
            id(F::FREQUENCIES_ID),
            fixture.input[&id(F::FREQUENCIES_ID)].clone(),
        ),
    ]);
    drop(time);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    for (name, stride) in [(F::TEXT_ID, TEXT * H * 2), (F::LATENTS_ID, IMAGE * H * 2)] {
        let source = &fixture.input[&id(name)];
        let mut data = Vec::with_capacity(stride * rows.len());
        for &(row, _) in rows {
            data.extend_from_slice(&source[row * stride..(row + 1) * stride]);
        }
        inputs.insert(id(name), data);
    }
    let family = TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    execute(
        &family,
        config.compile_options().unwrap(),
        &fixture.full.source,
        inputs,
    )
    .remove(&id(F::VELOCITY_ID))
    .unwrap()
}

#[test]
fn independent_progress_and_reordered_rows_match_standalone_dit_forwards() {
    let fixture = Fixture::new();
    let first = run_rows(&fixture, &[(0, 1000.0)]);
    let later = run_rows(&fixture, &[(1, 320.0)]);
    let batched = run_rows(&fixture, &[(0, 1000.0), (1, 320.0)]);
    assert_eq!(&batched[..IMAGE * H], first);
    assert_eq!(&batched[IMAGE * H..], later);

    let reordered = run_rows(&fixture, &[(1, 320.0), (0, 1000.0)]);
    assert_eq!(&reordered[..IMAGE * H], later);
    assert_eq!(&reordered[IMAGE * H..], first);

    // Advancing one request's time must affect its velocity without changing
    // the other request's attention, conditioning or target-row selection.
    let changed = run_rows(&fixture, &[(0, 1000.0), (1, 1000.0)]);
    assert_eq!(&changed[..IMAGE * H], first);
    assert_ne!(&changed[IMAGE * H..], later);
}
