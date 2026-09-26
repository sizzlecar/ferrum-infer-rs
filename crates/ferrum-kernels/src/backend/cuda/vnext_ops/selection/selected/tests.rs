use super::*;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;

#[test]
fn cuda_selected_argmax_keeps_scalar_partition_boundary_and_each_participant() {
    let on = SloStructuredCostCapture::HostSettledV1;
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        for (vocabulary, dispatches, blocks) in [(65_535, 1, 1), (65_536, 2, 33)] {
            for participants in [1, 3, 8] {
                let selected = evidence(
                    precision,
                    participants,
                    (0..participants).map(|_| (vocabulary, 512)),
                    on,
                )
                .unwrap();
                selected
                    .validate_command(
                        u64::from(participants),
                        u64::from(participants) * dispatches,
                        0,
                    )
                    .unwrap();
                assert_eq!(
                    selected.work().grid_blocks,
                    u64::from(participants) * blocks
                );
                selected
                    .algorithm_work()
                    .unwrap()
                    .unwrap()
                    .validate_command(&selected)
                    .unwrap();
            }
        }
    }
}

#[test]
fn cuda_selected_argmax_replay_rejects_changed_vocabulary_history_and_precision() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let base = evidence(ArgmaxPrecision::F32, 1, [(248_320, 512)], on).unwrap();
    let resident = SelectedReplayAlgorithmTemplateV1::from_selected(&base, 1, 2, 0).unwrap();
    resident.validate_binding(&base).unwrap();
    for (precision, vocabulary, history) in [
        (ArgmaxPrecision::F32, 248_319, 512),
        (ArgmaxPrecision::F32, 248_320, 513),
        (ArgmaxPrecision::F16, 248_320, 512),
    ] {
        let changed = evidence(precision, 1, [(vocabulary, history)], on).unwrap();
        assert!(resident.validate_binding(&changed).is_err());
    }
    let scaled = evidence(ArgmaxPrecision::F32, 1, [(131_072, 256)], on).unwrap();
    assert_eq!(base.family_signature(), scaled.family_signature());
    assert_ne!(base.work(), scaled.work());
}

#[test]
fn cuda_selected_argmax_disabled_never_visits_rows_and_incomplete_work_is_unknown() {
    let unvisited = std::iter::from_fn(|| -> Option<(i32, i32)> {
        panic!("disabled capture must not construct per-row selected work")
    });
    assert!(evidence(
        ArgmaxPrecision::F32,
        8,
        unvisited,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    let on = SloStructuredCostCapture::HostSettledV1;
    for (count, rows) in [
        (0, vec![]),
        (2, vec![(8, 1)]),
        (1, vec![(8, 1), (8, 1)]),
        (1, vec![(0, 1)]),
        (1, vec![(8, 0)]),
    ] {
        assert!(evidence(ArgmaxPrecision::F32, count, rows, on).is_none());
    }
}
