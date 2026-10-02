use super::*;

fn case(width: usize, prefix: PrefixKind) -> Case {
    Case {
        product: OpportunityProduct::Prefill,
        template: 0,
        width,
        maximum_output: NonZeroUsize::new(3).unwrap(),
        release_generated: 0,
        suffix_tokens: 3,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix,
        route: CalibrationDecodeRoute::Actual,
        reset: true,
        acquisition: None,
    }
}

#[test]
fn row_ceiling_keeps_width_and_charges_joint_and_serial_prefill_work() {
    let ceiling = NonZeroU32::new(2);
    for (width, expected_chunk) in [(1, 2), (2, 2), (3, 2), (4, 1), (7, 1)] {
        // Walk the declared original token stream, including its short tail.
        let mut left = 5;
        let mut prefills = 0;
        while left > 0 {
            left -= left.min(expected_chunk);
            prefills += 1;
        }
        let ordinary = case(width, PrefixKind::Ordinary);
        assert_eq!(
            ordinary.waves_with_row_ceiling(5, 7, ceiling).unwrap(),
            (prefills + 2, width * (prefills + 2)),
        );
        let prepared = case(width, PrefixKind::Clean);
        assert_eq!(
            prepared.waves_with_row_ceiling(5, 7, ceiling).unwrap(),
            (width * prefills + 2, width * (prefills + 2)),
            "prepared prefix preparation still executes every original row separately",
        );
    }
    assert!(case(8, PrefixKind::Ordinary)
        .waves_with_row_ceiling(5, 7, ceiling)
        .is_err());
}

#[test]
fn row_ceiling_continuation_offsets_follow_the_same_actual_per_row_schedule() {
    for (whole, width, prompt, expected) in [(2, 2, 3, vec![1, 2]), (8, 3, 5, vec![1, 4])] {
        let mut cases = vec![case(width, PrefixKind::Ordinary)];
        append_continuation_prefill_cases_with_row_ceiling(
            &mut cases,
            &[prompt],
            whole,
            NonZeroU32::new(1),
            usize::MAX,
        )
        .unwrap();
        let offsets: Vec<_> = cases[1..]
            .iter()
            .map(|span| match span.product {
                OpportunityProduct::ContinuationPrefill { offset } => offset,
                _ => panic!("expected typed continuation target"),
            })
            .collect();
        assert_eq!(offsets, expected);
        for span in &cases[1..] {
            assert_eq!(span.width, width);
            assert_eq!(span.maximum_output, cases[0].maximum_output);
            assert_eq!(span.preset, cases[0].preset);
            assert_eq!(span.route, cases[0].route);
            assert_eq!(
                span.waves_with_row_ceiling(prompt, whole, NonZeroU32::new(1))
                    .unwrap(),
                cases[0]
                    .waves_with_row_ceiling(prompt, whole, NonZeroU32::new(1))
                    .unwrap(),
                "target selection never removes the original complete source trajectory",
            );
        }
    }
}
