use super::*;

#[test]
fn every_step_matches_hand_calculated_binary_fraction_vectors() {
    // Independent arithmetic fixture: all values are exactly representable.
    // Deltas are -1/4, -1/4, -3/8, -1/8. Each expected row was calculated
    // directly as x + delta * v, not captured from this implementation.
    let schedule = FlowMatchEulerSchedule::new(&[1.0, 0.75, 0.5, 0.125], 0.0, 1000.0).unwrap();
    let velocities = [
        [2.0, -4.0, 1.0],
        [-2.0, 2.0, 0.5],
        [4.0, 0.0, -2.0],
        [-8.0, 4.0, 2.0],
    ];
    let expected = [
        [0.5, -1.0, 0.25],
        [1.0, -1.5, 0.125],
        [-0.5, -1.5, 0.875],
        [0.5, -2.0, 0.625],
    ];
    let expected_times = [1000.0, 750.0, 500.0, 125.0];
    let mut solver = schedule.start();
    let mut sample = [1.0, -2.0, 0.5];
    for (index, velocity) in velocities.iter().enumerate() {
        let step = solver.current_step().unwrap();
        assert_eq!(step.index(), index);
        assert_eq!(step.timestep().as_f32(), expected_times[index]);
        assert!(step.delta_sigma() < 0.0);
        assert_eq!(solver.advance_f32(&mut sample, velocity).unwrap(), step);
        assert_eq!(sample, expected[index]);
        assert_eq!(solver.step_index(), index + 1);
    }
    assert!(solver.is_complete());
    assert_eq!(solver.current_step(), None);
    assert_eq!(
        solver.advance_f32(&mut sample, &[0.0; 3]),
        Err(EulerStepError::Complete)
    );
    assert_eq!(sample, expected[3]);
}

#[test]
fn terminal_sigma_is_an_explicit_endpoint_not_an_extra_evaluation() {
    let schedule = FlowMatchEulerSchedule::new(&[1.0, 0.5], 0.25, 1.0).unwrap();
    assert_eq!(schedule.steps().len(), 2);
    assert_eq!(schedule.steps()[1].next_sigma(), 0.25);
    let mut solver = schedule.start();
    let mut sample = [1.0];
    solver.advance_f32(&mut sample, &[1.0]).unwrap();
    assert_eq!(sample, [0.5]);
    solver.advance_f32(&mut sample, &[1.0]).unwrap();
    assert_eq!(sample, [0.25]);
    assert!(solver.is_complete());
}

#[test]
fn checkpoint_grid_preserves_evaluation_sigmas_and_appends_zero() {
    // Qwen-Image-2.1-Turbo model_index.json, revision
    // fc13ba1986fcb905854dc8de2f52eee395f49b5b. This validates schedule
    // interpretation only; these are not generated-image or denoiser oracles.
    let sigmas = [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
    ];
    let expected_times = [
        1000.0, 978.453, 954.18, 926.626, 895.08, 845.148, 704.534, 414.568,
    ];
    let schedule = FlowMatchEulerSchedule::new(&sigmas, 0.0, 1000.0).unwrap();
    assert_eq!(schedule.steps().len(), sigmas.len());
    for (index, step) in schedule.steps().iter().enumerate() {
        assert_eq!(step.sigma(), sigmas[index]);
        assert!((step.timestep().as_f32() - expected_times[index]).abs() <= f32::EPSILON * 1000.0);
    }
    assert_eq!(schedule.steps().last().unwrap().next_sigma(), 0.0);
}

#[test]
fn invalid_schedules_fail_before_starting_a_request() {
    assert!(matches!(
        FlowMatchEulerSchedule::new(&[], 0.0, 1000.0),
        Err(ScheduleError::EmptySchedule)
    ));
    for sigma in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        assert!(matches!(
            FlowMatchEulerSchedule::new(&[sigma], 0.0, 1000.0),
            Err(ScheduleError::InvalidSigma { index: 0 })
        ));
        assert!(matches!(
            FlowMatchEulerSchedule::new(&[1.0], sigma, 1000.0),
            Err(ScheduleError::InvalidTerminalSigma)
        ));
    }
    for sigmas in [&[0.5, 0.75][..], &[0.5, 0.5], &[1.0, 0.0]] {
        assert!(matches!(
            FlowMatchEulerSchedule::new(sigmas, 0.0, 1000.0),
            Err(ScheduleError::NonDecreasingInterval { .. })
        ));
    }
    assert!(matches!(
        FlowMatchEulerSchedule::new(&[0.5], 0.75, 1000.0),
        Err(ScheduleError::NonDecreasingInterval { index: 0 })
    ));
}

#[test]
fn invalid_or_unrepresentable_timestep_scales_are_rejected() {
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            FlowMatchEulerSchedule::new(&[1.0], 0.0, scale),
            Err(ScheduleError::InvalidTimestepScale)
        ));
    }
    assert!(matches!(
        FlowMatchEulerSchedule::new(&[0.25], 0.0, f32::from_bits(1)),
        Err(ScheduleError::UnrepresentableTimestep { index: 0 })
    ));
    assert!(matches!(
        FlowMatchEulerSchedule::new(&[1.0, 0.75], 0.0, f32::from_bits(1)),
        Err(ScheduleError::UnrepresentableTimestep { index: 1 })
    ));
}

#[test]
fn element_count_and_empty_samples_leave_progress_unchanged() {
    let mut solver = FlowMatchEulerSchedule::new(&[1.0], 0.0, 1.0)
        .unwrap()
        .start();
    let mut sample = [2.0, 3.0];
    assert_eq!(
        solver.advance_f32(&mut sample, &[1.0]),
        Err(EulerStepError::ElementCountMismatch {
            sample: 2,
            model_output: 1
        })
    );
    assert_eq!(sample, [2.0, 3.0]);
    assert_eq!(
        solver.advance_f32(&mut [], &[]),
        Err(EulerStepError::EmptySample)
    );
    assert_eq!(solver.step_index(), 0);
}

#[test]
fn nonfinite_inputs_and_overflow_do_not_partially_mutate_latents() {
    let schedule = FlowMatchEulerSchedule::new(&[1.0], 0.0, 1.0).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut solver = schedule.start();
        let mut sample = [1.0, 2.0];
        assert_eq!(
            solver.advance_f32(&mut sample, &[1.0, bad]),
            Err(EulerStepError::NonFiniteModelOutput { index: 1 })
        );
        assert_eq!(sample, [1.0, 2.0]);
        let mut bad_sample = [1.0, bad];
        assert_eq!(
            solver.advance_f32(&mut bad_sample, &[1.0, 1.0]),
            Err(EulerStepError::NonFiniteSample { index: 1 })
        );
        assert_eq!(bad_sample.map(f32::to_bits), [1.0, bad].map(f32::to_bits));
        assert_eq!(solver.step_index(), 0);
    }
    let mut solver = schedule.start();
    let mut sample = [1.0, f32::MAX];
    assert_eq!(
        solver.advance_f32(&mut sample, &[1.0, -f32::MAX]),
        Err(EulerStepError::NonFiniteResult { index: 1 })
    );
    assert_eq!(sample, [1.0, f32::MAX]);
    assert_eq!(solver.step_index(), 0);
    solver.advance_f32(&mut sample, &[1.0, 0.0]).unwrap();
    assert_eq!(sample, [0.0, f32::MAX]);
    assert!(solver.is_complete());
}

#[test]
fn interleaved_requests_share_only_the_immutable_schedule() {
    let schedule = FlowMatchEulerSchedule::new(&[1.0, 0.5], 0.0, 1000.0).unwrap();
    let mut a = schedule.start();
    let mut b = schedule.start();
    let mut sample_a = [4.0];
    let mut sample_b = [-3.0];
    a.advance_f32(&mut sample_a, &[2.0]).unwrap();
    assert_eq!(b.step_index(), 0);
    b.advance_f32(&mut sample_b, &[-2.0]).unwrap();
    b.advance_f32(&mut sample_b, &[4.0]).unwrap();
    assert_eq!(a.step_index(), 1);
    a.advance_f32(&mut sample_a, &[-6.0]).unwrap();
    assert_eq!(sample_a, [6.0]);
    assert_eq!(sample_b, [-4.0]);
    assert!(a.is_complete() && b.is_complete());
    assert_eq!(schedule.start().step_index(), 0);
}
