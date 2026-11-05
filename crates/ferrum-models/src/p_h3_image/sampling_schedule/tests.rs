use super::*;
use serde_json::{json, Value};

fn index(sigmas: Value) -> Value {
    json!({"_class_name":"PH3ImageModularPipeline", "_blocks_class_name":"PH3ImageBlocks",
        "scheduler":["diffusers","MiniMaxH3Scheduler",{}],
        "audio_scheduler":["diffusers","MiniMaxH3Scheduler",{}],
        "sample_sigmas":sigmas})
}

fn scheduler(shift: f64) -> Value {
    json!({"_class_name":"MiniMaxH3Scheduler", "_diffusers_version":"0.42.0.dev0",
        "shift":shift,"shift_provided_sigmas":false})
}

fn limits() -> PH3ImageSamplingLimits {
    PH3ImageSamplingLimits {
        max_metadata_bytes: 4096,
        max_evaluations: 16,
    }
}

fn parse(index: &Value, video: &Value, audio: &Value) -> Result<PH3ImageSamplingSchedule> {
    PH3ImageSamplingSchedule::parse(
        &serde_json::to_vec(index)?,
        &serde_json::to_vec(video)?,
        &serde_json::to_vec(audio)?,
        limits(),
    )
}

fn simple(sigmas: Value) -> PH3ImageSamplingSchedule {
    parse(&index(sigmas), &scheduler(12.0), &scheduler(12.0)).unwrap()
}

#[test]
fn released_grid_preserves_verbatim_sigmas_and_h3_time_units() {
    let sigmas = [
        1.0_f32, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0,
    ];
    let schedule = simple(json!(sigmas));
    assert_eq!(schedule.evaluation_count(), 8);
    for (step, interval) in schedule.steps().iter().zip(sigmas.windows(2)) {
        assert_eq!(step.sigma(), interval[0]);
        assert_eq!(step.next_sigma(), interval[1]);
        assert_eq!(step.timestep(), 1.0_f32 - interval[0]);
    }
    assert_eq!(schedule.steps()[0].timestep(), 0.0);
    assert_eq!(schedule.steps().last().unwrap().next_sigma(), 0.0);
    // Both shifts are inactive for a finished grid and may differ by modality.
    let different_shift = parse(&index(json!(sigmas)), &scheduler(1.0), &scheduler(9.0)).unwrap();
    assert_eq!(schedule.steps(), different_shift.steps());
}

#[test]
fn positive_data_ward_velocity_advances_both_streams_toward_clean_sample() {
    let mut request = simple(json!([1.0, 0.25, 0.0])).start();
    let mut image = [2.0, -4.0];
    let mut audio = [10.0];
    request
        .advance_f32(&mut image, &[4.0, 2.0], &mut audio, &[-2.0])
        .unwrap();
    assert_eq!(image, [5.0, -2.5]);
    assert_eq!(audio, [8.5]);
    assert_eq!(request.current_step().unwrap().timestep(), 0.75);
    request
        .advance_f32(&mut image, &[4.0, 2.0], &mut audio, &[-2.0])
        .unwrap();
    assert_eq!(image, [6.0, -2.0]);
    assert_eq!(audio, [8.0]);
    assert!(request.is_complete());
    assert!(request.current_step().is_none());
    assert!(request
        .advance_f32(&mut image, &[4.0, 2.0], &mut audio, &[-2.0])
        .is_err());
}

#[test]
fn timestep_round_trip_is_kept_separate_from_grid_sigma() {
    // At this finite sigma, FP32 t=1-sigma rounds to 1. The reference recovers
    // sigma from that t for x0, so the sample stays at 1. Algebraically replacing
    // it with the grid sigma would instead yield 2 for the chosen velocity.
    let mut request = simple(json!([1e-8, 0.0])).start();
    assert_eq!(request.current_step().unwrap().timestep(), 1.0);
    let mut image = [1.0];
    let mut audio = [1.0];
    request
        .advance_f32(&mut image, &[1e8], &mut audio, &[1e8])
        .unwrap();
    assert_eq!(image, [1.0]);
    assert_eq!(audio, [1.0]);
}

#[test]
fn interleaved_requests_have_independent_progress_and_state() {
    let schedule = simple(json!([1.0, 0.5, 0.0]));
    let mut first = schedule.start();
    let second = schedule.start();
    let mut image = [0.0];
    let mut audio = [0.0];
    first
        .advance_f32(&mut image, &[2.0], &mut audio, &[4.0])
        .unwrap();
    assert_eq!(first.step_index(), 1);
    assert_eq!(second.step_index(), 0);
    assert_eq!(second.current_step().unwrap().timestep(), 0.0);
    assert_eq!(schedule.evaluation_count(), 2);
}

#[test]
fn late_audio_failure_cannot_partially_mutate_image_or_cursor() {
    let mut request = simple(json!([1.0, 0.0])).start();
    for bad_velocity in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut image = [2.0];
        let mut audio = [f32::MAX];
        assert!(request
            .advance_f32(&mut image, &[1.0], &mut audio, &[bad_velocity])
            .is_err());
        assert_eq!(image, [2.0]);
        assert_eq!(audio, [f32::MAX]);
        assert_eq!(request.step_index(), 0);
    }
    assert!(request
        .advance_f32(&mut [1.0], &[1.0], &mut [], &[])
        .is_err());
    assert!(request
        .advance_f32(&mut [1.0], &[], &mut [1.0], &[1.0])
        .is_err());
    assert_eq!(request.step_index(), 0);
}

#[test]
fn invalid_or_ambiguous_metadata_and_unsupported_scheduler_modes_are_rejected() {
    for sigmas in [
        json!([]),
        json!([0.0]),
        json!([1.0, 0.5]),
        json!([1.0, 0.5, 0.5, 0.0]),
        json!([1.1, 0.0]),
        json!([1.0, -0.1, 0.0]),
        json!([1.0, true, 0.0]),
        json!([1.0, "0.5", 0.0]),
        json!([1.0, 0.999999999, 0.0]),
    ] {
        assert!(parse(&index(sigmas), &scheduler(12.0), &scheduler(3.0)).is_err());
    }
    for (field, value) in [
        ("_class_name", json!("FlowMatchEulerDiscreteScheduler")),
        ("shift", json!(0.0)),
        ("shift_provided_sigmas", json!(true)),
        ("stochastic_sampling", json!(true)),
        ("new_rule", json!(true)),
    ] {
        let mut config = scheduler(3.0);
        config[field] = value;
        assert!(
            parse(&index(json!([1.0, 0.0])), &scheduler(12.0), &config).is_err(),
            "{field}"
        );
    }
    let original = serde_json::to_string(&scheduler(12.0)).unwrap();
    let duplicate = format!("{{\"shift_provided_sigmas\":true,{}", &original[1..]);
    assert!(PH3ImageSamplingSchedule::parse(
        &serde_json::to_vec(&index(json!([1.0, 0.0]))).unwrap(),
        duplicate.as_bytes(),
        &serde_json::to_vec(&scheduler(3.0)).unwrap(),
        limits()
    )
    .is_err());
}

#[test]
fn declarations_required_fields_and_explicit_determinism_are_checked() {
    let original = index(json!([1.0, 0.0]));
    for (field, value) in [
        ("_class_name", json!("QwenImage21Pipeline")),
        ("_blocks_class_name", json!("OtherBlocks")),
        ("scheduler", json!(["other", "MiniMaxH3Scheduler", {}])),
        (
            "audio_scheduler",
            json!(["diffusers", "FlowMatchEulerDiscreteScheduler", {}]),
        ),
    ] {
        let mut altered = original.clone();
        altered[field] = value;
        assert!(
            parse(&altered, &scheduler(12.0), &scheduler(12.0)).is_err(),
            "{field}"
        );
    }
    let mut absent_audio = original.clone();
    absent_audio
        .as_object_mut()
        .unwrap()
        .remove("audio_scheduler");
    assert!(parse(&absent_audio, &scheduler(12.0), &scheduler(12.0)).is_err());
    let mut absent_rule = scheduler(12.0);
    absent_rule
        .as_object_mut()
        .unwrap()
        .remove("shift_provided_sigmas");
    assert!(parse(&original, &absent_rule, &scheduler(12.0)).is_err());
    let mut deterministic = scheduler(12.0);
    deterministic["stochastic_sampling"] = json!(false);
    assert!(parse(&original, &deterministic, &deterministic).is_ok());
    let original = serde_json::to_string(&original).unwrap();
    let duplicate = format!("{{\"sample_sigmas\":[1.0,0.25,0.0],{}", &original[1..]);
    let config = serde_json::to_vec(&deterministic).unwrap();
    assert!(
        PH3ImageSamplingSchedule::parse(duplicate.as_bytes(), &config, &config, limits()).is_err()
    );
    assert!(PH3ImageSamplingSchedule::parse(b"[]", &config, &config, limits()).is_err());
}

#[test]
fn caller_declared_byte_and_retained_evaluation_capacities_are_enforced() {
    let index = serde_json::to_vec(&index(json!([1.0, 0.5, 0.0]))).unwrap();
    let scheduler = serde_json::to_vec(&scheduler(12.0)).unwrap();
    let mut capacity = PH3ImageSamplingLimits {
        max_metadata_bytes: index.len(),
        max_evaluations: 2,
    };
    assert!(PH3ImageSamplingSchedule::parse(&index, &scheduler, &scheduler, capacity).is_ok());
    capacity.max_evaluations = 1;
    assert!(PH3ImageSamplingSchedule::parse(&index, &scheduler, &scheduler, capacity).is_err());
    capacity.max_evaluations = 2;
    capacity.max_metadata_bytes -= 1;
    assert!(PH3ImageSamplingSchedule::parse(&index, &scheduler, &scheduler, capacity).is_err());
}
