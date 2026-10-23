use super::*;
use serde_json::{json, Value};

fn scheduler() -> Value {
    json!({"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
        "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,
        "max_image_seq_len":8192,"max_shift":0.9,"num_train_timesteps":1000,
        "shift":1.0,"shift_terminal":null,"stochastic_sampling":false,
        "time_shift_type":"exponential","use_beta_sigmas":false,
        "use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false})
}

fn index(sigmas: Value) -> Value {
    json!({"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":sigmas})
}

fn parse(index: &Value, scheduler: &Value) -> Result<QwenImage21SamplingSchedule> {
    QwenImage21SamplingSchedule::parse(&serde_json::to_vec(index)?, &serde_json::to_vec(scheduler)?)
}

#[test]
fn evaluation_count_is_metadata_driven_including_public_capacity_boundary() {
    for count in [1, 2, 8, QWEN_IMAGE21_MAX_SAMPLING_EVALUATIONS] {
        // Binary fractions remain exact through JSON -> F32 and times 1000.
        let sigmas: Vec<_> = (0..count)
            .map(|i| (count - i) as f64 / count as f64)
            .collect();
        let schedule = parse(&index(json!(sigmas)), &scheduler()).unwrap();
        assert_eq!(schedule.evaluation_count(), count);
        for (i, step) in schedule.steps().iter().enumerate() {
            assert_eq!(step.index(), i);
            assert_eq!(step.sigma().to_bits(), (sigmas[i] as f32).to_bits());
            assert_eq!(step.timestep().as_f32(), sigmas[i] as f32 * 1000.);
            let endpoint = if i + 1 == count {
                0.
            } else {
                sigmas[i + 1] as f32
            };
            assert_eq!(step.next_sigma().to_bits(), endpoint.to_bits());
        }
        assert_eq!(schedule.terminal_sigma().to_bits(), 0);
        assert_eq!(schedule.timestep_scale(), 1000.);
        assert_eq!(schedule.metadata_diffusers_version(), "fixture");
        assert_eq!(schedule.clone().into_solver().steps(), schedule.steps());
    }
    let count = QWEN_IMAGE21_MAX_SAMPLING_EVALUATIONS + 1;
    let sigmas: Vec<_> = (0..count)
        .map(|i| (count - i) as f64 / count as f64)
        .collect();
    assert!(parse(&index(json!(sigmas)), &scheduler()).is_err());
}

#[test]
fn fp32_conversion_precedes_time_and_retains_no_request_progress() {
    let original = 0.7_f64;
    let schedule = parse(&index(json!([original, 0.5])), &scheduler()).unwrap();
    assert_eq!(schedule.steps()[0].sigma().to_bits(), 0x3f33_3333);
    assert_eq!(
        schedule.steps()[0].timestep().as_f32().to_bits(),
        700_f32.to_bits()
    );
    let mut first = schedule.solver().start();
    let other = schedule.solver().start();
    let mut sample = [1.];
    first.advance_f32(&mut sample, &[1.]).unwrap();
    assert_eq!(first.step_index(), 1);
    assert_eq!(other.step_index(), 0);
    assert_eq!(schedule.evaluation_count(), 2);
    // Distinct JSON values collapsing to the same FP32 sigma are not repaired.
    assert!(parse(&index(json!([1., 0.999999999])), &scheduler()).is_err());
}

#[test]
fn unsupported_scheduler_semantics_and_unknown_fields_fail_closed() {
    for (field, value) in [
        ("_class_name", json!("OtherEulerScheduler")),
        ("num_train_timesteps", json!(999)),
        ("shift", json!(2.)),
        ("shift_terminal", json!(0.)),
        ("time_shift_type", json!("linear")),
        ("invert_sigmas", json!(true)),
        ("stochastic_sampling", json!(true)),
        ("use_beta_sigmas", json!(true)),
        ("use_dynamic_shifting", json!(true)),
        ("use_exponential_sigmas", json!(true)),
        ("use_karras_sigmas", json!(true)),
        ("base_image_seq_len", json!(0)),
        ("max_image_seq_len", json!(255)),
        ("new_schedule_rule", json!(true)),
    ] {
        let mut config = scheduler();
        config[field] = value;
        assert!(parse(&index(json!([1., 0.5])), &config).is_err(), "{field}");
    }
    for field in [
        "_class_name",
        "num_train_timesteps",
        "shift",
        "use_dynamic_shifting",
    ] {
        let mut config = scheduler();
        config.as_object_mut().unwrap().remove(field);
        assert!(
            parse(&index(json!([1., 0.5])), &config).is_err(),
            "missing {field}"
        );
    }
    // The existing metadata contract treats absent optional terminal shift as
    // null, while a numeric zero is still an unsupported shift request.
    let mut absent_optional = scheduler();
    absent_optional
        .as_object_mut()
        .unwrap()
        .remove("shift_terminal");
    let schedule = parse(&index(json!([1., 0.5])), &absent_optional).unwrap();
    assert_eq!(schedule.steps()[1].next_sigma().to_bits(), 0);
}

#[test]
fn pipeline_scheduler_identity_and_explicit_sigma_shape_are_required() {
    for declaration in [
        json!(null),
        json!("FlowMatchEulerDiscreteScheduler"),
        json!(["other", "FlowMatchEulerDiscreteScheduler"]),
        json!(["diffusers", "OtherScheduler"]),
        json!(["diffusers"]),
        json!(["diffusers", "FlowMatchEulerDiscreteScheduler", "extra"]),
    ] {
        let mut pipeline = index(json!([1., 0.5]));
        pipeline["scheduler"] = declaration;
        assert!(parse(&pipeline, &scheduler()).is_err());
    }
    for sigmas in [
        json!([]),
        json!([0.]),
        json!([-0.]),
        json!([1., 0.]),
        json!([1., 1.]),
        json!([0.5, 1.]),
        json!([1.1]),
        json!([-0.1]),
        json!([1e39]),
        json!([1e-50]),
        json!(["1.0"]),
        json!([null]),
        json!([{}]),
        json!(1.),
        json!(null),
    ] {
        assert!(
            parse(&index(sigmas.clone()), &scheduler()).is_err(),
            "{sigmas}"
        );
    }
    for field in ["scheduler", "sample_sigmas"] {
        let mut pipeline = index(json!([1., 0.5]));
        pipeline.as_object_mut().unwrap().remove(field);
        assert!(parse(&pipeline, &scheduler()).is_err());
    }
}

#[test]
fn duplicated_consumed_fields_are_rejected_from_original_bytes() {
    let scheduler = serde_json::to_vec(&scheduler()).unwrap();
    for raw in [
        r#"{"scheduler":["diffusers","Other"],"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1,0.5]}"#,
        r#"{"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1,1],"sample_sigmas":[1,0.5]}"#,
        r#"{"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1,0.5],"sample_sigmas":[1,1]}"#,
    ] {
        assert!(QwenImage21SamplingSchedule::parse(raw.as_bytes(), &scheduler).is_err());
    }
    let index = serde_json::to_vec(&index(json!([1., 0.5]))).unwrap();
    let text = String::from_utf8(scheduler).unwrap();
    let duplicate = format!(r#"{{"shift":2.0,{}"#, &text[1..]);
    assert!(QwenImage21SamplingSchedule::parse(&index, duplicate.as_bytes()).is_err());
}

#[test]
fn named_metadata_objects_cannot_be_replaced_with_positional_struct_sequences() {
    let config = serde_json::to_vec(&scheduler()).unwrap();
    let positional_index = br#"[["diffusers","FlowMatchEulerDiscreteScheduler"],[1,0.5]]"#;
    assert!(QwenImage21SamplingSchedule::parse(positional_index, &config).is_err());
    let index = serde_json::to_vec(&index(json!([1., 0.5]))).unwrap();
    let positional_scheduler = json!([
        "FlowMatchEulerDiscreteScheduler",
        "fixture",
        256,
        0.5,
        false,
        8192,
        0.9,
        1000,
        1.0,
        null,
        false,
        "exponential",
        false,
        false,
        false,
        false
    ]);
    assert!(QwenImage21SamplingSchedule::parse(
        &index,
        &serde_json::to_vec(&positional_scheduler).unwrap()
    )
    .is_err());
    let mut spaced = b" \t\r\n".to_vec();
    spaced.extend_from_slice(&index);
    assert!(QwenImage21SamplingSchedule::parse(&spaced, &config).is_ok());
}

#[test]
fn unrelated_components_and_producer_version_do_not_dispatch_the_schedule() {
    let mut pipeline = index(json!([1., 0.5]));
    pipeline["_class_name"] = json!("CallerDeclaredPipeline");
    pipeline["transformer"] = json!(["another_package", "SomeTransformer"]);
    pipeline["_name_or_path"] = json!("not-a-repository-selector");
    let mut config = scheduler();
    config["_diffusers_version"] = json!("future-informational-version");
    // Unused shifting geometry is still checked but not silently applied.
    config["base_shift"] = json!(0.125);
    config["max_shift"] = json!(3.);
    let schedule = parse(&pipeline, &config).unwrap();
    assert_eq!(
        schedule.metadata_diffusers_version(),
        "future-informational-version"
    );
    assert_eq!(schedule.steps()[0].sigma(), 1.);
    assert_eq!(schedule.steps()[1].sigma(), 0.5);
    assert_eq!(schedule.steps()[1].next_sigma(), 0.);
}
