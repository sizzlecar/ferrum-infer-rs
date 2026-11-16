use super::*;

fn schedule() -> FastH3PddSchedule {
    FastH3PddSchedule::new(32, (0..=32).step_by(4).collect(), 12.0, 3.0, 32).unwrap()
}

#[test]
fn schedule_roundtrip_reconstructs_every_clock_and_coefficient() {
    let source = schedule();
    let bytes = serde_json::to_vec(&source).unwrap();
    let decoded: FastH3PddSchedule = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, source);
    let first = FastH3PddPlan::from_schedule(source, 96, 32, 32).unwrap();
    let second = FastH3PddPlan::from_schedule(decoded, 96, 32, 32).unwrap();
    assert_eq!(first.schedule(), second.schedule());
    assert_eq!(first.fine_grid(), second.fine_grid());
    assert_eq!(first.steps().len(), 8);
    for (a, b) in first.steps().iter().zip(second.steps()) {
        assert_eq!(a.heads, b.heads);
        for (x, y) in [(&a.video, &b.video), (&a.audio, &b.audio)] {
            assert_eq!(x.sigma().to_bits(), y.sigma().to_bits());
            assert_eq!(x.next_sigma().to_bits(), y.next_sigma().to_bits());
            assert_eq!(x.timestep().to_bits(), y.timestep().to_bits());
            assert_eq!(x.integration_weights(), y.integration_weights());
            assert_eq!(x.normalized_head_weights(), y.normalized_head_weights());
        }
    }
    assert_ne!(
        first.steps()[0].video.integration_weights(),
        first.steps()[0].audio.integration_weights()
    );
}

#[test]
fn invalid_or_unbounded_descriptors_never_construct_a_plan() {
    assert!(schedule().validate(31).is_err());
    assert!(FastH3PddPlan::from_schedule(schedule(), 96, 32, 31).is_err());
    for indices in [
        vec![],
        vec![0],
        vec![1, 32],
        vec![0, 31],
        vec![0, 4, 4, 32],
        vec![0, 33, 32],
    ] {
        assert!(FastH3PddSchedule::new(32, indices, 12.0, 3.0, 32).is_err());
    }
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(FastH3PddSchedule::new(32, vec![0, 32], bad, 3.0, 32).is_err());
        assert!(FastH3PddSchedule::new(32, vec![0, 32], 12.0, bad, 32).is_err());
    }
    for grid in [0, 1, usize::MAX] {
        assert!(FastH3PddSchedule::new(grid, vec![0, grid], 1.0, 1.0, usize::MAX).is_err());
    }
    for width in [0, usize::MAX] {
        assert!(FastH3PddPlan::from_schedule(schedule(), width, 32, 32).is_err());
    }
    let mut wire = serde_json::to_value(schedule()).unwrap();
    wire["partition_indices"] = serde_json::json!([0, 4, 3, 32]);
    let decoded = serde_json::from_value::<FastH3PddSchedule>(wire.clone()).unwrap();
    assert!(decoded.validate(32).is_err());
    assert!(FastH3PddPlan::from_schedule(decoded, 96, 32, 32).is_err());
    wire["derived_sigma"] = serde_json::json!(0.5);
    assert!(serde_json::from_value::<FastH3PddSchedule>(wire).is_err());
}
