use super::{fixture::*, *};
use ferrum_models::qwen_image21::QwenImage21TransformerPhase;

#[test]
fn prefix_handoff_counts_two_caches_and_one_weight_installation_before_execution() {
    let (mut pipeline, _directory) = super::adapter::pipeline(1);
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(6, 2, 2, vec![6]).unwrap(),
    )
    .unwrap()
    .with_reference_image(2, 2, 16, 16)
    .unwrap();
    let prefill = pipeline
        .prepare_denoiser(
            &config
                .clone()
                .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
                .unwrap(),
            1,
        )
        .unwrap();
    let decode = pipeline
        .prepare_denoiser(
            &config
                .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
                .unwrap(),
            1,
        )
        .unwrap();
    let pm = prefill
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    let dm = decode
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    assert!(dm.minimum_request_bytes() > 0);
    assert!(pm.static_bytes() >= dm.static_bytes());
    let ordinary = pm.static_bytes()
        + pm.minimum_runnable_request_bytes()
            .max(dm.minimum_runnable_request_bytes());
    let transition = pm.static_bytes()
        + pm.minimum_runnable_request_bytes()
        + dm.minimum_request_bytes()
        + dm.minimum_sequence_bytes();
    assert!(
        transition > ordinary,
        "reference-prefix fixture must exercise simultaneous cache pressure"
    );
    for policy in [
        ImageResidencyPolicy::Cold,
        ImageResidencyPolicy::AllModelComponents,
    ] {
        pipeline.config.residency = policy;
        pipeline.config.device_capacity_bytes = transition - 1;
        assert!(pipeline
            .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)),)
            .is_err());
        pipeline.config.device_capacity_bytes = transition;
        pipeline
            .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
            .unwrap();
    }
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}
