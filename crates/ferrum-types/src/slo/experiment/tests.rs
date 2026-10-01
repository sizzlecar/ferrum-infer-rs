use super::*;

fn configured(stage: SloExperimentStageV1) -> SloConfig {
    let mut config = SloConfig {
        mode: SloMode::Enforce,
        experiment_stage: Some(stage),
        default_service_class: Some("interactive".into()),
        services: vec![ServiceSloConfig {
            id: "interactive".into(),
            server_token_commit: SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(500).unwrap(),
                tpot_ms: NonZeroU64::new(40).unwrap(),
                itl_ms: NonZeroU64::new(80).unwrap(),
            },
            client_visible: None,
            attainment: Default::default(),
        }],
        ..Default::default()
    };
    config.output.transport = stage.output_transport();
    config
}

#[test]
fn stage_ablation_omission_preserves_existing_wire_and_mode_policy() {
    for mode in [SloMode::Off, SloMode::Observe, SloMode::Enforce] {
        let config = SloConfig {
            mode,
            ..Default::default()
        };
        let encoded = serde_json::to_value(&config).unwrap();
        assert!(encoded.get("experiment_stage").is_none());
        let decoded: SloConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, config);
        let policy = decoded.execution_policy();
        assert_eq!(policy.single_wave, mode == SloMode::Enforce);
        assert_eq!(policy.cost_observation, mode != SloMode::Off);
        assert_eq!(policy.time_admission, mode != SloMode::Off);
    }
    assert!(
        serde_json::from_str::<SloConfig>(r#"{"experiment_stage":"historical_old_binary"}"#)
            .is_err()
    );
}

#[test]
fn stage_ablation_declares_independent_execution_deltas() {
    use SloExperimentStageV1::*;
    let baseline = configured(ControlledAdaptiveBaseline).execution_policy();
    let wave = configured(SingleWave).execution_policy();
    assert!(!baseline.single_wave);
    assert_eq!(
        SloExecutionPolicy {
            single_wave: true,
            ..baseline
        },
        wave
    );
    assert_eq!(wave, configured(OutputIsolation).execution_policy());
    let deadline = configured(DeadlineOnly).execution_policy();
    assert_eq!(
        SloExecutionPolicy {
            controller: SloControllerPolicy::DeadlineOnly,
            ..wave
        },
        deadline
    );
    let costs = configured(CostCandidates).execution_policy();
    assert!(costs.cost_observation);
    assert!(!costs.time_admission);
    assert_eq!(costs.controller, SloControllerPolicy::CostCandidates);
    assert_eq!(
        SloExecutionPolicy {
            time_admission: true,
            ..costs
        },
        configured(Complete).execution_policy()
    );
}

#[test]
fn stage_ablation_rejects_conflicting_output_admission_prefix_and_cost_consumers() {
    use SloExperimentStageV1::*;
    for stage in [
        ControlledAdaptiveBaseline,
        SingleWave,
        OutputIsolation,
        DeadlineOnly,
        CostCandidates,
        Complete,
    ] {
        let config = configured(stage);
        config.validate().unwrap();
        let mut incompatible = config.clone();
        incompatible.admission.time_policy = SloTimeAdmissionPolicy::RequireSlo;
        assert!(incompatible.validate().is_err());
        let mut incompatible = config.clone();
        incompatible.mode = SloMode::Observe;
        assert!(incompatible.validate().is_err());
        let mut incompatible = config.clone();
        incompatible.output.transport = match config.output.transport {
            SloOutputTransport::Legacy => SloOutputTransport::Credited,
            SloOutputTransport::Credited => SloOutputTransport::Legacy,
        };
        assert!(incompatible.validate().is_err());
        assert_eq!(
            config
                .validate_experiment_prefix(NonZeroU64::new(1))
                .is_ok(),
            stage == Complete
        );
        config.validate_experiment_prefix(None).unwrap();
        if !config.execution_policy().cost_observation {
            let mut incompatible = config.clone();
            incompatible.cost_profile = Some(PathBuf::from("explicit-cost.json"));
            assert!(incompatible.validate().is_err());
        }
    }
}
