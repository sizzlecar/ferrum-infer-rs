use super::*;
use serde_json::json;

fn budgets(ttft: u64, tpot: u64, itl: u64) -> SloLatencyBudgets {
    SloLatencyBudgets {
        ttft_ms: NonZeroU64::new(ttft).unwrap(),
        tpot_ms: NonZeroU64::new(tpot).unwrap(),
        itl_ms: NonZeroU64::new(itl).unwrap(),
    }
}

fn service() -> ServiceSloConfig {
    ServiceSloConfig {
        id: "interactive".to_owned(),
        server_token_commit: budgets(500, 40, 80),
        attainment: SloAttainmentTargets::default(),
        client_visible: None,
    }
}

fn configured(mode: SloMode) -> SloConfig {
    SloConfig {
        mode,
        default_service_class: Some("interactive".to_owned()),
        services: vec![service()],
        cost_profile: (mode == SloMode::Enforce).then(|| PathBuf::from("profiles/local.json")),
        ..Default::default()
    }
}

#[test]
fn absent_slo_fields_preserve_off_without_inventing_a_service_commitment() {
    let config: SloConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(config, SloConfig::default());
    assert_eq!(config.mode, SloMode::Off);
    assert!(config.default_service().is_none());
    assert!(config.services.is_empty());
    assert!(config.cost_profile.is_none());
    config.validate().unwrap();
    let mut serialized = serde_json::to_value(crate::SchedulerConfig::default()).unwrap();
    serialized.as_object_mut().unwrap().remove("slo");
    let restored: crate::SchedulerConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(restored.slo, SloConfig::default());
}

#[test]
fn time_policy_defaults_to_completion_and_requires_explicit_strict_selection() {
    let mut legacy = serde_json::to_value(configured(SloMode::Observe)).unwrap();
    legacy["admission"]
        .as_object_mut()
        .unwrap()
        .remove("time_policy");
    let restored: SloConfig = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        restored.admission.time_policy,
        SloTimeAdmissionPolicy::CompleteRequests
    );
    for policy in [
        SloTimeAdmissionPolicy::CompleteRequests,
        SloTimeAdmissionPolicy::RequireSlo,
    ] {
        let mut config = restored.clone();
        config.admission.time_policy = policy;
        let snapshot =
            crate::RuntimeConfigSnapshot::from_entries([crate::RuntimeConfigEntry::new(
                SLO_CONFIG_RUNTIME_KEY,
                serde_json::to_string(&config).unwrap(),
                crate::RuntimeConfigSource::ConfigFile,
            )]);
        let mut engine = crate::EngineConfig::default();
        engine.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(engine.scheduler.slo.admission.time_policy, policy);
    }
    for value in [json!("reject"), json!(true), json!(null)] {
        assert!(
            serde_json::from_value::<SloAdmissionConfig>(json!({"time_policy":value})).is_err()
        );
    }
}

#[test]
fn output_transport_defaults_preserve_off_and_observe_legacy_behavior() {
    for mode in [SloMode::Off, SloMode::Observe] {
        let mut source = serde_json::to_value(configured(mode)).unwrap();
        source["output"]
            .as_object_mut()
            .unwrap()
            .remove("transport");
        let restored: SloConfig = serde_json::from_value(source).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.mode, mode);
        assert_eq!(restored.output.transport, SloOutputTransport::Legacy);
    }
    assert_eq!(
        serde_json::to_value(SloOutputConfig::default()).unwrap()["transport"],
        "legacy"
    );
}

#[test]
fn credited_output_transport_roundtrips_through_the_typed_runtime_snapshot() {
    for mode in [SloMode::Off, SloMode::Observe] {
        let mut policy = configured(mode);
        policy.output.transport = SloOutputTransport::Credited;
        let wire = serde_json::to_value(&policy).unwrap();
        assert_eq!(wire["output"]["transport"], "credited");
        let snapshot =
            crate::RuntimeConfigSnapshot::from_entries([crate::RuntimeConfigEntry::new(
                SLO_CONFIG_RUNTIME_KEY,
                serde_json::to_string(&policy).unwrap(),
                crate::RuntimeConfigSource::ConfigFile,
            )]);
        let mut engine = crate::EngineConfig::default();
        engine.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(engine.scheduler.slo, policy);
    }
    for value in [
        json!("automatic"),
        json!("Credited"),
        json!(true),
        json!(null),
    ] {
        assert!(serde_json::from_value::<SloOutputConfig>(json!({"transport": value})).is_err());
    }
}

#[test]
fn slo_snapshot_applies_validated_typed_policy_without_environment_aliases() {
    let policy = configured(SloMode::Observe);
    let entry = crate::RuntimeConfigEntry::new(
        SLO_CONFIG_RUNTIME_KEY,
        serde_json::to_string(&policy).unwrap(),
        crate::RuntimeConfigSource::ConfigFile,
    );
    let mut engine = crate::EngineConfig::default();
    let mut snapshot = crate::RuntimeConfigSnapshot::from_entries([entry.clone()]);
    engine.apply_runtime_config_snapshot(&snapshot).unwrap();
    assert_eq!(engine.scheduler.slo, policy);
    snapshot.entries[0].source = crate::RuntimeConfigSource::Env;
    assert!(engine.apply_runtime_config_snapshot(&snapshot).is_err());
    snapshot.entries[0].source = crate::RuntimeConfigSource::Cli;
    snapshot.entries[0].effective_value = "{\"mode\":\"observe\"}".to_owned();
    assert!(engine.apply_runtime_config_snapshot(&snapshot).is_err());
    assert_eq!(
        engine.scheduler.slo, policy,
        "invalid replacement cannot discard the current policy"
    );
}

#[test]
fn enabled_slo_requires_an_explicit_resolvable_default_class() {
    for mode in [SloMode::Observe, SloMode::Enforce] {
        let mut config = configured(mode);
        config.validate().unwrap();
        assert_eq!(config.default_service(), config.service("interactive"));
        assert!(config.service("not-configured").is_none());

        config.default_service_class = None;
        assert!(config
            .validate()
            .unwrap_err()
            .contains("default_service_class"));
        config.default_service_class = Some("missing".to_owned());
        assert!(config
            .validate()
            .unwrap_err()
            .contains("no service configuration"));
        config.default_service_class = Some("interactive".to_owned());
        config.services.clear();
        assert!(config.validate().is_err());
    }
}

#[test]
fn only_strict_time_admission_requires_a_profile_to_validate_configuration() {
    configured(SloMode::Observe).validate().unwrap();
    let mut config = configured(SloMode::Enforce);
    config.cost_profile = None;
    config.validate().unwrap();
    config.admission.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    assert!(config.validate().unwrap_err().contains("cost_profile"));
    config.cost_profile = Some(PathBuf::new());
    assert!(config.validate().unwrap_err().contains("must not be empty"));
    config.cost_profile = Some(PathBuf::from("profile-not-opened-by-types.json"));
    config.validate().unwrap();
    // Structural validation does not claim that this file exists or that its
    // model/shape coverage has been verified. The composition root must do so.
}

#[test]
fn controller_retry_backoff_is_typed_and_checked() {
    let mut config = configured(SloMode::Observe);
    assert_eq!(config.planner.retry_backoff_ms.get(), 1);
    config.planner.retry_backoff_ms = NonZeroU64::new(7).unwrap();
    let roundtrip: SloConfig =
        serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
    assert_eq!(roundtrip.planner.retry_backoff_ms.get(), 7);
    config.planner.retry_backoff_ms = NonZeroU64::new(u64::MAX).unwrap();
    assert!(config.validate().is_err());
}

#[test]
fn service_identity_rejects_duplicates_and_invalid_names_even_when_off() {
    let mut config = configured(SloMode::Off);
    config.services.push(service());
    assert!(config
        .validate()
        .unwrap_err()
        .contains("duplicate service class"));
    for invalid in ["", " interactive", "interactive ", "interactive\n", "a\0b"] {
        let mut config = configured(SloMode::Off);
        config.services[0].id = invalid.to_owned();
        assert!(config.validate().is_err());
    }
    let mut config = configured(SloMode::Off);
    config.default_service_class = Some("interactive ".to_owned());
    assert!(config.validate().is_err());
}

#[test]
fn internal_and_client_visible_targets_round_trip_without_budget_conversion() {
    let mut config = configured(SloMode::Enforce);
    assert!(config.services[0].client_visible.is_none());
    config.services[0].client_visible = Some(SloClientVisibleConfig {
        latency: budgets(800, 60, 120),
        attainment: SloAttainmentTargets {
            ttft_percentile: 95.0,
            min_accepted_joint_attainment: 0.97,
            min_offered_joint_attainment: Some(0.96),
            itl_percentile_scope: SloItlPercentileScope::RequestMaximumGap,
            ..Default::default()
        },
    });
    config.validate().unwrap();
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(json["services"][0]["server_token_commit"]["ttft_ms"], 500);
    assert_eq!(
        json["services"][0]["client_visible"]["latency"]["ttft_ms"],
        800
    );
    let restored: SloConfig = serde_json::from_value(json).unwrap();
    assert_eq!(restored, config);
    let service = restored.default_service().unwrap();
    assert_eq!(
        service.server_token_commit.ttft(),
        Duration::from_millis(500)
    );
    assert_eq!(
        service.server_token_commit.tpot(),
        Duration::from_millis(40)
    );
    assert_eq!(service.server_token_commit.itl(), Duration::from_millis(80));
    assert_eq!(service.attainment.min_accepted_joint_attainment, 0.99);
    assert_eq!(service.attainment.min_offered_joint_attainment, None);
    assert_eq!(
        service
            .client_visible
            .as_ref()
            .unwrap()
            .attainment
            .min_accepted_joint_attainment,
        0.97
    );
    assert_eq!(
        service
            .client_visible
            .as_ref()
            .unwrap()
            .attainment
            .min_offered_joint_attainment,
        Some(0.96)
    );
}

#[test]
fn unknown_fields_fail_at_each_slo_configuration_boundary() {
    let mut config = configured(SloMode::Enforce);
    config.services[0].client_visible = Some(SloClientVisibleConfig {
        latency: budgets(800, 60, 120),
        attainment: SloAttainmentTargets::default(),
    });
    let source = serde_json::to_value(config).unwrap();
    for path in [
        "",
        "/planner",
        "/admission",
        "/output",
        "/services/0",
        "/services/0/server_token_commit",
        "/services/0/attainment",
        "/services/0/client_visible",
        "/services/0/client_visible/latency",
        "/services/0/client_visible/attainment",
    ] {
        let mut value = source.clone();
        value
            .pointer_mut(path)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("misspelled_setting".to_owned(), json!(1));
        assert!(
            serde_json::from_value::<SloConfig>(value).is_err(),
            "{path}"
        );
    }
    let mut value = source;
    value["services"][0]
        .as_object_mut()
        .unwrap()
        .remove("server_token_commit");
    assert!(serde_json::from_value::<SloConfig>(value).is_err());
}

#[test]
fn nonzero_timing_counts_and_buffer_limits_reject_zero_and_negative_wire_values() {
    let source = serde_json::to_value(configured(SloMode::Observe)).unwrap();
    for path in [
        "/services/0/server_token_commit/ttft_ms",
        "/services/0/server_token_commit/tpot_ms",
        "/services/0/server_token_commit/itl_ms",
        "/planner/candidate_limit",
        "/planner/beam_width",
        "/planner/lookahead_waves",
        "/planner/max_planning_us",
        "/planner/max_replan_attempts",
        "/admission/max_active_requests",
        "/admission/max_waiting_requests",
        "/admission/max_waiting_prompt_tokens",
        "/admission/max_waiting_prompt_bytes",
        "/admission/max_wait_ms",
        "/admission/max_sequence_tokens",
        "/output/max_queued_events_per_request",
        "/output/max_queued_bytes_per_request",
        "/output/max_projection_bytes_per_request",
        "/output/terminal_reserve_bytes_per_request",
        "/output/max_total_buffer_bytes",
        "/output/slow_consumer_timeout_ms",
    ] {
        for invalid in [0, -1] {
            let mut value = source.clone();
            *value.pointer_mut(path).unwrap() = json!(invalid);
            assert!(
                serde_json::from_value::<SloConfig>(value).is_err(),
                "{path}"
            );
        }
    }
}

#[test]
fn attainment_targets_validate_finite_percentiles_and_fraction_endpoints() {
    for field in [
        "ttft_percentile",
        "tpot_percentile",
        "itl_percentile",
        "min_accepted_joint_attainment",
        "min_offered_joint_attainment",
        "max_reject_rate",
        "max_error_rate",
    ] {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
            let mut targets = SloAttainmentTargets::default();
            let target = match field {
                "ttft_percentile" => &mut targets.ttft_percentile,
                "tpot_percentile" => &mut targets.tpot_percentile,
                "itl_percentile" => &mut targets.itl_percentile,
                "min_accepted_joint_attainment" => &mut targets.min_accepted_joint_attainment,
                "min_offered_joint_attainment" => targets.min_offered_joint_attainment.insert(0.99),
                "max_reject_rate" => &mut targets.max_reject_rate,
                _ => &mut targets.max_error_rate,
            };
            *target = invalid;
            assert!(targets.validate().unwrap_err().contains(field));
        }
    }
    let targets = SloAttainmentTargets {
        ttft_percentile: 100.0,
        tpot_percentile: 100.0,
        itl_percentile: 100.0,
        min_accepted_joint_attainment: 1.0,
        max_reject_rate: 0.0,
        max_error_rate: 1.0,
        ..Default::default()
    };
    targets.validate().unwrap();
    for (field, invalid) in [
        ("ttft_percentile", 0.0),
        ("tpot_percentile", 100.01),
        ("itl_percentile", 100.01),
        ("min_accepted_joint_attainment", 0.0),
        ("min_accepted_joint_attainment", 1.01),
        ("min_offered_joint_attainment", 0.0),
        ("min_offered_joint_attainment", 1.01),
        ("max_reject_rate", 1.01),
        ("max_error_rate", 1.01),
    ] {
        let mut value = serde_json::to_value(&targets).unwrap();
        value[field] = json!(invalid);
        let parsed: SloAttainmentTargets = serde_json::from_value(value).unwrap();
        assert!(parsed.validate().unwrap_err().contains(field));
    }
}

#[test]
fn planner_weights_allow_ablation_but_not_nonfinite_or_negative_scores() {
    let mut planner = SloPlannerConfig::default();
    planner.prefill_credit_beta = 0.0;
    planner.prefill_debt_gamma = 0.0;
    planner.enable_prefill_milestones = false;
    planner.validate().unwrap();
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        planner.prefill_credit_beta = invalid;
        assert!(planner
            .validate()
            .unwrap_err()
            .contains("prefill_credit_beta"));
        planner.prefill_credit_beta = 1.0;
        planner.prefill_debt_gamma = invalid;
        assert!(planner
            .validate()
            .unwrap_err()
            .contains("prefill_debt_gamma"));
        planner.prefill_debt_gamma = 1.0;
    }
}

#[test]
fn planner_route_domain_limits_are_typed_serialized_and_bounded() {
    let mut planner = SloPlannerConfig::default();
    assert_eq!(planner.max_route_states.get(), 16);
    assert_eq!(planner.max_shape_alternatives.get(), 32);
    planner.max_route_states = NonZeroUsize::new(8).unwrap();
    planner.max_shape_alternatives = NonZeroUsize::new(12).unwrap();
    planner.validate().unwrap();
    let encoded = serde_json::to_value(&planner).unwrap();
    let decoded: SloPlannerConfig = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(decoded, planner);
    for field in ["max_route_states", "max_shape_alternatives"] {
        let mut invalid = encoded.clone();
        invalid[field] = serde_json::json!(0);
        assert!(serde_json::from_value::<SloPlannerConfig>(invalid).is_err());
        let mut invalid = encoded.clone();
        invalid[field] = serde_json::json!(257);
        assert!(serde_json::from_value::<SloPlannerConfig>(invalid)
            .unwrap()
            .validate()
            .is_err());
    }
}

#[test]
fn arithmetic_bounds_reject_duration_search_and_queue_storage_overflow() {
    let mut latency = budgets(500, 40, 80);
    latency.ttft_ms = NonZeroU64::new(u64::MAX).unwrap();
    assert!(latency.validate().unwrap_err().contains("ttft_ms"));

    let mut planner = SloPlannerConfig::default();
    planner.candidate_limit = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(planner.validate().unwrap_err().contains("expansion bound"));
    planner = SloPlannerConfig::default();
    planner.max_planning_us = NonZeroU64::new(u64::MAX).unwrap();
    assert!(planner.validate().unwrap_err().contains("max_planning_us"));

    let mut admission = SloAdmissionConfig::default();
    admission.max_waiting_requests = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(admission.validate().unwrap_err().contains("request count"));
    admission = SloAdmissionConfig::default();
    admission.max_waiting_prompt_tokens = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(admission.validate().unwrap_err().contains("prompt storage"));
    admission = SloAdmissionConfig::default();
    admission.max_wait_ms = NonZeroU64::new(u64::MAX).unwrap();
    assert!(admission.validate().unwrap_err().contains("max_wait_ms"));

    let mut output = SloOutputConfig::default();
    output.max_queued_bytes_per_request = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(output
        .validate()
        .unwrap_err()
        .contains("per-request storage"));
    output = SloOutputConfig::default();
    output.slow_consumer_timeout_ms = NonZeroU64::new(u64::MAX).unwrap();
    assert!(output
        .validate()
        .unwrap_err()
        .contains("slow_consumer_timeout_ms"));
}

#[test]
fn cumulative_decode_deadlines_must_fit_the_admitted_context_range() {
    let mut config = configured(SloMode::Observe);
    config.services[0].server_token_commit.tpot_ms = NonZeroU64::new(u64::MAX / 1_000_000).unwrap();
    config.services[0].validate().unwrap();
    assert!(config.validate().unwrap_err().contains("cumulative TPOT"));
}

#[test]
fn output_budget_preserves_terminal_capacity_and_isolated_active_request_shares() {
    let mut config = configured(SloMode::Observe);
    config.output.terminal_reserve_bytes_per_request = config.output.max_queued_bytes_per_request;
    assert!(config.validate().unwrap_err().contains("terminal reserve"));
    config.output = SloOutputConfig::default();
    config.output.max_total_buffer_bytes = NonZeroUsize::new(1).unwrap();
    assert!(config.validate().unwrap_err().contains("one request"));
    config.output.max_total_buffer_bytes = NonZeroUsize::new(2 * 1_048_576).unwrap();
    assert!(config
        .validate()
        .unwrap_err()
        .contains("max_active_requests"));
    config.admission.max_active_requests = NonZeroUsize::new(1).unwrap();
    config.validate().unwrap();
}

#[test]
fn planner_phase_shares_default_in_old_config_and_reject_empty_replay_share() {
    let config: SloPlannerConfig = serde_json::from_str(r#"{"max_planning_us":1}"#).unwrap();
    assert_eq!(config.search_budget_percent, 60);
    assert_eq!(config.publication_reserve_percent, 20);
    config.validate().unwrap();
    let serialized = serde_json::to_value(&config).unwrap();
    assert_eq!(serialized["search_budget_percent"], 60);
    assert_eq!(serialized["publication_reserve_percent"], 20);
    for (search, publication) in [(0, 20), (60, 0), (80, 20), (100, 1), (255, 255)] {
        let mut invalid = config.clone();
        invalid.search_budget_percent = search;
        invalid.publication_reserve_percent = publication;
        assert!(invalid.validate().is_err(), "{search}/{publication}");
    }
}
