use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1 as PlanningEstimator;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    NonNegativeEnvelopeContractV1, OwnerBlockScheduleV1, StructuredCostTemplatePolicyV1,
    StructuredPopulationPolicyV1, StructuredServiceDomainPolicyV1, WorkAxisAndBranchChallengesV1,
};
use ferrum_types::SloAutomaticCalibrationNumericalStrategyV1 as NumericalStrategy;

pub(super) fn declaration(
    automatic: &SloAutomaticCalibrationSettingsV1,
    domain: CostWorkloadDomainV1,
) -> Result<StructuredServiceDeclarationV7> {
    if !automatic.population_schedule.uses_owner_blocks() {
        return Err(error(
            "prepared startup requires the explicitly selected owner-block population",
        ));
    }
    let mut numerical =
        crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(automatic);
    let mut schedule = OwnerBlockScheduleV1::new(
        automatic.discovery_offered_waves.get(),
        automatic.phase_offered_waves.map(|n| n.get()),
        [numerical.min_phase_samples; 3],
    )
    .map_err(|e| error(format!("automatic cost probe original schedule: {e:?}")))?;
    schedule.prediction_validity =
        crate::continuous_engine::inner::cost_observation::automatic_prediction_validity(automatic);
    numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
    numerical
        .validate()
        .map_err(|e| error(format!("automatic cost probe numerical settings: {e:?}")))?;
    if automatic.maximum_window_ns.get() > numerical.max_sample_age_ns {
        return Err(error(
            "automatic cost probe window exceeds original numerical age",
        ));
    }
    // Match the existing automatic retained-origin sharing rule. The startup
    // source does not get a second copy of the configured numerical allowance.
    let bytes = automatic.maximum_retained_numeric_bytes.get();
    let retained = automatic.maximum_retained_generations.get();
    let share = bytes
        .checked_mul(retained)
        .and_then(|n| n.checked_div(retained.checked_add(3)?))
        .map(|n| n.min(bytes / 2))
        .filter(|n| *n > 0)
        .ok_or_else(|| error("automatic cost probe shared retained budget overflow"))?;
    Ok(StructuredServiceDeclarationV7 {
        schedule,
        route_population: automatic.route_population,
        domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
        nonnegative_envelope: Some(NonNegativeEnvelopeContractV1 {
            algorithm_universe: None,
            population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            planning_estimator: match automatic.numerical_strategy {
                NumericalStrategy::IdentifiedEnvelopeV2 => PlanningEstimator::IdentifiedEnvelopeV2,
                NumericalStrategy::SameSourceJointCellsV1 => {
                    PlanningEstimator::IdentifiedFitJointCellsV1
                }
                NumericalStrategy::IdentifiedFitGlobalResidualV1 => {
                    PlanningEstimator::IdentifiedFitGlobalResidualV1
                }
            },
            template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
            workload_domain: domain,
            settings: Default::default(),
            challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
        }),
        settings: numerical,
        maximum_window_ns: automatic.maximum_window_ns.get(),
        maximum_owners: automatic.maximum_owners.get(),
        maximum_discovery_bytes: automatic.maximum_discovery_bytes.get(),
        maximum_retained_numeric_bytes: share,
    })
}
