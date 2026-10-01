use super::*;

/// Explicit numerical population semantics shared by run and serve. The
/// candidate never changes the independent reference or execution protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloAutomaticCalibrationNumericalStrategyV1 {
    #[default]
    IdentifiedEnvelopeV2,
    /// One identified Fit, one input-stopped joint-cell Residual bank, and one
    /// later independent Qualification boundary. Unqualified cells stay Unknown
    /// for this source; a new source is required to learn another bank.
    SameSourceJointCellsV1,
    /// Preserve the identified Fit and one global independent Residual margin;
    /// every Qualification member must pass. Physical/algorithm and branch
    /// gates still apply. This empirical point prediction permits magnitude
    /// extrapolation within that declared domain; it is not a coefficient-set
    /// upper bound. Automatic owner blocks use V3 geometry with a frozen Fit
    /// target for the input-only Residual/Qualification stopping rule.
    IdentifiedFitGlobalResidualV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_cell_strategy_is_explicit_and_rejects_fixed_windows() {
        let mut value = SloAutomaticCalibrationSettingsV1::default();
        assert_eq!(
            value.numerical_strategy,
            SloAutomaticCalibrationNumericalStrategyV1::IdentifiedEnvelopeV2
        );
        value.numerical_strategy =
            SloAutomaticCalibrationNumericalStrategyV1::SameSourceJointCellsV1;
        assert!(value.validate().is_ok());
        value.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1;
        assert!(value.validate().is_err());
    }

    #[test]
    fn global_residual_strategy_roundtrips_without_changing_defaults_or_silently_replacing_geometry(
    ) {
        let default = SloAutomaticCalibrationSettingsV1::default();
        assert_eq!(
            default.numerical_strategy,
            SloAutomaticCalibrationNumericalStrategyV1::IdentifiedEnvelopeV2
        );
        let mut value = default.clone();
        value.numerical_strategy =
            SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1;
        assert!(value.validate().is_ok());
        let encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(
            encoded["numerical_strategy"],
            "identified_fit_global_residual_v1"
        );
        assert_eq!(
            serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(encoded).unwrap(),
            value
        );
        value.population_schedule =
            SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2;
        assert!(value.validate().is_ok());
        value.population_schedule = SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1;
        assert!(value.validate().unwrap_err().contains("owner blocks"));
        value.population_schedule = default.population_schedule;
        for unsupported in [
            SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {},
            SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV1 {
                maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
                maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
            },
            SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV2 {
                maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
                maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
            },
        ] {
            value.input_readiness = unsupported;
            assert!(value
                .validate()
                .unwrap_err()
                .contains("work_axes_and_branches_v3"));
            let mut historical = value.clone();
            historical.numerical_strategy = default.numerical_strategy;
            assert!(
                historical.validate().is_ok(),
                "historical strategy combinations keep their meaning"
            );
        }
    }
}
