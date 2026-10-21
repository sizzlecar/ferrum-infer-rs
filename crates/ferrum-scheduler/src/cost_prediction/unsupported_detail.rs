//! Fixed-size subcounts; these partition the unsupported-domain parent bucket.
use super::{CostUnknownReason, CostUnsupportedDetail};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CostUnsupportedCounts {
    pub unspecified: u64,
    pub execution_composition: u64,
    pub invalid_work_projection: u64,
    pub execution_identity_mismatch: u64,
    pub feature_domain_mismatch: u64,
    pub outside_observed_support: u64,
    pub invalid_base_prediction: u64,
    pub unrepresentable_duration: u64,
    pub invalid_scenario_distribution: u64,
}

impl CostUnsupportedCounts {
    pub fn record(&mut self, reason: CostUnknownReason) {
        let Some(detail) = reason.unsupported_detail() else {
            return;
        };
        let count = match detail {
            CostUnsupportedDetail::Unspecified => &mut self.unspecified,
            CostUnsupportedDetail::ExecutionComposition => &mut self.execution_composition,
            CostUnsupportedDetail::InvalidWorkProjection => &mut self.invalid_work_projection,
            CostUnsupportedDetail::ExecutionIdentityMismatch => {
                &mut self.execution_identity_mismatch
            }
            CostUnsupportedDetail::FeatureDomainMismatch => &mut self.feature_domain_mismatch,
            CostUnsupportedDetail::OutsideObservedSupport => &mut self.outside_observed_support,
            CostUnsupportedDetail::InvalidBasePrediction => &mut self.invalid_base_prediction,
            CostUnsupportedDetail::UnrepresentableDuration => &mut self.unrepresentable_duration,
            CostUnsupportedDetail::InvalidScenarioDistribution => {
                &mut self.invalid_scenario_distribution
            }
        };
        *count = count.saturating_add(1);
    }

    pub fn merge(&mut self, other: Self) {
        self.unspecified = self.unspecified.saturating_add(other.unspecified);
        self.execution_composition = self
            .execution_composition
            .saturating_add(other.execution_composition);
        self.invalid_work_projection = self
            .invalid_work_projection
            .saturating_add(other.invalid_work_projection);
        self.execution_identity_mismatch = self
            .execution_identity_mismatch
            .saturating_add(other.execution_identity_mismatch);
        self.feature_domain_mismatch = self
            .feature_domain_mismatch
            .saturating_add(other.feature_domain_mismatch);
        self.outside_observed_support = self
            .outside_observed_support
            .saturating_add(other.outside_observed_support);
        self.invalid_base_prediction = self
            .invalid_base_prediction
            .saturating_add(other.invalid_base_prediction);
        self.unrepresentable_duration = self
            .unrepresentable_duration
            .saturating_add(other.unrepresentable_duration);
        self.invalid_scenario_distribution = self
            .invalid_scenario_distribution
            .saturating_add(other.invalid_scenario_distribution);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cost_prediction::JointUnsupportedCounts, joint_policy::Unsupported};

    #[test]
    fn detailed_rejections_preserve_the_original_coarse_histogram_partition() {
        let details = [
            CostUnsupportedDetail::ExecutionComposition,
            CostUnsupportedDetail::InvalidWorkProjection,
            CostUnsupportedDetail::ExecutionIdentityMismatch,
            CostUnsupportedDetail::FeatureDomainMismatch,
            CostUnsupportedDetail::OutsideObservedSupport,
            CostUnsupportedDetail::InvalidBasePrediction,
            CostUnsupportedDetail::UnrepresentableDuration,
            CostUnsupportedDetail::InvalidScenarioDistribution,
        ];
        let mut coarse = JointUnsupportedCounts::default();
        let mut detail_counts = CostUnsupportedCounts::default();
        for reason in std::iter::once(CostUnknownReason::UnsupportedDomain)
            .chain(details.map(CostUnknownReason::UnsupportedWithDetail))
        {
            assert_eq!(reason.coarse(), CostUnknownReason::UnsupportedDomain);
            coarse.record(Unsupported::Cost(reason));
            detail_counts.record(reason);
        }
        detail_counts.record(CostUnknownReason::Stale);
        assert_eq!(
            serde_json::to_value(coarse).unwrap()["cost_unsupported_domain"],
            9
        );
        assert_eq!(
            serde_json::to_value(detail_counts)
                .unwrap()
                .as_object()
                .unwrap()
                .values()
                .map(|value| value.as_u64().unwrap())
                .sum::<u64>(),
            9
        );
        assert_eq!(detail_counts.unspecified, 1);
        let mut merged = CostUnsupportedCounts {
            outside_observed_support: u64::MAX,
            ..Default::default()
        };
        merged.merge(detail_counts);
        assert_eq!(merged.outside_observed_support, u64::MAX);
        assert_eq!(merged.unspecified, 1);
        assert_eq!(merged.invalid_scenario_distribution, 1);
    }
}
