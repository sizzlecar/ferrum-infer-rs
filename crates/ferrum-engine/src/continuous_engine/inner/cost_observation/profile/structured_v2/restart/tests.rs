use super::*;
use crate::continuous_engine::inner::cost_observation::selected_feedback::{
    Comparison, FeedbackKind, FeedbackObservation, Monitor, RestartFeedbackAllowance,
};
use crate::continuous_engine::inner::cost_observation::structured_epoch;
use std::num::{NonZeroU64, NonZeroUsize};

impl EngineCostSnapshot {
    /// Called only by the real source8 CPU driver after original file replay.
    pub(in crate::continuous_engine::inner) fn check_restart_feedback_handoff_for_test(
        &self,
        children: Vec<ImportedStructuredModelV2>,
        workload_domain: &ferrum_interfaces::execution_cost::CostWorkloadDomainV1,
        domain: &CostMonotonicDomainV1,
        now: u64,
        destination: &Path,
    ) {
        let max_age = children.iter().map(|c| c.runtime_limits().1).min().unwrap();
        let max_wave = children.iter().map(|c| c.runtime_limits().0).min().unwrap();
        let policy = SloSelectedFeedbackSettingsV1 {
            window_samples: NonZeroUsize::new(2).unwrap(),
            minimum_underestimates: NonZeroUsize::new(2).unwrap(),
            minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
            trigger_excess_ns: NonZeroU64::new(2).unwrap(),
            correction_padding_ns: 5,
            maximum_family_margin_ns: NonZeroU64::new(max_wave.min(100)).unwrap(),
            maximum_consumption_lag_ns: NonZeroU64::new(max_age.min(50)).unwrap(),
            maximum_uncomparable_observations: 0,
            maximum_failed_or_partial: 0,
            maximum_queue_drops: 0,
            maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
        };
        let replayed = Self::live_catalog_with_domain(
            children.clone(),
            self.fingerprint.clone(),
            structured_epoch::View::initial(),
            now,
            Some(workload_domain),
        )
        .unwrap();
        let proof_bytes = self
            .restart_verification_budget(&replayed, &policy)
            .unwrap();
        assert!(self
            .verify_restart_replay(&replayed, &policy, domain, now, proof_bytes - 1)
            .is_err());
        let proof = self
            .verify_restart_replay(&replayed, &policy, domain, now, proof_bytes)
            .unwrap();
        assert_ne!(
            proof.original_binding().profile_sha256,
            proof.persisted_binding().profile_sha256
        );
        let mut monitor = Monitor::open_bound(
            &policy,
            &ferrum_types::SloSelectedFeedbackStorageV1::MemoryOnly,
            proof.original_binding().clone(),
            proof.scopes().len(),
            FeedbackKind::StructuredV2,
            Some(proof.scopes().clone()),
        )
        .unwrap();
        let scope = proof.scopes()[0];
        for ordinal in 1..=2 {
            monitor.observe_classified(
                ordinal,
                FeedbackObservation::Compared(Comparison {
                    family: scope,
                    base_planning_ns: 100,
                    actual_ns: 120,
                    observed_at_ns: now,
                    consumed_at_ns: now,
                }),
            );
        }
        monitor.publish(Some(0)).unwrap();
        assert_eq!(monitor.current_view().margin(&scope), 25);
        let old = monitor.audit();
        let budget = monitor.restart_budget(destination).unwrap();
        monitor
            .finish_for_restart(
                &proof,
                destination,
                2,
                RestartFeedbackAllowance {
                    transient_bytes: budget.maximum_transient_bytes,
                    disk_bytes: budget.maximum_disk_bytes,
                },
                || Some(now),
            )
            .unwrap();
        let budget = Monitor::restart_resume_budget(&policy, destination).unwrap();
        let mut resumed = Monitor::resume_for_restart(
            &policy,
            destination,
            proof.persisted_binding().clone(),
            proof.scopes().clone(),
            RestartFeedbackAllowance {
                transient_bytes: budget.maximum_transient_bytes,
                disk_bytes: budget.maximum_disk_bytes,
            },
        )
        .unwrap();
        assert_eq!(resumed.audit().compared, old.compared);
        assert_eq!(resumed.audit().corrections, old.corrections);
        assert_eq!(resumed.current_view().margin(&scope), 25);
        assert!(!resumed.current_view().current());
        assert!(
            proof
                .validate_freshness(now.checked_add(max_age).unwrap().checked_add(1).unwrap())
                .is_err(),
            "handoff does not renew original sample TTL"
        );
        let other = CostMonotonicDomainV1::new_macos_continuous([5; 16]).unwrap();
        assert!(self
            .verify_restart_replay(&replayed, &policy, &other, now, proof_bytes)
            .is_err());
        // A same-clock, same-source but incomplete replay catalog is not an
        // equivalent selected inventory; source provenance cannot fill it in.
        if children.len() > 1 {
            let subset = Self::live_catalog_with_domain(
                vec![children[0].clone()],
                self.fingerprint.clone(),
                structured_epoch::View::initial(),
                now,
                Some(workload_domain),
            )
            .unwrap();
            assert!(self
                .verify_restart_replay(&subset, &policy, domain, now, proof_bytes)
                .is_err());
        }
        resumed.finish();
    }
}
