//! Test-only reconstruction of the previous source7 Fit binding. The supported
//! fixture has no optional phase policies: every byte below is the historical
//! binding transcript, with the prospective-work semantic tag absent.
use super::*;

impl FittedStructuredModelV2 {
    pub(crate) fn previous_completion_binding_for_test(&self) -> [u8; 32] {
        let physical = self.numerical.physical().unwrap();
        let c = &physical.contract;
        assert_eq!(
            c.planning_estimator,
            NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1
        );
        assert!(physical.phase_support.is_none());
        assert!(c.algorithm_universe.is_none());
        assert_eq!(
            envelope::axes(&self.exemplar).unwrap(),
            envelope::model_axes(&self.exemplar, c.planning_estimator).unwrap()
        );
        let mut h = Sha256::new();
        h.update(b"ferrum.structured-fit-state.v2\0");
        h.update(MODEL_REVISION_V2.as_bytes());
        for value in [
            self.fingerprint.model_weights,
            self.fingerprint.numerical_policy,
            self.fingerprint.device_runtime,
            self.fingerprint.execution_config,
            self.source.capture_identity,
            self.source.protocol,
            self.source.membership_rule,
            self.source.cohort_manifest,
            self.domain_signature,
        ] {
            h.update(value);
        }
        for value in [
            self.settings.min_phase_samples as u64,
            self.settings.min_fit_redundancy as u64,
            self.settings.max_phase_samples as u64,
            self.settings.max_axes as u64,
            self.settings.max_rank as u64,
            self.settings.max_wave_ns,
            self.settings.max_sample_age_ns,
            self.settings.static_margin_ns,
            self.source.phase_members[0] as u64,
            self.source.phase_members[1] as u64,
            self.source.phase_members[2] as u64,
            self.fit_samples as u64,
            self.frozen_at_ns,
        ] {
            h.update(value.to_le_bytes());
        }
        if !self.settings.learned_drift.is_disabled() {
            h.update(self.settings.learned_drift.signature());
        }
        self.scope.bind_parameters(&mut h);
        self.state.bind_parameters(&mut h);
        previous_contract(c, &mut h);
        physical.fit.bind_parameters(&mut h);
        h.update(physical.fit_error_floor_ns.to_le_bytes());
        physical.coverage[0].as_ref().unwrap().bind(&mut h);
        let previous_fit: [u8; 32] = h.finalize().into();

        let population = self.owner_blocks.as_ref().unwrap();
        let p = &population.contract;
        assert!(p.schedule.input_readiness.is_none());
        assert!(p.schedule.phase_support.is_none());
        assert!(p.schedule.algorithm_universe.is_none());
        assert!(p.schedule.opening_frontier.is_none());
        assert!(p.schedule.prediction_validity.is_none());
        assert!(p.input_target.is_none());
        let mut h = Sha256::new();
        h.update(b"ferrum.structured-owner-block-fit.v1\0");
        h.update(previous_fit);
        h.update(b"ferrum.structured-owner-block-contract.v1\0");
        for value in [
            p.capture_identity,
            p.protocol,
            p.membership_rule,
            p.declaration_sha256,
        ] {
            h.update(value);
        }
        for value in [
            p.owner_attempt_id,
            p.schedule.block_offered as u64,
            p.discovery_block,
            p.discovery_offered_cutoff,
            p.discovery_fifo_cutoff,
            p.discovery_closed_at_ns,
            p.expires_at_ns,
        ] {
            h.update(value.to_le_bytes());
        }
        for values in [
            p.schedule.phase_min_offered,
            p.schedule.min_members,
            p.schedule.maximum_phase_members,
        ] {
            for value in values {
                h.update((value as u64).to_le_bytes());
            }
        }
        assert_eq!(
            p.domain_policy,
            StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
        );
        h.update([2]);
        previous_contract(p.nonnegative_envelope.as_ref().unwrap(), &mut h);
        h.update(population.closes[0].as_ref().unwrap().signature());
        h.finalize().into()
    }
}
fn previous_contract(c: &NonNegativeEnvelopeContractV1, h: &mut Sha256) {
    assert!(c.algorithm_universe.is_none());
    h.update(b"ferrum.nonnegative-physical-envelope.contract.v1\0");
    h.update(b"ferrum.identified-fit.global-residual.v1\0");
    h.update(c.workload_domain.sha256());
    if !c.template_policy.is_ordered() {
        h.update(b"ferrum.nonnegative-template-policy.installed-algorithm-set.v1\0");
    }
    if c.population_policy == StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1 {
        h.update(b"ferrum.homogeneous-ordinary-decode-population.v1\0");
    }
    h.update(c.settings.maximum_coordinate_visits.to_le_bytes());
    h.update((c.settings.maximum_query_certificates as u64).to_le_bytes());
    h.update([0]);
}
