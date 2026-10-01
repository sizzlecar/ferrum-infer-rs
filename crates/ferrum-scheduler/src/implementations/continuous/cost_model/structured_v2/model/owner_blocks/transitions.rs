use super::*;

impl FittedStructuredModelV2 {
    pub fn fit_owner_blocks(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredOwnerPhaseContractV1,
        close: StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<Self> {
        Self::fit_owner_blocks_parts(fingerprint, settings, scope, contract, close, samples, None)
    }

    pub fn fit_owner_blocks_from_certificate(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredOwnerPhaseContractV1,
        close: StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
        certificate: super::super::super::nonnegative::FitCertificate,
    ) -> Result<Self> {
        if contract.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Self::fit_owner_blocks_parts(
            fingerprint,
            settings,
            scope,
            contract,
            close,
            samples,
            Some(certificate),
        )
    }

    fn fit_owner_blocks_parts(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredOwnerPhaseContractV1,
        close: StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
        certificate: Option<super::super::super::nonnegative::FitCertificate>,
    ) -> Result<Self> {
        scope.validate()?;
        contract.validate_close(
            &settings,
            &close,
            samples,
            StructuredPhaseV2::Fit,
            None,
            contract.input_target.as_ref(),
        )?;
        let input_target = contract
            .schedule
            .input_readiness
            .as_ref()
            .map(|_| OwnerInputTargetV1::from_samples(samples))
            .transpose()?;
        let frozen_at_ns = close.boundary.frozen_at_ns;
        let expires_at_ns = contract.expires_at_ns;
        let original_sample_validity = matches!(
            contract.schedule.prediction_validity,
            Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1)
        );
        // These are actual member counts, not guessed future populations.
        let source = StructuredSourceContractV2 {
            capture_identity: contract.capture_identity,
            protocol: contract.protocol,
            membership_rule: contract.membership_rule,
            cohort_manifest: contract.declaration_sha256,
            phase_members: [samples.len(), 0, 0],
        };
        source.complete_population(samples, StructuredPhaseV2::Fit)?;
        let population = Population {
            contract,
            closes: [Some(close), None, None],
            fit_signature: [0; 32],
            calibrated_signature: [0; 32],
            input_target,
        };
        let mut model = Self::fit_parts(
            fingerprint,
            settings,
            scope,
            source,
            samples,
            frozen_at_ns,
            None,
            Some(population),
            certificate,
        )?;
        // Every phase's validate_close still checks the original collection
        // deadline. Only an explicitly bound new policy separates that work
        // limit from the expiry already derived from original member ages.
        if !original_sample_validity {
            model.state.expires_at_ns = model.state.expires_at_ns.min(expires_at_ns);
        }
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-owner-block-fit.v1\0");
        // All original mathematics, fingerprint, scope and sample-age facts.
        digest.update(model.legacy_parameters_signature());
        let population = model.owner_blocks.as_ref().unwrap();
        population.contract.bind(&mut digest);
        digest.update(population.closes[0].as_ref().unwrap().signature());
        model.owner_blocks.as_mut().unwrap().fit_signature = digest.finalize().into();
        Ok(model)
    }

    /// Consumes this phase state. A failed mathematical transition cannot be
    /// retried by adding a later block to the same fitted candidate.
    pub fn calibrate_owner_blocks(
        mut self,
        close: StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<CalibratedStructuredModelV2> {
        if self
            .owner_blocks
            .as_ref()
            .is_some_and(|p| p.contract.schedule.phase_support.is_some())
        {
            for sample in samples {
                if self.service_input_membership(&sample.input)?
                    != StructuredServiceInputMembershipV1::Eligible
                {
                    return Err(StructuredUnknown::UnidentifiedDirection);
                }
            }
        }
        let population = self
            .owner_blocks
            .as_mut()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        population.contract.validate_close(
            &self.settings,
            &close,
            samples,
            StructuredPhaseV2::Residual,
            Some((
                population.closes[0].as_ref().unwrap(),
                population.fit_signature,
            )),
            population.input_target.as_ref(),
        )?;
        let frozen_at_ns = close.boundary.frozen_at_ns;
        population.closes[1] = Some(close);
        self.source.phase_members[1] = samples.len();
        let mut calibrated = self.calibrate_parts(samples, frozen_at_ns)?;
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-owner-block-calibrated.v1\0");
        digest.update(calibrated.legacy_parameters_signature());
        digest.update(
            calibrated.fitted.owner_blocks.as_ref().unwrap().closes[1]
                .as_ref()
                .unwrap()
                .signature(),
        );
        calibrated
            .fitted
            .owner_blocks
            .as_mut()
            .unwrap()
            .calibrated_signature = digest.finalize().into();
        Ok(calibrated)
    }
}

impl CalibratedStructuredModelV2 {
    pub fn qualify_owner_blocks(
        mut self,
        close: StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<QualifiedStructuredModelV2> {
        if self
            .fitted
            .owner_blocks
            .as_ref()
            .is_some_and(|p| p.contract.schedule.phase_support.is_some())
        {
            for sample in samples {
                if self.service_input_membership(&sample.input)?
                    != StructuredServiceInputMembershipV1::Eligible
                {
                    return Err(StructuredUnknown::QualificationCoverage);
                }
            }
        }
        let population = self
            .fitted
            .owner_blocks
            .as_mut()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        population.contract.validate_close(
            &self.fitted.settings,
            &close,
            samples,
            StructuredPhaseV2::Qualification,
            Some((
                population.closes[1].as_ref().unwrap(),
                population.calibrated_signature,
            )),
            population.input_target.as_ref(),
        )?;
        let frozen_at_ns = close.boundary.frozen_at_ns;
        population.closes[2] = Some(close);
        self.fitted.source.phase_members[2] = samples.len();
        self.qualify_parts(samples, frozen_at_ns)
    }
}

impl QualifiedStructuredModelV2 {
    pub fn owner_block_contract(&self) -> Option<&StructuredOwnerPhaseContractV1> {
        self.calibrated
            .fitted
            .owner_blocks
            .as_ref()
            .map(|v| &v.contract)
    }

    pub fn owner_block_phase_signatures(&self) -> Option<[[u8; 32]; 3]> {
        self.calibrated.fitted.owner_blocks.as_ref().map(|v| {
            [
                v.fit_signature,
                v.calibrated_signature,
                self.parameters_signature(),
            ]
        })
    }
}
