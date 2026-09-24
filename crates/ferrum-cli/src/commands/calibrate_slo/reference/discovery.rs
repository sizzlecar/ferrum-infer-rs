use super::*;

pub(in crate::commands::calibrate_slo) struct DiscoverySet {
    config: ReferenceConfig,
    conditions: serde_json::Value,
    conditions_sha256: [u8; 32],
    preprocessing: [u8; 32],
    curves: Vec<Option<Vec<CalibrationReferenceDiscoverySample>>>,
    decode: Option<CalibrationReferenceDiscoverySample>,
    retained: usize,
}

pub(in crate::commands::calibrate_slo) struct FrozenReference {
    config: ReferenceConfig,
    collector: CalibrationReferenceCollector,
    frozen_plan: files::FrozenPlanReceipt,
}

#[derive(Debug, Serialize)]
pub(in crate::commands::calibrate_slo) struct ReferenceReceipt {
    pub frozen_plan: files::FrozenPlanReceipt,
    pub artifact: CalibrationReferenceArtifact,
    pub scope: &'static str,
}

impl DiscoverySet {
    pub(in crate::commands::calibrate_slo) fn observer(
        &self,
        target: DiscoveryTarget,
    ) -> Result<DiscoveryObserver> {
        let remaining = self
            .config
            .limits
            .max_samples
            .get()
            .checked_sub(self.retained)
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| invalid("discovery retention capacity exhausted"))?;
        let mut config = self.config.clone();
        config.limits.max_samples = remaining;
        DiscoveryObserver::new(&config, target)
    }
    pub(in crate::commands::calibrate_slo) fn new(
        config: ReferenceConfig,
        manifest: &manifest::Manifest,
        inputs: &inputs::PreparedInputs,
        session: &CalibrationSession,
    ) -> Result<Self> {
        config.validate(manifest, inputs)?;
        let conditions = serde_json::json!({
            "schema_version":1,
            "effective_engine":session.configuration(),
            "prepared_inputs":inputs.provenance(),
            "protocol":manifest.protocol,
            "reference_declaration":config,
        });
        let bytes = files::bounded_json(&conditions, config.limits.max_file_bytes.get())?;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.cli.reference-conditions.v1\0");
        hash.update(bytes);
        Ok(Self {
            curves: (0..config.curve_prompt_indices.len())
                .map(|_| None)
                .collect(),
            config,
            conditions,
            conditions_sha256: hash.finalize().into(),
            preprocessing: manifest.input_preprocessing_sha256,
            decode: None,
            retained: 0,
        })
    }

    /// Call only after the discovery request's normal terminal and output
    /// completion. No successful target can rescue a failed full request.
    pub(in crate::commands::calibrate_slo) fn accept(
        &mut self,
        observer: DiscoveryObserver,
    ) -> Result<()> {
        let mut observer = observer.finish()?;
        let count = observer.samples.len();
        let next = self
            .retained
            .checked_add(count)
            .filter(|n| *n <= self.config.limits.max_samples.get())
            .ok_or_else(|| invalid("all discovery receipts exceed the sample bound"))?;
        match observer.target {
            DiscoveryTarget::Prefill { curve } => {
                let slot = self
                    .curves
                    .get_mut(curve)
                    .ok_or_else(|| invalid("unknown discovery curve"))?;
                if slot.is_some() || observer.samples.is_empty() {
                    return Err(invalid("duplicate or empty discovery curve"));
                }
                *slot = Some(observer.samples);
            }
            DiscoveryTarget::Decode => {
                if self.decode.is_some() || observer.samples.len() != 1 {
                    return Err(invalid("duplicate or missing decode discovery"));
                }
                self.decode = observer.samples.pop();
            }
        }
        self.retained = next;
        Ok(())
    }

    pub(in crate::commands::calibrate_slo) async fn freeze(
        self,
        session: &mut CalibrationSession,
    ) -> Result<FrozenReference> {
        let decode = self
            .decode
            .ok_or_else(|| invalid("decode discovery is missing"))?;
        if decode.host_features().state.generated_tokens_before
            != u64::from(self.config.decode_unit.generated_before.get())
        {
            return Err(invalid(
                "decode discovery is not the declared actual generation",
            ));
        }
        let decode_input = *decode.request_evidence();
        let decode_shape = decode.shape();
        let decode_host = decode.host_features();
        let mut curves = Vec::with_capacity(self.curves.len());
        let mut discovery = Vec::with_capacity(self.retained);
        let mut host = None;
        let mut lengths = std::collections::BTreeSet::new();
        let mut segments = 0usize;
        for partition in self.curves {
            let partition = partition.ok_or_else(|| invalid("curve discovery is missing"))?;
            let first = partition
                .first()
                .ok_or_else(|| invalid("empty curve discovery"))?;
            let input = *first.request_evidence();
            let total = input_length(input)?;
            if !lengths.insert(total) {
                return Err(invalid("V1 curves require distinct actual token lengths; inputs cannot be deduplicated after discovery"));
            }
            for sample in &partition {
                if *sample.request_evidence() != input
                    || host.is_some_and(|expected| expected != sample.host_features())
                {
                    return Err(invalid("V1 requires one actual fixed prefill host policy/input per curve; declared reference output policy cannot override measured host incompatibility"));
                }
                host = Some(sample.host_features());
            }
            segments = segments
                .checked_add(partition.len())
                .ok_or_else(|| invalid("reference segment overflow"))?;
            curves.push(CalibrationReferenceCurve {
                total_prompt_tokens: total,
                input_tokens_sha256: input.original_input_tokens_sha256,
                partition: partition.iter().map(|sample| sample.shape()).collect(),
            });
            discovery.extend(partition);
        }
        retention_bound(
            &self.config,
            segments,
            curves.len(),
            decode_input.original_input_tokens,
        )?;
        discovery.push(decode);
        let plan = CalibrationReferencePlan {
            piecewise: self.config.piecewise.clone(),
            reference_revision: self.config.revision,
            protocol: ReferenceProtocolV1 {
                granule_tokens: self.config.granule_tokens,
                repetitions: self.config.repetitions,
                estimator: ReferenceEstimator::UpperMedianWallV1,
                input_preprocessing_sha256: self.preprocessing,
                measurement_conditions_sha256: self.conditions_sha256,
                prefill_host: host.ok_or_else(|| invalid("prefill discovery is missing"))?,
                decode_host,
                decode_shape,
            },
            decode_input_tokens: input_length(decode_input)?,
            decode_input_tokens_sha256: decode_input.original_input_tokens_sha256,
            curves,
            limits: self.config.limits.clone(),
        };
        let ordinals: Vec<_> = discovery
            .iter()
            .map(|sample| sample.accepted_ordinal())
            .collect();
        let collector = session
            .freeze_reference_plan(plan.clone(), discovery)
            .await?;
        let document = FrozenPlanDocument {
            schema_version: if plan.piecewise.is_some() { 2 } else { 1 },
            artifact_type: "ferrum.calibration-reference-frozen-plan",
            reference_plan: &plan,
            measurement_conditions: &self.conditions,
            plan_sha256: collector.plan_sha256(),
            frozen_accepted_ordinal: collector.frozen_accepted_ordinal(),
            discovery_accepted_ordinals: &ordinals,
        };
        // Persist and sync before giving the driver any fresh-trial handle.
        let frozen_plan = files::publish(
            &self.config.frozen_plan_path,
            &document,
            self.config.limits.max_file_bytes.get(),
        )?;
        Ok(FrozenReference {
            config: self.config,
            collector,
            frozen_plan,
        })
    }
}

#[derive(Serialize)]
struct FrozenPlanDocument<'a> {
    schema_version: u32,
    artifact_type: &'static str,
    reference_plan: &'a CalibrationReferencePlan,
    measurement_conditions: &'a serde_json::Value,
    plan_sha256: [u8; 32],
    frozen_accepted_ordinal: u64,
    discovery_accepted_ordinals: &'a [u64],
}

impl FrozenReference {
    pub(in crate::commands::calibrate_slo) fn frozen_plan(&self) -> &files::FrozenPlanReceipt {
        &self.frozen_plan
    }
    pub(in crate::commands::calibrate_slo) fn trial(
        &mut self,
        key: CalibrationReferenceTrial,
    ) -> TrialObserver<'_> {
        TrialObserver::new(&mut self.collector, key)
    }
    /// Use the exported training cut before independent heldout. The engine
    /// rechecks all selected original records/commits against the cut's hash.
    pub(in crate::commands::calibrate_slo) fn finish(
        self,
        cut: &CalibrationProfileArtifact,
    ) -> Result<ReferenceReceipt> {
        files::verify(&self.frozen_plan)?;
        let artifact = self.collector.finish(cut, &self.config.artifact_path)?;
        Ok(ReferenceReceipt { frozen_plan: self.frozen_plan, artifact,
            scope: "Fixed actual singleton route and original-input endpoints, measured UpperMedianWallV1; not a latency bound, cost-prediction coverage proof, or assertion that an opaque route hash proves greedy sampling." })
    }
}

fn input_length(input: CalibrationRequestEvidence) -> Result<NonZeroU32> {
    u32::try_from(input.original_input_tokens)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| invalid("original tokenized input is empty or too long"))
}

/// Include decode preparation and prior generations, even though only the
/// chosen decode is the unit. Those original receipts must survive source join.
pub(super) fn retention_bound(
    config: &ReferenceConfig,
    segments: usize,
    curves: usize,
    decode_input: usize,
) -> Result<()> {
    let decode_preparation = match &config.piecewise {
        Some(spec) => spec
            .segment_count(
                u32::try_from(decode_input).map_err(|_| invalid("decode input overflow"))?,
            )
            .map_err(|e| invalid(e.to_string()))?,
        None => decode_input.div_ceil(config.granule_tokens.get() as usize),
    };
    let per_repetition = segments
        .checked_add(decode_preparation)
        .and_then(|n| n.checked_add(config.decode_unit.generated_before.get() as usize));
    let retained = per_repetition
        .and_then(|n| n.checked_mul(config.repetitions.get()))
        .and_then(|n| n.checked_add(segments))
        .and_then(|n| n.checked_add(1));
    if retained.is_none_or(|n| n > config.limits.max_samples.get())
        || segments
            .checked_add(curves)
            .is_none_or(|n| n > config.limits.max_points.get())
    {
        return Err(invalid(
            "full preparation + selected targets exceed reference retention/point bounds",
        ));
    }
    Ok(())
}
