use super::*;

/// Slow-path assembly from actual segmented trials. No candidate cost or
/// online model version is accepted as a reference input.
pub struct ReferenceCalibrationBuilder {
    artifact: ReferenceCalibrationV1,
    limits: SloPrefillReferenceLimits,
    piecewise: Option<PiecewiseReferenceSpec>,
}
impl ReferenceCalibrationBuilder {
    pub fn new(
        revision: NonZeroU64,
        fingerprint: ProfileFingerprint,
        protocol: ReferenceProtocolV1,
        generated_unix_ns: NonZeroU64,
        limits: SloPrefillReferenceLimits,
    ) -> Result<Self, ReferenceError> {
        limits
            .validate()
            .map_err(|_| ReferenceError::Limit("configuration"))?;
        validate_protocol(&protocol)?;
        Ok(Self {
            artifact: ReferenceCalibrationV1 {
                schema_version: PREFILL_REFERENCE_SCHEMA_V1,
                reference_revision: revision,
                fingerprint,
                generated_unix_ns,
                protocol,
                decode_samples: Vec::new(),
                curves: Vec::new(),
            },
            limits,
            piecewise: None,
        })
    }
    pub fn with_piecewise(mut self, spec: PiecewiseReferenceSpec) -> Result<Self, ReferenceError> {
        spec.validate()?;
        self.piecewise = Some(spec);
        Ok(self)
    }
    pub fn finish_bytes(self) -> Result<Vec<u8>, ReferenceError> {
        if let Some(spec) = &self.piecewise {
            compile_partitioned(
                &self.artifact,
                &self.artifact.fingerprint.clone().into(),
                spec.protocol_sha256(&self.artifact.protocol)?,
                &self.limits,
                [0; 32],
                Some(spec),
            )?;
            let artifact = ReferenceCalibrationV2::from_evidence(self.artifact, spec.clone());
            let bytes = serde_json::to_vec(&artifact)?;
            if bytes.len() > self.limits.max_file_bytes.get() {
                return Err(ReferenceError::Limit("file bytes"));
            }
            Ok(bytes)
        } else {
            Ok(serde_json::to_vec(&self.finish()?)?)
        }
    }
    pub fn set_decode_samples(
        &mut self,
        samples: Vec<ReferenceObservedSample>,
    ) -> Result<(), ReferenceError> {
        if !self.artifact.decode_samples.is_empty()
            || samples.len() != self.artifact.protocol.repetitions.get()
        {
            return Err(ReferenceError::Evidence(
                "duplicate or incomplete decode trials",
            ));
        }
        if samples.len() > self.limits.max_samples.get() {
            return Err(ReferenceError::Limit("samples"));
        }
        self.artifact.decode_samples = samples;
        Ok(())
    }
    pub fn add_curve(&mut self, curve: ReferenceCurveInput) -> Result<(), ReferenceError> {
        if self
            .artifact
            .curves
            .iter()
            .any(|prior| prior.total_prompt_tokens == curve.total_prompt_tokens)
        {
            return Err(ReferenceError::Evidence("duplicate prompt length"));
        }
        // Check aggregate budgets before retaining another potentially large trial.
        counts(
            self.artifact.curves.iter().chain(std::iter::once(&curve)),
            self.artifact.decode_samples.len(),
            &self.limits,
        )?;
        self.artifact.curves.push(curve);
        Ok(())
    }
    pub fn finish(self) -> Result<ReferenceCalibrationV1, ReferenceError> {
        if self.piecewise.is_some() {
            return Err(ReferenceError::Evidence(
                "V2 requires the versioned artifact output",
            ));
        }
        compile(
            &self.artifact,
            &self.artifact.fingerprint.clone().into(),
            self.artifact.protocol.sha256()?,
            &self.limits,
            [0; 32],
        )?;
        let bytes = serde_json::to_vec(&self.artifact)?;
        if bytes.len() > self.limits.max_file_bytes.get() {
            return Err(ReferenceError::Limit("file bytes"));
        }
        Ok(self.artifact)
    }
}

fn counts<'a>(
    curves: impl Iterator<Item = &'a ReferenceCurveInput>,
    decode: usize,
    limits: &SloPrefillReferenceLimits,
) -> Result<(usize, usize), ReferenceError> {
    let (mut count, mut points, mut samples) = (0_usize, 0_usize, decode);
    for curve in curves {
        count += 1;
        let n = curve
            .partition
            .len()
            .checked_add(1)
            .ok_or(ReferenceError::Overflow)?;
        points = points.checked_add(n).ok_or(ReferenceError::Overflow)?;
        if n < 2
            || n > limits.max_points_per_curve.get()
            || points > limits.max_points.get()
            || count > limits.max_curves.get()
            || curve.trials.len() > PREFILL_REFERENCE_MAX_REPETITIONS
        {
            return Err(ReferenceError::Limit("curve or point budget"));
        }
        for trial in &curve.trials {
            if trial.samples.len() != curve.partition.len() {
                return Err(ReferenceError::Evidence("incomplete trial partition"));
            }
            samples = samples
                .checked_add(trial.samples.len())
                .ok_or(ReferenceError::Overflow)?;
        }
        if samples > limits.max_samples.get() {
            return Err(ReferenceError::Limit("sample budget"));
        }
    }
    if decode > limits.max_samples.get() {
        return Err(ReferenceError::Limit("decode sample budget"));
    }
    Ok((points, samples))
}

pub(super) fn compile(
    artifact: &ReferenceCalibrationV1,
    fingerprint: &ExecutionFingerprint,
    protocol_sha256: [u8; 32],
    limits: &SloPrefillReferenceLimits,
    artifact_sha256: [u8; 32],
) -> Result<LoadedPrefillReference, ReferenceError> {
    compile_partitioned(
        artifact,
        fingerprint,
        protocol_sha256,
        limits,
        artifact_sha256,
        None,
    )
}

pub(super) fn compile_partitioned(
    artifact: &ReferenceCalibrationV1,
    fingerprint: &ExecutionFingerprint,
    protocol_sha256: [u8; 32],
    limits: &SloPrefillReferenceLimits,
    artifact_sha256: [u8; 32],
    piecewise: Option<&PiecewiseReferenceSpec>,
) -> Result<LoadedPrefillReference, ReferenceError> {
    limits
        .validate()
        .map_err(|_| ReferenceError::Limit("configuration"))?;
    if artifact.schema_version != PREFILL_REFERENCE_SCHEMA_V1 {
        return Err(ReferenceError::Schema(artifact.schema_version));
    }
    if artifact.fingerprint != ProfileFingerprint::from(fingerprint)
        || match piecewise {
            Some(spec) => spec.protocol_sha256(&artifact.protocol)?,
            None => artifact.protocol.sha256()?,
        } != protocol_sha256
    {
        return Err(ReferenceError::Incompatible);
    }
    let protocol = &artifact.protocol;
    validate_protocol(protocol)?;
    if let Some(spec) = piecewise {
        spec.validate()?;
    }
    if artifact.curves.is_empty() || artifact.decode_samples.len() != protocol.repetitions.get() {
        return Err(ReferenceError::Evidence("missing calibration trials"));
    }
    let (points, samples) = counts(
        artifact.curves.iter(),
        artifact.decode_samples.len(),
        limits,
    )?;
    let mut sources = HashSet::new();
    sources
        .try_reserve(samples)
        .map_err(|_| ReferenceError::Limit("source allocation"))?;
    let mut decode_times = Vec::with_capacity(artifact.decode_samples.len());
    let decode_context = protocol.decode_shape.exact.decode_kv_tokens[0];
    let decode_generated = u32::try_from(protocol.decode_host.state.generated_tokens_before)
        .map_err(|_| ReferenceError::Overflow)?;
    for sample in &artifact.decode_samples {
        validate_sample(
            sample,
            &protocol.decode_shape,
            artifact.generated_unix_ns.get(),
            &mut sources,
        )?;
        let commit = &sample.commit;
        if commit.origin != ReferenceStateOrigin::PreparedDecode
            || commit.previous_record.is_some()
            || commit.prefix_before != decode_context
            || commit.prefix_after
                != decode_context
                    .checked_add(1)
                    .ok_or(ReferenceError::Overflow)?
            || commit.generated_before != decode_generated
            || commit.generated_after
                != decode_generated
                    .checked_add(1)
                    .ok_or(ReferenceError::Overflow)?
        {
            return Err(ReferenceError::Evidence(
                "decode unit lacks exact singleton commit",
            ));
        }
        decode_times.push(sample.observation.timing.wall_total_ns);
    }
    let tau_ref_ns = upper_median(&mut decode_times)?;
    let mut curves = BTreeMap::new();
    for curve in &artifact.curves {
        let total = curve.total_prompt_tokens.get();
        if curves.contains_key(&total) || curve.trials.len() != protocol.repetitions.get() {
            return Err(ReferenceError::Evidence(
                "duplicate length or incomplete repetitions",
            ));
        }
        let expected_segments = match piecewise {
            Some(spec) => spec.segment_count(total)? as u64,
            None => u64::from(total).div_ceil(u64::from(protocol.granule_tokens.get())),
        };
        if u64::try_from(curve.partition.len()).ok() != Some(expected_segments) {
            return Err(ReferenceError::Evidence(
                "missing declared partition endpoint",
            ));
        }
        let mut endpoints = Vec::with_capacity(curve.partition.len());
        let mut offset = 0;
        for shape in &curve.partition {
            let count = match piecewise {
                Some(spec) => spec.next_count(total, offset)?.get(),
                None => protocol.granule_tokens.get().min(total - offset),
            };
            let end = offset.checked_add(count).ok_or(ReferenceError::Overflow)?;
            validate_shape(
                protocol.graph_routes,
                shape,
                &protocol.prefill_host,
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens: total,
                },
            )?;
            endpoints.push(end);
            offset = end;
        }
        let mut segment_times =
            vec![Vec::with_capacity(protocol.repetitions.get()); curve.partition.len()];
        let input = curve.trials[0].input_tokens_sha256;
        if input == [0; 32] {
            return Err(ReferenceError::Evidence("missing input token digest"));
        }
        let mut owners = HashSet::new();
        for (index, trial) in curve.trials.iter().enumerate() {
            if trial.trial_index != index || trial.input_tokens_sha256 != input {
                return Err(ReferenceError::Evidence(
                    "trial order or fixed input changed",
                ));
            }
            let owner = trial.samples[0].commit.owner_incarnation;
            if !owners.insert(owner) {
                return Err(ReferenceError::Evidence(
                    "fresh trials reuse owner incarnation",
                ));
            }
            let mut previous: Option<&ReferenceObservedSample> = None;
            for (index, sample) in trial.samples.iter().enumerate() {
                validate_sample(
                    sample,
                    &curve.partition[index],
                    artifact.generated_unix_ns.get(),
                    &mut sources,
                )?;
                let commit = &sample.commit;
                let start = if index == 0 { 0 } else { endpoints[index - 1] };
                let end = endpoints[index];
                if commit.owner_incarnation != owner
                    || commit.prefix_before != start
                    || commit.prefix_after != end
                    || commit.generated_before != 0
                    || commit.generated_after != u32::from(end == total)
                {
                    return Err(ReferenceError::Evidence(
                        "prefill receipt does not commit exact reference work",
                    ));
                }
                match previous {
                    None if commit.origin == ReferenceStateOrigin::Fresh
                        && commit.previous_record.is_none() => {}
                    Some(prior)
                        if commit.origin == ReferenceStateOrigin::CommittedContinuation
                            && commit.previous_record == Some(prior.record)
                            && commit.work_generation > prior.commit.work_generation
                            && sample.observation.measured_unix_ns
                                >= prior.observation.measured_unix_ns => {}
                    _ => {
                        return Err(ReferenceError::Evidence(
                            "reference continuation chain is not fresh committed work",
                        ))
                    }
                }
                segment_times[index].push(sample.observation.timing.wall_total_ns);
                previous = Some(sample);
            }
        }
        let mut work = Vec::with_capacity(endpoints.len() + 1);
        work.push(ReferenceWorkPoint {
            prompt_tokens: 0,
            cumulative_work_ns: 0,
        });
        let mut cumulative = 0_u64;
        for (end, mut samples) in endpoints.into_iter().zip(segment_times) {
            cumulative = cumulative
                .checked_add(upper_median(&mut samples)?.get())
                .ok_or(ReferenceError::Overflow)?;
            work.push(ReferenceWorkPoint {
                prompt_tokens: end,
                cumulative_work_ns: cumulative,
            });
        }
        curves.insert(
            total,
            Arc::new(PrefillReferenceWork {
                evaluation: Default::default(),
                version: artifact.reference_revision.get(),
                points: work,
            }),
        );
    }
    let piecewise = piecewise
        .map(|spec| piecewise::PiecewiseDefinition::compile(spec, &curves).map(Arc::new))
        .transpose()?;
    Ok(LoadedPrefillReference {
        identity: ReferenceIdentity {
            revision: artifact.reference_revision,
            artifact_sha256,
            protocol_sha256,
        },
        fingerprint: fingerprint.clone(),
        protocol: artifact.protocol.clone(),
        tau_ref_ns,
        curves,
        piecewise,
        points,
        samples,
        generated_unix_ns: artifact.generated_unix_ns.get(),
        source_path: None,
    })
}

fn validate_protocol(protocol: &ReferenceProtocolV1) -> Result<(), ReferenceError> {
    use ferrum_interfaces::execution_cost::satisfied_completion_cost_signature;
    if protocol.repetitions.get() > PREFILL_REFERENCE_MAX_REPETITIONS
        || protocol.input_preprocessing_sha256 == [0; 32]
        || protocol.measurement_conditions_sha256 == [0; 32]
        || protocol.prefill_host.policy != protocol.decode_host.policy
        || protocol.prefill_host.state.generated_tokens_before != 0
        || protocol.decode_host.state.generated_tokens_before == 0
        || protocol.prefill_host.state.maximum_output_tokens
            != protocol.decode_host.state.maximum_output_tokens
        || [protocol.prefill_host, protocol.decode_host]
            .iter()
            .any(|host| {
                host.state.pending_decoded_utf8
                    || host.state.completion_state_signature
                        != satisfied_completion_cost_signature()
            })
    {
        return Err(ReferenceError::Evidence("unsupported reference protocol"));
    }
    let contexts = &protocol.decode_shape.exact.decode_kv_tokens;
    if contexts.len() != 1 || contexts[0] == 0 {
        return Err(ReferenceError::Evidence(
            "decode reference is not singleton",
        ));
    }
    validate_shape(
        protocol.graph_routes,
        &protocol.decode_shape,
        &protocol.decode_host,
        ActualRowWork::Decode {
            kv_tokens: contexts[0],
        },
    )
}

fn validate_shape(
    graph_routes: ReferenceGraphRoutes,
    shape: &ProfileWaveShapeV2,
    host: &HostCostFeaturesV1,
    work: ActualRowWork,
) -> Result<(), ReferenceError> {
    let exact = &shape.exact;
    if exact.path != ProfileExecutionPath::PlanRuntime
        || !graph_routes.accepts(exact.graph_state)
        || exact.order != ProfileBatchOrder::Ordered
        || exact.provider_signature == [0; 32]
        || exact.output_policy_signature == [0; 32]
        || exact.restore_bytes != 0
        || exact.maintenance_bytes != 0
        || exact.maintenance_units != 0
    {
        return Err(ReferenceError::Evidence("unsupported reference route"));
    }
    let output = match work {
        ActualRowWork::Decode { kv_tokens }
            if exact.kind == ProfileWaveKind::Decode
                && exact.decode_kv_tokens == [kv_tokens]
                && exact.prefill_chunks.is_empty() =>
        {
            CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0_f32.to_bits(),
            }
        }
        ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens,
        } if exact.kind == ProfileWaveKind::Prefill
            && exact.decode_kv_tokens.is_empty()
            && exact.prefill_chunks.len() == 1
            && exact.prefill_chunks[0].offset == offset
            && exact.prefill_chunks[0].count.get() == count
            && exact.prefill_chunks[0].total_prompt_tokens.get() == total_prompt_tokens =>
        {
            CostRowOutput::Prefill {
                final_logits: offset.checked_add(count) == Some(total_prompt_tokens),
            }
        }
        _ => {
            return Err(ReferenceError::Evidence(
                "reference work differs from declared partition",
            ))
        }
    };
    let features = shape
        .numeric_features
        .as_ref()
        .ok_or(ReferenceError::Evidence(
            "reference requires numeric v2 category evidence",
        ))?;
    features
        .validate(1)
        .map_err(|_| ReferenceError::Evidence("numeric row evidence"))?;
    let projected = project_host_cost_features(*host, work, output)
        .map_err(|_| ReferenceError::Evidence("fixed host state is invalid"))?;
    if features.rows != [projected] || features.output_policy_signature == [0; 32] {
        return Err(ReferenceError::Evidence("reference host work changed"));
    }
    Ok(())
}

fn validate_sample(
    sample: &ReferenceObservedSample,
    shape: &ProfileWaveShapeV2,
    generated_unix_ns: u64,
    sources: &mut HashSet<ReferenceRecordId>,
) -> Result<(), ReferenceError> {
    let observed = &sample.observation;
    if sample.record.source_sha256 == [0; 32]
        || sample.record.ordinal != observed.source_record
        || !sources.insert(sample.record)
        || observed.measured_unix_ns == 0
        || observed.measured_unix_ns > generated_unix_ns
        || observed.shape != *shape
        || observed.boundary != ProfileCostBoundary::PreparationToCommit
        || observed.outcome != (ProfileObservationOutcome::Completed {})
        || observed.timing.stages.restore.is_some()
        || observed.timing.stages.maintenance.is_some()
    {
        return Err(ReferenceError::Evidence(
            "sample source, shape, outcome or isolated boundary",
        ));
    }
    validate_timing(&observed.timing.clone().into(), u64::MAX)
        .map_err(|_| ReferenceError::Evidence("invalid wall or diagnostic timing"))
}
fn upper_median(times: &mut [u64]) -> Result<NonZeroU64, ReferenceError> {
    if times.is_empty() {
        return Err(ReferenceError::Evidence("missing estimator samples"));
    }
    times.sort_unstable();
    NonZeroU64::new(times[times.len() / 2]).ok_or(ReferenceError::Evidence("zero reference work"))
}
