use super::*;
#[derive(Debug, Clone)]
pub struct ImportedWholeWaveModelV1 {
    pub provenance: WholeWaveImportProvenanceV1,
    model: WholeWaveModelV1,
    pub clock: ProfileObservationClock,
    pub file_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub capture_identity_sha256: [u8; 32],
    pub fit_parameters_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
    pub fit_records: usize,
    pub residual_records: usize,
    pub conservative_clock_error_ns: u64,
}
#[derive(Debug, Clone)]
pub struct WholeWaveImportProvenanceV1 {
    pub loaded_from: Option<PathBuf>,
    pub file_bytes: usize,
    pub generated_unix_ns: u64,
    pub loaded_unix_ns: u64,
    pub oldest_imported_age_ns: u64,
    pub newest_imported_age_ns: u64,
    pub source: ProfileSource,
    pub fit_through_ordinal: u64,
    pub residual_through_ordinal: u64,
}
impl ImportedWholeWaveModelV1 {
    pub fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &CanonicalWaveCostShape,
        evidence: &StatisticalWaveEvidenceV1,
        local_now_ns: u64,
    ) -> Result<WholeWavePredictionV1, ModelUnknown> {
        self.model.predict(
            fingerprint,
            shape,
            evidence,
            self.clock
                .model_now_ns(local_now_ns)
                .map_err(|_| ModelUnknown::Clock)?,
        )
    }
    pub fn predict_input(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &super::super::super::cost_model::statistical::StatisticalModelInputV1,
        local_now_ns: u64,
    ) -> Result<WholeWavePredictionV1, ModelUnknown> {
        self.model.predict_input(
            fingerprint,
            input,
            self.clock
                .model_now_ns(local_now_ns)
                .map_err(|_| ModelUnknown::Clock)?,
        )
    }
    pub fn segment_count(&self) -> usize {
        self.model.segment_count()
    }
}
pub fn load_whole_wave_profile_v6(
    path: &Path,
    fingerprint: &ExecutionFingerprint,
    settings: &WholeWaveSettingsV1,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    limits.validate()?;
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(CostProfileError::Metadata("profile must be a regular file"));
    }
    let length = usize::try_from(metadata.len())
        .ok()
        .filter(|n| *n <= limits.max_file_bytes.get())
        .ok_or(CostProfileError::Limit("file byte limit exceeded"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| CostProfileError::Limit("profile allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0u8; 1])? != 0 {
        return Err(CostProfileError::Metadata("profile grew while reading"));
    }
    let mut loaded =
        load_whole_wave_profile_v6_bytes(&bytes, fingerprint, settings, limits, clock)?;
    loaded.provenance.loaded_from = Some(path.to_owned());
    Ok(loaded)
}
pub fn load_whole_wave_profile_v6_bytes(
    bytes: &[u8],
    fingerprint: &ExecutionFingerprint,
    settings: &WholeWaveSettingsV1,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    limits.validate()?;
    settings
        .validate()
        .map_err(|_| CostProfileError::Metadata("invalid whole-wave settings"))?;
    if bytes.len() > limits.max_file_bytes.get() {
        return Err(CostProfileError::Limit("file byte limit exceeded"));
    }
    // Reject older envelopes before interpreting their sample payload.
    #[derive(Deserialize)]
    struct Version {
        schema_version: u32,
    }
    let version: Version = serde_json::from_slice(bytes)?;
    if version.schema_version != COST_PROFILE_SCHEMA_VERSION_V6 {
        return Err(CostProfileError::UnsupportedVersion(version.schema_version));
    }
    let file: CostProfileFileV6 = serde_json::from_slice(bytes)?;
    if file.model_revision != MODEL_REVISION {
        return Err(CostProfileError::Metadata(
            "unsupported whole-wave model revision",
        ));
    }
    if file.fingerprint != ProfileFingerprint::from(fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    if file.settings != WholeWaveProfileSettingsV6::from(settings) {
        return Err(CostProfileError::SettingsMismatch);
    }
    if file.samples.is_empty()
        || file.samples.len() > limits.max_samples.get()
        || file.samples.len() > settings.max_retained_samples.get()
    {
        return Err(CostProfileError::Limit("sample limit exceeded"));
    }
    for field in [
        &file.source.generator,
        &file.source.generator_revision,
        &file.source.measurement_protocol,
    ] {
        if field.is_empty()
            || field.len() > limits.max_source_field_bytes.get()
            || field.chars().any(char::is_control)
        {
            return Err(CostProfileError::Metadata("invalid source field"));
        }
    }
    if file.source.observation_artifact_sha256 == [0; 32] {
        return Err(CostProfileError::Metadata("unsealed observation source"));
    }
    let wall = clock
        .wall_unix_ns
        .filter(|x| *x > 0)
        .ok_or(CostProfileError::Clock("trusted local wall clock required"))?;
    let local_error = clock
        .wall_max_error_ns
        .ok_or(CostProfileError::Clock("local uncertainty required"))?;
    let source_error = file
        .source_clock_max_error_ns
        .ok_or(CostProfileError::Clock("source uncertainty required"))?;
    if source_error > limits.max_clock_error_ns || local_error > limits.max_clock_error_ns {
        return Err(CostProfileError::Clock("clock uncertainty exceeds policy"));
    }
    let uncertainty = source_error
        .checked_add(local_error)
        .ok_or(CostProfileError::Clock("clock uncertainty overflow"))?;
    let age = |at: u64| {
        wall.checked_sub(at)
            .filter(|_| at > 0)
            .and_then(|n| n.checked_add(uncertainty))
            .ok_or(CostProfileError::Clock("future or missing wall timestamp"))
    };
    if age(file.generated_unix_ns)? > limits.max_profile_age_ns.get() {
        return Err(CostProfileError::Clock("profile generation too old"));
    }
    let anchor = settings.max_sample_age_ns.get();
    let model_clock = ProfileObservationClock {
        source_monotonic_anchor_ns: clock.monotonic_now_ns,
        model_anchor_ns: anchor,
    };
    let partition = CalibrationPartitionV1 {
        source_sha256: file.capture_identity_sha256,
        protocol_sha256: file.protocol_sha256,
        fit_through_ordinal: file.fit_through_ordinal,
        residual_through_ordinal: file.residual_through_ordinal,
    };
    let mut fit = Vec::new();
    let mut residual = Vec::new();
    let mut rows = 0usize;
    let mut ordinals = HashSet::new();
    let mut calls = HashSet::new();
    let mut residual_started = false;
    let mut oldest_age = 0;
    let mut newest_age = u64::MAX;
    for record in file.samples {
        if !ordinals.insert(record.accepted_ordinal) || !calls.insert(record.call_id) {
            return Err(CostProfileError::Metadata(
                "duplicate accepted ordinal or call ID",
            ));
        }
        if record.measured_unix_ns > file.generated_unix_ns {
            return Err(CostProfileError::Clock("sample follows profile generation"));
        }
        let sample_age = age(record.measured_unix_ns)?;
        oldest_age = oldest_age.max(sample_age);
        newest_age = newest_age.min(sample_age);
        let observed_at_ns = anchor
            .checked_sub(sample_age)
            .ok_or(CostProfileError::Clock(
                "whole-wave calibration contains expired sample",
            ))?;
        rows = rows
            .checked_add(record.shape.rows.len())
            .ok_or(CostProfileError::Limit("row count overflow"))?;
        if rows > limits.max_total_shape_rows.get() || rows > settings.max_retained_shape_rows.get()
        {
            return Err(CostProfileError::Limit("shape row limit exceeded"));
        }
        let exact = record.shape.canonical()?;
        let selected = StatisticalWaveEvidenceV1::from_wire_v1(record.selected, &exact)
            .map_err(|_| CostProfileError::Metadata("unbound selected-algorithm evidence"))?;
        let sample = WholeWaveObservationV1 {
            source_sha256: partition.source_sha256,
            accepted_ordinal: record.accepted_ordinal,
            call_id: record.call_id,
            fingerprint: fingerprint.clone(),
            exact,
            selected,
            boundary: CostBoundary::PreparationToHostSettledV1,
            outcome: record.outcome.into(),
            observed_at_ns,
            wall_ns: record.wall_ns,
        };
        match record.phase {
            WholeWaveProfilePhaseV6::Fit if !residual_started => fit.push(sample),
            WholeWaveProfilePhaseV6::Fit => {
                return Err(CostProfileError::Metadata("fit records after residual cut"))
            }
            WholeWaveProfilePhaseV6::Residual => {
                residual_started = true;
                residual.push(sample);
            }
        }
    }
    let frozen = FittedWholeWaveModelV1::fit(
        fingerprint.clone(),
        settings.clone(),
        partition,
        &fit,
        anchor,
    )
    .map_err(|_| CostProfileError::Metadata("invalid independent fit calibration"))?;
    if frozen.parameter_signature() != file.fit_parameters_sha256 {
        return Err(CostProfileError::Metadata(
            "frozen fit parameter digest mismatch",
        ));
    }
    let model = frozen
        .calibrate(&residual, anchor)
        .map_err(|_| CostProfileError::Metadata("invalid independent residual calibration"))?;
    let source_sha256 = file.source.observation_artifact_sha256;
    Ok(ImportedWholeWaveModelV1 {
        provenance: WholeWaveImportProvenanceV1 {
            loaded_from: None,
            file_bytes: bytes.len(),
            generated_unix_ns: file.generated_unix_ns,
            loaded_unix_ns: wall,
            oldest_imported_age_ns: oldest_age,
            newest_imported_age_ns: newest_age,
            source: file.source,
            fit_through_ordinal: partition.fit_through_ordinal,
            residual_through_ordinal: partition.residual_through_ordinal,
        },
        model,
        clock: model_clock,
        file_sha256: Sha256::digest(bytes).into(),
        source_sha256,
        capture_identity_sha256: partition.source_sha256,
        fit_parameters_sha256: file.fit_parameters_sha256,
        protocol_sha256: partition.protocol_sha256,
        fit_records: fit.len(),
        residual_records: residual.len(),
        conservative_clock_error_ns: uncertainty,
    })
}
