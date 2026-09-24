//! Exclusive manual calibration capture, independent of the normal worker.
//! Only the engine's private completed call capture can supply an observation.
use super::*;
use model::statistical::model::{
    CalibrationPartitionV1, FittedWholeWaveModelV1, WholeWaveModelRevision, WholeWaveObservationV1,
    WholeWaveSettingsV1,
};
use model::statistical::SelectedStatisticalFamily;
use profile::statistical_v6::{CostProfileFileV6, WholeWaveProfileShapeV6};
use profile::statistical_v7::CostProfileFileV7;
use profile::statistical_v8::CostProfileFileV8;
mod raw;
use raw::RawSource;

#[derive(Debug, Clone, Serialize)]
pub struct SelectedFitFreezeReceipt {
    pub capture_identity_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
    pub accepted_ordinal: u64,
    pub retained_fit_samples: usize,
    pub fit_parameters_sha256: [u8; 32],
    pub source_prefix_sha256: [u8; 32],
    pub frozen_monotonic_ns: u64,
}

pub(in crate::continuous_engine::inner) struct SelectedCalibrationCapture {
    options: SloCostProfileExportConfig,
    settings: WholeWaveSettingsV1,
    family: SelectedStatisticalFamily,
    revision: WholeWaveModelRevision,
    fingerprint: model::ExecutionFingerprint,
    identity: [u8; 32],
    protocol: [u8; 32],
    clock: Arc<dyn CostObservationClock>,
    opening: ExportClockReading,
    source: RawSource,
    fit: Vec<WholeWaveObservationV1>,
    residual: Vec<WholeWaveObservationV1>,
    frozen: Option<(FittedWholeWaveModelV1, SelectedFitFreezeReceipt)>,
    last_ordinal: u64,
    rows: usize,
    attempted: u64,
    unavailable: u64,
    poisoned: bool,
}
impl SelectedCalibrationCapture {
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn observation(
        capture: &CostCalibrationCapture,
        reconciled: bool,
        identity: [u8; 32],
    ) -> Result<WholeWaveObservationV1, model::statistical::model::ModelUnknown> {
        use model::statistical::model::ModelUnknown;
        if !reconciled {
            return Err(ModelUnknown::InvalidSample);
        }
        let ordinal = match capture.host_stage_queue() {
            Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(n),
            }) if n > 0 => n,
            _ => return Err(ModelUnknown::WrongSource),
        };
        let CostCalibrationStatus::Complete(result) = capture.status() else {
            return Err(ModelUnknown::InvalidSample);
        };
        let stages = capture.host_stages();
        let entry = match result.as_ref() {
            CostCalibrationResult::Observed {
                sample,
                accepted_ordinal,
                disposition: CostCallDisposition::Published,
                ..
            } if *accepted_ordinal == Some(ordinal) => CostEvidenceEntry::Training {
                sample: (**sample).clone(),
                stages,
            },
            CostCalibrationResult::Rejected(reason) => CostEvidenceEntry::StagesOnly {
                stages: stages.ok_or(ModelUnknown::Evidence(
                    StatisticalEvidenceUnknown::MissingProducer,
                ))?,
                legacy_rejection: *reason,
            },
            _ => return Err(ModelUnknown::InvalidSample),
        };
        super::super::trainer::whole_wave_observation(&entry, ordinal, identity)
    }

    pub fn new(
        options: SloCostProfileExportConfig,
        config: &ferrum_types::SloCostObservationConfig,
        fingerprint: model::ExecutionFingerprint,
        protocol: [u8; 32],
        clock: Arc<dyn CostObservationClock>,
    ) -> Result<Self, ExportError> {
        options.validate().map_err(ExportError::Config)?;
        config.validate().map_err(ExportError::Config)?;
        if options
            .declared_clock_max_error_ns
            .is_some_and(|error| error > config.profile_import.max_clock_error_ns)
        {
            return Err(ExportError::Clock(
                "export clock declaration exceeds product import policy",
            ));
        }
        if protocol == [0; 32] || !config.predictor.is_selected() {
            return Err(ExportError::Source(
                "explicit selected predictor and nonzero protocol are required",
            ));
        }
        let revision = match config.predictor {
            ferrum_types::SloCostPredictor::SelectedWholeWaveV1 => {
                WholeWaveModelRevision::OrderedV1
            }
            ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2 => {
                WholeWaveModelRevision::IndependentAttentionV2
            }
            ferrum_types::SloCostPredictor::SelectedWorkSupportV1 => {
                WholeWaveModelRevision::IndependentAttentionWorkSupportV1
            }
            _ => unreachable!("selected checked above"),
        };
        let family = revision.family();
        let settings = WholeWaveSettingsV1::from_policy_limits(
            &super::super::profile::model_settings(&config.model),
        );
        settings
            .validate()
            .map_err(|e| ExportError::Config(format!("{e:?}")))?;
        let profile_path = files::destination(&options.path)?;
        let source_path = files::destination(&options.observations_path)?;
        if profile_path == source_path {
            return Err(ExportError::Source("profile and raw source alias"));
        }
        let producer = ProducerIdentity::current()?;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.selected-calibration.capture.v1\0");
        hash.update(uuid::Uuid::new_v4().as_bytes());
        hash.update(protocol);
        let identity = hash.finalize().into();
        let opening = ExportClockReading::opening(clock.as_ref())?;
        let mut source = RawSource::create(&source_path, options.max_file_bytes.get() as u64)?;
        let mut header = serde_json::json!({"artifact_type":"ferrum.selected-whole-wave-source", "schema_version":1,
            "capture_identity_sha256":identity, "protocol_sha256":protocol, "producer":producer,
            "fingerprint":profile::ProfileFingerprint::from(&fingerprint),
            "settings":profile::statistical_v6::WholeWaveProfileSettingsV6::from(&settings),
            "opening":opening,"declared_clock_max_error_ns":options.declared_clock_max_error_ns});
        if family == SelectedStatisticalFamily::IndependentAttentionV2 {
            header["schema_version"] =
                if revision == WholeWaveModelRevision::IndependentAttentionWorkSupportV1 {
                    3
                } else {
                    2
                }
                .into();
            header["model_revision"] = revision.as_str().into();
        }
        source.record(&header)?;
        Ok(Self {
            options,
            settings,
            family,
            revision,
            fingerprint,
            identity,
            protocol,
            clock,
            opening,
            source,
            fit: Vec::new(),
            residual: Vec::new(),
            frozen: None,
            last_ordinal: 0,
            rows: 0,
            attempted: 0,
            unavailable: 0,
            poisoned: false,
        })
    }

    pub fn record(
        &mut self,
        capture: &CostCalibrationCapture,
        reconciled: bool,
    ) -> Result<(), ExportError> {
        if self.poisoned {
            return Err(ExportError::Source("selected capture already failed"));
        }
        let result = self.record_inner(capture, reconciled);
        self.poisoned |= result.is_err();
        result
    }
    fn record_inner(
        &mut self,
        capture: &CostCalibrationCapture,
        reconciled: bool,
    ) -> Result<(), ExportError> {
        self.attempted = self
            .attempted
            .checked_add(1)
            .ok_or(ExportError::Source("attempt count overflow"))?;
        let queue = capture.host_stage_queue();
        let stages = capture.host_stages();
        let observed = Self::observation(capture, reconciled, self.identity).and_then(|sample| {
            if sample.accepted_ordinal <= self.last_ordinal {
                Err(model::statistical::model::ModelUnknown::DuplicateRecord)
            } else {
                selected_family(&sample.selected, self.family)?;
                Ok(sample)
            }
        });
        let phase = if self.frozen.is_some() {
            "residual"
        } else {
            "fit"
        };
        let mut raw = match &observed {
            Ok(sample) => {
                serde_json::json!({"kind":"observation", "phase":phase,"accepted_ordinal":sample.accepted_ordinal,
                "call_id":sample.call_id,"observed_at_ns":sample.observed_at_ns,"wall_ns":sample.wall_ns,
                "shape":WholeWaveProfileShapeV6::try_from(&sample.exact).map_err(|_| ExportError::Source("unsupported profile shape"))?,
                "selected":sample.selected.to_wire_v1(),"host_stages":stages.as_deref(),"queue":queue})
            }
            Err(reason) => {
                serde_json::json!({"kind":"unavailable","phase":phase,"reason":format!("{reason:?}"),"queue":queue,"host_stages":stages.as_deref()})
            }
        };
        if self.family == SelectedStatisticalFamily::IndependentAttentionV2 {
            if let Ok(sample) = &observed {
                raw["independent_attention"] = serde_json::to_value(
                    sample
                        .selected
                        .independent_attention_v2()
                        .expect("validated above")
                        .to_wire_v2(),
                )
                .map_err(|_| ExportError::Source("encode independent-attention evidence"))?;
            }
        }
        self.source.record(&raw)?;
        let sample = match observed {
            Ok(sample) => sample,
            Err(_) => {
                self.unavailable += 1;
                return Ok(());
            }
        };
        if sample.fingerprint != self.fingerprint {
            return Err(ExportError::Source("capture fingerprint changed"));
        }
        let count = self
            .fit
            .len()
            .checked_add(self.residual.len())
            .and_then(|n| n.checked_add(1))
            .ok_or(ExportError::Source("sample count overflow"))?;
        let rows = self
            .rows
            .checked_add(sample.exact.rows.len())
            .ok_or(ExportError::Source("row count overflow"))?;
        if count
            > self
                .options
                .max_samples
                .get()
                .min(self.settings.max_retained_samples.get())
            || rows
                > self
                    .options
                    .max_total_shape_rows
                    .get()
                    .min(self.settings.max_retained_shape_rows.get())
        {
            return Err(ExportError::Source(
                "selected calibration retention limit reached",
            ));
        }
        self.rows = rows;
        self.last_ordinal = sample.accepted_ordinal;
        let target = if self.frozen.is_some() {
            &mut self.residual
        } else {
            &mut self.fit
        };
        target
            .try_reserve(1)
            .map_err(|_| ExportError::Source("calibration retention allocation failed"))?;
        target.push(sample);
        Ok(())
    }

    pub fn freeze_fit(&mut self, cut: u64) -> Result<SelectedFitFreezeReceipt, ExportError> {
        if self.poisoned || self.frozen.is_some() || cut == 0 || self.last_ordinal > cut {
            return Err(ExportError::Source("invalid fit cut"));
        }
        let now = self
            .clock
            .now_ns()
            .ok_or(ExportError::Clock("fit clock unavailable"))?;
        let partition = CalibrationPartitionV1 {
            source_sha256: self.identity,
            protocol_sha256: self.protocol,
            fit_through_ordinal: cut,
            residual_through_ordinal: u64::MAX,
        };
        let fitted = self
            .revision
            .fit(
                self.fingerprint.clone(),
                self.settings.clone(),
                partition,
                &self.fit,
                now,
            )
            .map_err(|e| ExportError::Config(format!("whole-wave fit: {e:?}")))?;
        let receipt = SelectedFitFreezeReceipt {
            capture_identity_sha256: self.identity,
            protocol_sha256: self.protocol,
            accepted_ordinal: cut,
            retained_fit_samples: self.fit.len(),
            fit_parameters_sha256: fitted.parameter_signature(),
            source_prefix_sha256: self.source.prefix_digest(),
            frozen_monotonic_ns: now,
        };
        if let Err(error) = self
            .source
            .record(&serde_json::json!({"kind":"fit_frozen","receipt":receipt}))
        {
            self.poisoned = true;
            return Err(error);
        }
        self.frozen = Some((fitted, receipt.clone()));
        Ok(receipt)
    }

    pub fn finish(
        mut self,
        cut: u64,
    ) -> Result<(CostProfileCutReceipt, serde_json::Value), ExportError> {
        if self.poisoned {
            return Err(ExportError::Source("selected capture already failed"));
        }
        let (fitted, freeze) = self
            .frozen
            .take()
            .ok_or(ExportError::Source("fit was not frozen"))?;
        if self.last_ordinal > cut {
            return Err(ExportError::Source("residual cut precedes accepted sample"));
        }
        let closing = ExportClockReading::closing(self.clock.as_ref())?;
        let drift = (i128::from(closing.wall_unix_ns) - i128::from(self.opening.wall_unix_ns))
            - (i128::from(closing.monotonic_ns) - i128::from(self.opening.monotonic_ns));
        if closing.monotonic_ns < self.opening.monotonic_ns
            || drift.unsigned_abs()
                > u128::from(self.options.declared_clock_max_error_ns.unwrap()).saturating_mul(2)
        {
            return Err(ExportError::Clock(
                "capture wall/monotonic drift exceeds declaration",
            ));
        }
        let model = fitted
            .seal_residual_cut(cut)
            .and_then(|f| f.calibrate(&self.residual, closing.monotonic_ns))
            .map_err(|e| ExportError::Config(format!("whole-wave residual: {e:?}")))?;
        let mut families = std::collections::BTreeMap::<
            [u8; 32],
            (
                usize,
                usize,
                usize,
                std::collections::BTreeMap<String, usize>,
            ),
        >::new();
        for (residual, samples) in [
            (false, self.fit.as_slice()),
            (true, self.residual.as_slice()),
        ] {
            for sample in samples {
                let family = families
                    .entry(
                        *selected_family(&sample.selected, self.family).map_err(|_| {
                            ExportError::Source("retained sample lost its versioned evidence")
                        })?,
                    )
                    .or_default();
                if residual {
                    family.1 += 1;
                } else {
                    family.0 += 1;
                }
                match model.predict(
                    &sample.fingerprint,
                    &sample.exact,
                    &sample.selected,
                    closing.monotonic_ns,
                ) {
                    Ok(_) => family.2 += 1,
                    Err(reason) => *family.3.entry(format!("{reason:?}")).or_default() += 1,
                }
            }
        }
        // These are retained training/residual points, explicitly NOT heldout
        // coverage and NOT a claim that any intervening joint work is covered.
        let support: Vec<_> = families.into_iter().map(|(family,(fit,residual,known,unknown))|
            serde_json::json!({"family_signature":family,"fit_samples":fit,"residual_samples":residual,
                "retained_point_known":known,"retained_point_unknown":unknown})).collect();
        let summary = serde_json::json!({"kind":"completed_residual_cut","accepted_ordinal":cut,
            "closing":closing,"fit_records":self.fit.len(),"residual_records":self.residual.len(),
            "attempted":self.attempted,"unavailable":self.unavailable,"supported_segments":model.segment_count(),
            "family_support":support,"coverage_scope":"retained fit/residual points only; independent heldout required"});
        self.source.record(&summary)?;
        let source = self.source.finish()?;
        let partition = CalibrationPartitionV1 {
            source_sha256: self.identity,
            protocol_sha256: self.protocol,
            fit_through_ordinal: freeze.accepted_ordinal,
            residual_through_ordinal: cut,
        };
        let profile_source = profile::ProfileSource {
            generator: "ferrum.calibrate-slo".into(),
            generator_revision: env!("CARGO_PKG_VERSION").into(),
            measurement_protocol: match self.revision {
                WholeWaveModelRevision::OrderedV1 => "selected-whole-wave-fit-residual-v1",
                WholeWaveModelRevision::IndependentAttentionV2 => {
                    "selected-independent-attention-fit-residual-v2"
                }
                WholeWaveModelRevision::IndependentAttentionWorkSupportV1 => {
                    "selected-work-support-fit-residual-v1"
                }
            }
            .into(),
            observation_artifact_sha256: source.digest,
        };
        let mut output =
            StagedFile::create(&self.options.path, self.options.max_file_bytes.get() as u64)?;
        match self.revision {
            WholeWaveModelRevision::OrderedV1 => output.json(
                &CostProfileFileV6::from_capture_observations(
                    &self.fingerprint,
                    &self.settings,
                    partition,
                    profile_source,
                    closing.monotonic_ns,
                    closing.wall_unix_ns,
                    self.options.declared_clock_max_error_ns.unwrap(),
                    freeze.fit_parameters_sha256,
                    &self.fit,
                    &self.residual,
                )
                .map_err(|e| ExportError::Config(e.to_string()))?,
            )?,
            WholeWaveModelRevision::IndependentAttentionV2 => output.json(
                &CostProfileFileV7::from_capture_observations(
                    &self.fingerprint,
                    &self.settings,
                    partition,
                    profile_source,
                    closing.monotonic_ns,
                    closing.wall_unix_ns,
                    self.options.declared_clock_max_error_ns.unwrap(),
                    freeze.fit_parameters_sha256,
                    &self.fit,
                    &self.residual,
                )
                .map_err(|e| ExportError::Config(e.to_string()))?,
            )?,
            WholeWaveModelRevision::IndependentAttentionWorkSupportV1 => output.json(
                &CostProfileFileV8::from_capture_observations(
                    &self.fingerprint,
                    &self.settings,
                    partition,
                    profile_source,
                    closing.monotonic_ns,
                    closing.wall_unix_ns,
                    self.options.declared_clock_max_error_ns.unwrap(),
                    freeze.fit_parameters_sha256,
                    &self.fit,
                    &self.residual,
                )
                .map_err(|e| ExportError::Config(e.to_string()))?,
            )?,
        }
        let published = output.publish()?;
        Ok((
            CostProfileCutReceipt {
                accepted_ordinal: cut,
                profile: published.path,
                profile_sha256: published.sha256,
                profile_bytes: published.bytes,
                source: source.path,
                source_sha256: source.sha256,
                source_digest: source.digest,
                source_bytes: source.bytes,
                retained_samples: (self.fit.len() + self.residual.len()) as u64,
                raw_retained_observations: (self.fit.len() + self.residual.len()) as u64,
            },
            summary,
        ))
    }
}

fn selected_family(
    evidence: &ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
    family: SelectedStatisticalFamily,
) -> Result<&[u8; 32], model::statistical::model::ModelUnknown> {
    match family {
        SelectedStatisticalFamily::OrderedV1 => Ok(evidence.family_signature()),
        SelectedStatisticalFamily::IndependentAttentionV2 => evidence
            .independent_attention_v2()
            .map(|v| v.family_signature())
            .ok_or(model::statistical::model::ModelUnknown::Evidence(
                StatisticalEvidenceUnknown::MissingProducer,
            )),
    }
}
