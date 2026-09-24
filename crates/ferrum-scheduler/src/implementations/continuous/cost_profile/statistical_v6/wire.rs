use super::*;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeWaveProfileSettingsV6 {
    pub max_buckets: NonZeroUsize,
    pub max_samples_per_bucket: NonZeroUsize,
    pub min_samples: NonZeroUsize,
    pub max_retained_samples: NonZeroUsize,
    pub max_retained_shape_rows: NonZeroUsize,
    pub residual_quantile: f64,
    pub drift_margin_ns: u64,
    pub max_wave_ns: NonZeroU64,
    pub max_sample_age_ns: NonZeroU64,
    pub shape_limits: ProfileShapeLimits,
}
impl From<&WholeWaveSettingsV1> for WholeWaveProfileSettingsV6 {
    fn from(s: &WholeWaveSettingsV1) -> Self {
        Self {
            max_buckets: s.max_buckets,
            max_samples_per_bucket: s.max_samples_per_bucket,
            min_samples: s.min_samples,
            max_retained_samples: s.max_retained_samples,
            max_retained_shape_rows: s.max_retained_shape_rows,
            residual_quantile: s.residual_quantile,
            drift_margin_ns: s.drift_margin_ns,
            max_wave_ns: s.max_wave_ns,
            max_sample_age_ns: s.max_sample_age_ns,
            shape_limits: ProfileShapeLimits {
                max_rows: s.shape_limits.max_rows,
                max_context_tokens: s.shape_limits.max_context_tokens,
                max_prefill_tokens_per_wave: s.shape_limits.max_prefill_tokens_per_wave,
                max_state_bytes: s.shape_limits.max_state_bytes,
                max_maintenance_units: s.shape_limits.max_maintenance_units,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeWaveProfilePhaseV6 {
    Fit,
    Residual,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeWaveProfileKindV6 {
    Decode,
    Prefill,
    Mixed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WholeWaveProfileWorkV6 {
    Decode {
        kv_tokens: u32,
    },
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
}
impl WholeWaveProfileWorkV6 {
    fn actual(self) -> ActualRowWork {
        match self {
            Self::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
            Self::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeWaveProfileShapeV6 {
    pub kind: WholeWaveProfileKindV6,
    pub provider_signature: [u8; 32],
    pub output_policy_signature: [u8; 32],
    pub numeric_features: CanonicalWaveCostFeatures,
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    pub row_multiset_features: HostRowMultisetCostFeaturesV2,
    #[serde(deserialize_with = "bounded_rows")]
    pub rows: Vec<WholeWaveProfileWorkV6>,
    pub recurrent_state_bytes: u64,
}
impl TryFrom<&CanonicalWaveCostShape> for WholeWaveProfileShapeV6 {
    type Error = CostProfileError;
    fn try_from(s: &CanonicalWaveCostShape) -> Result<Self, Self::Error> {
        if s.path != ActualWavePath::PlanRuntime
            || s.graph != ActualWaveGraphState::Disabled
            || s.row_order != ActualWaveRowOrder::Ordered
        {
            return Err(CostProfileError::Metadata(
                "v6 requires eager ordered PlanRuntime",
            ));
        }
        Ok(Self {
            kind: match s.kind {
                ActualWaveKind::Decode => WholeWaveProfileKindV6::Decode,
                ActualWaveKind::Prefill => WholeWaveProfileKindV6::Prefill,
                ActualWaveKind::Mixed => WholeWaveProfileKindV6::Mixed,
                _ => return Err(CostProfileError::Metadata("unsupported v6 wave")),
            },
            provider_signature: s.provider_signature,
            output_policy_signature: s.output_policy_signature,
            numeric_features: s
                .numeric_features
                .clone()
                .ok_or(CostProfileError::Metadata(
                    "missing actual numeric evidence",
                ))?,
            host_content_features: s.host_content_features,
            row_multiset_features: s
                .row_multiset_features
                .clone()
                .ok_or(CostProfileError::Metadata("missing actual host categories"))?,
            rows: s
                .rows
                .iter()
                .map(|w| match *w {
                    ActualRowWork::Decode { kv_tokens } => {
                        Ok(WholeWaveProfileWorkV6::Decode { kv_tokens })
                    }
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    } => Ok(WholeWaveProfileWorkV6::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    }),
                    _ => Err(CostProfileError::Metadata("unsupported v6 row")),
                })
                .collect::<Result<_, _>>()?,
            recurrent_state_bytes: s.recurrent_state_bytes,
        })
    }
}
impl WholeWaveProfileShapeV6 {
    pub(in crate::implementations::continuous::cost_profile) fn canonical(
        self,
    ) -> Result<CanonicalWaveCostShape, CostProfileError> {
        let decode = self
            .rows
            .iter()
            .any(|w| matches!(w, WholeWaveProfileWorkV6::Decode { .. }));
        let prefill = self
            .rows
            .iter()
            .any(|w| matches!(w, WholeWaveProfileWorkV6::Prefill { .. }));
        if !matches!(
            (self.kind, decode, prefill),
            (WholeWaveProfileKindV6::Decode, true, false)
                | (WholeWaveProfileKindV6::Prefill, false, true)
                | (WholeWaveProfileKindV6::Mixed, true, true)
        ) {
            return Err(CostProfileError::Metadata("inconsistent v6 physical roles"));
        }
        Ok(CanonicalWaveCostShape {
            kind: match self.kind {
                WholeWaveProfileKindV6::Decode => ActualWaveKind::Decode,
                WholeWaveProfileKindV6::Prefill => ActualWaveKind::Prefill,
                WholeWaveProfileKindV6::Mixed => ActualWaveKind::Mixed,
            },
            path: ActualWavePath::PlanRuntime,
            graph: ActualWaveGraphState::Disabled,
            row_order: ActualWaveRowOrder::Ordered,
            provider_signature: self.provider_signature,
            output_policy_signature: self.output_policy_signature,
            numeric_features: Some(self.numeric_features),
            host_content_features: self.host_content_features,
            row_multiset_features: Some(self.row_multiset_features),
            rows: self
                .rows
                .into_iter()
                .map(WholeWaveProfileWorkV6::actual)
                .collect(),
            recurrent_state_bytes: self.recurrent_state_bytes,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeWaveProfileSampleV6 {
    pub phase: WholeWaveProfilePhaseV6,
    pub accepted_ordinal: u64,
    pub call_id: u64,
    pub measured_unix_ns: u64,
    pub shape: WholeWaveProfileShapeV6,
    pub selected: StatisticalWaveEvidenceWireV1,
    pub boundary: v3::ProfileCostBoundaryV3,
    pub outcome: ProfileObservationOutcome,
    pub wall_ns: u64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "S: Deserialize<'de>"))]
pub struct WholeWaveProfileFile<S> {
    /// Stable capture identity, distinct from the eventually sealed raw bytes.
    pub capture_identity_sha256: [u8; 32],
    /// Frozen before residual collection; loading must reproduce these parameters.
    pub fit_parameters_sha256: [u8; 32],
    pub schema_version: u32,
    pub model_revision: String,
    pub fingerprint: ProfileFingerprint,
    pub settings: WholeWaveProfileSettingsV6,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    pub protocol_sha256: [u8; 32],
    pub fit_through_ordinal: u64,
    pub residual_through_ordinal: u64,
    #[serde(deserialize_with = "bounded_samples")]
    pub samples: Vec<S>,
}
pub type CostProfileFileV6 = WholeWaveProfileFile<WholeWaveProfileSampleV6>;
fn bounded_samples<'de, D: Deserializer<'de>, S: Deserialize<'de>>(
    d: D,
) -> Result<Vec<S>, D::Error> {
    bounded_vec::<D, _, HARD_SAMPLES>(d)
}
impl CostProfileFileV6 {
    /// Paired real source clocks; wall times are derived without resetting the
    /// age of measurements. Complete raw records remain the source artifact.
    pub fn from_observations(
        fingerprint: &ExecutionFingerprint,
        settings: &WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        source: ProfileSource,
        generated_monotonic_ns: u64,
        generated_unix_ns: u64,
        source_clock_max_error_ns: u64,
        fit: &[WholeWaveObservationV1],
        residual: &[WholeWaveObservationV1],
    ) -> Result<Self, CostProfileError> {
        if source.observation_artifact_sha256 != partition.source_sha256 {
            return Err(CostProfileError::Metadata("source digest mismatch"));
        }
        let frozen = FittedWholeWaveModelV1::fit(
            fingerprint.clone(),
            settings.clone(),
            partition,
            fit,
            generated_monotonic_ns,
        )
        .map_err(|_| CostProfileError::Metadata("invalid fit population"))?;
        Self::from_capture_observations(
            fingerprint,
            settings,
            partition,
            source,
            generated_monotonic_ns,
            generated_unix_ns,
            source_clock_max_error_ns,
            frozen.parameter_signature(),
            fit,
            residual,
        )
    }

    /// Export a live three-phase capture only after the training source is
    /// sealed. The session ID was fixed before collecting; it never pretends
    /// that later observations were present in an earlier file digest.
    pub fn from_capture_observations(
        fingerprint: &ExecutionFingerprint,
        settings: &WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        source: ProfileSource,
        generated_monotonic_ns: u64,
        generated_unix_ns: u64,
        source_clock_max_error_ns: u64,
        frozen_fit_parameters_sha256: [u8; 32],
        fit: &[WholeWaveObservationV1],
        residual: &[WholeWaveObservationV1],
    ) -> Result<Self, CostProfileError> {
        if source.observation_artifact_sha256 == [0; 32] || frozen_fit_parameters_sha256 == [0; 32]
        {
            return Err(CostProfileError::Metadata(
                "unsealed source or missing frozen fit",
            ));
        }
        let frozen = FittedWholeWaveModelV1::fit(
            fingerprint.clone(),
            settings.clone(),
            partition,
            fit,
            generated_monotonic_ns,
        )
        .map_err(|_| CostProfileError::Metadata("invalid independent whole-wave fit"))?;
        if frozen.parameter_signature() != frozen_fit_parameters_sha256 {
            return Err(CostProfileError::Metadata(
                "fit changed after its pre-residual freeze",
            ));
        }
        frozen
            .calibrate(residual, generated_monotonic_ns)
            .map_err(|_| CostProfileError::Metadata("invalid independent whole-wave residual"))?;
        let mut samples = Vec::with_capacity(fit.len() + residual.len());
        for (phase, rows) in [
            (WholeWaveProfilePhaseV6::Fit, fit),
            (WholeWaveProfilePhaseV6::Residual, residual),
        ] {
            for row in rows {
                let age = generated_monotonic_ns
                    .checked_sub(row.observed_at_ns)
                    .ok_or(CostProfileError::Clock("future monotonic observation"))?;
                let measured_unix_ns = generated_unix_ns
                    .checked_sub(age)
                    .filter(|x| *x > 0)
                    .ok_or(CostProfileError::Clock("wall timestamp underflow"))?;
                samples.push(WholeWaveProfileSampleV6 {
                    phase,
                    accepted_ordinal: row.accepted_ordinal,
                    call_id: row.call_id,
                    measured_unix_ns,
                    shape: (&row.exact).try_into()?,
                    selected: row.selected.to_wire_v1(),
                    boundary: v3::ProfileCostBoundaryV3::PreparationToHostSettledV1,
                    outcome: ProfileObservationOutcome::Completed {},
                    wall_ns: row.wall_ns,
                });
            }
        }
        Ok(Self {
            capture_identity_sha256: partition.source_sha256,
            fit_parameters_sha256: frozen_fit_parameters_sha256,
            schema_version: COST_PROFILE_SCHEMA_VERSION_V6,
            model_revision: MODEL_REVISION.into(),
            fingerprint: fingerprint.into(),
            settings: settings.into(),
            generated_unix_ns,
            source_clock_max_error_ns: Some(source_clock_max_error_ns),
            source,
            protocol_sha256: partition.protocol_sha256,
            fit_through_ordinal: partition.fit_through_ordinal,
            residual_through_ordinal: partition.residual_through_ordinal,
            samples,
        })
    }
}
impl<S: Serialize> WholeWaveProfileFile<S> {
    pub fn to_bounded_bytes(&self, max_file_bytes: usize) -> Result<Vec<u8>, CostProfileError> {
        if max_file_bytes == 0 || max_file_bytes > HARD_FILE_BYTES {
            return Err(CostProfileError::Limit("invalid export byte bound"));
        }
        struct Writer {
            data: Vec<u8>,
            limit: usize,
        }
        impl std::io::Write for Writer {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if self
                    .data
                    .len()
                    .checked_add(b.len())
                    .is_none_or(|n| n > self.limit)
                {
                    return Err(std::io::Error::other("profile byte limit"));
                }
                self.data
                    .try_reserve_exact(b.len())
                    .map_err(std::io::Error::other)?;
                self.data.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut writer = Writer {
            data: Vec::new(),
            limit: max_file_bytes,
        };
        serde_json::to_writer(&mut writer, self)?;
        Ok(writer.data)
    }
}
