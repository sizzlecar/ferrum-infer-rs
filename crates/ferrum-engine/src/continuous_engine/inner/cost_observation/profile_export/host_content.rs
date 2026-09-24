//! Distinct profile-v3 records, preserving the accepted entry and old clocks.
use super::*;
use serde::ser::{SerializeSeq, SerializeStruct};

pub(super) const COVERAGE: &str = "bounded original evidence entries; legacy token-commit observations/source_record remain separate and do not train the empirical-host-content model; complete eligible host-settled stages train profile v3 with independent host_source_record and the same accepted_ordinal; failed/missing/additional work never becomes zero-cost success; original receipt time is preserved; full wall and content residuals are empirical, not worst-case or client-visible guarantees; queue/training/export populations and unknown coverage remain separate; retrospective pre-update prediction is not future-query coverage";
pub(super) fn coverage(plan: &ExportPlan) -> &'static str {
    if ProfileExporter::prompt_range_mode(plan) {
        "bounded original evidence entries; profile v5 explicitly selects empirical prompt-total joint support using complete host-settled row evidence; actual total/count/order and accepted/source ordinals stay original; total is not an exact statistical equality key but must be within one complete observed joint support point and its measured range; chunk count, final/output/host classes and provider route remain exact; original clocks/TTL/limits are unchanged; this is an empirical estimate, not a worst-case or client-visible guarantee"
    } else if ProfileExporter::row_multiset_mode(plan) {
        "bounded original evidence entries; profile v4 requires actual ordered row-multiset and numeric evidence plus complete host-settled receipts; only the statistical model canonicalizes supported row tuples; legacy token-commit source_record and host_source_record remain independent, joined by accepted_ordinal; missing/failed evidence is not training; original clocks, TTL and row capacity limits remain unchanged; empirical residuals are not worst-case or client-visible guarantees"
    } else if ProfileExporter::host_content_mode(plan) {
        COVERAGE
    } else {
        super::COVERAGE
    }
}

pub(super) fn retain(
    plan: &ExportPlan,
    accepted_ordinal: u64,
    host_source_record: Option<u64>,
    evaluation: HostContentEvaluation,
    observation: Option<&model::WaveCostObservation>,
) -> Result<RetainedHostContent, ExportError> {
    if !ProfileExporter::host_content_mode(plan) {
        return Err(ExportError::Source(
            "host-content training requires explicit model mode",
        ));
    }
    let sample = observation
        .map(|sample| {
            if sample.boundary != model::CostBoundary::PreparationToHostSettledV1
                || sample.outcome != model::WaveObservationOutcome::Completed
                || (evaluation.training.recorded() && sample.fingerprint != plan.fingerprint)
                || evaluation.rejection.is_some()
            {
                return Err(ExportError::Source(
                    "invalid host-content original observation",
                ));
            }
            let source_record = host_source_record.ok_or(ExportError::Source(
                "host-content sample has no original host stages",
            ))?;
            let measured_unix_ns = sample
                .observed_at_ns
                .checked_sub(plan.opening.monotonic_ns)
                .and_then(|elapsed| plan.opening.wall_unix_ns.checked_add(elapsed))
                .ok_or(ExportError::Clock(
                    "host receipt predates source anchor or overflows",
                ))?;
            let timing = profile::ProfileWaveTiming {
                wall_total_ns: sample.timing.wall_total_ns,
                device_elapsed_ns: None,
                stages: Default::default(),
            };
            if ProfileExporter::row_multiset_mode(plan) {
                Ok(RetainedHostSample::V4(profile_v4::ProfileSampleV4 {
                    source_record,
                    accepted_ordinal,
                    measured_unix_ns,
                    shape: (&sample.actual_shape)
                        .try_into()
                        .map_err(|_| ExportError::Source("invalid row-multiset shape"))?,
                    boundary: profile_v4::ProfileCostBoundaryV4::PreparationToHostSettledV1,
                    outcome: profile::ProfileObservationOutcome::Completed {},
                    timing,
                }))
            } else {
                Ok(RetainedHostSample::V3(profile_v3::ProfileSampleV3 {
                    source_record,
                    accepted_ordinal,
                    measured_unix_ns,
                    shape: (&sample.actual_shape)
                        .try_into()
                        .map_err(|_| ExportError::Source("invalid host-content shape"))?,
                    boundary: profile_v3::ProfileCostBoundaryV3::PreparationToHostSettledV1,
                    outcome: profile::ProfileObservationOutcome::Completed {},
                    timing,
                }))
            }
        })
        .transpose()?;
    if evaluation.training.recorded() && sample.is_none() {
        return Err(ExportError::Source(
            "Recorded host-content training has no original sample",
        ));
    }
    Ok(RetainedHostContent {
        host_source_record,
        evaluation,
        sample,
        row_multiset: ProfileExporter::row_multiset_mode(plan),
    })
}

pub(super) fn retained_rows(row: &RetainedHostContent) -> Option<usize> {
    let Some(sample) = &row.sample else {
        return Some(0);
    };
    let (shape, extra) = match sample {
        RetainedHostSample::V3(sample) => (&sample.shape.exact, 0),
        RetainedHostSample::V4(sample) => (
            &sample.shape.exact,
            sample.shape.row_multiset_features.rows.capacity(),
        ),
    };
    shape
        .exact
        .decode_kv_tokens
        .capacity()
        .checked_add(shape.exact.prefill_chunks.capacity())?
        .checked_add(
            shape
                .numeric_features
                .as_ref()
                .map_or(0, |features| features.rows.capacity()),
        )?
        .checked_add(extra)
}

struct BorrowedProfile<'a> {
    header: profile_v2::CostProfileFileV2,
    host_content: bool,
    row_multiset: bool,
    prompt_range: bool,
    samples: &'a [Retained],
}
impl Serialize for BorrowedProfile<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Samples<'a> {
            rows: &'a [Retained],
            host_content: bool,
        }
        impl Serialize for Samples<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut sequence = serializer.serialize_seq(None)?;
                for entry in self.rows {
                    if self.host_content {
                        if let Some(row) = &entry.host_content {
                            if row.evaluation.training.recorded() {
                                if let Some(sample) = &row.sample {
                                    sequence.serialize_element(sample)?;
                                }
                            }
                        }
                    } else if let Some(row) = &entry.observation {
                        if row.training.recorded() {
                            sequence.serialize_element(&row.sample)?;
                        }
                    }
                }
                sequence.end()
            }
        }
        let h = &self.header;
        let mut out = serializer.serialize_struct("CostProfileFile", 7)?;
        let version = if self.prompt_range {
            profile_v5::COST_PROFILE_SCHEMA_VERSION_V5
        } else if self.row_multiset {
            profile_v4::COST_PROFILE_SCHEMA_VERSION_V4
        } else if self.host_content {
            profile_v3::COST_PROFILE_SCHEMA_VERSION_V3
        } else {
            h.schema_version
        };
        out.serialize_field("schema_version", &version)?;
        out.serialize_field("fingerprint", &h.fingerprint)?;
        out.serialize_field("settings", &h.settings)?;
        out.serialize_field("generated_unix_ns", &h.generated_unix_ns)?;
        out.serialize_field("source_clock_max_error_ns", &h.source_clock_max_error_ns)?;
        out.serialize_field("source", &h.source)?;
        out.serialize_field(
            "samples",
            &Samples {
                rows: self.samples,
                host_content: self.host_content,
            },
        )?;
        out.end()
    }
}
pub(super) fn profile_size(
    plan: &ExportPlan,
    producer: &ProducerIdentity,
) -> Result<u64, ExportError> {
    let mut header =
        ProfileExporter::profile_file(plan, producer, [u8::MAX; 32], u64::MAX, Vec::new());
    header.source.measurement_protocol = coverage(plan).into();
    json_size(&BorrowedProfile {
        header,
        host_content: ProfileExporter::host_content_mode(plan),
        row_multiset: ProfileExporter::row_multiset_mode(plan),
        prompt_range: ProfileExporter::prompt_range_mode(plan),
        samples: &[],
    })
}
impl ProfileExporter {
    pub(super) fn profile_sample_count(&self) -> u64 {
        if Self::host_content_mode(&self.plan) {
            self.counts.retained_host_content_samples
        } else {
            self.counts.retained_samples
        }
    }
    pub(super) fn write_profile(
        &self,
        path: &Path,
        source_hash: [u8; 32],
        generated_unix_ns: u64,
        coverage: &str,
    ) -> Result<PublishedFile, ExportError> {
        let mut header = Self::profile_file(
            &self.plan,
            &self.producer,
            source_hash,
            generated_unix_ns,
            Vec::new(),
        );
        header.source.measurement_protocol = coverage.into();
        let mut file = StagedFile::create(path, self.plan.options.max_file_bytes.get() as u64)?;
        file.json(&BorrowedProfile {
            header,
            host_content: Self::host_content_mode(&self.plan),
            row_multiset: Self::row_multiset_mode(&self.plan),
            prompt_range: Self::prompt_range_mode(&self.plan),
            samples: &self.samples,
        })?;
        file.publish()
    }
}
