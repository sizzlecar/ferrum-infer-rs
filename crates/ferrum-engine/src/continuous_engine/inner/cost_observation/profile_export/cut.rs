//! Publish a borrowed, immutable training prefix without closing live export.
use super::*;
use serde::ser::{SerializeSeq, SerializeStruct};

const CUT_COVERAGE: &str = "original locally accepted evidence entries through an explicit FIFO ordinal; host_stages_v1 are auxiliary and never train or occupy a profile source_record; bounded retained subset, only Recorded cost observations enter profile v2; imported seed is not copied; original measurement timestamps remain unchanged; this is training evidence, not serialization of the online predictor; validate the product-loader imported artifact independently before use; sink counters are a contemporaneous non-atomic funnel, not the accepted cut";

#[derive(Debug, Clone)]
pub(in crate::continuous_engine) struct CostProfileCutPaths {
    pub profile: PathBuf,
    pub source: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine) struct CostProfileCutReceipt {
    pub accepted_ordinal: u64,
    pub profile: PathBuf,
    pub profile_sha256: String,
    pub profile_bytes: u64,
    pub source: PathBuf,
    pub source_sha256: String,
    pub source_digest: [u8; 32],
    pub source_bytes: u64,
    pub retained_samples: u64,
    pub raw_retained_observations: u64,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CutRecord<'a> {
    Header {
        artifact_type: &'static str,
        schema_version: u32,
        accepted_ordinal: u64,
        fingerprint: &'a profile::ProfileFingerprint,
        settings: &'a profile_v2::ProfileModelSettingsV2,
        producer: &'a ProducerIdentity,
        opening: ExportClockReading,
        declared_clock_max_error_ns: u64,
        coverage: &'static str,
    },
    Summary {
        accepted_ordinal: u64,
        closing: ExportClockReading,
        counts: &'a ExportCounts,
        sink: SinkEvidence,
        training: &'a TrainingAuditSnapshot,
        coverage: &'static str,
    },
}

/// Stream the existing v2 wire representation without cloning retained rows.
struct BorrowedProfile<'a> {
    header: profile_v2::CostProfileFileV2,
    samples: &'a [Retained],
}
impl Serialize for BorrowedProfile<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Samples<'a>(&'a [Retained]);
        impl Serialize for Samples<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let count = self
                    .0
                    .iter()
                    .filter_map(|entry| entry.observation.as_ref())
                    .filter(|row| row.training.recorded())
                    .count();
                let mut sequence = serializer.serialize_seq(Some(count))?;
                for row in self
                    .0
                    .iter()
                    .filter_map(|entry| entry.observation.as_ref())
                    .filter(|row| row.training.recorded())
                {
                    sequence.serialize_element(&row.sample)?;
                }
                sequence.end()
            }
        }
        let h = &self.header;
        let mut out = serializer.serialize_struct("CostProfileFileV2", 7)?;
        out.serialize_field("schema_version", &h.schema_version)?;
        out.serialize_field("fingerprint", &h.fingerprint)?;
        out.serialize_field("settings", &h.settings)?;
        out.serialize_field("generated_unix_ns", &h.generated_unix_ns)?;
        out.serialize_field("source_clock_max_error_ns", &h.source_clock_max_error_ns)?;
        out.serialize_field("source", &h.source)?;
        out.serialize_field("samples", &Samples(self.samples))?;
        out.end()
    }
}

impl ProfileExporter {
    pub(super) fn write_cut(
        &self,
        paths: CostProfileCutPaths,
        cutoff: u64,
        closing: ExportClockReading,
        stats: CostSampleStats,
        training: &TrainingAuditSnapshot,
    ) -> Result<CostProfileCutReceipt, ExportError> {
        if self.last_accepted_ordinal != cutoff
            || self
                .samples
                .iter()
                .any(|sample| sample.accepted_ordinal > cutoff)
        {
            return Err(ExportError::Source(
                "training export is not at the requested cut",
            ));
        }
        let profile_path = files::destination(&paths.profile)?;
        let source_path = files::destination(&paths.source)?;
        if profile_path == source_path
            || [&profile_path, &source_path]
                .iter()
                .any(|path| **path == self.profile_path || **path == self.raw_path)
        {
            return Err(ExportError::Source(
                "training cut aliases live export destinations",
            ));
        }
        self.validate_closing(closing)?;
        let host_content = Self::host_content_mode(&self.plan);
        let coverage = if host_content {
            host_content::coverage(&self.plan)
        } else {
            CUT_COVERAGE
        };
        let fingerprint = profile::ProfileFingerprint::from(&self.plan.fingerprint);
        let settings = profile_v2::ProfileModelSettingsV2::from(&self.plan.settings);
        let limit = self.plan.options.max_file_bytes.get() as u64;
        let mut raw = StagedFile::create(&source_path, limit)?;
        raw.json_line(&CutRecord::Header {
            artifact_type: "ferrum.cost-training-cut",
            schema_version: if Self::row_multiset_mode(&self.plan) {
                4
            } else if host_content {
                3
            } else {
                2
            },
            accepted_ordinal: cutoff,
            fingerprint: &fingerprint,
            settings: &settings,
            producer: &self.producer,
            opening: self.plan.opening,
            declared_clock_max_error_ns: self
                .plan
                .options
                .declared_clock_max_error_ns
                .expect("validated clock declaration"),
            coverage,
        })?;
        for entry in &self.samples {
            if let Some(record) = entry.observation_record() {
                raw.json_line(&record)?;
            }
            if let Some(record) = entry.stages_record() {
                raw.json_line(&record)?;
            }
            if let Some(record) = entry.host_content_record() {
                raw.json_line(&record)?;
            }
        }
        raw.json_line(&CutRecord::Summary {
            accepted_ordinal: cutoff,
            closing,
            counts: &self.counts,
            sink: SinkEvidence::from_stats(&stats),
            training,
            coverage,
        })?;
        let source = raw.publish()?;
        if self.profile_sample_count() == 0 {
            return Err(ExportError::SourceOnly {
                path: source.path,
                reason: "no successful observations retained at training cut".into(),
            });
        }
        let result = (|| {
            if host_content {
                return self.write_profile(
                    &profile_path,
                    source.digest,
                    closing.wall_unix_ns,
                    coverage,
                );
            }
            let mut header = Self::profile_file(
                &self.plan,
                &self.producer,
                source.digest,
                closing.wall_unix_ns,
                Vec::new(),
            );
            header.source.measurement_protocol = CUT_COVERAGE.into();
            let mut file = StagedFile::create(&profile_path, limit)?;
            file.json(&BorrowedProfile {
                header,
                samples: &self.samples,
            })?;
            file.publish()
        })();
        let profile = result.map_err(|error: ExportError| ExportError::SourceOnly {
            path: source.path.clone(),
            reason: error.to_string(),
        })?;
        Ok(CostProfileCutReceipt {
            accepted_ordinal: cutoff,
            profile: profile.path,
            profile_sha256: profile.sha256,
            profile_bytes: profile.bytes,
            source: source.path,
            source_sha256: source.sha256,
            source_digest: source.digest,
            source_bytes: source.bytes,
            retained_samples: self.profile_sample_count(),
            raw_retained_observations: self.counts.raw_retained_observations,
        })
    }
}
