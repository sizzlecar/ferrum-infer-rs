use super::*;
use ferrum_engine::continuous_engine::StructuredCalibrationGroupArtifactV2;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{StructuredOwnerKeyV2, MODEL_REVISION_V2},
    cost_profile::{export_structured_profile_v10, ProfileFingerprint},
};
use std::io::Write;

pub(super) fn require_group_exportable(report: &GroupReportV2) -> Result<()> {
    if report.group_failure.is_some() || report.children.is_empty() {
        return Err(FerrumError::config(
            "whole structured group failed; no catalog is qualified",
        ));
    }
    for child in &report.children {
        report::require_exportable(
            child
                .source
                .as_ref()
                .ok_or_else(|| FerrumError::config("group child source is absent"))?,
            true,
        )?;
    }
    Ok(())
}
#[derive(Serialize)]
struct Catalog {
    artifact_type: &'static str,
    schema_version: u32,
    model_revision: &'static str,
    fingerprint: ProfileFingerprint,
    children: Vec<CatalogChild>,
}
#[derive(Serialize)]
struct CatalogChild {
    profile_path: PathBuf,
    profile_sha256: [u8; 32],
    owner: StructuredOwnerKeyV2,
    domain_signature: [u8; 32],
}
pub(super) fn export_and_inspect(
    session: &CalibrationSession,
    capture: &GroupCaptureConfigV2,
    source: StructuredCalibrationGroupArtifactV2,
    report: &mut GroupReportV2,
    artifacts: &mut super::super::super::report::Artifacts,
) -> Result<()> {
    let limits = structured::export_limits(
        &session
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .profile_import,
    );
    if source.children.len() != capture.children.len()
        || report.children.len() != capture.children.len()
    {
        return Err(FerrumError::internal("group export cardinality changed"));
    }
    if capture.shared_source.is_some() {
        return export_shared_and_inspect(session, capture, source, report, artifacts, &limits);
    }
    let mut catalog = Catalog {
        artifact_type: "ferrum.structured-v2-catalog",
        schema_version: 1,
        model_revision: MODEL_REVISION_V2,
        fingerprint: ProfileFingerprint::from(&source.fingerprint),
        children: Vec::with_capacity(source.children.len()),
    };
    for ((configured, source), child) in capture
        .children
        .iter()
        .zip(source.children)
        .zip(&mut report.children)
    {
        let model = source
            .model
            .as_ref()
            .ok_or_else(|| FerrumError::config("group child has no qualified live model"))?;
        let exported = export_structured_profile_v10(
            &source.source_path,
            source.source_sha256,
            &configured.profile,
            configured.declared_source_clock_error_ns,
            &limits,
        )
        .map_err(|e| FerrumError::config(format!("group profile10 export: {e}")))?;
        if exported.parameters_sha256 != model.parameters_signature()
            || model.owner() != &configured.scope.owner
        {
            return Err(FerrumError::internal(
                "group export changed its original parameters/owner",
            ));
        }
        let profile_path = std::fs::canonicalize(&exported.path)
            .map_err(|e| FerrumError::config(format!("resolve exported profile: {e}")))?;
        catalog.children.push(CatalogChild {
            profile_path,
            profile_sha256: exported.file_sha256,
            owner: model.owner().clone(),
            domain_signature: *model.domain_signature(),
        });
        child.exported_profile = Some(exported);
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_child_exported","receipt":child.exported_profile}))?;
    }
    // Match the existing product catalog bound. A failed final inspection can
    // leave diagnostic files, but never a success receipt or installed model.
    let mut bytes = BoundedBytes::new(2 * 1024 * 1024);
    serde_json::to_writer(&mut bytes, &catalog)
        .map_err(|e| FerrumError::config(format!("bounded group catalog: {e}")))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&capture.catalog)
        .map_err(|e| FerrumError::config(format!("create group catalog: {e}")))?;
    file.write_all(&bytes.bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| FerrumError::config(format!("write group catalog: {e}")))?;
    drop(file);
    let receipt = session.inspect_structured_cost_profile_v2(&capture.catalog)?;
    let v2 = receipt
        .structured_whole_wave_v2
        .as_ref()
        .ok_or_else(|| FerrumError::internal("group loader returned no V2 receipt"))?;
    if v2.artifact_kind != ferrum_types::SloStructuredArtifactKindV2::CatalogV1
        || v2.child_count != capture.children.len()
    {
        return Err(FerrumError::internal(
            "group loader did not verify every catalog child",
        ));
    }
    report.verified_catalog = Some(receipt);
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_catalog_verified","receipt":report.verified_catalog}))?;
    Ok(())
}
/// Stop during serialization, including a single large escaped string; do not
/// first build an unbounded JSON value/vector and inspect its final length.
struct BoundedBytes {
    bytes: Vec<u8>,
    limit: usize,
}
impl BoundedBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }
}
impl Write for BoundedBytes {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(buf.len())
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| std::io::Error::other("catalog metadata byte bound exceeded"))?;
        self.bytes
            .try_reserve(next - self.bytes.len())
            .map_err(std::io::Error::other)?;
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[cfg(test)]
#[test]
fn structured_group_cli_catalog_serialization_is_bounded_during_escaped_text() {
    let mut bytes = BoundedBytes::new(32);
    assert!(serde_json::to_writer(&mut bytes, &"\"界\n".repeat(1024)).is_err());
    assert!(bytes.bytes.len() <= 32);
    let mut bytes = BoundedBytes::new(32);
    serde_json::to_writer(&mut bytes, &"\"界\n").unwrap();
    assert_eq!(
        serde_json::from_slice::<String>(&bytes.bytes).unwrap(),
        "\"界\n"
    );
}

fn export_shared_and_inspect(
    session: &CalibrationSession,
    capture: &GroupCaptureConfigV2,
    source: StructuredCalibrationGroupArtifactV2,
    report: &mut GroupReportV2,
    artifacts: &mut super::super::super::report::Artifacts,
    limits: &ferrum_scheduler::implementations::continuous::cost_profile::CostProfileLoadLimits,
) -> Result<()> {
    let first = source
        .children
        .first()
        .ok_or_else(|| FerrumError::config("empty shared source"))?;
    if source.failure.is_some()
        || source.children.iter().any(|c| {
            c.source_path != first.source_path
                || c.source_sha256 != first.source_sha256
                || c.source_bytes != first.source_bytes
                || c.failure.is_some()
                || c.model.is_none()
        })
    {
        return Err(FerrumError::config(
            "shared group did not close one qualified immutable source",
        ));
    }
    let errors = capture
        .children
        .iter()
        .map(|c| c.declared_source_clock_error_ns)
        .collect::<Vec<_>>();
    let exported =
        ferrum_scheduler::implementations::continuous::cost_profile::export_structured_profile_v11(
            &first.source_path,
            first.source_sha256,
            &capture.catalog,
            &errors,
            limits,
        )
        .map_err(|e| FerrumError::config(format!("group profile11 export: {e}")))?;
    if exported.children.len() != source.children.len() {
        return Err(FerrumError::internal("shared export child count changed"));
    }
    for (((configured, original), receipt), child) in capture
        .children
        .iter()
        .zip(&source.children)
        .zip(&exported.children)
        .zip(&mut report.children)
    {
        let model = original.model.as_ref().unwrap();
        if receipt.parameters_sha256 != model.parameters_signature()
            || model.owner() != &configured.scope.owner
        {
            return Err(FerrumError::internal(
                "shared export changed original child parameters/owner",
            ));
        }
        child.exported_profile = Some(receipt.clone());
    }
    report.exported_shared_profile = Some(exported);
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_shared_profile_exported","receipt":report.exported_shared_profile}))?;
    let receipt = session.inspect_structured_cost_profile_v2(&capture.catalog)?;
    let v2 = receipt
        .structured_whole_wave_v2
        .as_ref()
        .ok_or_else(|| FerrumError::internal("shared loader returned no V2 receipt"))?;
    if v2.artifact_kind != ferrum_types::SloStructuredArtifactKindV2::SharedCatalogV11
        || v2.child_count != capture.children.len()
    {
        return Err(FerrumError::internal(
            "shared loader did not verify every declared child",
        ));
    }
    report.verified_catalog = Some(receipt);
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_shared_catalog_verified","receipt":report.verified_catalog}))?;
    Ok(())
}
