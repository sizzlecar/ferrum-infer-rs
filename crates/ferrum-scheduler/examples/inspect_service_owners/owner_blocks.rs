//! Cold diagnostic using the actual source7 canonical collector. No import,
//! activation or synthetic population; records after the first error are raw only.
use ferrum_scheduler::implementations::continuous::cost_profile::{
    record_bytes_v7, CostProfileLoadLimits, StructuredServiceCollectorV7,
    StructuredServiceHeaderV7, StructuredServiceRecordV7,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::Path};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAX_LINE: usize = 8 * 1024 * 1024;
const MAX_EVENTS: usize = 512;

pub(super) fn diagnose(path: &Path, limits: &CostProfileLoadLimits) -> Result<Value> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(
            u64::try_from(limits.max_file_bytes.get())?
                .checked_add(1)
                .ok_or("byte bound overflow")?,
        )
        .read_to_end(&mut bytes)?;
    diagnose_bytes(&bytes, limits)
}

fn diagnose_bytes(bytes: &[u8], limits: &CostProfileLoadLimits) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() {
        return Err("empty or oversized source7 journal".into());
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines.next().ok_or("missing source7 header")?;
    if first.len() > MAX_LINE || first.last() != Some(&b'\n') {
        return Err("incomplete or oversized source7 header".into());
    }
    let header: StructuredServiceHeaderV7 = serde_json::from_slice(first)?;
    if record_bytes_v7(&header)? != first {
        return Err("noncanonical source7 header".into());
    }
    let generation = header.generation;
    let declaration = serde_json::to_value(&header.declaration)?;
    let mut collector = StructuredServiceCollectorV7::new(header, limits.clone())?;
    let mut verified = 0usize;
    let mut raw_records = 0usize;
    let mut stop = None;
    let mut events = Vec::new();
    let mut events_truncated = false;
    let mut raw_kinds = BTreeMap::<String, usize>::new();
    let mut last_audit = serde_json::to_value(collector.audit())?;
    let mut last_receipt = collector.source_receipt();
    let mut last_checkpoint = None;
    let mut footer_complete = false;
    for (index, line) in lines.enumerate() {
        raw_records += 1;
        let raw = if line.len() <= MAX_LINE {
            serde_json::from_slice::<Value>(line).ok()
        } else {
            None
        };
        let kind = raw
            .as_ref()
            .and_then(|r| r["kind"].as_str())
            .unwrap_or("unparsed");
        let kind = match kind {
            "block_open"
            | "completed"
            | "outside_declared_route"
            | "not_submitted"
            | "block_close"
            | "checkpoint"
            | "failed"
            | "footer"
            | "unparsed" => kind,
            _ => "unrecognized",
        };
        *raw_kinds.entry(kind.to_owned()).or_default() += 1;
        // Do not feed later records to a mutated/rejected collector or restart it.
        if stop.is_some() {
            continue;
        }
        let mut numerical_diagnostics = Vec::new();
        let result: Result<StructuredServiceRecordV7> = (|| {
            if line.len() > MAX_LINE || line.last() != Some(&b'\n') {
                return Err("incomplete or oversized source7 record".into());
            }
            let r: StructuredServiceRecordV7 = serde_json::from_slice(line)?;
            if record_bytes_v7(&r)? != line {
                return Err("noncanonical source7 record".into());
            }
            if matches!(&r, StructuredServiceRecordV7::BlockClose { .. }) {
                numerical_diagnostics = collector.diagnose_pending_residuals();
            }
            collector.push(&r)?;
            // A diagnostic becomes reportable only after the original close
            // recomputes and verifies all freezes/certificates/populations.
            if let StructuredServiceRecordV7::BlockClose { freezes, .. } = &r {
                numerical_diagnostics.retain(|diagnostic| {
                    freezes.iter().any(|freeze| {
                        diagnostic["owner_attempt_id"].as_u64() == Some(freeze.owner_attempt_id)
                            && freeze.failure.as_deref() == diagnostic["detail"]["reason"].as_str()
                            && freeze.failure.is_some()
                    })
                });
            }
            Ok(r)
        })();
        let r = match result {
            Ok(r) => r,
            Err(e) => {
                stop = Some(json!({"record":index+1,"line":index+2,"reason":e.to_string()}));
                continue;
            }
        };
        verified += 1;
        last_receipt = collector.source_receipt();
        last_audit = serde_json::to_value(collector.audit())?;
        let event = match &r {
            StructuredServiceRecordV7::Completed { .. }
            | StructuredServiceRecordV7::OutsideDeclaredRoute { .. }
            | StructuredServiceRecordV7::NotSubmitted { .. } => None,
            StructuredServiceRecordV7::BlockClose {
                block,
                offered,
                accepted_fifo_cutoff,
                discoveries,
                freezes,
                route_population,
                ..
            } => Some(json!({
                "kind":"block_close", "block":block,"offered":offered,"accepted_fifo_cutoff":accepted_fifo_cutoff,
                "route_population":route_population,
                "discoveries":discoveries.iter().map(|d|json!({"owner_attempt_id":d.owner_attempt_id,"owner":d.scope.owner,"contract":d.contract})).collect::<Vec<_>>(),
                "freezes":freezes,"numerical_diagnostics":numerical_diagnostics
            })),
            StructuredServiceRecordV7::Checkpoint { .. } => {
                let mut checkpoint = serde_json::to_value(&r)?;
                checkpoint["verified_prefix_bytes"] = json!(last_receipt.0);
                checkpoint["verified_prefix_sha256"] = json!(last_receipt.1);
                checkpoint["qualified_children"] = json!(collector.qualified_children());
                last_checkpoint = Some(checkpoint.clone());
                Some(checkpoint)
            }
            StructuredServiceRecordV7::Footer {
                incomplete_block, ..
            } => {
                footer_complete = !incomplete_block;
                Some(serde_json::to_value(&r)?)
            }
            _ => Some(serde_json::to_value(&r)?),
        };
        if let Some(event) = event {
            if events.len() < MAX_EVENTS {
                events.push(event);
            } else {
                events_truncated = true;
            }
        }
    }
    let all_records_verified = stop.is_none();
    let complete = all_records_verified
        && footer_complete
        && last_audit["closed"] == true
        && last_audit["poisoned"] == false;
    Ok(json!({
        "diagnostic_schema":"ferrum.original-owner-block-audit.v1",
        "scope":"canonical source7 replay diagnostics only; no model import, activation, live publication or SLO acceptance inferred from exit status",
        "source_schema_version":7,"generation":generation,"declaration":declaration,
        "source_bytes":bytes.len(),"source_sha256":<[u8;32]>::from(Sha256::digest(bytes)),
        "verified_records":verified,"raw_records":raw_records,"raw_record_kind_counts":raw_kinds,
        "raw_unverified_records":raw_records-verified,"all_records_verified":all_records_verified,
        "complete_source_verified":complete,"stop":stop,
        "verified_prefix_bytes":last_receipt.0,"verified_prefix_sha256":last_receipt.1,
        "last_verified_audit":last_audit,"last_verified_checkpoint":last_checkpoint,
        "verified_boundary_events":events,"boundary_events_truncated":events_truncated,
        "raw_after_stop_semantics":"counts only; later syntax or a footer cannot repair an invalid original population",
        "live_publication_verified":false,"hardware_functionality_pass":null
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scheduler::implementations::continuous::{
        cost_model::structured_v2::{
            OwnerBlockScheduleV1, StructuredServiceDomainPolicyV1, StructuredSettingsV2,
        },
        cost_profile::{
            ProfileFingerprint, StructuredServiceClockV7, StructuredServiceDeclarationV7,
        },
    };
    fn header() -> StructuredServiceHeaderV7 {
        let mut settings = StructuredSettingsV2::default();
        settings.max_phase_samples = 256 + settings.min_phase_samples - 1;
        StructuredServiceHeaderV7::new(
            [1; 32],
            1,
            ProfileFingerprint {
                model_weights: [1; 32],
                numerical_policy: [2; 32],
                device_runtime: [3; 32],
                execution_config: [4; 32],
            },
            json!({"executable_path":"cpu-diagnostic-fixture"}),
            StructuredServiceClockV7 {
                wall_unix_ns: 1000,
                monotonic_ns: 1,
            },
            StructuredServiceDeclarationV7 {
                schedule: OwnerBlockScheduleV1::new(256, [256; 3], [settings.min_phase_samples; 3])
                    .unwrap(),
                route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
                domain_policy: StructuredServiceDomainPolicyV1::AllOffered,
                nonnegative_envelope: None,
                maximum_window_ns: settings.max_sample_age_ns,
                settings,
                maximum_owners: 8,
                maximum_retained_numeric_bytes: 16 * 1024 * 1024,
                maximum_discovery_bytes: 1024 * 1024,
            },
            16 * 1024 * 1024,
        )
        .unwrap()
    }
    #[test]
    fn source7_diagnostic_empty_complete_footer_is_not_a_qualified_checkpoint() {
        let h = header();
        let mut bytes = record_bytes_v7(&h).unwrap();
        let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
        bytes.extend(
            record_bytes_v7(
                &c.stop(StructuredServiceClockV7 {
                    wall_unix_ns: 1001,
                    monotonic_ns: 2,
                })
                .unwrap(),
            )
            .unwrap(),
        );
        let r = diagnose_bytes(&bytes, &CostProfileLoadLimits::default()).unwrap();
        assert_eq!(r["complete_source_verified"], true);
        assert!(r["last_verified_checkpoint"].is_null());
        assert_eq!(r["live_publication_verified"], false);
    }
    #[test]
    fn source7_diagnostic_never_resumes_after_invalid_assignment() {
        let h = header();
        let mut bytes = record_bytes_v7(&h).unwrap();
        let header_bytes = bytes.len();
        let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
        let valid = c.open_block(2, 0).unwrap();
        let mut invalid = valid.clone();
        if let StructuredServiceRecordV7::BlockOpen { block, .. } = &mut invalid {
            *block = 9;
        }
        bytes.extend(record_bytes_v7(&invalid).unwrap());
        bytes.extend(record_bytes_v7(&valid).unwrap());
        let r = diagnose_bytes(&bytes, &CostProfileLoadLimits::default()).unwrap();
        assert_eq!(r["verified_records"], 0);
        assert_eq!(r["raw_unverified_records"], 2);
        assert_eq!(r["verified_prefix_bytes"], header_bytes);
        assert_eq!(r["last_verified_audit"]["block"], 0);
        assert_eq!(r["complete_source_verified"], false);
        assert!(r["last_verified_checkpoint"].is_null());
    }
}
