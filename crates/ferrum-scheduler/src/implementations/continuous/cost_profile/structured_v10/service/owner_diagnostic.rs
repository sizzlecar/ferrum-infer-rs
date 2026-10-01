//! Read-only source diagnostics. No model, execution authority, or successful
//! import is returned, including when a failed source has a valid prefix.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOwnerDifferenceV1 {
    Rows,
    Role,
    Product,
    Readback,
    ProviderTemplate,
    AlgorithmDomain,
    InstalledPolicy,
}

#[derive(Debug, Serialize)]
pub struct StructuredOutsideOwnerDiagnosticV1 {
    pub phase: StructuredPhaseV2,
    pub ticket: u64,
    pub fifo: u64,
    pub call_id: u64,
    pub actual_owner: StructuredOwnerKeyV2,
    /// Fewest unequal fields among the seven owner fields; ties use the first
    /// declaration index. This is a diagnostic distance, never compatibility.
    pub nearest_declaration_index: usize,
    pub nearest_declared_owner: StructuredOwnerKeyV2,
    pub differences: Vec<StructuredOwnerDifferenceV1>,
}

/// Read-only comparison of an original verified wave with the new numerical
/// interpretation. It neither rewrites old sources nor qualifies a model.
#[derive(Debug, Serialize)]
pub struct StructuredNumericalFamilyProjectionV1 {
    pub original_owner: StructuredOwnerKeyV2,
    pub projected_owner: StructuredOwnerKeyV2,
    pub projected_domain: [u8; 32],
    pub first_ticket: u64,
    pub first_phase: StructuredPhaseV2,
    pub verified_waves: usize,
}

#[derive(Debug, Serialize)]
pub struct StructuredOwnerReplayFailureV1 {
    /// One-based record number after the header.
    pub record: usize,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct StructuredServiceOwnerDiagnosticV1 {
    pub source_sha256: [u8; 32],
    pub source_bytes: usize,
    pub generation: u64,
    pub declared_owners: usize,
    pub verified_records: usize,
    /// A successful original footer and every preceding record were verified.
    /// Even true does not constitute a profile import or execution permission.
    pub complete_source_verified: bool,
    pub stop: Option<StructuredOwnerReplayFailureV1>,
    /// At most one original validated OutsideCatalog wave per phase.
    pub first_outside_by_phase: [Option<StructuredOutsideOwnerDiagnosticV1>; 3],
    pub route_population: [StructuredServiceRouteCountsV1; 3],
    /// At most 128 distinct original/new identity pairs, after the unmodified
    /// original validator. Physical-domain-less legacy sources have no entry.
    pub numerical_family_projections: Vec<StructuredNumericalFamilyProjectionV1>,
    pub numerical_family_projection_truncated: bool,
}

fn differences(
    actual: &StructuredOwnerKeyV2,
    declared: &StructuredOwnerKeyV2,
) -> Vec<StructuredOwnerDifferenceV1> {
    use StructuredOwnerDifferenceV1::*;
    [
        (actual.rows != declared.rows, Rows),
        (actual.role != declared.role, Role),
        (actual.product != declared.product, Product),
        (actual.readback != declared.readback, Readback),
        (
            actual.provider_template != declared.provider_template,
            ProviderTemplate,
        ),
        (
            actual.algorithm_domain != declared.algorithm_domain,
            AlgorithmDomain,
        ),
        (
            actual.installed_policy != declared.installed_policy,
            InstalledPolicy,
        ),
    ]
    .into_iter()
    .filter_map(|(different, field)| different.then_some(field))
    .collect()
}

/// Inspect a bounded original source6, including the verified prefix of a
/// failed diagnostic source. The original collector validates every record,
/// freeze certificate, clock, private-receipt replay and population boundary.
pub fn diagnose_structured_service_owners_v6(
    path: &Path,
    limits: &CostProfileLoadLimits,
) -> Result<StructuredServiceOwnerDiagnosticV1, CostProfileError> {
    limits.validate()?;
    let bytes = read_bounded(path, limits.max_file_bytes.get())?;
    diagnose_bytes(&bytes, limits)
}

pub(super) fn diagnose_bytes(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<StructuredServiceOwnerDiagnosticV1, CostProfileError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() {
        return Err(invalid("source6 empty/oversized diagnostic stream"));
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines
        .next()
        .ok_or_else(|| invalid("source6 missing header"))?;
    if first.len() > 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit("source6 header line limit"));
    }
    let header: StructuredServiceHeaderV6 = serde_json::from_slice(first)?;
    if replay::record_bytes(&header)? != first {
        return Err(invalid("source6 requires canonical header encoding"));
    }
    let mut collector = StructuredServiceCollectorV6::new(header.clone(), limits.clone())?;
    let mut report = StructuredServiceOwnerDiagnosticV1 {
        source_sha256: Sha256::digest(bytes).into(),
        source_bytes: bytes.len(),
        generation: header.generation,
        declared_owners: header.declaration.scopes.len(),
        verified_records: 0,
        complete_source_verified: false,
        stop: None,
        first_outside_by_phase: [None, None, None],
        route_population: [StructuredServiceRouteCountsV1::default(); 3],
        numerical_family_projections: Vec::new(),
        numerical_family_projection_truncated: false,
    };
    // The second cold-path projection borrows each original DTO. It avoids a
    // diagnostic hook, retained inputs, or new work in the live collector.
    // Cross-wave frontiers (including OutsideDeclaredRoute) were already checked
    // by collector.push; the extra projection needs only this wave's facts.
    let mut opened_at_ns = None;
    for (index, line) in lines.enumerate() {
        let outcome = (|| {
            if line.len() > 8 * 1024 * 1024 {
                return Err(CostProfileError::Limit("source6 record line limit"));
            }
            let record: StructuredServiceRecordV6 = serde_json::from_slice(line)?;
            if replay::record_bytes(&record)? != line {
                return Err(invalid("source6 requires canonical record encoding"));
            }
            collector.push(&record)?;
            match &record {
                StructuredServiceRecordV6::PhaseOpen {
                    opened_at_ns: at, ..
                } => {
                    opened_at_ns = Some(*at);
                }
                StructuredServiceRecordV6::Completed { wave } => {
                    let (input, _, _) = physical::validate(
                        &header,
                        opened_at_ns.ok_or_else(|| invalid("diagnostic phase unavailable"))?,
                        wave,
                        &mut physical::Frontiers::default(),
                    )?;
                    let actual = input.owner();
                    if let Some((owner, domain)) = input.cost_template_identity(
                        crate::implementations::continuous::cost_model::structured_v2::StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1) {
                        if let Some(pair) = report.numerical_family_projections.iter_mut()
                            .find(|p| &p.original_owner == actual && &p.projected_owner == owner
                                && &p.projected_domain == domain) {
                            pair.verified_waves += 1;
                        } else if report.numerical_family_projections.len() < 128 {
                            report.numerical_family_projections.push(StructuredNumericalFamilyProjectionV1 {
                                original_owner: actual.clone(), projected_owner: owner.clone(),
                                projected_domain: *domain, first_ticket: wave.ticket,
                                first_phase: wave.phase, verified_waves: 1,
                            });
                        } else {
                            report.numerical_family_projection_truncated = true;
                        }
                    }
                    if report.first_outside_by_phase[phase_index(wave.phase)].is_none()
                        && !header.declaration.scopes.iter().any(|s| &s.owner == actual)
                    {
                        let (child, difference) = header
                            .declaration
                            .scopes
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (i, differences(actual, &s.owner)))
                            .min_by_key(|(i, fields)| (fields.len(), *i))
                            .ok_or_else(|| invalid("diagnostic declaration has no owners"))?;
                        report.first_outside_by_phase[phase_index(wave.phase)] =
                            Some(StructuredOutsideOwnerDiagnosticV1 {
                                phase: wave.phase,
                                ticket: wave.ticket,
                                fifo: wave.fifo,
                                call_id: wave.host_stages.call_id,
                                actual_owner: actual.clone(),
                                nearest_declaration_index: child,
                                nearest_declared_owner: header.declaration.scopes[child]
                                    .owner
                                    .clone(),
                                differences: difference,
                            });
                    }
                }
                StructuredServiceRecordV6::Footer { .. } => {
                    report.complete_source_verified = true;
                }
                _ => {}
            }
            Ok::<(), CostProfileError>(())
        })();
        if let Err(error) = outcome {
            report.complete_source_verified = false;
            report.stop = Some(StructuredOwnerReplayFailureV1 {
                record: index + 1,
                reason: error.to_string(),
            });
            break;
        }
        report.verified_records += 1;
        report.route_population = collector.route_population_counts();
    }
    if report.stop.is_none() && !report.complete_source_verified {
        report.stop = Some(StructuredOwnerReplayFailureV1 {
            record: report.verified_records + 1,
            reason: "source6 missing successful footer".into(),
        });
    }
    Ok(report)
}
