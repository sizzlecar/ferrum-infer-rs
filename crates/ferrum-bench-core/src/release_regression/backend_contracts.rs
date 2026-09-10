//! Exact device lifecycle assertions. GPU tests require their device and fail
//! when it is unavailable; CPU execution cannot stand in for an accelerator.
use super::contracts::{verify_contract_report, ContractGroup, ContractReport, ContractTest};
use super::{Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, ObligationScope};
use serde::{Deserialize, Serialize};

const PATH: &str = "production-plan-runtime";

pub fn submission_scope(backend: Backend) -> ObligationScope {
    ObligationScope::ExecutionPath {
        backend,
        execution_path: PATH.into(),
    }
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    }
}

fn checker_id(backend: Backend) -> String {
    format!(
        "backend-contract.{}.vnext-submission",
        backend_name(backend)
    )
}

pub fn contract_groups(backend: Backend) -> Vec<ContractGroup> {
    let (module, names): (&str, &[&str]) = match backend {
        Backend::Cpu => (
            "backend::cpu::vnext_runtime::tests",
            &[
                "upload_owns_source_bytes_and_is_charged_until_submission_finishes",
                "failure_after_write_returns_failed_quiescent_fence_and_prevents_retry",
                "foreign_command_is_rejected_before_any_earlier_command_runs",
            ],
        ),
        Backend::Metal => (
            "backend::metal::vnext_runtime::tests",
            &[
                "ordered_transfer_batch_is_async_and_readback_is_exact",
                "encode_failure_is_definitely_not_submitted_and_restores_stream",
            ],
        ),
        Backend::Cuda => (
            "backend::cuda::vnext_runtime::on_demand_tests",
            &[
                "on_demand_capture_executes_state_once_and_eviction_preserves_fallback",
                "on_demand_capture_rejection_keeps_ordinary_execution_available",
                "on_demand_logical_catalog_is_bounded_and_stale_references_miss_before_launch",
            ],
        ),
    };
    vec![ContractGroup {
        id: checker_id(backend),
        behavior: Behavior::SubmissionCompletion,
        entrypoints: Vec::new(),
        tests: names
            .iter()
            .map(|name| ContractTest {
                package: "ferrum-kernels".into(),
                target: "ferrum_kernels".into(),
                kind: "lib".into(),
                name: format!("{module}::{name}"),
            })
            .collect(),
    }]
}

/// One representative anchors a model-independent device execution route. No
/// descriptor is generated for legacy execution or an unimplemented backend.
pub fn check_descriptors(targets: &[ExecutionTarget]) -> Vec<CheckDescriptor> {
    [Backend::Cpu, Backend::Metal, Backend::Cuda]
        .into_iter()
        .filter_map(|backend| {
            let target = targets
                .iter()
                .filter(|t| t.backend == backend && t.execution_path == PATH)
                .min_by(|a, b| {
                    a.architecture
                        .cmp(&b.architecture)
                        .then_with(|| a.precision.cmp(&b.precision))
                        .then_with(|| format!("{:?}", a.protocol).cmp(&format!("{:?}", b.protocol)))
                });
            target.map(|target| CheckDescriptor {
                id: checker_id(backend),
                behavior: Behavior::SubmissionCompletion,
                layer: EvidenceLayer::BackendNumerics,
                entrypoints: Vec::new(),
                target: Some(target.clone()),
            })
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendContractReport {
    pub schema_version: u32,
    pub backend: Backend,
    pub execution: ContractReport,
}

pub struct VerifiedBackendContracts {
    backend: Backend,
}
impl VerifiedBackendContracts {
    pub fn covers(&self, obligation: &super::Obligation) -> bool {
        obligation.layer == EvidenceLayer::BackendNumerics
            && obligation.behavior == Behavior::SubmissionCompletion
            && obligation.scope == submission_scope(self.backend)
            && obligation.entrypoints.is_empty()
            && obligation.checkers == [checker_id(self.backend)]
    }
}

/// CI provenance is verified by the delivery consumer, independently of this
/// report. All registered assertions must have actually executed successfully.
pub fn verify_report(
    expected: Backend,
    report: &BackendContractReport,
) -> Result<VerifiedBackendContracts, String> {
    if report.schema_version != 1 || report.backend != expected {
        return Err("backend contract report has the wrong schema or device".into());
    }
    verify_contract_report(&contract_groups(expected), &report.execution)
        .map_err(|issues| issues.join("; "))?;
    Ok(VerifiedBackendContracts { backend: expected })
}

#[cfg(test)]
mod tests {
    use super::super::contracts::{
        CommandObservation, CommandStatus, ContractGroupResult, ContractStatus, ContractTestResult,
        HarnessArtifact,
    };
    use super::*;

    fn report_fixture(backend: Backend) -> BackendContractReport {
        let observation = CommandObservation {
            status: CommandStatus::Passed,
            exit_code: Some(0),
            elapsed_ms: 1,
            stdout: "stdout.log".into(),
            stderr: "stderr.log".into(),
            error: None,
        };
        BackendContractReport {
            schema_version: 1,
            backend,
            execution: ContractReport {
                schema_version: 1,
                status: ContractStatus::Passed,
                groups: contract_groups(backend)
                    .into_iter()
                    .map(|group| ContractGroupResult {
                        id: group.id,
                        behavior: group.behavior,
                        status: ContractStatus::Passed,
                        tests: group
                            .tests
                            .into_iter()
                            .map(|binding| ContractTestResult {
                                artifact: Some(HarnessArtifact {
                                    package: binding.package.clone(),
                                    package_id: binding.package.clone(),
                                    target: binding.target.clone(),
                                    kind: binding.kind.clone(),
                                    executable: "device-tests".into(),
                                    manifest_path: "Cargo.toml".into(),
                                }),
                                binding,
                                registered: true,
                                listing: Some(observation.clone()),
                                execution: Some(observation.clone()),
                                error: None,
                            })
                            .collect(),
                    })
                    .collect(),
            },
        }
    }

    #[test]
    fn device_reports_reject_missing_execution_and_cross_backend_substitution() {
        for backend in [Backend::Cpu, Backend::Metal, Backend::Cuda] {
            let report = report_fixture(backend);
            verify_report(backend, &report).unwrap();
            let mut changed = report.clone();
            changed.execution.groups[0].tests[0].execution = None;
            assert!(verify_report(backend, &changed).is_err());
            changed = report.clone();
            changed.execution.groups[0].tests[0].registered = false;
            assert!(verify_report(backend, &changed).is_err());
            changed = report.clone();
            changed.execution.groups[0].tests[0]
                .execution
                .as_mut()
                .unwrap()
                .exit_code = Some(1);
            assert!(verify_report(backend, &changed).is_err());
            changed = report.clone();
            changed.execution.groups[0].tests.clear();
            assert!(verify_report(backend, &changed).is_err());
            changed = report.clone();
            changed.schema_version += 1;
            assert!(verify_report(backend, &changed).is_err());
            for other in [Backend::Cpu, Backend::Metal, Backend::Cuda] {
                if other != backend {
                    assert!(verify_report(other, &report).is_err());
                    changed = report.clone();
                    changed.backend = other;
                    assert!(verify_report(other, &changed).is_err());
                }
            }
        }
    }

    #[test]
    fn device_receipt_cannot_be_rebound_to_a_different_route_or_behavior() {
        let receipt = VerifiedBackendContracts {
            backend: Backend::Cuda,
        };
        let mut obligation = super::super::Obligation {
            behavior: Behavior::SubmissionCompletion,
            layer: EvidenceLayer::BackendNumerics,
            scope: ObligationScope::ExecutionPath {
                backend: Backend::Cuda,
                execution_path: PATH.into(),
            },
            checkers: vec![checker_id(Backend::Cuda)],
            entrypoints: Vec::new(),
            reason: "test".into(),
        };
        assert!(receipt.covers(&obligation));
        obligation.scope = ObligationScope::ExecutionPath {
            backend: Backend::Metal,
            execution_path: PATH.into(),
        };
        assert!(!receipt.covers(&obligation));
        obligation.scope = ObligationScope::ExecutionPath {
            backend: Backend::Cuda,
            execution_path: "legacy-model-executor".into(),
        };
        assert!(!receipt.covers(&obligation));
        obligation.scope = ObligationScope::ExecutionPath {
            backend: Backend::Cuda,
            execution_path: PATH.into(),
        };
        obligation.behavior = Behavior::KernelNumerics;
        assert!(!receipt.covers(&obligation));
    }
}
