//! Exact device lifecycle and operator assertions. GPU tests require their device and fail
//! when it is unavailable; CPU execution cannot stand in for an accelerator.
use super::contracts::{verify_contract_report, ContractGroup, ContractReport, ContractTest};
use super::{Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, ObligationScope};
use serde::{Deserialize, Serialize};
mod compatibility;
#[path = "backend_contracts/numerical.rs"]
mod numerical;

const PATH: &str = "production-plan-runtime";
const LEGACY_CUDA_CHECK: &str = "backend-contract.cuda.legacy-submission";

fn legacy_cuda_scope() -> ObligationScope {
    ObligationScope::ExecutionPath {
        backend: Backend::Cuda,
        execution_path: super::submission::LEGACY_EXECUTION_PATH.into(),
    }
}

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
                "profiling_preserves_cpu_execution_and_marks_device_clocks_unavailable",
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
    let mut groups = vec![ContractGroup {
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
    }];
    if backend == Backend::Cuda {
        groups.push(ContractGroup {
            id: LEGACY_CUDA_CHECK.into(),
            behavior: Behavior::SubmissionCompletion,
            entrypoints: Vec::new(),
            tests: vec![ContractTest {
                package: "ferrum-kernels".into(),
                target: "ferrum_kernels".into(),
                kind: "lib".into(),
                name: "backend::cuda::submission_tests::legacy_stream_preserves_upload_compute_copy_order_and_context_reuse".into(),
            }],
        });
    }
    groups.extend(numerical::groups(backend));
    groups.extend(compatibility::groups(backend));
    if backend == Backend::Cuda {
        // Dense Marlin coverage requires these matrices and the numerical
        // group's F16 providers, workspace lifecycle and projection stitching.
        let tests: Vec<_> = [
            "gemma4_symmetric_compressed_tensors_w4a16_matches_two_shapes_and_batches_1_4",
            "qwen38_compressed_tensors_w4a16_matches_cpu_reference_for_four_fixed_fixtures",
            "qwen38_block_fp8_marlin_matches_four_locked_quality_vector_cases",
        ]
        .into_iter()
        .map(|name| ContractTest {
            package: "ferrum-kernels".into(),
            target: "compressed_tensors_marlin_eq".into(),
            kind: "test".into(),
            name: name.into(),
        })
        .collect();
        for (suffix, behavior) in [
            ("numerics", Behavior::KernelNumerics),
            ("boundaries", Behavior::KernelBoundaries),
        ] {
            groups.push(ContractGroup {
                id: format!("backend-contract.cuda.marlin-matrix-{suffix}"),
                behavior,
                entrypoints: Vec::new(),
                tests: tests.clone(),
            });
        }
    }
    groups
}

/// One representative anchors a model-independent device execution route.
/// Numerical descriptors additionally require their exact supported target.
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
        .chain(
            targets
                .iter()
                .filter(|target| {
                    target.backend == Backend::Cuda
                        && target.execution_path == super::submission::LEGACY_EXECUTION_PATH
                })
                .min_by_key(|target| (&target.architecture, &target.precision))
                .map(|target| CheckDescriptor {
                    id: LEGACY_CUDA_CHECK.into(),
                    behavior: Behavior::SubmissionCompletion,
                    layer: EvidenceLayer::BackendNumerics,
                    target: Some(target.clone()),
                    entrypoints: Vec::new(),
                }),
        )
        .chain(numerical::descriptors(targets))
        .chain(compatibility::descriptors(targets))
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
        expected_checker(self.backend, obligation).is_some_and(|id| obligation.checkers == [id])
    }
}

fn expected_checker(backend: Backend, obligation: &super::Obligation) -> Option<String> {
    if obligation.layer != EvidenceLayer::BackendNumerics || !obligation.entrypoints.is_empty() {
        return None;
    }
    if obligation.behavior == Behavior::SubmissionCompletion
        && obligation.scope == submission_scope(backend)
    {
        Some(checker_id(backend))
    } else if backend == Backend::Cuda
        && obligation.behavior == Behavior::SubmissionCompletion
        && obligation.scope == legacy_cuda_scope()
    {
        Some(LEGACY_CUDA_CHECK.into())
    } else {
        numerical::expected_checker(backend, obligation)
            .or_else(|| compatibility::expected_checker(backend, obligation))
    }
}

pub fn required_backend(obligation: &super::Obligation) -> Option<Backend> {
    [Backend::Cpu, Backend::Metal, Backend::Cuda]
        .into_iter()
        .find(|backend| expected_checker(*backend, obligation).is_some())
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

    #[test]
    fn compatibility_receipts_require_route_assertions_and_exact_declared_formats() {
        let targets = [
            (
                "gemma4_dense",
                Backend::Cuda,
                "compressed-tensors-w4a16",
                ferrum_types::ModelOutputProtocol::GemmaThought,
                PATH,
            ),
            (
                "gpt_oss_moe",
                Backend::Cuda,
                "mxfp4",
                ferrum_types::ModelOutputProtocol::HarmonyGptOss,
                PATH,
            ),
            (
                "qwen3_5_hybrid_moe",
                Backend::Cuda,
                "block-fp8-e4m3",
                ferrum_types::ModelOutputProtocol::Text,
                PATH,
            ),
            (
                "qwen3_attention_moe",
                Backend::Cuda,
                "gptq-int4",
                ferrum_types::ModelOutputProtocol::Text,
                PATH,
            ),
            (
                "qwen3_5_hybrid_moe",
                Backend::Metal,
                "gguf-q4_k_s",
                ferrum_types::ModelOutputProtocol::Text,
                PATH,
            ),
            (
                "qwen3_attention_moe",
                Backend::Metal,
                "gguf-q4_k_m",
                ferrum_types::ModelOutputProtocol::Text,
                PATH,
            ),
            (
                "llama_dense",
                Backend::Metal,
                "gguf-q4_k_m",
                ferrum_types::ModelOutputProtocol::Text,
                super::super::submission::LEGACY_EXECUTION_PATH,
            ),
            (
                "llama_dense",
                Backend::Cuda,
                "safetensors-bf16",
                ferrum_types::ModelOutputProtocol::Text,
                super::super::submission::LEGACY_EXECUTION_PATH,
            ),
        ];
        for (architecture, backend, precision, protocol, execution_path) in targets {
            let target = ExecutionTarget {
                architecture: architecture.into(),
                backend,
                precision: precision.into(),
                protocol,
                execution_path: execution_path.into(),
            };
            let report = report_fixture(backend);
            let receipt = verify_report(backend, &report).unwrap();
            let descriptors = compatibility::descriptors(std::slice::from_ref(&target));
            for behavior in [Behavior::KernelNumerics, Behavior::KernelBoundaries] {
                let descriptor = descriptors
                    .iter()
                    .find(|item| item.behavior == behavior)
                    .expect("existing route has an explicit operator binding");
                let mut obligation = super::super::Obligation {
                    behavior: descriptor.behavior,
                    layer: descriptor.layer,
                    scope: ObligationScope::Target {
                        target: target.clone(),
                    },
                    checkers: vec![descriptor.id.clone()],
                    entrypoints: Vec::new(),
                    reason: "existing execution route".into(),
                };
                assert!(receipt.covers(&obligation));
                let mut unsupported = target.clone();
                unsupported.precision = "unregistered-weight-format".into();
                obligation.scope = ObligationScope::Target {
                    target: unsupported,
                };
                assert!(!receipt.covers(&obligation));
            }
            let mut shared_only = report;
            shared_only
                .execution
                .groups
                .retain(|group| !group.id.contains("existing-routes"));
            assert!(verify_report(backend, &shared_only).is_err());
        }
    }

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

    #[test]
    fn native_numerical_receipts_require_the_operator_suite_and_exact_target_scope() {
        let backend = Backend::Cuda;
        let target = ExecutionTarget {
            architecture: "qwen3_5_dense_hybrid".into(),
            protocol: ferrum_types::ModelOutputProtocol::Text,
            precision: "gguf-mixed-4bit".into(),
            backend,
            execution_path: PATH.into(),
        };
        let descriptors = check_descriptors(std::slice::from_ref(&target));
        let report = report_fixture(backend);
        let receipt = verify_report(backend, &report).unwrap();
        for behavior in [Behavior::KernelNumerics, Behavior::KernelBoundaries] {
            let descriptor = descriptors.iter().find(|d| d.behavior == behavior).unwrap();
            let mut obligation = super::super::Obligation {
                behavior,
                layer: EvidenceLayer::BackendNumerics,
                scope: ObligationScope::Target {
                    target: target.clone(),
                },
                checkers: vec![descriptor.id.clone()],
                entrypoints: Vec::new(),
                reason: "native operator evidence".into(),
            };
            assert_eq!(required_backend(&obligation), Some(backend));
            assert!(receipt.covers(&obligation));
            let mut other = target.clone();
            other.precision = "gptq-int4".into();
            obligation.scope = ObligationScope::Target { target: other };
            assert!(!receipt.covers(&obligation));
            assert_eq!(required_backend(&obligation), None);
            obligation.scope = ObligationScope::Target {
                target: target.clone(),
            };
            obligation.entrypoints.push(super::super::Entrypoint::Run);
            assert!(!receipt.covers(&obligation));
        }
        let mut submission_only = report.clone();
        submission_only
            .execution
            .groups
            .retain(|group| group.behavior == Behavior::SubmissionCompletion);
        assert!(verify_report(backend, &submission_only).is_err());
        let mut missing_oracle = report;
        missing_oracle
            .execution
            .groups
            .iter_mut()
            .find(|group| group.behavior == Behavior::KernelNumerics)
            .unwrap()
            .tests[0]
            .execution = None;
        assert!(verify_report(backend, &missing_oracle).is_err());
    }

    #[test]
    fn legacy_cuda_requires_its_own_execution_and_cannot_certify_operator_numerics() {
        let target = ExecutionTarget {
            architecture: "llama_dense".into(),
            protocol: ferrum_types::ModelOutputProtocol::Text,
            precision: "safetensors-bf16".into(),
            backend: Backend::Cuda,
            execution_path: super::super::submission::LEGACY_EXECUTION_PATH.into(),
        };
        let descriptors = check_descriptors(&[target]);
        let descriptor = descriptors
            .iter()
            .find(|d| d.id == LEGACY_CUDA_CHECK)
            .unwrap();
        let mut obligation = super::super::Obligation {
            behavior: descriptor.behavior,
            layer: descriptor.layer,
            scope: legacy_cuda_scope(),
            checkers: vec![descriptor.id.clone()],
            entrypoints: Vec::new(),
            reason: "legacy stream lifecycle".into(),
        };
        let report = report_fixture(Backend::Cuda);
        let receipt = verify_report(Backend::Cuda, &report).unwrap();
        assert!(receipt.covers(&obligation));
        assert_eq!(required_backend(&obligation), Some(Backend::Cuda));
        obligation.behavior = Behavior::KernelNumerics;
        assert!(!receipt.covers(&obligation));
        obligation.behavior = Behavior::SubmissionCompletion;
        obligation.checkers = vec![checker_id(Backend::Cuda)];
        assert!(!receipt.covers(&obligation));
        let mut vnext_only = report;
        vnext_only
            .execution
            .groups
            .retain(|group| group.id != LEGACY_CUDA_CHECK);
        assert!(verify_report(Backend::Cuda, &vnext_only).is_err());
        assert!(!contract_groups(Backend::Cpu)
            .iter()
            .any(|group| group.id == LEGACY_CUDA_CHECK));
        assert!(!contract_groups(Backend::Metal)
            .iter()
            .any(|group| group.id == LEGACY_CUDA_CHECK));
    }

    #[test]
    fn cuda_dense_float_coverage_requires_the_f16_provider_assertions() {
        let target = ExecutionTarget {
            architecture: "qwen3_5_dense_hybrid".into(),
            protocol: ferrum_types::ModelOutputProtocol::Text,
            precision: "safetensors-bf16-f32".into(),
            backend: Backend::Cuda,
            execution_path: PATH.into(),
        };
        let report = report_fixture(Backend::Cuda);
        let receipt = verify_report(Backend::Cuda, &report).unwrap();
        let descriptors = check_descriptors(std::slice::from_ref(&target));
        for behavior in [Behavior::KernelNumerics, Behavior::KernelBoundaries] {
            let descriptor = descriptors.iter().find(|d| d.behavior == behavior).unwrap();
            let obligation = super::super::Obligation {
                behavior: descriptor.behavior,
                layer: descriptor.layer,
                scope: ObligationScope::Target {
                    target: target.clone(),
                },
                checkers: vec![descriptor.id.clone()],
                entrypoints: Vec::new(),
                reason: "dense F16 provider evidence".into(),
            };
            assert!(receipt.covers(&obligation));
        }
        let mut native_only = report;
        for group in &mut native_only.execution.groups {
            group
                .tests
                .retain(|test| !test.binding.name.contains("::f16_tests::"));
        }
        assert!(verify_report(Backend::Cuda, &native_only).is_err());
    }

    #[test]
    fn dense_marlin_requires_matrix_and_provider_evidence_without_covering_moe() {
        let report = report_fixture(Backend::Cuda);
        let receipt = verify_report(Backend::Cuda, &report).unwrap();
        for precision in ["compressed-tensors-int4", "block-fp8-e4m3"] {
            let target = ExecutionTarget {
                architecture: "qwen3_5_dense_hybrid".into(),
                protocol: ferrum_types::ModelOutputProtocol::Text,
                precision: precision.into(),
                backend: Backend::Cuda,
                execution_path: PATH.into(),
            };
            let descriptors = check_descriptors(std::slice::from_ref(&target));
            for behavior in [Behavior::KernelNumerics, Behavior::KernelBoundaries] {
                let descriptor = descriptors.iter().find(|d| d.behavior == behavior).unwrap();
                let mut obligation = super::super::Obligation {
                    behavior,
                    layer: EvidenceLayer::BackendNumerics,
                    scope: ObligationScope::Target {
                        target: target.clone(),
                    },
                    checkers: vec![descriptor.id.clone()],
                    entrypoints: Vec::new(),
                    reason: "dense Marlin operator chain".into(),
                };
                assert!(receipt.covers(&obligation));
                let mut moe = target.clone();
                moe.architecture = "qwen3_5_hybrid_moe".into();
                obligation.scope = ObligationScope::Target { target: moe };
                assert!(!receipt.covers(&obligation));
                assert_eq!(
                    required_backend(&obligation),
                    (precision == "block-fp8-e4m3").then_some(Backend::Cuda)
                );
            }
        }
        let mut no_matrix = report.clone();
        no_matrix
            .execution
            .groups
            .retain(|g| !g.id.contains("marlin-matrix"));
        assert!(verify_report(Backend::Cuda, &no_matrix).is_err());
        for missing in ["::marlin_tests::", "::projection_stitch_tests::"] {
            let mut incomplete = report.clone();
            for group in &mut incomplete.execution.groups {
                group
                    .tests
                    .retain(|test| !test.binding.name.contains(missing));
            }
            assert!(verify_report(Backend::Cuda, &incomplete).is_err());
        }
    }
}
