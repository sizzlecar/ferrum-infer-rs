//! Exact device lifecycle and operator assertions. GPU tests require their device and fail
//! when it is unavailable; CPU execution cannot stand in for an accelerator.
use super::contracts::{verify_contract_report, ContractGroup, ContractReport, ContractTest};
use super::{Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, ObligationScope};
use serde::{Deserialize, Serialize};
mod compatibility;
#[path = "backend_contracts/numerical.rs"]
mod numerical;

const PATH: &str = "production-plan-runtime";

fn legacy_scope(backend: Backend) -> ObligationScope {
    ObligationScope::ExecutionPath {
        backend,
        execution_path: super::submission::LEGACY_EXECUTION_PATH.into(),
    }
}

fn legacy_checker_id(backend: Backend) -> String {
    format!(
        "backend-contract.{}.legacy-submission",
        backend_name(backend)
    )
}

fn model_frame_tests(backend: Backend) -> Vec<ContractTest> {
    let name = backend_name(backend);
    let mut tests = vec![
        format!("models::llama_family::device_execution::tests::single_device_frame_restores_original_scratch_after_real_{name}_forward_failure"),
        format!("models::llama_family_forward_batched::tests::physical_batch_frame_preserves_real_{name}_logits_and_scratch_on_failure"),
        format!("models::llama_family_forward_batched::tests::selection_failure_full_readback_observes_real_{name}_preparation"),
        format!("models::llama_family::device_execution::preparation_tests::preparation_frame_real_{name}_allocator_and_contiguous_growth"),
    ];
    if backend == Backend::Metal {
        tests.push("models::llama_family::device_execution::preparation_tests::preparation_frame_real_metal_pending_context_blocks_allocation".into());
        tests.push("models::llama_family::device_execution::tests::metal_frame_failure_before_first_encode_restores_owner_and_full_numerical_execution".into());
        tests.extend([
            "qualification_unwrapped_metal_completes_and_rejects_invalid_close_without_losing_progress",
            "qualification_unwrapped_metal_cancelled_delivery_closes_original_and_keeps_peer_usable",
            "encoded_failure::qualification_unwrapped_metal_encoded_failure_retires_once_and_peer_matches_cpu",
            "qualification_unwrapped_metal_backbone_and_partial_stage_remain_unsupported",
        ].into_iter().map(|test| format!("executor::llm_executor::token_execution::qualification_tests::{test}")));
    }
    if backend != Backend::Cpu {
        tests.extend([
            format!("lora::runtime_tests::{name}_lora_workspace_growth_matches_complete_outputs"),
            format!("models::llama_family::lora_tests::instance_workspace::{name}_lora_instance_workspace_interleaves_adapters_and_survives_peer_close"),
        ]);
    }
    if backend == Backend::Cuda {
        tests.push("models::llama_family::device_execution::tests::model_frame_cancels_failed_cuda_capture_without_executing_prefix".into());
        tests.extend([
            "qualification_unwrapped_cuda_completion_and_original_owner_close",
            "qualification_unwrapped_cuda_cold_owner_closes_without_admission",
            "qualification_unwrapped_cuda_encoded_failure_keeps_peer_and_closes_original",
            "qualification_unwrapped_cuda_encoded_cancellation_preserves_peer",
            "qualification_unwrapped_cuda_cancelled_close_waiter_joins_original_worker",
            "int8::qualification_unwrapped_cuda_int8_completion_and_original_owner_close",
            "int8::qualification_unwrapped_cuda_int8_cold_owner_closes_without_admission",
            "int8::qualification_unwrapped_cuda_int8_encoded_failure_keeps_peer_and_closes_original",
            "int8::qualification_unwrapped_cuda_int8_encoded_cancellation_preserves_peer",
            "int8::qualification_unwrapped_cuda_int8_cancelled_close_waiter_joins_original_worker",
            "int8::qualification_unwrapped_cuda_int8_serial_page_crossing_keeps_completed_peer",
        ].into_iter().map(|test| format!("executor::llm_executor::token_execution::qualification_tests::cuda::{test}")));
    }
    tests
        .into_iter()
        .map(|name| ContractTest {
            package: "ferrum-models".into(),
            target: "ferrum_models".into(),
            kind: "lib".into(),
            name,
        })
        .collect()
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
                "completed_device_status_is_checked_after_fence_and_fails_quiescently",
                "completed_device_status_reused_slot_is_cleared_by_each_submission",
                "static_weight_no_copy_region_retains_page_owner_after_source_drop",
                "static_weight_import_copies_retained_regions_without_a_safe_native_view",
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
    let mut submission_tests: Vec<_> = names
        .iter()
        .map(|name| ContractTest {
            package: "ferrum-kernels".into(),
            target: "ferrum_kernels".into(),
            kind: "lib".into(),
            name: format!("{module}::{name}"),
        })
        .collect();
    submission_tests.push(ContractTest {
        package: "ferrum-models".into(),
        target: "ferrum_models".into(),
        kind: "lib".into(),
        name: format!(
            "executor::driver_contract_tests::sampling::shutdown::native_shutdown_{}_joins_cancelled_submission_after_waiter_drop",
            backend_name(backend),
        ),
    });
    if backend == Backend::Cuda {
        submission_tests.push(ContractTest {
            package: "ferrum-kernels".into(),
            target: "ferrum_kernels".into(),
            kind: "lib".into(),
            name: "backend::cuda::vnext_runtime::tests::numerical_status_snapshot_survives_eager_scratch_reuse_and_keeps_stream_usable".into(),
        });
    }
    if backend != Backend::Cpu {
        let target = format!("vnext_{}_checkpoint_continuation", backend_name(backend));
        let mut names = vec![
            "gated_delta_hadamard_pq2_shared_signs_and_mixed_projections_resume_public_checkpoint",
            "causal_int8_kv_q4k_provider_resumes_payload_and_scales_across_a_page_boundary",
            "causal_int8_kv_captures_completed_input_and_restores_an_appended_suffix",
            "causal_int8_kv_eager_failure_does_not_poison_the_execution_lane",
        ];
        match backend {
            Backend::Metal => {
                names.push("causal_int8_kv_checkpoint_rejects_a_same_token_f16_target")
            }
            Backend::Cuda => {
                names.push("causal_int8_kv_typed_slots_replay_and_recover_after_nonfinite_input")
            }
            Backend::Cpu => unreachable!(),
        }
        submission_tests.extend(names.into_iter().map(|name| ContractTest {
            package: "ferrum-models".into(),
            target: target.clone(),
            kind: "test".into(),
            name: name.into(),
        }));
    }
    let mut groups = vec![ContractGroup {
        id: checker_id(backend),
        behavior: Behavior::SubmissionCompletion,
        entrypoints: Vec::new(),
        tests: submission_tests,
    }];
    if backend != Backend::Cpu {
        let mut tests = model_frame_tests(backend);
        if backend == Backend::Metal {
            tests.extend(
                [
                    "backend::metal::checked_sync_tests::missing_submission_is_not_success",
                    "backend::metal::instance_queue_tests::metal_instance_queues_keep_peer_unsubmitted_work_through_retirement",
                    "backend::metal::instance_queue_tests::metal_instance_queues_consume_original_buffer_after_checked_handoff",
                    "attention::metal::pipelines::quantized::tests::quantized_catalog_lazy_slots_belong_to_each_catalog_and_share_by_borrow",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: name.into(),
                }),
            );
            tests.extend(
                [
                    "metal_injected_failure_blocks_host_access_and_encoding_until_explicit_begin",
                    "metal_injected_unresolved_observation_preserves_submission_and_blocks_host_access",
                    "metal_host_synchronization_preserves_full_outputs_and_successful_continuation",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: format!("backend::metal::host_access_tests::{name}"),
                }),
            );
            tests.extend(
                [
                    "metal_no_copy_quant_weights_survive_drop_before_commit",
                    "metal_no_copy_raw_buffer_clone_outlives_context_and_weight",
                    "metal_no_copy_peer_survives_independent_and_shared_mapping_retirement",
                    "metal_registered_partial_page_copies_exact_tensor_and_releases_source",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: format!("backend::metal::mmap_lifetime_tests::{name}"),
                }),
            );
            tests.extend(
                [
                    "metal_instance_weight_policy_is_frozen_across_interleaved_loads_and_peer_retirement",
                    "metal_instance_attention_routes_and_full_outputs_survive_interleaving_and_peer_retirement",
                    "metal_query_tiled_partial_kv_tiles_match_cpu_without_padding",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: format!("backend::metal::instance_config_tests::{name}"),
                }),
            );
            tests.extend(
                [
                    "copy_slice_real_metal_f32_preserves_offsets_and_canaries",
                    "copy_slice_real_metal_f16_preserves_offsets_and_canaries",
                    "copy_slice_real_metal_rejects_dtype_and_extent_errors_before_encoding",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: format!("backend::metal::copy_slice_tests::{name}"),
                }),
            );
        }
        if backend == Backend::Cuda {
            tests.extend(
                [
                    "qkv_capture_reuses_original_scratch_and_replays_after_eager_growth",
                    "qkv_unprepared_capture_failure_discards_prefix_and_reuses_original_context",
                    "invalidated_capture_is_confirmed_ended_before_original_context_reuse",
                ]
                .into_iter()
                .map(|name| ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: format!("backend::cuda::qkv::tests::{name}"),
                }),
            );
            tests.extend([
                ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: "backend::cuda::instance_owner_tests::equal_graph_keys_remain_owned_across_peer_reset_and_retirement".into(),
                },
                ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "ferrum_kernels".into(),
                    kind: "lib".into(),
                    name: "backend::cuda::submission_tests::legacy_stream_preserves_upload_compute_copy_order_and_context_reuse".into(),
                },
                ContractTest {
                    package: "ferrum-models".into(),
                    target: "ferrum_models".into(),
                    kind: "lib".into(),
                    name: "models::llama_family::lora_tests::lora_cuda_existing_base_graph_cannot_bypass_adapter".into(),
                },
                ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "int8_kv_parity".into(),
                    kind: "test".into(),
                    name: "int8_kv_append_then_decode_e2e".into(),
                },
                ContractTest {
                    package: "ferrum-kernels".into(),
                    target: "int8_kv_parity".into(),
                    kind: "test".into(),
                    name: "kv_cache_quant_int8_e2e".into(),
                },
            ]);
        }
        groups.push(ContractGroup {
            id: legacy_checker_id(backend),
            behavior: Behavior::SubmissionCompletion,
            entrypoints: Vec::new(),
            tests,
        });
    }
    groups.extend(numerical::groups(backend));
    groups.extend(compatibility::groups(backend));
    if backend != Backend::Cpu {
        // Lifecycle recovery compares independent models at the same physical
        // batch shape. Numerical qualification stays independently mandatory:
        // CUDA checks each shape against the CPU full-vocabulary oracle;
        // Metal retains its existing strict M=1/M=2 comparison.
        groups.push(ContractGroup {
            id: format!(
                "backend-contract.{}.legacy-physical-batch-numerics",
                backend_name(backend),
            ),
            behavior: Behavior::KernelNumerics,
            entrypoints: Vec::new(),
            tests: vec![ContractTest {
                package: "ferrum-models".into(),
                target: "ferrum_models".into(),
                kind: "lib".into(),
                name: if backend == Backend::Cuda {
                    "models::llama_family_forward_batched::tests::physical_batch_and_single_logits_match_cpu_oracle_on_cuda".into()
                } else {
                    format!(
                        "models::llama_family_forward_batched::tests::physical_batch_logits_match_single_item_decode_on_{}",
                        backend_name(backend),
                    )
                },
            }],
        });
        // Shared legacy weight loading has its own numerical contract. These
        // checks do not establish qualification for a typed model plan.
        groups.push(ContractGroup {
            id: format!("backend-contract.{}.gguf-fused-bias", backend_name(backend)),
            behavior: Behavior::KernelNumerics,
            entrypoints: Vec::new(),
            tests: vec![ContractTest {
                package: "ferrum-quantization".into(),
                target: "gguf_loader_test".into(),
                kind: "test".into(),
                name: format!(
                    "fused_bias::{}_fused_bias_matches_complete_oracles",
                    backend_name(backend),
                ),
            }],
        });
        groups.push(ContractGroup {
            id: format!(
                "backend-contract.{}.safetensors-direct-bias",
                backend_name(backend),
            ),
            behavior: Behavior::KernelNumerics,
            entrypoints: Vec::new(),
            tests: vec![ContractTest {
                package: "ferrum-quantization".into(),
                target: "native_safetensors_loader_test".into(),
                kind: "test".into(),
                name: format!(
                    "{}_direct_bias_matches_complete_oracles",
                    backend_name(backend),
                ),
            }],
        });
    }
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
            [Backend::Metal, Backend::Cuda]
                .into_iter()
                .filter_map(|backend| {
                    targets
                        .iter()
                        .filter(|target| {
                            target.backend == backend
                                && target.execution_path == super::submission::LEGACY_EXECUTION_PATH
                        })
                        .min_by_key(|target| (&target.architecture, &target.precision))
                        .map(|target| CheckDescriptor {
                            id: legacy_checker_id(backend),
                            behavior: Behavior::SubmissionCompletion,
                            layer: EvidenceLayer::BackendNumerics,
                            target: Some(target.clone()),
                            entrypoints: Vec::new(),
                        })
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
    } else if backend != Backend::Cpu
        && obligation.behavior == Behavior::SubmissionCompletion
        && obligation.scope == legacy_scope(backend)
    {
        Some(legacy_checker_id(backend))
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
    fn legacy_lifecycle_receipt_cannot_replace_physical_batch_numerical_execution() {
        for backend in [Backend::Metal, Backend::Cuda] {
            let report = report_fixture(backend);
            verify_report(backend, &report).unwrap();
            let numerical_id = format!(
                "backend-contract.{}.legacy-physical-batch-numerics",
                backend_name(backend),
            );
            let mut failed = report.clone();
            let execution = failed
                .execution
                .groups
                .iter_mut()
                .find(|group| group.id == numerical_id)
                .unwrap()
                .tests[0]
                .execution
                .as_mut()
                .unwrap();
            execution.exit_code = Some(1);
            assert!(verify_report(backend, &failed).is_err());
            let mut missing = report;
            missing
                .execution
                .groups
                .retain(|group| group.id != numerical_id);
            assert!(verify_report(backend, &missing).is_err());
        }
    }

    #[test]
    fn cuda_physical_batch_receipt_requires_the_cpu_oracle_execution() {
        let mut report = report_fixture(Backend::Cuda);
        verify_report(Backend::Cuda, &report).unwrap();
        let numerical = report
            .execution
            .groups
            .iter_mut()
            .find(|group| group.id == "backend-contract.cuda.legacy-physical-batch-numerics")
            .unwrap();
        assert_eq!(
            numerical.tests[0].binding.name,
            "models::llama_family_forward_batched::tests::physical_batch_and_single_logits_match_cpu_oracle_on_cuda",
        );
        // Even a purported passing receipt for the former cross-GPU-shape
        // comparison cannot substitute for the independent CPU-oracle test.
        numerical.tests[0].binding.name =
            "models::llama_family_forward_batched::tests::physical_batch_logits_match_single_item_decode_on_cuda".into();
        assert!(verify_report(Backend::Cuda, &report).is_err());
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
    fn legacy_accelerators_require_model_frame_execution_and_cannot_certify_operator_numerics() {
        for backend in [Backend::Metal, Backend::Cuda] {
            let target = ExecutionTarget {
                architecture: "llama_dense".into(),
                protocol: ferrum_types::ModelOutputProtocol::Text,
                precision: match backend {
                    Backend::Metal => "gguf-q4_k_m",
                    _ => "safetensors-bf16",
                }
                .into(),
                backend,
                execution_path: super::super::submission::LEGACY_EXECUTION_PATH.into(),
            };
            let descriptors = check_descriptors(&[target]);
            let descriptor = descriptors
                .iter()
                .find(|d| d.id == legacy_checker_id(backend))
                .unwrap();
            let mut obligation = super::super::Obligation {
                behavior: descriptor.behavior,
                layer: descriptor.layer,
                scope: legacy_scope(backend),
                checkers: vec![descriptor.id.clone()],
                entrypoints: Vec::new(),
                reason: "legacy stream lifecycle".into(),
            };
            let report = report_fixture(backend);
            let receipt = verify_report(backend, &report).unwrap();
            assert!(receipt.covers(&obligation));
            assert_eq!(required_backend(&obligation), Some(backend));
            obligation.behavior = Behavior::KernelNumerics;
            assert!(!receipt.covers(&obligation));
            obligation.behavior = Behavior::SubmissionCompletion;
            obligation.checkers = vec![checker_id(backend)];
            assert!(!receipt.covers(&obligation));
            for expected in model_frame_tests(backend) {
                let mut listed_only = report.clone();
                let group = listed_only
                    .execution
                    .groups
                    .iter_mut()
                    .find(|group| group.id == legacy_checker_id(backend))
                    .unwrap();
                let test = group
                    .tests
                    .iter_mut()
                    .find(|test| test.binding == expected)
                    .unwrap();
                test.execution = None;
                assert!(
                    verify_report(backend, &listed_only).is_err(),
                    "listing the frame assertion is not execution evidence"
                );
            }
            let mut vnext_only = report;
            vnext_only
                .execution
                .groups
                .retain(|group| group.id != legacy_checker_id(backend));
            assert!(verify_report(backend, &vnext_only).is_err());
            assert!(!contract_groups(Backend::Cpu)
                .iter()
                .any(|group| group.id == legacy_checker_id(backend)));
        }
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
    fn native_override_retirement_requires_each_actual_ffi_entrypoint() {
        let report = report_fixture(Backend::Cuda);
        assert!(verify_report(Backend::Cuda, &report).is_ok());
        for required in [
            "native_classic_dense_tile_retirement_preserves_cpu_reference",
            "native_classic_fused_tile_retirement_preserves_cpu_reference",
            "native_moe_f16_override_retirement_preserves_cpu_reference",
            "native_moe_fp8_override_retirement_preserves_cpu_reference",
            "native_moe_mxfp4_override_retirement_preserves_source_reference",
            "native_moe_profile_retirement_ignores_environment_after_real_math",
        ] {
            let mut missing = report.clone();
            for group in &mut missing.execution.groups {
                group
                    .tests
                    .retain(|test| !test.binding.name.ends_with(required));
            }
            assert!(
                verify_report(Backend::Cuda, &missing).is_err(),
                "a receipt must not omit native FFI evidence for {required}",
            );
        }
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
