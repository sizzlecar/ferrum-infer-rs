//! Exact portable causal composition for the original bounded CUDA context.
//! Native module bytes are caller-declared, separately admitted allowances.
use super::super::vnext_runtime::{CudaCausalProviderRetention, CudaTensorExecutionProfile};
use super::*;

/// Adds the closed dense-F16 causal capabilities and their paged KV layout.
/// This does not select a module allowance or change the caller's stream count.
pub fn cuda_tensor_and_dense_f16_causal_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    dense_causal_runtime_config(ordinal, device_id, false)
}

pub fn cuda_tensor_and_dense_bf16_causal_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    dense_causal_runtime_config(ordinal, device_id, true)
}

fn dense_causal_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
    bf16: bool,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    let mut config = cuda_tensor_runtime_config(ordinal, device_id)?;
    let capabilities: &[&str] = if bf16 {
        &crate::backend::cuda_tensor_causal_policy::DENSE_BF16_CAUSAL_CAPABILITIES
    } else {
        &crate::backend::cuda_tensor_causal_policy::DENSE_F16_CAUSAL_CAPABILITIES
    };
    for capability in capabilities {
        config.capabilities.insert(CapabilityId::new(*capability)?);
    }
    config
        .dynamic_storage_profiles
        .insert(DynamicStorageProfile::new(
            DynamicStorageAllocator::FixedBlockArena {
                block_bytes: VNEXT_KV_PAGE_BYTES,
            },
            DynamicStorageView::PagedRegions {
                block_bytes: VNEXT_KV_PAGE_BYTES,
            },
        )?);
    config.runtime_implementation_fingerprint = implementation_fingerprint(&[
        if bf16 {
            b"ferrum.cuda.bounded-tensor-and-dense-bf16-causal.v1"
        } else {
            b"ferrum.cuda.bounded-tensor-and-dense-f16-causal.v1"
        },
        config.runtime_implementation_fingerprint.as_bytes(),
        include_bytes!("dense_f16_causal.rs"),
    ]);
    Ok(config)
}

pub fn cuda_vnext_dense_f16_causal_contracts() -> Result<Vec<Box<dyn OperationContract>>, VNextError>
{
    Ok(vec![
        Box::new(token_embedding_contract()?),
        Box::new(dense_linear_contract()?),
        Box::new(last_token_dense_linear_contract()?),
        Box::new(last_token_masked_argmax_contract()?),
        Box::new(ferrum_runtime::tail_two_greedy_f16_contract()?),
        Box::new(ferrum_runtime::concat_columns_f16_contract()?),
        Box::new(rms_norm_contract()?),
        Box::new(dense_swiglu_contract()?),
        Box::new(residual_add_contract()?),
        Box::new(gated_delta_recurrent_attention_contract()?),
        Box::new(ferrum_runtime::gated_delta_candidate_scan_f16_contract()?),
        Box::new(causal_paged_attention_contract()?),
    ])
}

pub fn cuda_vnext_dense_bf16_causal_contracts(
) -> Result<Vec<Box<dyn OperationContract>>, VNextError> {
    Ok(vec![
        Box::new(ferrum_runtime::token_embedding_bf16_contract()?),
        Box::new(ferrum_runtime::last_token_dense_linear_bf16_contract()?),
        Box::new(ferrum_runtime::last_token_masked_argmax_bf16_contract()?),
        Box::new(ferrum_runtime::rms_norm_bf16_contract()?),
        Box::new(ferrum_runtime::dense_swiglu_bf16_contract()?),
        Box::new(ferrum_runtime::residual_add_bf16_contract()?),
        Box::new(ferrum_runtime::gated_delta_recurrent_attention_bf16_contract()?),
        Box::new(ferrum_runtime::causal_paged_attention_bf16_contract()?),
    ])
}

pub fn cuda_tensor_and_dense_bf16_visual_causal_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    let mut config = cuda_tensor_and_dense_bf16_causal_runtime_config(ordinal, device_id)?;
    config.capabilities.extend([
        CapabilityId::new(ferrum_runtime::ROW_SELECT_BF16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_CAPABILITY_ID)?,
    ]);
    config.runtime_implementation_fingerprint = implementation_fingerprint(&[
        b"ferrum.cuda.bounded-tensor-and-dense-bf16-visual-causal.v1",
        config.runtime_implementation_fingerprint.as_bytes(),
        include_bytes!("transformer/visual_row_select_f16.rs"),
    ]);
    Ok(config)
}

pub fn cuda_vnext_dense_bf16_visual_causal_contracts(
) -> Result<Vec<Box<dyn OperationContract>>, VNextError> {
    let mut contracts = cuda_vnext_dense_bf16_causal_contracts()?;
    contracts.push(Box::new(ferrum_runtime::row_select_bf16_contract()?));
    contracts.push(Box::new(
        ferrum_runtime::causal_paged_attention_explicit_rope_bf16_contract()?,
    ));
    Ok(contracts)
}

pub fn cuda_tensor_and_dense_f16_visual_causal_runtime_config(
    ordinal: usize,
    device_id: DeviceId,
) -> Result<CudaDeviceRuntimeConfig, VNextError> {
    let mut config = cuda_tensor_and_dense_f16_causal_runtime_config(ordinal, device_id)?;
    config.capabilities.extend([
        CapabilityId::new(ferrum_runtime::ROW_SELECT_F16_CAPABILITY_ID)?,
        CapabilityId::new(ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID)?,
    ]);
    config.runtime_implementation_fingerprint = implementation_fingerprint(&[
        b"ferrum.cuda.bounded-tensor-and-dense-f16-visual-causal.v1",
        config.runtime_implementation_fingerprint.as_bytes(),
        include_bytes!("transformer/visual_row_select_f16.rs"),
    ]);
    Ok(config)
}

pub fn cuda_vnext_dense_f16_visual_causal_contracts(
) -> Result<Vec<Box<dyn OperationContract>>, VNextError> {
    let mut contracts = cuda_vnext_dense_f16_causal_contracts()?;
    contracts.push(Box::new(ferrum_runtime::row_select_f16_contract()?));
    contracts.push(Box::new(
        ferrum_runtime::causal_paged_attention_explicit_rope_f16_contract()?,
    ));
    Ok(contracts)
}

/// Loads the exact existing F16 implementations after admitting the entire
/// additional module group. No FP8, GGUF materializer, native attention policy,
/// F32-master or reusable-execution provider is installed by this factory.
/// The F16 implementations currently load their native block helper modules as
/// well; those loads are included in this group's opaque declaration.
pub fn cuda_vnext_dense_f16_causal_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers(runtime, false, false)
}

pub fn cuda_vnext_dense_f16_visual_causal_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers(runtime, true, false)
}

pub fn cuda_vnext_dense_bf16_causal_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers(runtime, false, true)
}
pub fn cuda_vnext_dense_bf16_visual_causal_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers(runtime, true, true)
}

fn providers(
    runtime: &CudaDeviceRuntime,
    visual: bool,
    bf16: bool,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let expected = if bf16 && visual {
        CudaTensorExecutionProfile::TensorAndDenseBf16VisualCausalV1
    } else if bf16 {
        CudaTensorExecutionProfile::TensorAndDenseBf16CausalV1
    } else if visual {
        CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
    } else {
        CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
    };
    if runtime.tensor_execution_profile() != Some(expected)
        || runtime.attention_execution_policy() != AttentionExecutionPolicy::Portable
    {
        return Err(CudaDeviceRuntimeError::contract(
            "dense-F16 causal providers require the explicit bounded tensor+causal Portable profile",
        ));
    }
    let owner = runtime.reserve_causal_provider()?;
    let provider_owner = Arc::clone(&owner);
    // Partial native construction is fail-closed, exactly like the original
    // bounded constructor: uncertain module cleanup never fabricates a refund.
    let (providers, _owner) = super::super::vnext_runtime::create_native(owner, || {
        let mut providers: Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> = if bf16 {
            vec![
                Box::new(CudaTokenEmbeddingProvider::with_precision(
                    runtime,
                    TokenPrecision::Bf16,
                )?),
                Box::new(CudaLastTokenDenseLinearProvider::with_precision(
                    runtime,
                    TokenPrecision::Bf16,
                )?),
                Box::new(CudaLastTokenMaskedArgmaxProvider::with_precision(
                    runtime,
                    ArgmaxPrecision::Bf16,
                )?),
                Box::new(transformer::CudaRmsNormProvider::new_bf16(runtime)?),
                Box::new(transformer::CudaDenseSwiGluProvider::new_dense_bf16(
                    runtime,
                )?),
                Box::new(transformer::CudaResidualAddProvider::new_bf16(runtime)?),
                Box::new(
                    transformer::CudaGatedDeltaRecurrentAttentionProvider::new_dense_bf16(runtime)?,
                ),
                Box::new(transformer::CudaCausalPagedAttentionProvider::new_dense_bf16(runtime)?),
            ]
        } else {
            vec![
                Box::new(CudaTokenEmbeddingProvider::with_precision(
                    runtime,
                    TokenPrecision::F16,
                )?),
                Box::new(CudaLastTokenDenseLinearProvider::with_precision(
                    runtime,
                    TokenPrecision::F16,
                )?),
                Box::new(CudaLastTokenMaskedArgmaxProvider::with_precision(
                    runtime,
                    ArgmaxPrecision::F16,
                )?),
                Box::new(tail_two_greedy::CudaTailTwoGreedyProvider::new(runtime)?),
                Box::new(concat_columns_f16::CudaConcatColumnsProvider::new(runtime)?),
                Box::new(transformer::CudaRmsNormProvider::new(runtime)?),
                Box::new(transformer::CudaDenseLinearProvider::new_dense_f16(
                    runtime,
                )?),
                Box::new(transformer::CudaDenseSwiGluProvider::new_dense_f16(
                    runtime,
                )?),
                Box::new(transformer::CudaResidualAddProvider::new(runtime)?),
                Box::new(
                    transformer::CudaGatedDeltaRecurrentAttentionProvider::new_dense_f16(runtime)?,
                ),
                Box::new(
                    transformer::CudaGatedDeltaRecurrentAttentionProvider::new_candidate(
                        runtime, false, true,
                    )?,
                ),
                Box::new(transformer::CudaCausalPagedAttentionProvider::new_dense_f16(runtime)?),
            ]
        };
        if visual {
            providers.push(Box::new(if bf16 {
                transformer::CudaVisualRowSelectF16Provider::new_bf16(runtime)?
            } else {
                transformer::CudaVisualRowSelectF16Provider::new(runtime)?
            }));
            providers.push(Box::new(if bf16 {
                transformer::CudaCausalPagedAttentionProvider::new_explicit_rope_bf16(runtime)?
            } else {
                transformer::CudaCausalPagedAttentionProvider::new_explicit_rope_f16(runtime)?
            }));
        }
        providers
            .into_iter()
            .map(|inner| {
                let descriptor = dense_descriptor(inner.descriptor()).map_err(contract_error)?;
                Ok(Box::new(AdmittedProvider {
                    inner,
                    descriptor,
                    _owner: Arc::clone(&provider_owner),
                })
                    as Box<dyn OperationProvider<CudaDeviceRuntime>>)
            })
            .collect()
    })?;
    Ok(providers)
}

fn dense_descriptor(
    d: &OperationProviderDescriptor,
) -> Result<OperationProviderDescriptor, VNextError> {
    Ok(OperationProviderDescriptor::new(
        d.provider_id().clone(),
        d.operation_id().clone(),
        d.operation_fingerprint(),
        d.provider_implementation_fingerprint(),
        d.execution_semantics(),
        d.version(),
        d.device_id().clone(),
        d.capabilities().clone(),
        d.accepted_weight_formats()
            .iter()
            .filter(|format| format.as_str() == DENSE_SAFETENSORS_FORMAT_ID)
            .cloned()
            .collect(),
        BTreeSet::new(),
        d.dynamic_storage_bindings().to_vec(),
        d.resource_estimator_id(),
        d.resource_estimator_version(),
        d.resource_estimator_implementation_fingerprint(),
    )?
    .with_checkpoint_capability(d.checkpoint_capability().clone())
    .with_candidate_state_effects(d.candidate_state_effects().clone()))
}

struct AdmittedProvider {
    // Destroy native functions before releasing their module allowance.
    inner: Box<dyn OperationProvider<CudaDeviceRuntime>>,
    descriptor: OperationProviderDescriptor,
    _owner: Arc<CudaCausalProviderRetention>,
}
impl OperationResourceEstimator for AdmittedProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        self.inner.estimate_resources(request)
    }
}
impl OperationProvider<CudaDeviceRuntime> for AdmittedProvider {
    fn segment_binding_declaration_support(
        &self,
    ) -> ferrum_runtime::SegmentBindingDeclarationSupport {
        self.inner.segment_binding_declaration_support()
    }

    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.inner.reusable_execution_topology(request)
    }
    fn reusable_binding_resources(&self) -> ferrum_runtime::ReusableBindingResources {
        self.inner.reusable_binding_resources()
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        // CudaCommandExecutable captures the same quota from its admitted CUDA
        // regions, after enqueue/native-function fields in destruction order.
        // A fence or unknown-submission quarantine therefore retains this grant
        // even after the registry and all providers have been dropped.
        self.inner.encode_selected(invocation)
    }

    fn encode_selected_with_segment_declaration(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<
        (
            EncodedDeviceOperation<CudaDeviceCommand>,
            Option<ferrum_runtime::SegmentBindingDeclaration>,
        ),
        OperationFailure,
    > {
        self.inner
            .encode_selected_with_segment_declaration(invocation)
    }

    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<ferrum_runtime::EncodedReusableExecutionBindings<CudaDeviceCommand>, OperationFailure>
    {
        self.inner.encode_reusable_execution_bindings(invocation)
    }
}

pub(super) fn reject_legacy_bounded_provider(
    runtime: &CudaDeviceRuntime,
) -> Result<(), CudaDeviceRuntimeError> {
    if runtime.tensor_execution_profile().is_some() {
        return Err(CudaDeviceRuntimeError::contract(
            "bounded causal modules must be built by the admitted dense-F16 causal provider group",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visual_configuration_is_an_explicit_exact_two_capability_extension() {
        let id = DeviceId::new("device.cuda.0").unwrap();
        let text = cuda_tensor_and_dense_f16_causal_runtime_config(0, id.clone()).unwrap();
        let visual = cuda_tensor_and_dense_f16_visual_causal_runtime_config(0, id).unwrap();
        let added = visual
            .capabilities
            .difference(&text.capabilities)
            .map(|c| c.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            added,
            [
                ferrum_runtime::ROW_SELECT_F16_CAPABILITY_ID,
                ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID
            ]
            .into_iter()
            .collect()
        );
        crate::backend::cuda_tensor_causal_policy::validate_capabilities(
            &visual.capabilities,
            CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1,
        )
        .unwrap();
        assert!(
            crate::backend::cuda_tensor_causal_policy::validate_capabilities(
                &visual.capabilities,
                CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
            )
            .is_err()
        );
        assert_eq!(
            cuda_vnext_dense_f16_visual_causal_contracts()
                .unwrap()
                .len(),
            cuda_vnext_dense_f16_causal_contracts().unwrap().len() + added.len()
        );
        assert_ne!(
            text.runtime_implementation_fingerprint,
            visual.runtime_implementation_fingerprint
        );
    }
    #[test]
    fn precise_configuration_keeps_tensor_policy_and_adds_only_the_declared_causal_set() {
        let id = DeviceId::new("device.cuda.0").unwrap();
        let old = cuda_tensor_runtime_config(0, id.clone()).unwrap();
        let new = cuda_tensor_and_dense_f16_causal_runtime_config(0, id).unwrap();
        let difference = new
            .capabilities
            .difference(&old.capabilities)
            .map(|c| c.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            difference,
            crate::backend::cuda_tensor_causal_policy::DENSE_F16_CAUSAL_CAPABILITIES
                .into_iter()
                .collect()
        );
        assert_eq!(
            new.attention_execution_policy,
            AttentionExecutionPolicy::Portable
        );
        assert!(old
            .dynamic_storage_profiles
            .is_subset(&new.dynamic_storage_profiles));
        assert_ne!(
            old.runtime_implementation_fingerprint,
            new.runtime_implementation_fingerprint
        );
        crate::backend::cuda_tensor_causal_policy::validate_capabilities(
            &new.capabilities,
            CudaTensorExecutionProfile::TensorAndDenseF16CausalV1,
        )
        .unwrap();
        let contracts = cuda_vnext_dense_f16_causal_contracts().unwrap();
        assert_eq!(contracts.len(), difference.len());
    }

    #[test]
    #[ignore = "requires an actual CUDA context and pinned native operator set; construction only, not model numerical qualification"]
    fn admitted_causal_and_tensor_groups_share_one_runtime_and_release_their_own_slots() {
        use super::super::super::vnext_runtime::{
            CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeConfig, CudaTensorRuntimeConfig,
        };
        // Both released text and image-conditioned paths must stay closed to
        // their exact dense 8/10-op sets under the full product feature bundle.
        // Each iteration is a separate qualification case, not two runtimes
        // hidden behind one product request or one provider group.
        for visual in [false, true] {
            let device = if visual {
                cuda_tensor_and_dense_f16_visual_causal_runtime_config(
                    0,
                    DeviceId::new("device.cuda.0").unwrap(),
                )
            } else {
                cuda_tensor_and_dense_f16_causal_runtime_config(
                    0,
                    DeviceId::new("device.cuda.0").unwrap(),
                )
            }
            .unwrap();
            let causal = CudaDenseF16CausalProfile::new(
                64 * 1024 * 1024,
                "experimental-unqualified-dense-f16-modules-v1",
            )
            .unwrap();
            let runtime =
                CudaDeviceRuntime::new_tensor_and_causal_eager(CudaTensorAndCausalRuntimeConfig {
                    tensor: CudaTensorRuntimeConfig {
                        device,
                        usable_memory_bytes: 1024 * 1024 * 1024,
                        bootstrap_memory_bytes: 128 * 1024 * 1024,
                        stream_memory_bytes: 16 * 1024 * 1024,
                        maximum_streams: 3,
                        maximum_in_flight_per_stream: 1,
                        maximum_commands_per_submission: 1024,
                        opaque_memory_profile: "experimental-unqualified-composition-v1".to_owned(),
                    },
                    causal: if visual {
                        causal.with_visual_prompt_v1()
                    } else {
                        causal
                    },
                })
                .unwrap();
            assert_eq!(
                runtime.tensor_execution_profile(),
                Some(if visual {
                    CudaTensorExecutionProfile::TensorAndDenseF16VisualCausalV1
                } else {
                    CudaTensorExecutionProfile::TensorAndDenseF16CausalV1
                })
            );
            assert!(cuda_vnext_operation_registry(&runtime).is_err());
            assert!(CudaTokenEmbeddingProvider::new(&runtime).is_err());
            let make_providers = || {
                if visual {
                    cuda_vnext_dense_f16_visual_causal_providers(&runtime)
                } else {
                    cuda_vnext_dense_f16_causal_providers(&runtime)
                }
            };
            let mut providers = make_providers().unwrap();
            let declared = if visual {
                cuda_vnext_dense_f16_visual_causal_contracts()
            } else {
                cuda_vnext_dense_f16_causal_contracts()
            }
            .unwrap();
            assert_eq!(
                providers.len(),
                declared.len(),
                "every declared operation needs its actual selected provider"
            );
            for provider in &providers {
                assert!(
                    provider
                        .descriptor()
                        .capabilities()
                        .is_subset(&runtime.descriptor().capabilities),
                    "the selected dense provider must be admitted by the original closed runtime capability set"
                );
                assert!(provider
                    .descriptor()
                    .accepted_quantization_formats()
                    .is_empty());
                assert!(provider
                    .descriptor()
                    .accepted_weight_formats()
                    .iter()
                    .all(|f| f.as_str() == DENSE_SAFETENSORS_FORMAT_ID));
                let contract = declared
                    .iter()
                    .find(|contract| {
                        &contract.descriptor().id == provider.descriptor().operation_id()
                    })
                    .unwrap();
                assert_eq!(
                    provider.descriptor().capabilities(),
                    &contract.descriptor().provider.required_capabilities,
                    "dense construction must require its actual operation contract, without unrelated compiled quantization capabilities"
                );
            }
            let swiglu = providers
                .iter()
                .find(|provider| {
                    provider.descriptor().operation_id().as_str()
                        == ferrum_runtime::DENSE_SWIGLU_OPERATION_ID
                })
                .unwrap();
            assert_eq!(
                swiglu.descriptor().capabilities(),
                &BTreeSet::from([CapabilityId::new(DENSE_SWIGLU_F16_CAPABILITY_ID).unwrap()])
            );
            assert!(swiglu
                .descriptor()
                .accepted_weight_formats()
                .contains(&WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).unwrap()));
            assert!(make_providers().is_err());
            providers.extend(
                cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::Silu]).unwrap(),
            );
            let mut contracts = declared;
            contracts.push(Box::new(ferrum_runtime::silu_bf16_contract().unwrap()));
            let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
            drop(registry);
            // Each group returns its own slot. No second runtime/context is made.
            drop(make_providers().unwrap());
            drop(cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::Silu]).unwrap());
        }
    }
}
