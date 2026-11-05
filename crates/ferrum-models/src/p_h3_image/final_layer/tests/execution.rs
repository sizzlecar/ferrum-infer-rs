use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use std::sync::Arc;
struct VerificationFamily {
    inner: Family,
    capture: bool,
}
impl ModelFamilyProvider for VerificationFamily {
    type Config = PH3ImageFinalLayerConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.inner.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.inner.external_metadata_ids()
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.inner.parse_config(r)
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.inner.validate_config_identity(r, c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.inner.validated_external_metadata_id(r, c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.inner.weights(c)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.inner.numerical_profiles(c)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self.inner.semantic_program(c, p)?;
        if !self.capture {
            return Ok(base);
        }
        let outputs = base
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| n.outputs.clone())
            .collect();
        ModelProgram::new(
            base.family_id().clone(),
            base.inputs().to_vec(),
            base.blocks().to_vec(),
            base.states().to_vec(),
            base.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(base.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.inner.semantic_metadata(c)
    }
}
fn execute<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    c: &PH3ImageFinalLayerConfig,
    source: &SafetensorsArchive,
    profile: &str,
    capture: bool,
) -> BTreeMap<ProgramValueId, (ElementType, Vec<u8>)> {
    let family = TypedFamilyRegistration::new(VerificationFamily {
        inner: Family::new(limits()).unwrap(),
        capture,
    })
    .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(profile))
    .unwrap();
    let d = runtime.descriptor();
    let materializers = WeightMaterializerRegistry::new(vec![]).unwrap();
    let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
    let options = c.compile_options().unwrap();
    let compiled = CompiledTensorComponent::compile(
        family,
        registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        &options,
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2 << 20,
                transfer_staging_bytes: 2 << 20,
                output_bytes: 2 << 20,
                weight_backing_bytes: 2 << 20,
            },
        )
        .unwrap(),
    );
    let checkpoint = PH3ImageFinalLayerCheckpoint::new(c, source, limits()).unwrap();
    let mut session = compiled
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            ComponentInitializationConfig {
                request_id: id("request.final.init"),
                run_id: id("run.final.init"),
                transaction_id: id("transaction.final.init"),
                static_policy: StaticInitializationPolicy::new(2 << 20, 256).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|f| panic!("final initialize failed: {f:?}"));
    let bits = input(c);
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bits.len() * 2)
        .unwrap();
    for (dst, bits) in allocation.as_mut().chunks_exact_mut(2).zip(bits) {
        dst.copy_from_slice(&bits.to_le_bytes());
    }
    let x = ManagedHostTensor::new(
        id(Family::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![c.input_rows().unwrap(), c.hidden],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        allocation.freeze(),
    )
    .unwrap();
    let basis = checkpoint
        .managed_basis(
            &host,
            &[0.25, 0.75],
            PH3ImageCurveLerpArithmetic::SeparateF32,
        )
        .unwrap();
    assert_eq!(
        basis.bytes().bytes(),
        f32_bytes([0.25, 0.625, 0.75, -0.125])
    );
    let outputs = session
        .execute(
            id("run.final.execute"),
            id("request.final.execute"),
            vec![x, basis],
        )
        .unwrap_or_else(|f| panic!("final execute failed: {f:?}"));
    let actual = outputs
        .iter()
        .map(|o| {
            assert!(host.owns(o.bytes()));
            assert_eq!(
                o.tensor().dimensions(),
                options.tensor_specs()[o.value_id()].dimensions
            );
            (
                o.value_id().clone(),
                (o.tensor().element_type(), o.bytes().bytes().to_vec()),
            )
        })
        .collect();
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|f| panic!("final close failed: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(outputs.iter().all(|o| host.owns(o.bytes())));
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    actual
}
fn verify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    profile: &str,
) {
    for many in [false, true] {
        let c = config(many);
        let (_d, source) = archive(&payloads(&c));
        let actual = execute(
            runtime.clone(),
            registry,
            catalog,
            &c,
            &source,
            profile,
            true,
        );
        for (image, port) in [
            (true, Family::IMAGE_VELOCITY_ID),
            (false, Family::AUDIO_VELOCITY_ID),
        ] {
            let observed = &actual[&id::<ProgramValueId>(port)];
            assert_eq!(observed.0, ElementType::Bf16);
            let wanted: Vec<u8> = expected(&c, &source, image)
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect();
            assert_eq!(observed.1, wanted, "positive BF16 velocity {port}");
        }
        let parameter_weight = f32_parameter(&source, "final_layer.adaln_proj.linear.weight");
        let parameter_bias = f32_parameter(&source, "final_layer.adaln_proj.linear.bias");
        let wanted_modulations: [[u32; 4]; 2] = [[0.25, 0.625], [0.75, -0.125]].map(|basis| {
            let p = affine(&basis, &parameter_weight, &parameter_bias);
            std::array::from_fn(|column| {
                let factor = 1.0_f32 + p[4 + column];
                let product = 1.5_f32 * factor;
                (product + p[column]).to_bits()
            })
        });
        for (port, (dtype, bytes)) in &actual {
            if port.as_str().ends_with(".modulated") {
                assert_eq!(*dtype, ElementType::F32);
                for row in bytes.chunks_exact(16) {
                    let first = f32::from_le_bytes(row[..4].try_into().unwrap());
                    assert_eq!(first.to_bits(), 2.0_f32.powi(-22).to_bits());
                    let bits: [u32; 4] = std::array::from_fn(|i| {
                        u32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap())
                    });
                    assert!(
                        wanted_modulations.contains(&bits),
                        "{port}: scale/shift lost their native F32 precision"
                    );
                }
            }
        }
        let image = &actual[&id::<ProgramValueId>(Family::IMAGE_VELOCITY_ID)].1;
        let audio = &actual[&id::<ProgramValueId>(Family::AUDIO_VELOCITY_ID)].1;
        assert_eq!(
            u16::from_le_bytes(image[..2].try_into().unwrap()),
            bf16::from_f32(4.5).to_bits()
        );
        assert_eq!(
            u16::from_le_bytes(audio[..2].try_into().unwrap()),
            bf16::from_f32(-1.75).to_bits()
        );
        let product = execute(
            runtime.clone(),
            registry,
            catalog,
            &c,
            &source,
            profile,
            false,
        );
        for port in [Family::IMAGE_VELOCITY_ID, Family::AUDIO_VELOCITY_ID] {
            assert_eq!(
                product[&id::<ProgramValueId>(port)],
                actual[&id::<ProgramValueId>(port)]
            );
        }
    }
}
#[test]
fn stored_reference_final_heads_preserve_f32_modulation_positive_bf16_velocity_and_chunk_routing() {
    let c = ReferenceVNextComposition::create(id("device.reference.h3_final")).unwrap();
    verify(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        Family::REFERENCE_PROFILE_ID,
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn stored_cuda_final_heads_preserve_f32_modulation_positive_bf16_velocity_and_chunk_routing() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{
            cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
            cuda_vnext_tensor_providers, CudaTensorOperation,
        },
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "unqualified-h3-final-heads-v1".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::RowGather,
            CudaTensorOperation::BatchedRowConcat,
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::RowSelectedModulationBf16F32,
            CudaTensorOperation::CastF32ToBf16,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(row_gather_bf16_contract().unwrap()),
            Box::new(batched_row_concat_bf16_contract().unwrap()),
            Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
            Box::new(row_selected_modulation_bf16_f32_contract().unwrap()),
            Box::new(affine_linear_f32_contract().unwrap()),
            Box::new(cast_f32_to_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.h3_final.cuda"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(d.clone(), vec![engine])
        .unwrap();
    verify(runtime, &registry, &catalog, Family::CUDA_PROFILE_ID);
}
