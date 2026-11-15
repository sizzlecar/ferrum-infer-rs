//! Actual single-block execution. Zero residual gates give an independent
//! identity oracle; active gates exercise both nonzero branches and repeat.
//! Finite/repeat evidence is not an independent general DiT numerical oracle.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::CudaTensorOperation, vnext_runtime::CudaDeviceRuntime,
};
use std::sync::atomic::{AtomicUsize, Ordering};
fn ident<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
struct Counted {
    inner: FastH3DitBlockSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
fn managed(
    account: &HostMemoryAccount,
    name: &str,
    spec: &ProgramTensorSpec,
    fill: impl Fn(usize) -> f32,
) -> ManagedHostTensor {
    assert_eq!(spec.element_type, ElementType::Bf16);
    let mut owner = account
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            spec.byte_len().unwrap() as usize,
        )
        .unwrap();
    for (i, dst) in owner.as_mut().chunks_exact_mut(2).enumerate() {
        dst.copy_from_slice(&bf16::from_f32(fill(i)).to_bits().to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )
        .unwrap(),
        owner.freeze(),
    )
    .unwrap()
}
#[test]
#[ignore = "requires locked SM120 Sage/NVFP4, cuDNN and compact FMA; B1 complete block identity-gate/finite-repeat/retirement, no full-model quality claim"]
fn dit_block_cuda_executes_both_residuals_attention_and_ffn_in_one_retained_session() {
    let account = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
        CudaDeviceRuntime::pinned_weight_upload_alignment().unwrap(),
    )
    .unwrap();
    let (_directory, package) = fixture();
    let layout = super::super::attention_core::layout();
    let schedule = schedule();
    let tags = tags();
    let bounds = capacities();
    let (c, source, regions) = FastH3DitBlockSource::prepare(
        package,
        0,
        &layout,
        &schedule,
        &[(1, &tags)],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        bounds,
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(bounds).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(c.profile_id()).unwrap(),
        )
        .unwrap();
    let source_count = match prepared.weights() {
        ModelWeights::Stored(s) => s.components.len(),
        _ => panic!("Stored"),
    };
    let operation_count = prepared
        .program()
        .blocks()
        .iter()
        .map(|b| b.nodes.len())
        .sum::<usize>();
    let (runtime, registry) = super::super::attention_core::cuda_factory::create(
        regions.clone(),
        &[
            CudaTensorOperation::RowSelectedRmsModulation,
            CudaTensorOperation::RowSelectedResidualRmsModulation,
            CudaTensorOperation::RowSelectedAddcmulFmaBf16,
            CudaTensorOperation::HeadRmsPartialRopeSingleRound,
            CudaTensorOperation::MaskedTileMean,
            CudaTensorOperation::PackedSwiGlu,
        ],
        vec![
            Box::new(row_selected_rms_modulation_bf16_contract().unwrap()),
            Box::new(row_selected_residual_rms_modulation_bf16_contract().unwrap()),
            Box::new(row_selected_addcmul_fma_bf16_contract().unwrap()),
            Box::new(head_rms_partial_rope_single_round_bf16_contract().unwrap()),
            Box::new(masked_tile_mean_bf16_f32_contract().unwrap()),
            Box::new(packed_swiglu_bf16_contract().unwrap()),
        ],
    );
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        ident("provider.engine.dit_block"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let options = c.compile_options(bounds).unwrap();
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared,
            &registry,
            &catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .unwrap(),
            &materializers,
            &options,
        )
        .unwrap(),
    );
    let source = Arc::new(Counted {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let specs = options.tensor_specs();
    let common = vec![
        managed(&account, F::HIDDEN_ID, &specs[&id(F::HIDDEN_ID)], |i| {
            [-0.5, 0.25, 0.5, 1.][i % 4]
        }),
        managed(&account, F::COSINE_ID, &specs[&id(F::COSINE_ID)], |_| 1.),
        managed(&account, F::SINE_ID, &specs[&id(F::SINE_ID)], |_| 0.),
    ];
    let width = c.hidden_size() as usize;
    let tables = [0., 1.].map(|gate| {
        managed(&account, F::TABLE_ID, &specs[&id(F::TABLE_ID)], |i| {
            let chunk = (i % (6 * width)) / width;
            if chunk == 2 || chunk == 5 {
                gate
            } else {
                0.
            }
        })
    });
    let owners = Arc::new((
        source.clone(),
        common.clone(),
        tables.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
        regions.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            account.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            ComponentInitializationConfig {
                run_id: ident("run.dit_block.init"),
                request_id: ident("request.dit_block.init"),
                transaction_id: ident("transaction.dit_block.init"),
                static_policy: StaticInitializationPolicy::new(8192, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
            owners.clone(),
        )
        .unwrap_or_else(|error| {
            let diagnostic = format!("{error:?}");
            std::mem::forget(error);
            panic!("block initialization retains owners: {diagnostic}")
        });
    assert_eq!(source.reads.load(Ordering::Relaxed), source_count);
    let mut retained: Option<Vec<TensorExecutionOutput>> = None;
    for trial in 0..3 {
        let mut inputs = common.clone();
        inputs.push(tables[usize::from(trial != 0)].clone());
        let outputs = match session.session_mut().execute(
            ident(format!("run.dit_block.{trial}")),
            ident(format!("request.dit_block.{trial}")),
            inputs,
        ) {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnostic = format!("{error:?}");
                std::mem::forget((error, session, owners));
                panic!("block execution retains owners: {diagnostic}");
            }
        };
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value_id(), &id(F::OUTPUT_ID));
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [c.rows().unwrap(), c.hidden_size()]
        );
        let bytes = outputs[0].bytes().bytes();
        assert!(bytes
            .chunks_exact(2)
            .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()));
        if trial == 0 {
            assert_eq!(
                bytes,
                common[0].bytes().bytes(),
                "zero gates preserve exact residual bits"
            );
        } else {
            assert_ne!(
                bytes,
                common[0].bytes().bytes(),
                "active nonzero branches must affect output"
            );
            if trial == 2 {
                assert_eq!(
                    bytes,
                    retained.as_ref().unwrap()[0].bytes().bytes(),
                    "same explicit table repeats"
                );
            }
        }
        assert_eq!(source.reads.load(Ordering::Relaxed), source_count);
        retained = Some(outputs);
    }
    let (returned, receipt) = session.close().unwrap_or_else(|error| {
        let diagnostic = format!("{error:?}");
        std::mem::forget(error);
        panic!("block close retains owners: {diagnostic}")
    });
    drop(receipt);
    drop(returned);
    drop(owners);
    drop(common);
    drop(tables);
    drop(compiled);
    drop(source);
    drop(registry);
    drop(regions);
    let retained = retained.unwrap();
    assert!(retained.iter().all(|o| account.owns(o.bytes())));
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    eprintln!("FastH3 complete DiT block: B1 rows={} H{} nodes={operation_count}, original Stored source_reads={source_count}, zero-gate bit identity then 2 active-gate finite/repeated runs, close/Host retirement; no arbitrary full-block numerical parity or service/CB claim",c.rows().unwrap(),c.hidden_size());
}
