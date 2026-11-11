//! A real Stored connector projection exercises both dispatch lanes. This is
//! transport/owner evidence, separate from complete quantized TE qualification.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
#[test]
fn stored_f32_connector_projection_uses_same_actor_or_direct_math_and_retires_owners() {
    let composition = ReferenceVNextComposition::create(id("device.reference.text_actor")).unwrap();
    let config = PH3ImageTextConnectorProjectionConfig::new(2, 2).unwrap();
    let family = TypedFamilyRegistration::new(
        PH3ImageTextConnectorProjectionFamily::new(PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 16,
        })
        .unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&config).unwrap(),
        &NumericalProfileId::new(PH3ImageTextConnectorProjectionFamily::REFERENCE_PROFILE_ID)
            .unwrap(),
    )
    .unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(PH3ImageTextConnectorProjectionFamily::INPUT_ID),
        config.input_spec(2).unwrap(),
    )]))
    .unwrap();
    let d = composition.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let component = CompiledTensorComponent::compile(
        family,
        composition.registry(),
        composition.catalog(),
        &policy,
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let weight = [1.0_f32, 2., -1., 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &weight).unwrap();
    safetensors::tensor::serialize_to_file(
        [("proj.weight", view)],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8192,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8192,
                output_bytes: 8192,
                transfer_staging_bytes: 8192,
                weight_backing_bytes: 8192,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&archive),
            ComponentInitializationConfig {
                run_id: id("run.text_actor.init"),
                request_id: id("request.text_actor.init"),
                transaction_id: id("transaction.text_actor.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut actor = TensorSerialActor::new(composition.runtime().clone()).unwrap();
    let mut previous = None;
    for use_actor in [false, true] {
        let mut allocation = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap();
        for (b, v) in allocation
            .as_mut()
            .chunks_exact_mut(4)
            .zip([2.0_f32, 3., -2., 4.])
        {
            b.copy_from_slice(&v.to_le_bytes());
        }
        let input = ManagedHostTensor::new(
            id(PH3ImageTextConnectorProjectionFamily::INPUT_ID),
            ResolvedTensorSpec::new(
                vec![2, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            allocation.freeze(),
        )
        .unwrap();
        let mut execution = if use_actor {
            TextExecution::Actor(&mut actor)
        } else {
            TextExecution::Direct
        };
        let outputs = execution
            .execute(
                &mut session,
                id(format!("run.text_actor.{use_actor}")),
                id(format!("request.text_actor.{use_actor}")),
                vec![input],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(host.owns(outputs[0].bytes()));
        let actual = outputs[0].bytes().bytes().to_vec();
        assert_eq!(
            actual
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
            [8., -0.5, 6., 4.]
        );
        if let Some(previous) = &previous {
            assert_eq!(&actual, previous);
        }
        previous = Some(actual);
        drop(outputs);
    }
    session
        .close()
        .unwrap_or_else(|failure| panic!("text actor close: {:?}", failure.failure()));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
