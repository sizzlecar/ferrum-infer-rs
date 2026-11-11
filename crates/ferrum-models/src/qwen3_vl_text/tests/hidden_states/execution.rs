//! Real managed execution with zero residual branches, nonzero embeddings and
//! distinct DeepStack updates. The exact residual values and analytic RMSNorm
//! expose selection before injection, stale aliases and an omitted final norm.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy,
};
use ferrum_execution::TensorExecutorConfig;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;

fn id<T>(value: &str) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn encoded(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

#[test]
fn reference_selected_states_retain_post_deepstack_values_and_normalize_last_state() {
    let config = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&hf(4, 8, 12, 4, 2, 2)).unwrap(),
        Qwen3VLTextLayout::new(2, vec![vec![Qwen3VLTextKeyRun::new(2, true)]]).unwrap(),
    )
    .unwrap()
    .with_deepstack(
        Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(0, 1).unwrap()]).unwrap(),
    )
    .unwrap()
    .with_hidden_state_indices(vec![1, 2, 4])
    .unwrap();
    let tensors: BTreeMap<_, _> = config
        .expected_weights()
        .unwrap()
        .into_iter()
        .map(|w| {
            let count = w.dimensions().iter().product::<u64>() as usize;
            let value = if w.external_name() == Family::FINAL_NORM_EXTERNAL_NAME {
                2.0
            } else if w.dimensions().len() == 1 {
                1.0
            } else {
                0.0
            };
            (
                w.external_name().to_owned(),
                (
                    w.dimensions()
                        .iter()
                        .map(|&x| x as usize)
                        .collect::<Vec<_>>(),
                    encoded(std::iter::repeat_n(value, count)),
                ),
            )
        })
        .collect();
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("weights.safetensors"),
    )
    .unwrap();
    let source =
        SafetensorsArchive::open_single_file(directory.path(), "weights.safetensors").unwrap();
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.text.selected"),
        1 << 20,
    )
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared(&config),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            descriptor.total_memory_bytes,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &config.compile_options().unwrap(),
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            12288,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 4096,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                request_id: id("request.text.selected.init"),
                run_id: id("run.text.selected.init"),
                transaction_id: id("transaction.text.selected.init"),
                static_policy: StaticInitializationPolicy::new(4096, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| panic!("initialization failed: {e:?}"));
    let embeddings = [
        1., 2., -1., 0., 3., -2., 1., 4., -2., 1., 3., -1., 0., 2., 4., -3.,
    ];
    let mut raw = BTreeMap::from([
        (
            id::<ProgramValueId>(Family::EMBEDDINGS_ID),
            encoded(embeddings),
        ),
        (id(Family::COSINE_ID), encoded([1.; 4])),
        (id(Family::SINE_ID), encoded([0.; 4])),
    ]);
    for (name, delta) in Family::DEEPSTACK_IDS.into_iter().zip([1., 2., 4.]) {
        raw.insert(id(name), encoded([delta; 8]));
    }
    let ports = compiled
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap();
    let inputs = ports
        .inputs()
        .iter()
        .map(|port| {
            let data = &raw[port.value_id()];
            assert_eq!(
                data.len() as u64,
                port.tensor().minimum_storage_bytes().unwrap()
            );
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
                .unwrap();
            bytes.as_mut().copy_from_slice(data);
            ManagedHostTensor::new(
                port.value_id().clone(),
                port.tensor().clone(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect();
    let outputs = session
        .execute(id("run.text.selected"), id("request.text.selected"), inputs)
        .unwrap_or_else(|e| panic!("execution failed: {e:?}"));
    let actual: BTreeMap<_, _> = outputs
        .iter()
        .map(|output| {
            assert_eq!(output.tensor().dimensions(), [2, 8]);
            assert_eq!(output.tensor().element_type(), ElementType::Bf16);
            (output.value_id().clone(), output.bytes().bytes().to_vec())
        })
        .collect();
    for (index, delta) in [(1, 1.), (2, 3.)] {
        let expected = embeddings
            .into_iter()
            .enumerate()
            .map(|(position, x)| x + if position < 8 { delta } else { 0. });
        assert_eq!(
            actual[&Family::hidden_state_output_id(index).unwrap()],
            encoded(expected)
        );
    }
    let pre_final = embeddings
        .into_iter()
        .enumerate()
        .map(|(position, x)| x + if position < 8 { 7. } else { 0. })
        .collect::<Vec<_>>();
    assert_eq!(
        actual[&id::<ProgramValueId>(Family::OUTPUT_ID)],
        encoded(pre_final.iter().copied())
    );
    let normalized = actual[&Family::hidden_state_output_id(4).unwrap()]
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect::<Vec<_>>();
    for (row, expected) in pre_final.chunks_exact(8).zip(normalized.chunks_exact(8)) {
        let inverse_rms = (row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / 8. + 0.000001)
            .sqrt()
            .recip();
        for (&x, &observed) in row.iter().zip(expected) {
            let mathematical = (f64::from(x) * inverse_rms * 2.) as f32;
            // Independent F64 math; allow the final BF16 rounding half-ULP.
            assert!((observed - mathematical).abs() <= mathematical.abs() / 256. + 1e-6);
        }
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
