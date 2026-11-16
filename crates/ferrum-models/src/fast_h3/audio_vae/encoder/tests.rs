use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn architecture() -> FastH3AudioVaeConfig {
    let mut a = super::super::tests::architecture();
    a.encoder_dim = 2;
    a.encoder_rates = vec![2, 3];
    a.decoder_dim = 4;
    a.decoder_rates = vec![2, 3];
    a.decoder_kernel_sizes = vec![4, 5];
    a.latent_dim = 4;
    a.num_attention_heads = 2;
    a.validate().unwrap();
    a
}
fn fixture(a: &FastH3AudioVaeConfig) -> (tempfile::TempDir, Arc<SafetensorsArchive>) {
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    for e in entries(a).unwrap() {
        let (shape, v_name, g_name) = match &e.kind {
            EntryKind::Original { source, shape } => {
                let values = vec![0.5f32; shape.iter().product::<u64>() as usize];
                tensors.insert(
                    source.clone(),
                    (
                        shape.iter().map(|&d| d as usize).collect(),
                        values.into_iter().flat_map(f32::to_le_bytes).collect(),
                    ),
                );
                continue;
            }
            EntryKind::WeightNorm { v, g, shape } => (
                shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                v.clone(),
                g.clone(),
            ),
            EntryKind::Zero => panic!("encoder trunk has original biases"),
            EntryKind::ConcatVectors { .. } => panic!("encoder trunk has no concatenated vectors"),
        };
        let k = shape[2];
        let (gain, tap) = if e.name.ends_with(".block.1.weight") && e.name.split('.').count() == 8 {
            (0.125f32, 4)
        } else if e.name.ends_with(".block.3.weight") && e.name.split('.').count() == 8 {
            (0.25, 0)
        } else if e.name.ends_with(".block.4.weight") && e.name.split('.').count() == 6 {
            (1., (k / 2).div_ceil(2) + 1)
        } else {
            (1., k / 2)
        };
        let mut v = vec![0f32; shape.iter().product()];
        for row in 0..shape[0] {
            v[row * shape[1] * k + tap] = 1.;
        }
        tensors.insert(
            v_name,
            (
                shape.clone(),
                v.into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
        );
        tensors.insert(
            g_name,
            (
                vec![shape[0], 1, 1],
                vec![gain; shape[0]]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
        );
    }
    // Original vector biases are exactly zero; Snake alpha is direct 0.5.
    for (name, (_, bytes)) in &mut tensors {
        if name.ends_with(".bias") {
            bytes.fill(0);
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &dir.path().join("encoder.safetensors")).unwrap();
    let source =
        Arc::new(SafetensorsArchive::open_single_file(dir.path(), "encoder.safetensors").unwrap());
    (dir, source)
}

// Independent scalar signal oracle: each matrix reads channel zero and all
// output channels are identical. Deliberately offset k7 taps exercise dilation
// and zero padding; downsample taps select offset 1, including right padding.
fn oracle(input: &[f32], padded: usize) -> Vec<f64> {
    let snake = |x: f64| x + (0.5 * x).sin().powi(2) / (0.5 + f64::from(1e-9f32));
    let mut x = input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    x.resize(padded, 0.);
    for stride in [2usize, 3] {
        for dilation in [1usize, 3, 9] {
            let y = x.iter().map(|&v| snake(v)).collect::<Vec<_>>();
            x = x
                .iter()
                .enumerate()
                .map(|(i, &v)| v + 0.25 * snake(0.125 * y.get(i + dilation).copied().unwrap_or(0.)))
                .collect();
        }
        x = x.iter().map(|&v| snake(v)).collect();
        x = (0..x.len() / stride)
            .map(|i| x.get(i * stride + 1).copied().unwrap_or(0.))
            .collect();
    }
    x.into_iter().map(snake).collect()
}

#[test]
fn original_encoder_trunk_executes_dilated_snake_stride_padding_and_independent_channels() {
    let a = architecture();
    let (_dir, archive) = fixture(&a);
    let host = super::super::tests::account(16 << 20);
    let limits = super::super::tests::limits();
    let expected_backing =
        PreparedFastH3AudioEncoderTrunk::host_weight_backing_bytes(&a, &host).unwrap();
    let prepared =
        PreparedFastH3AudioEncoderTrunk::prepare(&a, archive, 2, 7, &host, limits).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), expected_backing);
    assert_eq!(prepared.input_spec().dimensions(), [2, 1, 1, 12]);
    assert_eq!(prepared.output_spec().dimensions(), [2, 4, 1, 2]);
    let composition =
        ReferenceVNextComposition::create(id("device.audio.encoder.test").unwrap()).unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared.family().clone(),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        prepared.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&prepared),
            ComponentInitializationConfig {
                run_id: id("run.encoder.init").unwrap(),
                request_id: id("request.encoder.init").unwrap(),
                transaction_id: id("transaction.encoder.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let values = [
        -0.75f32, 0.25, 0.125, 0.5, -0.25, 0.375, -0.125, 0.0625, -0.5, 0.25, 0.375, 0.75, -0.75,
        0.125,
    ];
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (word, v) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        word.copy_from_slice(&v.to_le_bytes());
    }
    let raw = ManagedHostTensor::new(
        id("value.encoder.source").unwrap(),
        ResolvedTensorSpec::new(
            vec![2, 1, 7],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap();
    let input = prepared.managed_input(&raw, &host).unwrap();
    for b in 0..2 {
        assert!(input.bytes().bytes()[4 * (b * 12 + 7)..4 * (b + 1) * 12]
            .iter()
            .all(|&v| v == 0));
    }
    drop(raw);
    drop(prepared);
    let outputs = session
        .execute(
            id("run.encoder.test").unwrap(),
            id("request.encoder.test").unwrap(),
            vec![input.clone()],
        )
        .unwrap();
    let actual = outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    for b in 0..2 {
        let expected = oracle(&values[b * 7..b * 7 + 7], 12);
        for c in 0..4 {
            for t in 0..2 {
                let observed = f64::from(actual[(b * 4 + c) * 2 + t]);
                assert!(
                    (observed - expected[t]).abs() < 2e-6,
                    "batch={b} channel={c} time={t}: {observed} != {}",
                    expected[t]
                );
            }
        }
    }
    assert_ne!(actual[0], actual[8]);
    assert!(session.close().is_ok());
    drop(input);
    drop(outputs);
    drop(compiled);
    drop(composition);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn encoder_rejects_capacity_and_layout_before_execution_without_retaining_backing() {
    let a = architecture();
    assert_eq!(a.encoder_padded_samples(12).unwrap(), 12);
    assert!(a.encoder_padded_samples(0).is_err());
    assert!(a.encoder_padded_samples(u64::MAX).is_err());
    let (_dir, archive) = fixture(&a);
    let host = super::super::tests::account(16 << 20);
    let mut limits = super::super::tests::limits();
    limits.maximum_activation_bytes = 95;
    assert!(
        PreparedFastH3AudioEncoderTrunk::prepare(&a, archive.clone(), 2, 7, &host, limits).is_err()
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    limits.maximum_activation_bytes = 16 << 20;
    limits.maximum_stored_bytes = 1;
    assert!(
        PreparedFastH3AudioEncoderTrunk::prepare(&a, archive.clone(), 2, 7, &host, limits).is_err()
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let prepared = PreparedFastH3AudioEncoderTrunk::prepare(
        &a,
        archive,
        2,
        7,
        &host,
        super::super::tests::limits(),
    )
    .unwrap();
    let raw = ManagedHostTensor::new(
        id("value.encoder.bad_layout").unwrap(),
        ResolvedTensorSpec::new(
            vec![1, 2, 7],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        host.allocate_zeroed(HostMemoryClass::InputPreprocess, 56)
            .unwrap()
            .freeze(),
    )
    .unwrap();
    assert!(prepared.managed_input(&raw, &host).is_err());
    drop(raw);
    drop(prepared);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
