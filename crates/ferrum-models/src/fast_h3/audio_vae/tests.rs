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
use std::sync::atomic::{AtomicUsize, Ordering};
fn iid<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.to_owned()).unwrap()
}
pub(super) fn architecture() -> FastH3AudioVaeConfig {
    FastH3AudioVaeConfig::parse(&serde_json::to_vec(&serde_json::json!({"_class_name":"AutoencoderKLMiniMaxH3Audio","encoder_dim":2,"encoder_rates":[2,4,4,5,5],"num_attention_heads":1,"latent_channels":2,"latent_dim":2,"decoder_dim":128,"decoder_rates":[5,5,2,2,2,2,2],"decoder_kernel_sizes":[9,9,4,4,4,4,4],"resblock_kernel_sizes":[3,7,11],"resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]],"sampling_rate":32000,"latents_mean":[0.,0.],"latents_std":[1.,1.]})).unwrap(),4096).unwrap()
}
pub(super) fn limits() -> FastH3AudioDecoderLimits {
    FastH3AudioDecoderLimits {
        maximum_metadata_bytes: 8 << 20,
        maximum_source_bytes: 16 << 20,
        maximum_stored_bytes: 16 << 20,
        maximum_plan_nodes: 1024,
        maximum_activation_bytes: 16 << 20,
    }
}
pub(super) fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn fixture(a: &FastH3AudioVaeConfig) -> (tempfile::TempDir, Arc<SafetensorsArchive>) {
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    for e in source::entries(a).unwrap() {
        let names = e.names();
        if names[0].starts_with("derived.") {
            continue;
        }
        for name in names {
            if tensors.contains_key(&name) {
                continue;
            }
            let shape = if name.ends_with("weight_g") {
                vec![e.physical[0] as usize, 1, 1]
            } else if name.ends_with("weight_v") || name == "dec_in_proj.weight" {
                vec![
                    e.physical[0] as usize,
                    e.physical[1] as usize,
                    e.physical[3] as usize,
                ]
            } else if name.ends_with("filter") {
                vec![1, 1, 12]
            } else {
                e.physical.iter().map(|&n| n as usize).collect()
            };
            let mut values = vec![0f32; shape.iter().product()];
            if name.ends_with("act.alpha") {
                values.fill(-1000.);
            } else if name.ends_with("upsample.filter") {
                values[5] = 1.;
            } else if name.ends_with("downsample.lowpass.filter") {
                values[5] = 0.5;
            } else if name.ends_with("weight_g") {
                if !name.contains("resblocks.") {
                    values[0] = 1.;
                }
            } else if name.ends_with("weight_v") {
                let row = shape[1] * shape[2];
                let tap = if name.contains("decoder.ups.") {
                    (shape[2] - if shape[2] == 9 { 5 } else { 2 }) / 2
                } else {
                    shape[2] / 2
                };
                for r in 0..shape[0] {
                    values[r * row + tap] = 1.;
                }
            } else if name == "dec_in_proj.weight" {
                values[0] = 1.;
            }
            tensors.insert(
                name,
                (
                    shape,
                    values.into_iter().flat_map(f32::to_le_bytes).collect(),
                ),
            );
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
    serialize_to_file(views, &None, &dir.path().join("audio.safetensors")).unwrap();
    let source =
        Arc::new(SafetensorsArchive::open_single_file(dir.path(), "audio.safetensors").unwrap());
    (dir, source)
}
struct Count<'a> {
    source: &'a PreparedFastH3AudioDecoder,
    reads: AtomicUsize,
}
impl WeightComponentSource for Count<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}

#[test]
fn original_audio_complete_decoder_preserves_stereo_batches_and_retires_managed_sources() {
    let a = architecture();
    let (_dir, archive) = fixture(&a);
    let host = account(64 << 20);
    let prepared = PreparedFastH3AudioDecoder::prepare(&a, archive, 2, 2, &host, limits()).unwrap();
    assert_eq!(prepared.original_source_tensors(), 914);
    assert_eq!(prepared.input_spec().dimensions(), [4, 2, 1, 2]);
    assert_eq!(prepared.output_spec().dimensions(), [4, 1, 1, 1600]);
    let component_count = match prepared.family().weights() {
        ModelWeights::Stored(s) => s.components.len(),
        _ => panic!("original Stored audio"),
    };
    let source = Count {
        source: &prepared,
        reads: AtomicUsize::new(0),
    };
    let composition = ReferenceVNextComposition::create(iid("device.reference.audio")).unwrap();
    let descriptor = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared.family().clone(),
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
        prepared.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: iid("run.audio.init"),
                request_id: iid("request.audio.init"),
                transaction_id: iid("transaction.audio.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    assert_eq!(source.reads.load(Ordering::Relaxed), component_count);
    let input_spec = prepared.input_spec().clone();
    drop(source);
    drop(prepared);
    let values = [
        -2f32, -0.5, 99., 99., 0.25, 2., 99., 99., 0.5, -0.25, 99., 99., 1., -1., 99., 99.,
    ];
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (word, x) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        word.copy_from_slice(&x.to_le_bytes());
    }
    let input = ManagedHostTensor::new(
        iid(PreparedFastH3AudioDecoder::INPUT_ID),
        input_spec,
        bytes.freeze(),
    )
    .unwrap();
    let mut retained = None;
    for run in 0..2 {
        let outputs = session
            .execute(
                iid(&format!("run.audio.{run}")),
                iid(&format!("request.audio.{run}")),
                vec![input.clone()],
            )
            .unwrap();
        let actual = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for b in 0..4 {
            for t in 0..1600 {
                let expected = if t % 800 == 0 {
                    values[b * 4 + t / 800].clamp(-1., 1.)
                } else {
                    0.
                };
                assert_eq!(actual[b * 1600 + t], expected, "batch={b} sample={t}");
            }
        }
        retained = Some(outputs);
    }
    assert!(session.close().is_ok());
    drop(input);
    assert!(retained.as_ref().unwrap()[0].bytes().bytes().len() > 0);
    drop(retained);
    drop(compiled);
    drop(composition);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn audio_weight_derivation_capacity_failure_rolls_back_all_host_backing() {
    let a = architecture();
    let (_dir, archive) = fixture(&a);
    let host = account(4);
    assert!(PreparedFastH3AudioDecoder::prepare(&a, archive, 1, 1, &host, limits()).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
