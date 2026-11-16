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
#[cfg(feature = "cuda")]
mod cuda;
pub(in crate::fast_h3::video_vae) mod oracle;

pub(in crate::fast_h3::video_vae) fn architecture() -> FastH3VideoVaeConfig {
    let mut j = serde_json::to_value(super::super::tests::config()).unwrap();
    j["block_out_channels"] = serde_json::json!([2, 4]);
    j["layers_per_block"] = 2.into();
    j["norm_num_groups"] = 2.into();
    FastH3VideoVaeConfig::parse(&serde_json::to_vec(&j).unwrap(), 4096).unwrap()
}
pub(in crate::fast_h3::video_vae) fn limits() -> FastH3VideoEncoderLimits {
    FastH3VideoEncoderLimits {
        maximum_metadata_bytes: 4096,
        maximum_stored_bytes: 1 << 20,
        maximum_plan_nodes: 64,
        maximum_activation_bytes: 4 << 20,
    }
}
fn geometry(frames: u64) -> FastH3VideoEncoderGeometry {
    FastH3VideoEncoderGeometry {
        batch: 2,
        frames,
        height: 12,
        width: 14,
    }
}
pub(in crate::fast_h3::video_vae) type Weights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;
pub(in crate::fast_h3::video_vae) fn original_weights() -> Weights {
    fn conv(w: &mut Weights, name: &str, input: usize, output: usize, kernel: usize) {
        let shape = vec![output, input, kernel, kernel, kernel];
        // Nonzero, unequal taps across all three axes and all channels.
        let values = (0..shape.iter().product())
            .map(|i| ((i * 7 + 3) % 23) as f32 / 256. - 11. / 256.)
            .collect();
        w.insert(format!("{name}.weight"), (shape, values));
        w.insert(
            format!("{name}.bias"),
            (
                vec![output],
                (0..output).map(|i| (i as f32 - 1.) / 64.).collect(),
            ),
        );
    }
    fn norm(w: &mut Weights, name: &str, channels: usize) {
        w.insert(
            format!("{name}.weight"),
            (
                vec![channels],
                (0..channels).map(|i| 0.75 + i as f32 / 8.).collect(),
            ),
        );
        w.insert(
            format!("{name}.bias"),
            (
                vec![channels],
                (0..channels).map(|i| (i as f32 - 1.) / 16.).collect(),
            ),
        );
    }
    let mut w = Weights::new();
    conv(&mut w, "encoder.conv_in", 3, 2, 3);
    let mut input = 2;
    for (level, output) in [2, 4].into_iter().enumerate() {
        for block in 0..2 {
            let p = format!("encoder.down_blocks.{level}.resnets.{block}");
            norm(&mut w, &format!("{p}.norm1"), input);
            conv(&mut w, &format!("{p}.conv1"), input, output, 3);
            norm(&mut w, &format!("{p}.norm2"), output);
            conv(&mut w, &format!("{p}.conv2"), output, output, 3);
            if input != output {
                conv(&mut w, &format!("{p}.conv_shortcut"), input, output, 1);
            }
            input = output;
        }
        conv(
            &mut w,
            &format!("encoder.down_blocks.{level}.downsamplers.0.conv"),
            output,
            output,
            3,
        );
    }
    norm(&mut w, "encoder.norm_out", 4);
    conv(&mut w, "encoder.conv_out", 4, 4, 3);
    conv(&mut w, "quant_conv", 4, 4, 1);
    w
}
pub(in crate::fast_h3::video_vae) fn archive(
    w: &Weights,
) -> (tempfile::TempDir, Arc<SafetensorsArchive>) {
    let bytes: BTreeMap<_, Vec<u8>> = w
        .iter()
        .map(|(n, (_, values))| {
            (
                n.clone(),
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views = w.iter().map(|(n, (s, _))| {
        (
            n.as_str(),
            TensorView::new(Dtype::F32, s.clone(), &bytes[n]).unwrap(),
        )
    });
    let dir = tempfile::tempdir().unwrap();
    serialize_to_file(views, &None, &dir.path().join("encoder.safetensors")).unwrap();
    let a =
        Arc::new(SafetensorsArchive::open_single_file(dir.path(), "encoder.safetensors").unwrap());
    (dir, a)
}
pub(in crate::fast_h3::video_vae) fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8 << 20,
                output_bytes: 8 << 20,
                transfer_staging_bytes: 8 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
    )
}
fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    profile: FastH3VideoEncoderProfile,
) {
    let weights = original_weights();
    let (_dir, archive) = archive(&weights);
    let host = host();
    for frames in [1, 9] {
        let g = geometry(frames);
        let prepared = PreparedFastH3VideoEncoder::prepare(
            &architecture(),
            archive.clone(),
            g,
            limits(),
            profile,
        )
        .unwrap();
        assert_eq!(
            prepared.output_spec().dimensions(),
            [2, 4, frames.div_ceil(4), 4, 5]
        );
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
        let d = runtime.descriptor();
        let compiled = CompiledTensorComponent::compile(
            prepared.family().clone(),
            registry,
            &catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .unwrap(),
            &materializers,
            prepared.compile_options(),
        )
        .unwrap();
        let mut session = compiled
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(prepared.weight_source()),
                ComponentInitializationConfig {
                    run_id: id("run.video_encoder.init").unwrap(),
                    request_id: id("request.video_encoder.init").unwrap(),
                    transaction_id: id("transaction.video_encoder.init").unwrap(),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 128).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let len = prepared.input_spec().minimum_storage_bytes().unwrap() as usize / 4;
        let mut values: Vec<f32> = (0..len)
            .map(|i| ((i * 11 + i / 19) % 37) as f32 / 16. - 1.)
            .collect();
        let mut previous = vec![];
        for trial in 0..2 {
            if trial == 1 {
                for (i, v) in values.iter_mut().enumerate() {
                    if i as u64 / (g.height * g.width) % frames == frames - 1 && i % 3 == 0 {
                        *v += 2.;
                    }
                }
            }
            let expected = oracle::encode(&weights, &values, g);
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
                .unwrap();
            for (word, v) in bytes.as_mut().chunks_exact_mut(4).zip(&values) {
                word.copy_from_slice(&v.to_le_bytes());
            }
            let input = ManagedHostTensor::new(
                id(PreparedFastH3VideoEncoder::INPUT_ID).unwrap(),
                prepared.input_spec().clone(),
                bytes.freeze(),
            )
            .unwrap();
            let outputs = session
                .execute(
                    id("run.video_encoder.execute").unwrap(),
                    id("request.video_encoder.execute").unwrap(),
                    vec![input],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].tensor(), prepared.output_spec());
            let actual: Vec<f32> = outputs[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.is_finite() && (f64::from(a) - e).abs() < 8e-5 * (1. + e.abs()),
                    "T{frames}/trial{trial}/element{i}: {a} vs {e}"
                );
                if trial == 1 && frames > 1 && (i as u64 / 20 % frames.div_ceil(4)) * 4 < frames - 1
                {
                    assert_eq!(a, previous[i], "future frame leaked into an earlier latent");
                }
            }
            if trial == 1 {
                assert_ne!(actual, previous, "last temporal plane was not consumed");
            }
            previous = actual;
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!("full original causal encoder T={frames}, independent F64 moments and temporal isolation passed");
    }
}
#[test]
fn full_causal_encoder_matches_independent_f64_volume_and_future_isolation() {
    let c =
        ReferenceVNextComposition::create(id("device.video_encoder.reference").unwrap()).unwrap();
    qualify(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        FastH3VideoEncoderProfile::Reference,
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[test]
fn full_causal_encoder_rejects_capacity_and_last_tap_substitution_before_upload() {
    let a = architecture();
    let w = original_weights();
    let (_dir, archive) = archive(&w);
    for limit in [
        FastH3VideoEncoderLimits {
            maximum_plan_nodes: 4,
            ..limits()
        },
        FastH3VideoEncoderLimits {
            maximum_stored_bytes: 100,
            ..limits()
        },
        FastH3VideoEncoderLimits {
            maximum_activation_bytes: 100,
            ..limits()
        },
        FastH3VideoEncoderLimits {
            maximum_metadata_bytes: 10,
            ..limits()
        },
    ] {
        assert!(PreparedFastH3VideoEncoder::prepare(
            &a,
            archive.clone(),
            geometry(9),
            limit,
            FastH3VideoEncoderProfile::Reference
        )
        .is_err());
    }
    for g in [
        geometry(0),
        geometry(18),
        FastH3VideoEncoderGeometry {
            width: 2,
            ..geometry(9)
        },
        FastH3VideoEncoderGeometry {
            batch: u64::MAX,
            ..geometry(9)
        },
    ] {
        assert!(PreparedFastH3VideoEncoder::prepare(
            &a,
            archive.clone(),
            g,
            limits(),
            FastH3VideoEncoderProfile::Reference
        )
        .is_err());
    }
    let mut wrong = w;
    let (shape, data) = wrong.get_mut("encoder.conv_in.weight").unwrap();
    shape[2] = 1;
    data.truncate(shape.iter().product());
    let (_dir, archive) = self::archive(&wrong);
    assert!(PreparedFastH3VideoEncoder::prepare(
        &a,
        archive,
        geometry(9),
        limits(),
        FastH3VideoEncoderProfile::Reference
    )
    .is_err());
}
