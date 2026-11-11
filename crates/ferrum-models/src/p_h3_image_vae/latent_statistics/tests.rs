use super::*;
use crate::p_h3_image_vae::image_layout::PH3ImageVaeImageLayout;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;

pub(in crate::p_h3_image_vae) fn config() -> PH3ImageVaeDecoderConfig {
    PH3ImageVaeDecoderConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "_class_name":"AutoencoderKLPH3Image", "latent_channels":2,"out_channels":3,
            "spatial_downsample_factors":[2],"decoder_num_layers":2,"decoder_num_attention_heads":2,
            "decoder_attention_head_dim":16,"decoder_num_register_tokens":4,"decoder_ffn_mult":4,
            "decoder_rope_dim_ratio":0.75,"decoder_rope_theta":100,"decoder_norm_eps":0.00001
        }))
        .unwrap(),
    )
    .unwrap()
}
pub(in crate::p_h3_image_vae) fn source(
    dtype: safetensors::Dtype,
    mean: &[f32],
    std: &[f32],
) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let dir = tempfile::tempdir().unwrap();
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| {
                if dtype == safetensors::Dtype::F32 {
                    v.to_le_bytes().to_vec()
                } else {
                    half::bf16::from_f32(*v).to_bits().to_le_bytes().to_vec()
                }
            })
            .collect::<Vec<_>>()
    };
    let mean_bytes = encode(mean);
    let std_bytes = encode(std);
    let tensor =
        |bytes, length| safetensors::tensor::TensorView::new(dtype, vec![length], bytes).unwrap();
    safetensors::tensor::serialize_to_file(
        [
            ("latents_mean", tensor(&mean_bytes, mean.len())),
            ("latents_std", tensor(&std_bytes, std.len())),
        ],
        &None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        ComfyCanonicalWeightSource::new(SafetensorsArchive::open(dir.path()).unwrap()).unwrap();
    (dir, source)
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn input(
    host: &HostMemoryAccount,
    layout: &PH3ImageVaeImageLayout,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (b, v) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        b.copy_from_slice(&v.to_le_bytes());
    }
    ManagedHostTensor::new(
        ProgramValueId::new("value.normalized").unwrap(),
        layout.latent_spec().unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}

#[test]
fn original_statistics_keep_separate_f32_rounding_and_do_not_mutate_sampler_state() {
    let config = config();
    let (_dir, source) = source(
        safetensors::Dtype::F32,
        &[-1.5, 100.0],
        &[1.0 + f32::EPSILON, 0.5],
    );
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &source).unwrap();
    assert!(std::ptr::eq(statistics.source(), &source));
    assert_eq!(
        statistics.mean.as_ptr(),
        source
            .archive()
            .tensor("latents_mean")
            .unwrap()
            .bytes()
            .as_ptr()
    );
    let layout = PH3ImageVaeImageLayout::new(config, 2, 1, 1).unwrap();
    let host = host(128);
    let sample = input(&host, &layout, &[1.5, 4.0, 1.5, -2.0]);
    let original = sample.bytes().bytes().to_vec();
    let output = layout
        .denormalize_and_pack(
            &host,
            &sample,
            ProgramValueId::new("value.denoised").unwrap(),
            &statistics,
        )
        .unwrap();
    let values = output
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [
            f32::from_bits(0x34800000),
            102.0,
            f32::from_bits(0x34800000),
            99.0
        ]
    );
    assert_ne!(values[0], 1.5_f32.mul_add(1.0 + f32::EPSILON, -1.5));
    assert_eq!(sample.bytes().bytes(), original);
    drop((sample, output));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_original_headers_and_late_nonfinite_sample_fail_without_publishing() {
    let config = config();
    for (dtype, mean, std) in [
        (safetensors::Dtype::BF16, vec![0.0, 0.0], vec![1.0, 1.0]),
        (safetensors::Dtype::F32, vec![0.0], vec![1.0, 1.0]),
        (safetensors::Dtype::F32, vec![0.0, f32::NAN], vec![1.0, 1.0]),
        (safetensors::Dtype::F32, vec![0.0, 0.0], vec![1.0, 0.0]),
        (
            safetensors::Dtype::F32,
            vec![0.0, 0.0],
            vec![1.0, f32::INFINITY],
        ),
    ] {
        let (_dir, source) = source(dtype, &mean, &std);
        assert!(PH3ImageVaeLatentStatistics::new(&config, &source).is_err());
    }
    let (_dir, source) = source(safetensors::Dtype::F32, &[0.0, 0.0], &[1.0, 1.0]);
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &source).unwrap();
    let layout = PH3ImageVaeImageLayout::new(config, 2, 1, 1).unwrap();
    let host = host(128);
    let sample = input(&host, &layout, &[1.0, 2.0, 3.0, f32::NAN]);
    let before = host.usage().unwrap().committed_bytes();
    assert!(layout
        .denormalize_and_pack(
            &host,
            &sample,
            ProgramValueId::new("value.denoised").unwrap(),
            &statistics
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(sample);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
