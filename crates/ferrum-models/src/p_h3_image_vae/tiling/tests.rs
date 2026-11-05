use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn host(capacity: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            capacity,
            HostMemoryClassLimits {
                input_preprocess_bytes: capacity,
                output_bytes: capacity,
                weight_backing_bytes: 0,
                transfer_staging_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn tiny() -> PH3ImageVaeTilePlan {
    PH3ImageVaeTilePlan::new(
        6,
        6,
        2,
        PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels: 4,
            minimum_overlap_pixels: 2,
            max_tiles: 4,
        },
    )
    .unwrap()
}
fn output(host: &HostMemoryAccount, values: &[f32]) -> ManagedHostBytes {
    let mut b = host
        .allocate_zeroed(HostMemoryClass::Output, values.len() * 4)
        .unwrap();
    for (bytes, value) in b.as_mut().chunks_exact_mut(4).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    b.freeze()
}
fn tile(host: &HostMemoryAccount, index: usize, batch: usize, channels: usize) -> ManagedHostBytes {
    let values = (0..batch)
        .flat_map(|b| {
            (0..channels).flat_map(move |c| {
                std::iter::repeat_n((b * 100 + c * 1000 + index * 10) as f32, 16)
            })
        })
        .collect::<Vec<_>>();
    output(host, &values)
}
fn expected() -> [f32; 36] {
    [
        0., 0., 0., 5., 10., 10., 0., 0., 0., 5., 10., 10., 0., 0., 0., 5., 10., 10., 10., 10.,
        10., 15., 20., 20., 20., 20., 20., 25., 30., 30., 20., 20., 20., 25., 30., 30.,
    ]
}

fn statistics_source() -> (
    tempfile::TempDir,
    ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource,
    crate::p_h3_image_vae::PH3ImageVaeDecoderConfig,
) {
    use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
    let config = crate::p_h3_image_vae::PH3ImageVaeDecoderConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "_class_name":"AutoencoderKLPH3Image", "latent_channels":2,"out_channels":3,
            "spatial_downsample_factors":[2],"decoder_num_layers":2,"decoder_num_attention_heads":2,
            "decoder_attention_head_dim":16,"decoder_num_register_tokens":4,"decoder_ffn_mult":4,
            "decoder_rope_dim_ratio":0.75,"decoder_rope_theta":100,"decoder_norm_eps":0.00001
        }))
        .unwrap(),
    )
    .unwrap();
    let mean = [10.0_f32, 100.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let std = [2.0_f32, 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let tensor = |bytes| {
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], bytes).unwrap()
    };
    let dir = tempfile::tempdir().unwrap();
    safetensors::tensor::serialize_to_file(
        [
            ("latents_mean", tensor(&mean)),
            ("latents_std", tensor(&std)),
        ],
        &None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        ComfyCanonicalWeightSource::new(SafetensorsArchive::open(dir.path()).unwrap()).unwrap();
    (dir, source, config)
}

#[test]
fn tile_rows_crop_original_nchw_and_denormalize_in_tile_then_request_order() {
    let account = host(1 << 16);
    let plan = tiny();
    let (_dir, source, config) = statistics_source();
    let statistics = PH3ImageVaeLatentStatistics::new(&config, &source).unwrap();
    let values = (0..2)
        .flat_map(|b| {
            (0..2).flat_map(move |c| (0..9).map(move |p| (b * 100 + c * 1000 + p) as f32))
        })
        .collect::<Vec<_>>();
    let make_input = |values: &[f32]| {
        let mut bytes = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
            .unwrap();
        for (dst, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        ManagedHostTensor::new(
            ProgramValueId::new("value.sampler").unwrap(),
            f32_spec(&[2, 2, 3, 3]).unwrap(),
            bytes.freeze(),
        )
        .unwrap()
    };
    let input = make_input(&values);
    let original = input.bytes().bytes().to_vec();
    let before = account.usage().unwrap().committed_bytes();
    let id = ProgramValueId::new("value.tile_rows").unwrap();
    let rows = plan
        .managed_tile_rows(&account, &input, &statistics, 0, 2, id.clone())
        .unwrap();
    assert_eq!(rows.tensor().dimensions(), [16, 2]);
    let actual = rows
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            10., 600., 12., 600.5, 16., 601.5, 18., 602., 210., 650., 212., 650.5, 216., 651.5,
            218., 652., 12., 600.5, 14., 601., 18., 602., 20., 602.5, 212., 650.5, 214., 651.,
            218., 652., 220., 652.5,
        ]
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before + account.backing_bytes(rows.bytes().len()).unwrap()
    );
    assert_eq!(input.bytes().bytes(), original);
    drop(rows);
    for (start, count) in [(0, 0), (1, 2), (4, 1), (0, usize::MAX)] {
        assert!(plan
            .managed_tile_rows(&account, &input, &statistics, start, count, id.clone())
            .is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), before);
    }
    let foreign = host(1 << 16);
    assert!(plan
        .managed_tile_rows(&foreign, &input, &statistics, 0, 1, id.clone())
        .is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    let mut poisoned = values;
    poisoned[32] = f32::NAN; // Last request, second channel, last pixel of tile 1.
    let bad_input = make_input(&poisoned);
    let with_bad = account.usage().unwrap().committed_bytes();
    assert!(plan
        .managed_tile_rows(&account, &bad_input, &statistics, 0, 2, id)
        .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), with_bad);
    assert_eq!(input.bytes().bytes(), original);
    drop((input, bad_input));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn default_split_distributes_overlap_in_latent_units_and_covers_exact_canvas() {
    let p =
        PH3ImageVaeTilePlan::new(512, 256, 16, PH3ImageVaeSpatialPolicy::comfy_default(3)).unwrap();
    assert_eq!(p.x.starts, [0, 128, 256]);
    assert_eq!(p.x.overlaps, [128, 128]);
    assert_eq!(p.tile_count(), 3);
    let p = PH3ImageVaeTilePlan::new(1024, 256, 16, PH3ImageVaeSpatialPolicy::comfy_default(5))
        .unwrap();
    assert_eq!(p.x.starts, [0, 192, 384, 576, 768]);
    assert_eq!(p.x.overlaps, [64; 4]);
    let p =
        PH3ImageVaeTilePlan::new(272, 128, 16, PH3ImageVaeSpatialPolicy::comfy_default(2)).unwrap();
    assert_eq!(p.x.starts, [0, 16]);
    assert_eq!(p.x.overlaps, [240]);
    assert_eq!(p.y.extent, 128);
    let whole =
        PH3ImageVaeTilePlan::new(512, 256, 16, PH3ImageVaeSpatialPolicy::WholeImage).unwrap();
    assert_eq!(whole.tile_count(), 1);
    assert_eq!(whole.tile(0).unwrap().width, 512);
    for (width, height, scale, policy) in [
        (512, 256, 16, PH3ImageVaeSpatialPolicy::comfy_default(2)),
        (513, 256, 16, PH3ImageVaeSpatialPolicy::comfy_default(10)),
        (256, 0, 16, PH3ImageVaeSpatialPolicy::comfy_default(10)),
        (
            512,
            256,
            16,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 256,
                minimum_overlap_pixels: 256,
                max_tiles: 10,
            },
        ),
        (
            512,
            256,
            16,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 255,
                minimum_overlap_pixels: 64,
                max_tiles: 10,
            },
        ),
    ] {
        assert!(PH3ImageVaeTilePlan::new(width, height, scale, policy).is_err());
    }
}

#[test]
fn recursive_vertical_then_horizontal_seams_preserve_request_and_color_planes() {
    let host = host(1 << 16);
    let plan = tiny();
    let peak = PH3ImageVaeCanvas::output_peak(&host, &plan, 2, 2).unwrap();
    let mut canvas = PH3ImageVaeCanvas::new(plan, host.clone(), 2, 2).unwrap();
    let spec = f32_spec(&[2, 2, 4, 4]).unwrap();
    for i in 0..4 {
        let raw = tile(&host, i, 2, 2);
        canvas.push(i, &spec, &raw).unwrap();
        assert!(
            host.usage().unwrap().committed_bytes()
                <= peak + host.backing_bytes(raw.len()).unwrap()
        );
    }
    let (spec, bytes) = canvas.finish(PH3ImageVaeCanvasDomain::RawDecoder).unwrap();
    assert_eq!(spec.dimensions(), [2, 2, 6, 6]);
    for b in 0..2 {
        for c in 0..2 {
            for (p, value) in expected().into_iter().enumerate() {
                assert_eq!(
                    read(bytes.bytes(), ((b * 2 + c) * 36 + p) * 4),
                    value + (b * 100 + c * 1000) as f32
                );
            }
        }
    }
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        host.backing_bytes(bytes.len()).unwrap()
    );
    drop(bytes);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn late_strip_admission_failure_refunds_temporary_owners_and_retries_same_tile() {
    let probe = host(1 << 16);
    let plan = tiny();
    let source_bytes = probe.backing_bytes(4 * 4 * 4).unwrap();
    let peak = PH3ImageVaeCanvas::output_peak(&probe, &plan, 1, 1).unwrap();
    let first_peak = [6 * 6 * 4, 4 * 4 * 4, 4 * 4 * 4, 4 * 2 * 4, 2 * 6 * 4]
        .into_iter()
        .map(|n| probe.backing_bytes(n).unwrap())
        .sum::<usize>();
    let capacity = peak + source_bytes;
    let account = host(capacity);
    let mut canvas = PH3ImageVaeCanvas::new(plan, account.clone(), 1, 1).unwrap();
    let spec = f32_spec(&[1, 1, 4, 4]).unwrap();
    let raw = tile(&account, 0, 1, 1);
    let blocker = account
        .allocate_zeroed(HostMemoryClass::Output, capacity - first_peak + 1)
        .unwrap();
    let before = account.usage().unwrap().committed_bytes();
    assert!(canvas.push(0, &spec, &raw).is_err());
    assert_eq!(canvas.completed_tiles(), 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), before);
    drop(blocker);
    canvas.push(0, &spec, &raw).unwrap();
    drop(raw);
    for i in 1..4 {
        canvas.push(i, &spec, &tile(&account, i, 1, 1)).unwrap();
    }
    let (_, bytes) = canvas.finish(PH3ImageVaeCanvasDomain::RawDecoder).unwrap();
    let actual = bytes
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected());
    drop(bytes);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bad_tile_order_nonfinite_or_foreign_owner_cannot_advance_or_publish_partial_canvas() {
    let account = host(1 << 16);
    let mut canvas = PH3ImageVaeCanvas::new(tiny(), account.clone(), 1, 1).unwrap();
    let spec = f32_spec(&[1, 1, 4, 4]).unwrap();
    let raw = tile(&account, 0, 1, 1);
    assert!(canvas.push(1, &spec, &raw).is_err());
    let foreign = host(1 << 16);
    assert!(canvas.push(0, &spec, &tile(&foreign, 0, 1, 1)).is_err());
    let mut poisoned = [0.0; 16];
    poisoned[15] = f32::NAN;
    assert!(canvas.push(0, &spec, &output(&account, &poisoned)).is_err());
    assert_eq!(canvas.completed_tiles(), 0);
    canvas.push(0, &spec, &raw).unwrap();
    drop(raw);
    assert!(canvas.finish(PH3ImageVaeCanvasDomain::RawDecoder).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn unit_rgb_finalization_happens_after_raw_blending_and_streams_through_shared_pixels() {
    let account = host(1 << 16);
    let mut canvas = PH3ImageVaeCanvas::new(tiny(), account.clone(), 1, 3).unwrap();
    let spec = f32_spec(&[1, 3, 4, 4]).unwrap();
    for i in 0..4 {
        // Extreme raw tiles make pre-blend clamping observably wrong.
        let values = [-10., 10., -10., 10.];
        canvas
            .push(i, &spec, &output(&account, &[values[i]; 48]))
            .unwrap();
    }
    let (spec, data) = canvas.finish(PH3ImageVaeCanvasDomain::UnitRgb).unwrap();
    // At x=3 the raw blend is zero; final means must survive rather than the
    // 0.5 that early per-tile clipping would produce.
    for (c, mean) in [0.485_f32, 0.456, 0.406].into_iter().enumerate() {
        assert_eq!(read(data.bytes(), (c * 36 + 3) * 4), mean);
    }
    let pixels = crate::image_pixels::ManagedImagePixels::new(
        &account,
        &spec,
        &data,
        crate::image_pixels::ImagePixelEncoding::F32UnitRgbRne,
    )
    .unwrap();
    let mut encoded = Vec::new();
    pixels
        .write_frame_rgba8(0, &mut [0; 7], &mut encoded)
        .unwrap();
    assert_eq!(encoded.len(), 6 * 6 * 4);
    assert!(encoded.chunks_exact(4).all(|p| p[3] == 255));
}
