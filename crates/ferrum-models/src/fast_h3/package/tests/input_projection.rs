use super::*;
use crate::fast_h3::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy, TensorComponentSession,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_runtime::*;
use half::bf16;
use std::sync::atomic::{AtomicUsize, Ordering};

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn input_limits() -> FastH3InputLimits {
    FastH3InputLimits {
        maximum_rows: 4096,
        maximum_segments: 32,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
    }
}
pub(super) fn host(bytes: usize) -> HostMemoryAccount {
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
fn layout() -> FastH3PackedLayout {
    FastH3PackedLayout::ref2va(
        3,
        &[
            FastH3Reference::Image {
                latent_height: 2,
                latent_width: 4,
            },
            FastH3Reference::Audio {
                latents_per_channel: 2,
            },
            FastH3Reference::Video {
                latent_frames: 2,
                latent_height: 4,
                latent_width: 2,
                audio_latents_per_channel: Some(3),
            },
            FastH3Reference::Image {
                latent_height: 2,
                latent_width: 2,
            },
        ],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        4096,
    )
    .unwrap()
}
pub(super) fn prepare<F: ModelFamilyProvider>(
    family: F,
    c: &F::Config,
    profile: &str,
) -> PreparedModelFamily
where
    F::Config: serde::Serialize,
{
    TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(profile))
        .unwrap()
}
pub(super) fn session(
    composition: &ReferenceVNextComposition,
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: ComponentWeightSource<'_>,
    host: &HostMemoryAccount,
) -> TensorComponentSession<ReferenceDeviceRuntime> {
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            source,
            ComponentInitializationConfig {
                run_id: id("run.input.init"),
                request_id: id("request.input.init"),
                transaction_id: id("transaction.input.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("input init failed; native owner retained")
        })
}
pub(super) fn close(s: TensorComponentSession<ReferenceDeviceRuntime>) {
    assert!(matches!(
        s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("input close failed; native owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
pub(super) fn execute(
    s: &mut TensorComponentSession<ReferenceDeviceRuntime>,
    inputs: Vec<ManagedHostTensor>,
) -> Vec<ferrum_execution::TensorExecutionOutput> {
    s.execute(id("run.input.execute"), id("request.input.execute"), inputs)
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("input execute failed; native owner retained")
        })
}
pub(super) fn managed(
    host: &HostMemoryAccount,
    name: &str,
    rows: u64,
    width: u64,
    dtype: ElementType,
    bytes: &[u8],
) -> ManagedHostTensor {
    let mut b = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    b.as_mut().copy_from_slice(bytes);
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(vec![rows, width], dtype, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        b.freeze(),
    )
    .unwrap()
}
fn bbytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}
fn values(f: &Fixture, name: &str) -> Vec<f32> {
    f.dense[name]
        .2
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
pub(super) struct Counting<'a> {
    pub(super) source: &'a dyn WeightComponentSource,
    pub(super) reads: AtomicUsize,
}
impl WeightComponentSource for Counting<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.source.component(c)
    }
}
fn fixture() -> Fixture {
    let mut f = Fixture::new();
    let h = config().hidden_size();
    for (module, width) in [
        ("proj_in", config().video_head_width()),
        ("audio_proj_in", config().audio_channels()),
        ("context_embedder", config().text_dim()),
    ] {
        for row in 0..h {
            for col in 0..width {
                let v = if row == 0 {
                    if col == 0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    ((row * 3 + col * 5) % 9) as f32 / 16. - 0.25
                };
                let i = (row * width + col) * 2;
                f.dense.get_mut(&format!("{module}.weight")).unwrap().2[i..i + 2]
                    .copy_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
            let bias = if row == 0 {
                -1.0
            } else {
                (row % 5) as f32 / 16.
            };
            f.dense.get_mut(&format!("{module}.bias")).unwrap().2[row * 2..row * 2 + 2]
                .copy_from_slice(&bf16::from_f32(bias).to_bits().to_le_bytes());
        }
    }
    f
}
#[test]
fn actual_dense_input_sources_preserve_f32_media_then_pack_with_explicit_refined_text() {
    let f = fixture();
    let parameters = ["proj_in", "audio_proj_in", "context_embedder"].map(|m| {
        (
            values(&f, &format!("{m}.weight")),
            values(&f, &format!("{m}.bias")),
        )
    });
    let (_dir, p) = f.open(limits());
    let p = p.unwrap();
    let layout = layout();
    let media =
        FastH3InputProjectionConfig::media_from_package(&p, &layout, input_limits()).unwrap();
    let text = FastH3InputProjectionConfig::text_from_package(&p, 3, input_limits()).unwrap();
    let rows = [media.rows()[0], media.rows()[1], text.rows()[0]];
    let widths = [
        media.input_widths()[0],
        media.input_widths()[1],
        text.input_widths()[0],
    ];
    let hidden = media.hidden_size();
    let account = host(4 << 20);
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.input_projection"),
        4 << 20,
    )
    .unwrap();
    let names = [
        FastH3InputProjectionFamily::VIDEO_INPUT_ID,
        FastH3InputProjectionFamily::AUDIO_INPUT_ID,
        FastH3InputProjectionFamily::TEXT_INPUT_ID,
    ];
    let out_names = [
        FastH3InputProjectionFamily::VIDEO_OUTPUT_ID,
        FastH3InputProjectionFamily::AUDIO_OUTPUT_ID,
        FastH3InputProjectionFamily::PROJECTED_TEXT_ID,
    ];
    let inputs = std::array::from_fn::<_, 3, _>(|i| {
        (0..rows[i] * widths[i])
            .map(|n| {
                if n % widths[i] == 0 {
                    if i < 2 {
                        1.0 + (1 + n / widths[i] % 4) as f32 / 256.
                    } else {
                        1.0078125
                    }
                } else {
                    ((n * 3 % 11) as f32 - 5.) / 8.
                }
            })
            .collect::<Vec<_>>()
    });
    let expected = std::array::from_fn::<_, 3, _>(|i| {
        inputs[i]
            .chunks_exact(widths[i] as usize)
            .flat_map(|x| {
                parameters[i]
                    .0
                    .chunks_exact(widths[i] as usize)
                    .zip(&parameters[i].1)
                    .map(move |(w, &b)| {
                        let dot = w
                            .iter()
                            .zip(x)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum::<f64>();
                        bf16::from_f32((dot as f32) + b).to_bits()
                    })
            })
            .collect::<Vec<_>>()
    });
    let mut projected = Vec::new();
    for c in [&media, &text] {
        let indices: &[usize] = if c.phase() == FastH3InputProjectionPhase::Media {
            &[0, 1]
        } else {
            &[2]
        };
        let source = FastH3InputProjectionSource::new(&p, c, &account, input_limits()).unwrap();
        let counted = Counting {
            source: &source,
            reads: AtomicUsize::new(0),
        };
        let prepared = prepare(
            FastH3InputProjectionFamily::new(input_limits()).unwrap(),
            c,
            c.phase().profile_id(),
        );
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("stored input sources")
        };
        assert_eq!(schema.components.len(), indices.len() * 2);
        assert!(schema
            .components
            .iter()
            .all(|component| component.physical_element_type()
                == if c.phase() == FastH3InputProjectionPhase::Media {
                    ElementType::F32
                } else {
                    ElementType::Bf16
                }));
        for component in &schema.components {
            let payload = source.component(component).unwrap();
            assert_eq!(
                payload.bytes().len() as u64,
                component.physical_bytes().unwrap()
            );
            assert!(payload.retained_host_memory().is_some());
        }
        let mut session = session(
            &composition,
            prepared,
            c.compile_options().unwrap(),
            ComponentWeightSource::Stored(&counted),
            &account,
        );
        assert_eq!(counted.reads.load(Ordering::SeqCst), indices.len() * 2);
        let mut retained = None;
        for _ in 0..2 {
            let tensors = indices
                .iter()
                .map(|&i| {
                    let bytes = if i < 2 {
                        inputs[i]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>()
                    } else {
                        bbytes(&inputs[i])
                    };
                    managed(
                        &account,
                        names[i],
                        rows[i],
                        widths[i],
                        if i < 2 {
                            ElementType::F32
                        } else {
                            ElementType::Bf16
                        },
                        &bytes,
                    )
                })
                .collect();
            let outputs = execute(&mut session, tensors);
            assert_eq!(outputs.len(), indices.len());
            for &i in indices {
                let out = outputs
                    .iter()
                    .find(|o| o.value_id().as_str() == out_names[i])
                    .unwrap();
                let actual = out
                    .bytes()
                    .bytes()
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual, expected[i],
                    "independent FP64 dot, F32 bias, final BF16"
                );
            }
            retained = Some(outputs);
        }
        assert_eq!(counted.reads.load(Ordering::SeqCst), indices.len() * 2);
        close(session);
        projected.extend(retained.unwrap());
    }
    assert_eq!(
        expected[0][0],
        bf16::from_f32(1. / 256.).to_bits(),
        "media input must not round BF16 before affine"
    );
    let packing = FastH3InputPackingConfig::from_layout(&layout, hidden, input_limits()).unwrap();
    let prepared = prepare(
        FastH3InputPackingFamily::new(input_limits()).unwrap(),
        &packing,
        FastH3InputPackingFamily::PROFILE_ID,
    );
    assert!(matches!(prepared.weights(), ModelWeights::None));
    let mut pack = session(
        &composition,
        prepared,
        packing.compile_options().unwrap(),
        ComponentWeightSource::None,
        &account,
    );
    // Explicit external refiner result. The context projection is intentionally
    // not used as this value: no text refiner has executed in this slice.
    let refined = (0..rows[2] * hidden)
        .map(|i| 8.0 + (i % 7) as f32 / 8.)
        .collect::<Vec<_>>();
    let mut operands = vec![managed(
        &account,
        FastH3InputPackingFamily::REFINED_TEXT_ID,
        rows[2],
        hidden,
        ElementType::Bf16,
        &bbytes(&refined),
    )];
    for (original, name, rows) in [
        (
            FastH3InputProjectionFamily::VIDEO_OUTPUT_ID,
            FastH3InputPackingFamily::VIDEO_ID,
            rows[0],
        ),
        (
            FastH3InputProjectionFamily::AUDIO_OUTPUT_ID,
            FastH3InputPackingFamily::AUDIO_ID,
            rows[1],
        ),
    ] {
        let original = projected
            .iter()
            .find(|o| o.value_id().as_str() == original)
            .unwrap();
        assert!(account.owns(original.bytes()));
        operands.push(managed(
            &account,
            name,
            rows,
            hidden,
            ElementType::Bf16,
            original.bytes().bytes(),
        ));
    }
    let outputs = execute(&mut pack, operands);
    close(pack);
    let actual = outputs[0].bytes().bytes();
    let h = hidden as usize;
    let mut cursors = [0usize; 3];
    let refined = refined
        .iter()
        .map(|&x| bf16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    for segment in layout.segments() {
        let stream = match segment.kind {
            FastH3SegmentKind::Text => 2,
            FastH3SegmentKind::ReferenceAudio | FastH3SegmentKind::TargetAudio => 1,
            _ => 0,
        };
        let source = if stream == 2 {
            &refined
        } else {
            &expected[stream]
        };
        for row in segment.packed_rows.clone() {
            for col in 0..h {
                let i = (row * h + col) * 2;
                assert_eq!(
                    u16::from_le_bytes(actual[i..i + 2].try_into().unwrap()),
                    source[cursors[stream] * h + col]
                );
            }
            cursors[stream] += 1;
        }
    }
    assert_eq!(
        cursors,
        [rows[0] as usize, rows[1] as usize, rows[2] as usize]
    );
    assert!(account.owns(outputs[0].bytes()));
    drop(projected);
    drop(outputs);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
#[test]
fn input_source_and_packing_reject_changed_dtype_incomplete_partition_and_failed_promotion_budget()
{
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    let l = layout();
    let c = FastH3InputProjectionConfig::media_from_package(&p, &l, input_limits()).unwrap();
    let small = host(4);
    let source = FastH3InputProjectionSource::new(&p, &c, &small, input_limits()).unwrap();
    let family = FastH3InputProjectionFamily::new(input_limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&c).unwrap() else {
        panic!("stored")
    };
    assert!(source.component(&schema.components[0]).is_err());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["widths"][0] = 1.into();
    let bad: FastH3InputProjectionConfig = serde_json::from_value(wire).unwrap();
    assert!(FastH3InputProjectionSource::new(&p, &bad, &small, input_limits()).is_err());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["components"][0]["encoding"] = serde_json::to_value(WeightEncoding::Dense {
        element_type: ElementType::Bf16,
    })
    .unwrap();
    assert!(family.parse_config(&wire).is_err());
    let bad: FastH3InputProjectionConfig = serde_json::from_value(wire).unwrap();
    assert!(FastH3InputProjectionSource::new(&p, &bad, &small, input_limits()).is_err());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["phase"] = "text".into();
    assert!(family.parse_config(&wire).is_err());
    let bad: FastH3InputProjectionConfig = serde_json::from_value(wire).unwrap();
    assert!(bad.compile_options().is_err());
    assert!(FastH3InputProjectionSource::new(&p, &bad, &small, input_limits()).is_err());
    assert!(TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(FastH3InputProjectionFamily::TEXT_PROFILE_ID),
        )
        .is_err());
    let mut too_small = input_limits();
    too_small.maximum_stored_bytes = 1;
    assert!(FastH3InputProjectionConfig::media_from_package(&p, &l, too_small).is_err());
    too_small = input_limits();
    too_small.maximum_segments = 2;
    assert!(FastH3InputPackingConfig::from_layout(&l, c.hidden_size(), too_small).is_err());
    let c = FastH3InputPackingConfig::from_layout(&l, c.hidden_size(), input_limits()).unwrap();
    let family = FastH3InputPackingFamily::new(input_limits()).unwrap();
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["runs"][0]["rows"] = 1.into();
    assert!(family.parse_config(&wire).is_err());
    wire = serde_json::to_value(&c).unwrap();
    wire["runs"][1]["stream"] = "refined_text".into();
    assert!(family.parse_config(&wire).is_err());
}

#[test]
fn deserialized_packing_compile_options_reject_overflow_and_incomplete_sources() {
    let canonical = serde_json::json!({
        "rows": [1, 1, 1],
        "hidden": 4,
        "runs": [
            { "stream": "refined_text", "rows": 1 },
            { "stream": "audio", "rows": 1 },
            { "stream": "video", "rows": 1 }
        ]
    });
    let c: FastH3InputPackingConfig = serde_json::from_value(canonical.clone()).unwrap();
    assert!(c.compile_options().is_ok());
    let mut extent = canonical.clone();
    extent["rows"][0] = u64::MAX.into();
    let mut incomplete = canonical.clone();
    incomplete["runs"][1]["rows"] = 2.into();
    let mut run_sum = canonical;
    run_sum["runs"] = serde_json::json!([
        { "stream": "refined_text", "rows": 1 },
        { "stream": "video", "rows": u64::MAX },
        { "stream": "audio", "rows": 1 },
        { "stream": "video", "rows": 1 }
    ]);
    for bad in [extent, incomplete, run_sum] {
        let c: FastH3InputPackingConfig = serde_json::from_value(bad).unwrap();
        assert!(c.compile_options().is_err());
    }
}
