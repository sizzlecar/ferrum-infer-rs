use super::*;
use crate::fast_h3::{FastH3Checkpoint, FastH3MetadataLimits};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use half::bf16;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
type F = FastH3PrefixAttentionFamily;
#[cfg(feature = "cuda")]
mod cuda;
fn limits() -> FastH3PrefixAttentionLimits {
    FastH3PrefixAttentionLimits {
        center: FastH3KeyCenterLimits {
            maximum_batches: 2,
            maximum_rows: 4096,
            maximum_columns: 256,
            maximum_tensor_bytes: 2 << 20,
            maximum_metadata_bytes: 4096,
        },
        transport: FastH3VsaRowTransportLimits {
            maximum_rows: 4096,
            maximum_columns: 256,
            maximum_batches: 2,
            maximum_map_bytes: 16384,
            maximum_metadata_bytes: 4096,
        },
        maximum_plan_nodes: 32,
        maximum_stored_bytes: 16384,
        maximum_metadata_bytes: 1 << 20,
    }
}
fn transformer() -> FastH3TransformerConfig {
    let mut wire: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../fixtures/transformer.json")).unwrap();
    wire["num_attention_heads"] = 1.into();
    wire["attention_head_dim"] = 128.into();
    wire["rope_freq_dim"] = 16.into();
    FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        &serde_json::to_vec(&wire).unwrap(),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap()
    .transformer()
    .clone()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 16 << 20,
                output_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    )
}
struct Counted {
    source: FastH3PrefixAttentionSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
fn managed(h: &HostMemoryAccount, name: &str, shape: Vec<u64>, v: &[u16]) -> ManagedHostTensor {
    let mut bytes = h
        .allocate_zeroed(HostMemoryClass::InputPreprocess, v.len() * 2)
        .unwrap();
    for (b, x) in bytes.as_mut().chunks_exact_mut(2).zip(v) {
        b.copy_from_slice(&x.to_le_bytes());
    }
    ManagedHostTensor::new(id(name).unwrap(), resolved(shape).unwrap(), bytes.freeze()).unwrap()
}

// Independent nearest-neighbor code table: no production quantization helper.
fn roundtrip(x: &[u16], d: usize) -> Vec<u16> {
    let table = (0..127)
        .map(|code| {
            let e = code >> 3;
            let m = code & 7;
            if e == 0 {
                m as f32 / 512.
            } else {
                (1. + m as f32 / 8.) * 2f32.powi(e - 7)
            }
        })
        .collect::<Vec<_>>();
    x.chunks_exact(d)
        .flat_map(|row| {
            let max = row
                .iter()
                .map(|&b| bf16::from_bits(b).to_f32().abs())
                .fold(0f32, f32::max);
            let sf = max / 448.;
            let inv = if sf == 0. { 0. } else { 1. / sf };
            row.iter()
                .map(|&b| {
                    let v = (bf16::from_bits(b).to_f32() * inv).clamp(-448., 448.);
                    let mut best = 0;
                    let mut distance = f64::INFINITY;
                    for (i, &code) in table.iter().enumerate() {
                        let delta = (f64::from(v.abs()) - f64::from(code)).abs();
                        if delta < distance || (delta == distance && i % 2 == 0) {
                            best = i;
                            distance = delta;
                        }
                    }
                    bf16::from_f32(table[best].copysign(v) * sf).to_bits()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
fn values(c: &FastH3PrefixAttentionConfig) -> [Vec<u16>; 3] {
    std::array::from_fn(|port| {
        (0..c.batches() * c.padded_rows() * c.columns())
            .map(|i| {
                let b = i / (c.padded_rows() * c.columns());
                let r = (i / c.columns()) % c.padded_rows();
                let channel = i % c.columns();
                let x = match port {
                    0 => ((r + channel * 3 + b) % 17) as f32 / 16. - 0.5,
                    1 => ((r * 7 + channel + b * 3) % 19) as f32 / 32. - 0.25,
                    _ => 0.5 + ((r * 11 + channel * 3 + b) % 23) as f32 / 32.,
                };
                bf16::from_f32(x).to_bits()
            })
            .collect()
    })
}
fn oracle(
    c: &FastH3PrefixAttentionConfig,
    layout: &FastH3VsaLayout,
    x: &[Vec<u16>; 3],
) -> Vec<u16> {
    let b = c.batches() as usize;
    let s = c.padded_rows() as usize;
    let cols = c.columns() as usize;
    let p = c.prefix_rows as usize;
    let d = c.head_dim as usize;
    let mut key = x[1].clone();
    for batch in 0..b {
        for col in 0..cols {
            let sum = (0..s)
                .map(|r| f64::from(bf16::from_bits(x[1][(batch * s + r) * cols + col]).to_f32()))
                .sum::<f64>();
            let mean = bf16::from_f32((sum / s as f64) as f32).to_f32();
            for r in 0..s {
                let i = (batch * s + r) * cols + col;
                key[i] = bf16::from_f32(bf16::from_bits(key[i]).to_f32() - mean).to_bits();
            }
        }
    }
    let key = roundtrip(&key, d);
    let query = roundtrip(&x[0], d);
    let valid = (0..layout.tile_count())
        .flat_map(|tile| {
            (0..usize::from(layout.valid_rows_per_tile()[tile])).map(move |r| tile * 128 + r)
        })
        .collect::<Vec<_>>();
    let mut out = vec![0; b * p * cols];
    let scale = f64::from((1. / (d as f64).sqrt()) as f32);
    for batch in 0..b {
        for row in 0..p {
            for head in 0..cols / d {
                let qi = (batch * s + row) * cols + head * d;
                let scores = valid
                    .iter()
                    .map(|&r| {
                        let ki = (batch * s + r) * cols + head * d;
                        (0..d)
                            .map(|j| {
                                f64::from(bf16::from_bits(query[qi + j]).to_f32())
                                    * f64::from(bf16::from_bits(key[ki + j]).to_f32())
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities = scores.iter().map(|&v| (v - max).exp()).collect::<Vec<_>>();
                let sum = probabilities.iter().sum::<f64>();
                for j in 0..d {
                    let y = valid
                        .iter()
                        .zip(&probabilities)
                        .map(|(&r, &prob)| {
                            prob / sum
                                * f64::from(
                                    bf16::from_bits(x[2][(batch * s + r) * cols + head * d + j])
                                        .to_f32(),
                                )
                        })
                        .sum::<f64>();
                    out[(batch * p + row) * cols + head * d + j] =
                        bf16::from_f32(y as f32).to_bits();
                }
            }
        }
    }
    out
}
pub(super) fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    arithmetic: FastH3PrefixAttentionArithmetic,
) {
    let layout = super::super::tests::layout();
    let h = host();
    let (c, source) =
        FastH3PrefixAttentionSource::prepare(&layout, 2, &transformer(), &h, limits()).unwrap();
    let c = c.with_attention_arithmetic(arithmetic, limits()).unwrap();
    let source = Counted {
        source,
        reads: AtomicUsize::new(0),
    };
    let prepared = TypedFamilyRegistration::new(F::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(c.profile_id()).unwrap(),
        )
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("stored map")
    };
    assert_eq!(schema.components.len(), 2);
    assert_eq!(schema.tensors.len(), 2);
    let d = runtime.descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        registry,
        catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        materializers,
        &c.compile_options(limits()).unwrap(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            runtime.clone(),
            h.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.prefix.init").unwrap(),
                request_id: id("request.prefix.init").unwrap(),
                transaction_id: id("transaction.prefix.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(16384, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("prefix initialization owners retained")
        });
    assert_eq!(source.reads.load(Ordering::Relaxed), 2);
    drop(source);
    let x = values(&c);
    let expected = oracle(&c, &layout, &x);
    let mut retained = None;
    let mut first = None;
    for _ in 0..2 {
        let inputs = [F::QUERY_ID, F::KEY_ID, F::VALUE_ID]
            .iter()
            .zip(&x)
            .map(|(name, v)| managed(&h, name, c.input_shape(), v))
            .collect();
        let outputs = match session.execute(
            id("run.prefix.execute").unwrap(),
            id("request.prefix.execute").unwrap(),
            inputs,
        ) {
            Ok(outputs) => outputs,
            Err(e) => {
                let reason = format!("{e:?}");
                std::mem::forget((session, e));
                panic!("prefix execution owners retained: {reason}");
            }
        };
        assert_eq!(outputs.len(), 2);
        let prefix = outputs
            .iter()
            .find(|o| o.value_id().as_str() == F::OUTPUT_ID)
            .unwrap();
        let centered = outputs
            .iter()
            .find(|o| o.value_id().as_str() == F::CENTERED_KEY_ID)
            .unwrap();
        assert_eq!(centered.tensor().dimensions(), c.input_shape());
        assert_eq!(prefix.tensor().dimensions(), c.output_shape());
        let actual = prefix
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        // Declared before CUDA execution: finite positive outputs, at most two
        // BF16 ULP vs independent FP64 dot/exp/PV and table-based RN FP8. This
        // is a tiny component bound, not upstream full-pipeline bit parity.
        assert!(actual
            .iter()
            .all(|&b| bf16::from_bits(b).is_finite() && b < 0x8000));
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(max <= 2, "prefix maximum BF16 ULP={max}");
        if let Some(first) = &first {
            assert_eq!(first, &actual);
        } else {
            first = Some(actual);
        }
        retained = Some(outputs);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("prefix close owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(h.owns(retained.as_ref().unwrap()[0].bytes()));
    drop(retained);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn prefix_reference_centers_quantizes_then_gathers_keys_in_one_managed_graph() {
    let r = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.prefix").unwrap(),
        32 << 20,
    )
    .unwrap();
    qualify(
        r.runtime().clone(),
        r.registry(),
        r.catalog(),
        r.weight_materializers(),
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
    );
    assert_eq!(r.runtime().snapshot().live_allocations, 0);
}
#[test]
fn prefix_source_shares_one_sorted_valid_map_and_rejects_capacity_or_wire_mismatch() {
    let layout = super::super::tests::layout();
    let h = host();
    let (c, s) =
        FastH3PrefixAttentionSource::prepare(&layout, 2, &transformer(), &h, limits()).unwrap();
    let p = program::compose(&c, limits()).unwrap();
    assert!(p.program.blocks()[0]
        .nodes
        .iter()
        .any(|n| n.operation_id.as_str() == INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID));
    assert!(!p.options.tensor_specs().contains_key(
        &id::<ProgramValueId>("value.fast_h3.vsa.prefix_attention.key_fp8").unwrap()
    ));
    assert_eq!(p.schema.components.len(), 2);
    let payload = s.component(s.valid.component_spec()).unwrap();
    let map = payload.bytes();
    let actual = map
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = (0..layout.tile_count())
        .flat_map(|t| (0..layout.valid_rows_per_tile()[t] as i32).map(move |r| t as i32 * 128 + r))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(actual.windows(2).all(|x| x[0] < x[1]));
    assert_ne!(
        actual,
        layout
            .packed_to_tile_rows()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>()
    );
    drop(payload);
    drop(s);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let family = TypedFamilyRegistration::new(F::new(limits()).unwrap());
    for key in ["prefix_rows", "head_dim"] {
        let mut wire = serde_json::to_value(&c).unwrap();
        wire[key] = u64::MAX.into();
        assert!(family
            .prepare_with_profile(&wire, &NumericalProfileId::new(F::PROFILE_ID).unwrap())
            .is_err());
    }
    for which in 0..3 {
        let mut cap = limits();
        match which {
            0 => cap.maximum_plan_nodes = 1,
            1 => cap.transport.maximum_map_bytes = c.valid_key_rows() * 4 - 1,
            _ => cap.maximum_metadata_bytes = 1,
        };
        assert!(FastH3PrefixAttentionSource::prepare(&layout, 2, &transformer(), &h, cap).is_err());
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    // Valid map is admitted first. A real Host refusal on the second query
    // map must release the first allocation, without keeping a partial source.
    let bytes = c.valid_key_rows() as usize * 4;
    let bounded = HostMemoryAccount::new(
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
    );
    assert!(
        FastH3PrefixAttentionSource::prepare(&layout, 2, &transformer(), &bounded, limits())
            .is_err()
    );
    assert_eq!(bounded.usage().unwrap().committed_bytes(), 0);
    let fused = c
        .with_attention_arithmetic(FastH3PrefixAttentionArithmetic::VendorFused, limits())
        .unwrap();
    let graph = program::compose(&fused, limits()).unwrap();
    assert!(graph.program.blocks()[0]
        .nodes
        .iter()
        .any(|n| n.operation_id.as_str() == CROSS_ATTENTION_FUSED_BF16_OPERATION_ID));
    assert_ne!(c.profile_id(), fused.profile_id());
    assert!(family
        .prepare_with_profile(
            &serde_json::to_value(&fused).unwrap(),
            &NumericalProfileId::new(c.profile_id()).unwrap()
        )
        .is_err());
}
