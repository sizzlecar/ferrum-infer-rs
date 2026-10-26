//! Explicit fused providers; finite-domain FP64 qualification and strict fallback.
//! Output correctness alone does not prove that a GPU admission flag stayed set.
//! The backend's production-PSO flag probe supplies that independent evidence.
use super::*;
use ferrum_execution::component::{CompiledTensorComponent, ComponentWeightSource};
use transforms::fixture::TransformFamily;
use transforms::{decode, initialization, upload_values};

struct FusedCase {
    layout: Case,
    keys: usize,
    cross: bool,
}
impl FusedCase {
    fn new(query: usize, keys: usize, cross: bool) -> Self {
        Self {
            layout: Case {
                sequence: query,
                heads: 2,
                kv_heads: 2,
                width: 128,
                segments: vec![(query, true)],
                keys: vec![vec![(keys, true)]; 2],
                scale: CanonicalRational::new(1, 8).unwrap(),
            },
            keys,
            cross,
        }
    }
    fn fixture(&self) -> Fixture {
        if !self.cross {
            return self
                .layout
                .fixture(MetalTensorOperation::StructuredAttentionFused);
        }
        let mut attrs = self.layout.attributes();
        for name in [
            "sequence_length",
            "segment_lengths",
            "segment_bidirectional",
        ] {
            attrs.remove(&id::<AttributeId>(name));
        }
        attrs.insert(
            id("batch"),
            SemanticValue::Unsigned(self.layout.keys.len() as u64),
        );
        Fixture::new(
            MetalTensorOperation::CrossAttentionFused,
            [self.layout.sequence, self.keys, self.keys]
                .into_iter()
                .map(|rows| {
                    spec(
                        vec![
                            (self.layout.keys.len() * rows) as u64,
                            (self.layout.heads * 128) as u64,
                        ],
                        ElementType::Bf16,
                    )
                })
                .collect(),
            attrs,
        )
    }
    fn visible(&self, batch: usize, query: usize, key: usize) -> bool {
        if self.cross {
            self.layout.key_valid(batch, key)
        } else {
            self.layout.visible(batch, query, key)
        }
    }
    fn values(&self) -> Vec<Vec<f32>> {
        let c = &self.layout;
        [c.sequence, self.keys, self.keys]
            .into_iter()
            .enumerate()
            .map(|(port, rows)| {
                (0..c.keys.len() * rows * c.heads * 128)
                    .map(|i| {
                        let row = i / (c.heads * 128);
                        let batch = row / rows;
                        let key = row % rows;
                        let head = i / 128 % c.heads;
                        let column = i % 128;
                        if port > 0 && !c.key_valid(batch, key) {
                            return if port == 1 { f32::NAN } else { f32::INFINITY };
                        }
                        if port < 2 {
                            // The full FP64 dot is a sum of exact quarter-integers.
                            (((i + 1 + port * 2) % 7) as f32 - 3.0) / 2.0
                        } else {
                            // Signed, nonconstant V exposes cancellation and every stride.
                            (((key + 2 * column + 3 * head + 5 * batch) % 7) as f32 - 3.0) / 2.0
                        }
                    })
                    .collect()
            })
            .collect()
    }
    fn check(&self, values: &[Vec<f32>], actual: &[f32]) {
        let c = &self.layout;
        assert_eq!(actual.len(), c.keys.len() * c.sequence * c.heads * 128);
        let scale = c.scale.numerator() as f64 / c.scale.denominator() as f64;
        for batch in 0..c.keys.len() {
            for query in 0..c.sequence {
                let visible: Vec<_> = (0..self.keys)
                    .filter(|&k| self.visible(batch, query, k))
                    .collect();
                for head in 0..c.heads {
                    let q = ((batch * c.sequence + query) * c.heads + head) * 128;
                    let kv = |key| ((batch * self.keys + key) * c.heads + head) * 128;
                    if visible.is_empty() {
                        assert!(
                            actual[q..q + 128].iter().all(|v| v.to_bits() == 0),
                            "empty row must ignore poisoned Q/K/V"
                        );
                        continue;
                    }
                    let scores: Vec<_> = visible
                        .iter()
                        .map(|&k| {
                            (0..128)
                                .map(|d| values[0][q + d] as f64 * values[1][kv(k) + d] as f64)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    if scores.iter().any(|s| !s.is_finite()) {
                        assert!(
                            actual[q..q + 128].iter().all(|v| v.is_nan()),
                            "visible nonfinite score poisons only its row/head"
                        );
                        continue;
                    }
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    // Scope of the predeclared log2/exp2 error envelope below.
                    assert!(scores.iter().all(|s| s.abs() < 17.0
                        && (s * 32.0).fract() == 0.0
                        && maximum - s < 24.5));
                    let weights: Vec<_> = scores.iter().map(|s| libm::exp(s - maximum)).collect();
                    let denominator: f64 = weights.iter().sum();
                    let (dn, dd) = error_envelope(self.keys);
                    for d in 0..128 {
                        let numerator: f64 = visible
                            .iter()
                            .zip(&weights)
                            .map(|(&k, &w)| w * values[2][kv(k) + d] as f64)
                            .sum();
                        let expected = numerator / denominator;
                        let got = actual[q + d];
                        if expected.is_nan() {
                            assert!(got.is_nan(), "visible mixed-sign infinity/NaN V");
                        } else if expected.is_infinite() {
                            assert_eq!(got as f64, expected, "visible infinity V");
                        } else {
                            let l1: f64 = visible
                                .iter()
                                .zip(&weights)
                                .map(|(&k, &w)| w * (values[2][kv(k) + d] as f64).abs())
                                .sum::<f64>()
                                / denominator;
                            // Cancellation is bounded by the weighted L1 norm,
                            // never by relative error in a possibly zero result.
                            let arithmetic =
                                ((dn * l1 + dd * expected.abs()) / (1.0 - dd)).next_up();
                            let division = (arithmetic
                                + (expected.abs() + arithmetic) * 2_f64.powi(-23))
                            .next_up();
                            let bound = (division
                                + (expected.abs() + division) * 2_f64.powi(-8)
                                + 1024.0 * f64::EPSILON * (l1 + expected.abs() + 1.0)
                                + 2_f64.powi(-134))
                            .next_up();
                            assert!(got.is_finite() && (got as f64-expected).abs() <= bound,
                                "fused b={batch} q={query} h={head} d={d}: {got} != {expected}, L1={l1}, bound={bound}");
                        }
                    }
                }
            }
        }
    }
}

fn error_envelope(keys: usize) -> (f64, f64) {
    // Same declared moderate score domain as the independent Steel diagnostic:
    // normal exp2 arguments in [-36,0], <=75 ULP plus log2-scale perturbation
    // give eP=2^-14. eA=2^-20 covers exp2(0), without assuming exact one.
    // Count every possible tile rescale even where the maximum stays fixed.
    // u=2^-23 covers RN/RTZ. Signed numerator uses an absolute L1 forward-error
    // bound for at most two rounded operations per key plus rescale/reduction.
    // This finite fixture bound does not replace the strict IEEE oracle.
    let m = keys.div_ceil(16) as f64;
    let u = 2_f64.powi(-23);
    let base = 2_f64.powi(-14) + m * 2_f64.powi(-20);
    let td = (base + (2.0 * m + 16.0) * u).next_up();
    let tn = (base + (2.0 * keys as f64 + 2.0 * m + 32.0) * u).next_up();
    assert!(tn < 1.0 && td < 1.0);
    (
        (tn / (1.0 - tn).next_down()).next_up(),
        (td / (1.0 - td).next_down()).next_up(),
    )
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
) {
    let run = |case: &FusedCase, values: &[Vec<f32>]| {
        run_case(
            runtime,
            registry,
            catalog,
            policy,
            host,
            &case.fixture(),
            values,
            |out| case.check(values, out),
        );
    };
    // Whole query tiles begin inside the last bidirectional segment. Prefix
    // rows remain causal; B2 uses the ORIGINAL physical batch pitch throughout.
    for (prefix, suffix) in [(32, 31), (32, 32), (32, 33), (17, 33)] {
        let mut case = FusedCase::new(prefix + suffix, prefix + suffix, false);
        case.layout.segments = vec![(prefix, false), (suffix, true)];
        run(&case, &case.values());
    }
    for keys in [15, 16, 17] {
        let case = FusedCase::new(33, keys, true);
        run(&case, &case.values());
    }
    // Invalid K/V are deliberately nonfinite in values(). Cover an entirely
    // empty first tile, a hole crossing 31/32, a middle empty tile and the tail.
    // Both batch pitches remain physical even when validity differs by batch.
    for cross in [false, true] {
        let mut holes = FusedCase::new(if cross { 33 } else { 65 }, 65, cross);
        holes.layout.keys = vec![
            vec![(16, false), (15, true), (2, false), (15, true), (17, false)],
            vec![(17, true), (16, false), (32, true)],
        ];
        run(&holes, &holes.values());

        let mut suffix = FusedCase::new(97, if cross { 65 } else { 97 }, cross);
        if !cross {
            suffix.layout.segments = vec![(17, false), (80, true)];
        }
        suffix.layout.keys = vec![
            vec![
                (32, false),
                (16, true),
                (16, false),
                (suffix.keys - 64, true),
            ],
            vec![(suffix.keys, false)],
        ];
        let mut values = suffix.values();
        // Entirely masked rows must return +0 without evaluating poisoned Q.
        values[0][97 * 2 * 128..].fill(f32::NAN);
        if !cross {
            values[0][..17 * 2 * 128].fill(f32::NAN);
        }
        run(&suffix, &values);

        let mut reusable = FusedCase::new(65, 65, cross);
        reusable.layout.keys = vec![
            vec![(16, true), (1, false), (48, true)],
            vec![(32, false), (33, true)],
        ];
        reuse_with_guard_changes(runtime, registry, catalog, policy, host, &reusable);
    }
    // First or last complete K tile empty, with exactly one visible key. This
    // catches exp(-Inf - -Inf), stale online state and tail-key off-by-one.
    let mut one_key = FusedCase::new(33, 17, true);
    one_key.layout.keys = vec![vec![(16, false), (1, true)], vec![(1, true), (16, false)]];
    run(&one_key, &one_key.values());
    for cross in [false, true] {
        let mut masked = FusedCase::new(33, 33, cross);
        masked.layout.keys = vec![vec![(16, true), (1, false), (16, true)], vec![(33, false)]];
        let mut values = masked.values();
        values[0][33 * 2 * 128..].fill(f32::NAN);
        run(&masked, &values);

        let case = FusedCase::new(65, if cross { 17 } else { 65 }, cross);
        reuse_with_guard_changes(runtime, registry, catalog, policy, host, &case);
    }
    let mut causal = FusedCase::new(33, 33, false);
    causal.layout.segments = vec![(33, false)];
    let mut values = causal.values();
    values[1][32 * 2 * 128] = f32::NAN;
    run(&causal, &values);
    // Equal weights and paired signed values have exact zero output. This is
    // an explicit cancellation check in addition to the L1-bound cases above.
    let cancel = FusedCase::new(33, 33, false);
    let mut values = cancel.values();
    values[0].fill(0.0);
    values[1].fill(0.0);
    values[2].fill(0.0);
    for b in 0..2 {
        for h in 0..2 {
            for d in 0..128 {
                values[2][(b * 33 * 2 + h) * 128 + d] = -1.0;
                values[2][((b * 33 + 1) * 2 + h) * 128 + d] = 1.0;
            }
        }
    }
    run_case(
        runtime,
        registry,
        catalog,
        policy,
        host,
        &cancel.fixture(),
        &values,
        |out| {
            cancel.check(&values, out);
            assert!(out.iter().all(|v| *v == 0.0));
        },
    );
    eprintln!("Metal fused structured/cross: signed FP64/L1, suffix and rectangular B2 strides, guards, reused flags, empty/masked semantics and Host retirement passed; fast-job flag evidence is separate");
}

fn reuse_with_guard_changes(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
    case: &FusedCase,
) {
    let fixture = case.fixture();
    let kind = fixture.kind();
    let prepared = TypedFamilyRegistration::new(TransformFamily)
        .prepare_with_profile(
            &serde_json::to_value(&fixture).unwrap(),
            &id(kind.profile()),
        )
        .unwrap();
    assert_eq!(
        prepared.program().blocks()[0].nodes[0].operation_id,
        kind.contract().unwrap().descriptor().id
    );
    let component = CompiledTensorComponent::compile(
        prepared,
        registry,
        catalog,
        policy,
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &ProgramPlanCompileOptions::new(fixture.specs()).unwrap(),
    )
    .unwrap();
    // Optional dense-PSO support is not a hardware gate here. When available,
    // run_case above checks the real guard scratch at exact/minus-one capacity;
    // the backend flag probe separately verifies actual fast-job admission.
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization("metal.fused.reuse"),
        )
        .unwrap();
    let mut previous_clean: Option<ManagedHostBytes> = None;
    for (iteration, change) in [
        "clean",
        "nan_query",
        "clean",
        "nan_key",
        "clean",
        "inf_value",
        "clean",
        "wide_head",
        "clean",
    ]
    .into_iter()
    .enumerate()
    {
        let mut values = case.values();
        match change {
            "nan_query" => values[0][10 * 2 * 128] = f32::NAN,
            "nan_key" => values[1][7 * 2 * 128] = f32::NAN,
            "inf_value" => {
                values[2][7 * 2 * 128 + 128] = f32::NEG_INFINITY;
                values[2][7 * 2 * 128 + 128 + 127] = f32::INFINITY;
            }
            "wide_head" => {
                // Reject only batch0/head0 while keeping exact moderate QK products.
                for q in 0..case.layout.sequence {
                    for d in 0..128 {
                        values[0][q * 2 * 128 + d] *= 2_f32.powi(21);
                    }
                }
                for k in 0..case.keys {
                    for d in 0..128 {
                        values[1][k * 2 * 128 + d] *= 2_f32.powi(-21);
                    }
                }
            }
            _ => {}
        }
        let input = fixture
            .inputs
            .iter()
            .zip(&values)
            .enumerate()
            .map(|(i, (spec, values))| upload_values(host, input_id(i), spec, values))
            .collect();
        let output = session
            .execute(
                id(format!("run.metal.fused.{iteration}")),
                id(format!("request.metal.fused.{iteration}")),
                input,
            )
            .unwrap();
        assert_eq!(output.len(), 1);
        case.check(&values, &decode(output[0].bytes().bytes()));
        if change == "clean" {
            if let Some(previous) = &previous_clean {
                assert_eq!(output[0].bytes().bytes(), previous.bytes());
            }
            previous_clean = Some(output[0].bytes().clone());
        }
        drop(output);
    }
    assert!(matches!(
        session.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("fused close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    case.check(
        &case.values(),
        &decode(previous_clean.as_ref().unwrap().bytes()),
    );
    drop(previous_clean);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
