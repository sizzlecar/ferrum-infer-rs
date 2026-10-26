//! Independent visibility and FP64 softmax oracle, exercised on actual Metal.
use super::*;
use ferrum_execution::component::TensorComponentPolicy;
use transforms::{fixture::Fixture, run_case, spec};

#[path = "attention/accumulation.rs"]
mod accumulation;
#[path = "attention/cross.rs"]
mod cross;
#[path = "attention/exp_range.rs"]
mod exp_range;
#[path = "attention/fused.rs"]
mod fused;
#[path = "attention/oracle.rs"]
mod oracle;
#[path = "attention/underflow.rs"]
mod underflow;

pub(super) struct Case {
    pub sequence: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub segments: Vec<(usize, bool)>,
    pub keys: Vec<Vec<(usize, bool)>>,
    pub scale: CanonicalRational,
}
impl Case {
    pub fn attributes(&self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut attrs = BTreeMap::from([
            (
                id("sequence_length"),
                SemanticValue::Unsigned(self.sequence as u64),
            ),
            (id("head_dim"), SemanticValue::Unsigned(self.width as u64)),
            (id("scale"), SemanticValue::Rational(self.scale)),
        ]);
        for (name, values) in [
            (
                "segment_lengths",
                self.segments.iter().map(|v| v.0 as i64).collect(),
            ),
            (
                "segment_bidirectional",
                self.segments.iter().map(|v| i64::from(v.1)).collect(),
            ),
            (
                "key_run_counts",
                self.keys.iter().map(|v| v.len() as i64).collect(),
            ),
            (
                "key_run_lengths",
                self.keys.iter().flatten().map(|v| v.0 as i64).collect(),
            ),
            (
                "key_run_valid",
                self.keys.iter().flatten().map(|v| i64::from(v.1)).collect(),
            ),
        ] {
            attrs.insert(id(name), SemanticValue::Integers(values));
        }
        attrs
    }
    fn fixture(&self, kind: MetalTensorOperation) -> Fixture {
        Fixture::new(
            kind,
            [self.heads, self.kv_heads, self.kv_heads]
                .into_iter()
                .map(|h| {
                    spec(
                        vec![
                            (self.keys.len() * self.sequence) as u64,
                            (h * self.width) as u64,
                        ],
                        ElementType::Bf16,
                    )
                })
                .collect(),
            self.attributes(),
        )
    }
    pub fn check(&self, values: &[Vec<f32>], actual: &[f32]) {
        oracle::check(self, values, actual);
    }
    fn key_valid(&self, batch: usize, key: usize) -> bool {
        let mut start = 0;
        for &(len, valid) in &self.keys[batch] {
            if (start..start + len).contains(&key) {
                return valid;
            }
            start += len;
        }
        panic!("key outside fixture");
    }
    fn visible(&self, batch: usize, query: usize, key: usize) -> bool {
        let mut start = 0;
        for &(len, bidirectional) in &self.segments {
            if (start..start + len).contains(&query) {
                return self.key_valid(batch, key)
                    && (key <= query || (bidirectional && (start..start + len).contains(&key)));
            }
            start += len;
        }
        panic!("query outside fixture");
    }
    fn values(&self) -> Vec<Vec<f32>> {
        [self.heads, self.kv_heads, self.kv_heads]
            .into_iter()
            .enumerate()
            .map(|(port, heads)| {
                (0..self.keys.len() * self.sequence * heads * self.width)
                    .map(|i| {
                        let token = i / (heads * self.width);
                        if port != 0
                            && !self.key_valid(token / self.sequence, token % self.sequence)
                        {
                            // Invalid keys must be ignored before any arithmetic use.
                            return if port == 1 { f32::NAN } else { f32::INFINITY };
                        }
                        rounds(((i * 13 + port * 27 + token * 7) % 37) as f32 / 16.0 - 1.0)
                    })
                    .collect()
            })
            .collect()
    }
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    use MetalTensorOperation::*;
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    for kind in [StructuredAttention, StructuredGqaAttention] {
        // Exercise every packed mask bit and batch slot, then reuse a group
        // beyond the native grid cap; both are real indexing boundaries.
        for case in [
            Case {
                sequence: 32,
                width: 2,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 1 },
                segments: vec![(1, true); 32],
                keys: (0..16)
                    .map(|batch| (0..32).map(|key| (1, (batch + key) % 2 == 0)).collect())
                    .collect(),
                scale: CanonicalRational::new(1, 2).unwrap(),
            },
            Case {
                sequence: 1,
                width: 2,
                heads: 4097,
                kv_heads: if kind == StructuredAttention { 4097 } else { 1 },
                segments: vec![(1, true)],
                keys: vec![vec![(1, true)]; 16],
                scale: CanonicalRational::new(1, 2).unwrap(),
            },
            Case {
                sequence: 33,
                width: 8,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 1 },
                segments: vec![(33, true)],
                keys: vec![vec![(33, false)]],
                scale: CanonicalRational::new(1, 8).unwrap(),
            },
            Case {
                sequence: 97,
                width: 128,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 1 },
                segments: vec![(97, true)],
                keys: vec![vec![(97, true)], vec![(97, false)]],
                scale: CanonicalRational::new(1, 8).unwrap(),
            },
            Case {
                sequence: 97,
                width: 128,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 2 },
                segments: vec![(97, true)],
                // Complete key tiles on either side of a masked nonfinite key,
                // plus a partial final tile, must share the same online state.
                keys: vec![vec![(32, true), (1, false), (64, true)]],
                scale: CanonicalRational::new(1, 8).unwrap(),
            },
        ] {
            let mut values = case.values();
            if case.keys.iter().all(|runs| runs.iter().all(|v| !v.1)) {
                values[0].fill(f32::NAN);
            }
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &case.fixture(kind),
                &values,
                |actual| case.check(&values, actual),
            );
            eprintln!(
                "Metal {kind:?}: mask/grid/empty boundary B={} S={} Hq={} Hkv={}, passed",
                case.keys.len(),
                case.sequence,
                case.heads,
                case.kv_heads
            );
        }
        for (width, sequence) in [
            (2, 1),
            (4, 15),
            (8, 16),
            (16, 17),
            (32, 31),
            (64, 32),
            (128, 33),
            (256, 65),
        ] {
            let case = Case {
                sequence,
                width,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 2 },
                segments: if sequence < 3 {
                    vec![(sequence, true)]
                } else {
                    vec![
                        (1, false),
                        (sequence / 2, true),
                        (sequence - sequence / 2 - 1, false),
                    ]
                },
                keys: if sequence < 3 {
                    vec![vec![(sequence, true)], vec![(sequence, false)]]
                } else {
                    vec![
                        vec![(1, false), (sequence - 2, true), (1, false)],
                        vec![(sequence / 2, true), (sequence - sequence / 2, false)],
                    ]
                },
                scale: CanonicalRational::new(1, 8).unwrap(),
            };
            let mut values = case.values();
            // Empty first query, including all-invalid second batch for S=1.
            if sequence > 1 {
                values[0][0] = f32::NAN;
            }
            if sequence == 1 {
                values[0][4 * width] = f32::INFINITY;
            }
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &case.fixture(kind),
                &values,
                |actual| case.check(&values, actual),
            );
            eprintln!("Metal {kind:?}: B=2 S={sequence} D={width}, mixed masks, padded query/key, FP64 bound, repeat and retirement passed");
        }
        // Future masked nonfinite keys and values must not contaminate earlier
        // causal rows. Visible score poison is whole-head; V poison per channel.
        for (width, sequence, bidirectional, poison) in
            [(8, 33, false), (128, 33, false), (128, 97, true)]
                .into_iter()
                .flat_map(|(width, sequence, bidirectional)| {
                    ["key", "value", "query", "score_overflow"]
                        .into_iter()
                        .map(move |poison| (width, sequence, bidirectional, poison))
                })
        {
            let case = Case {
                sequence,
                width,
                heads: 4,
                kv_heads: if kind == StructuredAttention { 4 } else { 1 },
                segments: vec![(sequence, bidirectional)],
                keys: vec![vec![(sequence, true)]],
                scale: CanonicalRational::new(1, 8).unwrap(),
            };
            let mut values = case.values();
            match poison {
                "key" => values[1][32 * case.kv_heads * case.width] = f32::NAN,
                "value" => {
                    let offset = 7 * case.kv_heads * case.width;
                    values[2][offset] = f32::INFINITY;
                    values[2][offset + 1] = f32::NEG_INFINITY;
                    values[2][offset + 2] = f32::NAN;
                }
                "query" => values[0][16 * case.heads * case.width] = f32::INFINITY,
                _ => {
                    values[0].fill(2_f32.powi(80));
                    values[1].fill(2_f32.powi(80));
                }
            }
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &case.fixture(kind),
                &values,
                |actual| case.check(&values, actual),
            );
            eprintln!(
                "Metal {kind:?}: S={sequence} D={width} bidirectional={bidirectional} {poison}, mask and nonfinite classification passed"
            );
        }
        // Both operands are valid BF16 values outside FP16's range. Their
        // product is moderate: an implicit FP16 conversion would be observable.
        let case = Case {
            sequence: 65,
            width: 128,
            heads: 4,
            kv_heads: if kind == StructuredAttention { 4 } else { 2 },
            segments: vec![(65, true)],
            keys: vec![vec![(65, true)]],
            scale: CanonicalRational::new(1, 8).unwrap(),
        };
        let mut values = case.values();
        for value in &mut values[0] {
            *value *= 2_f32.powi(80);
        }
        for value in &mut values[1] {
            *value *= 2_f32.powi(-80);
        }
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &case.fixture(kind),
            &values,
            |actual| case.check(&values, actual),
        );
        // Keep Q/K moderate so the original softmax oracle applies. V alone
        // spans outside FP16's finite and nonzero ranges, exercising the FP32
        // PV operands independently of the QK range case above. S=65 also
        // crosses complete key tiles and a partial final query/key tile.
        for exponent in [80, -80] {
            let mut values = case.values();
            for value in &mut values[2] {
                *value *= 2_f32.powi(exponent);
            }
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &case.fixture(kind),
                &values,
                |actual| case.check(&values, actual),
            );
            eprintln!("Metal {kind:?}: finite V scaled by 2^{exponent}, FP32 PV range, FP64 bound and retirement passed");
        }
    }
    underflow::verify(runtime, registry, catalog, &policy, host);
    accumulation::verify(runtime, registry, catalog, &policy, host);
    exp_range::verify(runtime, registry, catalog, &policy, host);
    cross::verify(runtime, registry, catalog, host);
    fused::verify(runtime, registry, catalog, &policy, host);
}
