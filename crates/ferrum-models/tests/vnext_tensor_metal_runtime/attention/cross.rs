//! Rectangular queries/KV, with an independent FP64 oracle and real Metal IO.
use super::*;

#[path = "cross_oracle.rs"]
mod oracle;
#[path = "cross_register.rs"]
mod register;

struct CrossCase {
    query_length: usize,
    key_length: usize,
    heads: usize,
    width: usize,
    keys: Vec<Vec<(usize, bool)>>,
    scale: CanonicalRational,
}
impl CrossCase {
    fn fixture(&self) -> Fixture {
        let mut attributes = BTreeMap::from([
            (id("batch"), SemanticValue::Unsigned(self.keys.len() as u64)),
            (id("head_dim"), SemanticValue::Unsigned(self.width as u64)),
            (id("scale"), SemanticValue::Rational(self.scale)),
        ]);
        for (name, values) in [
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
            attributes.insert(id(name), SemanticValue::Integers(values));
        }
        Fixture::new(
            MetalTensorOperation::CrossAttention,
            [self.query_length, self.key_length, self.key_length]
                .into_iter()
                .map(|rows| {
                    spec(
                        vec![
                            (self.keys.len() * rows) as u64,
                            (self.heads * self.width) as u64,
                        ],
                        ElementType::Bf16,
                    )
                })
                .collect(),
            attributes,
        )
    }
    fn key_valid(&self, batch: usize, key: usize) -> bool {
        let mut start = 0;
        for &(length, valid) in &self.keys[batch] {
            if (start..start + length).contains(&key) {
                return valid;
            }
            start += length;
        }
        panic!("key outside fixture");
    }
    fn values(&self) -> Vec<Vec<f32>> {
        [self.query_length, self.key_length, self.key_length]
            .into_iter()
            .enumerate()
            .map(|(port, rows)| {
                (0..self.keys.len() * rows * self.heads * self.width)
                    .map(|i| {
                        let row = i / (self.heads * self.width);
                        if port != 0 && !self.key_valid(row / rows, row % rows) {
                            return if port == 1 { f32::NAN } else { f32::INFINITY };
                        }
                        rounds(((i * 13 + port * 27 + row * 7) % 37) as f32 / 16.0 - 1.0)
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
    let run = |case: &CrossCase, values: &[Vec<f32>]| {
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &case.fixture(),
            values,
            |actual| oracle::check(case, values, actual),
        );
    };
    // Both rectangular orientations, query/key tile tails, the full head-width
    // domain, per-batch mask offsets and invalid keys containing NaN/Inf.
    for (width, q, kv) in [
        (2, 1, 3),
        (4, 15, 33),
        (8, 17, 16),
        (16, 31, 17),
        (32, 33, 31),
        (64, 17, 65),
        (72, 33, 17),
        (128, 33, 65),
        (128, 65, 33),
        (256, 17, 33),
    ] {
        let case = CrossCase {
            query_length: q,
            key_length: kv,
            heads: 2,
            width,
            keys: vec![
                vec![(1, false), (kv - 2, true), (1, false)],
                vec![(kv, false)],
            ],
            scale: CanonicalRational::new(1, 8).unwrap(),
        };
        let mut values = case.values();
        values[0][q * case.heads * width..].fill(f32::NAN);
        run(&case, &values);
    }
    // Every packed validity bit and all sixteen batch slots, with Q != KV.
    let bits = CrossCase {
        query_length: 17,
        key_length: 32,
        heads: 2,
        width: 2,
        keys: (0..16)
            .map(|b| (0..32).map(|k| (1, (b + k) % 2 == 0)).collect())
            .collect(),
        scale: CanonicalRational::new(1, 2).unwrap(),
    };
    run(&bits, &bits.values());
    // The existing structured suite exercises the shared >65535-job loop;
    // these rectangular cases keep the original Host fixture budget unchanged.
    let finite = CrossCase {
        query_length: 33,
        key_length: 65,
        heads: 2,
        width: 128,
        keys: vec![vec![(65, true)]; 2],
        scale: CanonicalRational::new(1, 8).unwrap(),
    };
    run(&finite, &finite.values());
    for poison in ["key", "query", "value", "overflow", "wide_exponent"] {
        let mut values = finite.values();
        match poison {
            "key" => values[1][32 * finite.heads * finite.width] = f32::NAN,
            "query" => values[0][16 * finite.heads * finite.width] = f32::INFINITY,
            "value" => {
                let base = 7 * finite.heads * finite.width;
                values[2][base] = f32::INFINITY;
                values[2][base + 1] = f32::NEG_INFINITY;
                values[2][base + 2] = f32::NAN;
            }
            "overflow" => {
                values[0].fill(2_f32.powi(80));
                values[1].fill(2_f32.powi(80));
            }
            _ => {
                for x in &mut values[0] {
                    *x *= 2_f32.powi(80);
                }
                for x in &mut values[1] {
                    *x *= 2_f32.powi(-80);
                }
            }
        }
        run(&finite, &values);
    }
    register::verify(runtime, registry, catalog, &policy, host);
    eprintln!("Metal cross attention: rectangular FP64/IEEE, mask, empty, repeat and retirement checks passed");
}
