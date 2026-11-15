use super::fixture::{id, Config, INPUT, OUTPUT};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::sync::Arc;

/// Domain covers unequal rows/halves, a partial CUDA block, the known staged
/// rounding counterexample, signed zero, overflowed exp2, and NaN propagation.
/// Independent FP64-exp/exp2 oracles retain each specified F32 and BF16 boundary. CUDA
/// ex2.approx/div.full is allowed <=1 BF16 ULP on finite nonzero results;
/// zero sign and special-value classes must match exactly.
pub fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    c: Config,
) {
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.packed_swiglu.test"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let component = c.compile(runtime.as_ref(), registry, &catalog, &materializers);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.packed_swiglu.init"),
                request_id: id("request.packed_swiglu.init"),
                transaction_id: id("transaction.packed_swiglu.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let (bytes, expected) = data(c);
    let mut retained = None;
    let mut worst = 0;
    for trial in 0..2 {
        let outputs = session
            .execute(
                id(format!("run.packed_swiglu.{trial}")),
                id(format!("request.packed_swiglu.{trial}")),
                vec![c.managed(&host, INPUT, &bytes)],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(OUTPUT));
        let actual = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        let witness = match c.profile {
            PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound => 0x3f3d,
            PackedSwiGluBf16Profile::SiluBf16ThenMultiply => 0x3f3c,
        };
        assert_eq!(
            actual[0], witness,
            "must retain the declared rounding boundary"
        );
        for (index, (actual, expected)) in actual.into_iter().zip(&expected).enumerate() {
            let a = bf16::from_bits(actual);
            let e = bf16::from_bits(*expected);
            if e.is_nan() {
                assert!(a.is_nan(), "NaN at{index}");
                continue;
            }
            if !e.is_finite() || e.to_f32() == 0. {
                assert_eq!(actual, *expected, "special value at{index}");
                continue;
            }
            assert!(a.is_finite(), "nonfinite at{index}");
            let distance = ordered(actual).abs_diff(ordered(*expected));
            worst = worst.max(distance);
            assert!(
                distance <= 1,
                "BF16 ULP {distance} at {index}: actual{actual:04x}, expected{expected:04x}"
            );
        }
        retained = Some(outputs);
    }
    let closed = session
        .close()
        .unwrap_or_else(|e| panic!("SwiGLU close: {:?}", e.failure()));
    assert!(matches!(closed, PlanRuntimeCloseOutcome::Closed(_)));
    let retained = retained.unwrap();
    assert!(host.owns(retained[0].bytes()));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    eprintln!(
        "packed SwiGLU {:?}: max_bf16_ulp={worst}, repeated=2, owner_closed=true",
        c
    );
}
fn ordered(v: u16) -> i32 {
    if v & 0x8000 != 0 {
        -(i32::from(v & 0x7fff))
    } else {
        i32::from(v)
    }
}
fn data(c: Config) -> (Vec<u8>, Vec<u16>) {
    let width = c.width as usize;
    let rows = c.rows as usize;
    let mut bytes = vec![0; rows * width * 4];
    let mut expected = Vec::new();
    for r in 0..rows {
        for k in 0..width {
            let (value, gate) = match k {
                0 => (1.0078125, 1.),
                1 => (1., -0.),
                2 => (1., 100.),
                3 => (1., -100.),
                4 => (0., f32::INFINITY),
                5 => (1., f32::NAN),
                6 => (-1., f32::NEG_INFINITY),
                7 => (-1., 0.),
                8 => (1., bf16::from_bits(1).to_f32()),
                9 => (1., bf16::from_bits(0x8001).to_f32()),
                _ => (
                    1.0078125 + ((r * 3 + k) % 11) as f32 / 8.,
                    ((r * 7 + k) % 51) as f32 / 2. - 12.5,
                ),
            };
            let v = bf16::from_f32(value);
            let g = bf16::from_f32(gate);
            let base = (r * width * 2 + k) * 2;
            bytes[base..base + 2].copy_from_slice(&v.to_bits().to_le_bytes());
            bytes[base + width * 2..base + width * 2 + 2]
                .copy_from_slice(&g.to_bits().to_le_bytes());
            // Independent FP64 transcendental implementations, retaining the
            // declared expression order and each F32/BF16 rounding boundary.
            let silu = match c.profile {
                PackedSwiGluBf16Profile::Exp2SigmoidF32SingleRound => {
                    let exponent = (0f32 - g.to_f32()) * f32::from_bits(0x3fb8aa3b);
                    let e = 2f64.powf(exponent as f64) as f32;
                    let sigmoid = 1f32 / (1f32 + e);
                    g.to_f32() * sigmoid
                }
                PackedSwiGluBf16Profile::SiluBf16ThenMultiply => {
                    let e = (-(g.to_f32() as f64)).exp() as f32;
                    bf16::from_f32(g.to_f32() / (1f32 + e)).to_f32()
                }
            };
            expected.push(bf16::from_f32(v.to_f32() * silu).to_bits());
        }
    }
    (bytes, expected)
}
