//! An independent finite-fixture oracle for two ordinary F32 boundaries.
//! The eight-F32-ULP SiLU limit below is declared for these selected operands,
//! separately from exact addition; it does not claim a uniform transcendental
//! bound, libm/libdevice equality, or Torch binary equivalence.
use crate::fixture::{id, Config, Kind, INPUT, OUTPUT, WEIGHT};
use ferrum_execution::{component::TensorComponentSession, TensorExecutionOutput};
use ferrum_runtime::*;

pub fn config(kind: Kind) -> Config {
    assert!(matches!(kind, Kind::F32ResidualAdd | Kind::F32SiluMul));
    Config {
        rows: 3,
        inputs: 8,
        outputs: 8,
        kind,
    }
}
fn operands(kind: Kind) -> ([f32; 24], [f32; 24]) {
    match kind {
        Kind::F32ResidualAdd => (
            [
                1.000244140625,
                1.,
                1.,
                -0.,
                f32::from_bits(1),
                f32::MAX,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                1e-30,
                -1e-30,
                16777216.,
                -16777216.,
                0.5,
                -0.5,
                0.,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                3.,
                -3.,
                1e30,
                -1e30,
                0.0001220703125,
                -0.0001220703125,
            ],
            [
                -1.,
                2f32.powi(-24),
                3. * 2f32.powi(-24),
                -0.,
                f32::from_bits(1),
                -f32::MAX,
                f32::NEG_INFINITY,
                -1.,
                1.,
                1e-30,
                1e-30,
                1.,
                -1.,
                2f32.powi(-25),
                -2f32.powi(-25),
                -0.,
                -f32::from_bits(1),
                f32::from_bits(1),
                -2.999755859375,
                2.999755859375,
                1e30,
                -1e30,
                1.,
                -1.,
            ],
        ),
        Kind::F32SiluMul => (
            [
                0.,
                -0.,
                1.000244140625,
                -1.,
                1.,
                -2.,
                2.,
                -20.,
                20.,
                -80.,
                80.,
                -100.,
                100.,
                1e-30,
                -1e-30,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                0.,
                1.,
                -5.000001,
                0.000244140625,
            ],
            [
                1.,
                1.,
                1.00048828125,
                0.75,
                -0.5,
                2.,
                -3.,
                1.,
                0.125,
                1.,
                -0.25,
                1.,
                0.25,
                0.5,
                -0.5,
                1.,
                1.,
                1.,
                1.,
                1.,
                f32::INFINITY,
                f32::NAN,
                0.75,
                1.000244140625,
            ],
        ),
        _ => unreachable!(),
    }
}
fn ulp(x: f32) -> f64 {
    let bits = x.abs().to_bits();
    if bits == 0 {
        return f64::from(f32::from_bits(1));
    }
    f64::from(f32::from_bits(bits + 1)) - f64::from(f32::from_bits(bits))
}
pub fn execute<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    kind: Kind,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    let config = config(kind);
    let (x, update) = operands(kind);
    let x_bytes: Vec<_> = x.into_iter().flat_map(f32::to_le_bytes).collect();
    let update_bytes: Vec<_> = update.into_iter().flat_map(f32::to_le_bytes).collect();
    let outputs = session
        .execute(
            id(format!("run.pointwise_f32.{trial}")),
            id(format!("request.pointwise_f32.{trial}")),
            vec![
                config.managed(host, INPUT, &x_bytes),
                config.managed(host, WEIGHT, &update_bytes),
            ],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert_eq!(output.value_id().as_str(), OUTPUT);
    assert_eq!(output.tensor().dimensions(), &[3, 8]);
    assert_eq!(output.tensor().element_type(), ElementType::F32);
    assert!(host.owns(output.bytes()));
    let actual: Vec<_> = output
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(actual.len(), x.len());
    for (i, ((&gate, &up), &actual)) in x.iter().zip(&update).zip(&actual).enumerate() {
        let expected = match kind {
            Kind::F32ResidualAdd => (f64::from(gate) + f64::from(up)) as f32,
            Kind::F32SiluMul => {
                // F64 exponential is rounded to the explicit F32 exp boundary
                // before ordinary F32 add/divide/multiply. Thus -100 correctly
                // overflows expf to infinity and yields negative zero.
                let exponent = f64::from(-gate).exp() as f32;
                (gate / (1.0_f32 + exponent)) * up
            }
            _ => unreachable!(),
        };
        if expected.is_nan() {
            assert!(actual.is_nan(), "NaN class at {i}");
        } else if matches!(kind, Kind::F32ResidualAdd) || expected == 0. || expected.is_infinite() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "exact boundary at {i}"
            );
        } else {
            let bound = 8. * ulp(expected);
            assert!(
                actual.is_finite() && (f64::from(actual) - f64::from(expected)).abs() <= bound,
                "SiLU finite fixture {i}: {actual} vs {expected}, bound {bound}"
            );
        }
    }
    if matches!(kind, Kind::F32ResidualAdd) {
        assert_eq!(actual[0].to_bits(), 0.000244140625_f32.to_bits());
    } else {
        assert_ne!(
            actual[2],
            half::bf16::from_f32(actual[2]).to_f32(),
            "ordinary SiLU*up retains F32 output"
        );
    }
    outputs
}
