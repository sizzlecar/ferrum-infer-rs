//! Small managed FP8 projection and an independent finite-domain scalar oracle.
//! CPU checks define the fixture; ignored CUDA checks execute the pinned native
//! artifact. Neither substitutes for full-model quality or throughput evidence.
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(any(
    feature = "cuda-cutlass-fp8",
    all(feature = "metal", target_os = "macos")
))]
use std::sync::Arc;
use std::sync::OnceLock;

#[path = "vnext_tensor_fp8/family.rs"]
mod family;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "vnext_tensor_fp8/metal.rs"]
mod metal;
#[path = "vnext_tensor_fp8/oracle.rs"]
mod oracle;
#[cfg(feature = "cuda-cutlass-fp8")]
#[path = "vnext_tensor_fp8/runtime.rs"]
mod runtime;

const X: &str = "value.fp8.input";
const W: &str = "value.fp8.weight";
const Y: &str = "value.fp8.output";
const OP: &str = "operation.dense_linear_fp8.bf16";
const PROFILE: &str = "cuda.cutlass.w8a8.bf16.rowwise_e4m3fn.v1";
const METAL_PROFILE: &str = "metal.simdgroup.f32.w8a8.bf16.rowwise_e4m3fn.v1";
fn id<T: TryFrom<String>>(name: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(name.into()).unwrap()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    m: u64,
    n: u64,
    k: u64,
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

struct Payload {
    weights: Vec<u8>,
    scale: [u8; 4],
}
impl Payload {
    fn new(shape: &Shape) -> Self {
        // Distinct rows and asymmetric columns catch transposition/broadcast errors.
        let weights = (0..shape.n)
            .flat_map(|n| {
                (0..shape.k).map(move |k| {
                    let magnitude = 24 + ((n * 13 + k * 7) % 80) as u8;
                    magnitude | if (n + 2 * k) % 3 == 0 { 128 } else { 0 }
                })
            })
            .collect();
        Self {
            weights,
            scale: 0.03125_f32.to_le_bytes(),
        }
    }
    fn scale(&self) -> f32 {
        f32::from_le_bytes(self.scale)
    }
}
impl WeightComponentSource for Payload {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let bytes: &[u8] = match component.role {
            WeightComponentRole::PackedValues => &self.weights,
            WeightComponentRole::Scales => &self.scale,
            _ => return Err(invalid("unexpected FP8 fixture component")),
        };
        WeightComponentPayload::new(
            component,
            component.external_names[0].clone(),
            "generated-tiny-fp8-fixture",
            component.dimensions.clone(),
            component.physical_element_type(),
            bytes,
        )
    }
}
fn input(shape: &Shape) -> Vec<bf16> {
    (0..shape.m)
        .flat_map(|m| {
            (0..shape.k).map(move |k| {
                let value = if m % 7 == 0 && shape.m > 1 {
                    if k % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    let amplitude = if shape.m == 1 {
                        0.125
                    } else {
                        [0.0001_f32, 0.125, 1.0, 16.0][m as usize % 4]
                    };
                    ((k * 11 + m * 3) % 31) as f32 / 8.0 * amplitude - amplitude
                };
                bf16::from_f32(value)
            })
        })
        .collect()
}

#[test]
fn single_row_fixture_is_nonzero_and_has_nontrivial_quantized_projection() {
    let shape = Shape { m: 1, n: 32, k: 16 };
    let x = input(&shape);
    assert!(x.iter().any(|value| value.to_f32() != 0.0));
    let payload = Payload::new(&shape);
    let expected = oracle::expected(&x, &payload.weights, payload.scale(), 16, 32);
    assert!(expected.activation_bytes.iter().any(|bits| bits & 127 != 0));
    assert!(expected.quantized_dot.iter().any(|value| *value != 0.0));
}

#[test]
fn tiny_weight_source_retains_whole_tensor_encoding_and_independent_scale() {
    let shape = Shape { m: 3, n: 32, k: 16 };
    let prepared = family::prepare(&shape);
    let schema = prepared.stored_weight_schema().unwrap();
    let payload = Payload::new(&shape);
    for component in &schema.components {
        payload.component(component).unwrap();
    }
    assert_eq!(
        schema.physical_bytes(&id("weight.fp8.projection")).unwrap(),
        shape.n * shape.k + 4
    );
    assert_eq!(schema.tensors[0].logical_element_type, ElementType::Bf16);
    assert!(matches!(
        schema.tensors[0].physical_layout,
        PhysicalWeightLayout::QuantizedTensor { .. }
    ));
    let expected = oracle::expected(
        &input(&shape),
        &payload.weights,
        payload.scale(),
        shape.k as usize,
        shape.n as usize,
    );
    assert_eq!(expected.quantized_dot.len(), (shape.m * shape.n) as usize);
    assert!(expected.quantized_dot.iter().all(|x| x.is_finite()));
    assert!(expected.quantized_dot[..shape.n as usize]
        .iter()
        .all(|x| *x == 0.0));
}

#[cfg(feature = "cuda-cutlass-fp8")]
#[test]
#[ignore = "requires exclusive SM89 or SM120 CUDA and its locked CUTLASS FP8 native artifact"]
fn managed_fp8_projection_independent_oracle_shapes_and_retirement() {
    runtime::verify();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn managed_metal_fp8_projection_matches_independent_oracle_and_retires() {
    metal::verify();
}
