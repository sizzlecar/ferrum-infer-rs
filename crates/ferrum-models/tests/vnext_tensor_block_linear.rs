//! Native GGUF matrices through a managed fixed-tensor component. Candle's
//! decoder supplies an independent weight oracle; no full model is required.
use candle_core::{
    quantized::{gguf_file, GgmlDType, QTensor},
    Device, Tensor,
};
use ferrum_quantization::{gguf::gguf_weight_encoding, GgufWeightComponentSource, TensorArchive};
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

#[path = "vnext_tensor_block_linear/family.rs"]
mod family;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "vnext_tensor_block_linear/metal_runtime.rs"]
mod metal_runtime;
#[cfg(feature = "cuda")]
#[path = "vnext_tensor_block_linear/runtime.rs"]
mod runtime;

const X: &str = "value.block.input";
const W: &str = "value.block.weight";
const Y: &str = "value.block.output";
const PROFILE: &str = "cuda.native_block.decode_f32_rn_bf16.cublas_f32.v1";
fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Debug, Clone, Copy)]
enum Format {
    Q8,
    Q4K,
    Q5K,
    Q6K,
}
const FORMATS: [Format; 4] = [Format::Q8, Format::Q4K, Format::Q5K, Format::Q6K];
impl Format {
    fn dtype(self) -> GgmlDType {
        match self {
            Self::Q8 => GgmlDType::Q8_0,
            Self::Q4K => GgmlDType::Q4K,
            Self::Q5K => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K,
        }
    }
    fn blocks(self) -> BlockQuantizationSpec {
        let code = match self {
            Self::Q8 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        };
        match gguf_weight_encoding(code).unwrap() {
            WeightEncoding::BlockQuantized(block) => block,
            _ => unreachable!(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    m: u64,
    n: u64,
    k: u64,
    blocks: BlockQuantizationSpec,
}
struct Fixture {
    _file: tempfile::NamedTempFile,
    source: GgufWeightComponentSource,
    decoded: Vec<bf16>,
}
impl Fixture {
    fn new(shape: &Shape, format: Format) -> Self {
        // Different rows and superblocks expose transposition, lane/subgroup
        // indexing and repeated-block mistakes, including signed scale paths.
        let values: Vec<f32> = (0..shape.n)
            .flat_map(|n| {
                (0..shape.k).map(move |k| {
                    let amplitude = [0.001, 0.25, 1.0, 8.0][((k / 16 + n) % 4) as usize];
                    (((k * 17 + n * 43 + k / 256 * 19) % 97) as f32 - 48.0) * amplitude
                })
            })
            .collect();
        let dense =
            Tensor::from_vec(values, (shape.n as usize, shape.k as usize), &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&dense, format.dtype()).unwrap();
        let decoded = quantized
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let mut file = tempfile::NamedTempFile::with_suffix(".gguf").unwrap();
        let architecture = gguf_file::Value::String("linear-fixture".into());
        gguf_file::write(
            file.as_file_mut(),
            &[("general.architecture", &architecture)],
            &[("projection.weight", &quantized)],
        )
        .unwrap();
        let source = GgufWeightComponentSource::open(file.path()).unwrap();
        Self {
            _file: file,
            source,
            decoded,
        }
    }
}
fn input(shape: &Shape, basis: bool) -> Vec<bf16> {
    (0..shape.m)
        .flat_map(|m| {
            (0..shape.k).map(move |k| {
                bf16::from_f32(if basis {
                    if m == k {
                        1.0
                    } else {
                        0.0
                    }
                } else if m % 7 == 6 {
                    0.0
                } else {
                    (((m * 7 + k * 11) % 31) as f32 - 15.0) / 16.0
                })
            })
        })
        .collect()
}
fn check(shape: &Shape, inputs: &[bf16], weights: &[bf16], bytes: &[u8], basis: bool) {
    assert_eq!(bytes.len() as u64, shape.m * shape.n * 2);
    let u = 2_f64.powi(-24);
    let operations = (2 * shape.k + 8) as f64;
    let gamma = operations * u / (1.0 - operations * u);
    for (index, bytes) in bytes.chunks_exact(2).enumerate() {
        let (m, n) = (index / shape.n as usize, index % shape.n as usize);
        let actual = bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32();
        if basis {
            // Unit rows isolate every decoded weight, with no accumulation
            // tolerance that could hide a decoding or permutation error.
            assert_eq!(
                actual,
                weights[n * shape.k as usize + m].to_f32(),
                "basis m={m} n={n} format={}",
                shape.blocks.format_id
            );
        } else {
            let mut sum = 0_f64;
            let mut absolute = 0_f64;
            for k in 0..shape.k as usize {
                let product = f64::from(inputs[m * shape.k as usize + k].to_f32())
                    * f64::from(weights[n * shape.k as usize + k].to_f32());
                sum += product;
                absolute += product.abs();
            }
            let accumulation = gamma * absolute;
            let bound = accumulation + (sum.abs() + accumulation) / 256.0 + 2_f64.powi(-130);
            assert!(
                actual.is_finite() && (f64::from(actual) - sum).abs() <= bound,
                "m={m} n={n}: observed={actual} decoded-F64={sum} bound={bound}"
            );
        }
    }
}

#[test]
fn native_fixture_keeps_packed_bytes_and_bf16_logical_shape() {
    for format in FORMATS {
        let shape = Shape {
            m: 3,
            n: 7,
            k: 512,
            blocks: format.blocks(),
        };
        let fixture = Fixture::new(&shape, format);
        let schema = family::prepare(&shape)
            .stored_weight_schema()
            .unwrap()
            .clone();
        schema.validate(&id("family.tensor.block.fixture")).unwrap();
        let view = fixture.source.tensor("projection.weight").unwrap();
        assert_eq!(view.shape(), [shape.n, shape.k]);
        assert_eq!(view.encoding(), &schema.components[0].encoding);
        let physical = shape.n * shape.k / u64::from(shape.blocks.logical_values_per_block)
            * u64::from(shape.blocks.bytes_per_block);
        assert_eq!(
            schema
                .physical_bytes(&id("weight.block.projection"))
                .unwrap(),
            physical
        );
        let component = fixture.source.component(&schema.components[0]).unwrap();
        assert_eq!(component.bytes().as_ptr(), view.bytes().as_ptr());
        assert_eq!(component.bytes().len() as u64, physical);
        assert!(fixture.decoded.iter().all(|value| value.is_finite()));
        let mut incorrect = schema.components[0].clone();
        incorrect.dimensions[1] += 1;
        assert!(fixture.source.component(&incorrect).is_err());
    }
}
#[test]
fn basis_oracle_checks_every_weight_without_relaxing_decode_tolerance() {
    let shape = Shape {
        m: 512,
        n: 3,
        k: 512,
        blocks: Format::Q4K.blocks(),
    };
    let fixture = Fixture::new(&shape, Format::Q4K);
    let expected: Vec<u8> = (0..shape.k)
        .flat_map(|k| {
            let weights = &fixture.decoded;
            (0..shape.n).flat_map(move |n| weights[(n * shape.k + k) as usize].to_le_bytes())
        })
        .collect();
    check(
        &shape,
        &input(&shape, true),
        &fixture.decoded,
        &expected,
        true,
    );
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an exclusive CUDA test window"]
fn managed_native_block_projection_oracle_capacity_and_retirement() {
    runtime::verify();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn managed_metal_block_projection_oracle_capacity_and_retirement() {
    metal_runtime::verify();
}
