//! Four deterministic conversion vectors are a small materializer contract,
//! not model-quality or serving evidence. References are constructed from
//! independently specified coefficients, never decoded from candidate output.
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use half::f16;
use serde_json::{json, Value};

const EXECUTION_CONTRACT:&str="gguf-original-f32-decoder/rn-even-binary16-once/finite-no-overflow/contiguous-projection-components/v1";
const RELATIVE_L2_DENOMINATOR: u64 = 1024;

pub(super) struct Vector {
    pub format: GgufBlockFormat,
    pub source: Vec<u8>,
    pub reference: Vec<f32>,
}
pub(super) fn vectors() -> Vec<Vector> {
    let d = f16::from_bits(0x237b).to_f32();
    let m = f16::from_bits(0x159d).to_f32();
    [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8_0,
    ]
    .into_iter()
    .map(|format| {
        let mut source = vec![0; format.block_bytes()];
        let mut reference = Vec::with_capacity(format.block_values());
        match format {
            GgufBlockFormat::Q4K | GgufBlockFormat::Q5K => {
                source[..2].copy_from_slice(&0x237bu16.to_le_bytes());
                source[2..4].copy_from_slice(&0x159du16.to_le_bytes());
                source[4..12].fill(1);
                source[12..16].fill(0x11); // each group scale=minimum=1
                let values = if format == GgufBlockFormat::Q4K {
                    16
                } else {
                    32
                };
                let start = if format == GgufBlockFormat::Q4K {
                    16
                } else {
                    48
                };
                for i in 0..256 {
                    let q = ((i * 13 + 7) % values) as u8;
                    source[start + (i / 64) * 32 + i % 32] |= (q & 15) << (4 * ((i % 64) / 32));
                    if format == GgufBlockFormat::Q5K && q >= 16 {
                        source[16 + i % 32] |= 1 << (i / 32);
                    }
                    reference.push(d * f32::from(q) - m);
                }
            }
            GgufBlockFormat::Q6K => {
                source[192..208].fill(3);
                source[208..210].copy_from_slice(&0x237bu16.to_le_bytes());
                for i in 0..256 {
                    let code = (i % 64) as u8;
                    let group = (i % 128) / 32;
                    source[(i / 128) * 64 + (group % 2) * 32 + i % 32] |=
                        (code & 15) << (4 * (group / 2));
                    source[128 + (i / 128) * 32 + i % 32] |= (code >> 4) << (2 * group);
                    reference.push((d * 3.0) * (i32::from(code) - 32) as f32);
                }
            }
            GgufBlockFormat::Q8_0 => {
                source[..2].copy_from_slice(&0x237bu16.to_le_bytes());
                for i in 0..32 {
                    let q = ((i * 17) % 255) as i16 - 127;
                    source[2 + i] = q as i8 as u8;
                    reference.push(d * q as f32);
                }
            }
            _ => unreachable!(),
        }
        Vector {
            format,
            source,
            reference,
        }
    })
    .collect()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn reference_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}
fn checkpoint() -> Value {
    json!({"id":"checkpoint.ferrum.gguf-rn-f16.synthetic-v1","repository":"ferrum/gguf-rn-f16-conversion-vectors","revision":sha(include_bytes!("quality.rs"))})
}
fn payload() -> Value {
    let vectors = vectors();
    json!({
        "activation_batches":[],
        "activation_contract":{"kind":"weight-conversion-only-no-gemm"},
        "cases":vectors.iter().map(|v|json!({"case_id":v.format.format_id(),"source_bytes":v.source,"source_sha256":sha(&v.source),"reference_f32le_sha256":sha(&reference_bytes(&v.reference))})).collect::<Vec<_>>(),
        "checkpoint":checkpoint(),"fixture_id":"gguf-rn-f16-conversion-v1",
        "generator":{"algorithm":"independent-coefficient-q4-q5-q6-q8-v1"},
        "reference_contract":{"dtype":"f32","arithmetic":"original-gguf-coefficient-order"},"schema_version":1,
        "source_contract":{"rounding":"rn-even","nonfinite":"reject","overflow":"reject"},
        "weight_shapes":[[1,256],[1,256],[1,256],[1,32]],
    })
}
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(values) => {
            let sorted = values
                .into_iter()
                .map(|(k, v)| (k, canonical(v)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        v => v,
    }
}
fn encode(value: Value) -> Result<Vec<u8>, VNextError> {
    serde_json::to_vec(&canonical(value)).map_err(|e| invalid(e.to_string()))
}
pub(super) fn contract() -> Result<ApproximateWeightQualityContract, VNextError> {
    ApproximateWeightQualityContract::new(
        sha(EXECUTION_CONTRACT.as_bytes()),
        sha(&encode(payload())?),
        4,
        CanonicalRational::new(1, RELATIVE_L2_DENOMINATOR)?,
        0,
        0,
    )
}
pub(super) fn artifact(
    descriptor: &WeightMaterializerDescriptor,
    source: &WeightSchema,
    execution: &WeightSchema,
) -> Result<Vec<u8>, VNextError> {
    let mut cases = Vec::new();
    for vector in vectors() {
        let converted = conversion::convert(vector.format, &vector.source)?;
        let actual = converted
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>();
        let reference = vector
            .reference
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        cases.push(json!({"case_id":vector.format.format_id(),"actual_f16_bits":actual,"actual_f16le_sha256":sha(&converted),"reference_f32_bits":reference,"reference_f32le_sha256":sha(&reference_bytes(&vector.reference)),"relative_l2_upper_bound":CanonicalRational::new(1,RELATIVE_L2_DENOMINATOR)?,"nan_count":0,"inf_count":0}));
    }
    let contract = descriptor
        .approximate_quality_contract()
        .ok_or_else(|| invalid("RN-F16 locked quality contract absent"))?;
    encode(json!({
        "schema_id":NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID,
        "authority":{"id":NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID,"version":ContractVersion::new(1,0),"implementation_fingerprint":numeric_weight_quality_authority_implementation_fingerprint()?},
        "checkpoint":checkpoint(),
        "materializer":{"id":descriptor.id(),"version":descriptor.version(),"implementation_fingerprint":descriptor.implementation_fingerprint(),"fidelity":WeightMaterializationFidelity::Approximate},
        "source":{"weight_format_id":source.format_id},
        "execution":{"weight_format_id":execution.format_id,"weight_layout_id":execution.layout_id,"quantization_format_ids":execution.quantization_formats()},
        "contract":{"execution_contract_fingerprint":contract.execution_contract_fingerprint(),"quality_vector_digest":contract.quality_vector_digest()},
        "quality_vector_payload":payload(),"cases":cases,
    }))
}
