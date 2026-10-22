//! Four bounded conversion vectors. Original BF16 values independently define
//! the F32 references; no reference is inferred from quantized candidate bytes.
//! This artifact qualifies only this small conversion contract, not model
//! quality or CUDA GEMM accuracy.
use super::*;
use half::bf16;
use serde_json::{json, Value};

const EXECUTION: &str = "finite-bf16/ordered-group-amax/f32-div448/static-f32-reciprocal-multiply/e4m3fn-satfinite-rne/zero-scale1/reject-invalid-scale/v1";

fn vectors() -> [(&'static str, [u16; 8]); 4] {
    [
        ("zeros", [0, 0x8000, 0, 0x8000, 0, 0, 0, 0]),
        (
            "ties",
            [
                0x43e0, 0xc3e0, 0x3f88, 0x3f98, 0xbf88, 0xbf98, 0x3a80, 0x3b40,
            ],
        ),
        (
            "scale",
            [
                0x40e0, 0xc0e0, 0x3f81, 0xbf83, 0x3e81, 0xbe83, 0x3d01, 0xbd03,
            ],
        ),
        (
            "range",
            [
                0x4380, 0xc300, 0x4201, 0xc201, 0x3f87, 0xbf8b, 0x3c01, 0xbc07,
            ],
        ),
    ]
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn references(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn checkpoint() -> Value {
    json!({"id":"checkpoint.ferrum.bf16-fp8.conversion-v1", "repository":"ferrum/bf16-fp8-conversion-vectors", "revision":sha(include_bytes!("quality.rs"))})
}
fn payload() -> Value {
    json!({
        "schema_version":1,"fixture_id":"bf16-fp8-conversion-v1", "checkpoint":checkpoint(),
        "generator":{"algorithm":"explicit-bf16-bits-v1"},
        "source_contract":{"profile":EXECUTION},
        "reference_contract":{"dtype":"f32","arithmetic":"exact-bf16-expansion"},
        "activation_batches":[],"activation_contract":{"kind":"weight-conversion-only-no-gemm"},
        "weight_shapes":[[1,8],[1,8],[1,8],[1,8]],
        "cases":vectors().iter().map(|(name,bits)|json!({"case_id":name,"source_bf16_bits":bits,
            "reference_f32le_sha256":sha(&f32_bytes(&references(bits)))})).collect::<Vec<_>>()
    })
}
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k, canonical(v)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        value => value,
    }
}
fn encode(value: Value) -> Result<Vec<u8>, VNextError> {
    serde_json::to_vec(&canonical(value)).map_err(|e| invalid(e.to_string()))
}
pub(super) fn contract() -> Result<ApproximateWeightQualityContract, VNextError> {
    ApproximateWeightQualityContract::new(
        sha(EXECUTION.as_bytes()),
        sha(&encode(payload())?),
        4,
        CanonicalRational::new(1, 16)?,
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
    for (name, bits) in vectors() {
        let reference = references(&bits);
        let maximum = reference.iter().fold(0_f32, |max, v| max.max(v.abs()));
        let (scale, inverse) = conversion::scale(maximum)?;
        let actual: Vec<_> = reference
            .iter()
            .map(|x| {
                bf16::from_f32(conversion::decode(conversion::encode(x * inverse)) * scale)
                    .to_bits()
            })
            .collect();
        let actual_bytes: Vec<_> = actual.iter().flat_map(|x| x.to_le_bytes()).collect();
        cases.push(json!({"case_id":name,"actual_bf16_bits":actual,"actual_bf16le_sha256":sha(&actual_bytes),
            "reference_f32_bits":reference.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            "reference_f32le_sha256":sha(&f32_bytes(&reference)),
            "relative_l2_upper_bound":CanonicalRational::new(1,16)?,"nan_count":0,"inf_count":0}));
    }
    let contract = descriptor
        .approximate_quality_contract()
        .ok_or_else(|| invalid("quality contract absent"))?;
    encode(json!({
        "schema_id":NUMERIC_WEIGHT_QUALITY_BF16_ARTIFACT_SCHEMA_ID,
        "authority":{"id":NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID,"version":ContractVersion::new(2,0),
            "implementation_fingerprint":numeric_bf16_weight_quality_authority_implementation_fingerprint()?},
        "checkpoint":checkpoint(),
        "materializer":{"id":descriptor.id(),"version":descriptor.version(),
            "implementation_fingerprint":descriptor.implementation_fingerprint(),"fidelity":WeightMaterializationFidelity::Approximate},
        "source":{"weight_format_id":source.format_id},
        "execution":{"weight_format_id":execution.format_id,"weight_layout_id":execution.layout_id,
            "quantization_format_ids":execution.quantization_formats()},
        "contract":{"execution_contract_fingerprint":contract.execution_contract_fingerprint(),
            "quality_vector_digest":contract.quality_vector_digest()},
        "quality_vector_payload":payload(),"cases":cases
    }))
}
