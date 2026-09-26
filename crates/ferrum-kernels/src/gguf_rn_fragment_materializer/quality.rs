//! Four deterministic conversion vectors are a small materializer contract,
//! not model-quality or serving evidence. References are constructed from
//! independently specified coefficients, never decoded from candidate output.
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use crate::gguf_f16_projection_materializer as legacy;
use serde_json::{json, Value};

const EXECUTION_CONTRACT:&str="gguf-original-f32-decoder/rn-even-binary16-once/finite-no-overflow/dense-plus-exact-rn-fragment-N16-K32-v1";
const RELATIVE_L2_DENOMINATOR: u64 = 1024;

fn vectors() -> Vec<legacy::quality::Vector> {
    legacy::quality::vectors()
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
    json!({"id":"checkpoint.ferrum.gguf-rn-fragment.synthetic-v1","repository":"ferrum/gguf-rn-fragment-conversion-vectors","revision":sha(include_bytes!("quality.rs"))})
}
fn payload() -> Value {
    let vectors = vectors();
    json!({
        "activation_batches":[],
        "activation_contract":{"kind":"weight-conversion-only-no-gemm"},
        "cases":vectors.iter().map(|v|json!({"case_id":v.format.format_id(),"source_bytes":v.source,"source_sha256":sha(&v.source),"reference_f32le_sha256":sha(&reference_bytes(&v.reference))})).collect::<Vec<_>>(),
        "checkpoint":checkpoint(),"fixture_id":"gguf-rn-fragment-conversion-v1",
        "generator":{"algorithm":"independent-coefficient-q4-q5-q6-q8-v1"},
        "reference_contract":{"dtype":"f32","arithmetic":"original-gguf-coefficient-order"},"schema_version":1,
        "source_contract":{"rounding":"rn-even","nonfinite":"reject","overflow":"reject","fragment":"original quant codes and F32 coefficients; every reconstructed RN coefficient equals original dense conversion; whole-N tile padding"},
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
        let converted = legacy::conversion::convert(vector.format, &vector.source)?;
        let format = match vector.format {
            GgufBlockFormat::Q4K => Some(RnF16FragmentSourceFormatV1::Q4K),
            GgufBlockFormat::Q5K => Some(RnF16FragmentSourceFormatV1::Q5K),
            GgufBlockFormat::Q6K => Some(RnF16FragmentSourceFormatV1::Q6K),
            _ => None, // retained legacy RN Q8 attention vector
        };
        if let Some(format) = format {
            let plan = RnF16FragmentPlanV1::new(format, 1, 256)?;
            crate::gguf_rn_fragment::pack_checked(&plan, &[&vector.source], Some(&converted))?;
        }
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
