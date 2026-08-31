use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::vnext::{
    CanonicalRational, QuantizationFormatId, WeightComponentPayload, WeightComponentSource,
    WeightComponentSpec, WeightFormatId, WeightLayoutId,
};

use super::{
    canonical_fingerprint, canonical_json, invalid_plan, is_canonical_sha256, CapabilityCatalog,
    CapabilityId, ContractVersion, Deserialize, DeviceDescriptor, ModelFamilyId,
    PreparedModelFamily, ResourceId, Serialize, VNextError, WeightId, WeightMaterializerId,
    WeightSchema,
};

pub const IDENTITY_WEIGHT_MATERIALIZER_ID: &str = "weight-materializer.identity";
const IDENTITY_MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(2, 0);
pub const MAX_WEIGHT_MATERIALIZERS: usize = 64;
pub const NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID: &str =
    "quality-approval.weight-materializer.numeric.v1";
pub const NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID: &str = "quality-approval-authority.ferrum.numeric";
pub const MAX_APPROXIMATE_WEIGHT_QUALITY_ARTIFACT_BYTES: usize = 64 * 1024;
pub const STATIC_WEIGHT_TRANSFORM_SCRATCH_ALIGNMENT_BYTES: u64 = 256;

const NUMERIC_WEIGHT_QUALITY_AUTHORITY_VERSION: ContractVersion = ContractVersion::new(1, 0);
const REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES: usize = 4;
const MAX_NUMERIC_WEIGHT_QUALITY_VALUES_PER_CASE: usize = 8 * 1024;
const MAX_NUMERIC_WEIGHT_QUALITY_VALUES: usize = 16 * 1024;

#[derive(Serialize)]
struct NumericWeightQualityAuthorityFingerprint<'a> {
    id: &'a str,
    version: ContractVersion,
    artifact_schema_id: &'a str,
    actual_encoding: &'a str,
    reference_encoding: &'a str,
    metric: &'a str,
    verification_contract: &'a str,
    artifact_max_bytes: usize,
    required_cases: usize,
    maximum_values_per_case: usize,
    maximum_total_values: usize,
}

/// Canonical identity of the crate-owned numeric-artifact verification
/// contract. It changes only when the parser, encodings, metric, or containment
/// limits change; it is not a digest of a compiler binary or candidate SHA.
pub fn numeric_weight_quality_authority_implementation_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &NumericWeightQualityAuthorityFingerprint {
            id: NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID,
            version: NUMERIC_WEIGHT_QUALITY_AUTHORITY_VERSION,
            artifact_schema_id: NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID,
            actual_encoding: "ieee754-binary16-little-endian-bits",
            reference_encoding: "ieee754-binary32-little-endian-bits",
            metric: "norm(actual-reference)_2/max(norm(reference)_2,1e-6)",
            verification_contract: "strict-canonical-json+locked-vector-payload-sha256+case-reference-sha256+raw-vector-sha256+recomputed-nonfinite-and-relative-l2+live-schema-binding",
            artifact_max_bytes: MAX_APPROXIMATE_WEIGHT_QUALITY_ARTIFACT_BYTES,
            required_cases: REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES,
            maximum_values_per_case: MAX_NUMERIC_WEIGHT_QUALITY_VALUES_PER_CASE,
            maximum_total_values: MAX_NUMERIC_WEIGHT_QUALITY_VALUES,
        },
        "fingerprint approximate weight numeric quality authority",
    )
}

/// Typed compiler selection for one weight materializer.
///
/// Artifact bytes remain inert until the process-local registry verifies them
/// against the selected implementation, its checked-in quality contract, and
/// the live source and execution schemas. The bytes therefore cannot act as a
/// global precision override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightMaterializerSelection {
    materializer_id: WeightMaterializerId,
    numeric_quality_artifact: Option<Arc<[u8]>>,
}

impl WeightMaterializerSelection {
    pub fn exact(materializer_id: WeightMaterializerId) -> Self {
        Self {
            materializer_id,
            numeric_quality_artifact: None,
        }
    }

    pub fn numeric_quality_artifact(
        materializer_id: WeightMaterializerId,
        artifact_bytes: impl Into<Vec<u8>>,
    ) -> Result<Self, VNextError> {
        let artifact_bytes = artifact_bytes.into();
        decode_numeric_weight_quality_artifact(&artifact_bytes)?;
        Ok(Self {
            materializer_id,
            numeric_quality_artifact: Some(Arc::from(artifact_bytes)),
        })
    }

    pub fn materializer_id(&self) -> &WeightMaterializerId {
        &self.materializer_id
    }

    pub fn has_numeric_quality_artifact(&self) -> bool {
        self.numeric_quality_artifact.is_some()
    }

    fn numeric_quality_artifact_bytes(&self) -> Option<&[u8]> {
        self.numeric_quality_artifact.as_deref()
    }
}

/// Whether a physical weight transformation preserves source values.
///
/// Kernel availability is not authorization to change model precision.
/// Approximate materializers may be registered for capability discovery, but
/// require a separate numerical-quality approval path before plan selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMaterializationFidelity {
    Exact,
    Approximate,
}

/// Stable physical ABI of cached or device-produced execution weights.
///
/// This identity deliberately excludes the materializer implementation
/// fingerprint, compiler SHA, device model, and worker count. Those values are
/// useful runtime evidence, but none of them changes the bytes accepted by an
/// execution provider. Cache compatibility is therefore tied to this typed
/// ABI plus the source/component identity rather than to one producer build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightArtifactAbi {
    version: ContractVersion,
    weight_format_id: WeightFormatId,
    weight_layout_id: WeightLayoutId,
    quantization_format_ids: BTreeSet<QuantizationFormatId>,
}

impl WeightArtifactAbi {
    const VERSION: ContractVersion = ContractVersion::new(1, 0);

    fn from_schema(schema: &WeightSchema) -> Result<Self, VNextError> {
        let abi = Self {
            version: Self::VERSION,
            weight_format_id: schema.format_id.clone(),
            weight_layout_id: schema.layout_id.clone(),
            quantization_format_ids: schema.quantization_formats(),
        };
        abi.validate()?;
        Ok(abi)
    }

    pub const fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn weight_format_id(&self) -> &WeightFormatId {
        &self.weight_format_id
    }

    pub fn weight_layout_id(&self) -> &WeightLayoutId {
        &self.weight_layout_id
    }

    pub fn quantization_format_ids(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.quantization_format_ids
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint weight artifact ABI")
    }

    fn validate(&self) -> Result<(), VNextError> {
        if self.version != Self::VERSION {
            return Err(invalid_plan(
                "weight artifact ABI has an unsupported contract version",
            ));
        }
        Ok(())
    }
}

/// One required cold-path device transform from source checkpoint components
/// into final execution-weight components.
///
/// The first supported producer preserves official E4M3 bytes, converts the
/// BF16 inverse-scale grid into Marlin's grouped F16 scale ABI, and repacks one
/// logical matrix at a time. `matrices_per_output == 2` represents the routed
/// gate/up fusion `[E, 2, N, K] -> [E, 2N, K]`; all other projections use one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StaticWeightTransformPlan {
    BlockFp8ToMarlinFp8Group128 {
        source_values_id: WeightId,
        source_scales_id: WeightId,
        packed_values_id: WeightId,
        scales_id: WeightId,
        logical_dimensions: Vec<u64>,
        matrices_per_output: u32,
    },
    GptOssMxfp4ToMarlin {
        source_blocks_id: WeightId,
        source_scales_id: WeightId,
        packed_values_id: WeightId,
        scales_id: WeightId,
        logical_dimensions: Vec<u64>,
    },
}

impl StaticWeightTransformPlan {
    pub fn source_component_ids(&self) -> [&WeightId; 2] {
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                source_values_id,
                source_scales_id,
                ..
            } => [source_values_id, source_scales_id],
            Self::GptOssMxfp4ToMarlin {
                source_blocks_id,
                source_scales_id,
                ..
            } => [source_blocks_id, source_scales_id],
        }
    }

    pub fn execution_component_ids(&self) -> [&WeightId; 2] {
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                packed_values_id,
                scales_id,
                ..
            } => [packed_values_id, scales_id],
            Self::GptOssMxfp4ToMarlin {
                packed_values_id,
                scales_id,
                ..
            } => [packed_values_id, scales_id],
        }
    }

    pub fn logical_dimensions(&self) -> &[u64] {
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                logical_dimensions, ..
            } => logical_dimensions,
            Self::GptOssMxfp4ToMarlin {
                logical_dimensions, ..
            } => logical_dimensions,
        }
    }

    pub const fn matrices_per_output(&self) -> u32 {
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                matrices_per_output,
                ..
            } => *matrices_per_output,
            Self::GptOssMxfp4ToMarlin { .. } => 1,
        }
    }

    /// Device scratch is shared across transforms and sized for the largest
    /// single fused output matrix, never for the model or expert count.
    pub fn scratch_bytes(&self) -> Result<u64, VNextError> {
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                logical_dimensions,
                matrices_per_output,
                ..
            } => {
                if logical_dimensions.len() < 2 {
                    return Err(invalid_plan(
                        "static block-FP8 transform requires at least two dimensions",
                    ));
                }
                let [n, k] = logical_dimensions[logical_dimensions.len() - 2..] else {
                    unreachable!("two-axis slice has exact length")
                };
                n.checked_mul(u64::from(*matrices_per_output))
                    .and_then(|rows| rows.checked_mul(k))
                    .ok_or_else(|| {
                        invalid_plan("static block-FP8 transform scratch size overflows u64")
                    })
            }
            Self::GptOssMxfp4ToMarlin {
                logical_dimensions, ..
            } => {
                let [_, n, k] = logical_dimensions.as_slice() else {
                    return Err(invalid_plan(
                        "static GPT-OSS MXFP4 transform requires [E,N,K] dimensions",
                    ));
                };
                n.checked_mul(*k)
                    .and_then(|weights| weights.checked_div(2))
                    .ok_or_else(|| {
                        invalid_plan("static GPT-OSS MXFP4 transform scratch size overflows u64")
                    })
            }
        }
    }

    fn validate(&self) -> Result<(), VNextError> {
        let source_ids = self.source_component_ids();
        let execution_ids = self.execution_component_ids();
        if source_ids[0] == source_ids[1] || execution_ids[0] == execution_ids[1] {
            return Err(invalid_plan(
                "static weight transform has repeated source or execution component identities",
            ));
        }
        match self {
            Self::BlockFp8ToMarlinFp8Group128 {
                logical_dimensions,
                matrices_per_output,
                ..
            } => {
                if logical_dimensions.len() < 2 {
                    return Err(invalid_plan(
                        "static block-FP8 transform requires matrix dimensions",
                    ));
                }
                let n = logical_dimensions[logical_dimensions.len() - 2];
                let k = logical_dimensions[logical_dimensions.len() - 1];
                let source_matrix_count = logical_dimensions[..logical_dimensions.len() - 2]
                    .iter()
                    .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| invalid_plan("static block-FP8 matrix count overflows u64"))?;
                let fused_prefix_is_typed = *matrices_per_output == 1
                    || (*matrices_per_output == 2
                        && logical_dimensions.len() >= 4
                        && logical_dimensions[logical_dimensions.len() - 3] == 2);
                if n == 0
                    || k == 0
                    || !n.is_multiple_of(128)
                    || !k.is_multiple_of(128)
                    || !matches!(*matrices_per_output, 1 | 2)
                    || !source_matrix_count.is_multiple_of(u64::from(*matrices_per_output))
                    || !fused_prefix_is_typed
                    || self.scratch_bytes()? == 0
                {
                    return Err(invalid_plan(
                        "static block-FP8 group-128 transform has invalid shape, fusion, or scratch demand",
                    ));
                }
            }
            Self::GptOssMxfp4ToMarlin {
                logical_dimensions, ..
            } => {
                let [experts, n, k] = logical_dimensions.as_slice() else {
                    return Err(invalid_plan(
                        "static GPT-OSS MXFP4 transform requires [E,N,K] dimensions",
                    ));
                };
                if *experts == 0
                    || *n == 0
                    || *k == 0
                    || !n.is_multiple_of(64)
                    || !k.is_multiple_of(64)
                    || self.scratch_bytes()? == 0
                {
                    return Err(invalid_plan(
                        "static GPT-OSS MXFP4 transform requires positive E and 64-aligned N/K",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Checked-in numerical policy for an approximate materializer.
///
/// This is not an approval record. The public compiler remains exact-only;
/// M3 must add a crate-owned verifier that consumes real numeric artifact
/// bytes before an approximate materializer can be selected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproximateWeightQualityContract {
    execution_contract_fingerprint: String,
    quality_vector_digest: String,
    required_case_count: u32,
    relative_l2_max: CanonicalRational,
    nan_count_max: u64,
    inf_count_max: u64,
}

impl ApproximateWeightQualityContract {
    pub fn new(
        execution_contract_fingerprint: impl Into<String>,
        quality_vector_digest: impl Into<String>,
        required_case_count: u32,
        relative_l2_max: CanonicalRational,
        nan_count_max: u64,
        inf_count_max: u64,
    ) -> Result<Self, VNextError> {
        let contract = Self {
            execution_contract_fingerprint: execution_contract_fingerprint.into(),
            quality_vector_digest: quality_vector_digest.into(),
            required_case_count,
            relative_l2_max,
            nan_count_max,
            inf_count_max,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn execution_contract_fingerprint(&self) -> &str {
        &self.execution_contract_fingerprint
    }

    pub fn quality_vector_digest(&self) -> &str {
        &self.quality_vector_digest
    }

    pub const fn required_case_count(&self) -> u32 {
        self.required_case_count
    }

    pub const fn relative_l2_max(&self) -> CanonicalRational {
        self.relative_l2_max
    }

    pub const fn nan_count_max(&self) -> u64 {
        self.nan_count_max
    }

    pub const fn inf_count_max(&self) -> u64 {
        self.inf_count_max
    }

    fn validate(&self) -> Result<(), VNextError> {
        if !is_canonical_sha256(&self.execution_contract_fingerprint)
            || !is_canonical_sha256(&self.quality_vector_digest)
            || self.required_case_count == 0
            || self.relative_l2_max.numerator() <= 0
        {
            return Err(invalid_plan(
                "approximate weight quality contract has invalid digests, case count, or threshold",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifact {
    schema_id: String,
    authority: NumericWeightQualityArtifactAuthority,
    checkpoint: NumericWeightQualityArtifactCheckpoint,
    materializer: NumericWeightQualityArtifactMaterializer,
    source: NumericWeightQualityArtifactSource,
    execution: NumericWeightQualityArtifactExecution,
    contract: NumericWeightQualityArtifactContract,
    quality_vector_payload: serde_json::Value,
    cases: Vec<NumericWeightQualityArtifactCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactAuthority {
    id: String,
    version: ContractVersion,
    implementation_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactCheckpoint {
    id: String,
    repository: String,
    revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactMaterializer {
    id: WeightMaterializerId,
    version: ContractVersion,
    implementation_fingerprint: String,
    fidelity: WeightMaterializationFidelity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactSource {
    weight_format_id: WeightFormatId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactExecution {
    weight_format_id: WeightFormatId,
    weight_layout_id: WeightLayoutId,
    quantization_format_ids: BTreeSet<QuantizationFormatId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactContract {
    execution_contract_fingerprint: String,
    quality_vector_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericWeightQualityArtifactCase {
    case_id: String,
    actual_f16le_sha256: String,
    actual_f16_bits: Vec<u16>,
    reference_f32le_sha256: String,
    reference_f32_bits: Vec<u32>,
    relative_l2_upper_bound: CanonicalRational,
    nan_count: u64,
    inf_count: u64,
}

/// Serializable receipt embedded into an approximate execution-weight plan.
///
/// This record is evidence, not authority: only the non-serializable
/// [`TrustedExecutionWeightPlan`] produced by the process-local registry can
/// carry it into an executable plan. It binds reusable numeric evidence to the
/// exact live source and execution schema fingerprints selected for this plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproximateWeightQualityApprovalRecord {
    authority_id: String,
    authority_version: ContractVersion,
    authority_implementation_fingerprint: String,
    artifact_sha256: String,
    source_schema_fingerprint: String,
    execution_schema_fingerprint: String,
    execution_contract_fingerprint: String,
    quality_vector_digest: String,
    completed_case_count: u32,
    relative_l2_max_observed: CanonicalRational,
    nan_count: u64,
    inf_count: u64,
}

impl ApproximateWeightQualityApprovalRecord {
    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub const fn authority_version(&self) -> ContractVersion {
        self.authority_version
    }

    pub fn authority_implementation_fingerprint(&self) -> &str {
        &self.authority_implementation_fingerprint
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn source_schema_fingerprint(&self) -> &str {
        &self.source_schema_fingerprint
    }

    pub fn execution_schema_fingerprint(&self) -> &str {
        &self.execution_schema_fingerprint
    }

    pub fn execution_contract_fingerprint(&self) -> &str {
        &self.execution_contract_fingerprint
    }

    pub fn quality_vector_digest(&self) -> &str {
        &self.quality_vector_digest
    }

    pub const fn completed_case_count(&self) -> u32 {
        self.completed_case_count
    }

    pub const fn relative_l2_max_observed(&self) -> CanonicalRational {
        self.relative_l2_max_observed
    }

    pub const fn nan_count(&self) -> u64 {
        self.nan_count
    }

    pub const fn inf_count(&self) -> u64 {
        self.inf_count
    }

    fn validate_structure(&self) -> Result<(), VNextError> {
        if self.authority_id != NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID
            || self.authority_version != NUMERIC_WEIGHT_QUALITY_AUTHORITY_VERSION
            || self.authority_implementation_fingerprint
                != numeric_weight_quality_authority_implementation_fingerprint()?
            || !is_canonical_sha256(&self.artifact_sha256)
            || !is_canonical_sha256(&self.source_schema_fingerprint)
            || !is_canonical_sha256(&self.execution_schema_fingerprint)
            || !is_canonical_sha256(&self.execution_contract_fingerprint)
            || !is_canonical_sha256(&self.quality_vector_digest)
            || self.completed_case_count != REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES as u32
            || self.relative_l2_max_observed.numerator() < 0
        {
            return Err(invalid_plan(
                "approximate weight quality approval record is structurally invalid",
            ));
        }
        Ok(())
    }

    fn validate_against(
        &self,
        source_schema_fingerprint: &str,
        execution_schema_fingerprint: &str,
        quality_contract: &ApproximateWeightQualityContract,
    ) -> Result<(), VNextError> {
        self.validate_structure()?;
        if self.source_schema_fingerprint != source_schema_fingerprint
            || self.execution_schema_fingerprint != execution_schema_fingerprint
            || self.execution_contract_fingerprint
                != quality_contract.execution_contract_fingerprint()
            || self.quality_vector_digest != quality_contract.quality_vector_digest()
            || self.completed_case_count != quality_contract.required_case_count()
            || !nonnegative_rational_le(
                self.relative_l2_max_observed,
                quality_contract.relative_l2_max(),
            )
            || self.nan_count > quality_contract.nan_count_max()
            || self.inf_count > quality_contract.inf_count_max()
        {
            return Err(invalid_plan(
                "approximate weight quality approval differs from the live materializer or schema contract",
            ));
        }
        Ok(())
    }
}

fn decode_numeric_weight_quality_artifact(
    artifact_bytes: &[u8],
) -> Result<NumericWeightQualityArtifact, VNextError> {
    if artifact_bytes.is_empty()
        || artifact_bytes.len() > MAX_APPROXIMATE_WEIGHT_QUALITY_ARTIFACT_BYTES
    {
        return Err(invalid_plan(format!(
            "approximate weight quality artifact must contain 1..={MAX_APPROXIMATE_WEIGHT_QUALITY_ARTIFACT_BYTES} bytes"
        )));
    }
    let artifact: NumericWeightQualityArtifact =
        serde_json::from_slice(artifact_bytes).map_err(|error| {
            invalid_plan(format!(
                "approximate weight quality artifact is not strict schema-valid JSON: {error}"
            ))
        })?;
    let canonical_bytes = serde_json::to_value(&artifact)
        .map(canonical_json)
        .and_then(|value| serde_json::to_vec(&value))
        .map_err(|error| VNextError::Serialization {
            context: "canonicalize approximate weight quality artifact",
            message: error.to_string(),
        })?;
    if canonical_bytes != artifact_bytes {
        return Err(invalid_plan(
            "approximate weight quality artifact is not canonical compact JSON",
        ));
    }
    if artifact.schema_id != NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID
        || artifact.authority.id != NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID
        || artifact.authority.version != NUMERIC_WEIGHT_QUALITY_AUTHORITY_VERSION
        || artifact.authority.implementation_fingerprint
            != numeric_weight_quality_authority_implementation_fingerprint()?
    {
        return Err(invalid_plan(
            "approximate weight quality artifact names a different verification authority",
        ));
    }
    if !portable_artifact_text(&artifact.checkpoint.id)
        || !portable_artifact_text(&artifact.checkpoint.repository)
        || !canonical_revision(&artifact.checkpoint.revision)
    {
        return Err(invalid_plan(
            "approximate weight quality artifact checkpoint identity is invalid",
        ));
    }
    if artifact.materializer.version.major == 0
        || !is_canonical_sha256(&artifact.materializer.implementation_fingerprint)
        || artifact.materializer.fidelity != WeightMaterializationFidelity::Approximate
        || !is_canonical_sha256(&artifact.contract.execution_contract_fingerprint)
        || !is_canonical_sha256(&artifact.contract.quality_vector_digest)
        || artifact.cases.len() != REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES
    {
        return Err(invalid_plan(
            "approximate weight quality artifact has invalid materializer, contract, or case structure",
        ));
    }
    let quality_vector_bytes = serde_json::to_vec(&canonical_json(
        artifact.quality_vector_payload.clone(),
    ))
    .map_err(|error| VNextError::Serialization {
        context: "canonicalize approximate weight quality vector payload",
        message: error.to_string(),
    })?;
    if format!("{:x}", Sha256::digest(&quality_vector_bytes))
        != artifact.contract.quality_vector_digest
    {
        return Err(invalid_plan(
            "approximate weight quality artifact does not contain the locked quality vector payload",
        ));
    }
    let vector_references =
        quality_vector_references(&artifact.quality_vector_payload, &artifact.checkpoint)?;
    let mut case_ids = BTreeSet::new();
    let mut total_values = 0_usize;
    for case in &artifact.cases {
        let value_count = case.actual_f16_bits.len();
        if !portable_artifact_text(&case.case_id)
            || !case_ids.insert(case.case_id.clone())
            || value_count == 0
            || value_count != case.reference_f32_bits.len()
            || value_count > MAX_NUMERIC_WEIGHT_QUALITY_VALUES_PER_CASE
            || !is_canonical_sha256(&case.actual_f16le_sha256)
            || !is_canonical_sha256(&case.reference_f32le_sha256)
            || case.relative_l2_upper_bound.numerator() < 0
        {
            return Err(invalid_plan(
                "approximate weight quality artifact has an invalid case identity, vector, digest, or metric structure",
            ));
        }
        if vector_references.get(&case.case_id) != Some(&case.reference_f32le_sha256) {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` reference differs from the locked quality vector",
                case.case_id
            )));
        }
        total_values = total_values.checked_add(value_count).ok_or_else(|| {
            invalid_plan("approximate weight quality artifact value count overflows usize")
        })?;
        if total_values > MAX_NUMERIC_WEIGHT_QUALITY_VALUES {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact exceeds {MAX_NUMERIC_WEIGHT_QUALITY_VALUES} total values"
            )));
        }
    }
    if case_ids != vector_references.keys().cloned().collect() {
        return Err(invalid_plan(
            "approximate weight quality artifact cases differ from the locked quality vector",
        ));
    }
    Ok(artifact)
}

fn verify_numeric_weight_quality_artifact(
    artifact_bytes: &[u8],
    descriptor: &WeightMaterializerDescriptor,
    family: &PreparedModelFamily,
    execution_schema: &WeightSchema,
) -> Result<ApproximateWeightQualityApprovalRecord, VNextError> {
    let artifact = decode_numeric_weight_quality_artifact(artifact_bytes)?;
    let quality_contract = descriptor.approximate_quality_contract().ok_or_else(|| {
        invalid_plan(format!(
            "approximate weight materializer `{}` has no numerical quality contract",
            descriptor.id()
        ))
    })?;
    if artifact.materializer.id != *descriptor.id()
        || artifact.materializer.version != descriptor.version()
        || artifact.materializer.implementation_fingerprint
            != descriptor.implementation_fingerprint()
        || artifact.materializer.fidelity != descriptor.fidelity()
        || descriptor.fidelity() != WeightMaterializationFidelity::Approximate
    {
        return Err(invalid_plan(
            "approximate weight quality artifact differs from the selected materializer",
        ));
    }
    if artifact.source.weight_format_id != family.weight_schema().format_id
        || artifact.execution.weight_format_id != execution_schema.format_id
        || artifact.execution.weight_layout_id != execution_schema.layout_id
        || artifact.execution.quantization_format_ids != execution_schema.quantization_formats()
    {
        return Err(invalid_plan(
            "approximate weight quality artifact differs from the live source or execution format contract",
        ));
    }
    if artifact.contract.execution_contract_fingerprint
        != quality_contract.execution_contract_fingerprint()
        || artifact.contract.quality_vector_digest != quality_contract.quality_vector_digest()
        || quality_contract.required_case_count() as usize != REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES
        || artifact.cases.len() != REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES
    {
        return Err(invalid_plan(
            "approximate weight quality artifact differs from the checked-in quality contract",
        ));
    }
    let quality_vector_bytes = serde_json::to_vec(&canonical_json(
        artifact.quality_vector_payload.clone(),
    ))
    .map_err(|error| VNextError::Serialization {
        context: "canonicalize approximate weight quality vector payload",
        message: error.to_string(),
    })?;
    if format!("{:x}", Sha256::digest(&quality_vector_bytes))
        != quality_contract.quality_vector_digest()
    {
        return Err(invalid_plan(
            "approximate weight quality artifact does not contain the locked quality vector payload",
        ));
    }
    let vector_references =
        quality_vector_references(&artifact.quality_vector_payload, &artifact.checkpoint)?;

    let mut case_ids = BTreeSet::new();
    let mut total_values = 0_usize;
    let mut total_nan_count = 0_u64;
    let mut total_inf_count = 0_u64;
    let mut maximum_upper_bound = CanonicalRational::new(0, 1)?;
    for case in &artifact.cases {
        let value_count = case.actual_f16_bits.len();
        if !portable_artifact_text(&case.case_id)
            || !case_ids.insert(case.case_id.clone())
            || value_count == 0
            || value_count != case.reference_f32_bits.len()
            || value_count > MAX_NUMERIC_WEIGHT_QUALITY_VALUES_PER_CASE
        {
            return Err(invalid_plan(
                "approximate weight quality artifact has an invalid case identity or vector size",
            ));
        }
        if vector_references.get(&case.case_id) != Some(&case.reference_f32le_sha256) {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` reference differs from the locked quality vector",
                case.case_id
            )));
        }
        total_values = total_values.checked_add(value_count).ok_or_else(|| {
            invalid_plan("approximate weight quality artifact value count overflows usize")
        })?;
        if total_values > MAX_NUMERIC_WEIGHT_QUALITY_VALUES {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact exceeds {MAX_NUMERIC_WEIGHT_QUALITY_VALUES} total values"
            )));
        }
        if digest_little_endian_u16(&case.actual_f16_bits) != case.actual_f16le_sha256
            || digest_little_endian_u32(&case.reference_f32_bits) != case.reference_f32le_sha256
        {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` raw-vector digest differs",
                case.case_id
            )));
        }

        let (relative_l2, nan_count, inf_count) = recompute_relative_l2(case)?;
        if nan_count != case.nan_count || inf_count != case.inf_count {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` reports incorrect NaN or Inf counts",
                case.case_id
            )));
        }
        total_nan_count = total_nan_count.checked_add(nan_count).ok_or_else(|| {
            invalid_plan("approximate weight quality artifact NaN count overflows u64")
        })?;
        total_inf_count = total_inf_count.checked_add(inf_count).ok_or_else(|| {
            invalid_plan("approximate weight quality artifact Inf count overflows u64")
        })?;
        if case.relative_l2_upper_bound.numerator() < 0
            || relative_l2 > rational_as_f64(case.relative_l2_upper_bound)
            || !nonnegative_rational_le(
                case.relative_l2_upper_bound,
                quality_contract.relative_l2_max(),
            )
        {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` exceeds or understates its relative-L2 contract",
                case.case_id
            )));
        }
        if nonnegative_rational_le(maximum_upper_bound, case.relative_l2_upper_bound) {
            maximum_upper_bound = case.relative_l2_upper_bound;
        }
    }
    if case_ids != vector_references.keys().cloned().collect() {
        return Err(invalid_plan(
            "approximate weight quality artifact cases differ from the locked quality vector",
        ));
    }
    if total_nan_count > quality_contract.nan_count_max()
        || total_inf_count > quality_contract.inf_count_max()
    {
        return Err(invalid_plan(
            "approximate weight quality artifact exceeds its non-finite output contract",
        ));
    }

    let record = ApproximateWeightQualityApprovalRecord {
        authority_id: NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID.to_owned(),
        authority_version: NUMERIC_WEIGHT_QUALITY_AUTHORITY_VERSION,
        authority_implementation_fingerprint:
            numeric_weight_quality_authority_implementation_fingerprint()?,
        artifact_sha256: format!("{:x}", Sha256::digest(artifact_bytes)),
        source_schema_fingerprint: family.weight_schema().fingerprint()?,
        execution_schema_fingerprint: execution_schema.fingerprint()?,
        execution_contract_fingerprint: quality_contract
            .execution_contract_fingerprint()
            .to_owned(),
        quality_vector_digest: quality_contract.quality_vector_digest().to_owned(),
        completed_case_count: u32::try_from(artifact.cases.len()).map_err(|_| {
            invalid_plan("approximate weight quality artifact case count exceeds u32")
        })?,
        relative_l2_max_observed: maximum_upper_bound,
        nan_count: total_nan_count,
        inf_count: total_inf_count,
    };
    record.validate_against(
        &family.weight_schema().fingerprint()?,
        &execution_schema.fingerprint()?,
        quality_contract,
    )?;
    Ok(record)
}

fn quality_vector_references(
    payload: &serde_json::Value,
    checkpoint: &NumericWeightQualityArtifactCheckpoint,
) -> Result<BTreeMap<String, String>, VNextError> {
    const ROOT_KEYS: [&str; 10] = [
        "activation_batches",
        "activation_contract",
        "cases",
        "checkpoint",
        "fixture_id",
        "generator",
        "reference_contract",
        "schema_version",
        "source_contract",
        "weight_shapes",
    ];
    let root = payload.as_object().ok_or_else(|| {
        invalid_plan("approximate weight quality vector payload must be an object")
    })?;
    if root.keys().map(String::as_str).collect::<BTreeSet<_>>() != ROOT_KEYS.into_iter().collect() {
        return Err(invalid_plan(
            "approximate weight quality vector payload has unexpected root fields",
        ));
    }
    let payload_checkpoint = root
        .get("checkpoint")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            invalid_plan("approximate weight quality vector checkpoint must be an object")
        })?;
    if payload_checkpoint
        .get("id")
        .and_then(serde_json::Value::as_str)
        != Some(checkpoint.id.as_str())
        || payload_checkpoint
            .get("repository")
            .and_then(serde_json::Value::as_str)
            != Some(checkpoint.repository.as_str())
        || payload_checkpoint
            .get("revision")
            .and_then(serde_json::Value::as_str)
            != Some(checkpoint.revision.as_str())
    {
        return Err(invalid_plan(
            "approximate weight quality artifact checkpoint differs from its locked vector",
        ));
    }
    let cases = root
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_plan("approximate weight quality vector cases must be an array"))?;
    if cases.len() != REQUIRED_NUMERIC_WEIGHT_QUALITY_CASES {
        return Err(invalid_plan(
            "approximate weight quality vector must contain exactly four cases",
        ));
    }
    let mut references = BTreeMap::new();
    for case in cases {
        let case = case.as_object().ok_or_else(|| {
            invalid_plan("approximate weight quality vector case must be an object")
        })?;
        let case_id = case
            .get("case_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| portable_artifact_text(value))
            .ok_or_else(|| {
                invalid_plan("approximate weight quality vector case identity is invalid")
            })?;
        let reference = case
            .get("reference_f32le_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| is_canonical_sha256(value))
            .ok_or_else(|| {
                invalid_plan("approximate weight quality vector reference digest is invalid")
            })?;
        if references
            .insert(case_id.to_owned(), reference.to_owned())
            .is_some()
        {
            return Err(invalid_plan(
                "approximate weight quality vector has duplicate case identities",
            ));
        }
    }
    Ok(references)
}

fn portable_artifact_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 160
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        })
}

fn canonical_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_little_endian_u16(values: &[u16]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn digest_little_endian_u32(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn recompute_relative_l2(
    case: &NumericWeightQualityArtifactCase,
) -> Result<(f64, u64, u64), VNextError> {
    let mut error_squared = 0_f64;
    let mut reference_squared = 0_f64;
    let mut nan_count = 0_u64;
    let mut inf_count = 0_u64;
    for (&actual_bits, &reference_bits) in case.actual_f16_bits.iter().zip(&case.reference_f32_bits)
    {
        let actual = binary16_as_f32(actual_bits);
        if actual.is_nan() {
            nan_count += 1;
            continue;
        }
        if actual.is_infinite() {
            inf_count += 1;
            continue;
        }
        let reference = f32::from_bits(reference_bits);
        if !reference.is_finite() {
            return Err(invalid_plan(format!(
                "approximate weight quality artifact case `{}` has a non-finite reference",
                case.case_id
            )));
        }
        let error = f64::from(actual - reference);
        error_squared += error * error;
        let reference = f64::from(reference);
        reference_squared += reference * reference;
    }
    if nan_count != 0 || inf_count != 0 {
        return Ok((f64::INFINITY, nan_count, inf_count));
    }
    let relative_l2 = error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6);
    if !relative_l2.is_finite() {
        return Err(invalid_plan(format!(
            "approximate weight quality artifact case `{}` produced a non-finite relative L2",
            case.case_id
        )));
    }
    Ok((relative_l2, nan_count, inf_count))
}

fn binary16_as_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 {
        1.0_f32
    } else {
        -1.0_f32
    };
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    match (exponent, fraction) {
        (0, 0) => sign * 0.0,
        (0, fraction) => sign * fraction as f32 * 2_f32.powi(-24),
        (0x1f, 0) => sign * f32::INFINITY,
        (0x1f, _) => f32::NAN,
        (exponent, fraction) => {
            sign * (1.0 + fraction as f32 / 1024.0) * 2_f32.powi(exponent as i32 - 15)
        }
    }
}

fn rational_as_f64(value: CanonicalRational) -> f64 {
    value.numerator() as f64 / value.denominator() as f64
}

fn nonnegative_rational_le(left: CanonicalRational, right: CanonicalRational) -> bool {
    left.numerator() >= 0
        && right.numerator() >= 0
        && i128::from(left.numerator()) * i128::from(right.denominator())
            <= i128::from(right.numerator()) * i128::from(left.denominator())
}

#[derive(Serialize)]
struct IdentityMaterializerFingerprint<'a> {
    id: &'a str,
    version: ContractVersion,
    contract: &'a str,
}

fn identity_materializer_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &IdentityMaterializerFingerprint {
            id: IDENTITY_WEIGHT_MATERIALIZER_ID,
            version: IDENTITY_MATERIALIZER_VERSION,
            contract: "execution-weight-plan.identity.v2",
        },
        "fingerprint identity weight materializer",
    )
}

/// Serializable capability identity for one trusted weight transformation.
///
/// The descriptor advertises availability, but cannot construct an execution
/// schema. Only the process-local [`WeightMaterializerRegistry`] retains the
/// implementation object that may produce a trusted plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightMaterializerDescriptor {
    id: WeightMaterializerId,
    version: ContractVersion,
    implementation_fingerprint: String,
    fidelity: WeightMaterializationFidelity,
    required_capabilities: BTreeSet<CapabilityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    approximate_quality_contract: Option<ApproximateWeightQualityContract>,
}

impl WeightMaterializerDescriptor {
    pub fn new(
        id: WeightMaterializerId,
        version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        fidelity: WeightMaterializationFidelity,
        required_capabilities: BTreeSet<CapabilityId>,
    ) -> Result<Self, VNextError> {
        let descriptor = Self {
            id,
            version,
            implementation_fingerprint: implementation_fingerprint.into(),
            fidelity,
            required_capabilities,
            approximate_quality_contract: None,
        };
        descriptor.validate_structure()?;
        Ok(descriptor)
    }

    pub(crate) fn identity() -> Result<Self, VNextError> {
        Self::new(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID)?,
            IDENTITY_MATERIALIZER_VERSION,
            identity_materializer_fingerprint()?,
            WeightMaterializationFidelity::Exact,
            BTreeSet::new(),
        )
    }

    pub fn id(&self) -> &WeightMaterializerId {
        &self.id
    }

    pub const fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn implementation_fingerprint(&self) -> &str {
        &self.implementation_fingerprint
    }

    pub const fn fidelity(&self) -> WeightMaterializationFidelity {
        self.fidelity
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn with_approximate_quality_contract(
        mut self,
        contract: ApproximateWeightQualityContract,
    ) -> Result<Self, VNextError> {
        if self.fidelity != WeightMaterializationFidelity::Approximate {
            return Err(invalid_plan(format!(
                "exact weight materializer `{}` cannot carry an approximate quality contract",
                self.id
            )));
        }
        contract.validate()?;
        self.approximate_quality_contract = Some(contract);
        Ok(self)
    }

    pub fn approximate_quality_contract(&self) -> Option<&ApproximateWeightQualityContract> {
        self.approximate_quality_contract.as_ref()
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint weight materializer descriptor")
    }

    pub(crate) fn validate_for_device(&self, device: &DeviceDescriptor) -> Result<(), VNextError> {
        self.validate_structure()?;
        if !self.required_capabilities.is_subset(&device.capabilities) {
            return Err(invalid_plan(format!(
                "weight materializer `{}` requires capabilities absent from device `{}`",
                self.id, device.id
            )));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), VNextError> {
        if self.version.major == 0 || !is_canonical_sha256(&self.implementation_fingerprint) {
            return Err(invalid_plan(format!(
                "weight materializer descriptor `{}` has invalid version or implementation identity",
                self.id
            )));
        }
        if let Some(contract) = &self.approximate_quality_contract {
            contract.validate()?;
            if self.fidelity != WeightMaterializationFidelity::Approximate {
                return Err(invalid_plan(format!(
                    "exact weight materializer `{}` cannot carry an approximate quality contract",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Trusted authority for one physical weight transformation.
///
/// Implementations are invoked once while compiling an immutable plan. They
/// must derive the complete execution schema from the prepared family and
/// device descriptor without allocating device memory or reading request
/// state. During static initialization the same retained implementation turns
/// source components into the exact execution components recorded in that
/// plan; no provider or weight source may bypass this authority.
pub trait WeightMaterializer: Send + Sync {
    fn descriptor(&self) -> &WeightMaterializerDescriptor;

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError>;

    /// Ordered source-component identities for every execution component.
    ///
    /// The default only permits a same-id mapping. Materializers that create
    /// derived component identities or combine multiple source components must
    /// declare that provenance explicitly.
    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        identity_component_sources(family, execution_schema)
    }

    /// Required device-side transforms for final execution artifacts.
    ///
    /// An empty set means every execution component is produced as a host
    /// payload through [`Self::materialize_components`]. A non-empty set is a
    /// fail-closed contract: static initialization must use a runtime that
    /// implements the exact transform and must never call the host
    /// materializer for those outputs.
    fn static_weight_transforms(
        &self,
        _family: &PreparedModelFamily,
        _execution_schema: &WeightSchema,
    ) -> Result<Vec<StaticWeightTransformPlan>, VNextError> {
        Ok(Vec::new())
    }

    /// Materialize one execution component on the cold initialization path.
    ///
    /// `source_components` is resolved from the immutable plan's ordered source
    /// map. Implementations may borrow source bytes for identity layouts or
    /// return owned bytes for repacked/quantized layouts.
    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError>;

    /// Materialize execution components that share one exact ordered source set.
    ///
    /// The default preserves existing one-component implementations. Expensive
    /// transforms may override this method to decode, quantize, or repack their
    /// common source once and return one payload per requested component in the
    /// same order.
    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        execution_components
            .iter()
            .map(|component| self.materialize_component(source, source_components, component))
            .collect()
    }
}

struct IdentityWeightMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl IdentityWeightMaterializer {
    fn new() -> Result<Self, VNextError> {
        Ok(Self {
            descriptor: WeightMaterializerDescriptor::identity()?,
        })
    }
}

impl WeightMaterializer for IdentityWeightMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        Ok(family.weight_schema().clone())
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [source_component] = source_components else {
            return Err(invalid_plan(
                "identity weight materializer requires exactly one source component",
            ));
        };
        if *source_component != execution_component {
            return Err(invalid_plan(format!(
                "identity weight materializer cannot transform component `{}`",
                execution_component.id
            )));
        }
        source.component(source_component)
    }
}

fn identity_component_sources(
    family: &PreparedModelFamily,
    execution_schema: &WeightSchema,
) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
    let source_ids = family
        .weight_schema()
        .components
        .iter()
        .map(|component| component.id.clone())
        .collect::<BTreeSet<_>>();
    execution_schema
        .components
        .iter()
        .map(|component| {
            if !source_ids.contains(&component.id) {
                return Err(invalid_plan(format!(
                    "weight materializer must declare sources for derived component `{}`",
                    component.id
                )));
            }
            Ok((component.id.clone(), vec![component.id.clone()]))
        })
        .collect()
}

/// Process-local registry retaining the exact implementations authorized to
/// transform checkpoint schemas. It is deliberately neither serializable nor
/// reconstructible from a capability catalog.
pub struct WeightMaterializerRegistry {
    materializers: BTreeMap<WeightMaterializerId, Arc<dyn WeightMaterializer>>,
}

impl WeightMaterializerRegistry {
    pub fn new(materializers: Vec<Box<dyn WeightMaterializer>>) -> Result<Self, VNextError> {
        if materializers.len() >= MAX_WEIGHT_MATERIALIZERS {
            return Err(invalid_plan(format!(
                "weight materializer registry exceeds {} non-identity entries",
                MAX_WEIGHT_MATERIALIZERS - 1
            )));
        }
        let identity: Arc<dyn WeightMaterializer> = Arc::new(IdentityWeightMaterializer::new()?);
        let mut entries = BTreeMap::from([(identity.descriptor().id().clone(), identity)]);
        for materializer in materializers {
            materializer.descriptor().validate_structure()?;
            let id = materializer.descriptor().id().clone();
            if entries
                .insert(id.clone(), Arc::from(materializer))
                .is_some()
            {
                return Err(invalid_plan(format!(
                    "duplicate weight materializer `{id}`"
                )));
            }
        }
        Ok(Self {
            materializers: entries,
        })
    }

    pub fn identity_only() -> Result<Self, VNextError> {
        Self::new(Vec::new())
    }

    /// Extends a capability catalog with descriptors from these exact
    /// process-local implementation objects.
    pub fn augment_catalog(
        &self,
        catalog: CapabilityCatalog,
    ) -> Result<CapabilityCatalog, VNextError> {
        catalog.with_weight_materializer_descriptors(self.descriptors())
    }

    /// Invokes the selected trusted implementation and returns a witness that
    /// cannot be forged by deserializing an [`ExecutionWeightPlan`].
    /// Selects a materializer that is value-preserving by contract.
    ///
    /// Approximate transforms require a future typed quality approval carrying
    /// checked-in numerical-tolerance evidence. Keeping that authority out of
    /// this entrypoint prevents a build feature or backend capability from
    /// silently changing checkpoint precision.
    pub fn select_exact(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        materializer_id: &WeightMaterializerId,
    ) -> Result<TrustedExecutionWeightPlan, VNextError> {
        let materializer = self.registered_materializer(catalog, materializer_id)?;
        let descriptor = materializer.descriptor();
        if descriptor.fidelity() != WeightMaterializationFidelity::Exact {
            return Err(VNextError::WeightMaterializerQualityApprovalRequired {
                materializer_id: materializer_id.to_string(),
            });
        }
        descriptor.validate_for_device(catalog.device())?;
        let mut schema = materializer.execution_schema(family, catalog.device())?;
        schema.normalize();
        let component_sources = materializer.component_sources(family, &schema)?;
        let static_weight_transforms = materializer.static_weight_transforms(family, &schema)?;
        let plan = ExecutionWeightPlan::from_materializer(
            family,
            descriptor,
            schema,
            component_sources,
            static_weight_transforms,
        )?;
        Ok(TrustedExecutionWeightPlan {
            plan,
            descriptor: descriptor.clone(),
            materializer: Arc::clone(materializer),
        })
    }

    /// Selects either an exact materializer or an approximate materializer
    /// carrying strict, crate-verified numeric artifact bytes.
    pub fn select(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        selection: &WeightMaterializerSelection,
    ) -> Result<TrustedExecutionWeightPlan, VNextError> {
        let Some(artifact_bytes) = selection.numeric_quality_artifact_bytes() else {
            return self.select_exact(family, catalog, selection.materializer_id());
        };
        self.select_with_numeric_quality_artifact(
            family,
            catalog,
            selection.materializer_id(),
            artifact_bytes,
        )
    }

    pub fn select_with_numeric_quality_artifact(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        materializer_id: &WeightMaterializerId,
        artifact_bytes: &[u8],
    ) -> Result<TrustedExecutionWeightPlan, VNextError> {
        let materializer = self.registered_materializer(catalog, materializer_id)?;
        let descriptor = materializer.descriptor();
        if descriptor.fidelity() != WeightMaterializationFidelity::Approximate {
            return Err(invalid_plan(format!(
                "exact weight materializer `{materializer_id}` cannot consume an approximate quality artifact"
            )));
        }
        descriptor.validate_for_device(catalog.device())?;
        let mut schema = materializer.execution_schema(family, catalog.device())?;
        schema.normalize();
        let approval =
            verify_numeric_weight_quality_artifact(artifact_bytes, descriptor, family, &schema)?;
        let component_sources = materializer.component_sources(family, &schema)?;
        let static_weight_transforms = materializer.static_weight_transforms(family, &schema)?;
        let plan = ExecutionWeightPlan::from_materializer_with_approval(
            family,
            descriptor,
            schema,
            component_sources,
            static_weight_transforms,
            Some(approval),
        )?;
        Ok(TrustedExecutionWeightPlan {
            plan,
            descriptor: descriptor.clone(),
            materializer: Arc::clone(materializer),
        })
    }

    fn registered_materializer<'registry>(
        &'registry self,
        catalog: &CapabilityCatalog,
        materializer_id: &WeightMaterializerId,
    ) -> Result<&'registry Arc<dyn WeightMaterializer>, VNextError> {
        let materializer = self.materializers.get(materializer_id).ok_or_else(|| {
            invalid_plan(format!(
                "weight materializer `{materializer_id}` is not registered"
            ))
        })?;
        if materializer.descriptor() != catalog.weight_materializer(materializer_id)? {
            return Err(invalid_plan(format!(
                "weight materializer `{materializer_id}` differs from its capability catalog descriptor"
            )));
        }
        Ok(materializer)
    }

    pub fn descriptors(&self) -> BTreeMap<WeightMaterializerId, WeightMaterializerDescriptor> {
        self.materializers
            .iter()
            .map(|(id, materializer)| (id.clone(), materializer.descriptor().clone()))
            .collect()
    }
}

/// Non-serializable proof that a process-local registry implementation
/// produced and validated this physical execution schema.
#[derive(Clone)]
pub struct TrustedExecutionWeightPlan {
    plan: ExecutionWeightPlan,
    descriptor: WeightMaterializerDescriptor,
    materializer: Arc<dyn WeightMaterializer>,
}

impl fmt::Debug for TrustedExecutionWeightPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedExecutionWeightPlan")
            .field("plan", &self.plan)
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl PartialEq for TrustedExecutionWeightPlan {
    fn eq(&self, other: &Self) -> bool {
        self.plan == other.plan && self.descriptor == other.descriptor
    }
}

impl Eq for TrustedExecutionWeightPlan {}

impl TrustedExecutionWeightPlan {
    pub(crate) fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let materializer: Arc<dyn WeightMaterializer> =
            Arc::new(IdentityWeightMaterializer::new()?);
        let descriptor = materializer.descriptor().clone();
        let schema = family.weight_schema().clone();
        let component_sources = materializer.component_sources(family, &schema)?;
        Ok(Self {
            plan: ExecutionWeightPlan::from_materializer(
                family,
                &descriptor,
                schema,
                component_sources,
                Vec::new(),
            )?,
            descriptor,
            materializer,
        })
    }

    pub fn plan(&self) -> &ExecutionWeightPlan {
        &self.plan
    }

    pub(crate) fn validate_against_catalog(
        &self,
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
    ) -> Result<(), VNextError> {
        self.validate_runtime_authority()?;
        let catalog_descriptor = catalog.weight_materializer(self.descriptor.id())?;
        if &self.descriptor != catalog_descriptor {
            return Err(invalid_plan(format!(
                "weight materializer `{}` differs from its capability catalog descriptor",
                self.descriptor.id()
            )));
        }
        self.plan
            .validate_against_materializer(family, &self.descriptor)?;
        let mut expected_transforms = self
            .materializer
            .static_weight_transforms(family, self.plan.schema())?;
        expected_transforms.sort();
        if expected_transforms != self.plan.static_weight_transforms {
            return Err(invalid_plan(format!(
                "weight materializer `{}` static transform authority differs from the trusted plan",
                self.descriptor.id()
            )));
        }
        Ok(())
    }

    pub(crate) fn materialize_components<'source>(
        &self,
        family: &PreparedModelFamily,
        source: &'source dyn WeightComponentSource,
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.validate_runtime_authority()?;
        let Some(first_execution_component) = execution_components.first() else {
            return Err(invalid_plan(
                "weight materializer received an empty execution component group",
            ));
        };
        let mut planned_components = Vec::with_capacity(execution_components.len());
        for execution_component in execution_components {
            let planned_component_index = self
                .plan
                .schema
                .components
                .binary_search_by(|component| component.id.cmp(&execution_component.id))
                .map_err(|_| {
                    invalid_plan(format!(
                        "execution component `{}` is absent from the trusted weight plan",
                        execution_component.id
                    ))
                })?;
            let planned_component = &self.plan.schema.components[planned_component_index];
            if planned_component != *execution_component {
                return Err(invalid_plan(format!(
                    "execution component `{}` differs from the trusted weight plan",
                    execution_component.id
                )));
            }
            planned_components.push(planned_component);
        }
        let source_ids = self
            .plan
            .component_sources
            .get(&first_execution_component.id)
            .ok_or_else(|| {
                invalid_plan(format!(
                    "execution component `{}` has no source mapping",
                    first_execution_component.id
                ))
            })?;
        if execution_components
            .iter()
            .skip(1)
            .any(|component| self.plan.component_sources.get(&component.id) != Some(source_ids))
        {
            return Err(invalid_plan(
                "grouped execution components do not share one ordered source mapping",
            ));
        }
        let source_components = source_ids
            .iter()
            .map(|source_id| {
                let source_component_index = family
                    .weight_schema()
                    .components
                    .binary_search_by(|component| component.id.cmp(source_id))
                    .map_err(|_| {
                        invalid_plan(format!(
                            "execution component `{}` references unknown source component `{source_id}`",
                            first_execution_component.id
                        ))
                    })?;
                Ok(&family.weight_schema().components[source_component_index])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let payloads = self.materializer.materialize_components(
            source,
            &source_components,
            &planned_components,
        )?;
        if payloads.len() != planned_components.len() {
            return Err(invalid_plan(format!(
                "weight materializer `{}` returned {} payloads for {} execution components",
                self.descriptor.id,
                payloads.len(),
                planned_components.len()
            )));
        }
        for (payload, execution_component) in payloads.iter().zip(&planned_components) {
            if payload.component_id() != &execution_component.id
                || payload.external_names() != execution_component.external_names.as_slice()
                || payload.dimensions() != execution_component.dimensions.as_slice()
                || payload.element_type() != execution_component.physical_element_type()
                || u64::try_from(payload.bytes().len()).ok()
                    != Some(execution_component.physical_bytes()?)
            {
                return Err(invalid_plan(format!(
                    "weight materializer `{}` returned invalid or reordered payload for execution component `{}`",
                    self.descriptor.id, execution_component.id
                )));
            }
        }
        Ok(payloads)
    }

    pub(crate) fn static_weight_transform_for_components(
        &self,
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Option<&StaticWeightTransformPlan>, VNextError> {
        self.validate_runtime_authority()?;
        if execution_components.is_empty() {
            return Err(invalid_plan(
                "static weight transform lookup received an empty component group",
            ));
        }
        let requested = execution_components
            .iter()
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        let matching = self
            .plan
            .static_weight_transforms
            .iter()
            .filter(|transform| {
                transform
                    .execution_component_ids()
                    .into_iter()
                    .any(|component_id| requested.contains(component_id))
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Ok(None);
        }
        let [transform] = matching.as_slice() else {
            return Err(invalid_plan(
                "execution component group spans multiple static weight transforms",
            ));
        };
        let expected = transform
            .execution_component_ids()
            .into_iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if requested != expected {
            return Err(invalid_plan(
                "execution component group contains a partial or mixed static weight transform",
            ));
        }
        Ok(Some(transform))
    }

    fn validate_runtime_authority(&self) -> Result<(), VNextError> {
        if self.materializer.descriptor() != &self.descriptor {
            return Err(invalid_plan(format!(
                "weight materializer `{}` runtime authority differs from its trusted descriptor",
                self.descriptor.id
            )));
        }
        Ok(())
    }
}

/// Trusted physical weight contract selected for one immutable execution plan.
///
/// The prepared family remains the source/checkpoint contract. This plan owns
/// the physical schema consumed by providers, static memory planning, and
/// initialization. Keeping those contracts separate allows a backend to
/// select a prepared layout without changing model semantics or hiding an
/// allocation behind a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionWeightPlan {
    source_schema_fingerprint: String,
    materializer_id: WeightMaterializerId,
    materializer_version: ContractVersion,
    materializer_implementation_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    approximate_quality_approval: Option<ApproximateWeightQualityApprovalRecord>,
    component_sources: BTreeMap<WeightId, Vec<WeightId>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    static_weight_transforms: Vec<StaticWeightTransformPlan>,
    schema: WeightSchema,
}

impl ExecutionWeightPlan {
    pub fn identity(family: &PreparedModelFamily) -> Result<Self, VNextError> {
        let descriptor = WeightMaterializerDescriptor::identity()?;
        let schema = family.weight_schema().clone();
        let component_sources = identity_component_sources(family, &schema)?;
        Self::from_materializer(family, &descriptor, schema, component_sources, Vec::new())
    }

    fn from_materializer(
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
        schema: WeightSchema,
        component_sources: BTreeMap<WeightId, Vec<WeightId>>,
        static_weight_transforms: Vec<StaticWeightTransformPlan>,
    ) -> Result<Self, VNextError> {
        Self::from_materializer_with_approval(
            family,
            descriptor,
            schema,
            component_sources,
            static_weight_transforms,
            None,
        )
    }

    fn from_materializer_with_approval(
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
        schema: WeightSchema,
        component_sources: BTreeMap<WeightId, Vec<WeightId>>,
        mut static_weight_transforms: Vec<StaticWeightTransformPlan>,
        approximate_quality_approval: Option<ApproximateWeightQualityApprovalRecord>,
    ) -> Result<Self, VNextError> {
        static_weight_transforms.sort();
        let plan = Self {
            source_schema_fingerprint: family.weight_schema().fingerprint()?,
            materializer_id: descriptor.id().clone(),
            materializer_version: descriptor.version(),
            materializer_implementation_fingerprint: descriptor
                .implementation_fingerprint()
                .to_owned(),
            approximate_quality_approval,
            component_sources,
            static_weight_transforms,
            schema,
        };
        plan.validate_against_materializer(family, descriptor)?;
        Ok(plan)
    }

    pub fn source_schema_fingerprint(&self) -> &str {
        &self.source_schema_fingerprint
    }

    pub fn materializer_id(&self) -> &WeightMaterializerId {
        &self.materializer_id
    }

    pub const fn materializer_version(&self) -> ContractVersion {
        self.materializer_version
    }

    pub fn materializer_implementation_fingerprint(&self) -> &str {
        &self.materializer_implementation_fingerprint
    }

    pub fn artifact_abi(&self) -> Result<WeightArtifactAbi, VNextError> {
        WeightArtifactAbi::from_schema(&self.schema)
    }

    pub fn approximate_quality_approval(&self) -> Option<&ApproximateWeightQualityApprovalRecord> {
        self.approximate_quality_approval.as_ref()
    }

    pub fn schema(&self) -> &WeightSchema {
        &self.schema
    }

    pub fn component_sources(&self) -> &BTreeMap<WeightId, Vec<WeightId>> {
        &self.component_sources
    }

    pub fn static_weight_transforms(&self) -> &[StaticWeightTransformPlan] {
        &self.static_weight_transforms
    }

    pub fn maximum_static_weight_transform_scratch_bytes(&self) -> Result<u64, VNextError> {
        self.static_weight_transforms
            .iter()
            .map(StaticWeightTransformPlan::scratch_bytes)
            .try_fold(0_u64, |maximum, bytes| {
                bytes.map(|bytes| maximum.max(bytes))
            })
    }

    pub fn static_weight_transform_scratch_resource_id(
        &self,
    ) -> Result<Option<ResourceId>, VNextError> {
        if self.static_weight_transforms.is_empty() {
            return Ok(None);
        }
        let artifact_abi = self.artifact_abi()?;
        let digest = canonical_fingerprint(
            &(&artifact_abi, &self.static_weight_transforms),
            "fingerprint static weight transform scratch identity",
        )?;
        ResourceId::new(format!(
            "resource/static-weight-transform-scratch/sha256/{digest}"
        ))
        .map(Some)
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(self, "fingerprint execution weight plan")
    }

    pub(super) fn validate_structure(&self, family_id: &ModelFamilyId) -> Result<(), VNextError> {
        if !is_canonical_sha256(&self.source_schema_fingerprint)
            || !is_canonical_sha256(&self.materializer_implementation_fingerprint)
            || self.materializer_version.major == 0
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "execution weight plan provenance is invalid".to_owned(),
            });
        }
        if let Some(approval) = &self.approximate_quality_approval {
            approval.validate_structure()?;
        }
        self.schema.validate(family_id)?;
        self.artifact_abi()?.validate()?;
        let execution_component_ids = self
            .schema
            .components
            .iter()
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        let mapped_component_ids = self
            .component_sources
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if execution_component_ids != mapped_component_ids
            || self.component_sources.values().any(|source_ids| {
                source_ids.is_empty()
                    || source_ids.iter().collect::<BTreeSet<_>>().len() != source_ids.len()
            })
        {
            return Err(invalid_plan(
                "execution weight component source map is incomplete or contains duplicate sources",
            ));
        }
        if self
            .static_weight_transforms
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_plan(
                "static weight transforms are duplicate or non-canonical",
            ));
        }
        let mut transformed_components = BTreeSet::new();
        for transform in &self.static_weight_transforms {
            transform.validate()?;
            let source_ids = transform
                .source_component_ids()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>();
            for execution_id in transform.execution_component_ids() {
                if !execution_component_ids.contains(execution_id)
                    || self.component_sources.get(execution_id) != Some(&source_ids)
                    || !transformed_components.insert(execution_id.clone())
                {
                    return Err(invalid_plan(
                        "static weight transform outputs differ from the execution schema source map",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_against_family(
        &self,
        family: &PreparedModelFamily,
    ) -> Result<(), VNextError> {
        self.validate_structure(family.family_id())?;
        if self.source_schema_fingerprint != family.weight_schema().fingerprint()? {
            return Err(invalid_plan(
                "execution weight plan source schema differs from its prepared family",
            ));
        }
        let source_components = family
            .weight_schema()
            .components
            .iter()
            .map(|component| (&component.id, component))
            .collect::<BTreeMap<_, _>>();
        let mut referenced_source_components = BTreeSet::new();
        for (execution_component_id, source_ids) in &self.component_sources {
            for source_id in source_ids {
                if !source_components.contains_key(source_id) {
                    return Err(invalid_plan(format!(
                        "execution component `{execution_component_id}` references unknown source component `{source_id}`"
                    )));
                }
                referenced_source_components.insert(source_id.clone());
            }
        }
        if let Some(component) = source_components.values().find(|component| {
            component.required && !referenced_source_components.contains(&component.id)
        }) {
            return Err(invalid_plan(format!(
                "required source component `{}` is not represented in the execution weight plan",
                component.id
            )));
        }
        let source_tensors = family
            .weight_schema()
            .tensors
            .iter()
            .map(|tensor| (&tensor.id, tensor))
            .collect::<BTreeMap<_, _>>();
        let execution_tensors = self
            .schema
            .tensors
            .iter()
            .map(|tensor| (&tensor.id, tensor))
            .collect::<BTreeMap<_, _>>();
        if source_tensors.len() != execution_tensors.len()
            || source_tensors.iter().any(|(id, source)| {
                execution_tensors.get(id).is_none_or(|execution| {
                    source.dimensions != execution.dimensions
                        || source.logical_element_type != execution.logical_element_type
                        || source.required != execution.required
                })
            })
        {
            return Err(invalid_plan(
                "execution weight schema changes the prepared family's logical tensor contract",
            ));
        }
        Ok(())
    }

    fn validate_against_materializer(
        &self,
        family: &PreparedModelFamily,
        descriptor: &WeightMaterializerDescriptor,
    ) -> Result<(), VNextError> {
        self.validate_against_family(family)?;
        if &self.materializer_id != descriptor.id()
            || self.materializer_version != descriptor.version()
            || self.materializer_implementation_fingerprint
                != descriptor.implementation_fingerprint()
        {
            return Err(invalid_plan(
                "execution weight plan differs from its trusted materializer descriptor",
            ));
        }
        match (
            descriptor.fidelity(),
            descriptor.approximate_quality_contract(),
            &self.approximate_quality_approval,
        ) {
            (WeightMaterializationFidelity::Exact, None, None) => {}
            (
                WeightMaterializationFidelity::Approximate,
                Some(quality_contract),
                Some(approval),
            ) => approval.validate_against(
                &self.source_schema_fingerprint,
                &self.schema.fingerprint()?,
                quality_contract,
            )?,
            _ => {
                return Err(invalid_plan(
                    "execution weight plan fidelity differs from its numerical quality approval",
                ));
            }
        }
        Ok(())
    }
}
