use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{
    AttributeId, CapabilityCatalog, ContractVersion, DimensionConstraint, ElementType,
    OperationContract, OperationDescriptor, OperationId, OracleSpec, SemanticValue, VNextError,
    MAX_REFERENCE_ORACLE_DEPTH,
};

/// Maximum rank of one canonical host tensor passed through an oracle.
pub const MAX_ORACLE_TENSOR_RANK: usize = 16;
/// Maximum logical elements in one canonical host tensor.
pub const MAX_ORACLE_TENSOR_ELEMENTS: usize = 16 * 1024 * 1024;
/// Maximum encoded bytes in one canonical host tensor.
pub const MAX_ORACLE_TENSOR_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of input or output tensors in one oracle call.
pub const MAX_ORACLE_TENSORS: usize = 64;
/// Maximum cumulative tensor bytes in an oracle request or result.
pub const MAX_ORACLE_CALL_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of typed attributes in an oracle request.
pub const MAX_ORACLE_ATTRIBUTES: usize = 256;
/// Maximum canonical JSON bytes occupied by oracle request attributes.
pub const MAX_ORACLE_ATTRIBUTE_BYTES: usize = 1024 * 1024;
/// Maximum JSON wire bytes accepted before decoding any unvalidated oracle type.
pub const MAX_ORACLE_WIRE_BYTES: usize = 16 * 1024 * 1024;

fn invalid_oracle(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

/// Validates the raw availability boundary shared by all oracle wire decoders.
pub fn validate_oracle_wire_byte_length(byte_length: usize) -> Result<(), VNextError> {
    if byte_length > MAX_ORACLE_WIRE_BYTES {
        return Err(invalid_oracle(format!(
            "oracle wire bytes exceed {MAX_ORACLE_WIRE_BYTES}"
        )));
    }
    Ok(())
}

fn decode_untrusted_oracle_wire<T: DeserializeOwned>(
    bytes: &[u8],
    context: &'static str,
) -> Result<T, VNextError> {
    validate_oracle_wire_byte_length(bytes.len())?;
    serde_json::from_slice(bytes).map_err(|error| VNextError::Serialization {
        context,
        message: error.to_string(),
    })
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_oracle_identity(value: &str) -> Result<(), VNextError> {
    if value.is_empty() || value.len() > 160 {
        return Err(VNextError::InvalidIdentity {
            kind: "operation oracle",
            value: value.to_owned(),
            reason: "identity must contain between 1 and 160 bytes",
        });
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
    }) {
        return Err(VNextError::InvalidIdentity {
            kind: "operation oracle",
            value: value.to_owned(),
            reason: "identity contains a non-portable character",
        });
    }
    Ok(())
}

/// Stable identity of one checked-in oracle implementation contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct OperationOracleId(String);

impl OperationOracleId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        validate_oracle_identity(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for OperationOracleId {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<OperationOracleId> for String {
    fn from(value: OperationOracleId) -> Self {
        value.0
    }
}

impl fmt::Display for OperationOracleId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Trusted, Serialize-only identity of an executable operation oracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationOracleDescriptor {
    oracle_id: OperationOracleId,
    version: ContractVersion,
    implementation_fingerprint: String,
    operation_id: OperationId,
    operation_fingerprint: String,
}

/// Untrusted wire shape for an oracle descriptor. It is never executable until
/// reconstructed and matched to a registry-owned trait object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOperationOracleDescriptor {
    pub oracle_id: OperationOracleId,
    pub version: ContractVersion,
    pub implementation_fingerprint: String,
    pub operation_id: OperationId,
    pub operation_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationOracleDescriptorWire {
    oracle_id: OperationOracleId,
    version: ContractVersion,
    implementation_fingerprint: String,
    operation_id: OperationId,
    operation_fingerprint: String,
}

impl UnvalidatedOperationOracleDescriptor {
    pub fn revalidate(self) -> Result<OperationOracleDescriptor, VNextError> {
        OperationOracleDescriptor::new(
            self.oracle_id,
            self.version,
            self.implementation_fingerprint,
            self.operation_id,
            self.operation_fingerprint,
        )
    }
}

impl OperationOracleDescriptor {
    pub fn new(
        oracle_id: OperationOracleId,
        version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        operation_id: OperationId,
        operation_fingerprint: impl Into<String>,
    ) -> Result<Self, VNextError> {
        let implementation_fingerprint = implementation_fingerprint.into();
        let operation_fingerprint = operation_fingerprint.into();
        if version.major == 0 {
            return Err(invalid_oracle(format!(
                "oracle `{oracle_id}` has a zero contract major version"
            )));
        }
        if !is_canonical_sha256(&implementation_fingerprint) {
            return Err(invalid_oracle(format!(
                "oracle `{oracle_id}` implementation fingerprint is not canonical SHA-256"
            )));
        }
        if !is_canonical_sha256(&operation_fingerprint) {
            return Err(invalid_oracle(format!(
                "oracle `{oracle_id}` operation fingerprint is not canonical SHA-256"
            )));
        }
        Ok(Self {
            oracle_id,
            version,
            implementation_fingerprint,
            operation_id,
            operation_fingerprint,
        })
    }

    pub fn oracle_id(&self) -> &OperationOracleId {
        &self.oracle_id
    }

    pub const fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn implementation_fingerprint(&self) -> &str {
        &self.implementation_fingerprint
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize operation oracle descriptor",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn decode_untrusted(
        bytes: &[u8],
    ) -> Result<UnvalidatedOperationOracleDescriptor, VNextError> {
        let wire: OperationOracleDescriptorWire =
            decode_untrusted_oracle_wire(bytes, "decode untrusted operation oracle descriptor")?;
        Ok(UnvalidatedOperationOracleDescriptor {
            oracle_id: wire.oracle_id,
            version: wire.version,
            implementation_fingerprint: wire.implementation_fingerprint,
            operation_id: wire.operation_id,
            operation_fingerprint: wire.operation_fingerprint,
        })
    }
}

/// Canonical row-major, little-endian host tensor used by all oracles. Float
/// encodings must be finite; exact comparison is bit-exact, including distinct
/// positive and negative zero encodings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OracleTensor {
    dimensions: Vec<u64>,
    element_type: ElementType,
    bytes: Vec<u8>,
}

/// Untrusted wire tensor. `revalidate` checks all rank, extent, byte and scalar
/// encoding bounds before producing a trusted host tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOracleTensor {
    pub dimensions: Vec<u64>,
    pub element_type: ElementType,
    pub bytes: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleTensorWire {
    dimensions: Vec<u64>,
    element_type: ElementType,
    bytes: Vec<u8>,
}

impl UnvalidatedOracleTensor {
    pub fn revalidate(self) -> Result<OracleTensor, VNextError> {
        OracleTensor::new(self.dimensions, self.element_type, self.bytes)
    }
}

impl OracleTensor {
    pub fn new(
        dimensions: Vec<u64>,
        element_type: ElementType,
        bytes: Vec<u8>,
    ) -> Result<Self, VNextError> {
        if dimensions.len() > MAX_ORACLE_TENSOR_RANK {
            return Err(invalid_oracle(format!(
                "oracle tensor rank exceeds {MAX_ORACLE_TENSOR_RANK}"
            )));
        }
        if dimensions.iter().any(|extent| *extent == 0) {
            return Err(invalid_oracle("oracle tensor has a zero extent"));
        }
        let elements = dimensions.iter().try_fold(1usize, |elements, extent| {
            let extent = usize::try_from(*extent)
                .map_err(|_| invalid_oracle("oracle tensor extent does not fit usize"))?;
            elements
                .checked_mul(extent)
                .ok_or_else(|| invalid_oracle("oracle tensor element count overflows usize"))
        })?;
        if elements > MAX_ORACLE_TENSOR_ELEMENTS {
            return Err(invalid_oracle(format!(
                "oracle tensor elements exceed {MAX_ORACLE_TENSOR_ELEMENTS}"
            )));
        }
        let element_bytes = usize::try_from(element_type.size_bytes())
            .map_err(|_| invalid_oracle("oracle tensor element width does not fit usize"))?;
        let expected_bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid_oracle("oracle tensor byte count overflows usize"))?;
        if expected_bytes > MAX_ORACLE_TENSOR_BYTES || bytes.len() != expected_bytes {
            return Err(invalid_oracle(format!(
                "oracle tensor requires exactly {expected_bytes} bytes within the {MAX_ORACLE_TENSOR_BYTES} byte limit"
            )));
        }
        validate_scalar_encodings(element_type, &bytes)?;
        Ok(Self {
            dimensions,
            element_type,
            bytes,
        })
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn element_count(&self) -> usize {
        self.bytes.len() / self.element_type.size_bytes() as usize
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedOracleTensor, VNextError> {
        let wire: OracleTensorWire =
            decode_untrusted_oracle_wire(bytes, "decode untrusted oracle tensor")?;
        Ok(UnvalidatedOracleTensor {
            dimensions: wire.dimensions,
            element_type: wire.element_type,
            bytes: wire.bytes,
        })
    }

    fn numeric_value(&self, index: usize) -> Result<f64, VNextError> {
        if index >= self.element_count() {
            return Err(invalid_oracle(
                "oracle tensor element index is out of bounds",
            ));
        }
        let width = self.element_type.size_bytes() as usize;
        let offset = index * width;
        let bytes = &self.bytes[offset..offset + width];
        let value = match self.element_type {
            ElementType::Bool => {
                return Err(invalid_oracle(
                    "tolerance comparison is not defined for boolean tensors",
                ));
            }
            ElementType::U8 => f64::from(bytes[0]),
            ElementType::I8 => f64::from(i8::from_le_bytes([bytes[0]])),
            ElementType::U32 => f64::from(u32::from_le_bytes(
                bytes.try_into().expect("validated width"),
            )),
            ElementType::I32 => f64::from(i32::from_le_bytes(
                bytes.try_into().expect("validated width"),
            )),
            ElementType::F16 => f16_to_f64(u16::from_le_bytes(
                bytes.try_into().expect("validated width"),
            )),
            ElementType::Bf16 => f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes(
                    bytes.try_into().expect("validated width"),
                )) << 16,
            )),
            ElementType::F32 => f64::from(f32::from_le_bytes(
                bytes.try_into().expect("validated width"),
            )),
        };
        Ok(value)
    }
}

fn validate_scalar_encodings(element_type: ElementType, bytes: &[u8]) -> Result<(), VNextError> {
    match element_type {
        ElementType::Bool if bytes.iter().any(|value| !matches!(value, 0 | 1)) => Err(
            invalid_oracle("canonical boolean oracle tensors contain only 0 or 1"),
        ),
        ElementType::F16
            if bytes.chunks_exact(2).any(|bytes| {
                u16::from_le_bytes(bytes.try_into().expect("two-byte chunk")) & 0x7c00 == 0x7c00
            }) =>
        {
            Err(invalid_oracle(
                "oracle tensors reject non-finite f16 values",
            ))
        }
        ElementType::Bf16
            if bytes.chunks_exact(2).any(|bytes| {
                u16::from_le_bytes(bytes.try_into().expect("two-byte chunk")) & 0x7f80 == 0x7f80
            }) =>
        {
            Err(invalid_oracle(
                "oracle tensors reject non-finite bf16 values",
            ))
        }
        ElementType::F32
            if bytes.chunks_exact(4).any(|bytes| {
                !f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")).is_finite()
            }) =>
        {
            Err(invalid_oracle(
                "oracle tensors reject non-finite f32 values",
            ))
        }
        _ => Ok(()),
    }
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x03ff) / 1024.0;
    if exponent == 0 {
        sign * fraction * 2.0_f64.powi(-14)
    } else {
        sign * (1.0 + fraction) * 2.0_f64.powi(exponent - 15)
    }
}

/// Canonical, bounded request delivered to an `OperationOracle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationOracleRequest {
    operation_id: OperationId,
    operation_fingerprint: String,
    inputs: Vec<OracleTensor>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOperationOracleRequest {
    pub operation_id: OperationId,
    pub operation_fingerprint: String,
    pub inputs: Vec<UnvalidatedOracleTensor>,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationOracleRequestWire {
    operation_id: OperationId,
    operation_fingerprint: String,
    inputs: Vec<OracleTensorWire>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
}

impl UnvalidatedOperationOracleRequest {
    pub fn revalidate(self) -> Result<OperationOracleRequest, VNextError> {
        OperationOracleRequest::new(
            self.operation_id,
            self.operation_fingerprint,
            self.inputs
                .into_iter()
                .map(UnvalidatedOracleTensor::revalidate)
                .collect::<Result<Vec<_>, _>>()?,
            self.attributes,
        )
    }
}

impl OperationOracleRequest {
    pub fn new(
        operation_id: OperationId,
        operation_fingerprint: impl Into<String>,
        inputs: Vec<OracleTensor>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, VNextError> {
        let operation_fingerprint = operation_fingerprint.into();
        if !is_canonical_sha256(&operation_fingerprint) {
            return Err(invalid_oracle(
                "oracle request operation fingerprint is not canonical SHA-256",
            ));
        }
        validate_tensor_collection("oracle request inputs", &inputs, false)?;
        validate_oracle_attributes(&attributes)?;
        Ok(Self {
            operation_id,
            operation_fingerprint,
            inputs,
            attributes,
        })
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn inputs(&self) -> &[OracleTensor] {
        &self.inputs
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        &self.attributes
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedOperationOracleRequest, VNextError> {
        let wire: OperationOracleRequestWire =
            decode_untrusted_oracle_wire(bytes, "decode untrusted operation oracle request")?;
        Ok(UnvalidatedOperationOracleRequest {
            operation_id: wire.operation_id,
            operation_fingerprint: wire.operation_fingerprint,
            inputs: wire
                .inputs
                .into_iter()
                .map(|tensor| UnvalidatedOracleTensor {
                    dimensions: tensor.dimensions,
                    element_type: tensor.element_type,
                    bytes: tensor.bytes,
                })
                .collect(),
            attributes: wire.attributes,
        })
    }
}

/// Canonical, bounded outputs produced by an oracle implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationOracleResult {
    outputs: Vec<OracleTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOperationOracleResult {
    pub outputs: Vec<UnvalidatedOracleTensor>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationOracleResultWire {
    outputs: Vec<OracleTensorWire>,
}

impl UnvalidatedOperationOracleResult {
    pub fn revalidate(self) -> Result<OperationOracleResult, VNextError> {
        OperationOracleResult::new(
            self.outputs
                .into_iter()
                .map(UnvalidatedOracleTensor::revalidate)
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

impl OperationOracleResult {
    pub fn new(outputs: Vec<OracleTensor>) -> Result<Self, VNextError> {
        validate_tensor_collection("oracle result outputs", &outputs, true)?;
        Ok(Self { outputs })
    }

    pub fn outputs(&self) -> &[OracleTensor] {
        &self.outputs
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedOperationOracleResult, VNextError> {
        let wire: OperationOracleResultWire =
            decode_untrusted_oracle_wire(bytes, "decode untrusted operation oracle result")?;
        Ok(UnvalidatedOperationOracleResult {
            outputs: wire
                .outputs
                .into_iter()
                .map(|tensor| UnvalidatedOracleTensor {
                    dimensions: tensor.dimensions,
                    element_type: tensor.element_type,
                    bytes: tensor.bytes,
                })
                .collect(),
        })
    }
}

fn validate_tensor_collection(
    context: &str,
    tensors: &[OracleTensor],
    require_nonempty: bool,
) -> Result<(), VNextError> {
    if tensors.len() > MAX_ORACLE_TENSORS || (require_nonempty && tensors.is_empty()) {
        return Err(invalid_oracle(format!(
            "{context} must contain {} to {MAX_ORACLE_TENSORS} tensors",
            usize::from(require_nonempty)
        )));
    }
    let total_bytes = tensors.iter().try_fold(0usize, |total, tensor| {
        total
            .checked_add(tensor.bytes.len())
            .ok_or_else(|| invalid_oracle(format!("{context} byte count overflows usize")))
    })?;
    if total_bytes > MAX_ORACLE_CALL_BYTES {
        return Err(invalid_oracle(format!(
            "{context} exceeds {MAX_ORACLE_CALL_BYTES} cumulative bytes"
        )));
    }
    Ok(())
}

fn validate_oracle_attributes(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<(), VNextError> {
    if attributes.len() > MAX_ORACLE_ATTRIBUTES {
        return Err(invalid_oracle(format!(
            "oracle request exceeds {MAX_ORACLE_ATTRIBUTES} attributes"
        )));
    }
    for value in attributes.values() {
        value.validate("oracle request attributes")?;
    }
    let bytes = serde_json::to_vec(attributes).map_err(|error| VNextError::Serialization {
        context: "serialize operation oracle attributes",
        message: error.to_string(),
    })?;
    if bytes.len() > MAX_ORACLE_ATTRIBUTE_BYTES {
        return Err(invalid_oracle(format!(
            "oracle request attributes exceed {MAX_ORACLE_ATTRIBUTE_BYTES} canonical bytes"
        )));
    }
    Ok(())
}

/// Object-safe executable correctness oracle. Runtime callers receive a
/// registry-bound handle and never supply an oracle implementation per call.
pub trait OperationOracle: Send + Sync {
    fn descriptor(&self) -> &OperationOracleDescriptor;

    fn invoke(&self, request: &OperationOracleRequest)
        -> Result<OperationOracleResult, VNextError>;
}

/// Composition-root registration that independently anchors the expected
/// descriptor before an implementation can enter the trusted registry.
pub struct OperationOracleRegistration {
    expected_descriptor: OperationOracleDescriptor,
    oracle: Box<dyn OperationOracle>,
}

impl OperationOracleRegistration {
    pub fn new(
        expected_descriptor: OperationOracleDescriptor,
        oracle: Box<dyn OperationOracle>,
    ) -> Result<Self, VNextError> {
        if oracle.descriptor() != &expected_descriptor {
            return Err(invalid_oracle(
                "operation oracle implementation differs from its trusted registration descriptor",
            ));
        }
        Ok(Self {
            expected_descriptor,
            oracle,
        })
    }

    pub fn descriptor(&self) -> &OperationOracleDescriptor {
        &self.expected_descriptor
    }
}

struct RegisteredOracle {
    descriptor: OperationOracleDescriptor,
    oracle: Box<dyn OperationOracle>,
}

/// Trusted concrete registry that owns contracts and exact oracle objects for
/// the lifetime of every bound invocation handle.
pub struct OperationOracleRegistry {
    catalog_fingerprint: String,
    operations: BTreeMap<OperationId, OperationDescriptor>,
    contracts: BTreeMap<OperationId, Box<dyn OperationContract>>,
    terminal_operations: BTreeMap<OperationId, OperationId>,
    oracles: BTreeMap<OperationId, RegisteredOracle>,
}

impl OperationOracleRegistry {
    pub fn new(
        catalog: &CapabilityCatalog,
        contracts: Vec<Box<dyn OperationContract>>,
        registrations: Vec<OperationOracleRegistration>,
    ) -> Result<Self, VNextError> {
        let operations = catalog.operations().clone();
        let mut contract_map = BTreeMap::new();
        for contract in contracts {
            let descriptor = contract.descriptor();
            descriptor.validate()?;
            let catalog_descriptor = operations.get(&descriptor.id).ok_or_else(|| {
                invalid_oracle(format!(
                    "oracle registry contract `{}` is absent from the capability catalog",
                    descriptor.id
                ))
            })?;
            if descriptor != catalog_descriptor
                || descriptor.fingerprint()? != catalog_descriptor.fingerprint()?
            {
                return Err(invalid_oracle(format!(
                    "oracle registry contract `{}` differs from the capability catalog",
                    descriptor.id
                )));
            }
            contract.validate_signature(&catalog_descriptor.inputs, &catalog_descriptor.outputs)?;
            let operation_id = descriptor.id.clone();
            if contract_map
                .insert(operation_id.clone(), contract)
                .is_some()
            {
                return Err(invalid_oracle(format!(
                    "oracle registry has duplicate contract `{operation_id}`"
                )));
            }
        }
        if contract_map.keys().collect::<BTreeSet<_>>()
            != operations.keys().collect::<BTreeSet<_>>()
        {
            return Err(invalid_oracle(
                "oracle registry requires exactly one contract for every catalog operation",
            ));
        }

        let terminal_operations = resolve_terminal_operations(&operations)?;
        let terminal_ids = terminal_operations
            .values()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut oracle_ids = BTreeSet::new();
        let mut oracle_map = BTreeMap::new();
        for registration in registrations {
            let OperationOracleRegistration {
                expected_descriptor,
                oracle,
            } = registration;
            if oracle.descriptor() != &expected_descriptor {
                return Err(invalid_oracle(
                    "operation oracle descriptor changed after trusted registration",
                ));
            }
            let operation = operations
                .get(expected_descriptor.operation_id())
                .ok_or_else(|| {
                    invalid_oracle(format!(
                        "oracle `{}` targets an operation absent from the capability catalog",
                        expected_descriptor.oracle_id()
                    ))
                })?;
            if matches!(operation.oracle, OracleSpec::ReferenceOperation { .. }) {
                return Err(invalid_oracle(format!(
                    "reference operation `{}` cannot register a direct oracle",
                    operation.id
                )));
            }
            if expected_descriptor.operation_fingerprint() != operation.fingerprint()? {
                return Err(invalid_oracle(format!(
                    "oracle `{}` operation fingerprint differs from `{}`",
                    expected_descriptor.oracle_id(),
                    operation.id
                )));
            }
            if !oracle_ids.insert(expected_descriptor.oracle_id().clone()) {
                return Err(invalid_oracle(format!(
                    "oracle registry has duplicate identity `{}`",
                    expected_descriptor.oracle_id()
                )));
            }
            let operation_id = operation.id.clone();
            if oracle_map
                .insert(
                    operation_id.clone(),
                    RegisteredOracle {
                        descriptor: expected_descriptor,
                        oracle,
                    },
                )
                .is_some()
            {
                return Err(invalid_oracle(format!(
                    "terminal operation `{operation_id}` has multiple oracles"
                )));
            }
        }
        if oracle_map.keys().cloned().collect::<BTreeSet<_>>() != terminal_ids {
            return Err(invalid_oracle(
                "every terminal non-reference operation must have exactly one oracle",
            ));
        }

        Ok(Self {
            catalog_fingerprint: catalog.fingerprint()?,
            operations,
            contracts: contract_map,
            terminal_operations,
            oracles: oracle_map,
        })
    }

    pub fn catalog_fingerprint(&self) -> &str {
        &self.catalog_fingerprint
    }

    pub fn contract(
        &self,
        operation_id: &OperationId,
    ) -> Result<&dyn OperationContract, VNextError> {
        self.contracts
            .get(operation_id)
            .map(Box::as_ref)
            .ok_or_else(|| invalid_oracle(format!("operation `{operation_id}` is not registered")))
    }

    pub fn bind<'registry>(
        &'registry self,
        operation_id: &OperationId,
    ) -> Result<BoundOperationOracle<'registry>, VNextError> {
        let requested_operation = self.operations.get(operation_id).ok_or_else(|| {
            invalid_oracle(format!("operation `{operation_id}` is not registered"))
        })?;
        let terminal_id = self.terminal_operations.get(operation_id).ok_or_else(|| {
            invalid_oracle(format!(
                "operation `{operation_id}` has no validated terminal oracle"
            ))
        })?;
        let terminal_operation = self
            .operations
            .get(terminal_id)
            .ok_or_else(|| invalid_oracle("validated terminal operation disappeared"))?;
        let registered = self
            .oracles
            .get(terminal_id)
            .ok_or_else(|| invalid_oracle("validated terminal oracle disappeared"))?;
        if registered.oracle.descriptor() != &registered.descriptor {
            return Err(invalid_oracle(
                "registered oracle descriptor changed before binding",
            ));
        }
        Ok(BoundOperationOracle {
            requested_operation,
            terminal_operation,
            registered,
        })
    }
}

fn resolve_terminal_operations(
    operations: &BTreeMap<OperationId, OperationDescriptor>,
) -> Result<BTreeMap<OperationId, OperationId>, VNextError> {
    let mut resolved = BTreeMap::new();
    for (root_id, root) in operations {
        let mut current = root;
        let mut visited = BTreeSet::new();
        for _ in 0..MAX_REFERENCE_ORACLE_DEPTH {
            if !visited.insert(current.id.clone()) {
                return Err(invalid_oracle(format!(
                    "reference oracle chain from `{root_id}` contains a cycle"
                )));
            }
            let OracleSpec::ReferenceOperation {
                operation_id,
                version,
            } = &current.oracle
            else {
                resolved.insert(root_id.clone(), current.id.clone());
                break;
            };
            let reference = operations.get(operation_id).ok_or_else(|| {
                invalid_oracle(format!(
                    "reference oracle `{operation_id}` for `{}` is missing",
                    current.id
                ))
            })?;
            if !reference.version.satisfies(*version)
                || current.inputs != reference.inputs
                || current.outputs != reference.outputs
                || current.attributes != reference.attributes
            {
                return Err(invalid_oracle(format!(
                    "reference oracle `{operation_id}` is incompatible with `{}`",
                    current.id
                )));
            }
            current = reference;
        }
        if !resolved.contains_key(root_id) {
            return Err(invalid_oracle(format!(
                "reference oracle chain from `{root_id}` exceeds depth {MAX_REFERENCE_ORACLE_DEPTH}"
            )));
        }
    }
    Ok(resolved)
}

/// Non-cloneable authority borrowed from one registry-owned oracle object.
pub struct BoundOperationOracle<'registry> {
    requested_operation: &'registry OperationDescriptor,
    terminal_operation: &'registry OperationDescriptor,
    registered: &'registry RegisteredOracle,
}

impl BoundOperationOracle<'_> {
    pub fn requested_operation_id(&self) -> &OperationId {
        &self.requested_operation.id
    }

    pub fn terminal_operation_id(&self) -> &OperationId {
        &self.terminal_operation.id
    }

    pub fn descriptor(&self) -> &OperationOracleDescriptor {
        &self.registered.descriptor
    }

    pub fn comparison_policy(&self) -> &OracleSpec {
        &self.terminal_operation.oracle
    }

    pub fn invoke(
        &self,
        inputs: Vec<OracleTensor>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<OperationOracleResult, VNextError> {
        self.invoke_internal(inputs, attributes)
            .map(|(result, _)| result)
    }

    pub fn invoke_and_compare(
        &self,
        inputs: Vec<OracleTensor>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
        actual: &OperationOracleResult,
    ) -> Result<bool, VNextError> {
        let (reference, mut symbols) = self.invoke_internal(inputs, attributes)?;
        validate_tensors_against_contracts(
            "actual oracle comparison outputs",
            actual.outputs(),
            &self.requested_operation.outputs,
            &mut symbols,
        )?;
        compare_oracle_results(self.comparison_policy(), actual, &reference)
    }

    fn invoke_internal(
        &self,
        inputs: Vec<OracleTensor>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<(OperationOracleResult, BTreeMap<String, u64>), VNextError> {
        if self.registered.oracle.descriptor() != &self.registered.descriptor {
            return Err(invalid_oracle(
                "registered oracle descriptor changed before invocation",
            ));
        }
        let mut symbols = BTreeMap::new();
        validate_tensors_against_contracts(
            "oracle request inputs",
            &inputs,
            &self.requested_operation.inputs,
            &mut symbols,
        )?;
        self.requested_operation.validate_attributes(&attributes)?;
        let request = OperationOracleRequest::new(
            self.terminal_operation.id.clone(),
            self.terminal_operation.fingerprint()?,
            inputs,
            attributes,
        )?;
        let result = self.registered.oracle.invoke(&request)?;
        if self.registered.oracle.descriptor() != &self.registered.descriptor {
            return Err(invalid_oracle(
                "registered oracle descriptor changed during invocation",
            ));
        }
        validate_tensors_against_contracts(
            "oracle result outputs",
            result.outputs(),
            &self.terminal_operation.outputs,
            &mut symbols,
        )?;
        Ok((result, symbols))
    }
}

fn validate_tensors_against_contracts(
    context: &str,
    tensors: &[OracleTensor],
    contracts: &[super::TensorContract],
    symbols: &mut BTreeMap<String, u64>,
) -> Result<(), VNextError> {
    if tensors.len() != contracts.len() {
        return Err(invalid_oracle(format!(
            "{context} count {} differs from contract count {}",
            tensors.len(),
            contracts.len()
        )));
    }
    for (index, (tensor, contract)) in tensors.iter().zip(contracts).enumerate() {
        if tensor.dimensions.len() != contract.dimensions().len()
            || !contract.element_types().contains(&tensor.element_type)
        {
            return Err(invalid_oracle(format!(
                "{context}[{index}] rank or dtype differs from the operation contract"
            )));
        }
        for (axis, (extent, constraint)) in tensor
            .dimensions
            .iter()
            .zip(contract.dimensions())
            .enumerate()
        {
            let accepted = match constraint {
                DimensionConstraint::Exact(expected) => extent == expected,
                DimensionConstraint::Range { minimum, maximum } => {
                    minimum <= extent && extent <= maximum
                }
                DimensionConstraint::Symbol(symbol) => match symbols.get(symbol) {
                    Some(expected) => expected == extent,
                    None => {
                        symbols.insert(symbol.clone(), *extent);
                        true
                    }
                },
            };
            if !accepted {
                return Err(invalid_oracle(format!(
                    "{context}[{index}] axis {axis} violates the operation contract"
                )));
            }
        }
    }
    Ok(())
}

/// Applies a terminal operation's exact, absolute or relative `OracleSpec` to
/// two already canonical result sets. Relative tolerance is
/// `abs(actual-reference) <= tolerance * abs(reference)`, so a zero reference
/// accepts only an exact zero difference.
pub fn compare_oracle_results(
    policy: &OracleSpec,
    actual: &OperationOracleResult,
    reference: &OperationOracleResult,
) -> Result<bool, VNextError> {
    if actual.outputs.len() != reference.outputs.len() {
        return Err(invalid_oracle(
            "oracle comparison result counts are different",
        ));
    }
    for (index, (actual, reference)) in actual.outputs.iter().zip(&reference.outputs).enumerate() {
        if actual.dimensions != reference.dimensions
            || actual.element_type != reference.element_type
        {
            return Err(invalid_oracle(format!(
                "oracle comparison tensor {index} shape or dtype is different"
            )));
        }
    }

    match policy {
        OracleSpec::Exact => Ok(actual
            .outputs
            .iter()
            .zip(&reference.outputs)
            .all(|(actual, reference)| actual.bytes == reference.bytes)),
        OracleSpec::AbsoluteTolerance { tolerance } => {
            compare_with_tolerance(actual, reference, rational_to_f64(*tolerance)?, false)
        }
        OracleSpec::RelativeTolerance { tolerance } => {
            compare_with_tolerance(actual, reference, rational_to_f64(*tolerance)?, true)
        }
        OracleSpec::ReferenceOperation { .. } => Err(invalid_oracle(
            "reference operation policy must be resolved to a terminal oracle before comparison",
        )),
    }
}

fn rational_to_f64(value: super::CanonicalRational) -> Result<f64, VNextError> {
    if value.numerator() < 0 {
        return Err(invalid_oracle("oracle tolerance must not be negative"));
    }
    let tolerance = value.numerator() as f64 / value.denominator() as f64;
    if !tolerance.is_finite() {
        return Err(invalid_oracle("oracle tolerance must be finite"));
    }
    Ok(tolerance)
}

fn compare_with_tolerance(
    actual: &OperationOracleResult,
    reference: &OperationOracleResult,
    tolerance: f64,
    relative: bool,
) -> Result<bool, VNextError> {
    for (actual, reference) in actual.outputs.iter().zip(&reference.outputs) {
        for index in 0..actual.element_count() {
            let actual = actual.numeric_value(index)?;
            let reference = reference.numeric_value(index)?;
            let difference = (actual - reference).abs();
            let allowed = if relative {
                tolerance * reference.abs()
            } else {
                tolerance
            };
            if difference > allowed {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
