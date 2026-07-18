use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, Write};

use super::model::PreparedModelFamilyWire;
use super::{
    AdmissionFitPolicy, CapabilityCatalog, ContractVersion, DeviceDescriptor,
    DynamicStorageProfile, ExecutablePlanView, ExecutionPlan, ModelFamilyRegistry,
    PlanNodeResolution, PreparedModelFamily, ProviderId, RuntimePolicy, SpecialTokenRole,
    TokenizerDescriptor, UnvalidatedExecutionPlan, UnvalidatedExecutionPlanWire,
    UnvalidatedPreparedModelFamily, VNextError,
};

/// Maximum raw byte length accepted for one resolution source artifact.
pub const MAX_RESOLUTION_SOURCE_BYTES: usize = 32 * 1024 * 1024;
/// Maximum serialized byte length accepted by resolved-plan wire decoding.
pub const MAX_RESOLVED_MODEL_PLAN_WIRE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum container nesting depth in a parsed resolution source document.
pub const MAX_RESOLUTION_JSON_DEPTH: usize = 128;
/// Maximum total JSON values in a parsed resolution source document.
pub const MAX_RESOLUTION_JSON_NODES: usize = 1_000_000;
/// Maximum cumulative bytes across object keys and string values.
pub const MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES: usize = MAX_RESOLUTION_SOURCE_BYTES;
/// Maximum cumulative string bytes in one resolution source provenance record.
pub const MAX_RESOLUTION_PROVENANCE_BYTES: usize = 4 * 1024;
/// Maximum number of source JSON pointers recorded by one artifact.
pub const MAX_RESOLUTION_FIELD_PATHS: usize = 4_096;
/// Maximum byte length of one source JSON pointer.
pub const MAX_RESOLUTION_FIELD_PATH_BYTES: usize = 512;
/// Maximum cumulative bytes across all source JSON pointers in one artifact.
pub const MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES: usize = 1024 * 1024;

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_fingerprint<T: Serialize + ?Sized>(
    value: &T,
    context: &'static str,
) -> Result<String, VNextError> {
    let value = serde_json::to_value(value).map_err(|error| VNextError::Serialization {
        context,
        message: error.to_string(),
    })?;
    canonical_value_fingerprint(&canonicalize_json(value), context)
}

struct Sha256Writer(Sha256);

impl Write for Sha256Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn canonical_value_fingerprint(
    value: &serde_json::Value,
    context: &'static str,
) -> Result<String, VNextError> {
    let mut writer = Sha256Writer(Sha256::new());
    serde_json::to_writer(&mut writer, value).map_err(|error| VNextError::Serialization {
        context,
        message: error.to_string(),
    })?;
    Ok(format!("{:x}", writer.0.finalize()))
}

fn canonical_json_value<T: Serialize + ?Sized>(
    value: &T,
    context: &'static str,
) -> Result<serde_json::Value, VNextError> {
    serde_json::to_value(value)
        .map(canonicalize_json)
        .map_err(|error| VNextError::Serialization {
            context,
            message: error.to_string(),
        })
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        value => value,
    }
}

fn invalid_plan(field: impl Into<String>, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidResolvedModelPlan {
        field: field.into(),
        reason: reason.into(),
    }
}

fn validate_resolution_source_bytes(source_bytes: &[u8]) -> Result<(), VNextError> {
    if source_bytes.is_empty() || source_bytes.len() > MAX_RESOLUTION_SOURCE_BYTES {
        return Err(invalid_plan(
            "resolution_source_evidence.source_bytes",
            format!("must contain between 1 and {MAX_RESOLUTION_SOURCE_BYTES} bytes"),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ResolutionJsonBudget {
    maximum_depth: usize,
    maximum_nodes: usize,
    maximum_key_and_string_bytes: usize,
}

impl ResolutionJsonBudget {
    const SOURCE: Self = Self {
        maximum_depth: MAX_RESOLUTION_JSON_DEPTH,
        maximum_nodes: MAX_RESOLUTION_JSON_NODES,
        maximum_key_and_string_bytes: MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES,
    };
}

struct ResolutionJsonPreflight {
    budget: ResolutionJsonBudget,
    nodes: usize,
    key_and_string_bytes: usize,
    violation: Option<VNextError>,
}

impl ResolutionJsonPreflight {
    fn new(budget: ResolutionJsonBudget) -> Self {
        Self {
            budget,
            nodes: 0,
            key_and_string_bytes: 0,
            violation: None,
        }
    }

    fn fail<E: serde::de::Error>(&mut self, error: VNextError) -> Result<(), E> {
        self.violation = Some(error);
        Err(E::custom(
            "resolution source JSON exceeds its structural budget",
        ))
    }

    fn account_node<E: serde::de::Error>(&mut self, depth: usize) -> Result<(), E> {
        if depth > self.budget.maximum_depth {
            return self.fail(invalid_plan(
                "resolution_source_evidence.document.depth",
                format!("must not exceed {}", self.budget.maximum_depth),
            ));
        }
        let Some(nodes) = self.nodes.checked_add(1) else {
            return self.fail(invalid_plan(
                "resolution_source_evidence.document.nodes",
                "node count overflowed",
            ));
        };
        if nodes > self.budget.maximum_nodes {
            return self.fail(invalid_plan(
                "resolution_source_evidence.document.nodes",
                format!("must not exceed {}", self.budget.maximum_nodes),
            ));
        }
        self.nodes = nodes;
        Ok(())
    }

    fn account_text<E: serde::de::Error>(&mut self, bytes: usize) -> Result<(), E> {
        let Some(key_and_string_bytes) = self.key_and_string_bytes.checked_add(bytes) else {
            return self.fail(invalid_plan(
                "resolution_source_evidence.document.key_and_string_bytes",
                "byte count overflowed",
            ));
        };
        if key_and_string_bytes > self.budget.maximum_key_and_string_bytes {
            return self.fail(invalid_plan(
                "resolution_source_evidence.document.key_and_string_bytes",
                format!(
                    "must not exceed {}",
                    self.budget.maximum_key_and_string_bytes
                ),
            ));
        }
        self.key_and_string_bytes = key_and_string_bytes;
        Ok(())
    }
}

struct ResolutionJsonValueSeed<'a> {
    preflight: &'a mut ResolutionJsonPreflight,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for ResolutionJsonValueSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.preflight.account_node::<D::Error>(self.depth)?;
        deserializer.deserialize_any(ResolutionJsonValueVisitor {
            preflight: self.preflight,
            depth: self.depth,
        })
    }
}

struct ResolutionJsonValueVisitor<'a> {
    preflight: &'a mut ResolutionJsonPreflight,
    depth: usize,
}

impl<'de> Visitor<'de> for ResolutionJsonValueVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("one JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let child_depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| serde::de::Error::custom("resolution source JSON depth overflowed"))?;
        while sequence
            .next_element_seed(ResolutionJsonValueSeed {
                preflight: &mut *self.preflight,
                depth: child_depth,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let child_depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| serde::de::Error::custom("resolution source JSON depth overflowed"))?;
        while object
            .next_key_seed(ResolutionJsonKeySeed {
                preflight: &mut *self.preflight,
            })?
            .is_some()
        {
            object.next_value_seed(ResolutionJsonValueSeed {
                preflight: &mut *self.preflight,
                depth: child_depth,
            })?;
        }
        Ok(())
    }
}

struct ResolutionJsonKeySeed<'a> {
    preflight: &'a mut ResolutionJsonPreflight,
}

impl<'de> DeserializeSeed<'de> for ResolutionJsonKeySeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ResolutionJsonKeyVisitor {
            preflight: self.preflight,
        })
    }
}

struct ResolutionJsonKeyVisitor<'a> {
    preflight: &'a mut ResolutionJsonPreflight,
}

impl<'de> Visitor<'de> for ResolutionJsonKeyVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object key")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.preflight.account_text::<E>(value.len())
    }
}

fn preflight_resolution_json_depth(
    source_bytes: &[u8],
    maximum_depth: usize,
) -> Result<(), VNextError> {
    let mut closing_delimiters = Vec::with_capacity(maximum_depth.saturating_add(1));
    let mut in_string = false;
    let mut escaped = false;

    for byte in source_bytes.iter().copied() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => {
                if closing_delimiters.len() > maximum_depth {
                    return Err(invalid_plan(
                        "resolution_source_evidence.document.depth",
                        format!("must not exceed {maximum_depth}"),
                    ));
                }
                in_string = true;
            }
            b'[' | b'{' => {
                if closing_delimiters.len() > maximum_depth {
                    return Err(invalid_plan(
                        "resolution_source_evidence.document.depth",
                        format!("must not exceed {maximum_depth}"),
                    ));
                }
                closing_delimiters.push(if byte == b'[' { b']' } else { b'}' });
            }
            b']' | b'}' => {
                if closing_delimiters.last() == Some(&byte) {
                    closing_delimiters.pop();
                }
            }
            b' ' | b'\t' | b'\r' | b'\n' | b',' | b':' => {}
            _ if closing_delimiters.len() > maximum_depth => {
                return Err(invalid_plan(
                    "resolution_source_evidence.document.depth",
                    format!("must not exceed {maximum_depth}"),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn preflight_resolution_json(
    source_bytes: &[u8],
    budget: ResolutionJsonBudget,
) -> Result<(), VNextError> {
    // serde_json's built-in recursion guard can fire before our public depth
    // contract. This quote-aware pass preserves the contract's structured
    // depth error without allocating a JSON tree.
    preflight_resolution_json_depth(source_bytes, budget.maximum_depth)?;

    let mut preflight = ResolutionJsonPreflight::new(budget);
    let result = {
        let mut deserializer = serde_json::Deserializer::from_slice(source_bytes);
        ResolutionJsonValueSeed {
            preflight: &mut preflight,
            depth: 0,
        }
        .deserialize(&mut deserializer)
        .and_then(|()| deserializer.end())
    };
    if let Some(error) = preflight.violation {
        return Err(error);
    }
    result.map_err(|error| VNextError::Serialization {
        context: "parse resolution source JSON",
        message: error.to_string(),
    })
}

fn validate_resolution_json_tree(document: &serde_json::Value) -> Result<(), VNextError> {
    let mut stack = vec![(document, 0usize)];
    let mut nodes = 0usize;
    let mut key_and_string_bytes = 0usize;

    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_RESOLUTION_JSON_DEPTH {
            return Err(invalid_plan(
                "resolution_source_evidence.document.depth",
                format!("must not exceed {MAX_RESOLUTION_JSON_DEPTH}"),
            ));
        }
        nodes = nodes.checked_add(1).ok_or_else(|| {
            invalid_plan(
                "resolution_source_evidence.document.nodes",
                "node count overflowed",
            )
        })?;
        if nodes > MAX_RESOLUTION_JSON_NODES {
            return Err(invalid_plan(
                "resolution_source_evidence.document.nodes",
                format!("must not exceed {MAX_RESOLUTION_JSON_NODES}"),
            ));
        }

        match value {
            serde_json::Value::String(value) => {
                key_and_string_bytes =
                    key_and_string_bytes
                        .checked_add(value.len())
                        .ok_or_else(|| {
                            invalid_plan(
                                "resolution_source_evidence.document.key_and_string_bytes",
                                "byte count overflowed",
                            )
                        })?;
            }
            serde_json::Value::Array(values) => {
                let child_depth = depth.checked_add(1).ok_or_else(|| {
                    invalid_plan(
                        "resolution_source_evidence.document.depth",
                        "depth overflowed",
                    )
                })?;
                stack.extend(values.iter().map(|value| (value, child_depth)));
            }
            serde_json::Value::Object(values) => {
                let child_depth = depth.checked_add(1).ok_or_else(|| {
                    invalid_plan(
                        "resolution_source_evidence.document.depth",
                        "depth overflowed",
                    )
                })?;
                for (key, value) in values {
                    key_and_string_bytes =
                        key_and_string_bytes.checked_add(key.len()).ok_or_else(|| {
                            invalid_plan(
                                "resolution_source_evidence.document.key_and_string_bytes",
                                "byte count overflowed",
                            )
                        })?;
                    stack.push((value, child_depth));
                }
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            }
        }

        if key_and_string_bytes > MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES {
            return Err(invalid_plan(
                "resolution_source_evidence.document.key_and_string_bytes",
                format!("must not exceed {MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES}"),
            ));
        }
    }

    Ok(())
}

fn drop_json_iteratively(document: serde_json::Value) {
    let mut stack = vec![document];
    while let Some(value) = stack.pop() {
        match value {
            serde_json::Value::Array(mut values) => stack.append(&mut values),
            serde_json::Value::Object(values) => stack.extend(values.into_values()),
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => {}
        }
    }
}

fn validate_and_canonicalize_resolution_json(
    document: serde_json::Value,
    fingerprint_context: &'static str,
) -> Result<(serde_json::Value, String), VNextError> {
    if let Err(error) = validate_resolution_json_tree(&document) {
        drop_json_iteratively(document);
        return Err(error);
    }
    let document = canonicalize_json(document);
    let fingerprint = canonical_value_fingerprint(&document, fingerprint_context)?;
    Ok((document, fingerprint))
}

fn validate_portable_identifier(
    kind: &'static str,
    value: &str,
    maximum_length: usize,
) -> Result<(), VNextError> {
    if value.is_empty() || value.len() > maximum_length {
        return Err(invalid_plan(
            kind,
            format!("must contain between 1 and {maximum_length} bytes"),
        ));
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
    }) {
        return Err(invalid_plan(kind, "contains a non-portable character"));
    }
    Ok(())
}

fn validate_source_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.ends_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    LocalDirectory,
    LocalFile,
    Repository,
    ReleaseArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OriginalModelSource {
    pub kind: ModelSourceKind,
    pub location: String,
    pub requested_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFingerprint {
    pub relative_path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelSource {
    pub canonical_location: String,
    pub resolved_revision: String,
    pub files: Vec<FileFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfigFingerprint {
    pub source_file: String,
    pub sha256: String,
    pub typed_config_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSelection {
    pub provider_id: ProviderId,
    pub contract_version: ContractVersion,
    pub implementation_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingDiscipline {
    FirstReady,
    Priority,
    Deadline,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMemoryPolicy {
    pub capacity_bytes: u64,
    pub reserve_bytes: u64,
    pub maximum_active_sequences: u32,
    pub dynamic_storage_profile_order: Vec<DynamicStorageProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionPolicy {
    pub maximum_queue_depth: u32,
    pub maximum_scheduled_tokens: u64,
    pub sequence_fit_policy: AdmissionFitPolicy,
    pub allow_defer: bool,
    pub cancellation_check_interval_steps: u32,
}

#[derive(Serialize)]
struct RuntimePolicyFingerprintPayload<'a> {
    policy_id: &'a str,
    version: ContractVersion,
    scheduling: SchedulingDiscipline,
    memory: &'a RuntimeMemoryPolicy,
    admission: &'a AdmissionPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedRuntimePolicy {
    policy_id: String,
    version: ContractVersion,
    scheduling: SchedulingDiscipline,
    memory: RuntimeMemoryPolicy,
    admission: AdmissionPolicy,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedRuntimePolicyWire {
    policy_id: String,
    version: ContractVersion,
    scheduling: SchedulingDiscipline,
    memory: RuntimeMemoryPolicy,
    admission: AdmissionPolicy,
    fingerprint: String,
}

impl<'de> Deserialize<'de> for ResolvedRuntimePolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedRuntimePolicyWire::deserialize(deserializer)?;
        let policy = Self::new(
            wire.policy_id,
            wire.version,
            wire.scheduling,
            wire.memory,
            wire.admission,
        )
        .map_err(serde::de::Error::custom)?;
        if wire.fingerprint != policy.fingerprint {
            return Err(serde::de::Error::custom(format!(
                "runtime policy fingerprint mismatch: expected `{}`, actual `{}`",
                policy.fingerprint, wire.fingerprint
            )));
        }
        Ok(policy)
    }
}

impl ResolvedRuntimePolicy {
    pub fn new(
        policy_id: impl Into<String>,
        version: ContractVersion,
        scheduling: SchedulingDiscipline,
        memory: RuntimeMemoryPolicy,
        admission: AdmissionPolicy,
    ) -> Result<Self, VNextError> {
        let policy_id = policy_id.into();
        Self::validate_fields(&policy_id, version, &memory, &admission)?;
        let fingerprint =
            Self::compute_fingerprint(&policy_id, version, scheduling, &memory, &admission)?;
        Ok(Self {
            policy_id,
            version,
            scheduling,
            memory,
            admission,
            fingerprint,
        })
    }

    fn validate_fields(
        policy_id: &str,
        version: ContractVersion,
        memory: &RuntimeMemoryPolicy,
        admission: &AdmissionPolicy,
    ) -> Result<(), VNextError> {
        validate_portable_identifier("runtime_policy.policy_id", policy_id, 160)?;
        if version.major == 0 {
            return Err(invalid_plan(
                "runtime_policy.version",
                "major version must be non-zero",
            ));
        }
        if memory.capacity_bytes == 0
            || memory.reserve_bytes >= memory.capacity_bytes
            || memory.maximum_active_sequences == 0
            || memory.dynamic_storage_profile_order.is_empty()
            || memory
                .dynamic_storage_profile_order
                .iter()
                .enumerate()
                .any(|(index, profile)| {
                    memory.dynamic_storage_profile_order[index + 1..].contains(profile)
                })
        {
            return Err(invalid_plan(
                "runtime_policy.memory",
                "capacity and reserve must be valid and maximum_active_sequences must be non-zero",
            ));
        }
        if admission.maximum_queue_depth == 0
            || admission.maximum_scheduled_tokens == 0
            || admission.cancellation_check_interval_steps == 0
        {
            return Err(invalid_plan(
                "runtime_policy.admission",
                "queue depth, scheduled-token ceiling, and cancellation interval must be non-zero",
            ));
        }
        Ok(())
    }

    fn compute_fingerprint(
        policy_id: &str,
        version: ContractVersion,
        scheduling: SchedulingDiscipline,
        memory: &RuntimeMemoryPolicy,
        admission: &AdmissionPolicy,
    ) -> Result<String, VNextError> {
        canonical_fingerprint(
            &RuntimePolicyFingerprintPayload {
                policy_id,
                version,
                scheduling,
                memory,
                admission,
            },
            "serialize resolved runtime policy",
        )
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn scheduling(&self) -> SchedulingDiscipline {
        self.scheduling
    }

    pub fn memory(&self) -> &RuntimeMemoryPolicy {
        &self.memory
    }

    pub fn admission(&self) -> &AdmissionPolicy {
        &self.admission
    }

    pub fn fingerprint_str(&self) -> &str {
        &self.fingerprint
    }
}

impl RuntimePolicy for ResolvedRuntimePolicy {
    fn version(&self) -> ContractVersion {
        self.version
    }

    fn memory_capacity_bytes(&self) -> u64 {
        self.memory.capacity_bytes
    }

    fn memory_reserve_bytes(&self) -> u64 {
        self.memory.reserve_bytes
    }

    fn maximum_active_sequences(&self) -> u32 {
        self.memory.maximum_active_sequences
    }

    fn maximum_scheduled_tokens(&self) -> u64 {
        self.admission.maximum_scheduled_tokens
    }

    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.memory.dynamic_storage_profile_order
    }

    fn validate(&self) -> Result<(), VNextError> {
        Self::validate_fields(&self.policy_id, self.version, &self.memory, &self.admission)?;
        let computed = Self::compute_fingerprint(
            &self.policy_id,
            self.version,
            self.scheduling,
            &self.memory,
            &self.admission,
        )?;
        if self.fingerprint != computed {
            return Err(invalid_plan(
                "runtime_policy.fingerprint",
                format!(
                    "does not match typed fields: expected `{computed}`, actual `{}`",
                    self.fingerprint
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RationalValue {
    numerator: i64,
    denominator: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalValueWire {
    numerator: i64,
    denominator: u64,
}

impl<'de> Deserialize<'de> for RationalValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalValueWire::deserialize(deserializer)?;
        Self::new(wire.numerator, wire.denominator).map_err(serde::de::Error::custom)
    }
}

impl RationalValue {
    pub fn new(numerator: i64, denominator: u64) -> Result<Self, VNextError> {
        if denominator == 0 {
            return Err(invalid_plan(
                "sampling.rational",
                "denominator must be non-zero",
            ));
        }
        let divisor = greatest_common_divisor(numerator.unsigned_abs(), denominator);
        let numerator = ((numerator as i128) / (divisor as i128)) as i64;
        let denominator = denominator / divisor;
        Ok(Self {
            numerator,
            denominator,
        })
    }

    fn validate(&self, field: &str) -> Result<(), VNextError> {
        if self.denominator == 0
            || greatest_common_divisor(self.numerator.unsigned_abs(), self.denominator) != 1
            || (self.numerator == 0 && self.denominator != 1)
        {
            return Err(invalid_plan(
                field,
                "rational value must be reduced with a non-zero denominator",
            ));
        }
        Ok(())
    }

    fn compare(&self, numerator: i64, denominator: u64) -> Ordering {
        ((self.numerator as i128) * (denominator as i128))
            .cmp(&((numerator as i128) * (self.denominator as i128)))
    }

    pub fn numerator(&self) -> i64 {
        self.numerator
    }

    pub fn denominator(&self) -> u64 {
        self.denominator
    }
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriStatePolicy {
    ModelDefault,
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SamplingPolicy {
    temperature: RationalValue,
    top_p: RationalValue,
    top_k: Option<u32>,
    min_p: RationalValue,
    presence_penalty: RationalValue,
    repetition_penalty: RationalValue,
    seed: u64,
    thinking_policy: TriStatePolicy,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingPolicyWire {
    temperature: RationalValue,
    top_p: RationalValue,
    top_k: Option<u32>,
    min_p: RationalValue,
    presence_penalty: RationalValue,
    repetition_penalty: RationalValue,
    seed: u64,
    thinking_policy: TriStatePolicy,
}

impl<'de> Deserialize<'de> for SamplingPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SamplingPolicyWire::deserialize(deserializer)?;
        Self::new(
            wire.temperature,
            wire.top_p,
            wire.top_k,
            wire.min_p,
            wire.presence_penalty,
            wire.repetition_penalty,
            wire.seed,
            wire.thinking_policy,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl SamplingPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: RationalValue,
        top_p: RationalValue,
        top_k: Option<u32>,
        min_p: RationalValue,
        presence_penalty: RationalValue,
        repetition_penalty: RationalValue,
        seed: u64,
        thinking_policy: TriStatePolicy,
    ) -> Result<Self, VNextError> {
        let policy = Self {
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            repetition_penalty,
            seed,
            thinking_policy,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), VNextError> {
        self.temperature.validate("sampling.temperature")?;
        self.top_p.validate("sampling.top_p")?;
        self.min_p.validate("sampling.min_p")?;
        self.presence_penalty
            .validate("sampling.presence_penalty")?;
        self.repetition_penalty
            .validate("sampling.repetition_penalty")?;

        if self.temperature.compare(0, 1) == Ordering::Less {
            return Err(invalid_plan(
                "sampling.temperature",
                "must be greater than or equal to zero",
            ));
        }
        if self.top_p.compare(0, 1) != Ordering::Greater
            || self.top_p.compare(1, 1) == Ordering::Greater
        {
            return Err(invalid_plan(
                "sampling.top_p",
                "must be in the interval (0, 1]",
            ));
        }
        if self.top_k == Some(0) {
            return Err(invalid_plan(
                "sampling.top_k",
                "must be absent or greater than zero",
            ));
        }
        if self.min_p.compare(0, 1) == Ordering::Less
            || self.min_p.compare(1, 1) == Ordering::Greater
        {
            return Err(invalid_plan(
                "sampling.min_p",
                "must be in the interval [0, 1]",
            ));
        }
        if self.presence_penalty.compare(-2, 1) == Ordering::Less
            || self.presence_penalty.compare(2, 1) == Ordering::Greater
        {
            return Err(invalid_plan(
                "sampling.presence_penalty",
                "must be in the interval [-2, 2]",
            ));
        }
        if self.repetition_penalty.compare(0, 1) != Ordering::Greater {
            return Err(invalid_plan(
                "sampling.repetition_penalty",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn temperature(&self) -> RationalValue {
        self.temperature
    }

    pub fn top_p(&self) -> RationalValue {
        self.top_p
    }

    pub fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    pub fn min_p(&self) -> RationalValue {
        self.min_p
    }

    pub fn presence_penalty(&self) -> RationalValue {
        self.presence_penalty
    }

    pub fn repetition_penalty(&self) -> RationalValue {
        self.repetition_penalty
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn thinking_policy(&self) -> TriStatePolicy {
        self.thinking_policy
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopTokenCollisionPolicy {
    allowed_model_roles: BTreeSet<SpecialTokenRole>,
}

impl StopTokenCollisionPolicy {
    pub fn new(allowed_model_roles: BTreeSet<SpecialTokenRole>) -> Result<Self, VNextError> {
        if allowed_model_roles.contains(&SpecialTokenRole::Stop) {
            return Err(invalid_plan(
                "stop.collision_policy",
                "a stop-token collision policy can only name model-owned roles",
            ));
        }
        Ok(Self {
            allowed_model_roles,
        })
    }

    pub fn require_distinct() -> Self {
        Self {
            allowed_model_roles: BTreeSet::new(),
        }
    }

    pub fn allows(&self, model_role: SpecialTokenRole) -> bool {
        self.allowed_model_roles.contains(&model_role)
    }

    pub fn allowed_model_roles(&self) -> &BTreeSet<SpecialTokenRole> {
        &self.allowed_model_roles
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopPolicy {
    pub maximum_output_tokens: u32,
    pub token_ids: BTreeSet<u32>,
    pub strings: Vec<String>,
    pub collision_policy: StopTokenCollisionPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputPolicy {
    Disabled,
    JsonObject,
    JsonSchema { schema_sha256: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionField {
    OriginalSource,
    ResolvedSource,
    Config,
    ExternalMetadata,
    Family,
    WeightSchema,
    WeightFormat,
    Tokenizer,
    Template,
    SpecialTokens,
    Device,
    Capabilities,
    RuntimePreset,
    RuntimeMemory,
    Admission,
    Engine,
    ExecutionPlan,
    Sampling,
    Stop,
    StructuredOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionDecisionSource {
    UserInput,
    CommandLine,
    ConfigFile,
    ModelMetadata,
    TypedModelResolution,
    ProductDefault,
    RuntimePreset,
    CapabilityResolution,
    Planner,
}

impl ResolutionField {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OriginalSource => "original_source",
            Self::ResolvedSource => "resolved_source",
            Self::Config => "config",
            Self::ExternalMetadata => "external_metadata",
            Self::Family => "family",
            Self::WeightSchema => "weight_schema",
            Self::WeightFormat => "weight_format",
            Self::Tokenizer => "tokenizer",
            Self::Template => "template",
            Self::SpecialTokens => "special_tokens",
            Self::Device => "device",
            Self::Capabilities => "capabilities",
            Self::RuntimePreset => "runtime_preset",
            Self::RuntimeMemory => "runtime_memory",
            Self::Admission => "admission",
            Self::Engine => "engine",
            Self::ExecutionPlan => "execution_plan",
            Self::Sampling => "sampling",
            Self::Stop => "stop",
            Self::StructuredOutput => "structured_output",
        }
    }

    /// Returns whether a provenance source is allowed to author this decision.
    /// This matrix prevents a self-consistent package from relabeling a
    /// planner, model, or capability decision as a generic product default.
    pub const fn accepts_source(self, source: ResolutionDecisionSource) -> bool {
        use ResolutionDecisionSource as Source;
        use ResolutionField as Field;

        match self {
            Field::OriginalSource => matches!(
                source,
                Source::UserInput | Source::CommandLine | Source::ConfigFile
            ),
            Field::ResolvedSource => {
                matches!(source, Source::ModelMetadata | Source::TypedModelResolution)
            }
            Field::Config => matches!(
                source,
                Source::ConfigFile | Source::ModelMetadata | Source::TypedModelResolution
            ),
            Field::ExternalMetadata
            | Field::Family
            | Field::WeightSchema
            | Field::Tokenizer
            | Field::Template
            | Field::SpecialTokens => {
                matches!(source, Source::ModelMetadata | Source::TypedModelResolution)
            }
            Field::WeightFormat => matches!(
                source,
                Source::CommandLine
                    | Source::ConfigFile
                    | Source::ModelMetadata
                    | Source::TypedModelResolution
            ),
            Field::Device => matches!(
                source,
                Source::CommandLine
                    | Source::ConfigFile
                    | Source::ProductDefault
                    | Source::CapabilityResolution
            ),
            Field::Capabilities => matches!(source, Source::CapabilityResolution),
            Field::RuntimePreset => matches!(
                source,
                Source::CommandLine
                    | Source::ConfigFile
                    | Source::ProductDefault
                    | Source::RuntimePreset
            ),
            Field::RuntimeMemory | Field::Admission => matches!(
                source,
                Source::CommandLine | Source::ConfigFile | Source::RuntimePreset
            ),
            Field::Engine => {
                matches!(source, Source::RuntimePreset | Source::CapabilityResolution)
            }
            Field::ExecutionPlan => matches!(source, Source::Planner),
            Field::Sampling | Field::StructuredOutput => matches!(
                source,
                Source::UserInput
                    | Source::CommandLine
                    | Source::ConfigFile
                    | Source::ProductDefault
            ),
            Field::Stop => matches!(
                source,
                Source::UserInput
                    | Source::CommandLine
                    | Source::ConfigFile
                    | Source::ModelMetadata
                    | Source::ProductDefault
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResolutionReasonId(String);

impl ResolutionReasonId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        validate_portable_identifier("resolution_reason_id", &value, 160)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ResolutionReasonId {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ResolutionReasonId> for String {
    fn from(value: ResolutionReasonId) -> Self {
        value.0
    }
}

impl fmt::Display for ResolutionReasonId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResolutionFingerprint(String);

impl ResolutionFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        if !is_canonical_sha256(&value) {
            return Err(invalid_plan(
                "resolution_fingerprint",
                "must be a canonical lowercase SHA-256",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ResolutionFingerprint {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ResolutionFingerprint> for String {
    fn from(value: ResolutionFingerprint) -> Self {
        value.0
    }
}

impl fmt::Display for ResolutionFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResolutionArtifactId(String);

impl ResolutionArtifactId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        validate_portable_identifier("resolution_artifact_id", &value, 160)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ResolutionArtifactId {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ResolutionArtifactId> for String {
    fn from(value: ResolutionArtifactId) -> Self {
        value.0
    }
}

impl fmt::Display for ResolutionArtifactId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Externally anchored origin of resolution source bytes. A source is either
/// one exact file from the locked model snapshot or an explicitly identified
/// upstream producer. There is no unstructured locator variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ResolutionSourceProvenance {
    LockedModelFile {
        relative_path: String,
    },
    Upstream {
        producer_id: String,
        producer_version: ContractVersion,
        producer_implementation_fingerprint: ResolutionFingerprint,
        revision: String,
        artifact_locator: String,
    },
}

impl ResolutionSourceProvenance {
    fn validate(&self) -> Result<(), VNextError> {
        match self {
            Self::LockedModelFile { relative_path } => {
                if !validate_source_path(relative_path) {
                    return Err(invalid_plan(
                        "resolution_source_provenance.locked_model_file",
                        "relative path must identify one portable locked model file",
                    ));
                }
            }
            Self::Upstream {
                producer_id,
                producer_version,
                revision,
                artifact_locator,
                ..
            } => {
                validate_portable_identifier(
                    "resolution_source_provenance.producer_id",
                    producer_id,
                    160,
                )?;
                if producer_version.major == 0 {
                    return Err(invalid_plan(
                        "resolution_source_provenance.producer_version",
                        "producer contract major version must be non-zero",
                    ));
                }
                validate_portable_identifier(
                    "resolution_source_provenance.revision",
                    revision,
                    256,
                )?;
                validate_portable_identifier(
                    "resolution_source_provenance.artifact_locator",
                    artifact_locator,
                    512,
                )?;
            }
        }
        Ok(())
    }

    pub fn locator(&self) -> &str {
        match self {
            Self::LockedModelFile { relative_path } => relative_path,
            Self::Upstream {
                artifact_locator, ..
            } => artifact_locator,
        }
    }
}

fn resolution_provenance_bytes(
    provenance: &ResolutionSourceProvenance,
) -> Result<usize, VNextError> {
    match provenance {
        ResolutionSourceProvenance::LockedModelFile { relative_path } => Ok(relative_path.len()),
        ResolutionSourceProvenance::Upstream {
            producer_id,
            producer_implementation_fingerprint,
            revision,
            artifact_locator,
            ..
        } => [
            producer_id.len(),
            producer_implementation_fingerprint.as_str().len(),
            revision.len(),
            artifact_locator.len(),
        ]
        .into_iter()
        .try_fold(0usize, |total, length| {
            total.checked_add(length).ok_or_else(|| {
                invalid_plan(
                    "resolution_source_evidence.provenance",
                    "provenance byte count overflowed",
                )
            })
        }),
    }
}

fn validate_resolution_field_paths(field_paths: &BTreeSet<String>) -> Result<(), VNextError> {
    if field_paths.is_empty() || field_paths.len() > MAX_RESOLUTION_FIELD_PATHS {
        return Err(invalid_plan(
            "resolution_source_evidence.field_paths",
            format!("must contain between 1 and {MAX_RESOLUTION_FIELD_PATHS} unique paths"),
        ));
    }

    let mut total_bytes = 0usize;
    for path in field_paths {
        if !path.starts_with('/')
            || path.len() > MAX_RESOLUTION_FIELD_PATH_BYTES
            || path.trim() != path
            || path.bytes().any(|byte| byte.is_ascii_control())
        {
            return Err(invalid_plan(
                "resolution_source_evidence.field_paths",
                format!(
                    "each path must be a portable JSON pointer of at most {MAX_RESOLUTION_FIELD_PATH_BYTES} bytes"
                ),
            ));
        }
        total_bytes = total_bytes.checked_add(path.len()).ok_or_else(|| {
            invalid_plan(
                "resolution_source_evidence.field_paths",
                "field-path byte count overflowed",
            )
        })?;
        if total_bytes > MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES {
            return Err(invalid_plan(
                "resolution_source_evidence.field_paths",
                format!("total path bytes must not exceed {MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES}"),
            ));
        }
    }
    Ok(())
}

fn validate_resolution_source_availability(
    source_bytes: &[u8],
    provenance: &ResolutionSourceProvenance,
    field_paths: &BTreeSet<String>,
) -> Result<(), VNextError> {
    validate_resolution_source_bytes(source_bytes)?;
    let provenance_bytes = resolution_provenance_bytes(provenance)?;
    if provenance_bytes > MAX_RESOLUTION_PROVENANCE_BYTES {
        return Err(invalid_plan(
            "resolution_source_evidence.provenance",
            format!("must not exceed {MAX_RESOLUTION_PROVENANCE_BYTES} bytes"),
        ));
    }
    validate_resolution_field_paths(field_paths)?;
    provenance.validate()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolutionSourceArtifact {
    id: ResolutionArtifactId,
    source: ResolutionDecisionSource,
    provenance: ResolutionSourceProvenance,
    parser: ResolutionParserDescriptor,
    content_size_bytes: u64,
    content_fingerprint: ResolutionFingerprint,
    canonical_document_fingerprint: ResolutionFingerprint,
    fields: BTreeMap<String, ResolutionFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResolutionSourceArtifact {
    pub id: ResolutionArtifactId,
    pub source: ResolutionDecisionSource,
    pub provenance: ResolutionSourceProvenance,
    pub parser: ResolutionParserDescriptor,
    pub content_size_bytes: u64,
    pub content_fingerprint: ResolutionFingerprint,
    pub canonical_document_fingerprint: ResolutionFingerprint,
    pub fields: BTreeMap<String, ResolutionFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolutionParserDescriptor {
    id: String,
    version: ContractVersion,
    implementation_fingerprint: ResolutionFingerprint,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionParserDescriptorWire {
    id: String,
    version: ContractVersion,
    implementation_fingerprint: ResolutionFingerprint,
}

impl<'de> Deserialize<'de> for ResolutionParserDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolutionParserDescriptorWire::deserialize(deserializer)?;
        Self::new(wire.id, wire.version, wire.implementation_fingerprint)
            .map_err(serde::de::Error::custom)
    }
}

impl ResolutionParserDescriptor {
    pub fn new(
        id: impl Into<String>,
        version: ContractVersion,
        implementation_fingerprint: ResolutionFingerprint,
    ) -> Result<Self, VNextError> {
        let descriptor = Self {
            id: id.into(),
            version,
            implementation_fingerprint,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    fn validate(&self) -> Result<(), VNextError> {
        validate_portable_identifier("resolution_parser.id", &self.id, 160)?;
        if self.version.major == 0 {
            return Err(invalid_plan(
                "resolution_parser.version",
                "parser contract major version must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn implementation_fingerprint(&self) -> &ResolutionFingerprint {
        &self.implementation_fingerprint
    }
}

/// Trusted parser implementation supplied by the composition root. Core
/// records its exact identity and reruns it for every wire revalidation.
pub trait ResolutionSourceParser: Send + Sync {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError>;

    fn parse(
        &self,
        source: ResolutionDecisionSource,
        provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<serde_json::Value, VNextError>;
}

pub struct JsonResolutionSourceParser;

pub static JSON_RESOLUTION_SOURCE_PARSER: JsonResolutionSourceParser = JsonResolutionSourceParser;

impl ResolutionSourceParser for JsonResolutionSourceParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.core-json",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(canonical_fingerprint(
                &"ferrum.resolution-parser.core-json.v1",
                "fingerprint core JSON resolution parser",
            )?)?,
        )
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<serde_json::Value, VNextError> {
        validate_resolution_source_bytes(source_bytes)?;
        preflight_resolution_json(source_bytes, ResolutionJsonBudget::SOURCE)?;
        serde_json::from_slice(source_bytes).map_err(|error| VNextError::Serialization {
            context: "parse resolution source JSON",
            message: error.to_string(),
        })
    }
}

/// External source bytes plus the exact trusted parser used to derive typed
/// decision fields. This evidence is never serialized into a resolved plan.
#[derive(Clone)]
pub struct ResolutionSourceEvidence<'a> {
    id: ResolutionArtifactId,
    source: ResolutionDecisionSource,
    provenance: ResolutionSourceProvenance,
    source_bytes: Vec<u8>,
    field_paths: BTreeSet<String>,
    parser: &'a dyn ResolutionSourceParser,
}

impl<'a> ResolutionSourceEvidence<'a> {
    pub fn new(
        id: ResolutionArtifactId,
        source: ResolutionDecisionSource,
        provenance: ResolutionSourceProvenance,
        source_bytes: Vec<u8>,
        field_paths: BTreeSet<String>,
        parser: &'a dyn ResolutionSourceParser,
    ) -> Result<Self, VNextError> {
        validate_resolution_source_availability(&source_bytes, &provenance, &field_paths)?;
        Ok(Self {
            id,
            source,
            provenance,
            source_bytes,
            field_paths,
            parser,
        })
    }

    /// Runs the same deterministic parser verification used when constructing
    /// or revalidating a resolved plan. Callers may use this for an explicit
    /// preflight, but plan validation never relies on a prior call.
    pub fn validate(&self) -> Result<(), VNextError> {
        self.verify().map(drop)
    }

    fn verify(&self) -> Result<ResolutionSourceArtifact, VNextError> {
        validate_resolution_source_availability(
            &self.source_bytes,
            &self.provenance,
            &self.field_paths,
        )?;
        let parser = self.parser.descriptor()?;
        parser.validate()?;
        let first_document =
            self.parser
                .parse(self.source, &self.provenance, &self.source_bytes)?;
        let (first_document, first_fingerprint) = validate_and_canonicalize_resolution_json(
            first_document,
            "fingerprint first parsed resolution source document",
        )?;
        let parser_after_first_parse = self.parser.descriptor()?;
        parser_after_first_parse.validate()?;
        let second_document =
            self.parser
                .parse(self.source, &self.provenance, &self.source_bytes)?;
        let (second_document, second_fingerprint) = validate_and_canonicalize_resolution_json(
            second_document,
            "fingerprint second parsed resolution source document",
        )?;
        let parser_after_second_parse = self.parser.descriptor()?;
        parser_after_second_parse.validate()?;
        if parser != parser_after_first_parse
            || parser != parser_after_second_parse
            || first_fingerprint != second_fingerprint
            || first_document != second_document
        {
            return Err(invalid_plan(
                "resolution_source_evidence.parser",
                "parser identity changed or repeated parsing of identical bytes was nondeterministic",
            ));
        }
        ResolutionSourceArtifact::from_verified_document(
            self.id.clone(),
            self.source,
            self.provenance.clone(),
            &self.source_bytes,
            parser,
            &first_document,
            first_fingerprint,
            self.field_paths.clone(),
        )
    }

    pub fn id(&self) -> &ResolutionArtifactId {
        &self.id
    }

    pub const fn source(&self) -> ResolutionDecisionSource {
        self.source
    }

    pub fn provenance(&self) -> &ResolutionSourceProvenance {
        &self.provenance
    }

    pub fn locator(&self) -> &str {
        self.provenance.locator()
    }

    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }

    pub fn field_paths(&self) -> &BTreeSet<String> {
        &self.field_paths
    }
}

impl ResolutionSourceArtifact {
    fn from_verified_document(
        id: ResolutionArtifactId,
        source: ResolutionDecisionSource,
        provenance: ResolutionSourceProvenance,
        bytes: &[u8],
        parser: ResolutionParserDescriptor,
        document: &serde_json::Value,
        canonical_document_fingerprint: String,
        field_paths: BTreeSet<String>,
    ) -> Result<Self, VNextError> {
        validate_resolution_source_availability(bytes, &provenance, &field_paths)?;
        let content_size_bytes = u64::try_from(bytes.len()).map_err(|_| {
            invalid_plan(
                "resolution_source_artifact.content_size_bytes",
                "source byte length does not fit u64",
            )
        })?;
        let fields = field_paths
            .into_iter()
            .map(|path| {
                let value = document.pointer(&path).ok_or_else(|| {
                    invalid_plan(
                        "resolution_source_artifact.fields",
                        format!("JSON pointer `{path}` is absent from the source document"),
                    )
                })?;
                ResolutionFingerprint::new(canonical_value_fingerprint(
                    value,
                    "fingerprint resolution source field",
                )?)
                .map(|fingerprint| (path, fingerprint))
            })
            .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
        Ok(Self {
            id,
            source,
            provenance,
            parser,
            content_size_bytes,
            content_fingerprint: ResolutionFingerprint::new(format!(
                "{:x}",
                Sha256::digest(bytes)
            ))?,
            canonical_document_fingerprint: ResolutionFingerprint::new(
                canonical_document_fingerprint,
            )?,
            fields,
        })
    }

    pub fn id(&self) -> &ResolutionArtifactId {
        &self.id
    }

    pub fn source(&self) -> ResolutionDecisionSource {
        self.source
    }

    pub fn provenance(&self) -> &ResolutionSourceProvenance {
        &self.provenance
    }

    pub fn locator(&self) -> &str {
        self.provenance.locator()
    }

    pub fn content_size_bytes(&self) -> u64 {
        self.content_size_bytes
    }

    pub fn content_fingerprint(&self) -> &ResolutionFingerprint {
        &self.content_fingerprint
    }

    pub fn parser(&self) -> &ResolutionParserDescriptor {
        &self.parser
    }

    pub fn canonical_document_fingerprint(&self) -> &ResolutionFingerprint {
        &self.canonical_document_fingerprint
    }

    pub fn fields(&self) -> &BTreeMap<String, ResolutionFingerprint> {
        &self.fields
    }
}

impl UnvalidatedResolutionSourceArtifact {
    fn revalidate(
        self,
        expected: &ResolutionSourceArtifact,
    ) -> Result<ResolutionSourceArtifact, VNextError> {
        if self.id != expected.id
            || self.source != expected.source
            || self.provenance != expected.provenance
            || self.parser != expected.parser
            || self.content_size_bytes != expected.content_size_bytes
            || self.content_fingerprint != expected.content_fingerprint
            || self.canonical_document_fingerprint != expected.canonical_document_fingerprint
            || self.fields != expected.fields
        {
            return Err(invalid_plan(
                "source_artifacts",
                format!(
                    "serialized source artifact `{}` differs from externally verified evidence",
                    self.id
                ),
            ));
        }
        Ok(expected.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolutionDecisionEvidence {
    source_artifact_id: ResolutionArtifactId,
    source_field_path: String,
    chosen_value_fingerprint: ResolutionFingerprint,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionDecisionEvidenceWire {
    source_artifact_id: ResolutionArtifactId,
    source_field_path: String,
    chosen_value_fingerprint: ResolutionFingerprint,
}

impl<'de> Deserialize<'de> for ResolutionDecisionEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolutionDecisionEvidenceWire::deserialize(deserializer)?;
        Self::new(
            wire.source_artifact_id,
            wire.source_field_path,
            wire.chosen_value_fingerprint,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolutionDecisionEvidence {
    fn new(
        source_artifact_id: ResolutionArtifactId,
        source_field_path: impl Into<String>,
        chosen_value_fingerprint: ResolutionFingerprint,
    ) -> Result<Self, VNextError> {
        let source_field_path = source_field_path.into();
        if source_field_path.is_empty()
            || source_field_path.len() > 512
            || source_field_path.trim() != source_field_path
            || source_field_path
                .bytes()
                .any(|byte| byte.is_ascii_control())
        {
            return Err(invalid_plan(
                "decisions.evidence.source_field_path",
                "must be a non-empty portable path of at most 512 bytes",
            ));
        }
        Ok(Self {
            source_artifact_id,
            source_field_path,
            chosen_value_fingerprint,
        })
    }

    pub fn source_artifact_id(&self) -> &ResolutionArtifactId {
        &self.source_artifact_id
    }

    pub fn source_field_path(&self) -> &str {
        &self.source_field_path
    }

    pub fn chosen_value_fingerprint(&self) -> &ResolutionFingerprint {
        &self.chosen_value_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolutionDecision {
    field: ResolutionField,
    source: ResolutionDecisionSource,
    reason_id: ResolutionReasonId,
    evidence: ResolutionDecisionEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionDecisionWire {
    field: ResolutionField,
    source: ResolutionDecisionSource,
    reason_id: ResolutionReasonId,
    evidence: ResolutionDecisionEvidence,
}

impl<'de> Deserialize<'de> for ResolutionDecision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolutionDecisionWire::deserialize(deserializer)?;
        Ok(Self::new(
            wire.field,
            wire.source,
            wire.reason_id,
            wire.evidence,
        ))
    }
}

impl ResolutionDecision {
    fn new(
        field: ResolutionField,
        source: ResolutionDecisionSource,
        reason_id: ResolutionReasonId,
        evidence: ResolutionDecisionEvidence,
    ) -> Self {
        Self {
            field,
            source,
            reason_id,
            evidence,
        }
    }

    pub fn field(&self) -> ResolutionField {
        self.field
    }

    pub fn source(&self) -> ResolutionDecisionSource {
        self.source
    }

    pub fn reason_id(&self) -> &ResolutionReasonId {
        &self.reason_id
    }

    pub fn evidence(&self) -> &ResolutionDecisionEvidence {
        &self.evidence
    }
}

/// Construction-time link from a resolved field to externally supplied source
/// evidence. It intentionally carries no chosen-value fingerprint; core derives
/// that fingerprint independently from both sides after parsing raw evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionDecisionBinding {
    field: ResolutionField,
    source: ResolutionDecisionSource,
    reason_id: ResolutionReasonId,
    source_artifact_id: ResolutionArtifactId,
    source_field_path: String,
}

impl ResolutionDecisionBinding {
    pub fn new(
        field: ResolutionField,
        source: ResolutionDecisionSource,
        reason_id: ResolutionReasonId,
        source_artifact_id: ResolutionArtifactId,
        source_field_path: impl Into<String>,
    ) -> Result<Self, VNextError> {
        let source_field_path = source_field_path.into();
        if !field.accepts_source(source) {
            return Err(invalid_plan(
                "decision_bindings.source",
                format!("source `{source:?}` cannot author field `{field:?}`"),
            ));
        }
        if !source_field_path.starts_with('/')
            || source_field_path.len() > 512
            || source_field_path.trim() != source_field_path
            || source_field_path
                .bytes()
                .any(|byte| byte.is_ascii_control())
        {
            return Err(invalid_plan(
                "decision_bindings.source_field_path",
                "must be a portable JSON pointer of at most 512 bytes",
            ));
        }
        Ok(Self {
            field,
            source,
            reason_id,
            source_artifact_id,
            source_field_path,
        })
    }

    pub fn field(&self) -> ResolutionField {
        self.field
    }

    pub fn source(&self) -> ResolutionDecisionSource {
        self.source
    }

    pub fn reason_id(&self) -> &ResolutionReasonId {
        &self.reason_id
    }

    pub fn source_artifact_id(&self) -> &ResolutionArtifactId {
        &self.source_artifact_id
    }

    pub fn source_field_path(&self) -> &str {
        &self.source_field_path
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelPlanInputs {
    pub original_source: OriginalModelSource,
    pub resolved_source: ResolvedModelSource,
    pub config: ModelConfigFingerprint,
    pub external_metadata_id: super::ExternalModelMetadataId,
    pub prepared_family: PreparedModelFamily,
    pub tokenizer: TokenizerDescriptor,
    pub device: DeviceDescriptor,
    pub capabilities: CapabilityCatalog,
    pub runtime: ResolvedRuntimePolicy,
    pub engine: EngineSelection,
    pub execution_plan: ExecutionPlan,
    pub sampling: SamplingPolicy,
    pub stop: StopPolicy,
    pub structured_output: StructuredOutputPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedModelPlanParts {
    source_artifacts: Vec<ResolutionSourceArtifact>,
    pub original_source: OriginalModelSource,
    pub resolved_source: ResolvedModelSource,
    pub config: ModelConfigFingerprint,
    pub external_metadata_id: super::ExternalModelMetadataId,
    pub prepared_family: PreparedModelFamily,
    pub tokenizer: TokenizerDescriptor,
    pub device: DeviceDescriptor,
    pub capabilities: CapabilityCatalog,
    pub runtime: ResolvedRuntimePolicy,
    pub engine: EngineSelection,
    pub execution_plan: ExecutionPlan,
    pub sampling: SamplingPolicy,
    pub stop: StopPolicy,
    pub structured_output: StructuredOutputPolicy,
    decisions: Vec<ResolutionDecision>,
}

impl ResolvedModelPlanParts {
    pub fn source_artifacts(&self) -> &[ResolutionSourceArtifact] {
        &self.source_artifacts
    }

    pub fn decisions(&self) -> &[ResolutionDecision] {
        &self.decisions
    }
}

/// Trusted inputs that are intentionally outside a serialized resolved plan.
/// A wire payload cannot choose its model registry, source evidence, physical
/// node bindings, provider preference, or resource estimate and then validate
/// itself against those same values.
pub struct ResolvedPlanValidationContext<'a> {
    registry: &'a dyn ModelFamilyRegistry,
    source_evidence: &'a [ResolutionSourceEvidence<'a>],
    node_resolutions: &'a [PlanNodeResolution],
    device: &'a DeviceDescriptor,
    capabilities: &'a CapabilityCatalog,
    runtime: &'a ResolvedRuntimePolicy,
}

impl<'a> ResolvedPlanValidationContext<'a> {
    pub fn new(
        registry: &'a dyn ModelFamilyRegistry,
        source_evidence: &'a [ResolutionSourceEvidence<'a>],
        node_resolutions: &'a [PlanNodeResolution],
        device: &'a DeviceDescriptor,
        capabilities: &'a CapabilityCatalog,
        runtime: &'a ResolvedRuntimePolicy,
    ) -> Self {
        Self {
            registry,
            source_evidence,
            node_resolutions,
            device,
            capabilities,
            runtime,
        }
    }

    pub fn registry(&self) -> &dyn ModelFamilyRegistry {
        self.registry
    }

    fn verify_source_artifacts(&self) -> Result<Vec<ResolutionSourceArtifact>, VNextError> {
        self.source_evidence
            .iter()
            .map(ResolutionSourceEvidence::verify)
            .collect()
    }

    pub fn node_resolutions(&self) -> &[PlanNodeResolution] {
        self.node_resolutions
    }

    pub fn device(&self) -> &DeviceDescriptor {
        self.device
    }

    pub fn capabilities(&self) -> &CapabilityCatalog {
        self.capabilities
    }

    pub fn runtime(&self) -> &ResolvedRuntimePolicy {
        self.runtime
    }
}

/// The single validated, data-only result consumed by a product entrypoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedModelPlan {
    parts: ResolvedModelPlanParts,
    fingerprint: ResolutionFingerprint,
}

impl ExecutablePlanView for ResolvedModelPlan {
    fn execution_plan(&self) -> &ExecutionPlan {
        &self.parts.execution_plan
    }

    fn device(&self) -> &DeviceDescriptor {
        &self.parts.device
    }

    fn capabilities(&self) -> &CapabilityCatalog {
        &self.parts.capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResolvedModelPlanParts {
    source_artifacts: Vec<UnvalidatedResolutionSourceArtifact>,
    original_source: OriginalModelSource,
    resolved_source: ResolvedModelSource,
    config: ModelConfigFingerprint,
    external_metadata_id: super::ExternalModelMetadataId,
    prepared_family: PreparedModelFamilyWire,
    tokenizer: TokenizerDescriptor,
    device: DeviceDescriptor,
    capabilities: CapabilityCatalog,
    runtime: ResolvedRuntimePolicy,
    engine: EngineSelection,
    execution_plan: UnvalidatedExecutionPlanWire,
    sampling: SamplingPolicy,
    stop: StopPolicy,
    structured_output: StructuredOutputPolicy,
    decisions: Vec<ResolutionDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedResolvedModelPlan {
    parts: UnvalidatedResolvedModelPlanParts,
    fingerprint: ResolutionFingerprint,
}

#[derive(Serialize)]
struct ResolvedModelPlanWire {
    parts: UnvalidatedResolvedModelPlanParts,
    fingerprint: ResolutionFingerprint,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ResolvedModelPlanWireFields {
    parts: UnvalidatedResolvedModelPlanParts,
    fingerprint: ResolutionFingerprint,
}

impl<'de> Deserialize<'de> for ResolvedModelPlanWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let fields =
            ResolvedModelPlanWireFields::deserialize(&raw).map_err(serde::de::Error::custom)?;
        let canonical = serde_json::to_value(&fields).map_err(serde::de::Error::custom)?;
        if canonical != raw {
            return Err(serde::de::Error::custom(
                "resolved model plan wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(Self {
            parts: fields.parts,
            fingerprint: fields.fingerprint,
        })
    }
}

impl From<ResolvedModelPlanWire> for UnvalidatedResolvedModelPlan {
    fn from(wire: ResolvedModelPlanWire) -> Self {
        Self {
            parts: wire.parts,
            fingerprint: wire.fingerprint,
        }
    }
}

impl UnvalidatedResolvedModelPlan {
    pub fn revalidate(
        self,
        context: &ResolvedPlanValidationContext<'_>,
    ) -> Result<ResolvedModelPlan, VNextError> {
        let UnvalidatedResolvedModelPlanParts {
            source_artifacts,
            original_source,
            resolved_source,
            config,
            external_metadata_id,
            prepared_family,
            tokenizer,
            device,
            capabilities,
            runtime,
            engine,
            execution_plan,
            sampling,
            stop,
            structured_output,
            decisions,
        } = self.parts;
        let verified_source_artifacts = context.verify_source_artifacts()?;
        let source_artifacts =
            Self::revalidate_source_artifacts(source_artifacts, &verified_source_artifacts)?;
        let prepared_family =
            UnvalidatedPreparedModelFamily::from(prepared_family).revalidate(context.registry())?;
        if device != *context.device()
            || capabilities != *context.capabilities()
            || runtime != *context.runtime()
        {
            return Err(invalid_plan(
                "validation_context",
                "serialized device, capability catalog, or runtime policy differs from external trusted inputs",
            ));
        }
        let execution_plan = UnvalidatedExecutionPlan::from(execution_plan).revalidate(
            &prepared_family,
            context.capabilities(),
            context.runtime(),
            context.node_resolutions().to_vec(),
        )?;
        let serialized_decisions = decisions;
        let decision_bindings = serialized_decisions
            .iter()
            .map(|decision| {
                ResolutionDecisionBinding::new(
                    decision.field,
                    decision.source,
                    decision.reason_id.clone(),
                    decision.evidence.source_artifact_id.clone(),
                    decision.evidence.source_field_path.clone(),
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let rebuilt = ResolvedModelPlan::from_verified_inputs(
            ResolvedModelPlanInputs {
                original_source,
                resolved_source,
                config,
                external_metadata_id,
                prepared_family,
                tokenizer,
                device: context.device().clone(),
                capabilities: context.capabilities().clone(),
                runtime: context.runtime().clone(),
                engine,
                execution_plan,
                sampling,
                stop,
                structured_output,
            },
            decision_bindings,
            source_artifacts,
            context,
        )?;
        if rebuilt.parts.decisions != serialized_decisions {
            return Err(invalid_plan(
                "decisions",
                "serialized decisions differ from decisions rebuilt from external raw evidence",
            ));
        }
        if rebuilt.fingerprint != self.fingerprint {
            return Err(invalid_plan(
                "fingerprint",
                format!(
                    "does not match typed reconstruction: expected `{}`, actual `{}`",
                    rebuilt.fingerprint, self.fingerprint
                ),
            ));
        }
        Ok(rebuilt)
    }

    fn revalidate_source_artifacts(
        serialized: Vec<UnvalidatedResolutionSourceArtifact>,
        expected: &[ResolutionSourceArtifact],
    ) -> Result<Vec<ResolutionSourceArtifact>, VNextError> {
        let mut expected_by_id = BTreeMap::new();
        for artifact in expected {
            if expected_by_id
                .insert(artifact.id().clone(), artifact)
                .is_some()
            {
                return Err(invalid_plan(
                    "validation_context.source_artifacts",
                    format!("duplicate externally verified artifact `{}`", artifact.id()),
                ));
            }
        }
        if serialized.len() != expected_by_id.len() {
            return Err(invalid_plan(
                "source_artifacts",
                "serialized and externally verified source artifact sets differ",
            ));
        }

        let mut seen = BTreeSet::new();
        let mut validated = Vec::with_capacity(serialized.len());
        for artifact in serialized {
            if !seen.insert(artifact.id.clone()) {
                return Err(invalid_plan(
                    "source_artifacts",
                    format!("duplicate serialized artifact `{}`", artifact.id),
                ));
            }
            let expected = expected_by_id.get(&artifact.id).ok_or_else(|| {
                invalid_plan(
                    "source_artifacts",
                    format!(
                        "serialized artifact `{}` has no externally verified evidence",
                        artifact.id
                    ),
                )
            })?;
            validated.push(artifact.revalidate(expected)?);
        }
        Ok(validated)
    }
}

impl ResolvedModelPlan {
    pub fn new(
        inputs: ResolvedModelPlanInputs,
        decision_bindings: Vec<ResolutionDecisionBinding>,
        context: &ResolvedPlanValidationContext<'_>,
    ) -> Result<Self, VNextError> {
        // Raw source bytes are parsed before any trusted plan parts or
        // decisions exist. This is the only public construction path.
        let source_artifacts = context.verify_source_artifacts()?;
        Self::from_verified_inputs(inputs, decision_bindings, source_artifacts, context)
    }

    fn from_verified_inputs(
        inputs: ResolvedModelPlanInputs,
        decision_bindings: Vec<ResolutionDecisionBinding>,
        source_artifacts: Vec<ResolutionSourceArtifact>,
        context: &ResolvedPlanValidationContext<'_>,
    ) -> Result<Self, VNextError> {
        Self::validate_external_inputs(&inputs, context)?;
        let ResolvedModelPlanInputs {
            original_source,
            resolved_source,
            config,
            external_metadata_id,
            prepared_family,
            tokenizer,
            device,
            capabilities,
            runtime,
            engine,
            execution_plan,
            sampling,
            stop,
            structured_output,
        } = inputs;
        let mut parts = ResolvedModelPlanParts {
            source_artifacts,
            original_source,
            resolved_source,
            config,
            external_metadata_id,
            prepared_family,
            tokenizer,
            device,
            capabilities,
            runtime,
            engine,
            execution_plan,
            sampling,
            stop,
            structured_output,
            decisions: Vec::new(),
        };
        Self::normalize(&mut parts);
        parts.decisions = Self::bind_decisions(&parts, decision_bindings)?;
        Self::normalize(&mut parts);
        Self::validate(&parts, context.node_resolutions())?;
        let fingerprint = ResolutionFingerprint::new(canonical_fingerprint(
            &parts,
            "serialize resolved model plan",
        )?)?;
        Ok(Self { parts, fingerprint })
    }

    fn validate_external_inputs(
        inputs: &ResolvedModelPlanInputs,
        context: &ResolvedPlanValidationContext<'_>,
    ) -> Result<(), VNextError> {
        let family_registration = context
            .registry()
            .resolve(inputs.prepared_family.family_id())?;
        let metadata_registration = context
            .registry()
            .resolve_external(&inputs.external_metadata_id)?;
        if !family_registration
            .external_metadata_ids()
            .contains(&inputs.external_metadata_id)
            || !std::ptr::eq(family_registration, metadata_registration)
            || inputs.prepared_family.external_metadata_id() != &inputs.external_metadata_id
        {
            return Err(invalid_plan(
                "external_metadata_id",
                format!(
                    "external metadata `{}`, prepared identity `{}`, and internal family `{}` must identify the same registration and exact typed configuration",
                    inputs.external_metadata_id,
                    inputs.prepared_family.external_metadata_id(),
                    inputs.prepared_family.family_id(),
                ),
            ));
        }
        let externally_prepared =
            family_registration.prepare(inputs.prepared_family.canonical_config())?;
        if externally_prepared != inputs.prepared_family {
            return Err(invalid_plan(
                "validation_context.model_registry",
                "prepared model family differs from external typed registry reconstruction",
            ));
        }
        if inputs.device != *context.device()
            || inputs.capabilities != *context.capabilities()
            || inputs.runtime != *context.runtime()
        {
            return Err(invalid_plan(
                "validation_context",
                "device, capability catalog, or runtime policy differs from external trusted inputs",
            ));
        }
        Ok(())
    }

    fn bind_decisions(
        parts: &ResolvedModelPlanParts,
        bindings: Vec<ResolutionDecisionBinding>,
    ) -> Result<Vec<ResolutionDecision>, VNextError> {
        let expected_fingerprints = Self::decision_fingerprints(parts)?;
        let mut artifacts = BTreeMap::new();
        for artifact in &parts.source_artifacts {
            if artifacts.insert(artifact.id.clone(), artifact).is_some() {
                return Err(invalid_plan(
                    "source_artifacts",
                    format!("duplicate source artifact `{}`", artifact.id),
                ));
            }
        }
        let mut fields = BTreeSet::new();
        let mut used_artifacts = BTreeSet::new();
        let mut used_artifact_fields = BTreeSet::new();
        let mut decisions = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if !fields.insert(binding.field) {
                return Err(invalid_plan(
                    "decision_bindings",
                    format!("field `{:?}` has more than one binding", binding.field),
                ));
            }
            if !binding.field.accepts_source(binding.source) {
                return Err(invalid_plan(
                    "decision_bindings.source",
                    format!(
                        "source `{:?}` cannot author field `{:?}`",
                        binding.source, binding.field
                    ),
                ));
            }
            let expected = expected_fingerprints.get(&binding.field).ok_or_else(|| {
                invalid_plan(
                    "decision_bindings.field",
                    format!("field `{:?}` is not resolvable", binding.field),
                )
            })?;
            let artifact = artifacts.get(&binding.source_artifact_id).ok_or_else(|| {
                invalid_plan(
                    "decision_bindings.source_artifact_id",
                    format!(
                        "binding for `{:?}` references unknown source artifact `{}`",
                        binding.field, binding.source_artifact_id
                    ),
                )
            })?;
            if artifact.source != binding.source {
                return Err(invalid_plan(
                    "decision_bindings.source",
                    format!(
                        "binding for `{:?}` source differs from artifact `{}`",
                        binding.field, artifact.id
                    ),
                ));
            }
            let source_fingerprint =
                artifact
                    .fields
                    .get(&binding.source_field_path)
                    .ok_or_else(|| {
                        invalid_plan(
                            "decision_bindings.source_field_path",
                            format!(
                                "binding for `{:?}` references a field absent from artifact `{}`",
                                binding.field, artifact.id
                            ),
                        )
                    })?;
            if source_fingerprint.as_str() != expected {
                return Err(invalid_plan(
                    "decision_bindings.source_field_path",
                    format!(
                        "externally parsed field for `{:?}` differs from the resolved value",
                        binding.field
                    ),
                ));
            }
            used_artifacts.insert(artifact.id.clone());
            if !used_artifact_fields
                .insert((artifact.id.clone(), binding.source_field_path.clone()))
            {
                return Err(invalid_plan(
                    "decision_bindings.source_field_path",
                    "one parsed artifact field cannot issue more than one decision",
                ));
            }
            decisions.push(ResolutionDecision::new(
                binding.field,
                binding.source,
                binding.reason_id,
                ResolutionDecisionEvidence::new(
                    binding.source_artifact_id,
                    binding.source_field_path,
                    source_fingerprint.clone(),
                )?,
            ));
        }
        if fields
            != expected_fingerprints
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
        {
            return Err(invalid_plan(
                "decision_bindings",
                "every resolved field requires exactly one external evidence binding",
            ));
        }
        if used_artifacts != artifacts.keys().cloned().collect::<BTreeSet<_>>() {
            return Err(invalid_plan(
                "source_artifacts",
                "externally verified source evidence and referenced binding sets differ",
            ));
        }
        let available_artifact_fields = artifacts
            .values()
            .flat_map(|artifact| {
                artifact
                    .fields
                    .keys()
                    .cloned()
                    .map(|path| (artifact.id.clone(), path))
            })
            .collect::<BTreeSet<_>>();
        if used_artifact_fields != available_artifact_fields {
            return Err(invalid_plan(
                "source_artifacts.fields",
                "externally parsed source fields and decision binding fields differ",
            ));
        }
        Ok(decisions)
    }

    fn normalize(parts: &mut ResolvedModelPlanParts) {
        parts
            .source_artifacts
            .sort_by(|left, right| left.id.cmp(&right.id));
        parts
            .resolved_source
            .files
            .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        parts.stop.strings.sort();
        parts.decisions.sort_by_key(|decision| decision.field);
    }

    fn validate(
        parts: &ResolvedModelPlanParts,
        node_resolutions: &[PlanNodeResolution],
    ) -> Result<(), VNextError> {
        let mut source_artifacts = BTreeMap::new();
        for artifact in &parts.source_artifacts {
            if source_artifacts
                .insert(artifact.id.clone(), artifact)
                .is_some()
            {
                return Err(invalid_plan(
                    "source_artifacts",
                    format!("duplicate source artifact `{}`", artifact.id),
                ));
            }
            artifact.provenance.validate()?;
            match &artifact.provenance {
                ResolutionSourceProvenance::LockedModelFile { relative_path } => {
                    let Some(file) = parts
                        .resolved_source
                        .files
                        .iter()
                        .find(|file| &file.relative_path == relative_path)
                    else {
                        return Err(invalid_plan(
                            "source_artifacts.provenance",
                            format!(
                                "artifact `{}` does not resolve to locked file `{relative_path}`",
                                artifact.id
                            ),
                        ));
                    };
                    if file.sha256 != artifact.content_fingerprint.as_str()
                        || file.size_bytes != artifact.content_size_bytes
                    {
                        return Err(invalid_plan(
                            "source_artifacts.provenance",
                            format!(
                                "artifact `{}` bytes differ from locked file `{relative_path}` SHA or size",
                                artifact.id
                            ),
                        ));
                    }
                }
                ResolutionSourceProvenance::Upstream { .. }
                    if artifact.source == ResolutionDecisionSource::ModelMetadata =>
                {
                    return Err(invalid_plan(
                        "source_artifacts.provenance",
                        format!(
                            "model-metadata artifact `{}` must bind a locked model file",
                            artifact.id
                        ),
                    ));
                }
                ResolutionSourceProvenance::Upstream { .. } => {}
            }
        }
        if parts.original_source.location.trim().is_empty()
            || matches!(
                parts.original_source.requested_revision.as_deref(),
                Some(revision) if revision.trim().is_empty()
            )
            || parts.resolved_source.canonical_location.trim().is_empty()
            || parts.resolved_source.resolved_revision.trim().is_empty()
        {
            return Err(invalid_plan(
                "source",
                "source locations and revisions must be non-empty",
            ));
        }
        if parts.resolved_source.files.is_empty() {
            return Err(invalid_plan(
                "resolved_source.files",
                "at least one fingerprinted file is required",
            ));
        }
        let mut paths = BTreeSet::new();
        if parts.resolved_source.files.iter().any(|file| {
            !validate_source_path(&file.relative_path)
                || file.size_bytes == 0
                || !is_canonical_sha256(&file.sha256)
                || !paths.insert(file.relative_path.clone())
        }) {
            return Err(invalid_plan(
                "resolved_source.files",
                "file paths, sizes, and canonical hashes must be valid and unique",
            ));
        }
        let family = &parts.prepared_family;
        let program_fingerprint = family.program().fingerprint()?;
        let family_fingerprint = family.fingerprint()?;
        if parts.tokenizer.vocabulary_size == 0
            || family.metadata().template.template.trim().is_empty()
            || family.metadata().special_tokens.eos_token_ids.is_empty()
        {
            return Err(invalid_plan(
                "model_metadata",
                "program, tokenizer, template, and special-token metadata are incomplete",
            ));
        }
        Self::validate_source_file_binding(
            &parts.resolved_source,
            &parts.config.source_file,
            &parts.config.sha256,
            "config",
        )?;
        if parts.config.typed_config_sha256 != family.config_fingerprint() {
            return Err(invalid_plan(
                "config.typed_config_sha256",
                "does not match the typed model family configuration",
            ));
        }
        Self::validate_source_file_binding(
            &parts.resolved_source,
            &parts.tokenizer.source_file,
            &parts.tokenizer.sha256,
            "tokenizer",
        )?;
        Self::validate_source_file_binding(
            &parts.resolved_source,
            &family.metadata().template.source_file,
            &family.metadata().template.sha256,
            "template",
        )?;
        Self::validate_token_contract(parts)?;
        parts.runtime.validate()?;
        parts.execution_plan.validate_against(
            family,
            &parts.capabilities,
            &parts.runtime,
            node_resolutions,
        )?;
        let runtime_fingerprint = super::canonical_runtime_policy_fingerprint(&parts.runtime)?;
        let capability_fingerprint = parts.capabilities.fingerprint()?;
        if &parts.device != parts.capabilities.device()
            || &parts.device.id != parts.execution_plan.payload().device_id()
            || family.family_id() != parts.execution_plan.payload().family_id()
            || family_fingerprint != parts.execution_plan.payload().prepared_family_fingerprint()
            || runtime_fingerprint != parts.execution_plan.payload().policy_fingerprint()
            || program_fingerprint != parts.execution_plan.payload().program_fingerprint()
            || &family.weight_schema().format_id != parts.execution_plan.payload().weight_format()
            || family.weight_schema().quantization_formats()
                != *parts.execution_plan.payload().quantization_formats()
            || capability_fingerprint
                != parts
                    .execution_plan
                    .payload()
                    .capability_catalog_fingerprint()
        {
            return Err(invalid_plan(
                "resolved_contract_links",
                "family, full device descriptor, policy, capability, engine, and plan links disagree",
            ));
        }
        let engine = parts
            .capabilities
            .engine_provider(&parts.engine.provider_id, parts.engine.contract_version)?;
        if !is_canonical_sha256(&parts.engine.implementation_fingerprint)
            || engine.contract_version() != parts.engine.contract_version
            || engine.implementation_fingerprint() != parts.engine.implementation_fingerprint
            || engine.device_id() != &parts.device.id
        {
            return Err(invalid_plan(
                "engine",
                "engine provider version and implementation are not exactly bound to the resolved device",
            ));
        }
        for node in parts.execution_plan.payload().nodes() {
            let providers = parts.capabilities.providers_for(node.operation_id())?;
            let selected = providers
                .iter()
                .find(|provider| provider.provider_id() == node.selection().selected_provider())
                .ok_or_else(|| VNextError::UnsupportedOperation {
                    node_id: Some(node.id().to_string()),
                    operation_id: node.operation_id().to_string(),
                    device_id: parts.device.id.to_string(),
                    reason: format!(
                        "selected provider `{}` is absent from the resolved catalog",
                        node.selection().selected_provider()
                    ),
                })?;
            if !selected.version().satisfies(node.operation_version()) {
                return Err(VNextError::IncompatibleOperationVersion {
                    node_id: Some(node.id().to_string()),
                    operation_id: node.operation_id().to_string(),
                    required_major: node.operation_version().major,
                    required_minor: node.operation_version().minor,
                    available_major: selected.version().major,
                    available_minor: selected.version().minor,
                });
            }
            // Per-node weight/quantization requirements are derived from that
            // node's bound semantic values by ExecutionPlan::validate_against.
            // Applying the family-wide format to every operation would reject
            // valid mixed-format programs and recreate a model-level shortcut.
        }
        parts.sampling.validate()?;
        if parts.stop.maximum_output_tokens == 0
            || parts.stop.strings.iter().any(|stop| stop.is_empty())
            || parts.stop.strings.windows(2).any(|pair| pair[0] == pair[1])
            || matches!(
                &parts.structured_output,
                StructuredOutputPolicy::JsonSchema { schema_sha256 }
                    if !is_canonical_sha256(schema_sha256)
            )
        {
            return Err(invalid_plan(
                "generation_policy",
                "stop values must be non-empty and unique and structured-output hashes must be canonical",
            ));
        }
        let expected_fingerprints = Self::decision_fingerprints(parts)?;
        let mut actual_decisions = BTreeSet::new();
        let mut used_source_artifacts = BTreeSet::new();
        let mut used_source_fields = BTreeSet::new();
        for decision in &parts.decisions {
            if !actual_decisions.insert(decision.field) {
                return Err(invalid_plan(
                    "decisions",
                    format!("field `{:?}` has more than one decision", decision.field),
                ));
            }
            let expected = expected_fingerprints.get(&decision.field).ok_or_else(|| {
                invalid_plan(
                    "decisions",
                    format!("field `{:?}` is not resolvable", decision.field),
                )
            })?;
            if !decision.field.accepts_source(decision.source) {
                return Err(invalid_plan(
                    "decisions.source",
                    format!(
                        "source `{:?}` is not allowed to author field `{:?}`",
                        decision.source, decision.field
                    ),
                ));
            }
            if decision.evidence.chosen_value_fingerprint.as_str() != expected {
                return Err(invalid_plan(
                    format!("decisions.{:?}.chosen_value_fingerprint", decision.field),
                    format!(
                        "does not match the resolved value: expected `{expected}`, actual `{}`",
                        decision.evidence.chosen_value_fingerprint
                    ),
                ));
            }
            let artifact = source_artifacts
                .get(decision.evidence.source_artifact_id())
                .ok_or_else(|| {
                    invalid_plan(
                        "decisions.evidence.source_artifact_id",
                        format!(
                            "decision for `{:?}` references unknown source artifact `{}`",
                            decision.field,
                            decision.evidence.source_artifact_id()
                        ),
                    )
                })?;
            used_source_artifacts.insert(artifact.id.clone());
            if !used_source_fields.insert((
                artifact.id.clone(),
                decision.evidence.source_field_path.clone(),
            )) {
                return Err(invalid_plan(
                    "decisions.evidence.source_field_path",
                    "one parsed artifact field cannot issue more than one decision",
                ));
            }
            if artifact.source != decision.source {
                return Err(invalid_plan(
                    "decisions.source",
                    format!(
                        "decision for `{:?}` source differs from artifact `{}`",
                        decision.field, artifact.id
                    ),
                ));
            }
            let source_field = artifact
                .fields
                .get(decision.evidence.source_field_path())
                .ok_or_else(|| {
                    invalid_plan(
                        "decisions.evidence.source_field_path",
                        format!(
                            "decision for `{:?}` references a field absent from artifact `{}`",
                            decision.field, artifact.id
                        ),
                    )
                })?;
            if source_field != &decision.evidence.chosen_value_fingerprint {
                return Err(invalid_plan(
                    "decisions.evidence.chosen_value_fingerprint",
                    format!(
                        "decision for `{:?}` differs from source artifact field `{}`",
                        decision.field,
                        decision.evidence.source_field_path()
                    ),
                ));
            }
        }
        let required_decisions = expected_fingerprints
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        if actual_decisions != required_decisions {
            return Err(invalid_plan(
                "decisions",
                "every resolved field requires exactly one typed, fingerprinted decision",
            ));
        }
        if used_source_artifacts != source_artifacts.keys().cloned().collect::<BTreeSet<_>>() {
            return Err(invalid_plan(
                "source_artifacts",
                "every source artifact must be referenced by at least one resolution decision",
            ));
        }
        let available_source_fields = source_artifacts
            .values()
            .flat_map(|artifact| {
                artifact
                    .fields
                    .keys()
                    .cloned()
                    .map(|path| (artifact.id.clone(), path))
            })
            .collect::<BTreeSet<_>>();
        if used_source_fields != available_source_fields {
            return Err(invalid_plan(
                "source_artifacts.fields",
                "parsed source fields and resolution decision fields differ",
            ));
        }
        Ok(())
    }

    fn validate_source_file_binding(
        source: &ResolvedModelSource,
        source_file: &str,
        sha256: &str,
        field: &str,
    ) -> Result<(), VNextError> {
        if !validate_source_path(source_file) || !is_canonical_sha256(sha256) {
            return Err(invalid_plan(
                format!("{field}.source_file"),
                "source path or hash is not canonical",
            ));
        }
        match source
            .files
            .iter()
            .find(|file| file.relative_path == source_file)
        {
            Some(file) if file.sha256 == sha256 => Ok(()),
            Some(file) => Err(invalid_plan(
                format!("{field}.sha256"),
                format!(
                    "does not match resolved source row `{source_file}`: expected `{}`, actual `{sha256}`",
                    file.sha256
                ),
            )),
            None => Err(invalid_plan(
                format!("{field}.source_file"),
                format!("`{source_file}` is absent from resolved_source.files"),
            )),
        }
    }

    fn validate_token_contract(parts: &ResolvedModelPlanParts) -> Result<(), VNextError> {
        let vocabulary_size = parts.tokenizer.vocabulary_size;
        let special = &parts.prepared_family.metadata().special_tokens;
        if special.collision_policy.allowed().iter().any(|collision| {
            collision.first() == SpecialTokenRole::Stop
                || collision.second() == SpecialTokenRole::Stop
        }) {
            return Err(invalid_plan(
                "special_tokens.collision_policy",
                "model metadata cannot authorize product-owned stop-token collisions",
            ));
        }
        let mut roles = Vec::new();
        if let Some(token_id) = special.bos_token_id {
            roles.push((SpecialTokenRole::Bos, token_id));
        }
        roles.extend(
            special
                .eos_token_ids
                .iter()
                .copied()
                .map(|token_id| (SpecialTokenRole::Eos, token_id)),
        );
        if let Some(token_id) = special.pad_token_id {
            roles.push((SpecialTokenRole::Pad, token_id));
        }
        roles.extend(
            parts
                .stop
                .token_ids
                .iter()
                .copied()
                .map(|token_id| (SpecialTokenRole::Stop, token_id)),
        );
        if roles
            .iter()
            .any(|(_, token_id)| u64::from(*token_id) >= vocabulary_size)
        {
            return Err(invalid_plan(
                "special_tokens",
                "a model or stop token id exceeds the tokenizer vocabulary",
            ));
        }
        let mut observed_stop_collisions = BTreeSet::new();
        for (index, (role, token_id)) in roles.iter().enumerate() {
            for (other_role, other_token_id) in &roles[..index] {
                if token_id == other_token_id && role != other_role {
                    let (policy_field, allowed) = if *role == SpecialTokenRole::Stop
                        || *other_role == SpecialTokenRole::Stop
                    {
                        let model_role = if *role == SpecialTokenRole::Stop {
                            *other_role
                        } else {
                            *role
                        };
                        observed_stop_collisions.insert(model_role);
                        (
                            "stop.collision_policy",
                            parts.stop.collision_policy.allows(model_role),
                        )
                    } else {
                        (
                            "special_tokens.collision_policy",
                            special.collision_policy.allows(*role, *other_role),
                        )
                    };
                    if !allowed {
                        return Err(invalid_plan(
                            policy_field,
                            format!(
                                "token id {token_id} is shared by {role:?} and {other_role:?} without an explicit policy"
                            ),
                        ));
                    }
                }
            }
        }
        if observed_stop_collisions != *parts.stop.collision_policy.allowed_model_roles() {
            return Err(invalid_plan(
                "stop.collision_policy",
                "declared model-role collisions must exactly match observed stop-token aliases",
            ));
        }
        Ok(())
    }

    fn decision_values(
        parts: &ResolvedModelPlanParts,
    ) -> Result<BTreeMap<ResolutionField, serde_json::Value>, VNextError> {
        #[derive(Serialize)]
        struct RuntimePresetValue<'a> {
            policy_id: &'a str,
            version: ContractVersion,
            scheduling: SchedulingDiscipline,
        }

        let mut values = BTreeMap::new();
        macro_rules! insert_value {
            ($field:expr, $value:expr, $context:literal) => {
                values.insert($field, canonical_json_value($value, $context)?);
            };
        }

        insert_value!(
            ResolutionField::OriginalSource,
            &parts.original_source,
            "serialize original model source decision"
        );
        insert_value!(
            ResolutionField::ResolvedSource,
            &parts.resolved_source,
            "serialize resolved model source decision"
        );
        insert_value!(
            ResolutionField::Config,
            &parts.config,
            "serialize model config decision"
        );
        insert_value!(
            ResolutionField::ExternalMetadata,
            &parts.external_metadata_id,
            "serialize external metadata decision"
        );
        insert_value!(
            ResolutionField::Family,
            parts.prepared_family.family_id(),
            "serialize model family decision"
        );
        insert_value!(
            ResolutionField::WeightSchema,
            parts.prepared_family.weight_schema(),
            "serialize weight schema decision"
        );
        insert_value!(
            ResolutionField::WeightFormat,
            &parts.prepared_family.weight_schema().format_id,
            "serialize weight format decision"
        );
        insert_value!(
            ResolutionField::Tokenizer,
            &parts.tokenizer,
            "serialize tokenizer decision"
        );
        insert_value!(
            ResolutionField::Template,
            &parts.prepared_family.metadata().template,
            "serialize template decision"
        );
        insert_value!(
            ResolutionField::SpecialTokens,
            &parts.prepared_family.metadata().special_tokens,
            "serialize special-token decision"
        );
        insert_value!(
            ResolutionField::Device,
            &parts.device,
            "serialize device decision"
        );
        insert_value!(
            ResolutionField::Capabilities,
            &parts.capabilities,
            "serialize capability decision"
        );
        insert_value!(
            ResolutionField::RuntimePreset,
            &RuntimePresetValue {
                policy_id: parts.runtime.policy_id(),
                version: parts.runtime.version(),
                scheduling: parts.runtime.scheduling(),
            },
            "serialize runtime preset decision"
        );
        insert_value!(
            ResolutionField::RuntimeMemory,
            parts.runtime.memory(),
            "serialize runtime memory decision"
        );
        insert_value!(
            ResolutionField::Admission,
            parts.runtime.admission(),
            "serialize admission decision"
        );
        insert_value!(
            ResolutionField::Engine,
            &parts.engine,
            "serialize engine decision"
        );
        insert_value!(
            ResolutionField::ExecutionPlan,
            parts.execution_plan.plan_hash().as_str(),
            "serialize execution-plan decision"
        );
        insert_value!(
            ResolutionField::Sampling,
            &parts.sampling,
            "serialize sampling decision"
        );
        insert_value!(
            ResolutionField::Stop,
            &parts.stop,
            "serialize stop decision"
        );
        insert_value!(
            ResolutionField::StructuredOutput,
            &parts.structured_output,
            "serialize structured-output decision"
        );
        Ok(values)
    }

    fn decision_fingerprints(
        parts: &ResolvedModelPlanParts,
    ) -> Result<BTreeMap<ResolutionField, String>, VNextError> {
        Self::decision_values(parts)?
            .into_iter()
            .map(|(field, value)| {
                canonical_fingerprint(&value, "fingerprint resolved decision value")
                    .map(|fingerprint| (field, fingerprint))
            })
            .collect()
    }

    pub fn parts(&self) -> &ResolvedModelPlanParts {
        &self.parts
    }

    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.parts.execution_plan
    }

    pub fn fingerprint(&self) -> &str {
        self.fingerprint.as_str()
    }

    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize resolved model plan",
            message: error.to_string(),
        })
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedResolvedModelPlan, VNextError> {
        if bytes.len() > MAX_RESOLVED_MODEL_PLAN_WIRE_BYTES {
            return Err(invalid_plan(
                "resolved_model_plan.wire_bytes",
                format!("must not exceed {MAX_RESOLVED_MODEL_PLAN_WIRE_BYTES} bytes"),
            ));
        }
        serde_json::from_slice::<ResolvedModelPlanWire>(bytes)
            .map(Into::into)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted resolved model plan",
                message: error.to_string(),
            })
    }

    pub fn from_json_validated(
        bytes: &[u8],
        context: &ResolvedPlanValidationContext<'_>,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate(context)
    }
}
