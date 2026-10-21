use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;

use ferrum_runtime::WeightSchema;
use ferrum_types::Result;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use super::{invalid, ExpectedWeights, QwenImage21Config, StructureLimits};

/// Detached header bytes plus a separately supplied whole-file length.
/// The length is a claim checked for internal consistency, not an authenticated
/// filesystem or HTTP observation. No tensor payload is read by this type.
/// `header_json` must contain the exact on-disk header range, including trailing
/// padding, and exclude the preceding eight-byte length word. Trimming or
/// reserializing it changes the payload offset and invalidates that evidence.
#[derive(Debug, Clone, Copy)]
pub struct HeaderEvidence<'a> {
    shard: &'a str,
    header_json: &'a [u8],
    declared_file_bytes: u64,
}

impl<'a> HeaderEvidence<'a> {
    pub fn new(shard: &'a str, header_json: &'a [u8], declared_file_bytes: u64) -> Result<Self> {
        validate_shard_path(shard)?;
        let prefix = (header_json.len() as u64)
            .checked_add(8)
            .ok_or_else(|| invalid("header length overflows"))?;
        if header_json.is_empty() || declared_file_bytes <= prefix {
            return Err(invalid(
                "detached header requires non-empty header and positive declared payload length",
            ));
        }
        Ok(Self {
            shard,
            header_json,
            declared_file_bytes,
        })
    }
}

/// Deliberately no "verified" variant: this preflight never reads payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadVerification {
    NotVerified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorStructure {
    external_name: String,
    shard: String,
    dimensions: Vec<u64>,
    dtype: HeaderDtype,
    /// Relative to the shard payload, after its eight-byte length and header.
    data_offsets: [u64; 2],
}

impl TensorStructure {
    pub fn external_name(&self) -> &str {
        &self.external_name
    }
    pub fn shard(&self) -> &str {
        &self.shard
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub const fn element_type(&self) -> ferrum_runtime::ElementType {
        ferrum_runtime::ElementType::Bf16
    }
    pub const fn data_offsets(&self) -> [u64; 2] {
        self.data_offsets
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShardStructure {
    shard: String,
    header_bytes: u64,
    header_sha256: String,
    declared_file_bytes: u64,
    payload_bytes: u64,
    tensor_count: usize,
}

impl ShardStructure {
    pub fn shard(&self) -> &str {
        &self.shard
    }
    pub const fn declared_file_bytes(&self) -> u64 {
        self.declared_file_bytes
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub fn header_sha256(&self) -> &str {
        &self.header_sha256
    }
}

/// Validated agreement between semantic configuration, index, and detached
/// shard headers. All identities are metadata identities; there is no runtime
/// admission, payload authentication, or `WeightComponentSource` capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21Structure {
    schema_version: u32,
    config: QwenImage21Config,
    config_sha256: String,
    index_sha256: String,
    shards: Vec<ShardStructure>,
    tensors: BTreeMap<String, TensorStructure>,
    payload_bytes: u64,
    parameter_count: u64,
    payload_verification: PayloadVerification,
    structure_fingerprint: String,
}

impl QwenImage21Structure {
    pub fn inspect(
        config_json: &[u8],
        index_json: &[u8],
        headers: &[HeaderEvidence<'_>],
        limits: StructureLimits,
    ) -> Result<Self> {
        if headers.is_empty() || headers.len() > limits.maximum_shards() {
            return Err(invalid(
                "header evidence shard count exceeds its supported bounds",
            ));
        }
        let mut total_bytes = 0_u64;
        for (label, bytes) in [
            ("transformer config", config_json),
            ("transformer index", index_json),
        ]
        .into_iter()
        .chain(
            headers
                .iter()
                .map(|header| (header.shard, header.header_json)),
        ) {
            limits.check_metadata(label, bytes)?;
            total_bytes = total_bytes
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("metadata total overflows"))?;
            if total_bytes > limits.total_metadata_bytes() {
                return Err(invalid("structure metadata exceeds total read limit"));
            }
        }
        let config = QwenImage21Config::from_hf_json(config_json, limits)?;
        let expected = ExpectedWeights::from_config(&config, limits)?;
        let index: IndexWire = serde_json::from_slice(index_json)
            .map_err(|error| invalid(format!("invalid transformer index: {error}")))?;
        if index.weight_map.0.len() != expected.tensors().len()
            || index.metadata.total_size != expected.payload_bytes()
        {
            return Err(invalid(
                "transformer index tensor count or total_size differs from expected structure",
            ));
        }
        let mut declared_shards = BTreeSet::new();
        for (name, shard) in &index.weight_map.0 {
            if !expected.tensors().contains_key(name) {
                return Err(invalid(format!("unknown index tensor {name:?}")));
            }
            validate_shard_path(shard)?;
            declared_shards.insert(shard.as_str());
        }
        let mut supplied_shards = BTreeSet::new();
        for header in headers {
            if !supplied_shards.insert(header.shard) {
                return Err(invalid(format!(
                    "duplicate header evidence for {:?}",
                    header.shard
                )));
            }
        }
        if supplied_shards != declared_shards {
            return Err(invalid(
                "header evidence must cover exactly the index shards",
            ));
        }
        let mut tensors = BTreeMap::new();
        let mut shards = Vec::with_capacity(headers.len());
        for header in headers {
            let parsed: HeaderWire = serde_json::from_slice(header.header_json)
                .map_err(|error| invalid(format!("invalid header {:?}: {error}", header.shard)))?;
            if parsed.tensors.is_empty() || parsed.tensors.len() > limits.maximum_tensors() {
                return Err(invalid("header tensor inventory is empty or exceeds limit"));
            }
            let payload_bytes = header.declared_file_bytes - 8 - header.header_json.len() as u64;
            let mut intervals = Vec::with_capacity(parsed.tensors.len());
            for (name, tensor) in parsed.tensors {
                let shape_bytes = tensor
                    .shape
                    .iter()
                    .try_fold(2_u64, |bytes, extent| {
                        if *extent == 0 {
                            None
                        } else {
                            bytes.checked_mul(*extent)
                        }
                    })
                    .filter(|_| !tensor.shape.is_empty())
                    .ok_or_else(|| invalid(format!("invalid or overflowing shape for {name:?}")))?;
                let Some(spec) = expected.tensors().get(&name) else {
                    return Err(invalid(format!("unknown header tensor {name:?}")));
                };
                if tensor.shape != spec.dimensions() || shape_bytes != spec.bytes() {
                    return Err(invalid(format!(
                        "header tensor {name:?} shape differs from typed config"
                    )));
                }
                if index.weight_map.0.get(&name).map(String::as_str) != Some(header.shard) {
                    return Err(invalid(format!(
                        "header tensor {name:?} is not in its index-declared shard"
                    )));
                }
                let [start, end] = tensor.data_offsets;
                if end.checked_sub(start) != Some(shape_bytes) || end > payload_bytes {
                    return Err(invalid(format!(
                        "invalid byte range or byte length for {name:?}"
                    )));
                }
                intervals.push((start, end));
                let structure = TensorStructure {
                    external_name: name.clone(),
                    shard: header.shard.to_owned(),
                    dimensions: tensor.shape,
                    dtype: tensor.dtype,
                    data_offsets: tensor.data_offsets,
                };
                if tensors.insert(name.clone(), structure).is_some() {
                    return Err(invalid(format!("duplicate tensor {name:?} across headers")));
                }
            }
            intervals.sort_unstable();
            let mut cursor = 0_u64;
            for (start, end) in &intervals {
                if *start != cursor {
                    return Err(invalid(format!(
                        "header {:?} has overlapping or non-contiguous payload intervals",
                        header.shard
                    )));
                }
                cursor = *end;
            }
            if cursor != payload_bytes {
                return Err(invalid(format!(
                    "header {:?} does not exactly cover declared file length",
                    header.shard
                )));
            }
            shards.push(ShardStructure {
                shard: header.shard.to_owned(),
                header_bytes: header.header_json.len() as u64,
                header_sha256: sha256(header.header_json),
                declared_file_bytes: header.declared_file_bytes,
                payload_bytes,
                tensor_count: intervals.len(),
            });
        }
        if tensors.len() != expected.tensors().len() || tensors.keys().ne(expected.tensors().keys())
        {
            return Err(invalid("headers omit expected transformer tensors"));
        }
        let payload_bytes = shards
            .iter()
            .try_fold(0_u64, |bytes, shard| bytes.checked_add(shard.payload_bytes))
            .ok_or_else(|| invalid("shard payload total overflows"))?;
        if payload_bytes != expected.payload_bytes() {
            return Err(invalid(
                "header payload total differs from typed transformer structure",
            ));
        }
        shards.sort_by(|left, right| left.shard.cmp(&right.shard));
        let config_sha256 = sha256(config_json);
        let index_sha256 = sha256(index_json);
        let encoded = serde_json::to_vec(&(
            "ferrum.qwen-image21.structure",
            1_u32,
            &config,
            &config_sha256,
            &index_sha256,
            &shards,
            &tensors,
        ))
        .map_err(|error| invalid(format!("serialize structure fingerprint: {error}")))?;
        Ok(Self {
            schema_version: 1,
            config,
            config_sha256,
            index_sha256,
            shards,
            tensors,
            payload_bytes,
            parameter_count: expected.parameter_count(),
            payload_verification: PayloadVerification::NotVerified,
            structure_fingerprint: sha256(&encoded),
        })
    }
    pub fn config(&self) -> &QwenImage21Config {
        &self.config
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorStructure> {
        &self.tensors
    }
    pub fn shards(&self) -> &[ShardStructure] {
        &self.shards
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub const fn parameter_count(&self) -> u64 {
        self.parameter_count
    }
    pub const fn payload_verification(&self) -> PayloadVerification {
        self.payload_verification
    }
    pub fn structure_fingerprint(&self) -> &str {
        &self.structure_fingerprint
    }
    pub fn weight_schema(&self) -> Result<WeightSchema> {
        ExpectedWeights::with_tensor_limit(&self.config, self.tensors.len())?.weight_schema()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum HeaderDtype {
    #[serde(rename = "BF16")]
    Bf16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorWire {
    dtype: HeaderDtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct HeaderWire {
    tensors: BTreeMap<String, TensorWire>,
}

impl<'de> Deserialize<'de> for HeaderWire {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = HeaderWire;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a safetensors header with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut seen = BTreeSet::new();
                let mut tensors = BTreeMap::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !seen.insert(name.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate header key {name:?}"
                        )));
                    }
                    if name == "__metadata__" {
                        map.next_value::<UniqueMap<String>>()?;
                    } else {
                        tensors.insert(name, map.next_value::<TensorWire>()?);
                    }
                }
                Ok(HeaderWire { tensors })
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexWire {
    metadata: IndexMetadata,
    weight_map: UniqueMap<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}

struct UniqueMap<T>(BTreeMap<String, T>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for UniqueVisitor<T> {
            type Value = UniqueMap<T>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut output = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if output.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate metadata key {key:?}"
                        )));
                    }
                    output.insert(key, map.next_value()?);
                }
                Ok(UniqueMap(output))
            }
        }
        deserializer.deserialize_map(UniqueVisitor(PhantomData))
    }
}

fn validate_shard_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':', '\0'])
        || !path.ends_with(".safetensors")
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(invalid(format!(
            "invalid relative safetensors shard path {path:?}"
        )));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
