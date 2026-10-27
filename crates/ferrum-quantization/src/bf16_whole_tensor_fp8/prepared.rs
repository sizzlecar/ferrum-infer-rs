//! Explicit, verified physical execution weights. This is not an automatic cache:
//! opening verifies the selected immutable files, and a missing/mismatched output
//! never causes conversion. Original BF16 inventory remains available separately.
use super::*;
use crate::{ArchiveTensor, SafetensorsArchive, TensorArchive};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(test)]
mod tests;
mod writer;

const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MANIFEST_FILE: &str = "prepared-fp8.json";
// File-relative alignment covers the current 16-byte tensor ABI and native
// 4/16/64KiB pages. Runtime still checks the actual mmap address and native page
// size; this format does not promise that Metal will accept every no-copy view.
const STORAGE_ALIGNMENT_BYTES: u64 = 64 * 1024;
const STORAGE_LAYOUT: &str = "aligned64k.single-shard.v2";
const PADDING_PREFIX: &str = "__ferrum_prepared_padding_";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedFp8Manifest {
    pub schema_version: u32,
    pub storage_alignment_bytes: u64,
    /// Explicit zero-filled U8 tensors, not execution weights. The single shard
    /// contains exactly these tensors and `components`, without hidden gaps.
    pub padding: BTreeMap<String, u64>,
    pub numerical_profile: String,
    pub materializer: WeightMaterializerDescriptor,
    pub groups: Vec<Vec<WeightId>>,
    pub source_schema: WeightSchema,
    pub execution_schema: WeightSchema,
    pub component_sources: BTreeMap<WeightId, Vec<WeightId>>,
    /// Hash of the actual component specs, ordered source file names, and bytes.
    /// It is not a claimed hash of unconsumed checkpoint contents.
    pub source_identity_sha256: String,
    pub sources: Vec<PreparedFp8SourceComponent>,
    pub shards: Vec<PreparedFp8Shard>,
    pub components: Vec<PreparedFp8Component>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedFp8SourceComponent {
    pub spec: WeightComponentSpec,
    pub source_files: Vec<String>,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedFp8Shard {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedFp8Component {
    pub component_id: WeightId,
    pub shard: String,
    pub tensor: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedFp8ExportReceipt {
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub source_identity_sha256: String,
    pub payload_bytes: u64,
    pub shard_bytes: u64,
    /// Publication already succeeded. This reports only the following directory
    /// sync attempt, not a filesystem/power-loss durability guarantee.
    pub directory_sync: PreparedFp8DirectorySync,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedFp8DirectorySync {
    Synced,
    Failed { message: String },
    Unsupported,
}

pub struct PreparedFp8Archive {
    manifest: PreparedFp8Manifest,
    manifest_sha256: String,
    shards: BTreeMap<String, SafetensorsArchive>,
    components: BTreeMap<WeightId, usize>,
    payload_bytes: u64,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn canonical_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
fn io_error(error: impl std::fmt::Display) -> VNextError {
    invalid(format!("prepared FP8: {error}"))
}
fn portable_file(name: &str) -> bool {
    !name.is_empty()
        && !matches!(name, "." | "..")
        && !name.contains(['/', '\\', ':'])
        && !name.chars().any(char::is_control)
}

fn aligned_storage_end(bytes: u64) -> Result<u64, VNextError> {
    bytes
        .checked_add(STORAGE_ALIGNMENT_BYTES - 1)
        .map(|n| n & !(STORAGE_ALIGNMENT_BYTES - 1))
        .ok_or_else(|| invalid("prepared storage alignment overflows"))
}

fn validate_storage_header(whole: &[u8]) -> Result<u64, VNextError> {
    let prefix: [u8; 8] = whole
        .get(..8)
        .ok_or_else(|| invalid("prepared shard header is missing"))?
        .try_into()
        .unwrap();
    let header_bytes = u64::from_le_bytes(prefix);
    let data_start = header_bytes
        .checked_add(8)
        .filter(|n| *n <= whole.len() as u64)
        .ok_or_else(|| invalid("prepared shard header exceeds its owner"))?;
    if header_bytes > MAX_MANIFEST_BYTES
        || !data_start.is_multiple_of(STORAGE_ALIGNMENT_BYTES)
        || !(whole.len() as u64).is_multiple_of(STORAGE_ALIGNMENT_BYTES)
    {
        return Err(invalid("prepared shard header or EOF alignment differs"));
    }
    let header: serde_json::Value =
        serde_json::from_slice(&whole[8..data_start as usize]).map_err(io_error)?;
    if header.get("__metadata__")
        != Some(&serde_json::json!({
            "ferrum.prepared_fp8.layout": STORAGE_LAYOUT,
            "ferrum.prepared_fp8.alignment_bytes": STORAGE_ALIGNMENT_BYTES.to_string(),
        }))
    {
        return Err(invalid("prepared shard layout metadata differs"));
    }
    Ok(data_start)
}

impl PreparedFp8Archive {
    /// Hashes the selected manifest and every mapped payload once per opened
    /// owner. Files must remain immutable while retained, like raw archives.
    /// Hashing/opening belongs inside cold-process startup measurements.
    pub fn open(path: &Path, expected_manifest_sha256: &str) -> Result<Arc<Self>, VNextError> {
        let started = std::time::Instant::now();
        if !canonical_sha(expected_manifest_sha256) {
            return Err(invalid("prepared manifest SHA256 is not canonical"));
        }
        let file = fs::File::open(path).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() > MAX_MANIFEST_BYTES {
            return Err(invalid("prepared manifest exceeds metadata bound"));
        }
        let mut bytes = Vec::new();
        file.take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES || sha(&bytes) != expected_manifest_sha256 {
            return Err(invalid(
                "prepared manifest SHA256 mismatch or oversized metadata",
            ));
        }
        let manifest: PreparedFp8Manifest = serde_json::from_slice(&bytes).map_err(io_error)?;
        if manifest.schema_version != 2
            || manifest.storage_alignment_bytes != STORAGE_ALIGNMENT_BYTES
            || manifest.numerical_profile != BF16_WHOLE_TENSOR_FP8_NUMERICAL_PROFILE
            || manifest.groups.is_empty()
            || manifest.groups.len() > 4096
            || manifest.shards.len() != 1
            || manifest.components.is_empty()
            || manifest.materializer.fidelity() != WeightMaterializationFidelity::Approximate
            || manifest.materializer.approximate_quality_contract() != Some(&quality::contract()?)
        {
            return Err(invalid(
                "unsupported prepared FP8 version, recipe or quality contract",
            ));
        }
        let source_bytes = serde_json::to_vec(&manifest.sources).map_err(io_error)?;
        if sha(&source_bytes) != manifest.source_identity_sha256 {
            return Err(invalid("prepared source identity mismatch"));
        }
        let original: BTreeMap<_, _> = manifest
            .source_schema
            .components
            .iter()
            .map(|s| (&s.id, s))
            .collect();
        let mut seen_sources = BTreeSet::new();
        for record in &manifest.sources {
            if original.get(&record.spec.id).copied() != Some(&record.spec)
                || !seen_sources.insert(record.spec.id.clone())
                || !canonical_sha(&record.sha256)
                || record.source_files.len() != record.spec.external_names.len()
                || record.source_files.iter().any(|f| {
                    f.is_empty()
                        || f.starts_with('/')
                        || f.contains('\\')
                        || f.split('/').any(|v| matches!(v, "" | "." | ".."))
                })
            {
                return Err(invalid(
                    "prepared source identity is incomplete or inconsistent",
                ));
            }
        }
        let required_sources: BTreeSet<_> = manifest
            .component_sources
            .values()
            .flatten()
            .cloned()
            .collect();
        if seen_sources != required_sources {
            return Err(invalid("prepared source provenance coverage differs"));
        }
        let parent = path
            .parent()
            .ok_or_else(|| invalid("prepared manifest has no directory"))?;
        let mut shards = BTreeMap::new();
        let mut data_start = 0;
        for record in &manifest.shards {
            if !portable_file(&record.file)
                || !record.file.ends_with(".safetensors")
                || !canonical_sha(&record.sha256)
                || record.bytes == 0
                || shards.contains_key(&record.file)
            {
                return Err(invalid("prepared shard declaration is invalid or repeated"));
            }
            let archive = SafetensorsArchive::open_single_file_with_header_limit(
                parent,
                &record.file,
                NonZeroU64::new(MAX_MANIFEST_BYTES).unwrap(),
            )
            .map_err(io_error)?;
            let first = archive
                .tensor_names()
                .next()
                .ok_or_else(|| invalid("empty prepared shard"))?;
            // Validate the same retained mapping used later, not a separately
            // reopened path that could name different bytes.
            let tensor = archive.tensor(first).map_err(io_error)?;
            let whole = tensor.retained_host_memory().owner_bytes();
            if whole.len() as u64 != record.bytes || sha(whole) != record.sha256 {
                return Err(invalid("prepared shard size or SHA256 mismatch"));
            }
            data_start = validate_storage_header(whole)?;
            shards.insert(record.file.clone(), archive);
        }
        let specs: BTreeMap<_, _> = manifest
            .execution_schema
            .components
            .iter()
            .map(|s| (&s.id, s))
            .collect();
        let mut components = BTreeMap::new();
        let mut storage = BTreeSet::new();
        let mut payload_bytes = 0_u64;
        let mut ranges = Vec::new();
        for (i, record) in manifest.components.iter().enumerate() {
            let spec = specs
                .get(&record.component_id)
                .ok_or_else(|| invalid("prepared output absent from execution schema"))?;
            if components.insert(record.component_id.clone(), i).is_some()
                || !storage.insert((record.shard.clone(), record.tensor.clone()))
                || !canonical_sha(&record.sha256)
                || record.tensor != record.component_id.as_str()
                || record.tensor.starts_with(PADDING_PREFIX)
            {
                return Err(invalid("duplicate or invalid prepared output"));
            }
            let tensor = shards
                .get(&record.shard)
                .ok_or_else(|| invalid("prepared output references undeclared shard"))?
                .tensor(&record.tensor)
                .map_err(io_error)?;
            if tensor.shape() != spec.dimensions
                || tensor.element_type() != Some(spec.physical_element_type())
                || record.bytes != spec.physical_bytes()?
                || tensor.bytes().len() as u64 != record.bytes
                || sha(tensor.bytes()) != record.sha256
            {
                return Err(invalid(
                    "prepared output shape, encoding, size or SHA256 mismatch",
                ));
            }
            let offset = tensor.retained_host_memory().offset_bytes() as u64;
            if !offset.is_multiple_of(STORAGE_ALIGNMENT_BYTES) {
                return Err(invalid("prepared component is not storage-aligned"));
            }
            ranges.push((offset, record.bytes, false));
            if spec.encoding
                == WeightEncoding::Quantized(QuantizationSpec::fp8_e4m3fn_whole_tensor_f32())
                && tensor.bytes().iter().any(|byte| byte & 0x7f == 0x7f)
            {
                return Err(invalid("prepared finite FP8 recipe contains a NaN payload"));
            }
            if spec.role == WeightComponentRole::Scales
                && spec.encoding
                    == (WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    })
            {
                for scale in tensor.bytes().chunks_exact(4) {
                    let value = f32::from_le_bytes(scale.try_into().unwrap());
                    if !value.is_finite() || value <= 0.0 {
                        return Err(invalid("prepared FP8 scale is not finite positive"));
                    }
                }
            }
            payload_bytes = payload_bytes
                .checked_add(record.bytes)
                .ok_or_else(|| invalid("prepared payload size overflows"))?;
        }
        let shard = &manifest.shards[0];
        let archive = &shards[&shard.file];
        for (name, &bytes) in &manifest.padding {
            let ordinal = name.strip_prefix(PADDING_PREFIX).unwrap_or("");
            if ordinal.is_empty()
                || !ordinal.bytes().all(|c| c.is_ascii_digit())
                || bytes == 0
                || bytes >= STORAGE_ALIGNMENT_BYTES
                || !storage.insert((shard.file.clone(), name.clone()))
            {
                return Err(invalid("prepared padding declaration differs"));
            }
            let tensor = archive.tensor(name).map_err(io_error)?;
            let offset = tensor.retained_host_memory().offset_bytes() as u64;
            if tensor.element_type() != Some(ElementType::U8)
                || tensor.shape() != [bytes]
                || tensor.bytes().len() as u64 != bytes
                || tensor.bytes().iter().any(|&byte| byte != 0)
                || aligned_storage_end(offset)?.checked_sub(offset) != Some(bytes)
            {
                return Err(invalid(
                    "prepared padding shape, range or zero contents differ",
                ));
            }
            ranges.push((offset, bytes, true));
        }
        ranges.sort_unstable();
        let mut end = data_start;
        for (offset, bytes, _) in ranges {
            if offset != end {
                return Err(invalid("prepared tensor ranges leave gaps or overlap"));
            }
            end = end
                .checked_add(bytes)
                .ok_or_else(|| invalid("prepared tensor range overflows"))?;
        }
        if end != shard.bytes {
            return Err(invalid(
                "prepared tensors do not cover the complete data section",
            ));
        }
        let actual_storage: BTreeSet<_> = shards
            .iter()
            .flat_map(|(file, archive)| {
                archive
                    .tensor_names()
                    .map(move |name| (file.clone(), name.to_owned()))
            })
            .collect();
        if components.len() != specs.len()
            || components.len() != manifest.execution_schema.components.len()
            || storage != actual_storage
            || manifest.component_sources.len() != components.len()
            || !components.keys().eq(manifest.component_sources.keys())
        {
            return Err(invalid(
                "prepared output, schema or source coverage differs",
            ));
        }
        tracing::debug!(
            target: "ferrum_quantization::prepared_fp8",
            manifest_sha256 = expected_manifest_sha256,
            payload_bytes,
            shard_count = shards.len(),
            component_count = components.len(),
            padding_count = manifest.padding.len(),
            storage_layout = STORAGE_LAYOUT,
            open_verify_wall_us = started.elapsed().as_micros() as u64,
            verification = "whole_shard_and_component_hashes_finite_fp8_scales_aligned_zero_padding",
            "verified explicit prepared FP8 physical source"
        );
        Ok(Arc::new(Self {
            manifest,
            manifest_sha256: expected_manifest_sha256.into(),
            shards,
            components,
            payload_bytes,
        }))
    }
    pub fn manifest(&self) -> &PreparedFp8Manifest {
        &self.manifest
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Device-free recipe/schema compatibility; never hashes the original BF16
    /// payload again. The explicitly selected prepared bytes are authoritative.
    pub fn validate_recipe(
        &self,
        family: &PreparedModelFamily,
        groups: &[Vec<WeightId>],
    ) -> Result<(), VNextError> {
        let mut groups = groups.to_vec();
        groups.sort();
        self.validate_recipe_parts(
            family.stored_weight_schema()?,
            &plan::prepare(family, &groups)?,
            &groups,
        )
    }
    fn validate_recipe_parts(
        &self,
        source: &WeightSchema,
        plan: &plan::Prepared,
        groups: &[Vec<WeightId>],
    ) -> Result<(), VNextError> {
        if self.manifest.groups != groups
            || self.manifest.materializer.implementation_fingerprint()
                != recipe_fingerprint(groups)?
        {
            return Err(invalid(
                "prepared FP8 conversion recipe or implementation mismatch",
            ));
        }
        self.validate_plan(source, plan)
    }
    pub(super) fn validate_plan(
        &self,
        source: &WeightSchema,
        plan: &plan::Prepared,
    ) -> Result<(), VNextError> {
        if &self.manifest.source_schema != source
            || self.manifest.execution_schema != plan.schema
            || self.manifest.component_sources != plan.sources
        {
            return Err(invalid(
                "prepared FP8 source/execution schema or ordered provenance mismatch",
            ));
        }
        Ok(())
    }
    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let index = self.components.get(&spec.id).ok_or_else(|| {
            invalid("prepared output is missing; conversion fallback is forbidden")
        })?;
        if !self
            .manifest
            .execution_schema
            .components
            .iter()
            .any(|s| s == spec)
        {
            return Err(invalid(
                "prepared requested output differs from its exact schema",
            ));
        }
        let record = &self.manifest.components[*index];
        let tensor = self.shards[&record.shard]
            .tensor(&record.tensor)
            .map_err(io_error)?;
        WeightComponentPayload::from_ordered_sources(
            spec,
            spec.external_names.clone(),
            vec![record.shard.clone(); spec.external_names.len()],
            spec.dimensions.clone(),
            spec.physical_element_type(),
            Cow::Borrowed(tensor.bytes()),
        )?
        .with_retained_host_memory(tensor.retained_host_memory().clone())
    }
    pub(super) fn materialize<'a>(
        &self,
        source: &'a dyn WeightComponentSource,
        inputs: &[&WeightComponentSpec],
        outputs: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'a>>, VNextError> {
        if outputs.is_empty() {
            return Err(invalid("empty prepared output request"));
        }
        let ids: Vec<_> = inputs.iter().map(|s| s.id.clone()).collect();
        for input in inputs {
            if !self
                .manifest
                .source_schema
                .components
                .iter()
                .any(|s| s == *input)
            {
                return Err(invalid("prepared input provenance differs"));
            }
        }
        let mut seen = BTreeSet::new();
        outputs
            .iter()
            .map(|output| {
                if self.manifest.component_sources.get(&output.id) != Some(&ids)
                    || !seen.insert(&output.id)
                {
                    return Err(invalid(
                        "prepared ordered sources or output request differs",
                    ));
                }
                let expected = self.component(output)?;
                let payload = source.component(output)?;
                if payload.bytes().len() != expected.bytes().len()
                    || !std::ptr::eq(payload.bytes().as_ptr(), expected.bytes().as_ptr())
                    || payload.retained_host_memory().is_none()
                {
                    return Err(invalid(
                        "prepared source must supply the selected verified archive owner",
                    ));
                }
                Ok(payload)
            })
            .collect()
    }
}

/// Keeps the original logical inventory for model metadata, embedding and vision
/// consumers. Only exact declared execution components use prepared storage.
pub struct PreparedFp8TensorArchive {
    original: Arc<dyn TensorArchive>,
    prepared: Arc<PreparedFp8Archive>,
}
impl PreparedFp8TensorArchive {
    pub fn new(original: Arc<dyn TensorArchive>, prepared: Arc<PreparedFp8Archive>) -> Self {
        Self { original, prepared }
    }
}
impl TensorArchive for PreparedFp8TensorArchive {
    fn tensor(&self, name: &str) -> ferrum_types::Result<ArchiveTensor<'_>> {
        self.original.tensor(name)
    }
    fn tensor_count(&self) -> usize {
        self.original.tensor_count()
    }
    fn tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        self.original.tensor_names()
    }
}
impl WeightComponentSource for PreparedFp8TensorArchive {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if self.prepared.components.contains_key(&component.id) {
            return self.prepared.component(component);
        }
        if matches!(component.encoding, WeightEncoding::Quantized(_)) {
            return Err(invalid(
                "undeclared prepared quantized output; fallback is forbidden",
            ));
        }
        self.original.component(component)
    }
}

impl Bf16WholeTensorFp8Materializer {
    pub fn with_prepared_archive(
        self,
        family: &PreparedModelFamily,
        archive: Arc<PreparedFp8Archive>,
    ) -> Result<Self, VNextError> {
        archive.validate_recipe(family, &self.groups)?;
        self.bind_prepared(archive)
    }
    fn bind_prepared(mut self, archive: Arc<PreparedFp8Archive>) -> Result<Self, VNextError> {
        if self.host_cache.is_some() || self.prepared.is_some() {
            return Err(invalid(
                "prepared source cannot replace an existing cache or binding",
            ));
        }
        if archive.manifest.materializer != self.descriptor {
            return Err(invalid("prepared materializer identity differs"));
        }
        self.descriptor = WeightMaterializerDescriptor::new(
            self.descriptor.id().clone(),
            self.descriptor.version(),
            fingerprint(&[
                self.descriptor.implementation_fingerprint().as_bytes(),
                archive.manifest_sha256().as_bytes(),
                include_bytes!("prepared.rs"),
                include_bytes!("prepared/writer.rs"),
            ]),
            WeightMaterializationFidelity::Approximate,
            self.descriptor.required_capabilities().clone(),
        )?
        .with_approximate_quality_contract(quality::contract()?)?;
        self.prepared = Some(archive);
        Ok(self)
    }
}
