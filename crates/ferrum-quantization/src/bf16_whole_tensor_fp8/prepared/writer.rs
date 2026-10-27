use super::*;
use std::io::Write;

const SHARD_FILE: &str = "weights.safetensors";

struct ShardLayout {
    header: Vec<u8>,
    data_start: u64,
    file_bytes: u64,
    padding: BTreeMap<String, u64>,
}

// Metadata is bounded independently of model payload size. Header padding and
// tensor gaps are emitted from a fixed stack block, never a shard-sized Vec.
#[derive(Default)]
struct BoundedHeader(Vec<u8>);
impl Write for BoundedHeader {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = self.0.len().checked_add(bytes.len()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prepared header length overflow",
            )
        })?;
        if length as u64 > MAX_MANIFEST_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prepared header exceeds metadata bound",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn shard_layout<'a>(
    outputs: impl Iterator<Item = &'a WeightComponentSpec>,
) -> Result<ShardLayout, VNextError> {
    let mut header = BTreeMap::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "ferrum.prepared_fp8.layout": STORAGE_LAYOUT,
            "ferrum.prepared_fp8.alignment_bytes": STORAGE_ALIGNMENT_BYTES.to_string(),
        }),
    );
    let mut offset = 0_u64;
    let mut padding = BTreeMap::new();
    for (ordinal, spec) in outputs.enumerate() {
        let name = spec.id.as_str();
        let dtype = match spec.physical_element_type() {
            ElementType::U8 => "U8",
            ElementType::F32 => "F32",
            ElementType::Bf16 => "BF16",
            _ => return Err(invalid("prepared export has an unsupported physical dtype")),
        };
        let bytes = spec.physical_bytes()?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("prepared shard data length overflows"))?;
        if bytes == 0 || name.starts_with(PADDING_PREFIX) || name == "__metadata__" {
            return Err(invalid(
                "prepared output is empty or uses a reserved tensor name",
            ));
        }
        if header
            .insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": spec.dimensions,
                    "data_offsets": [offset, end],
                }),
            )
            .is_some()
        {
            return Err(invalid("prepared shard repeats an output tensor"));
        }
        offset = aligned_storage_end(end)?;
        if offset != end {
            let name = format!("{PADDING_PREFIX}{ordinal:06}");
            let bytes = offset - end;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [bytes],
                    "data_offsets": [end, offset],
                }),
            );
            padding.insert(name, bytes);
        }
    }
    let mut encoded = BoundedHeader::default();
    serde_json::to_writer(&mut encoded, &header).map_err(io_error)?;
    let data_start = aligned_storage_end(encoded.0.len() as u64 + 8)?;
    if data_start - 8 > MAX_MANIFEST_BYTES {
        return Err(invalid("prepared padded header exceeds metadata bound"));
    }
    let file_bytes = data_start
        .checked_add(offset)
        .filter(|n| *n <= isize::MAX as u64)
        .ok_or_else(|| invalid("prepared shard exceeds the address space"))?;
    Ok(ShardLayout {
        header: encoded.0,
        data_start,
        file_bytes,
        padding,
    })
}

fn write_fill(file: &mut impl Write, byte: u8, mut bytes: u64) -> Result<(), VNextError> {
    let block = [byte; 4096];
    while bytes != 0 {
        let n = bytes.min(block.len() as u64) as usize;
        file.write_all(&block[..n]).map_err(io_error)?;
        bytes -= n as u64;
    }
    Ok(())
}

fn sync_directory(destination: &Path) -> PreparedFp8DirectorySync {
    #[cfg(unix)]
    {
        match fs::File::open(destination).and_then(|file| file.sync_all()) {
            Ok(()) => PreparedFp8DirectorySync::Synced,
            Err(error) => PreparedFp8DirectorySync::Failed {
                message: error.to_string(),
            },
        }
    }
    #[cfg(not(unix))]
    {
        let _ = destination;
        PreparedFp8DirectorySync::Unsupported
    }
}

// The closure is a narrow commit-point test seam. Production always calls the
// actual OS directory-sync implementation; tests label injected failures as such.
pub(super) fn publish_manifest(
    temporary: &Path,
    manifest: &Path,
    destination: &Path,
    sync: impl FnOnce(&Path) -> PreparedFp8DirectorySync,
) -> Result<PreparedFp8DirectorySync, VNextError> {
    fs::rename(temporary, manifest).map_err(io_error)?;
    // After rename the complete artifact is already visible. Never return a
    // misleading export error or delete it because directory sync failed.
    let state = sync(destination);
    if state != PreparedFp8DirectorySync::Synced {
        tracing::warn!(
            target: "ferrum_quantization::prepared_fp8",
            manifest = %manifest.display(),
            directory_sync = ?state,
            "prepared FP8 archive is published; directory synchronization is not confirmed"
        );
    }
    Ok(state)
}

/// Retains one ordered original group through hashing and conversion. A second
/// component request borrows the same bytes, without another archive read-ahead
/// request or an unaccounted copy of a matrix.
struct GroupSource<'a> {
    specs: Vec<&'a WeightComponentSpec>,
    payloads: Vec<WeightComponentPayload<'a>>,
}
impl WeightComponentSource for GroupSource<'_> {
    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let index = self
            .specs
            .iter()
            .position(|s| *s == spec)
            .ok_or_else(|| invalid("export source differs from its ordered group"))?;
        let original = &self.payloads[index];
        let payload = WeightComponentPayload::from_ordered_sources(
            spec,
            original.external_names().to_vec(),
            original.source_files().to_vec(),
            original.dimensions().to_vec(),
            original.element_type(),
            Cow::Borrowed(original.bytes()),
        )?;
        match original.retained_host_memory() {
            Some(owner) => payload.with_retained_host_memory(owner.clone()),
            None => Err(invalid("export requires retained immutable source bytes")),
        }
    }
}

impl Bf16WholeTensorFp8Materializer {
    /// CPU-only export of this exact recipe, one complete source group at a time.
    /// Destination must not exist. The manifest is published last; failures
    /// before publication do not expose a usable artifact. After publication,
    /// directory-sync failures are reported in the successful receipt and a
    /// warning. No GPU runtime or full-stage Host cache is created, and
    /// converted bytes use this materializer's Host account.
    pub fn export_prepared(
        &self,
        family: &PreparedModelFamily,
        source: &dyn WeightComponentSource,
        destination: &Path,
    ) -> Result<PreparedFp8ExportReceipt, VNextError> {
        let prepared = plan::prepare(family, &self.groups)?;
        self.export_plan(
            family.stored_weight_schema()?,
            &prepared,
            source,
            destination,
        )
    }

    pub(super) fn export_plan(
        &self,
        original: &WeightSchema,
        plan: &plan::Prepared,
        source: &dyn WeightComponentSource,
        destination: &Path,
    ) -> Result<PreparedFp8ExportReceipt, VNextError> {
        if self.prepared.is_some() || self.host_cache.is_some() {
            return Err(invalid(
                "export requires an uncached runtime-conversion materializer",
            ));
        }
        if plan.maximum_group_accounted_bytes(&self.host)?
            > self
                .host
                .limits()
                .class_bytes(HostMemoryClass::WeightBacking) as u64
        {
            return Err(invalid(
                "prepared export conversion group exceeds WeightBacking capacity",
            ));
        }
        let mut grouped: BTreeMap<Vec<WeightId>, Vec<&WeightComponentSpec>> = BTreeMap::new();
        for output in &plan.schema.components {
            grouped
                .entry(
                    plan.sources
                        .get(&output.id)
                        .ok_or_else(|| invalid("export output has no source recipe"))?
                        .clone(),
                )
                .or_default()
                .push(output);
        }
        // Header offsets follow the same complete-group order used below, not
        // dtype sorting. Each component (including a 4B scale) starts aligned.
        let layout = shard_layout(grouped.values().flat_map(|outputs| outputs.iter().copied()))?;
        fs::create_dir(destination).map_err(io_error)?;
        let path = destination.join(SHARD_FILE);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(io_error)?;
        file.write_all(&(layout.data_start - 8).to_le_bytes())
            .map_err(io_error)?;
        file.write_all(&layout.header).map_err(io_error)?;
        write_fill(
            &mut file,
            b' ',
            layout.data_start - 8 - layout.header.len() as u64,
        )?;
        let originals: BTreeMap<_, _> = original
            .components
            .iter()
            .map(|spec| (&spec.id, spec))
            .collect();
        let mut source_records = BTreeMap::new();
        let mut components = Vec::new();
        let mut payload_bytes = 0_u64;
        for (source_ids, outputs) in grouped {
            let specs = source_ids
                .iter()
                .map(|id| {
                    originals
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("export provenance has no source component"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let payloads = specs
                .iter()
                .map(|spec| source.component(spec))
                .collect::<Result<Vec<_>, _>>()?;
            for (spec, payload) in specs.iter().zip(&payloads) {
                if payload.component_id() != &spec.id
                    || payload.dimensions() != spec.dimensions
                    || payload.element_type() != spec.physical_element_type()
                    || payload.external_names() != spec.external_names
                    || payload.bytes().len() as u64 != spec.physical_bytes()?
                    || payload.retained_host_memory().is_none()
                {
                    return Err(invalid(
                        "export source payload differs from its declared schema",
                    ));
                }
                let record = PreparedFp8SourceComponent {
                    spec: (*spec).clone(),
                    source_files: payload.source_files().to_vec(),
                    sha256: sha(payload.bytes()),
                };
                if let Some(previous) = source_records.insert(spec.id.clone(), record.clone()) {
                    if previous.sha256 != record.sha256
                        || previous.source_files != record.source_files
                    {
                        return Err(invalid("export source changed between groups"));
                    }
                }
            }
            let group = GroupSource { specs, payloads };
            let payloads = conversion::materialize(&self.host, &group, &group.specs, &outputs)?;
            for (spec, payload) in outputs.iter().zip(&payloads) {
                if payload.element_type() != spec.physical_element_type()
                    || payload.dimensions() != spec.dimensions
                    || payload.bytes().len() as u64 != spec.physical_bytes()?
                {
                    return Err(invalid(
                        "prepared converted payload differs from its planned extent",
                    ));
                }
                let key = spec.id.as_str();
                let bytes = payload.bytes().len() as u64;
                file.write_all(payload.bytes()).map_err(io_error)?;
                write_fill(&mut file, 0, aligned_storage_end(bytes)? - bytes)?;
                payload_bytes = payload_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("export payload bytes overflow"))?;
                components.push(PreparedFp8Component {
                    component_id: spec.id.clone(),
                    shard: SHARD_FILE.to_owned(),
                    tensor: key.to_owned(),
                    bytes,
                    sha256: sha(payload.bytes()),
                });
            }
            // All source/converted group payloads die before the next group.
        }
        file.sync_all().map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != layout.file_bytes {
            return Err(invalid(
                "prepared written shard length differs from planned extents",
            ));
        }
        drop(file);
        // Reuse immutable mmap indexing to hash the actual final file. No full
        // shard temporary buffer or second conversion is created.
        let written = SafetensorsArchive::open_single_file_with_header_limit(
            destination,
            SHARD_FILE,
            NonZeroU64::new(MAX_MANIFEST_BYTES).unwrap(),
        )
        .map_err(io_error)?;
        let first = written
            .tensor_names()
            .next()
            .ok_or_else(|| invalid("empty prepared shard"))?;
        let tensor = written.tensor(first).map_err(io_error)?;
        let whole = tensor.retained_host_memory().owner_bytes();
        validate_storage_header(whole)?;
        let shard_bytes = whole.len() as u64;
        let shards = vec![PreparedFp8Shard {
            file: SHARD_FILE.to_owned(),
            bytes: shard_bytes,
            sha256: sha(whole),
        }];
        components.sort_by(|a, b| a.component_id.cmp(&b.component_id));
        let sources: Vec<_> = source_records.into_values().collect();
        let source_identity_sha256 = sha(&serde_json::to_vec(&sources).map_err(io_error)?);
        let manifest = PreparedFp8Manifest {
            schema_version: 2,
            storage_alignment_bytes: STORAGE_ALIGNMENT_BYTES,
            padding: layout.padding,
            numerical_profile: BF16_WHOLE_TENSOR_FP8_NUMERICAL_PROFILE.into(),
            materializer: self.descriptor.clone(),
            groups: self.groups.clone(),
            source_schema: original.clone(),
            execution_schema: plan.schema.clone(),
            component_sources: plan.sources.clone(),
            source_identity_sha256: source_identity_sha256.clone(),
            sources,
            shards,
            components,
        };
        let bytes = serde_json::to_vec_pretty(&manifest).map_err(io_error)?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(invalid("export manifest exceeds metadata bound"));
        }
        let temporary = destination.join("prepared-fp8.json.partial");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        file.write_all(&bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        let manifest_path = destination.join(MANIFEST_FILE);
        let directory_sync =
            publish_manifest(&temporary, &manifest_path, destination, sync_directory)?;
        Ok(PreparedFp8ExportReceipt {
            manifest_path,
            manifest_sha256: sha(&bytes),
            source_identity_sha256,
            payload_bytes,
            shard_bytes,
            directory_sync,
        })
    }
}
