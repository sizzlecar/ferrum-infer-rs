use super::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::io::Write;

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
    /// Destination must not exist. Partial files are never a usable artifact:
    /// the manifest is published last. No GPU runtime or full-stage Host cache
    /// is created, and converted bytes use this materializer's Host account.
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
        fs::create_dir(destination).map_err(io_error)?;
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
        let originals: BTreeMap<_, _> = original
            .components
            .iter()
            .map(|spec| (&spec.id, spec))
            .collect();
        let mut source_records = BTreeMap::new();
        let mut components = Vec::new();
        let mut shards = Vec::new();
        let mut payload_bytes = 0_u64;
        let mut shard_bytes = 0_u64;
        for (ordinal, (source_ids, outputs)) in grouped.into_iter().enumerate() {
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
            let file = format!("group-{ordinal:06}.safetensors");
            let path = destination.join(&file);
            let mut views = Vec::with_capacity(outputs.len());
            for (spec, payload) in outputs.iter().zip(&payloads) {
                let dtype = match payload.element_type() {
                    ElementType::U8 => Dtype::U8,
                    ElementType::F32 => Dtype::F32,
                    ElementType::Bf16 => Dtype::BF16,
                    _ => return Err(invalid(
                        "prepared FP8 supports U8 packed, F32 scale and unchanged BF16 payloads",
                    )),
                };
                let shape = spec
                    .dimensions
                    .iter()
                    .map(|d| usize::try_from(*d).map_err(|_| invalid("export shape exceeds usize")))
                    .collect::<Result<Vec<_>, _>>()?;
                let key = spec.id.as_str();
                views.push((
                    key,
                    TensorView::new(dtype, shape, payload.bytes()).map_err(io_error)?,
                ));
                let bytes = payload.bytes().len() as u64;
                payload_bytes = payload_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("export payload bytes overflow"))?;
                components.push(PreparedFp8Component {
                    component_id: spec.id.clone(),
                    shard: file.clone(),
                    tensor: key.to_owned(),
                    bytes,
                    sha256: sha(payload.bytes()),
                });
            }
            serialize_to_file(views, &None, &path).map_err(io_error)?;
            fs::File::open(&path)
                .map_err(io_error)?
                .sync_all()
                .map_err(io_error)?;
            // Reuse immutable mmap indexing to hash the actual written file;
            // no shard-sized temporary Vec or second conversion is needed.
            let written =
                SafetensorsArchive::open_single_file(destination, &file).map_err(io_error)?;
            let tensor = written.tensor(outputs[0].id.as_str()).map_err(io_error)?;
            let whole = tensor.retained_host_memory().owner_bytes();
            let bytes = whole.len() as u64;
            shard_bytes = shard_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("export shard bytes overflow"))?;
            shards.push(PreparedFp8Shard {
                file,
                bytes,
                sha256: sha(whole),
            });
            // All source/converted payloads and the written mmap die here.
        }
        components.sort_by(|a, b| a.component_id.cmp(&b.component_id));
        let sources: Vec<_> = source_records.into_values().collect();
        let source_identity_sha256 = sha(&serde_json::to_vec(&sources).map_err(io_error)?);
        let manifest = PreparedFp8Manifest {
            schema_version: 1,
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
        fs::rename(&temporary, &manifest_path).map_err(io_error)?;
        #[cfg(unix)]
        fs::File::open(destination)
            .map_err(io_error)?
            .sync_all()
            .map_err(io_error)?;
        Ok(PreparedFp8ExportReceipt {
            manifest_path,
            manifest_sha256: sha(&bytes),
            source_identity_sha256,
            payload_bytes,
            shard_bytes,
        })
    }
}
