//! Process-owned disk backing for exact dense source normalization.
//!
//! Only the original managed converter writes tensor values. The finished
//! anonymous safetensors file is checked and mapped read-only after all writes;
//! no writable descriptor or caller Host account survives publication. Mapping
//! RSS, OS page cache and disk I/O remain real costs outside managed heap usage.
use crate::{ManagedDenseArchiveSource, TensorArchive};
use ferrum_runtime::{
    ImmutableFileMappedHostMemory, RetainedHostMemoryRegion, StableHostMemory, VNextError,
    WeightComponentPayload, WeightComponentSource, WeightComponentSpec,
};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path, sync::Arc};

mod layout;
use layout::Layout;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessMappedDenseLimits {
    pub maximum_file_bytes: u64,
    pub maximum_metadata_bytes: usize,
}

impl ProcessMappedDenseLimits {
    pub fn validate(self) -> Result<(), VNextError> {
        if self.maximum_file_bytes == 0
            || self.maximum_file_bytes > isize::MAX as u64
            || self.maximum_metadata_bytes == 0
            || self.maximum_metadata_bytes > 64 * 1024 * 1024
        {
            return Err(invalid(
                "file limit must be positive/addressable and metadata limit within 1..=64 MiB",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProcessMappedDenseRequirements {
    file_bytes: u64,
    payload_bytes: u64,
    metadata_bytes: usize,
    maximum_component_bytes: u64,
}
impl ProcessMappedDenseRequirements {
    pub const fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    /// Layout metadata at preflight; a completed build additionally includes
    /// its bounded original source-file lineage in this value.
    pub const fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
    pub const fn maximum_component_bytes(&self) -> u64 {
        self.maximum_component_bytes
    }
}

struct MappedFile {
    bytes: Mmap,
}
// SAFETY: the read-only file map owns fixed bytes until its final Arc drops.
// The anonymous file has no public path, and its sole writable descriptor is
// closed before this owner is constructed or any retained region is published.
unsafe impl StableHostMemory for MappedFile {
    fn stable_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
// SAFETY: this is an actual file-backed immutable read-only mapping, never a
// managed allocation, heap Vec or anonymous RAM map reclassified as a file.
unsafe impl ImmutableFileMappedHostMemory for MappedFile {}

pub struct ProcessMappedDenseWeights {
    mapping: Arc<MappedFile>,
    layout: Layout,
    source_files: Vec<Vec<String>>,
    requirements: ProcessMappedDenseRequirements,
    artifact_sha256: String,
}

impl ProcessMappedDenseWeights {
    /// Metadata only: no file, source tensor conversion or Host grant.
    pub fn requirements(
        components: &[WeightComponentSpec],
        limits: ProcessMappedDenseLimits,
    ) -> Result<ProcessMappedDenseRequirements, VNextError> {
        Ok(Layout::new(components, limits)?.requirements)
    }

    /// Build once on the original installation's Host authorization. The
    /// caller binds this result to its already verified source identity; this
    /// generic source does not accept a pathname or a claimed source digest as
    /// evidence for replacing an original archive.
    pub fn build(
        archive: &dyn TensorArchive,
        components: &[WeightComponentSpec],
        host: &ferrum_runtime::HostMemoryAccount,
        directory: &Path,
        limits: ProcessMappedDenseLimits,
    ) -> Result<Self, VNextError> {
        let layout = Layout::new(components, limits)?;
        let mut requirements = layout.requirements;
        let mut file = tempfile::tempfile_in(directory).map_err(io_error)?;
        let mut hash = Sha256::new();
        let prefix = (layout.header.len() as u64).to_le_bytes();
        file.write_all(&prefix).map_err(io_error)?;
        file.write_all(&layout.header).map_err(io_error)?;
        hash.update(prefix);
        hash.update(&layout.header);
        let source = ManagedDenseArchiveSource::new(archive, host);
        let mut source_files = vec![Vec::new(); components.len()];
        for &index in &layout.write_order {
            let entry = &layout.entries[index];
            // This exact existing converter preserves required F32 parameters,
            // ordered composites and DenseAffine's declared rounding once.
            let payload = source.component(&entry.spec)?;
            if payload.component_id() != &entry.spec.id
                || payload.external_names() != entry.spec.external_names
                || payload.dimensions() != entry.spec.dimensions
                || payload.element_type() != entry.spec.physical_element_type()
                || payload.bytes().len() != entry.relative.len()
            {
                return Err(invalid(
                    "converted payload differs from its exact component",
                ));
            }
            let lineage =
                layout::encoded_size(payload.source_files(), limits.maximum_metadata_bytes)?;
            requirements.metadata_bytes = requirements
                .metadata_bytes
                .checked_add(lineage)
                .ok_or_else(|| invalid("lineage metadata overflows"))?;
            if requirements.metadata_bytes > limits.maximum_metadata_bytes {
                return Err(invalid("source-file lineage exceeds metadata limit"));
            }
            source_files[index] = payload.source_files().to_vec();
            file.write_all(payload.bytes()).map_err(io_error)?;
            hash.update(payload.bytes());
            // Only this one component's converted Host owner remains live.
            drop(payload);
        }
        file.flush().map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != requirements.file_bytes {
            return Err(invalid(
                "completed file extent differs from its planned layout",
            ));
        }
        let expected_sha256 = format!("{:x}", hash.finalize());
        // SAFETY: tempfile_in returns an unlinked/private file. This builder
        // never exposes or clones its descriptor and has finished all writes.
        let mapping = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(io_error)?;
        drop(file); // No writable handle may escape with the read-only owner.
        if format!("{:x}", Sha256::digest(&mapping)) != expected_sha256 {
            return Err(invalid("mapped file checksum differs from converted bytes"));
        }
        layout.validate_mapping(&mapping)?;
        Ok(Self {
            mapping: Arc::new(MappedFile { bytes: mapping }),
            layout,
            source_files,
            requirements,
            artifact_sha256: expected_sha256,
        })
    }

    pub fn requirements_report(&self) -> &ProcessMappedDenseRequirements {
        &self.requirements
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// Exact component order and declarations, including source order and
    /// rational affine, must match the source contract used for this build.
    pub fn matches_components(&self, components: &[WeightComponentSpec]) -> bool {
        self.layout.entries.len() == components.len()
            && self
                .layout
                .entries
                .iter()
                .zip(components)
                .all(|(entry, spec)| &entry.spec == spec)
    }
}

impl WeightComponentSource for ProcessMappedDenseWeights {
    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let position = self
            .layout
            .lookup_order
            .binary_search_by(|&index| self.layout.entries[index].spec.id.cmp(&spec.id))
            .map_err(|_| invalid("requested component is absent from the normalized source"))?;
        let index = self.layout.lookup_order[position];
        let entry = &self.layout.entries[index];
        if &entry.spec != spec {
            return Err(invalid(
                "requested component differs from the normalized source contract",
            ));
        }
        let range = self.layout.data_range(entry)?;
        let retained = RetainedHostMemoryRegion::new_readonly_file_mapping(
            Arc::clone(&self.mapping),
            range.start,
            range.len(),
        )?;
        WeightComponentPayload::from_ordered_sources(
            spec,
            spec.external_names.clone(),
            self.source_files[index].clone(),
            spec.dimensions.clone(),
            spec.physical_element_type(),
            &self.mapping.bytes[range],
        )?
        .with_retained_host_memory(retained)
    }
}

fn invalid(reason: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("process-mapped dense source: {reason}"),
    }
}
fn io_error(error: std::io::Error) -> VNextError {
    invalid(format!("file I/O: {error}"))
}
