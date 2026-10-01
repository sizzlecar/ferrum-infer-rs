//! Consume the worker's sealed numerical result without replaying or fitting.
use super::*;

impl StructuredServiceCollectorV6 {
    /// The caller must own the original live collector and use the same
    /// monotonic clock for capture and publication. `source_*` are the receipt
    /// from the source writer, checked against this collector's canonical stream.
    /// This moves qualified models; it never renews their original sample age.
    ///
    /// The profile is persisted for audit. Without a declared source wall-clock
    /// accuracy it cannot be loaded into another process; the ordinary export
    /// and import path retains strict replay and cross-epoch clock validation.
    pub fn publish_same_process(
        self,
        source_path: &Path,
        source_bytes: u64,
        source_sha256: [u8; 32],
        destination: &Path,
        now: StructuredServiceClockV6,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedStructuredCatalogV13, CostProfileError> {
        limits.validate()?;
        if self.sealed_source()? != (source_bytes, source_sha256) {
            return Err(invalid(
                "live source writer receipt differs from sealed collector",
            ));
        }
        let (catalog, bytes) = self.same_process_catalog(
            Some(source_path.canonicalize()?),
            Some(destination),
            now,
            limits,
        )?;
        publish_new(destination, &bytes)?;
        Ok(catalog)
    }

    /// Activate the original sealed numerical result without filesystem IO.
    /// Original sample ages, all three independent populations and canonical
    /// source hashes are preserved. Raw source bytes are streamed into a hash
    /// during capture and are not retained or claimed as a persisted artifact.
    pub fn activate_same_process_memory(
        self,
        now: StructuredServiceClockV6,
        limits: &CostProfileLoadLimits,
    ) -> Result<ImportedStructuredCatalogV13, CostProfileError> {
        self.same_process_catalog(None, None, now, limits)
            .map(|(catalog, _)| catalog)
    }

    fn same_process_catalog(
        self,
        source_path: Option<PathBuf>,
        destination: Option<&Path>,
        now: StructuredServiceClockV6,
        limits: &CostProfileLoadLimits,
    ) -> Result<(ImportedStructuredCatalogV13, Vec<u8>), CostProfileError> {
        limits.validate()?;
        let (source_bytes, source_sha256) = self.sealed_source()?;
        let mut children = Vec::new();
        let mut mapped = Vec::new();
        for (i, model) in self.models() {
            mapped.push((
                i,
                clock::same_process(
                    clock_evidence(&self, i),
                    now.monotonic_ns,
                    now.wall_unix_ns,
                    limits,
                )?,
            ));
            children.push(Child {
                declaration_index: i,
                owner: model.owner().clone(),
                domain_signature: *model.domain_signature(),
                parameters_sha256: model.parameters_signature(),
                phases: self.phases[i]
                    .clone()
                    .try_into()
                    .map_err(|_| invalid("live source missing three freezes"))?,
            });
        }
        if children.is_empty() {
            return Err(invalid("live source has no qualified child"));
        }
        let declared = EnvelopeV13 {
            artifact_type: ARTIFACT.into(),
            schema_version: 13,
            model_revision: MODEL_REVISION_V2.into(),
            fingerprint: self.header.fingerprint.clone(),
            source_path,
            source_bytes,
            source_sha256,
            capture_protocol: self.header.protocol,
            source_clock_max_error_ns: None,
            children,
        };
        let mut bytes = serde_json::to_vec_pretty(&declared)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_METADATA
            || source_bytes
                .checked_add(bytes.len() as u64)
                .is_none_or(|n| n > limits.max_file_bytes.get() as u64)
        {
            return Err(CostProfileError::Limit(
                "live profile and source byte capacity",
            ));
        }
        // Build all fallible model metadata before publishing the profile name.
        let catalog = assemble_catalog(
            self,
            declared,
            destination,
            bytes.len() as u64,
            Sha256::digest(&bytes).into(),
            mapped,
            ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic,
        )?;
        Ok((catalog, bytes))
    }
}
