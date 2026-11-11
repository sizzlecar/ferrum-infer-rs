//! Exact prepared safetensors mappings with installation-local Host authority.
//! This accounts source representation conversion only. Executor upload staging,
//! materializer backing, and dynamic causal I/O remain separate requirements.

use super::{PreparedProductionModel, ProductionWeightArtifact};
use ferrum_quantization::{
    Bf16F16ConversionPolicy, ManagedDenseArchiveSource, SafetensorsArchive, TensorArchive,
};
use ferrum_runtime::{
    ElementType, ExecutionPlan, HostMemoryAccount, VNextError, WeightComponentPayload,
    WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use std::sync::{Arc, OnceLock};

mod backing_selection;
mod host_backing;
mod process_mapped;
mod reusable;
mod selected_archive;
mod streaming;
use backing_selection::Policy as SourceBackingPolicy;

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("managed production source: {}", reason.into()),
    }
}

fn model_error(error: impl std::fmt::Display) -> ferrum_types::FerrumError {
    ferrum_types::FerrumError::model(error.to_string())
}

/// An immutable mapping of the exact artifact frozen by a production
/// preparation. No constructor accepts a replacement archive or model config.
/// Mapping lifetime is independent of any short-lived execution Host scope.
pub struct ManagedProductionModel {
    prepared: Arc<PreparedProductionModel>,
    archive: Arc<SafetensorsArchive>,
    mapped_cache: Option<Arc<process_mapped::CacheBinding>>,
}

/// A source lease for one actual installation Host account. Keep this lease and
/// its model mapping alive until static initialization (including transfers)
/// completes. It creates no scope or additional grant. Fresh conversion owners
/// release individually; explicitly armed reusable backing reserves one actual
/// allocation at the core hook and retains its full charge until source drop.
pub struct ManagedProductionWeightSource<'source> {
    prepared: &'source PreparedProductionModel,
    archive: &'source SafetensorsArchive,
    host: &'source HostMemoryAccount,
    source: ManagedDenseArchiveSource<'source>,
    mapped_cache: Option<&'source Arc<process_mapped::CacheBinding>>,
    mapped_ready: OnceLock<Arc<ferrum_quantization::ProcessMappedDenseWeights>>,
    static_reuse: reusable::StaticReuse<'source>,
    static_streaming: streaming::StaticStreaming<'source>,
    backing_selection: backing_selection::Selection,
}

impl PreparedProductionModel {
    /// Remap only this preparation's selected safetensors artifact and verify
    /// the actual mapped bytes against its frozen source fingerprints.
    /// Quantized source encodings are intentionally unsupported by this dense
    /// normalization adapter and remain on their explicit existing loaders.
    pub fn managed_safetensors(self: &Arc<Self>) -> ferrum_types::Result<ManagedProductionModel> {
        let archive = self.sources().weights().open_safetensors_archive()?;
        selected_archive::validate(self, &archive)?;
        for component in &self
            .family()
            .stored_weight_schema()
            .map_err(model_error)?
            .components
        {
            conversion_bytes(&archive, component).map_err(model_error)?;
        }
        Ok(ManagedProductionModel {
            prepared: Arc::clone(self),
            archive: Arc::new(archive),
            mapped_cache: None,
        })
    }
}

impl ManagedProductionModel {
    pub fn prepared(&self) -> &PreparedProductionModel {
        &self.prepared
    }

    /// Read-only lease of the exact already fingerprint-verified mmap. A
    /// multimodal component must not reopen its mutable source path later.
    pub fn shared_archive(&self) -> Arc<SafetensorsArchive> {
        Arc::clone(&self.archive)
    }

    /// Rebind a separately validated profile/family from this exact source
    /// authority. Reuses the original verified mmap, even after path changes;
    /// it neither reopens the source nor hashes a large checkpoint again.
    pub fn with_prepared(
        &self,
        prepared: Arc<PreparedProductionModel>,
    ) -> ferrum_types::Result<Self> {
        if !Arc::ptr_eq(self.prepared.sources(), prepared.sources()) {
            return Err(model_error(
                "prepared family belongs to another frozen source authority",
            ));
        }
        for component in &prepared
            .family()
            .stored_weight_schema()
            .map_err(model_error)?
            .components
        {
            conversion_bytes(self.archive.as_ref(), component).map_err(model_error)?;
        }
        let mapped_cache = self.rebind_mapped_cache(&prepared)?;
        Ok(Self {
            prepared,
            archive: Arc::clone(&self.archive),
            mapped_cache,
        })
    }

    /// Bind the caller's original account, including an already granted phase
    /// scope view. No nested scope or additional ledger is introduced.
    pub fn weight_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> ManagedProductionWeightSource<'a> {
        ManagedProductionWeightSource {
            prepared: &self.prepared,
            archive: &self.archive,
            host,
            source: ManagedDenseArchiveSource::new(self.archive.as_ref(), host),
            mapped_cache: self.mapped_cache.as_ref(),
            mapped_ready: OnceLock::new(),
            static_reuse: reusable::StaticReuse::new(),
            static_streaming: streaming::StaticStreaming::new(),
            backing_selection: backing_selection::Selection::new(),
        }
    }
}

impl ManagedProductionWeightSource<'_> {
    /// Arm one explicit reusable installation against this exact compiled
    /// identity plan. This metadata-only step allocates no payload. Core must
    /// subsequently validate the actual copy route and Host scope before the
    /// first allocation or component publication. It cannot be armed twice or
    /// after the source has already materialized a component.
    pub fn prepare_reusable_static_backing(&self, plan: &ExecutionPlan) -> Result<(), VNextError> {
        self.backing_selection
            .prepare(SourceBackingPolicy::Reusable, || {
                self.static_reuse.prepare(self, plan)
            })
    }

    /// Select bounded, installation-local normalization. The physical ceiling
    /// includes Host alignment padding; this metadata preparation allocates no
    /// payload and creates no memory grant.
    pub fn prepare_bounded_streaming_static_backing(
        &self,
        plan: &ExecutionPlan,
        maximum_physical_bytes: usize,
    ) -> Result<(), VNextError> {
        self.prepare_bounded_streaming_static_backing_with_policy(
            plan,
            maximum_physical_bytes,
            Bf16F16ConversionPolicy::default(),
        )
    }

    /// Select bounded conversion within the same source slot. Workers split
    /// its output range and all join before publication; this does not reserve
    /// another payload, overlap uploads, or alter fallback backing requirements.
    pub fn prepare_bounded_streaming_static_backing_with_policy(
        &self,
        plan: &ExecutionPlan,
        maximum_physical_bytes: usize,
        conversion_policy: Bf16F16ConversionPolicy,
    ) -> Result<(), VNextError> {
        self.backing_selection
            .prepare(SourceBackingPolicy::Streaming, || {
                self.static_streaming
                    .prepare(self, plan, maximum_physical_bytes, conversion_policy)
            })
    }

    pub fn execution_input(
        &self,
    ) -> ferrum_types::Result<ferrum_execution::PreparedExecutionInput<'_>> {
        self.prepared.execution_input_with_source(self)
    }

    /// Extra source-owned WeightBacking at the actual plan's largest static
    /// initialization group. This excludes upload TransferStaging and the
    /// materializer's own backing query. Only the exact core identity plan is
    /// supported; other materializers need their own explicit lifetime rule.
    pub fn maximum_group_backing_bytes(&self, plan: &ExecutionPlan) -> Result<usize, VNextError> {
        if self.backing_selection.get(false)? == SourceBackingPolicy::Streaming {
            return self.static_streaming.maximum_backing_bytes(plan);
        }
        let installed_peak = host_backing::maximum(self, plan)?;
        if self.mapped_cache.is_none() {
            return Ok(installed_peak);
        }
        // The cold private file covers the full bound physical schema. Include
        // optional/unplaced components in its streaming build peak, using the
        // same actual Host alignment rather than granting a second account.
        self.prepared
            .family()
            .stored_weight_schema()?
            .components
            .iter()
            .try_fold(installed_peak, |peak, component| {
                let bytes = usize::try_from(self.component_backing_bytes(component)?)
                    .map_err(|_| invalid("cached component exceeds Host address space"))?;
                if bytes == 0 {
                    return Ok(peak);
                }
                let backing = self
                    .host
                    .backing_bytes(bytes)
                    .map_err(|e| invalid(e.to_string()))?;
                Ok(peak.max(backing))
            })
    }

    /// Actual allocation size for this source conversion, before Host alignment.
    /// Zero denotes an immutable borrowed view rather than a free allocation.
    pub fn component_backing_bytes(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<u64, VNextError> {
        self.validate_component(component)?;
        conversion_bytes(self.archive, component)
    }

    fn validate_component(&self, component: &WeightComponentSpec) -> Result<(), VNextError> {
        let schema = self.prepared.family().stored_weight_schema()?;
        if schema
            .components
            .iter()
            .any(|expected| expected == component)
        {
            Ok(())
        } else {
            Err(invalid(format!(
                "component `{}` differs from the exact prepared source schema",
                component.id
            )))
        }
    }
}

impl WeightComponentSource for ManagedProductionWeightSource<'_> {
    fn streaming_component(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<Option<ferrum_runtime::WeightComponentStreamSpec>, VNextError> {
        match self.backing_selection.get(true)? {
            SourceBackingPolicy::Streaming => {
                self.validate_component(component)?;
                self.static_streaming.component_spec(component)
            }
            _ => Ok(None),
        }
    }

    fn component_range(
        &self,
        request: &ferrum_runtime::WeightComponentRangeRequest,
    ) -> Result<ferrum_runtime::WeightComponentRange, VNextError> {
        if self.backing_selection.get(true)? != SourceBackingPolicy::Streaming {
            return Err(invalid(
                "component ranges require the selected streaming policy",
            ));
        }
        self.validate_component(request.component())?;
        self.static_streaming.component_range(request)
    }

    fn validate_static_initialization(
        &self,
        context: &ferrum_runtime::StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        match self.backing_selection.get(true)? {
            SourceBackingPolicy::Fresh => Ok(()),
            SourceBackingPolicy::Reusable => self.static_reuse.validate(self, context),
            SourceBackingPolicy::Streaming => self.static_streaming.validate(self, context),
        }
    }

    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.validate_component(component)?;
        // Check the complete declared representation before the converter can
        // allocate; never accept an unvalidated format through a fallback.
        conversion_bytes(self.archive, component)?;
        match self.backing_selection.get(true)? {
            SourceBackingPolicy::Reusable => {
                if !self.static_reuse.uses_reuse()? {
                    return Err(invalid("reusable source is not ready"));
                }
                return self.static_reuse.component(&self.source, component);
            }
            SourceBackingPolicy::Streaming => {
                self.static_streaming.validate_full_component(component)?;
            }
            SourceBackingPolicy::Fresh => {}
        }
        if let Some(binding) = self.mapped_cache {
            return self
                .mapped_ready
                .get()
                .ok_or_else(|| invalid("process-mapped weights have not been prepared with the installation Host account"))?
                .component_with_upload_policy(component, binding.upload_policy);
        }
        self.source.component(component)
    }
}

fn floating(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    )
}

fn conversion_bytes(
    archive: &SafetensorsArchive,
    component: &WeightComponentSpec,
) -> Result<u64, VNextError> {
    let target = match &component.encoding {
        WeightEncoding::Dense { element_type }
        | WeightEncoding::DenseAffine { element_type, .. }
            if floating(*element_type) =>
        {
            *element_type
        }
        _ => {
            return Err(invalid(
                "only dense F16/BF16/F32 target encodings are supported",
            ))
        }
    };
    if component.external_names.is_empty()
        || component.dimensions.is_empty()
        || component.dimensions.contains(&0)
    {
        return Err(invalid(
            "dense components require source names and positive dimensions",
        ));
    }
    let tensors = component
        .external_names
        .iter()
        .map(|name| {
            TensorArchive::tensor(archive, name).map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for tensor in &tensors {
        if !matches!(tensor.encoding(), WeightEncoding::Dense { element_type } if floating(*element_type))
        {
            return Err(invalid("only dense floating source tensors are supported"));
        }
        let elements = tensor
            .shape()
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d));
        if elements.and_then(|n| n.checked_mul(tensor.element_type()?.size_bytes()))
            != u64::try_from(tensor.bytes().len()).ok()
        {
            return Err(invalid(
                "source shape differs from its physical byte extent",
            ));
        }
    }
    if let [tensor] = tensors.as_slice() {
        if tensor.shape() != component.dimensions {
            return Err(invalid(format!(
                "component `{}` source shape differs",
                component.id
            )));
        }
        return if matches!(component.encoding, WeightEncoding::Dense { .. })
            && tensor.element_type() == Some(target)
        {
            Ok(0)
        } else {
            component.physical_bytes()
        };
    }
    if !matches!(component.encoding, WeightEncoding::Dense { .. }) {
        return Err(invalid("multi-source affine normalization is unsupported"));
    }
    let stacked = component.dimensions[0] == tensors.len() as u64
        && tensors
            .iter()
            .all(|tensor| tensor.shape() == &component.dimensions[1..]);
    let concatenated = tensors.iter().all(|tensor| {
        tensor.shape().len() == component.dimensions.len()
            && tensor.shape()[1..] == component.dimensions[1..]
    }) && tensors
        .iter()
        .try_fold(0u64, |rows, tensor| rows.checked_add(tensor.shape()[0]))
        == Some(component.dimensions[0]);
    if !stacked && !concatenated {
        return Err(invalid(
            "sources do not form the declared exact stack or axis-zero concatenation",
        ));
    }
    component.physical_bytes()
}
