//! Original BF16 checkpoint binding for the existing complete visual graph.
//! This selects no alternative graph and owns no allocation or device cache.
use super::{Qwen3VLVisionConfig, Qwen3VLVisionFamily};
use anyhow::{ensure, Context, Result};
use ferrum_quantization::{ManagedDenseArchiveSource, SafetensorsArchive, TensorArchive};
use ferrum_runtime::*;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLVisionCheckpointLimits {
    pub maximum_metadata_bytes: usize,
    pub maximum_plan_nodes: usize,
    /// Selected graph weights plus the learned position table.
    pub maximum_source_bytes: u64,
}

pub struct PreparedQwen3VLVisionCheckpoint {
    archive: Arc<SafetensorsArchive>,
    config: Qwen3VLVisionConfig,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    components: BTreeMap<WeightId, WeightComponentSpec>,
    source_bytes: u64,
}
impl PreparedQwen3VLVisionCheckpoint {
    pub fn prepare(
        config: Qwen3VLVisionConfig,
        archive: Arc<SafetensorsArchive>,
        profile: &NumericalProfileId,
        limits: Qwen3VLVisionCheckpointLimits,
    ) -> Result<Self> {
        ensure!(
            limits.maximum_metadata_bytes > 0
                && limits.maximum_plan_nodes > 0
                && limits.maximum_source_bytes > 0,
            "positive visual capacities required"
        );
        ensure!(
            config.architecture().depth() <= limits.maximum_plan_nodes as u64,
            "visual layers exceed plan capacity"
        );
        let family = TypedFamilyRegistration::new(Qwen3VLVisionFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(&config)?, profile)?;
        let schema = family.stored_weight_schema()?;
        let nodes = family
            .program()
            .blocks()
            .iter()
            .map(|b| b.nodes.len())
            .sum::<usize>();
        ensure!(
            nodes <= limits.maximum_plan_nodes,
            "visual plan exceeds node capacity"
        );
        ensure!(
            serde_json::to_vec(&(&config, schema, family.program()))?.len()
                <= limits.maximum_metadata_bytes,
            "visual plan exceeds metadata capacity"
        );
        let mut source_bytes = 0u64;
        for component in &schema.components {
            let [name] = component.external_names.as_slice() else {
                anyhow::bail!("original visual weights require one exact checkpoint tensor");
            };
            let tensor = archive.tensor(name)?;
            ensure!(
                tensor.shape() == component.dimensions
                    && tensor.element_type() == Some(ElementType::Bf16),
                "original visual BF16 source differs: {name}"
            );
            ensure!(
                tensor.bytes().len() as u64 == component.physical_bytes()?,
                "visual payload extent differs: {name}"
            );
            source_bytes = source_bytes
                .checked_add(component.physical_bytes()?)
                .context("visual source extent overflow")?;
        }
        let position = archive.tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)?;
        let a = config.architecture();
        ensure!(
            position.shape() == [a.position_embeddings(), a.hidden_size()]
                && position.element_type() == Some(ElementType::Bf16),
            "original BF16 learned visual position table differs"
        );
        source_bytes = source_bytes
            .checked_add(position.bytes().len() as u64)
            .context("visual source extent overflow")?;
        ensure!(
            source_bytes <= limits.maximum_source_bytes,
            "selected visual source exceeds capacity"
        );
        let components = schema
            .components
            .iter()
            .map(|c| (c.id.clone(), c.clone()))
            .collect();
        let options = config.compile_options()?;
        Ok(Self {
            archive,
            config,
            family,
            options,
            components,
            source_bytes,
        })
    }
    pub fn config(&self) -> &Qwen3VLVisionConfig {
        &self.config
    }
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn selected_source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn managed_source<'a>(
        &'a self,
        host: &'a HostMemoryAccount,
    ) -> Qwen3VLVisionCheckpointSource<'a> {
        Qwen3VLVisionCheckpointSource {
            checked: &self.components,
            inner: ManagedDenseArchiveSource::new(self.archive.as_ref(), host),
        }
    }
    pub fn managed_position_inputs(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<[ManagedHostTensor; 3]> {
        self.config.position_inputs_bytes(
            host,
            self.archive
                .tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)?
                .bytes(),
        )
    }
}

/// Delegates original BF16 bytes and mmap retention to the existing managed
/// archive source; foreign components cannot request a conversion or alias.
pub struct Qwen3VLVisionCheckpointSource<'a> {
    checked: &'a BTreeMap<WeightId, WeightComponentSpec>,
    inner: ManagedDenseArchiveSource<'a>,
}
impl WeightComponentSource for Qwen3VLVisionCheckpointSource<'_> {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        if self.checked.get(&component.id) != Some(component) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "visual source requires the original checked BF16 component".into(),
            });
        }
        self.inner.component(component)
    }
}

#[cfg(test)]
mod tests;
