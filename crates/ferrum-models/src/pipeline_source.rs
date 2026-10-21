//! Metadata preflight for component pipelines, independent of text execution.
//!
//! This module recognizes declared pipeline/component architectures and retains
//! the exact metadata bytes consumed during preflight. It never discovers,
//! opens, or validates weight payloads and grants no execution capability.
//! Numerical configuration validation and weight/header verification remain
//! mandatory when the component model programs are prepared.

mod declaration;
mod files;
mod identity;

pub use declaration::{
    ComponentArchitecture, ComponentDeclaration, ComponentRole, PipelineArchitecture,
    PipelineDeclaration,
};
pub use files::{PipelineSourceLimits, RetainedMetadataFile};
pub use identity::{MetadataFileDigest, PipelineMetadataLock, PipelineSourceManifest};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferrum_types::{FerrumError, Result};
use serde_json::Value;

/// Retained metadata for one declared component; no weights have been opened.
#[derive(Debug, Clone)]
pub struct PipelineComponentSources {
    declaration: ComponentDeclaration,
    config: RetainedMetadataFile,
    auxiliary: BTreeMap<String, RetainedMetadataFile>,
}

impl PipelineComponentSources {
    pub fn declaration(&self) -> &ComponentDeclaration {
        &self.declaration
    }

    pub fn config(&self) -> &RetainedMetadataFile {
        &self.config
    }

    /// Look up an exact retained package-relative path, without filesystem IO.
    pub fn auxiliary(&self, relative_path: &str) -> Option<&RetainedMetadataFile> {
        self.auxiliary.get(relative_path)
    }
}

/// A content-bound metadata snapshot. Moving it does not reopen source files.
///
/// The local root must contain all metadata (symlinks within it are allowed).
/// External HF-cache blob symlinks are deliberately not authorized by this
/// local-directory API. A future cache resolver must provide an explicit,
/// verified blob boundary rather than silently allowing arbitrary file reads.
/// Callers must keep package paths and contents stable while opening a bundle;
/// this API does not protect against concurrent filesystem path replacement.
/// Once opened, retained bytes are independent of subsequent source changes.
#[derive(Debug, Clone)]
pub struct PipelineSourceBundle {
    root: PathBuf,
    model_index: RetainedMetadataFile,
    declaration: PipelineDeclaration,
    components: BTreeMap<String, PipelineComponentSources>,
    manifest: PipelineSourceManifest,
}

impl PipelineSourceBundle {
    /// Validate all required small metadata before a caller opens any weights.
    ///
    /// Missing, empty, oversized or mismatched metadata fails here. JSON and
    /// tokenizer structures are parsed; chat templates are checked only for
    /// non-empty UTF-8. Template syntax/rendering and numerical configuration
    /// validation remain the component program's responsibility. A successful
    /// result is metadata preflight only, not a runnable model.
    pub fn open(root: impl AsRef<Path>, limits: PipelineSourceLimits) -> Result<Self> {
        let mut reader = files::MetadataReader::new(root.as_ref(), limits)?;
        let model_index = reader.read("model_index.json", false)?;
        let declaration = PipelineDeclaration::from_json(model_index.bytes())?;
        let mut components = BTreeMap::new();
        for component in declaration.components() {
            let config = reader.read(component.config_path(), false)?;
            validate_component_config(component, &config)?;
            let mut auxiliary = BTreeMap::new();
            match component.architecture() {
                ComponentArchitecture::Qwen3VlProcessor => {
                    let tokenizer_config = reader.read(
                        &format!("{}/tokenizer_config.json", component.directory()),
                        false,
                    )?;
                    let value = json_object(&tokenizer_config)?;
                    require_string(
                        &value,
                        "processor_class",
                        "Qwen3VLProcessor",
                        &tokenizer_config,
                    )?;
                    require_string(
                        &value,
                        "tokenizer_class",
                        "Qwen2Tokenizer",
                        &tokenizer_config,
                    )?;
                    auxiliary.insert(tokenizer_config.path().to_owned(), tokenizer_config);

                    let tokenizer =
                        reader.read(&format!("{}/tokenizer.json", component.directory()), true)?;
                    tokenizers::Tokenizer::from_bytes(tokenizer.bytes()).map_err(|error| {
                        FerrumError::model(format!("invalid {}: {error}", tokenizer.path()))
                    })?;
                    auxiliary.insert(tokenizer.path().to_owned(), tokenizer);

                    let template = reader.read(
                        &format!("{}/chat_template.jinja", component.directory()),
                        false,
                    )?;
                    let text = std::str::from_utf8(template.bytes()).map_err(|error| {
                        FerrumError::model(format!("{} is not UTF-8: {error}", template.path()))
                    })?;
                    if text.trim().is_empty() {
                        return Err(FerrumError::model(format!("{} is empty", template.path())));
                    }
                    auxiliary.insert(template.path().to_owned(), template);
                }
                ComponentArchitecture::Qwen3VlConditionEncoder => {
                    if let Some(generation) = reader.read_optional(&format!(
                        "{}/generation_config.json",
                        component.directory()
                    ))? {
                        json_object(&generation)?;
                        auxiliary.insert(generation.path().to_owned(), generation);
                    }
                }
                _ => {}
            }
            insert_component(
                &mut components,
                PipelineComponentSources {
                    declaration: component.clone(),
                    config,
                    auxiliary,
                },
            )?;
        }
        let mut digests = vec![model_index.digest().clone()];
        for component in components.values() {
            digests.push(component.config.digest().clone());
            digests.extend(
                component
                    .auxiliary
                    .values()
                    .map(|file| file.digest().clone()),
            );
        }
        let manifest = PipelineSourceManifest::new(declaration.clone(), digests)?;
        Ok(Self {
            root: reader.root().to_owned(),
            model_index,
            declaration,
            components,
            manifest,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn model_index(&self) -> &RetainedMetadataFile {
        &self.model_index
    }

    pub fn declaration(&self) -> &PipelineDeclaration {
        &self.declaration
    }

    /// Access a specific registered component instance, such as `text_encoder`.
    pub fn component(&self, id: &str) -> Option<&PipelineComponentSources> {
        self.components.get(id)
    }

    /// Iterate all component instances serving this role, in stable ID order.
    /// A role is a capability grouping and does not imply a unique instance.
    pub fn components_with_role(
        &self,
        role: ComponentRole,
    ) -> impl Iterator<Item = &PipelineComponentSources> {
        self.components
            .values()
            .filter(move |component| component.declaration.role() == role)
    }

    pub fn manifest(&self) -> &PipelineSourceManifest {
        &self.manifest
    }

    /// Match every consumed file against a separately trusted revision lock.
    /// This proves equality to the caller's lock; it does not authenticate HF
    /// or independently prove who published that lock.
    pub fn verify_metadata_lock(&self, expected: &PipelineMetadataLock) -> Result<()> {
        expected.verify(&self.manifest)
    }
}

fn insert_component(
    components: &mut BTreeMap<String, PipelineComponentSources>,
    component: PipelineComponentSources,
) -> Result<()> {
    use std::collections::btree_map::Entry;

    match components.entry(component.declaration.id().to_owned()) {
        Entry::Vacant(entry) => {
            entry.insert(component);
            Ok(())
        }
        Entry::Occupied(entry) => Err(FerrumError::model(format!(
            "duplicate registered pipeline component instance {:?}",
            entry.key()
        ))),
    }
}

fn json_object(file: &RetainedMetadataFile) -> Result<Value> {
    let value: Value = serde_json::from_slice(file.bytes())
        .map_err(|error| FerrumError::model(format!("invalid {}: {error}", file.path())))?;
    if !value.is_object() {
        return Err(FerrumError::model(format!(
            "{} must contain a JSON object",
            file.path()
        )));
    }
    Ok(value)
}

fn require_string(
    value: &Value,
    key: &str,
    expected: &str,
    file: &RetainedMetadataFile,
) -> Result<()> {
    if value.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(FerrumError::unsupported(format!(
            "{} must declare {key}={expected:?}",
            file.path()
        )));
    }
    Ok(())
}

fn validate_component_config(
    component: &ComponentDeclaration,
    file: &RetainedMetadataFile,
) -> Result<()> {
    let value = json_object(file)?;
    match component.architecture() {
        ComponentArchitecture::Qwen3VlConditionEncoder => {
            let architectures = value.get("architectures").and_then(Value::as_array);
            if !architectures.is_some_and(|items| {
                items.len() == 1 && items[0].as_str() == Some(component.architecture().class_name())
            }) {
                return Err(FerrumError::unsupported(format!(
                    "{} architectures must match declared component {}",
                    file.path(),
                    component.architecture().class_name()
                )));
            }
            require_string(&value, "model_type", "qwen3_vl", file)
        }
        ComponentArchitecture::Qwen3VlProcessor => require_string(
            &value,
            "processor_class",
            component.architecture().class_name(),
            file,
        ),
        _ => require_string(
            &value,
            "_class_name",
            component.architecture().class_name(),
            file,
        ),
    }
}

#[cfg(test)]
mod tests;
