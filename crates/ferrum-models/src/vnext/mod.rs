//! Production model-family packages consumed by the vNext planner and runtime.

use std::fs;
use std::path::Path;

use ferrum_interfaces::vnext::{ExternalModelMetadataId, PreparedModelFamily};
use serde_json::Value;

pub mod qwen35;

type PrepareFamily = fn(&Path) -> ferrum_types::Result<PreparedModelFamily>;

struct FamilyLoaderRegistration {
    external_metadata_id: &'static str,
    prepare: PrepareFamily,
}

const FAMILY_LOADERS: &[FamilyLoaderRegistration] = &[FamilyLoaderRegistration {
    external_metadata_id: qwen35::EXTERNAL_METADATA_ID,
    prepare: qwen35::prepare_from_model_dir,
}];

/// Explicit migration result. An unregistered family remains on its current
/// path without letting the engine infer model identity from an architecture
/// enum, model name, backend, or device size.
pub enum ProductionFamilySelection {
    Prepared(PreparedModelFamily),
    Unmigrated {
        external_metadata_id: ExternalModelMetadataId,
    },
}

pub fn prepare_registered_from_model_dir(
    model_dir: &Path,
) -> ferrum_types::Result<ProductionFamilySelection> {
    let external_metadata_id = external_metadata_id_from_model_dir(model_dir)?;
    let loader = FAMILY_LOADERS
        .iter()
        .find(|registration| registration.external_metadata_id == external_metadata_id.as_str());
    match loader {
        Some(registration) => {
            (registration.prepare)(model_dir).map(ProductionFamilySelection::Prepared)
        }
        None => Ok(ProductionFamilySelection::Unmigrated {
            external_metadata_id,
        }),
    }
}

fn external_metadata_id_from_model_dir(
    model_dir: &Path,
) -> ferrum_types::Result<ExternalModelMetadataId> {
    let path = model_dir.join("config.json");
    let raw = fs::read_to_string(&path)
        .map_err(|error| ferrum_types::FerrumError::model(format!("read {path:?}: {error}")))?;
    let config: Value = serde_json::from_str(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(format!("parse {path:?}: {error}")))?;
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ferrum_types::FerrumError::model(
                "config.json must declare exactly one architectures entry for typed family resolution",
            )
        })?;
    let architecture = match architectures.as_slice() {
        [value] => value.as_str().filter(|value| !value.is_empty()),
        _ => None,
    }
    .ok_or_else(|| {
        ferrum_types::FerrumError::model(
            "config.json must declare exactly one non-empty architecture identity",
        )
    })?;
    ExternalModelMetadataId::new(format!("hf.architecture.{architecture}"))
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))
}
