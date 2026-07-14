//! Production model-family packages consumed by the vNext planner and runtime.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ExternalModelMetadataId, PreparedModelFamily, WeightComponentSource,
};
use serde_json::Value;

pub mod qwen35;

type PrepareModel = fn(&Path) -> ferrum_types::Result<PreparedProductionModel>;

struct ModelLoaderRegistration {
    external_metadata_id: &'static str,
    prepare: PrepareModel,
}

const MODEL_LOADERS: &[ModelLoaderRegistration] = &[ModelLoaderRegistration {
    external_metadata_id: qwen35::EXTERNAL_METADATA_ID,
    prepare: qwen35::prepare_from_model_dir,
}];

/// One immutable semantic family and the exact indexed weight source used to
/// initialize its execution plan. Keeping them together prevents product code
/// from resolving a family from one model directory and loading bytes from
/// another.
pub struct PreparedProductionModel {
    family: PreparedModelFamily,
    weights: Arc<dyn WeightComponentSource>,
}

impl PreparedProductionModel {
    pub(super) fn new(
        family: PreparedModelFamily,
        weights: impl WeightComponentSource + 'static,
    ) -> Self {
        Self {
            family,
            weights: Arc::new(weights),
        }
    }

    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }

    pub fn weights(&self) -> &dyn WeightComponentSource {
        self.weights.as_ref()
    }

    pub fn weight_source(&self) -> &Arc<dyn WeightComponentSource> {
        &self.weights
    }
}

/// Explicit migration result. An unregistered family remains on its current
/// path without letting the engine infer model identity from an architecture
/// enum, model name, backend, or device size.
pub enum ProductionModelSelection {
    Prepared(PreparedProductionModel),
    Unmigrated {
        external_metadata_id: ExternalModelMetadataId,
    },
}

pub fn prepare_registered_model_from_dir(
    model_dir: &Path,
) -> ferrum_types::Result<ProductionModelSelection> {
    let external_metadata_id = external_metadata_id_from_model_dir(model_dir)?;
    let loader = MODEL_LOADERS
        .iter()
        .find(|registration| registration.external_metadata_id == external_metadata_id.as_str());
    match loader {
        Some(registration) => {
            (registration.prepare)(model_dir).map(ProductionModelSelection::Prepared)
        }
        None => Ok(ProductionModelSelection::Unmigrated {
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
