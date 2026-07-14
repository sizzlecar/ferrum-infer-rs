use std::borrow::Cow;

use super::{ElementType, VNextError, WeightComponentSpec, WeightId};

/// Validated physical bytes for one model weight component.
///
/// Borrowed bytes allow mmap-backed dense weights to avoid a host copy.
/// Owned bytes cover format adapters that must repack before device upload.
pub struct WeightComponentPayload<'source> {
    component_id: WeightId,
    external_names: Vec<String>,
    source_files: Vec<String>,
    dimensions: Vec<u64>,
    element_type: ElementType,
    bytes: Cow<'source, [u8]>,
}

impl<'source> WeightComponentPayload<'source> {
    pub fn new(
        component: &WeightComponentSpec,
        external_name: impl Into<String>,
        source_file: impl Into<String>,
        dimensions: Vec<u64>,
        element_type: ElementType,
        bytes: impl Into<Cow<'source, [u8]>>,
    ) -> Result<Self, VNextError> {
        Self::from_ordered_sources(
            component,
            vec![external_name.into()],
            vec![source_file.into()],
            dimensions,
            element_type,
            bytes,
        )
    }

    /// Construct a payload materialized from multiple ordered checkpoint
    /// tensors. Multi-source order is part of the component schema, so packed
    /// projections cannot silently swap their logical partitions.
    pub fn from_ordered_sources(
        component: &WeightComponentSpec,
        external_names: Vec<String>,
        source_files: Vec<String>,
        dimensions: Vec<u64>,
        element_type: ElementType,
        bytes: impl Into<Cow<'source, [u8]>>,
    ) -> Result<Self, VNextError> {
        let bytes = bytes.into();
        let valid_source_file = |source_file: &str| {
            !source_file.is_empty()
                && !source_file.starts_with('/')
                && !source_file.contains('\\')
                && source_file
                    .split('/')
                    .all(|component| !matches!(component, "" | "." | ".."))
        };
        let sources_match = !external_names.is_empty()
            && external_names == component.external_names
            && external_names.len() == source_files.len()
            && source_files.iter().all(|file| valid_source_file(file));
        let expected_bytes = component.physical_bytes()?;
        if !sources_match
            || dimensions != component.dimensions
            || element_type != component.physical_element_type()
            || u64::try_from(bytes.len()).ok() != Some(expected_bytes)
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "weight component `{}` payload differs from its schema identity, source, shape, type, or byte length",
                    component.id
                ),
            });
        }
        Ok(Self {
            component_id: component.id.clone(),
            external_names,
            source_files,
            dimensions,
            element_type,
            bytes,
        })
    }

    pub fn component_id(&self) -> &WeightId {
        &self.component_id
    }

    pub fn external_name(&self) -> &str {
        &self.external_names[0]
    }

    pub fn source_file(&self) -> &str {
        &self.source_files[0]
    }

    pub fn external_names(&self) -> &[String] {
        &self.external_names
    }

    pub fn source_files(&self) -> &[String] {
        &self.source_files
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Backend-neutral source of schema-addressed physical weight components.
/// Implementations own file-format discovery, validation, and optional
/// repacking; resource initialization owns placement and device submission.
pub trait WeightComponentSource: Send + Sync {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{WeightComponentRole, WeightEncoding};

    fn packed_component() -> WeightComponentSpec {
        WeightComponentSpec {
            id: WeightId::new("component.test.gate_up").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec!["gate.weight".to_owned(), "up.weight".to_owned()],
            dimensions: vec![2, 2, 2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }
    }

    #[test]
    fn packed_payload_preserves_ordered_source_identity() {
        let component = packed_component();
        let payload = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.clone(),
            vec![
                "model-1.safetensors".to_owned(),
                "model-2.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![0_u8; 16],
        )
        .unwrap();
        assert_eq!(payload.external_names(), component.external_names);
        assert_eq!(payload.source_files().len(), 2);

        let error = WeightComponentPayload::from_ordered_sources(
            &component,
            component.external_names.iter().rev().cloned().collect(),
            vec![
                "model-2.safetensors".to_owned(),
                "model-1.safetensors".to_owned(),
            ],
            component.dimensions.clone(),
            ElementType::F16,
            vec![0_u8; 16],
        )
        .err()
        .expect("source order is part of the packed component identity");
        assert!(error.to_string().contains("differs from its schema"));
    }
}
