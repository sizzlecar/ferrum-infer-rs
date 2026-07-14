use std::borrow::Cow;

use super::{ElementType, VNextError, WeightComponentSpec, WeightId};

/// Validated physical bytes for one model weight component.
///
/// Borrowed bytes allow mmap-backed dense weights to avoid a host copy.
/// Owned bytes cover format adapters that must repack before device upload.
pub struct WeightComponentPayload<'source> {
    component_id: WeightId,
    external_name: String,
    source_file: String,
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
        let external_name = external_name.into();
        let source_file = source_file.into();
        let bytes = bytes.into();
        let valid_source_file = !source_file.is_empty()
            && !source_file.starts_with('/')
            && !source_file.contains('\\')
            && source_file
                .split('/')
                .all(|component| !matches!(component, "" | "." | ".."));
        let expected_bytes = component.physical_bytes()?;
        if !component
            .external_names
            .iter()
            .any(|candidate| candidate == &external_name)
            || !valid_source_file
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
            external_name,
            source_file,
            dimensions,
            element_type,
            bytes,
        })
    }

    pub fn component_id(&self) -> &WeightId {
        &self.component_id
    }

    pub fn external_name(&self) -> &str {
        &self.external_name
    }

    pub fn source_file(&self) -> &str {
        &self.source_file
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
