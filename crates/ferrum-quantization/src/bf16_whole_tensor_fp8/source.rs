use super::*;
use crate::TensorArchive;

/// Borrow an archive while forbidding its optional dtype conversion for BF16
/// component requests. Validation happens before `component()` can allocate a
/// transcoding buffer. Non-BF16 requests retain the archive's existing behavior.
/// Immutable mmap residency remains outside Host backing-byte accounting.
pub struct ExactBf16TensorSource<'archive> {
    archive: &'archive dyn TensorArchive,
}
impl<'archive> ExactBf16TensorSource<'archive> {
    pub fn new(archive: &'archive dyn TensorArchive) -> Self {
        Self { archive }
    }
    fn validate(&self, component: &WeightComponentSpec) -> Result<(), VNextError> {
        if component.encoding
            == (WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            })
        {
            for name in &component.external_names {
                let tensor = self
                    .archive
                    .tensor(name)
                    .map_err(|e| invalid(e.to_string()))?;
                if tensor.element_type() != Some(ElementType::Bf16) {
                    return Err(invalid("BF16 source requests require actual BF16 payloads"));
                }
            }
        }
        Ok(())
    }
}
impl WeightComponentSource for ExactBf16TensorSource<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.validate(component)?;
        self.archive.component(component)
    }
    fn component_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentSegments<'source>, VNextError> {
        self.validate(component)?;
        self.archive.component_segments(component)
    }
}
