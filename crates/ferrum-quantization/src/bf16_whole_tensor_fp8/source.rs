use super::*;
use crate::SafetensorsArchive;

/// Borrow an archive while forbidding its optional dtype conversion for BF16
/// component requests. Validation happens before `component()` can allocate a
/// transcoding buffer. Non-BF16 requests retain the archive's existing behavior.
/// Immutable mmap residency remains outside Host backing-byte accounting.
pub struct ExactBf16SafetensorsSource<'archive> {
    archive: &'archive SafetensorsArchive,
}
impl<'archive> ExactBf16SafetensorsSource<'archive> {
    pub fn new(archive: &'archive SafetensorsArchive) -> Self {
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
                if tensor.dtype() != safetensors::Dtype::BF16 {
                    return Err(invalid(
                        "BF16 source requests require actual BF16 safetensors payloads",
                    ));
                }
            }
        }
        Ok(())
    }
}
impl WeightComponentSource for ExactBf16SafetensorsSource<'_> {
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
