use super::*;

/// Complete component identity and source order for bounded range production.
/// This is metadata, not a complete payload or authority to bypass a weight
/// materializer. Only core's live identity materializer permits passthrough.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightComponentStreamSpec {
    component: WeightComponentSpec,
    source_files: Vec<String>,
    maximum_range_bytes: usize,
}

impl WeightComponentStreamSpec {
    pub fn new(
        component: &WeightComponentSpec,
        source_files: Vec<String>,
        maximum_range_bytes: usize,
    ) -> Result<Self, VNextError> {
        let element = component.physical_element_type().size_bytes();
        let full_bytes = component.physical_bytes()?;
        if full_bytes == 0
            || maximum_range_bytes == 0
            || maximum_range_bytes as u64 % element != 0
            || !valid_ordered_sources(component, &component.external_names, &source_files)
        {
            return Err(invalid(
                "streamed component has invalid extent or ordered source metadata",
            ));
        }
        Ok(Self {
            component: component.clone(),
            source_files,
            maximum_range_bytes,
        })
    }

    pub fn component(&self) -> &WeightComponentSpec {
        &self.component
    }

    pub fn source_files(&self) -> &[String] {
        &self.source_files
    }

    pub const fn maximum_range_bytes(&self) -> usize {
        self.maximum_range_bytes
    }
}

/// Core-created request for one positive element-aligned interval in the
/// original complete component. It cannot be constructed from wire metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightComponentRangeRequest {
    spec: WeightComponentStreamSpec,
    offset_bytes: usize,
    length_bytes: usize,
}

impl WeightComponentRangeRequest {
    pub(crate) fn new(
        spec: &WeightComponentStreamSpec,
        offset_bytes: usize,
        length_bytes: usize,
    ) -> Result<Self, VNextError> {
        let element = spec.component.physical_element_type().size_bytes();
        let end = offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid("streamed component range overflows the host address space"))?;
        if length_bytes == 0
            || offset_bytes as u64 % element != 0
            || length_bytes as u64 % element != 0
            || length_bytes > spec.maximum_range_bytes
            || end as u64 > spec.component.physical_bytes()?
        {
            return Err(invalid(
                "streamed component range is empty, unaligned or out of bounds",
            ));
        }
        Ok(Self {
            spec: spec.clone(),
            offset_bytes,
            length_bytes,
        })
    }

    pub fn spec(&self) -> &WeightComponentStreamSpec {
        &self.spec
    }

    pub fn component(&self) -> &WeightComponentSpec {
        self.spec.component()
    }

    pub const fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> usize {
        self.length_bytes
    }
}

/// Owned accounted bytes for precisely one requested interval. The full
/// component contract remains in the request; a range is never a full payload.
pub struct WeightComponentRange {
    request: WeightComponentRangeRequest,
    bytes: ManagedHostBytes,
}

impl WeightComponentRange {
    pub fn new(
        request: &WeightComponentRangeRequest,
        bytes: ManagedHostBytes,
    ) -> Result<Self, VNextError> {
        if bytes.class() != HostMemoryClass::WeightBacking
            || bytes.bytes().len() != request.length_bytes
        {
            return Err(invalid(
                "streamed component range has the wrong Host class or length",
            ));
        }
        Ok(Self {
            request: request.clone(),
            bytes,
        })
    }

    pub(crate) fn validate_and_take(
        self,
        expected: &WeightComponentRangeRequest,
        host: &crate::HostMemoryAccount,
    ) -> Result<ManagedHostBytes, VNextError> {
        if &self.request != expected || !host.owns_in_scope(&self.bytes) {
            return Err(invalid(
                "streamed range differs from its request or original Host scope",
            ));
        }
        Ok(self.bytes)
    }
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
