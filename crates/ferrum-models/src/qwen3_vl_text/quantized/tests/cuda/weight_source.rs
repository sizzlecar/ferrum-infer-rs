//! Test-only binding of validated norms to the actual session Host scope.
//! Projection tensors and F8 scale carriers always retain canonical storage.
use super::*;
use ferrum_quantization::ManagedDenseArchiveSource;
pub(super) struct Source<'a> {
    canonical: &'a ComfyCanonicalWeightSource,
    schema: &'a WeightSchema,
    norms: BTreeSet<String>,
    dense: ManagedDenseArchiveSource<'a>,
}
impl<'a> Source<'a> {
    pub(super) fn new(
        checkpoint: &'a Qwen3VLTextComfyCheckpoint<'_>,
        schema: &'a WeightSchema,
        host: &'a HostMemoryAccount,
    ) -> Self {
        Self {
            canonical: checkpoint.source(),
            schema,
            norms: weights::norm_shapes(checkpoint.config().text())
                .into_iter()
                .map(|(name, _)| name)
                .collect(),
            dense: ManagedDenseArchiveSource::new(checkpoint.source().archive(), host),
        }
    }
}
impl WeightComponentSource for Source<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let invalid = |reason: &str| VNextError::InvalidExecutionPlan {
            reason: format!("TE fixture source: {reason}"),
        };
        if !self.schema.components.iter().any(|expected| expected == c) {
            return Err(invalid(
                "component differs from the real canonical Stored schema",
            ));
        }
        let [name] = c.external_names.as_slice() else {
            return Err(invalid("one original tensor required"));
        };
        if self.norms.contains(name) {
            if c.dimensions.len() != 1
                || c.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    })
            {
                return Err(invalid("only validated dense F32 norm vectors can promote"));
            }
            self.dense.component(c)
        } else {
            self.canonical.component(c)
        }
    }
}
