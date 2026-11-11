//! Diagnostic-only F32 Linear islands. The canonical archive remains immutable.
use anyhow::{ensure, Context, Result};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::*;
use half::bf16;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

pub(super) struct Selection {
    components: BTreeMap<WeightId, WeightComponentStreamSpec>,
    pub chunk_bytes: usize,
}

/// Selection follows original operation operands, never tensor-name prefixes.
pub(super) fn select(
    program: &ModelProgram,
    schema: &WeightSchema,
    source: &ComfyCanonicalWeightSource,
    maximum_chunk_bytes: usize,
) -> Result<Selection> {
    let mut selected = BTreeMap::new();
    let references = program
        .weights()
        .iter()
        .map(|w| (&w.value_id, w))
        .collect::<BTreeMap<_, _>>();
    for node in program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .filter(|n| n.operation_id.as_str() == AFFINE_LINEAR_F32_OPERATION_ID)
    {
        ensure!(
            node.inputs.len() == 3,
            "F32 affine must have exact input/weight/bias ports"
        );
        for value in &node.inputs[1..] {
            let reference = references
                .get(value)
                .context("F32 affine operand is not a Stored weight")?;
            let tensor = schema
                .tensors
                .iter()
                .find(|t| t.id == reference.weight_id)
                .context("weight tensor missing")?;
            let PhysicalWeightLayout::Dense { component_id } = &tensor.physical_layout else {
                anyhow::bail!("F32 affine diagnostic cannot transform a quantized weight")
            };
            let component = schema
                .components
                .iter()
                .find(|c| &c.id == component_id)
                .context("weight component missing")?;
            ensure!(
                tensor.logical_element_type == ElementType::F32
                    && reference.tensor.element_type == ElementType::F32
                    && component.encoding
                        == (WeightEncoding::Dense {
                            element_type: ElementType::F32
                        })
                    && component.dimensions == tensor.dimensions
                    && component.external_names.len() == 1,
                "selected operand must be one exact Dense F32 component"
            );
            let actual = source.archive().tensor(&component.external_names[0])?;
            ensure!(
                actual.element_type() == Some(ElementType::F32)
                    && actual.shape() == component.dimensions
                    && actual.bytes().len() as u64 == component.physical_bytes()?,
                "selected archive tensor must already be original F32 with exact shape"
            );
            selected.insert(
                component.id.clone(),
                (component.clone(), actual.source_file().to_owned()),
            );
        }
    }
    ensure!(
        !selected.is_empty(),
        "no original F32 affine islands were selected"
    );
    // Resolve all aliases through physical component identity, including other
    // value IDs referring to the same bytes. No quantized operand may be rounded.
    for node in program.blocks().iter().flat_map(|b| &b.nodes) {
        for (port, value) in node.inputs.iter().enumerate() {
            let Some(reference) = references.get(value) else {
                continue;
            };
            let tensor = schema
                .tensors
                .iter()
                .find(|t| t.id == reference.weight_id)
                .context("weight tensor missing")?;
            if let PhysicalWeightLayout::Dense { component_id } = &tensor.physical_layout {
                if selected.contains_key(component_id) {
                    ensure!(
                        node.operation_id.as_str() == AFFINE_LINEAR_F32_OPERATION_ID
                            && matches!(port, 1 | 2),
                        "selected F32 component is also used outside an affine weight/bias port"
                    );
                }
            }
        }
    }
    let largest = selected
        .values()
        .map(|(c, _)| c.physical_bytes())
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .context("empty selection")?;
    let chunk_bytes = maximum_chunk_bytes.min(usize::try_from(largest)?);
    let chunk_bytes = chunk_bytes - chunk_bytes % 4;
    ensure!(
        chunk_bytes > 0,
        "source chunk must hold a complete F32 element"
    );
    let components = selected
        .into_iter()
        .map(|(id, (c, file))| {
            Ok((
                id,
                WeightComponentStreamSpec::new(&c, vec![file], chunk_bytes)?,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Selection {
        components,
        chunk_bytes,
    })
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("diagnostic BF16-roundtrip source: {}", reason.into()),
    }
}

pub(super) fn round_into(input: &[u8], output: &mut [u8]) -> Result<(), VNextError> {
    if input.len() != output.len() || input.is_empty() || input.len() % 4 != 0 {
        return Err(invalid("conversion requires equal positive F32 extents"));
    }
    for (a, b) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let original = f32::from_le_bytes(a.try_into().unwrap());
        let rounded = bf16::from_f32(original).to_f32();
        if !original.is_finite() || !rounded.is_finite() {
            return Err(invalid(
                "selected weight contains nonfinite or BF16-overflowing values",
            ));
        }
        b.copy_from_slice(&rounded.to_le_bytes());
    }
    Ok(())
}

#[derive(Default)]
struct Progress {
    bytes: usize,
    ranges: usize,
    original: Sha256,
    rounded: Sha256,
}
pub(super) struct RoundtripSource<'a> {
    original: &'a ComfyCanonicalWeightSource,
    selection: &'a Selection,
    host: HostMemoryAccount,
    plan_hash: PlanHash,
    family_fingerprint: String,
    ready: AtomicBool,
    failed: AtomicBool,
    progress: Mutex<BTreeMap<WeightId, Progress>>,
}
impl<'a> RoundtripSource<'a> {
    #[cfg(feature = "cuda")]
    pub fn new(
        original: &'a ComfyCanonicalWeightSource,
        selection: &'a Selection,
        host: &HostMemoryAccount,
        plan: &ExecutionPlan,
    ) -> Result<Self> {
        ensure!(
            host.backing_bytes(selection.chunk_bytes)?
                <= host.limits().class_bytes(HostMemoryClass::WeightBacking),
            "single source chunk physical backing exceeds Host class"
        );
        Ok(Self {
            original,
            selection,
            host: host.clone(),
            plan_hash: plan.plan_hash().clone(),
            family_fingerprint: plan.payload().prepared_family_fingerprint().to_owned(),
            ready: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            progress: Mutex::new(BTreeMap::new()),
        })
    }
    fn spec(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<Option<&WeightComponentStreamSpec>, VNextError> {
        let Some(spec) = self.selection.components.get(&component.id) else {
            return Ok(None);
        };
        if spec.component() != component {
            return Err(invalid("selected component metadata drift"));
        }
        Ok(Some(spec))
    }
    fn materialize(
        &self,
        spec: &WeightComponentStreamSpec,
        offset: usize,
        length: usize,
    ) -> Result<ManagedHostBytes, VNextError> {
        if !self.ready.load(Ordering::Acquire) || self.failed.load(Ordering::Acquire) {
            return Err(invalid("source is not in its one admitted initialization"));
        }
        let result = (|| {
            let component = spec.component();
            let actual = self
                .original
                .archive()
                .tensor(&component.external_names[0])
                .map_err(|e| invalid(e.to_string()))?;
            let end = offset
                .checked_add(length)
                .filter(|&e| e <= actual.bytes().len())
                .ok_or_else(|| invalid("range exceeds original source"))?;
            if length == 0
                || offset % 4 != 0
                || length % 4 != 0
                || length > spec.maximum_range_bytes()
            {
                return Err(invalid("range is empty, unaligned, or exceeds one chunk"));
            }
            let mut progress = self
                .progress
                .lock()
                .map_err(|_| invalid("progress owner poisoned"))?;
            let progress = progress.entry(component.id.clone()).or_default();
            if progress.bytes != offset {
                return Err(invalid(
                    "range coverage must be ordered, unique and gap-free",
                ));
            }
            let mut bytes = self
                .host
                .allocate_zeroed(HostMemoryClass::WeightBacking, length)
                .map_err(|e| invalid(e.to_string()))?;
            round_into(&actual.bytes()[offset..end], bytes.as_mut())?;
            progress.original.update(&actual.bytes()[offset..end]);
            progress.rounded.update(bytes.as_ref());
            progress.bytes = end;
            progress.ranges += 1;
            Ok(bytes.freeze())
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
    pub fn receipt(&self) -> Result<serde_json::Value> {
        ensure!(
            self.ready.load(Ordering::Acquire) && !self.failed.load(Ordering::Acquire),
            "transformed source did not complete"
        );
        let progress = self
            .progress
            .lock()
            .map_err(|_| anyhow::anyhow!("source progress poisoned"))?;
        let components = self.selection.components.iter().map(|(id, spec)| {
            let p = progress.get(id).context("selected component was not uploaded")?;
            ensure!(p.bytes as u64 == spec.component().physical_bytes()?, "selected component upload coverage incomplete");
            Ok(json!({"component_id":id,"external_name":spec.component().external_names[0],"bytes":p.bytes,"ranges":p.ranges,
                "original_f32_sha256":format!("{:x}",p.original.clone().finalize()),"bf16_roundtrip_f32_sha256":format!("{:x}",p.rounded.clone().finalize())}))
        }).collect::<Result<Vec<_>>>()?;
        Ok(
            json!({"policy":"bf16_roundtrip","selection":"original InstallFullWeights AFFINE_LINEAR_F32 weight/bias operands only",
            "canonical_archive_bytes":false,"chunk_bytes":self.selection.chunk_bytes,"physical_chunk_bytes":self.host.backing_bytes(self.selection.chunk_bytes)?,"components":components}),
        )
    }
}
impl WeightComponentSource for RoundtripSource<'_> {
    fn streaming_component(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<Option<WeightComponentStreamSpec>, VNextError> {
        match self.spec(component)? {
            Some(spec) => Ok(Some(spec.clone())),
            None => self.original.streaming_component(component),
        }
    }
    fn validate_static_initialization(
        &self,
        context: &StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        if self.ready.swap(true, Ordering::AcqRel) {
            self.failed.store(true, Ordering::Release);
            return Err(invalid("source initialization is one-use"));
        }
        let result = (|| {
            let valid = context.matches_host_account(&self.host)
                && context.plan().plan_hash() == &self.plan_hash
                && context.family().fingerprint()? == self.family_fingerprint
                && context.source_retention() == StaticWeightSourceRetention::FencedCopy
                && context.supports_managed_source_upload()
                && context.maximum_source_chunk_bytes() >= 4;
            if !valid {
                return Err(invalid(
                    "requires original plan/family/Host and managed FencedCopy",
                ));
            }
            self.original.validate_static_initialization(context)
        })();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
    fn component_range(
        &self,
        request: &WeightComponentRangeRequest,
    ) -> Result<WeightComponentRange, VNextError> {
        let Some(spec) = self.spec(request.component())? else {
            return self.original.component_range(request);
        };
        if spec != request.spec() {
            self.failed.store(true, Ordering::Release);
            return Err(invalid("range spec differs from frozen selection"));
        }
        WeightComponentRange::new(
            request,
            self.materialize(spec, request.offset_bytes(), request.length_bytes())?,
        )
    }
    fn component<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        if self.spec(component)?.is_some() {
            return Err(invalid(
                "selected component cannot bypass transformation via full payload",
            ));
        }
        self.original.component(component)
    }
    fn component_segments<'s>(
        &'s self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentSegments<'s>, VNextError> {
        if self.spec(component)?.is_some() {
            return Err(invalid(
                "selected component cannot bypass transformation via mapped segments",
            ));
        }
        self.original.component_segments(component)
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
