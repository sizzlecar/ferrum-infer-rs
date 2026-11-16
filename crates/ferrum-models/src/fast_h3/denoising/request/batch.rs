//! Checked model metadata and one borrowed execution membership. Neither type
//! installs weights, owns a device session, or schedules requests.
use super::*;

#[derive(Clone)]
pub struct FastH3DenoisingBatchPlan {
    pub(super) config: FastH3DenoisingConfig,
    pub(super) options: ProgramPlanCompileOptions,
    // Exact declaration/interpretation equality, not payload authentication or
    // an installed-device-weight owner. Core must still admit each session.
    pub(super) schema_fingerprint: String,
    pub(super) regions: Arc<CheckedVsaRegions>,
}
impl FastH3DenoisingBatchPlan {
    /// Cold preparation reuses its one actual composition for both the source
    /// interpretation identity and compiler geometry. No graph is rebuilt by
    /// bind(), inputs(), output partitioning or solver publication.
    pub fn prepare(
        config: FastH3DenoisingConfig,
        limits: FastH3DenoisingLimits,
        regions: Arc<CheckedVsaRegions>,
    ) -> Result<Self, VNextError> {
        let built = compose(&config, limits)?;
        Self::validate_regions(&config, &regions)?;
        Ok(Self {
            config,
            options: built.options,
            schema_fingerprint: built.schema.fingerprint()?,
            regions,
        })
    }

    /// Prepare the semantic family and execution geometry from one composition.
    /// The original typed provider remains available for independent loading
    /// and rehydration; this path preserves its exact family identity.
    pub fn prepare_with_family(
        config: FastH3DenoisingConfig,
        limits: FastH3DenoisingLimits,
        regions: Arc<CheckedVsaRegions>,
    ) -> Result<(Self, PreparedModelFamily), VNextError> {
        FastH3DenoisingFamily::new(limits)?;
        let built = compose(&config, limits)?;
        Self::validate_regions(&config, &regions)?;
        let schema_fingerprint = built.schema.fingerprint()?;
        let (family, options) =
            built.prepare_family(&config, id(FastH3DenoisingFamily::METADATA_ID)?)?;
        Ok((
            Self {
                config,
                options,
                schema_fingerprint,
                regions,
            },
            family,
        ))
    }

    fn validate_regions(
        config: &FastH3DenoisingConfig,
        regions: &CheckedVsaRegions,
    ) -> Result<(), VNextError> {
        if regions.digest() != config.backbone.region_digest()
            || u64::from(regions.shape().batches()) != config.backbone.batches()
        {
            return Err(invalid(
                "batch native regions differ from the source-bound configuration",
            ));
        }
        Ok(())
    }
    pub fn configuration(&self) -> &FastH3DenoisingConfig {
        &self.config
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn bind<'a>(
        &'a self,
        requests: Vec<&'a mut FastH3DenoisingRequest>,
    ) -> Result<FastH3DenoisingBatch<'a>, VNextError> {
        if requests.len() as u64 != self.config.backbone.batches() || requests.is_empty() {
            return Err(invalid(
                "batch membership differs from the prepared geometry",
            ));
        }
        let first = &requests[0];
        let batch_regions = self.regions.checked().map_err(invalid)?;
        let mut identities = BTreeSet::new();
        for (slot, request) in requests.iter().enumerate() {
            let regions = request
                .batch_regions
                .as_ref()
                .ok_or_else(|| invalid("request has no original checked native batch geometry"))?
                .checked()
                .map_err(invalid)?;
            if request.is_complete()
                || !identities.insert(request.request().clone())
                || self.schema_fingerprint != request.schema_fingerprint
                || self.config.profile_id() != request.config.profile_id()
                || !self
                    .config
                    .packing
                    .same_request_partition(&request.config.packing)
                || self.config.backbone.layers() != request.config.backbone.layers()
                || self.config.backbone.rotary_dim() != request.config.backbone.rotary_dim()
                || request.config.backbone.batches() != 1
                || request.plan.schedule() != first.plan.schedule()
                || !Arc::ptr_eq(&request.tables, &first.tables)
                || !Arc::ptr_eq(&request.output_table, &first.output_table)
                || !first.host.owns(request.video().bytes())
                || !first.host.owns(request.audio().bytes())
                || !first.host.owns(request.text.bytes())
                || request.rotary.iter().any(|v| !first.host.owns(v.bytes()))
                || regions.shape().heads() != batch_regions.shape().heads()
                || regions.shape().query_blocks() != batch_regions.shape().query_blocks()
                || regions.shape().key_blocks() != batch_regions.shape().key_blocks()
                || regions.regions() != batch_regions.regions()
                || regions.half_valid() != batch_regions.half_valid()
            {
                return Err(invalid("batch member differs in source interpretation, layout, schedule, conditioning installation, ownership or request identity"));
            }
            // Static tag roles are checked against this exact slot. Evaluation
            // remains request-local and is deliberately absent from this test.
            // Cold block validation equates norm1/norm2 routing; the final FMA
            // uses norm2's same runs. No unvalidated second routing is skipped.
            for (batch, single) in self
                .config
                .backbone
                .blocks()
                .iter()
                .zip(request.config.backbone.blocks())
            {
                if !batch.same_numerical_geometry(single) {
                    return Err(invalid(
                        "batch member differs in normalization or QK/RoPE numerical geometry",
                    ));
                }
                let batch = batch.modulation_config();
                let single = single.modulation_config();
                let base = 5 * slot as u64;
                if !batch
                    .time_values()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(single.time_values().iter().map(|v| v.to_bits()))
                    || !batch
                        .parameter_row_runs()
                        .filter(|(_, r)| *r >= base && *r < base + 5)
                        .map(|(n, r)| (n, r - base))
                        .eq(single.parameter_row_runs())
                {
                    return Err(invalid(
                        "batch slot differs from the member's checked time/tag routing",
                    ));
                }
            }
        }
        Ok(FastH3DenoisingBatch {
            plan: self,
            requests,
        })
    }
}

/// Slots belong only to this borrow. A request may join a later compatible
/// batch in another slot without reconstructing or advancing its continuation.
/// The caller must bind the outputs of its submitted Core execution to this
/// membership at its completion fence; matching labels alone is not evidence.
pub struct FastH3DenoisingBatch<'a> {
    plan: &'a FastH3DenoisingBatchPlan,
    requests: Vec<&'a mut FastH3DenoisingRequest>,
}
impl FastH3DenoisingBatch<'_> {
    pub fn len(&self) -> usize {
        self.requests.len()
    }
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
    pub fn member(&self, slot: usize) -> Result<(&RequestIdentity, usize), VNextError> {
        let request = self
            .requests
            .get(slot)
            .ok_or_else(|| invalid("batch slot outside membership"))?;
        Ok((request.request(), request.evaluation()))
    }
    pub fn inputs(&self) -> Result<Vec<ManagedHostTensor>, VNextError> {
        self.inputs_in(&self.requests[0].host)
    }
    /// Admit transient joined inputs under the batch's own Host scope. Request
    /// continuations retain their original independent scopes and owners.
    pub fn inputs_in(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        let first = &self.requests[0];
        if self.requests.iter().any(|r| !host.same_parent(&r.host)) {
            return Err(invalid("batch input scope belongs to another Host account"));
        }
        let evaluations = self
            .requests
            .iter()
            .map(|r| r.evaluation())
            .collect::<Vec<_>>();
        let mut inputs = self.plan.config.backbone.managed_role_tables(
            &first.tables,
            &first.schedule,
            &evaluations,
            host,
        )?;
        for (name, index) in [
            (FastH3DenoisingFamily::VIDEO_INPUT_ID, 0),
            (FastH3DenoisingFamily::AUDIO_INPUT_ID, 1),
            (FastH3DenoisingFamily::TEXT_INPUT_ID, 2),
            (FastH3DenoisingFamily::COSINE_ID, 3),
            (FastH3DenoisingFamily::SINE_ID, 4),
        ] {
            let parts = self
                .requests
                .iter()
                .map(|r| match index {
                    0 => r.video(),
                    1 => r.audio(),
                    2 => &r.text,
                    3 => &r.rotary[0],
                    _ => &r.rotary[1],
                })
                .collect::<Vec<_>>();
            inputs.push(join_inputs(
                host,
                id(name)?,
                &self.plan.options.tensor_specs()[&id(name)?],
                &parts,
            )?);
        }
        Ok(inputs)
    }
    pub fn final_output_inputs(
        &self,
        slot: usize,
        config: &FastH3FinalOutputConfig,
        output: &TensorExecutionOutput,
    ) -> Result<Vec<ManagedHostTensor>, VNextError> {
        self.requests
            .get(slot)
            .ok_or_else(|| invalid("batch slot outside membership"))?
            .final_output_rows(config, output, self.requests.len() as u64, slot)
    }
    /// The exact borrowed member supplies its own evaluation and modulation;
    /// the declared row view selects only that member from the joint output.
    pub fn final_output_state_inputs(
        &self,
        slot: usize,
        config: &FastH3FinalOutputConfig,
    ) -> Result<(Vec<ManagedHostTensor>, TensorRequestStateMapping), VNextError> {
        let request = self
            .requests
            .get(slot)
            .ok_or_else(|| invalid("batch slot outside membership"))?;
        let inputs = request.final_output_state_inputs(config)?;
        let length = request.config.backbone.packed_rows();
        let start = length
            .checked_mul(slot as u64)
            .ok_or_else(|| invalid("prediction row offset overflow"))?;
        Ok((
            inputs,
            TensorRequestStateMapping {
                source_output: id(FastH3DenoisingFamily::OUTPUT_ID)?,
                destination_state: id(FastH3FinalOutputFamily::INPUT_ID)?,
                selection: TensorOutputSelection::LeadingAxis { start, length },
            },
        ))
    }

    /// All members validate and admit their complete video/audio successors
    /// before publication. A failed later member drops prior guards unchanged.
    pub fn advance(self, predictions: &[Vec<TensorExecutionOutput>]) -> anyhow::Result<()> {
        if predictions.len() != self.requests.len() {
            return Err(invalid("prediction partitions differ from exact batch membership").into());
        }
        let mut successors = Vec::with_capacity(self.requests.len());
        for (request, outputs) in self.requests.into_iter().zip(predictions) {
            successors.push(request.prepare_advance(outputs)?);
        }
        for successor in successors {
            successor.commit();
        }
        Ok(())
    }
}
fn join_inputs(
    host: &HostMemoryAccount,
    name: ProgramValueId,
    expected: &ProgramTensorSpec,
    parts: &[&ManagedHostTensor],
) -> Result<ManagedHostTensor, VNextError> {
    let mut one = expected.dimensions.clone();
    if parts.is_empty() || one.is_empty() || one[0] % parts.len() as u64 != 0 {
        return Err(invalid(
            "batch input has no exact leading request partition",
        ));
    }
    one[0] /= parts.len() as u64;
    for part in parts {
        if !host.owns(part.bytes())
            || part.bytes().class() != HostMemoryClass::InputPreprocess
            || part.tensor().dimensions() != one
            || part.tensor().element_type() != expected.element_type
            || part.tensor().layout() != &expected.layout
        {
            return Err(invalid(
                "request input differs from its checked batch partition",
            ));
        }
    }
    if parts.len() == 1 {
        return ManagedHostTensor::new(name, parts[0].tensor().clone(), parts[0].bytes().clone());
    }
    let spec = ResolvedTensorSpec::new(
        expected.dimensions.clone(),
        expected.element_type,
        expected.layout.clone(),
    )?;
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(spec.minimum_storage_bytes()?).map_err(invalid)?,
        )
        .map_err(invalid)?;
    let mut offset = 0;
    for part in parts {
        let len = part.bytes().len();
        bytes.as_mut()[offset..offset + len].copy_from_slice(part.bytes().bytes());
        offset += len;
    }
    ManagedHostTensor::new(name, spec, bytes.freeze())
}
