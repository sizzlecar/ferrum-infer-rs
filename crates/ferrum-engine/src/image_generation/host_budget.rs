//! Whole-request Host grants precede the first device upload. Each class uses
//! a peak of simultaneously live owners, with every distinct backing padded
//! separately. Returned images keep their ordinary parent charge after close.
use super::*;
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionFamily;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExecutionHostBudget {
    pub input: usize,
    pub output: usize,
    pub staging: usize,
    pub weights: usize,
}

impl ExecutionHostBudget {
    pub(super) fn claims(self) -> [(HostMemoryClass, usize); 4] {
        [
            (HostMemoryClass::InputPreprocess, self.input),
            (HostMemoryClass::Output, self.output),
            (HostMemoryClass::TransferStaging, self.staging),
            (HostMemoryClass::WeightBacking, self.weights),
        ]
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn prepare_execution_host(
        &self,
        prepared: &mut PreparedRequest<R>,
    ) -> anyhow::Result<()> {
        let (budget, backing) = self.execution_host_budget(prepared)?;
        self.execution.host.check_scope_capacity(&budget.claims())?;
        prepared.host_budget = budget;
        prepared.host_backing = backing;
        Ok(())
    }

    pub(super) fn reserve_execution_host(
        &self,
        prepared: &PreparedRequest<R>,
    ) -> anyhow::Result<HostMemoryScope> {
        let budget = prepared.host_budget;
        tracing::debug!(?budget, "image complete execution Host grant");
        Ok(self.execution.host.reserve_scope(&budget.claims())?)
    }

    fn execution_host_budget(
        &self,
        p: &PreparedRequest<R>,
    ) -> anyhow::Result<(ExecutionHostBudget, Vec<HostBackingPreparation>)> {
        let host = &self.execution.host;
        let batch = p.prompts.batch();
        let latent = host.backing_bytes(p.latent_shape.byte_len())?;
        let denoise = ports(&p.denoise_stage)?;
        let text = input_bytes(host, denoise, QwenImage21TransformerFamily::TEXT_ID)?;
        let reference = optional_input_bytes(
            host,
            denoise,
            QwenImage21TransformerFamily::REFERENCE_LATENTS_ID,
        )?;
        // Every Euler step allocates one successor while its current sample is
        // still live. Repack and VAE denormalization have the same two owners.
        let mut budget = ExecutionHostBudget {
            input: sum_ports(host, ports(&p.text_stage)?.inputs().iter())?
                .max(add(&[sum_ports(host, denoise.inputs().iter())?, latent])?)
                .max(add(&[text, reference, latent, host.backing_bytes(16)?])?)
                .max(add(&[text, reference, latent, latent])?),
            output: sum_ports(host, ports(&p.text_stage)?.outputs().iter())?
                .max(sum_ports(host, ports(&p.noise_stage)?.outputs().iter())?),
            ..ExecutionHostBudget::default()
        };
        if p.edit.is_none() && p.prefix_decode.is_none() {
            budget.input = budget
                .input
                .max(full_generation_input_peak(host, denoise, batch)?);
        }
        // Prefix KV outputs are copied directly into Request state on device;
        // only the velocity readback consumes Host Output credit.
        let cache_outputs = if p.prefix_decode.is_some() {
            p.transformer.prefix_cache_ports()?
        } else {
            Vec::new()
        };
        budget.output = budget.output.max(sum_ports(
            host,
            denoise.outputs().iter().filter(|port| {
                !cache_outputs
                    .iter()
                    .any(|cache| cache.value_id() == port.value_id())
            }),
        )?);
        if let Some(prefix) = &p.prefix_decode {
            let decode = ports(&prefix.stage)?;
            budget.output = budget.output.max(sum_ports(host, decode.outputs().iter())?);
            budget.input = budget.input.max(add(&[
                sum_ports(host, denoise.inputs().iter())?,
                latent,
                input_bytes(host, decode, QwenImage21TransformerFamily::FREQUENCIES_ID)?,
                input_bytes(host, decode, QwenImage21TransformerFamily::TIME_FEATURES_ID)?,
            ])?);
        }
        let decoder_output = sum_ports(host, ports(&p.decode_stage)?.outputs().iter())?;
        budget.output = budget.output.max(decoder_output_peak(
            decoder_output,
            batch,
            self.execution.config.decoder_batch_policy,
        )?);
        let mut stages = vec![
            &p.text_stage,
            &p.noise_stage,
            &p.denoise_stage,
            &p.decode_stage,
        ];
        if let Some(edit) = &p.edit {
            let vision = ports(&edit.vision_stage)?;
            budget.input = budget.input.max(sum_ports(
                host,
                vision
                    .inputs()
                    .iter()
                    .filter(|port| port.value_id().as_str() != Qwen3VLVisionFamily::PATCHES_ID),
            )?);
            // All vision samples coexist until the batched text bridge copies
            // them. Encoder outputs are normalized and retired one at a time.
            budget.output = budget
                .output
                .max(multiply(sum_ports(host, vision.outputs().iter())?, batch)?)
                .max(sum_ports(
                    host,
                    ports(&edit.encoder_stage)?.outputs().iter(),
                )?);
            if batch > 1 {
                let per_reference = input_logical_bytes(
                    denoise,
                    QwenImage21TransformerFamily::REFERENCE_LATENTS_ID,
                )? / batch;
                budget.input =
                    budget
                        .input
                        .max(add(&[text, reference, host.backing_bytes(per_reference)?])?);
            }
            stages.extend([&edit.vision_stage, &edit.encoder_stage]);
        }
        // Prefix plans share one materializer and identical weights. They must
        // not count cache fill twice. Other stages can retain converted cache
        // entries across one another, so their incremental bounds add.
        let mut backing = Vec::new();
        for stage in stages {
            if self.stage_needs_weight_initialization(stage) {
                let (bytes, retention) = self.stage_host_backing(stage)?;
                budget.weights = add(&[budget.weights, bytes])?;
                backing.push(retention);
                if self.execution.runtime.requires_managed_static_staging() {
                    budget.staging = budget.staging.max(staging_bytes(
                        host,
                        stage,
                        self.execution.config.static_initialization,
                    )?);
                }
            }
        }
        Ok((budget, backing))
    }

    fn stage_needs_weight_initialization(&self, stage: &PreparedStage<R>) -> bool {
        if stage.plan.static_bytes == 0 {
            return false;
        }
        // Adaptive residency may evict any installation between stages.
        if self.execution.config.residency == ImageResidencyPolicy::Adaptive {
            return true;
        }
        !self.execution.config.residency.retains(stage.plan.stage)
            || !self.execution.resident_stages.has_weights(stage.plan.stage)
    }

    pub(super) fn stage_host_backing(
        &self,
        stage: &PreparedStage<R>,
    ) -> anyhow::Result<(usize, HostBackingPreparation)> {
        self.execution.stage_host_backing(stage)
    }
}

pub(super) fn ports<R: DeviceRuntime>(
    stage: &PreparedStage<R>,
) -> anyhow::Result<&CompiledTensorPorts> {
    stage
        .component
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .context("image stage has no compiled tensor ports")
}

pub(super) fn sum_ports<'a>(
    host: &HostMemoryAccount,
    mut ports: impl Iterator<Item = &'a CompiledTensorPort>,
) -> anyhow::Result<usize> {
    ports.try_fold(0, |total, port| {
        add(&[
            total,
            host.backing_bytes(usize::try_from(port.tensor().minimum_storage_bytes()?)?)?,
        ])
    })
}

/// During one request-owned Euler wave the gathered batch, every old sample
/// and every prepared successor coexist with the fixed conditioning inputs.
/// Each request sample is a distinct allocation and needs its own padding.
pub(super) fn full_generation_input_peak(
    host: &HostMemoryAccount,
    ports: &CompiledTensorPorts,
    batch: usize,
) -> anyhow::Result<usize> {
    let latent = input_logical_bytes(ports, QwenImage21TransformerFamily::LATENTS_ID)?;
    ensure!(
        batch > 0 && latent % batch == 0,
        "invalid latent Host budget partition"
    );
    let samples = multiply(host.backing_bytes(latent / batch)?, batch)?;
    add(&[sum_ports(host, ports.inputs().iter())?, samples, samples])
}

fn input_logical_bytes(ports: &CompiledTensorPorts, name: &str) -> anyhow::Result<usize> {
    let port = ports
        .inputs()
        .iter()
        .find(|port| port.value_id().as_str() == name)
        .with_context(|| format!("image stage is missing input {name}"))?;
    Ok(usize::try_from(port.tensor().minimum_storage_bytes()?)?)
}

fn input_bytes(
    host: &HostMemoryAccount,
    ports: &CompiledTensorPorts,
    name: &str,
) -> anyhow::Result<usize> {
    Ok(host.backing_bytes(input_logical_bytes(ports, name)?)?)
}

fn optional_input_bytes(
    host: &HostMemoryAccount,
    ports: &CompiledTensorPorts,
    name: &str,
) -> anyhow::Result<usize> {
    if ports
        .inputs()
        .iter()
        .any(|port| port.value_id().as_str() == name)
    {
        input_bytes(host, ports, name)
    } else {
        Ok(0)
    }
}

pub(super) fn staging_bytes<R: DeviceRuntime>(
    host: &HostMemoryAccount,
    stage: &PreparedStage<R>,
    policy: StaticInitializationPolicy,
) -> anyhow::Result<usize> {
    stored_staging_bytes(
        host,
        stage
            .component
            .executable()
            .execution_plan()
            .payload()
            .execution_weights(),
        policy,
    )
}

/// The same per-allocation staging bound for an already compiled causal phase.
pub(super) fn stored_staging_bytes(
    host: &HostMemoryAccount,
    weights: &ExecutionWeights,
    policy: StaticInitializationPolicy,
) -> anyhow::Result<usize> {
    let ExecutionWeights::Stored(weights) = weights else {
        return Ok(0);
    };
    let mut largest_chunk = 0;
    let mut all_chunks = 0usize;
    for component in &weights.schema().components {
        let element = component.physical_element_type().size_bytes();
        let maximum = policy.maximum_staging_bytes() - policy.maximum_staging_bytes() % element;
        ensure!(
            maximum != 0,
            "static staging policy cannot hold one component element"
        );
        let bytes = component.physical_bytes()?;
        let chunk = bytes.min(maximum);
        if chunk > 0 {
            largest_chunk = largest_chunk.max(host.backing_bytes(usize::try_from(chunk)?)?);
        }
        let full_chunks = usize::try_from(bytes / maximum)?;
        if full_chunks > 0 {
            all_chunks = add(&[
                all_chunks,
                multiply(host.backing_bytes(usize::try_from(maximum)?)?, full_chunks)?,
            ])?;
        }
        let remainder = bytes % maximum;
        if remainder > 0 {
            all_chunks = add(&[all_chunks, host.backing_bytes(usize::try_from(remainder)?)?])?;
        }
    }
    if largest_chunk == 0 {
        return Ok(0);
    }
    // Each chunk has independent leading/alignment padding. Static upload can
    // flush pending chunks before its next claim when this class ceiling is
    // smaller than the command batching policy's full physical bound.
    let padding = multiply(host.allocation_alignment() - 1, 2)?;
    let ceiling = add(&[
        usize::try_from(policy.maximum_staging_bytes())?,
        multiply(padding, policy.maximum_commands_per_batch())?,
    ])?;
    // The stager visits every placed execution component once, with one
    // independently padded allocation per chunk. An Arc retained by a command
    // shares that allocation. Tiny plans cannot accumulate a full policy-sized
    // batch: even retaining every chunk simultaneously is this smaller bound.
    Ok(largest_chunk.max(
        ceiling
            .min(host.limits().class_bytes(HostMemoryClass::TransferStaging))
            .min(all_chunks),
    ))
}

fn add(bytes: &[usize]) -> anyhow::Result<usize> {
    bytes.iter().try_fold(0_usize, |total, bytes| {
        total
            .checked_add(*bytes)
            .context("image Host peak overflows")
    })
}

fn multiply(bytes: usize, count: usize) -> anyhow::Result<usize> {
    bytes
        .checked_mul(count)
        .context("image Host peak overflows")
}

pub(super) fn decoder_output_peak(
    stage_output: usize,
    batch: usize,
    policy: ImageDecoderBatchPolicy,
) -> anyhow::Result<usize> {
    match policy {
        ImageDecoderBatchPolicy::SingleImage => multiply(stage_output, batch),
        ImageDecoderBatchPolicy::FullBatch => Ok(stage_output),
    }
}
