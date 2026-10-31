//! Capacity-driven retirement of this pipeline's idle owners. This is planning
//! evidence, not a reservation or a query of other users of the device ledger.
use super::*;

impl<R: DeviceRuntime> ResidentStages<R> {
    #[cfg(test)]
    pub(in crate::image_generation) fn set_weight_reload_timing(
        &mut self,
        stage: ImageStage,
        duration_us: u64,
    ) {
        self.weights_slot(stage)
            .unwrap()
            .as_mut()
            .unwrap()
            .origin
            .initialization_wall_duration_us = duration_us;
    }

    fn stage(&self, stage: ImageStage) -> Option<&ActiveImageStage<R>> {
        match stage {
            ImageStage::TextEncoder => self.text_encoder.as_ref(),
            ImageStage::VisionEncoder => self.vision_encoder.as_ref(),
            ImageStage::VaeEncoder => self.vae_encoder.as_ref(),
            ImageStage::Denoiser => self.denoiser.as_ref(),
            ImageStage::Decoder => self.decoder.as_ref(),
            ImageStage::Noise => self.noise.as_ref(),
        }
    }

    fn weights(&self, stage: ImageStage) -> Option<&ResidentWeights<R>> {
        match stage {
            ImageStage::TextEncoder => self.text_encoder_weights.as_ref(),
            ImageStage::VisionEncoder => self.vision_encoder_weights.as_ref(),
            ImageStage::VaeEncoder => self.vae_encoder_weights.as_ref(),
            ImageStage::Denoiser => self.denoiser_weights.as_ref(),
            ImageStage::Decoder => self.decoder_weights.as_ref(),
            ImageStage::Noise => None,
        }
    }

    pub(super) fn touch(&mut self, stage: ImageStage) {
        // Rebase the six ranks before overflow without changing their order.
        if self.acquisition_clock == u64::MAX {
            let mut indices = [0, 1, 2, 3, 4, 5];
            indices.sort_by_key(|&index| (self.acquisition_order[index], index));
            for (rank, index) in indices.into_iter().enumerate() {
                self.acquisition_order[index] = rank as u64;
            }
            self.acquisition_clock = 6;
        }
        self.acquisition_clock += 1;
        self.acquisition_order[resident_stage_index(stage).expect("known stage")] =
            self.acquisition_clock;
    }

    fn eviction_order(&self, target: ImageStage) -> [ImageStage; 6] {
        let mut order = RETAINABLE_STAGES;
        order.sort_by_key(|&stage| {
            let index = resident_stage_index(stage).expect("known stage");
            // Preserve the next stage's reusable pool until all other idle
            // pools have been considered. Ties have a stable stage order.
            (stage == target, self.acquisition_order[index], index)
        });
        order
    }

    /// The fixed stage set makes an exact search cheap. Reclaiming several
    /// small, old installations before a larger one can reload all of them on
    /// the next batch even when the larger installation alone was sufficient.
    /// Use actual cold-installation timing; attached-session timing is not a
    /// weight reload observation. Missing observations retain the LRU fallback.
    fn minimum_reload_weight_evictions(
        &self,
        target: ImageStage,
        required_bytes: u64,
    ) -> Option<Vec<ImageStage>> {
        if required_bytes == 0 {
            return Some(Vec::new());
        }
        let mut candidates = Vec::new();
        for stage in self.eviction_order(target) {
            // Checked-out sessions own active execution or prefix state. The
            // target's installation is needed again immediately, so releasing
            // it would not satisfy the predicted steady-state capacity deficit.
            if stage == target || self.stage(stage).is_none() {
                continue;
            }
            let Some(weights) = self.weights(stage) else {
                continue;
            };
            let receipt = &weights.origin;
            if receipt.weights_reused || receipt.initialization_wall_duration_us == 0 {
                return None;
            }
            candidates.push((
                stage,
                receipt.plan.static_bytes,
                receipt.initialization_wall_duration_us,
            ));
        }
        let mut best = None;
        for mask in 1_usize..(1 << candidates.len()) {
            let mut bytes = 0_u128;
            let mut reload_us = 0_u128;
            for (index, &(_, size, cost)) in candidates.iter().enumerate() {
                if mask & (1 << index) != 0 {
                    bytes += u128::from(size);
                    reload_us += u128::from(cost);
                }
            }
            if bytes < u128::from(required_bytes) {
                continue;
            }
            // The mask follows LRU order and breaks otherwise identical costs
            // deterministically. Never evict an extra zero-benefit owner.
            let score = (
                reload_us,
                mask.count_ones(),
                std::cmp::Reverse(mask.reverse_bits()),
                mask,
            );
            if best.is_none_or(|previous| score < previous) {
                best = Some(score);
            }
        }
        let (_, _, _, mask) = best?;
        Some(
            candidates
                .into_iter()
                .enumerate()
                .filter_map(|(index, (stage, _, _))| (mask & (1 << index) != 0).then_some(stage))
                .collect(),
        )
    }
}

fn idle_receipt(receipt: &mut ImageStageExecutionReceipt) {
    receipt.completed_executions = 0;
    receipt.execution_wall_durations_us.clear();
    receipt.device_profiles.clear();
    receipt.initialization_wall_duration_us = 0;
    receipt.plan.compile_wall_duration_us = 0;
    receipt.session_reused = true;
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// Fresh preparation can count only this pipeline's idle, reclaimable
    /// allocations in addition to currently available physical storage.
    pub(in crate::image_generation) fn adaptive_capacity_ceiling(&self) -> anyhow::Result<u64> {
        if self.preparation_only {
            return Ok(self.config.device_capacity_bytes);
        }

        let Some(memory) = self.runtime.memory_availability()? else {
            return Ok(self.config.device_capacity_bytes);
        };
        let mut reclaimable = 0_u64;
        for stage in RETAINABLE_STAGES {
            if self.resident_stages.stage(stage).is_some() {
                reclaimable = reclaimable
                    .checked_add(self.resident_stages.dynamic_bytes(stage))
                    .and_then(|n| {
                        n.checked_add(
                            self.resident_stages
                                .weights(stage)
                                .map_or(0, |weights| weights.origin.plan.static_bytes),
                        )
                    })
                    .context("idle image capacity overflows")?;
            }
        }
        // The configured ceiling already applies utilization and explicit
        // account limits. Do not subtract total - ceiling from physical free
        // storage again: native/other owners have already reduced that value.
        // The report includes the native allowance, so leaving it in this
        // bound also reserves room for native storage not allocated yet.
        let capacity = memory
            .available_bytes()
            .saturating_add(reclaimable)
            .min(self.config.device_capacity_bytes);
        tracing::debug!(target: "ferrum_engine::image_generation::timing",
            capacity_bytes = capacity, configured_capacity_bytes = self.config.device_capacity_bytes,
            reclaimable_bytes = reclaimable, free_bytes = memory.free_bytes,
            "image preparation physical capacity observed");
        Ok(capacity)
    }

    /// A logical capacity estimate cannot guarantee a native allocation. On a
    /// rolled-back allocation failure, retire actual idle owners before retry.
    /// Each successful pass removes backing, so retries terminate without a
    /// configured retry count or reducing the user's batch limit.
    pub(in crate::image_generation) fn reclaim_after_allocation_failure(
        &mut self,
        target: ImageStage,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<bool, ImageGenerationFailure<R>> {
        let order = self.resident_stages.eviction_order(target);
        let mut reclaimed = false;
        for stage in order {
            if self.resident_stages.stage(stage).is_some()
                && self.resident_stages.dynamic_bytes(stage) != 0
            {
                self.adaptive_release_dynamic(stage, receipts)?;
                reclaimed = true;
            }
        }
        if reclaimed {
            return Ok(true);
        }
        for stage in order {
            // A missing session can mean that an active caller checked it out.
            // Only idle sessions authorize retirement of their shared weights.
            if self.resident_stages.stage(stage).is_some()
                && self.resident_stages.weights(stage).is_some()
            {
                self.adaptive_close_session(stage, receipts)?;
                self.adaptive_close_weights(stage, receipts)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn adaptive_required_bytes(&self, prepared: &PreparedStage<R>) -> anyhow::Result<u64> {
        let target = prepared.plan.stage;
        let mut bytes = self.device_opaque_allowance_bytes;
        for stage in RETAINABLE_STAGES {
            if let Some(weights) = self.resident_stages.weights(stage) {
                // A decode subset still charges the original full installation.
                bytes = bytes
                    .checked_add(weights.origin.plan.static_bytes)
                    .context("adaptive static capacity overflows")?;
            } else if stage == target {
                bytes = bytes
                    .checked_add(prepared.plan.static_bytes)
                    .context("adaptive target static capacity overflows")?;
            }
            if stage != target {
                bytes = bytes
                    .checked_add(self.resident_stages.dynamic_bytes(stage))
                    .context("adaptive idle dynamic capacity overflows")?;
            }
        }
        // For an identical plan this includes the actual retained chunks and
        // the missing per-pool demand. After trimming it must be recomputed.
        let dynamic = prepared.admission_dynamic_bytes(self.runnable_dynamic_bytes(prepared)?)?;
        bytes
            .checked_add(dynamic)
            .context("adaptive active capacity overflows")
    }

    fn adaptive_physical_demand(
        &self,
        prepared: &PreparedStage<R>,
    ) -> anyhow::Result<Option<(u64, u64)>> {
        let Some(memory) = self.runtime.memory_availability()? else {
            return Ok(None);
        };
        let target = prepared.plan.stage;
        let additional_static = if self.resident_stages.weights(target).is_some() {
            0
        } else {
            prepared.plan.static_bytes
        };
        let dynamic = prepared.admission_dynamic_bytes(self.runnable_dynamic_bytes(prepared)?)?;
        let additional = additional_static
            .checked_add(dynamic.saturating_sub(self.resident_stages.dynamic_bytes(target)))
            .context("adaptive physical demand overflows")?;
        // The logical check already enforces the configured account ceiling.
        // Physical free excludes allocated driver/native/other owners; reserve
        // only the declared native allowance for possible future native growth.
        let reserve = self.device_opaque_allowance_bytes;
        let available = memory.available_bytes().saturating_sub(reserve);
        tracing::debug!(target: "ferrum_engine::image_generation::timing",
            stage = ?target, additional_bytes = additional, available_bytes = available,
            free_bytes = memory.free_bytes, reusable_pool_bytes = memory.reusable_pool_bytes,
            reserve_bytes = reserve, "image stage physical capacity observed");
        Ok(Some((additional, available)))
    }

    fn adaptive_stage_fits(&self, prepared: &PreparedStage<R>) -> anyhow::Result<bool> {
        if self.adaptive_required_bytes(prepared)? > self.config.device_capacity_bytes {
            return Ok(false);
        }
        Ok(self
            .adaptive_physical_demand(prepared)?
            .is_none_or(|(required, available)| required <= available))
    }

    pub(super) fn admit_adaptive_stage(
        &mut self,
        prepared: &PreparedStage<R>,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        let preflight = ImageGenerationFailure::Preflight;
        let memory = prepared
            .component
            .executable()
            .execution_plan()
            .payload()
            .memory();
        let cold = prepared
            .admission_dynamic_bytes(memory.minimum_runnable_request_bytes())
            .and_then(|dynamic| {
                memory
                    .static_bytes()
                    .checked_add(dynamic)
                    .and_then(|n| n.checked_add(self.device_opaque_allowance_bytes))
                    .context("adaptive cold capacity overflows")
            })
            .map_err(preflight)?;
        // An impossible stage must not destroy useful previous residency. Full
        // requests also check this for every plan before their first upload.
        if cold > self.config.device_capacity_bytes {
            return Err(preflight(anyhow::anyhow!(
                "adaptive image stage {:?} needs at least {cold} device bytes, exceeding total ceiling {}",
                prepared.plan.stage, self.config.device_capacity_bytes
            )));
        }
        let target = prepared.plan.stage;
        if self
            .resident_stages
            .stage(target)
            .is_some_and(|old| !same_resident_plan(&old.receipt.plan, &prepared.plan))
        {
            // Plan replacement cannot retain an old shape's dynamic pools.
            // Its readonly weights remain eligible for exact attachment.
            self.adaptive_close_session(target, receipts)?;
        }
        let order = self.resident_stages.eviction_order(target);
        for stage in order {
            if self.adaptive_stage_fits(prepared).map_err(preflight)? {
                return Ok(());
            }
            if self.resident_stages.dynamic_bytes(stage) != 0 {
                self.adaptive_release_dynamic(stage, receipts)?;
            }
        }

        let logical_deficit = self
            .adaptive_required_bytes(prepared)
            .map_err(preflight)?
            .saturating_sub(self.config.device_capacity_bytes);
        let physical_deficit = self
            .adaptive_physical_demand(prepared)
            .map_err(preflight)?
            .map_or(0, |(required, available)| {
                required.saturating_sub(available)
            });
        let required_bytes = logical_deficit.max(physical_deficit);
        if let Some(evictions) = self
            .resident_stages
            .minimum_reload_weight_evictions(target, required_bytes)
        {
            tracing::debug!(target: "ferrum_engine::image_generation::timing",
                stage = ?target, required_bytes, stages = ?evictions,
                "image idle weight reclamation selected by observed reload cost");
            for stage in evictions {
                // This selection is only an estimate. Real free storage and
                // actual managed ownership remain authoritative after every
                // release, including when another owner changes capacity.
                if self.adaptive_stage_fits(prepared).map_err(preflight)? {
                    return Ok(());
                }
                self.adaptive_close_session(stage, receipts)?;
                self.adaptive_close_weights(stage, receipts)?;
            }
        }
        for stage in order {
            if self.adaptive_stage_fits(prepared).map_err(preflight)? {
                return Ok(());
            }
            // Only a slot owns an idle session. Never evict a checked-out
            // session, a prefix request, or a pending completion's lease.
            if self.resident_stages.stage(stage).is_some() {
                self.adaptive_close_session(stage, receipts)?;
                self.adaptive_close_weights(stage, receipts)?;
            } else if stage == target && self.resident_stages.weights(stage).is_some() {
                // A changed plan may have just closed its old session. Closing
                // the full snapshot is the final option before cold reload;
                // Referenced/failed close returns its actual recovery owner.
                self.adaptive_close_weights(stage, receipts)?;
            }
        }
        let required = self.adaptive_required_bytes(prepared).map_err(preflight)?;
        if required > self.config.device_capacity_bytes {
            return Err(preflight(anyhow::anyhow!(
                "adaptive image stage still needs {required} bytes after idle reclamation; active owners cannot be evicted"
            )));
        }
        if let Some((additional_bytes, available_bytes)) =
            self.adaptive_physical_demand(prepared).map_err(preflight)?
        {
            if additional_bytes > available_bytes {
                return Err(ImageGenerationFailure::CapacityRejected {
                    stage: target,
                    additional_bytes,
                    available_bytes,
                });
            }
        }
        Ok(())
    }

    fn adaptive_release_dynamic(
        &mut self,
        stage: ImageStage,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        let Some(mut active) = self.resident_stages.slot(stage).and_then(Option::take) else {
            return Ok(());
        };
        idle_receipt(&mut active.receipt);
        let started = std::time::Instant::now();
        let release = match active.session.try_release_idle_dynamic_backing() {
            Ok(IdleDynamicReleaseOutcome::Released(release)) => release,
            Ok(IdleDynamicReleaseOutcome::Busy(reason)) => {
                return Err(active.processing(anyhow::anyhow!(
                    "adaptive stage is not idle for reclamation: {reason:?}"
                )))
            }
            Err(error) => return Err(active.processing(error.into())),
        };
        self.resident_stages.set_dynamic_bytes(stage, 0);
        receipts.push(ImageStageReceipt {
            execution: active.receipt.clone(),
            close: None,
            retirement: ImageStageRetirement::Resident {
                idle_release: release,
            },
            close_wall_duration_us: trace_elapsed_us(started),
        });
        *self.resident_stages.slot(stage).expect("known stage") = Some(active);
        Ok(())
    }

    fn adaptive_close_session(
        &mut self,
        stage: ImageStage,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        if let Some(mut active) = self.resident_stages.slot(stage).and_then(Option::take) {
            idle_receipt(&mut active.receipt);
            receipts.push(active.close()?);
            self.resident_stages.set_dynamic_bytes(stage, 0);
        }
        Ok(())
    }

    fn adaptive_close_weights(
        &mut self,
        stage: ImageStage,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        if let Some(ResidentWeights {
            weights,
            mut origin,
        }) = self
            .resident_stages
            .weights_slot(stage)
            .and_then(Option::take)
        {
            idle_receipt(&mut origin);
            origin.session_reused = false;
            origin.weights_reused = true;
            let started = std::time::Instant::now();
            match weights.close() {
                Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => {
                    receipts.push(ImageStageReceipt {
                        execution: origin,
                        close: None,
                        retirement: ImageStageRetirement::SharedWeightsClosed { receipt },
                        close_wall_duration_us: trace_elapsed_us(started),
                    });
                }
                Ok(outcome @ SharedComponentWeightsCloseOutcome::Referenced { .. }) => {
                    return Err(ImageGenerationFailure::SharedWeightsReferenced { stage, outcome });
                }
                Err(source) => {
                    return Err(ImageGenerationFailure::SharedWeightsClose { stage, source })
                }
            }
        }
        Ok(())
    }
}
