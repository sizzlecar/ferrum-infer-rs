use super::*;

/// Actual retirement evidence, returned only after every session and the last
/// shared installation owner have closed. Host continuations remain untouched.
pub struct DenoisingSuspendReceipt {
    pub plans: Vec<PlanRuntimeCloseReceipt>,
    pub weights: SharedComponentWeightsCloseReceipt,
}
impl<R: DeviceRuntime> DenoisingStage<R> {
    /// Add an optional execution cardinality without replacing the existing
    /// stage or reuploading its weights. The caller reaches this method only
    /// at a fenced membership boundary. A suspended stage records metadata;
    /// a resident stage uses the original Core shared-installation attach.
    pub fn attach_batch(
        &mut self,
        batch: PreparedDenoisingBatch<R>,
        initialization: ComponentInitializationConfig,
    ) -> Result<(), DenoisingStageError<R>> {
        let count = batch.plan().configuration().backbone().batches() as usize;
        if count != 2
            || self.batches.contains_key(&count)
            || (!self.is_resident() && !self.is_suspended())
            || batch.compiled().family().weights() != self.batches[&1].compiled().family().weights()
        {
            return Err(invalid(
                "optional batch requires the same complete source schema and a usable stage",
            )
            .into());
        }
        if self.is_suspended() {
            self.batches.insert(count, batch);
            return Ok(());
        }
        self.failed = true;
        let session = batch
            .compiled()
            .initialize_with_shared_weights(
                self.runtime.clone(),
                self.host.clone(),
                self.weights.as_ref().unwrap(),
                initialization,
            )
            .map_err(DenoisingStageError::Initialize)?;
        self.sessions.insert(count, session);
        self.batches.insert(count, batch);
        self.failed = false;
        Ok(())
    }

    /// Reclaim idle temporaries at a completed phase boundary without closing
    /// the one shared weight owner. Busy or uncertain retirement stays fatal.
    pub fn release_idle_backing(&mut self) -> Result<(), DenoisingStageError<R>> {
        if !self.is_resident() {
            return Err(invalid("idle reclamation requires a complete resident stage").into());
        }
        self.failed = true;
        for session in self.sessions.values_mut() {
            match session.try_release_idle_dynamic_backing()? {
                IdleDynamicReleaseOutcome::Released(_) => {}
                IdleDynamicReleaseOutcome::Busy(owner) => {
                    return Err(DenoisingStageError::IdleBusy(owner));
                }
            }
            // The last step's Output account may belong to a completed batch
            // scope. Resident weights must not pin that transient Host credit.
            session
                .set_output_host_account(self.host.clone())
                .map_err(DenoisingStageError::Execute)?;
        }
        self.active_batch = None;
        self.failed = false;
        Ok(())
    }

    /// Reinstall after an auxiliary stage, or cold-start once. Initialization
    /// identities/budgets are explicit for each configured batch cardinality.
    pub fn restore(
        &mut self,
        initialization: &BTreeMap<usize, ComponentInitializationConfig>,
    ) -> Result<(), DenoisingStageError<R>> {
        if !self.is_suspended() || !initialization.keys().eq(self.batches.keys()) {
            return Err(invalid(
                "restore requires suspended state and exact plan initialization identities",
            )
            .into());
        }
        // Set failed before acquisition: partial initialization cannot be
        // mistaken for resident or suspended. Original errors retain custody.
        self.failed = true;
        let first = &self.batches[&1];
        let (session, weights) = first
            .compiled
            .initialize_shared_weights_with_pool_policy(
                self.runtime.clone(),
                self.host.clone(),
                ComponentWeightSource::Stored(self.sources[0].as_ref()),
                initialization[&1].clone(),
                ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
            )
            .map_err(DenoisingStageError::Initialize)?;
        self.weights = Some(weights);
        self.sessions.insert(1, session);
        for (&count, batch) in &self.batches {
            if count == 1 {
                continue;
            }
            let session = batch
                .compiled
                .initialize_with_shared_weights(
                    self.runtime.clone(),
                    self.host.clone(),
                    self.weights.as_ref().unwrap(),
                    initialization[&count].clone(),
                )
                .map_err(DenoisingStageError::Initialize)?;
            self.sessions.insert(count, session);
        }
        self.failed = false;
        self.active_batch = None;
        Ok(())
    }
    /// Fenced stage transition. This does not discard request continuations or
    /// claim that dropping a session releases native capacity.
    pub fn suspend(&mut self) -> Result<DenoisingSuspendReceipt, DenoisingStageError<R>> {
        if !self.is_resident() {
            return Err(invalid("suspend requires a complete resident stage").into());
        }
        self.failed = true;
        let mut receipts = Vec::with_capacity(self.sessions.len());
        while let Some((_, session)) = self.sessions.pop_first() {
            match session.close() {
                Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipts.push(receipt),
                original => return Err(DenoisingStageError::PlanClose(original)),
            }
        }
        let weights = match self.weights.take().unwrap().close() {
            Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => receipt,
            original => return Err(DenoisingStageError::WeightsClose(original)),
        };
        self.failed = false;
        self.active_batch = None;
        Ok(DenoisingSuspendReceipt {
            plans: receipts,
            weights,
        })
    }

    pub(super) fn select_active_batch(
        &mut self,
        count: usize,
    ) -> Result<(), DenoisingStageError<R>> {
        if self.active_batch == Some(count) {
            return Ok(());
        }
        self.failed = true;
        match self.workspace {
            DenoisingWorkspacePolicy::ActivePhaseOnly => {
                for (&other, session) in &mut self.sessions {
                    if other == count {
                        continue;
                    }
                    match session.try_release_idle_dynamic_backing()? {
                        IdleDynamicReleaseOutcome::Released(receipt) => {
                            tracing::debug!(
                                batch = other,
                                ?receipt,
                                "released inactive FastH3 batch backing"
                            );
                        }
                        IdleDynamicReleaseOutcome::Busy(owner) => {
                            return Err(DenoisingStageError::IdleBusy(owner))
                        }
                    }
                }
            }
        }
        self.active_batch = Some(count);
        self.failed = false;
        Ok(())
    }
}
