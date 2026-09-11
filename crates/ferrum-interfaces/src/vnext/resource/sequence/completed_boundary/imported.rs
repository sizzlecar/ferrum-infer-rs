use super::*;
use crate::vnext::SuccessfulRestoreFrontierSeal;

impl<R: DeviceRuntime> PreparedSequenceStateTransfer<R> {
    pub(crate) fn install_imported_frontier(
        &self,
        seal: &SuccessfulRestoreFrontierSeal,
        full_input: Arc<[u32]>,
    ) -> Result<Arc<CompletedSequenceBoundary>, VNextError> {
        let resources = self.session().resources();
        let plan = &resources.request.plan;
        let _lifecycle = plan.resources.read_lifecycle("install restored frontier")?;
        if self.kind() != SequenceStateTransferKind::RestoreWrite || !seal.matches_target(self) {
            return Err(invalid_resource(
                "restore frontier does not own this exact native target",
            ));
        }
        resources
            .admitted_work
            .validate_single_checkpoint_input(Arc::clone(&full_input))?;
        let source = seal.source();
        let end = source.completed_tokens();
        if source.plan_hash() != plan.plan_hash()
            || end == 0
            || full_input.get(..end) != Some(source.token_prefix())
        {
            return Err(invalid_resource(
                "restored state differs from the admitted exact token prefix",
            ));
        }
        self.with_reserved_state(|active| {
            if !matches!(active.completed_boundary, SequenceCompletedFrontier::Fresh) {
                return Err(invalid_resource("restore target is no longer fresh"));
            }
            let boundary = Arc::new(CompletedSequenceBoundary {
                plan_hash: plan.plan_hash().clone(),
                provenance: seal.provenance(),
                continuation: Some(Arc::new(SequenceCheckpointContinuation {
                    input_dependency: seal.input_dependency(),
                    suffix_constraints: seal.suffix_constraints().to_vec(),
                })),
                epoch: active.epoch,
                session_fingerprint: active.fingerprint.clone(),
                backing_generation: self.backing().generation(),
                start_token: source.capture_span_start(),
                end_token: end,
                full_input,
            });
            active.completed_boundary = SequenceCompletedFrontier::Proven(Arc::clone(&boundary));
            Ok(boundary)
        })
    }

    pub(crate) fn acknowledge_imported_frontier(
        &self,
        expected: &Arc<CompletedSequenceBoundary>,
    ) -> Result<(), VNextError> {
        let _lifecycle = self
            .session()
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("publish restored frontier")?;
        self.with_reserved_state(|active| {
            if self.kind() != SequenceStateTransferKind::RestoreWrite
                || !matches!(expected.provenance(), CompletedSequenceProvenance::ImportedCheckpoint { .. })
                || !matches!(&active.completed_boundary, SequenceCompletedFrontier::Proven(current) if Arc::ptr_eq(current, expected))
                || expected.epoch() != active.epoch
                || expected.session_fingerprint() != &active.fingerprint
                || expected.backing_generation() != self.backing().generation()
            {
                return Err(invalid_resource("restore publication lost its exact installed frontier"));
            }
            self.release_active_reservation(active)
        })
    }
}
