use super::foundation::invalid_operation;
use crate::{
    BatchOperationIdentity, BatchWorkShape, BoundSubmissionWork, DeviceRuntime,
    ExecutionFactsCapture, ExecutionNodeWork, PreparedStepSubmissionWave, ResidentDecodeUse,
    SubmittedOperationReceipt, VNextError,
};

/// Prepared before submission, then moved into evidence after the original
/// fence is armed. It contains no resources and never changes an identity.
pub(super) struct PreparedSubmissionWork {
    nodes: Box<[ExecutionNodeWork]>,
}

impl PreparedSubmissionWork {
    pub(super) fn for_wave<R: DeviceRuntime>(
        capture: ExecutionFactsCapture,
        identity: &BatchOperationIdentity,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<Option<Self>, VNextError> {
        if capture == ExecutionFactsCapture::Off {
            return Ok(None);
        }
        if identity.node_count() != wave.nodes().len() {
            return Err(invalid_operation(
                "observed wave node count differs from its identity",
            ));
        }
        let nodes = wave
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| {
                if identity.node_id_at(index) != Some(node.node_id()) {
                    return Err(invalid_operation(
                        "observed wave node order differs from its identity",
                    ));
                }
                Self::node(identity, index, node.work_shape())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            nodes: nodes.into_boxed_slice(),
        }))
    }

    pub(super) fn for_resident<R: DeviceRuntime>(
        capture: ExecutionFactsCapture,
        identity: &BatchOperationIdentity,
        use_: &ResidentDecodeUse<R>,
    ) -> Result<Option<Self>, VNextError> {
        if capture == ExecutionFactsCapture::Off {
            return Ok(None);
        }
        if identity.node_count() != use_.node_count() {
            return Err(invalid_operation(
                "observed resident node count differs from its identity",
            ));
        }
        let nodes = (0..use_.node_count())
            .map(|index| Self::node(identity, index, use_.work_shape()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            nodes: nodes.into_boxed_slice(),
        }))
    }

    fn node(
        identity: &BatchOperationIdentity,
        index: usize,
        work: &BatchWorkShape,
    ) -> Result<ExecutionNodeWork, VNextError> {
        if identity.work_shape_fingerprint_at(index) != Some(work.fingerprint())
            || identity.node_participant_count(index) != Some(work.participant_token_ranges().len())
        {
            return Err(invalid_operation(
                "observed token work differs from the bound node",
            ));
        }
        let index = u32::try_from(index)
            .map_err(|_| invalid_operation("observed node index exceeds u32"))?;
        Ok(ExecutionNodeWork::new(
            index,
            work.participant_token_ranges().into(),
        ))
    }

    pub(super) fn bind(self, submission: &SubmittedOperationReceipt) -> BoundSubmissionWork {
        BoundSubmissionWork::new(submission.clone(), self.nodes)
    }
}
