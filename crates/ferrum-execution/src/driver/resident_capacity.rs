use super::*;
use crate::model_executor::{ResidentCohortCapacitySnapshot, ResidentSequenceCapacityRequest};

impl<R: DeviceRuntime> ProgramExecutor<R> {
    pub(super) fn native_resident_cohort_capacity(
        &self,
        cohort: &[ResidentSequenceCapacityRequest],
    ) -> Option<ResidentCohortCapacitySnapshot> {
        if cohort.is_empty()
            || cohort.len() > MAX_RESIDENT_CAPACITY_PARTICIPANTS
            || cohort.len() > self.policy.memory().maximum_active_sequences as usize
            || cohort.len() as u64 > self.policy.admission().maximum_scheduled_tokens
        {
            return None;
        }
        let mut sequences = Vec::with_capacity(cohort.len());
        {
            let registry = self.sequences.try_lock()?;
            for (index, request) in cohort.iter().enumerate() {
                self.token_issuer.frontier(&request.sequence, 0).ok()?;
                if request.through_tokens == 0
                    || cohort[..index]
                        .iter()
                        .any(|other| other.sequence == request.sequence)
                {
                    return None;
                }
                let sequence = registry.token_index.get(&request.sequence)?.upgrade()?;
                if request.through_tokens > sequence.maximum_tokens
                    || u64::try_from(request.through_tokens).ok()?
                        > sequence
                            .session
                            .resources()
                            .request_resources()
                            .work_shape()
                            .fit_tokens()
                {
                    return None;
                }
                sequences.push(sequence);
            }
        }
        // No waiting on execution: the synchronous planning query is unknown
        // while any original operation gate is in use.
        let _operations = sequences
            .iter()
            .map(|sequence| sequence.operation.try_lock().ok())
            .collect::<Option<Vec<_>>>()?;
        let mut settlements = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            let settlement = sequence.settlement.try_lock()?;
            if !sequence.active.load(Ordering::Acquire) || !settlement.token_frontier_is_open() {
                return None;
            }
            settlements.push(settlement);
        }
        let sessions = sequences
            .iter()
            .map(|sequence| Arc::clone(&sequence.session))
            .collect::<Vec<_>>();
        SequenceSession::resident_cohort_persistent_capacity(&sessions)
            .ok()?
            .then_some(ResidentCohortCapacitySnapshot {
                participants: cohort.len(),
                // Only this observed resident width is certified. This is not
                // an assertion that configured spare slots are backed.
                max_batch_sequences: cohort.len(),
            })
    }
}
