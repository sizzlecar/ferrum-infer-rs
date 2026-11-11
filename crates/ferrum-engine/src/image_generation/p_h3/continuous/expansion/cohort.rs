//! A sealed auxiliary cohort retires each visual producer before installing
//! its one causal owner. Original members survive filtering and downstream handoff.
use super::*;
use crate::image_generation::p_h3::expansion::cohort::{
    ExpansionCohortProgress, ExpansionParticipantResult, PreparedExpansionCohort,
};
use crate::image_generation::p_h3::expansion::product::ProductExpansionCohort;
use crate::image_generation::service::{
    ImageDenoisingBatch, ImageInitializationCohortCompletion, ImageInitializationCohortGenerator,
    ImageInitializationCohortMember, ImageInitializationCohortOutcome,
    ImageInitializationCohortProgress, ImageInitializationCohortRequest,
};
use std::time::Instant;
use uuid::Uuid;
mod admission;
mod requirements;
mod state;
pub(super) mod synchronous;
#[cfg(test)]
mod tests;
use requirements::{CohortRequirements, VisualRequirements};
use state::{Cohort, CohortFailure, CohortPhase};

struct Member {
    request_id: Uuid,
    correlation: Option<ferrum_types::ImageRequestCorrelation>,
    identity: IngressIdentity,
    job: u64,
    output_credit: usize,
    ingress_backing_bytes: usize,
    prompt_trace: Option<u64>,
}
struct PreparedCohort<R: DeviceRuntime> {
    pipeline: Arc<()>,
    assets: Arc<NativeTextExpansion>,
    prepared: Option<PreparedExpansionCohort>,
    visuals: Vec<Option<PreparedVisual<R>>>,
    requirements: CohortRequirements,
    phase: Option<HostMemoryScope>,
    members: Vec<Member>,
    receipts: Vec<ImageStageReceipt>,
}
fn cohort_preparation<R: DeviceRuntime>(
    prepared: &mut ImageBatchPreparation,
) -> InitResult<Option<&mut PreparedExpansion<R>>> {
    Ok(match prepared.get_mut::<PreparedInitialization<R>>()? {
        PreparedInitialization::Expansion(expansion) if expansion.phase.is_none() => {
            Some(expansion)
        }
        _ => None,
    })
}

impl<R: DeviceRuntime> ImageInitializationCohortGenerator for PH3ImagePipeline<R> {
    fn maximum_initialization_cohort_members(
        &mut self,
        prepared: &mut ImageBatchPreparation,
    ) -> InitResult<usize> {
        Ok(if cohort_preparation::<R>(prepared)?.is_some() {
            self.execution.config.maximum_batch_size
        } else {
            0
        })
    }
    fn initialization_cohort_compatible(
        &mut self,
        first: &mut ImageBatchPreparation,
        next: &mut ImageBatchPreparation,
    ) -> InitResult<bool> {
        let (Some(first), Some(next)) = (
            cohort_preparation::<R>(first)?,
            cohort_preparation::<R>(next)?,
        ) else {
            return Ok(false);
        };
        if !Arc::ptr_eq(
            &first.pipeline,
            &self.execution.service_preparation_identity,
        ) || !Arc::ptr_eq(&first.pipeline, &next.pipeline)
            || !Arc::ptr_eq(&first.assets, &next.assets)
        {
            return Ok(false);
        }
        first
            .assets
            .cohort_compatible(&first.prepared, &next.prepared)
            .map_err(|e| preparation_error(e.into()))
    }
    fn prepare_initialization_cohort(
        &mut self,
        members: Vec<ImageInitializationCohortMember>,
    ) -> InitResult<ImageBatchPreparation> {
        if members.is_empty() || members.len() > self.execution.config.maximum_batch_size {
            return Err(ImageBatchError::Rejected(
                "H3 expansion cohort exceeds declared capacity".into(),
            ));
        }
        let mut metadata = Vec::with_capacity(members.len());
        let mut preparations = Vec::with_capacity(members.len());
        let mut visuals = Vec::with_capacity(members.len());
        let mut assets = None::<Arc<NativeTextExpansion>>;
        for member in members {
            let PreparedInitialization::Expansion(expansion) = member
                .preparation
                .into_inner::<PreparedInitialization<R>>(
            )?
            else {
                return Err(ImageBatchError::Rejected(
                    "H3 cohort requires prompt expansion".into(),
                ));
            };
            if expansion.phase.is_some()
                || !Arc::ptr_eq(
                    &expansion.pipeline,
                    &self.execution.service_preparation_identity,
                )
                || assets
                    .as_ref()
                    .is_some_and(|a| !Arc::ptr_eq(a, &expansion.assets))
                || metadata
                    .iter()
                    .any(|m: &Member| m.request_id == member.request_id)
            {
                return Err(ImageBatchError::Rejected(
                    "H3 expansion cohort source or admission identity differs".into(),
                ));
            }
            assets.get_or_insert_with(|| expansion.assets.clone());
            preparations.push(expansion.prepared);
            visuals.push(expansion.visual);
            metadata.push(Member {
                request_id: member.request_id,
                correlation: expansion.correlation,
                identity: expansion.identity,
                job: expansion.job,
                output_credit: member.resources.output_credit_bytes,
                ingress_backing_bytes: expansion.ingress_backing_bytes,
                prompt_trace: None,
            });
        }
        let assets = assets.expect("nonempty validated cohort");
        let mut prepared = assets
            .prepare_cohort(preparations)
            .map_err(|e| preparation_error(e.into()))?;
        let composition = self.expansion_composition();
        let host = &self.execution.host;
        let causal = assets
            .cohort_host_requirements(&composition, &prepared, host)
            .map_err(|e| preparation_error(e.into()))?;
        let visual_requirements = prepared
            .requests_mut()
            .iter()
            .zip(&visuals)
            .map(|(request, visual)| -> Result<_> {
                visual
                    .as_ref()
                    .map(|visual| {
                        Ok(VisualRequirements {
                            report: visual.requirements(host)?,
                            decoded_rgba: visual.decoded_backing_bytes(host)?,
                            persistent_input: request.retained_visual_input_bytes(host)?,
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()
            .map_err(preparation_error)?;
        let requirements =
            CohortRequirements::new(causal, visual_requirements).map_err(preparation_error)?;
        Ok(ImageBatchPreparation::new(PreparedCohort {
            pipeline: self.execution.service_preparation_identity.clone(),
            assets,
            prepared: Some(prepared),
            visuals,
            requirements,
            phase: None,
            members: metadata,
            receipts: vec![],
        }))
    }
    fn try_admit_initialization_cohort_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        batch: Option<&mut ImageDenoisingBatch>,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> InitResult<bool> {
        let prepared = prepared.get_mut::<PreparedCohort<R>>()?;
        let batch = batch
            .map(|b| b.get_mut::<ContinuousBatch<R>>())
            .transpose()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<ContinuousRequest<R>>())
            .collect::<InitResult<Vec<_>>>()?;
        self.admit_expansion_cohort(prepared, batch, &mut requests)
    }
    fn begin_initialization_cohort(
        &mut self,
        opaque: &mut ImageBatchPreparation,
        requests: &[ImageInitializationCohortRequest<'_>],
    ) -> InitResult<ImageRequestInitialization> {
        let prepared = opaque.get_mut::<PreparedCohort<R>>()?;
        validate_members(
            &prepared.pipeline,
            &self.execution.service_preparation_identity,
            &prepared.members,
            requests,
        )?;
        if prepared.phase.is_none() || prepared.prepared.is_none() {
            return Err(ImageBatchError::Rejected(
                "H3 expansion cohort requires its original Host grant and plan".into(),
            ));
        }
        let old = std::mem::take(opaque).into_inner::<PreparedCohort<R>>()?;
        let mut cohort = Cohort::new(old);
        for member in &mut cohort.members {
            member.prompt_trace = self.trace_phase_started(
                member.correlation,
                None,
                crate::image_generation::continuous_trace::ImageTracePhase::PromptExpansion,
            );
        }
        self.cancel_pre_native_members(&mut cohort, requests);
        // Keep the original text path's immediate installation. Visual work
        // returns to the worker before any producer or causal native call.
        if cohort.stage == CohortPhase::Install {
            if let Err(source) = self.install_cohort(&mut cohort) {
                self.execution.stopped = true;
                return Err(ImageBatchError::Fatal(Box::new(CohortFailure::new(
                    source, cohort,
                ))));
            }
        }
        Ok(ImageRequestInitialization::new(cohort))
    }
    fn advance_initialization_cohort(
        &mut self,
        opaque: &mut ImageRequestInitialization,
        requests: &[ImageInitializationCohortRequest<'_>],
        wake: &Waker,
    ) -> InitResult<ImageInitializationCohortProgress> {
        let cohort = opaque.get_mut::<Cohort<R>>()?;
        let result = (|| {
            validate_members(
                &cohort.pipeline,
                &self.execution.service_preparation_identity,
                &cohort.members,
                requests,
            )?;
            self.advance_cohort(cohort, requests, wake)
        })();
        match result {
            Ok(progress) => Ok(progress),
            Err(source) => {
                self.execution.stopped = true;
                let marker = Cohort::failed_marker(cohort.pipeline.clone(), cohort.assets.clone());
                let original = std::mem::replace(cohort, marker);
                Err(ImageBatchError::Fatal(Box::new(CohortFailure::new(
                    source, original,
                ))))
            }
        }
    }
    fn initialization_cohort_blocks_denoising(&self, opaque: &ImageRequestInitialization) -> bool {
        opaque
            .get::<Cohort<R>>()
            .map_or(true, |c| c.stage != CohortPhase::Done)
    }
}

/// Called only after the original native owner reports retirement. Close every
/// member at that one boundary before any member's downstream preparation; a
/// downstream rejection cannot change an already-completed expansion phase.
fn finish_retired_prompt_spans(
    members: &mut [Member],
    requests: &[ImageInitializationCohortRequest<'_>],
    outcomes: &[ExpansionParticipantResult],
    retired_at: Instant,
    mut record: impl FnMut(
        Option<u64>,
        crate::image_generation::continuous_trace::ImageTracePhaseOutcome,
        Instant,
    ),
) {
    use crate::image_generation::continuous_trace::ImageTracePhaseOutcome as TraceOutcome;
    let valid_membership = members.len() == requests.len() && members.len() == outcomes.len();
    for (index, member) in members.iter_mut().enumerate() {
        let outcome = if !valid_membership {
            // No result can be attributed to a member after a broken extent.
            // The caller preserves the original fatal protocol error.
            TraceOutcome::Failed
        } else if requests[index].cancelled {
            TraceOutcome::Cancelled
        } else {
            match &outcomes[index] {
                ExpansionParticipantResult::Expanded(_) => TraceOutcome::Completed,
                ExpansionParticipantResult::Cancelled => TraceOutcome::Cancelled,
                ExpansionParticipantResult::Failed(_) => TraceOutcome::Failed,
            }
        };
        record(member.prompt_trace.take(), outcome, retired_at);
    }
}

fn validate_members(
    pipeline: &Arc<()>,
    expected: &Arc<()>,
    members: &[Member],
    requests: &[ImageInitializationCohortRequest<'_>],
) -> InitResult<()> {
    if !Arc::ptr_eq(pipeline, expected)
        || members.len() != requests.len()
        || !members
            .iter()
            .zip(requests)
            .all(|(m, r)| m.request_id == r.request_id && m.identity.matches(r.request))
    {
        return Err(ImageBatchError::Rejected(
            "H3 expansion cohort ingress or order changed".into(),
        ));
    }
    Ok(())
}
