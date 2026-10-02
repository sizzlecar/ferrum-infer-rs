//! A bounded, predeclared complete-request probe population. The plan is not
//! execution evidence, a numerical qualification, or permission to change a
//! request's installed policy. All samples come from the ordinary source8 path.
use super::super::{
    cohort_driver::{ProbeCohortSettings, ProbePreflightCharge, ProbeRequest},
    token_preparation::*,
};
use super::*;
use crate::AutomaticCostProbeTemplate;
use ferrum_interfaces::{
    execution_cost::CostWorkloadDomainV1, model_executor::TokenPolicyResidencyInvalidation,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{
        prefixes::{StructuredPrefixCohortV5, StructuredPrefixPlanV5, StructuredPrefixSlotV5},
        windows::{CohortPlanV2, CohortRequestV2, CohortV2},
    },
    cost_profile::StructuredServiceDeclarationV7,
};
use ferrum_types::{
    SloAutomaticCalibrationSettingsV1, SloAutomaticCostProbeSamplingPresetV1,
    SloAutomaticCostProbeSettingsV1,
};
use std::num::{NonZeroU32, NonZeroUsize};

mod context_variants;
mod coverage;
mod geometry;
mod layout;
mod manifest;
mod population;
mod prefixes;
mod series;
mod templates;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PrefixKind {
    Ordinary,
    Clean,
    Pending,
    Mixed { pending_rows: usize },
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeCohort {
    pub pass: usize,
    pub ordinal: usize,
    template: usize,
    original_template_index: usize,
    width: usize,
    maximum_output: NonZeroUsize,
    suffix_tokens: usize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    prefix: PrefixKind,
    route: CalibrationDecodeRoute,
    reset_token_policy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_chunk: Option<NonZeroU32>,
    seed: u64,
}

#[derive(Debug, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbePlanAudit {
    /// Required input geometries, distinct from actually scheduled cohorts.
    pub required_geometry: geometry::ProbeGeometryRequirements,
    /// Planned input opportunities only. This cannot establish learned-model
    /// coverage, qualification or a joint execution-path guarantee.
    pub input_coverage: coverage::ProbeInputCoverage,
    pub prepared_prefix_unavailable: Vec<PreparedPrefixUnavailable>,
    /// Declaration opportunities; never the observed eligible population.
    pub input_opportunities: Option<ProbeInputOpportunityBudget>,
    pub checked_selection: Option<layout::selection::CheckedSelection>,
    pub excluded_template_indices: Vec<usize>,
    pub effective_context: usize,
    pub effective_maximum_rows: usize,
    pub selected_widths: Vec<usize>,
    pub planned_cohorts: usize,
    pub planned_requests: usize,
    /// No-retry serial action bound, including checkpoint setup and restores;
    /// natural terminals may perform less work.
    /// This is neither an observed population nor a promise of a complete block.
    pub serial_wave_bound: usize,
    /// Inference rows only. Native transfers do not become numerical samples.
    pub declared_offer_row_bound: usize,
    pub original_block_offered: usize,
    pub token_policy_invalidation: TokenPolicyResidencyInvalidation,
    pub skipped_endpoint_presets: usize,
    pub token_ids_examined: usize,
    pub token_bytes_charged: usize,
    pub token_utf8_transitions: usize,
    pub token_peak_search_states: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct PreparedPrefixUnavailable {
    pub original_template_index: usize,
    pub preset: SloAutomaticCostProbeSamplingPresetV1,
    pub reason: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct ProbeInputOpportunityBudget {
    pub minimum_input_family_opportunities_per_cycle: usize,
    pub minimum_original_offers_per_completed_cycle: usize,
    pub discovery_cycles: usize,
    pub phase_cycles: [usize; 3],
    pub planned_cycles: usize,
    pub required_original_offers: usize,
    pub successful_cycle_wave_upper_bound: usize,
    pub phase_original_offer_bounds: [usize; 3],
    pub maximum_fresh_member_span: [usize; 3],
}

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbePlan {
    declaration: StructuredPreparedOwnerBlockDeclarationV8,
    limits: CostProfileLoadLimits,
    execution: PreparedProbeExecutionPlan,
    preflight_charge: ProbePreflightCharge,
    /// Frozen input cursor retained while this child is collected.
    external_retained_bytes: usize,
}

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeExecutionPlan {
    pub cohorts: Vec<PreparedProbeCohort>,
    pub audit: PreparedProbePlanAudit,
    templates: Vec<AutomaticCostProbeTemplate>,
    prefill_chunk: NonZeroU32,
    prefill_row_ceiling: Option<NonZeroU32>,
}

impl PreparedProbePlan {
    pub(in crate::continuous_engine::inner::calibration) fn preflight_charge(
        &self,
    ) -> ProbePreflightCharge {
        self.preflight_charge
    }

    pub(in crate::continuous_engine::inner::calibration) fn audit(
        &self,
    ) -> &PreparedProbePlanAudit {
        &self.execution.audit
    }

    pub fn into_parts(
        self,
    ) -> (
        StructuredPreparedOwnerBlockDeclarationV8,
        CostProfileLoadLimits,
        PreparedProbeExecutionPlan,
    ) {
        (self.declaration, self.limits, self.execution)
    }
}

#[derive(Clone)]
struct PrefillCandidateWindow {
    template: usize,
    chunk: NonZeroU32,
}

#[derive(Clone)]
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeInputs {
    templates: Vec<AutomaticCostProbeTemplate>,
    prompts: Vec<usize>,
    outputs: Vec<NonZeroUsize>,
    pair: prefixes::PrefixPair,
    population: StructuredServiceDeclarationV7,
    limits: CostProfileLoadLimits,
    context: usize,
    width_limit: usize,
    configured_width_limit: usize,
    chunk: NonZeroU32,
    prefill_row_ceiling: Option<NonZeroU32>,
    // Extra product-rendered windows are only prefill cases, never a new
    // template × decode/preset/width Cartesian product.
    base_template_count: usize,
    prefill_candidate_chunks: Vec<NonZeroU32>,
    continuation_windows: Vec<PrefillCandidateWindow>,
    reset: bool,
    invalidation: TokenPolicyResidencyInvalidation,
    discovery: CalibrationPrefixTokenDiscoveryAuditV1,
    excluded_templates: Vec<usize>,
    context_coverage: Arc<ferrum_interfaces::vnext::ExecutorDecodeContextCoverage>,
    settings: SloAutomaticCostProbeSettingsV1,
    original_template_indices: Vec<usize>,
    required_geometry: geometry::ProbeGeometryRequirements,
    external_retained_bytes: usize,
    input_geometry_visit_limit: Option<std::num::NonZeroU64>,
    /// One independently qualified source owns one retained capture origin,
    /// regardless of its child count. Share the live runtime's actual limit.
    maximum_retained_sources: NonZeroUsize,
}

impl PreparedProbeInputs {
    /// Cold payload still alive through inventory/selection. Shared Arcs are
    /// conservatively charged in full, including identity strings and headers.
    pub(in crate::continuous_engine::inner::calibration) fn retained_payload_bytes(
        &self,
    ) -> Option<usize> {
        use ferrum_interfaces::vnext::{
            BoundDecodeContextCoverage, DecodeContextBoundary, DecodeContextCoverage,
            ExecutorDecodeContextCoverage,
        };
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.templates
                    .capacity()
                    .checked_mul(std::mem::size_of::<AutomaticCostProbeTemplate>())?,
            )?
            .checked_add(
                self.prompts
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.outputs
                    .capacity()
                    .checked_mul(std::mem::size_of::<NonZeroUsize>())?,
            )?
            .checked_add(
                self.excluded_templates
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.original_template_indices
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.settings
                    .sampling_presets
                    .capacity()
                    .checked_mul(std::mem::size_of::<SloAutomaticCostProbeSamplingPresetV1>())?,
            )?
            .checked_add(self.required_geometry.retained_payload_bytes()?)?
            .checked_add(
                self.prefill_candidate_chunks
                    .capacity()
                    .checked_mul(std::mem::size_of::<NonZeroU32>())?,
            )?
            .checked_add(
                self.continuation_windows
                    .capacity()
                    .checked_mul(std::mem::size_of::<PrefillCandidateWindow>())?,
            )?;
        if let Some(universe) = self
            .population
            .nonnegative_envelope
            .as_ref()
            .and_then(|contract| contract.algorithm_universe.as_ref())
        {
            bytes = bytes.checked_add(universe.retained_payload_bytes()?)?;
        }
        for template in &self.templates {
            bytes = bytes.checked_add(
                template
                    .retained_payload_bytes()?
                    .checked_sub(std::mem::size_of::<AutomaticCostProbeTemplate>())?,
            )?;
        }
        for slot in [&self.pair.clean, &self.pair.pending] {
            bytes = bytes
                .checked_add(
                    slot.token_ids
                        .capacity()
                        .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())?,
                )?
                .checked_add(
                    slot.token_bytes
                        .capacity()
                        .checked_mul(std::mem::size_of::<Vec<u8>>())?,
                )?;
            for fragment in &slot.token_bytes {
                bytes = bytes.checked_add(fragment.capacity())?;
            }
        }
        bytes = bytes
            .checked_add(std::mem::size_of::<ExecutorDecodeContextCoverage>())?
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add(
                self.context_coverage
                    .nodes
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoundDecodeContextCoverage>())?,
            )?;
        for node in &self.context_coverage.nodes {
            let capacity = match &node.coverage {
                DecodeContextCoverage::Unknown {
                    known_boundaries, ..
                } => known_boundaries.capacity(),
                DecodeContextCoverage::Declared { boundaries, .. } => boundaries.capacity(),
            };
            bytes = bytes
                .checked_add(capacity.checked_mul(std::mem::size_of::<DecodeContextBoundary>())?)?
                .checked_add(node.node_id.as_str().len())?
                .checked_add(node.provider_id.as_str().len())?
                .checked_add(4 * std::mem::size_of::<usize>())?;
        }
        Some(bytes)
    }

    pub async fn new(
        session: &mut CalibrationSession,
        automatic: &SloAutomaticCalibrationSettingsV1,
        templates: &[AutomaticCostProbeTemplate],
    ) -> Result<Self> {
        automatic.validate().map_err(FerrumError::config)?;
        let settings = &automatic.cost_probe;
        settings.validate().map_err(FerrumError::config)?;
        if templates.is_empty() || templates.len() > 16 {
            return Err(error(
                "automatic cost probe needs bounded real product templates",
            ));
        }
        // This discovers a real isolated capability, not a promise inferred
        // from a backend name. Every declared cold cohort repeats the operation.
        let invalidation = session.invalidate_token_policy_residency().await;
        let reset = match invalidation {
            TokenPolicyResidencyInvalidation::Cleared { .. } => true,
            TokenPolicyResidencyInvalidation::Unsupported => false,
            TokenPolicyResidencyInvalidation::Unavailable { .. } => {
                return Err(error(format!(
                    "automatic cost probe residency boundary unavailable: {invalidation:?}"
                )));
            }
        };
        let inner = &session.engine.inner;
        let domain = inner
            .cost_runtime
            .as_ref()
            .and_then(|r| r.workload_domain())
            .ok_or_else(|| {
                error("automatic cost probe requires the actual immutable workload domain")
            })?
            .clone();
        let context = crate::continuous_engine::effective_request_context_capacity(
            &inner.config,
            &inner.runtime_config,
            inner.model_executor.kv_capacity(),
        )
        .unwrap_or(session.context_capacity())
        .min(session.context_capacity())
        .min(domain.limits().maximum_context_tokens.get() as usize);
        // Compare the plan against the configured workload domain. The private
        // probe request limit must not hide wider configured serving inputs.
        let configured_width_limit = inner
            .config
            .scheduler
            .max_running_requests
            .min(inner.config.batching.max_batch_size)
            .min(inner.config.batching.max_num_batched_tokens)
            .min(inner.model_executor.capabilities().max_batch_size)
            .min(domain.limits().maximum_rows.get() as usize)
            .min(
                usize::try_from(domain.limits().maximum_scheduled_tokens_per_wave.get())
                    .unwrap_or(usize::MAX),
            );
        let width_limit = settings
            .maximum_concurrent_requests
            .get()
            .min(session.limits.maximum_requests().get())
            .min(configured_width_limit);
        let chunk = inner.config.batching.max_num_batched_tokens.min(
            usize::try_from(domain.limits().maximum_scheduled_tokens_per_wave.get())
                .unwrap_or(usize::MAX),
        );
        let chunk = NonZeroU32::new(u32::try_from(chunk).unwrap_or(u32::MAX))
            .ok_or_else(|| error("automatic cost probe has no actual prefill capacity"))?;
        let prefill_row_ceiling = [
            inner.config.scheduler.prefill_step_chunk,
            inner.runtime_config.chunked_prefill_size,
        ]
        .into_iter()
        .flatten()
        .min()
        .map(|ceiling| {
            let ceiling = u32::try_from(ceiling)
                .map_err(|_| error("automatic prefill row ceiling exceeds typed token domain"))?;
            NonZeroU32::new(ceiling)
                .ok_or_else(|| error("automatic prefill row ceiling must be positive"))
        })
        .transpose()?;
        let prefill_candidate_chunks = if let (true, Some(alignment)) =
            (reset, inner.model_executor.guarded_prefill_granularity())
        {
            use ferrum_scheduler::implementations::continuous::prefill_reference::{
                piecewise_chunk_candidates, ReferenceChunkLimits,
            };
            let alignment = u32::try_from(alignment.get())
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| error("probe prefill alignment exceeds typed token domain"))?;
            let maximum_tokens = super::super::geometry_projection::prefill_chunk_for_width(
                chunk,
                prefill_row_ceiling,
                1,
            )
            .ok_or_else(|| error("probe singleton prefill exceeds whole-wave capacity"))?;
            // Automatic startup's declared singleton reference protocol uses
            // unit granules (startup/plan.rs). Imported references retain their
            // original granule. No measured reference is forged here.
            let granule = inner
                .prefill_reference_runtime
                .as_ref()
                .map_or(NonZeroU32::MIN, |runtime| {
                    runtime.calibration().protocol().granule_tokens
                });
            if inner
                .prefill_reference_runtime
                .as_ref()
                .is_some_and(|runtime| runtime.calibration().piecewise_domain().is_none())
            {
                // Imported exact curves have their own endpoint menu. Keep
                // their existing calibration inputs rather than inventing a
                // piecewise declaration not present in that reference.
                Vec::new()
            } else {
                piecewise_chunk_candidates(
                    u32::try_from(context - 1)
                        .map_err(|_| error("probe context exceeds typed token domain"))?,
                    granule,
                    ReferenceChunkLimits {
                        maximum_tokens,
                        alignment,
                        allow_final_short_chunk: false,
                        maximum_candidates: NonZeroUsize::new(64).unwrap(),
                    },
                )
                .map_err(|reason| error(format!("probe declared prefill candidates: {reason:?}")))?
            }
        } else {
            Vec::new()
        };
        let tokenizer = inner.tokenizer.as_ref();
        let population = population::declaration(automatic, domain)?;
        let (templates, excluded_templates) =
            templates::select(templates, population.maximum_retained_numeric_bytes)?;
        let prompt_tokens = templates
            .iter()
            .map(|t| {
                let n = tokenizer.encode(t.prompt(), true)?.len();
                if n == 0 || n >= context {
                    return Err(error("probe prompt leaves no original context for output"));
                }
                Ok(n)
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = prompt_tokens
            .iter()
            .map(|n| {
                NonZeroUsize::new(settings.maximum_output_tokens.get().min(context - n))
                    .ok_or_else(|| error("probe prompt/output exceeds actual context"))
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum_prefix_output = outputs.iter().map(|n| n.get()).min().unwrap();
        let (pair, discovery) = prefixes::discover(
            tokenizer,
            &templates,
            NonZeroUsize::new(maximum_prefix_output).unwrap(),
            settings,
        )?;
        let import = &inner.config.scheduler.slo.cost_observation.profile_import;
        let limits = CostProfileLoadLimits {
            max_file_bytes: import.max_file_bytes,
            max_samples: import.max_samples,
            max_total_shape_rows: import.max_total_shape_rows,
            max_source_field_bytes: import.max_source_field_bytes,
            max_profile_age_ns: import.max_profile_age_ns,
            max_clock_error_ns: import.max_clock_error_ns,
        };
        let original_template_indices: Vec<_> = (0..templates.len() + excluded_templates.len())
            .filter(|index| !excluded_templates.contains(index))
            .collect();
        let required_geometry = geometry::ProbeGeometryRequirements::new(
            configured_width_limit,
            width_limit.min(chunk.get() as usize),
            context,
            prompt_tokens
                .iter()
                .copied()
                .min()
                .and_then(|n| n.checked_add(pair.clean.token_ids.len()))
                .ok_or_else(|| error("probe first suffix frontier overflow"))?,
            &inner.model_executor.decode_context_coverage(),
            population.maximum_retained_numeric_bytes,
        )?;
        let base_template_count = templates.len();
        let mut prepared = Self {
            templates,
            prompts: prompt_tokens,
            outputs,
            pair,
            population,
            limits,
            context,
            width_limit,
            configured_width_limit,
            chunk,
            prefill_row_ceiling,
            base_template_count,
            prefill_candidate_chunks,
            continuation_windows: Vec::new(),
            reset,
            invalidation,
            discovery,
            excluded_templates,
            context_coverage: inner.model_executor.decode_context_coverage(),
            settings: settings.clone(),
            original_template_indices,
            required_geometry,
            external_retained_bytes: 0,
            maximum_retained_sources: automatic.maximum_retained_generations,
            input_geometry_visit_limit: match automatic.input_readiness {
                ferrum_types::SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {} => None,
                ferrum_types::SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV1 {
                    maximum_geometry_visits,
                    ..
                }
                | ferrum_types::SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV2 {
                    maximum_geometry_visits,
                    ..
                }
                | ferrum_types::SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
                    maximum_geometry_visits,
                    ..
                } => Some(maximum_geometry_visits),
            },
        };
        context_variants::expand(&mut prepared, tokenizer)?;
        Ok(prepared)
    }

    pub(in crate::continuous_engine::inner::calibration) async fn finish(
        self,
        session: &mut CalibrationSession,
        budget: &mut super::super::cohort_driver::ProbeExecutionBudget,
    ) -> Result<PreparedProbePlan> {
        // The private startup driver composes several substantial executor
        // futures. Keep this cold phase off every outer engine-builder frame.
        Box::pin(layout::build_checked(session, self, budget)).await
    }

    pub(in crate::continuous_engine::inner::calibration) fn into_cursor(
        self,
    ) -> Result<layout::CheckedInputCursor> {
        layout::CheckedInputCursor::new(self)
    }
}

impl PreparedProbeExecutionPlan {
    pub fn cohorts(&self) -> &[PreparedProbeCohort] {
        &self.cohorts
    }
    pub fn requests_for(
        &self,
        cohort: &PreparedProbeCohort,
    ) -> Result<(Vec<ProbeRequest>, ProbeCohortSettings)> {
        let original = self
            .cohorts
            .iter()
            .find(|c| std::ptr::eq(*c, cohort))
            .ok_or_else(|| error("probe cohort is not in this frozen execution plan"))?;
        let template = self
            .templates
            .get(original.template)
            .ok_or_else(|| error("probe template index differs"))?;
        let mut requests = Vec::with_capacity(original.width);
        for slot in 0..original.width {
            let seed = original
                .seed
                .checked_add(slot as u64)
                .ok_or_else(|| error("probe seed overflow"))?;
            let (request, contract) =
                template.instantiate(original.maximum_output, seed, original.preset)?;
            requests.push(ProbeRequest { request, contract });
        }
        Ok((
            requests,
            ProbeCohortSettings {
                prefill_plan: if matches!(original.prefix, PrefixKind::Ordinary) {
                    super::super::cohort_driver::ProbePrefillPlan::Joint
                } else {
                    super::super::cohort_driver::ProbePrefillPlan::PreparedSequentialV1
                },
                // The frozen chunk is also bounded by the original whole-wave
                // token cap. Every row receives its checked share; the driver
                // can submit the entire declared width without exceeding it.
                prefill_chunk: {
                    let maximum = super::super::geometry_projection::prefill_chunk_for_width(
                        self.prefill_chunk,
                        self.prefill_row_ceiling,
                        original.width,
                    )
                    .ok_or_else(|| error("probe width exceeds whole-wave token capacity"))?;
                    match original.prefill_chunk {
                        Some(chunk) if chunk > maximum => {
                            return Err(error(
                                "frozen probe span exceeds original prefill capacity",
                            ))
                        }
                        Some(chunk) => chunk,
                        None => maximum,
                    }
                },
                decode_route: original.route,
                reset_token_policy: original.reset_token_policy,
            },
        ))
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        let mut n = std::mem::size_of::<Self>()
            .checked_add(
                self.cohorts
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedProbeCohort>())?,
            )?
            .checked_add(
                self.audit
                    .selected_widths
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.audit
                    .excluded_template_indices
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.audit
                    .prepared_prefix_unavailable
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
            )?
            .checked_add(
                self.templates
                    .capacity()
                    .checked_mul(std::mem::size_of::<AutomaticCostProbeTemplate>())?,
            )?;
        for template in &self.templates {
            n = n.checked_add(template.retained_payload_bytes()?)?;
        }
        n = n.checked_add(self.audit.input_coverage.retained_payload_bytes()?)?;
        n = n.checked_add(self.audit.required_geometry.retained_payload_bytes()?)?;
        if let Some(selection) = &self.audit.checked_selection {
            n = n.checked_add(selection.retained_payload_bytes()?)?;
        }
        Some(n)
    }
}

fn error(message: impl Into<String>) -> FerrumError {
    FerrumError::config(message.into())
}
