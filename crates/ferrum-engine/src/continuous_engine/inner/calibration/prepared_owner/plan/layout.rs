use super::*;
use prefixes::PrefixPair;
mod budget;
pub(super) use budget::finite::FinitePlan;
mod checked;
mod inventory;
mod populations;
#[cfg(test)]
mod row_capacity_tests;
pub(super) mod selection;
pub(super) mod source_inputs;
pub(super) mod work;
pub(super) use checked::build as build_checked;
pub(in crate::continuous_engine::inner::calibration) use checked::CheckedInputCursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
enum OpportunityProduct {
    Prefill,
    ContinuationPrefill {
        offset: usize,
    },
    /// Exact finite scheduler candidate, including its preparation trajectory.
    PrefillSpan {
        offset: usize,
        chunk: NonZeroU32,
    },
    Greedy,
    Full,
}

#[derive(Clone, serde::Serialize)]
pub(super) struct Case {
    product: OpportunityProduct,
    template: usize,
    width: usize,
    maximum_output: NonZeroUsize,
    release_generated: usize,
    suffix_tokens: usize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    prefix: PrefixKind,
    route: CalibrationDecodeRoute,
    reset: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    acquisition: Option<work::PreparedProbeAcquisition>,
}
impl Case {
    fn explicit_prefill_chunk(&self) -> Option<NonZeroU32> {
        match self.product {
            OpportunityProduct::PrefillSpan { chunk, .. } => Some(chunk),
            _ => None,
        }
    }

    fn prefill_chunk(
        &self,
        whole: NonZeroU32,
        row_ceiling: Option<NonZeroU32>,
        width: usize,
    ) -> Result<NonZeroU32> {
        let maximum = crate::continuous_engine::inner::calibration::geometry_projection::prefill_chunk_for_width(whole, row_ceiling, width)
            .ok_or_else(|| error("probe width exceeds frozen prefill capacity"))?;
        match self.explicit_prefill_chunk() {
            Some(chunk) if chunk > maximum => Err(error(
                "declared prefill span exceeds frozen prefill capacity",
            )),
            Some(chunk) => Ok(chunk),
            None => Ok(maximum),
        }
    }
    pub(super) fn waves(&self, prompt: usize, chunk: usize) -> Result<(usize, usize)> {
        self.waves_with_row_ceiling(prompt, chunk, None)
    }

    pub(super) fn waves_with_row_ceiling(
        &self,
        prompt: usize,
        whole_chunk: usize,
        prefill_row_ceiling: Option<NonZeroU32>,
    ) -> Result<(usize, usize)> {
        let work = work::case_work(self, prompt, whole_chunk, prefill_row_ceiling)?;
        Ok((work.declared_offers_upper, work.execution_actions))
    }
}

pub(super) fn cases(
    templates: &[AutomaticCostProbeTemplate],
    settings: &SloAutomaticCostProbeSettingsV1,
    outputs: &[NonZeroUsize],
    widths: &[usize],
    pair: &PrefixPair,
    reset: bool,
    vocabulary: usize,
    original_indices: &[usize],
    maximum_retained_bytes: usize,
) -> Result<(Vec<Case>, usize, Vec<PreparedPrefixUnavailable>)> {
    // Four ordinary cases and at most eight across the two decode products,
    // including the terminal padding below. Bound allocations before
    // expanding the template/preset/width product.
    let endpoints = templates
        .len()
        .checked_mul(settings.sampling_presets.len())
        .ok_or_else(|| error("probe case capacity overflow"))?;
    let capacity = endpoints
        .checked_mul(widths.len())
        .and_then(|n| n.checked_mul(12))
        .ok_or_else(|| error("probe case capacity overflow"))?;
    capacity
        .checked_mul(std::mem::size_of::<Case>())
        .and_then(|n| {
            n.checked_add(endpoints.checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?)
        })
        .and_then(|n| n.checked_add(8 * std::mem::size_of::<Case>()))
        .filter(|n| *n <= maximum_retained_bytes)
        .ok_or_else(|| error("probe case expansion exceeds retained capacity"))?;
    let mut cases = Vec::with_capacity(capacity);
    let mut skipped = 0;
    let mut unavailable = Vec::with_capacity(endpoints);
    for (template, actual) in templates.iter().enumerate() {
        for &preset in &settings.sampling_presets {
            if !templates::supports_sampling(actual, preset)? {
                skipped += 1;
                continue;
            }
            let prefix_unavailable =
                templates::prepared_prefix_unavailable(actual, preset, vocabulary)?;
            if let Some(reason) = prefix_unavailable {
                unavailable.push(PreparedPrefixUnavailable {
                    original_template_index: original_indices[template],
                    preset,
                    reason,
                });
            }
            let greedy = templates::declared_greedy_sampling(actual, preset)?;
            for &width in widths {
                // Four fresh ordinary cohorts preserve the prefill member
                // opportunities previously spent on duplicate forced routes.
                // The actual installed sampler chooses the execution product.
                for (maximum, cold) in [(1, true), (1, false), (2, true), (2, false)] {
                    if outputs[template].get() < maximum {
                        return Err(error("probe context/output cap cannot fit ordinary continuation and terminal"));
                    }
                    cases.push(Case {
                        product: OpportunityProduct::Prefill,
                        template,
                        width,
                        maximum_output: NonZeroUsize::new(maximum).unwrap(),
                        release_generated: 0,
                        suffix_tokens: maximum,
                        preset,
                        prefix: PrefixKind::Ordinary,
                        route: CalibrationDecodeRoute::Actual,
                        reset: reset && cold,
                        acquisition: None,
                    });
                }
                if prefix_unavailable.is_some() {
                    continue;
                }
                if pair.clean.token_ids.len() != pair.pending.token_ids.len() {
                    return Err(error(
                        "probe mixed rows need original equal-length prefix trajectories",
                    ));
                }
                let release = pair.clean.token_ids.len();
                // PhysicalEnvelope challenges zero/positive work directions,
                // not the RowSpace joint-count catalogue. Full pending covers
                // every eligible position; one proper subset also satisfies
                // the original Discovery intermediate challenge. Predictions
                // still pass all original numerical and branch checks.
                let counts: &[usize] = if width == 1 { &[0, 1] } else { &[0, 1, width] };
                for &pending_rows in counts {
                    let prefix = if pending_rows == 0 {
                        PrefixKind::Clean
                    } else if pending_rows == width {
                        PrefixKind::Pending
                    } else {
                        PrefixKind::Mixed { pending_rows }
                    };
                    for suffix_tokens in [1, 2] {
                        let maximum = release
                            .checked_add(suffix_tokens)
                            .ok_or_else(|| error("probe prefix/output overflow"))?;
                        if maximum > outputs[template].get() {
                            continue; // This real context leaves room only for the terminal suffix.
                        }
                        cases.push(Case {
                            product: if pending_rows != 0 || !greedy {
                                OpportunityProduct::Full
                            } else {
                                OpportunityProduct::Greedy
                            },
                            template,
                            width,
                            maximum_output: NonZeroUsize::new(maximum).unwrap(),
                            release_generated: release,
                            suffix_tokens,
                            preset,
                            prefix,
                            route: CalibrationDecodeRoute::Actual,
                            // Both real invalidation states recur in each
                            // cycle; do not multiply all input branches by an
                            // unnecessary cold/warm Cartesian product.
                            reset: reset && suffix_tokens == 1,
                            acquisition: None,
                        });
                    }
                }
            }
        }
    }
    if cases.is_empty() {
        return Err(error("no actual product preset supports cost probing"));
    }
    // A completed prepared cohort guarantees one original post-release
    // opportunity. Ordinary output may EOS immediately or leave pending UTF-8,
    // so it contributes only its guaranteed initial prefill here.
    let initial = cases.len();
    for index in 0..initial {
        if cases[index].product == OpportunityProduct::Prefill {
            continue;
        }
        let same = |c: &Case| {
            c.template == cases[index].template
                && c.preset == cases[index].preset
                && c.width == cases[index].width
                && c.product == cases[index].product
        };
        if cases[..index].iter().any(same) {
            continue;
        }
        let originals: Vec<_> = cases[..initial]
            .iter()
            .filter(|c| same(c))
            .cloned()
            .collect();
        let minimum = cases
            .iter()
            .filter(|c| {
                c.template == cases[index].template
                    && c.preset == cases[index].preset
                    && c.width == cases[index].width
                    && c.product == OpportunityProduct::Prefill
            })
            .count();
        let mut count = originals.len();
        while count < minimum {
            // The original pair already includes continuation. Additional fresh
            // terminal cohorts supply sample opportunities without repeating an
            // unnecessary second suffix token.
            cases.push(
                originals
                    .iter()
                    .find(|c| c.suffix_tokens == 1)
                    .ok_or_else(|| error("prepared opportunity lacks an original terminal case"))?
                    .clone(),
            );
            count += 1;
        }
    }
    Ok((cases, skipped, unavailable))
}

/// A completed ordinary cohort must execute its legal prompt spans before
/// sampling can observe EOS. These cases bind actual offsets, not synthetic
/// owners or observations; the checked inventory still proves the route.
/// Add only the declared singleton span windows. Decode/context endpoints and
/// wider-row opportunity cases remain the original independently bounded set.
fn append_scheduler_prefill_cases(
    cases: &mut Vec<Case>,
    input: &PreparedProbeInputs,
    maximum_retained_bytes: usize,
) -> Result<()> {
    if !input.reset {
        return Ok(());
    }
    let additional = input
        .continuation_windows
        .iter()
        .try_fold(0usize, |count, window| {
            let outputs = input.outputs.get(window.template)?.get().min(2);
            count.checked_add(outputs.checked_mul(3)?)
        })
        .ok_or_else(|| error("prefill candidate case count overflow"))?;
    let required = cases
        .len()
        .checked_add(additional)
        .ok_or_else(|| error("prefill candidate case count overflow"))?;
    if required > cases.capacity() {
        cases
            .capacity()
            .checked_add(required)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Case>()))
            .filter(|bytes| *bytes <= maximum_retained_bytes)
            .ok_or_else(|| error("prefill candidate cases exceed retained capacity"))?;
        cases
            .try_reserve_exact(additional)
            .map_err(|_| error("prefill candidate case allocation failed"))?;
    }
    cases
        .capacity()
        .checked_mul(std::mem::size_of::<Case>())
        .filter(|bytes| *bytes <= maximum_retained_bytes)
        .ok_or_else(|| error("prefill candidate cases exceed retained capacity"))?;
    for window in &input.continuation_windows {
        let prompt = *input
            .prompts
            .get(window.template)
            .ok_or_else(|| error("prefill candidate template differs"))?;
        let chunk = window.chunk.get() as usize;
        let middle = chunk
            .checked_mul(2)
            .ok_or_else(|| error("prefill candidate span overflow"))?;
        if prompt <= middle
            || prompt
                > chunk
                    .checked_mul(3)
                    .ok_or_else(|| error("prefill candidate span overflow"))?
        {
            return Err(error(
                "prefill candidate window lacks first/middle/final spans",
            ));
        }
        for maximum_output in 1..=input.outputs[window.template].get().min(2) {
            for offset in [0, chunk, middle] {
                cases.push(Case {
                    product: OpportunityProduct::PrefillSpan {
                        offset,
                        chunk: window.chunk,
                    },
                    template: window.template,
                    width: 1,
                    maximum_output: NonZeroUsize::new(maximum_output).unwrap(),
                    release_generated: 0,
                    suffix_tokens: maximum_output,
                    preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                    prefix: PrefixKind::Ordinary,
                    route: CalibrationDecodeRoute::Actual,
                    reset: true,
                    acquisition: None,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn append_continuation_prefill_cases(
    cases: &mut Vec<Case>,
    prompts: &[usize],
    whole_chunk: usize,
    maximum_retained_bytes: usize,
) -> Result<()> {
    append_continuation_prefill_cases_with_row_ceiling(
        cases,
        prompts,
        whole_chunk,
        None,
        maximum_retained_bytes,
    )
}

pub(super) fn append_continuation_prefill_cases_with_row_ceiling(
    cases: &mut Vec<Case>,
    prompts: &[usize],
    whole_chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    maximum_retained_bytes: usize,
) -> Result<()> {
    let whole_chunk = u32::try_from(whole_chunk)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| error("continuation prefill whole-wave capacity is invalid"))?;
    let offsets = |case: &Case| -> Result<Option<[usize; 2]>> {
        if case.product != OpportunityProduct::Prefill
            || !matches!(case.prefix, PrefixKind::Ordinary)
            || !case.reset
        {
            return Ok(None);
        }
        let prompt = *prompts
            .get(case.template)
            .ok_or_else(|| error("continuation prefill template differs"))?;
        let chunk = crate::continuous_engine::inner::calibration::geometry_projection::prefill_chunk_for_width(
            whole_chunk,
            prefill_row_ceiling,
            case.width,
        )
            .map(|chunk| chunk.get() as usize)
            .ok_or_else(|| error("continuation prefill width exceeds token capacity"))?;
        if prompt <= chunk {
            return Ok(None);
        }
        let last = (prompt - 1)
            .checked_div(chunk)
            .and_then(|n| n.checked_mul(chunk))
            .ok_or_else(|| error("continuation prefill offset overflow"))?;
        u32::try_from(last)
            .map_err(|_| error("continuation prefill exceeds physical context domain"))?;
        Ok(Some([chunk, last]))
    };
    let original = cases.len();
    let mut additional = 0usize;
    for case in &cases[..original] {
        if let Some([first, last]) = offsets(case)? {
            additional = additional
                .checked_add(1 + usize::from(first != last))
                .ok_or_else(|| error("continuation prefill case count overflow"))?;
        }
    }
    let required = original
        .checked_add(additional)
        .ok_or_else(|| error("continuation prefill case count overflow"))?;
    if required > cases.capacity() {
        // During reserve both the old and replacement backing may be live.
        cases
            .capacity()
            .checked_add(required)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Case>()))
            .filter(|n| *n <= maximum_retained_bytes)
            .ok_or_else(|| error("continuation prefill case capacity exhausted"))?;
        cases
            .try_reserve_exact(additional)
            .map_err(|_| error("continuation prefill case allocation failed"))?;
    }
    cases
        .capacity()
        .checked_mul(std::mem::size_of::<Case>())
        .filter(|n| *n <= maximum_retained_bytes)
        .ok_or_else(|| error("continuation prefill case capacity exhausted"))?;
    for index in 0..original {
        let Some([first, last]) = offsets(&cases[index])? else {
            continue;
        };
        for (position, offset) in [first, last].into_iter().enumerate() {
            if position == 1 && first == last {
                continue;
            }
            let mut span = cases[index].clone();
            span.product = OpportunityProduct::ContinuationPrefill { offset };
            cases.push(span);
        }
    }
    Ok(())
}

fn repetitions(
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    p: &StructuredServiceDeclarationV7,
) -> Result<([usize; 3], usize, usize, usize, ProbeInputOpportunityBudget)> {
    let mut minimum_cycle_offers = 0usize;
    let mut cycle_serial = 0usize;
    let mut cycle_offer_rows = 0usize;
    let mut cycle_requests = 0usize;
    for c in cases {
        let work = work::case_work(c, prompts[c.template], chunk, None)?;
        // A completed GreedyLength cohort really reaches its original output
        // budget. A Configured prefix must finish preparation and then at
        // least one original suffix wave; ordinary Configured may end at its
        // first token. Failed cohorts do not satisfy any of these bounds.
        minimum_cycle_offers = minimum_cycle_offers
            .checked_add(work.declared_offers_minimum)
            .ok_or_else(|| error("probe cycle work overflow"))?;
        cycle_serial = cycle_serial
            .checked_add(work.execution_actions)
            .ok_or_else(|| error("probe cycle work overflow"))?;
        cycle_offer_rows = cycle_offer_rows
            .checked_add(work.serial_declared_offer_rows)
            .ok_or_else(|| error("probe cycle offered rows overflow"))?;
        cycle_requests = cycle_requests
            .checked_add(work.requests)
            .ok_or_else(|| error("probe request count overflow"))?;
    }
    let input_opportunities = budget::plan(cases, prompts, chunk, p, minimum_cycle_offers)?;
    let rounds = input_opportunities.planned_cycles;
    let repetitions: [usize; 3] =
        std::array::from_fn(|phase| rounds / 3 + usize::from(phase < rounds % 3));
    let setup = work::setup_for_cases(cases)?;
    if repetitions
        .iter()
        .any(|n| n.checked_mul(cases.len()).is_none_or(|count| count > 4096))
    {
        return Err(error("probe cohort plan exceeds protocol capacity"));
    }
    Ok((
        repetitions,
        cycle_requests
            .checked_mul(rounds)
            .and_then(|n| n.checked_add(setup.requests))
            .ok_or_else(|| error("probe requests overflow"))?,
        cycle_serial
            .checked_mul(rounds)
            .and_then(|n| n.checked_add(setup.execution_actions))
            .ok_or_else(|| error("probe serial work overflow"))?,
        cycle_offer_rows
            .checked_mul(rounds)
            .ok_or_else(|| error("probe offered rows overflow"))?,
        input_opportunities,
    ))
}

fn payload_bound(
    cases: &[Case],
    order: &[Vec<usize>; 3],
    pair: &PrefixPair,
    templates: &[AutomaticCostProbeTemplate],
    unavailable_capacity: usize,
) -> Result<usize> {
    let count = || -> Option<usize> {
        let mut n = std::mem::size_of::<PreparedProbePlan>()
            .checked_add(std::mem::size_of::<StructuredPreparedOwnerBlockDeclarationV8>())?
            .checked_add(
                unavailable_capacity
                    .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
            )?
            .checked_add(cases.len().checked_mul(std::mem::size_of::<Case>())?)?;
        for t in templates {
            n = n.checked_add(t.retained_payload_bytes()?)?;
        }
        for &index in order.iter().flatten() {
            let c = cases.get(index)?;
            let mut cohort = std::mem::size_of::<PreparedProbeCohort>()
                .checked_add(std::mem::size_of::<CohortV2>())?
                .checked_add(std::mem::size_of::<Option<StructuredPrefixCohortV5>>())?
                .checked_add(
                    c.width
                        .checked_mul(std::mem::size_of::<CohortRequestV2>())?,
                )?;
            if !matches!(c.prefix, PrefixKind::Ordinary) {
                for position in 0..c.width {
                    let p = prefix_slot(c.prefix, position, &pair);
                    let mut slot = std::mem::size_of::<StructuredPrefixSlotV5>()
                        .checked_add(
                            p.token_ids
                                .len()
                                .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())?,
                        )?
                        .checked_add(
                            p.token_bytes
                                .len()
                                .checked_mul(std::mem::size_of::<Vec<u8>>())?,
                        )?;
                    for bytes in &p.token_bytes {
                        slot = slot.checked_add(bytes.len())?;
                    }
                    cohort = cohort.checked_add(slot)?;
                }
            }
            n = n.checked_add(cohort)?;
        }
        // Small bounded index/template vectors coexist while expanding.
        n.checked_add(
            templates
                .len()
                .checked_mul(std::mem::size_of::<AutomaticCostProbeTemplate>())?,
        )?
        .checked_add((templates.len() + 4).checked_mul(std::mem::size_of::<usize>())?)
    };
    count().ok_or_else(|| error("probe population retained payload overflow before expansion"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    templates: &[AutomaticCostProbeTemplate],
    settings: &SloAutomaticCostProbeSettingsV1,
    population: StructuredServiceDeclarationV7,
    limits: CostProfileLoadLimits,
    prompts: &[usize],
    outputs: &[NonZeroUsize],
    pair: PrefixPair,
    context: usize,
    width_limit: usize,
    configured_width_limit: usize,
    chunk: NonZeroU32,
    reset: bool,
    invalidation: TokenPolicyResidencyInvalidation,
    discovery: CalibrationPrefixTokenDiscoveryAuditV1,
    excluded_template_indices: Vec<usize>,
    context_coverage: &ferrum_interfaces::vnext::ExecutorDecodeContextCoverage,
) -> Result<PreparedProbePlan> {
    if templates.is_empty() || templates.len() != prompts.len() || prompts.len() != outputs.len() {
        return Err(error("probe actual template/token population differs"));
    }
    let original_indices: Vec<_> = (0..templates.len() + excluded_template_indices.len())
        .filter(|i| !excluded_template_indices.contains(i))
        .collect();
    if original_indices.len() != templates.len() {
        return Err(error("probe original template mapping differs"));
    }
    let required_geometry = geometry::ProbeGeometryRequirements::new(
        configured_width_limit,
        width_limit.min(chunk.get() as usize),
        context,
        prompts
            .iter()
            .copied()
            .min()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| error("probe geometry prompt frontier overflow"))?,
        context_coverage,
        population.maximum_retained_numeric_bytes,
    )?;
    let mut widths: Vec<_> = [1, 2, 4, 8]
        .into_iter()
        .filter(|n| *n <= width_limit && *n <= chunk.get() as usize)
        .collect();
    let vocabulary = usize::try_from(
        population
            .nonnegative_envelope
            .as_ref()
            .ok_or_else(|| error("probe physical domain absent"))?
            .workload_domain
            .limits()
            .output_vocabulary_elements
            .get(),
    )
    .map_err(|_| error("probe vocabulary does not fit host"))?;
    let (cases, skipped, unavailable, repeats, input_opportunities) = loop {
        if widths.is_empty() {
            return Err(error(
                "complete probe scenarios exceed the shared request/work budget",
            ));
        }
        let (cases, skipped, unavailable) = cases(
            templates,
            settings,
            outputs,
            &widths,
            &pair,
            reset,
            vocabulary,
            &original_indices,
            population.maximum_retained_numeric_bytes,
        )?;
        let (repeats, requests, serial, offered_rows, input_opportunities) =
            repetitions(&cases, prompts, chunk.get() as usize, &population)?;
        if requests <= settings.maximum_probe_requests.get()
            && serial <= settings.maximum_offered_waves.get()
            && offered_rows <= limits.max_samples.get()
            && offered_rows <= limits.max_total_shape_rows.get()
            // All declared input branches must be able to enter the original
            // first discovery block; otherwise a rare branch can arrive only
            // after this owner's catalogue has already frozen.
            && input_opportunities.successful_cycle_wave_upper_bound
                <= population.schedule.block_offered
        {
            break (cases, skipped, unavailable, repeats, input_opportunities);
        }
        // Input-budget admission only, before any measured cohort. Never remove
        // an already selected formal trial in response to its outcome/timing.
        widths.pop();
    };

    let order =
        std::array::from_fn(|phase| (0..repeats[phase]).flat_map(|_| 0..cases.len()).collect());
    freeze(
        PreparedProbeInputs {
            templates: templates.to_vec(),
            prompts: prompts.to_vec(),
            outputs: outputs.to_vec(),
            pair,
            population,
            limits,
            context,
            width_limit,
            configured_width_limit,
            chunk,
            prefill_row_ceiling: None,
            base_template_count: templates.len(),
            prefill_candidate_chunks: Vec::new(),
            continuation_windows: Vec::new(),
            prefix_acquisitions: Vec::new(),
            reset,
            invalidation,
            discovery,
            excluded_templates: excluded_template_indices,
            context_coverage: Arc::new(context_coverage.clone()),
            settings: settings.clone(),
            original_template_indices: original_indices,
            required_geometry,
            external_retained_bytes: 0,
            input_geometry_visit_limit: None,
            // This legacy path freezes one original source, not a series.
            maximum_retained_sources: NonZeroUsize::MIN,
        },
        FrozenCases {
            cases,
            order,
            widths,
            skipped,
            unavailable,
            input_opportunities: Some(input_opportunities),
            checked_selection: None,
            source_inputs: Vec::new(),
            preflight_charge: ProbePreflightCharge::default(),
        },
    )
}

struct FrozenCases {
    cases: Vec<Case>,
    order: [Vec<usize>; 3],
    widths: Vec<usize>,
    skipped: usize,
    unavailable: Vec<PreparedPrefixUnavailable>,
    input_opportunities: Option<ProbeInputOpportunityBudget>,
    checked_selection: Option<selection::CheckedSelection>,
    source_inputs: Vec<source_inputs::PreparedProbeSourceInputs>,
    preflight_charge: ProbePreflightCharge,
}

fn freeze(input: PreparedProbeInputs, selected: FrozenCases) -> Result<PreparedProbePlan> {
    let PreparedProbeInputs {
        templates,
        prompts,
        pair,
        population,
        limits,
        context,
        width_limit,
        configured_width_limit,
        chunk,
        prefill_row_ceiling,
        invalidation,
        discovery,
        excluded_templates: excluded_template_indices,
        context_coverage,
        settings,
        original_template_indices: original_indices,
        required_geometry,
        external_retained_bytes,
        ..
    } = input;
    let FrozenCases {
        cases,
        order,
        widths,
        skipped,
        unavailable,
        input_opportunities,
        checked_selection,
        source_inputs,
        preflight_charge,
    } = selected;
    let templates = templates.as_slice();
    let prompts = prompts.as_slice();
    let context_coverage = &context_coverage;
    let (mut requests, mut serial, offered_rows) =
        order
            .iter()
            .flatten()
            .try_fold((0usize, 0usize, 0usize), |(r, w, o), &i| {
                let c = cases
                    .get(i)
                    .ok_or_else(|| error("selected case index differs"))?;
                let work = work::case_work(
                    c,
                    prompts[c.template],
                    chunk.get() as usize,
                    prefill_row_ceiling,
                )?;
                Ok::<_, FerrumError>((
                    r.checked_add(work.requests)
                        .ok_or_else(|| error("request bound overflow"))?,
                    w.checked_add(work.execution_actions)
                        .ok_or_else(|| error("action bound overflow"))?,
                    o.checked_add(work.serial_declared_offer_rows)
                        .ok_or_else(|| error("offered row bound overflow"))?,
                ))
            })?;
    let mut add_setup = |setup: work::CaseWork| -> Result<()> {
        requests = requests
            .checked_add(setup.requests)
            .ok_or_else(|| error("setup request bound overflow"))?;
        serial = serial
            .checked_add(setup.execution_actions)
            .ok_or_else(|| error("setup action bound overflow"))?;
        Ok(())
    };
    if let Some(selection) = &checked_selection {
        for batch in selection.batches.iter().filter(|batch| batch.scheduled) {
            add_setup(work::setup_for_indices(
                &cases,
                &batch.representative_case_indices,
            )?)?;
        }
        if requests != selection.requests
            || serial != selection.serial_wave_upper_bound
            || offered_rows != selection.declared_offer_row_bound
        {
            return Err(error(
                "frozen source work differs from original selection reservation",
            ));
        }
    } else {
        add_setup(work::setup_for_cases(&cases)?)?;
    }
    if order.iter().any(|p| p.len() > 4096)
        || requests
            .checked_add(preflight_charge.readiness_reserved_requests)
            .is_none_or(|n| n > settings.maximum_probe_requests.get())
        || preflight_charge.planning_reserved_requests
            > settings.maximum_input_projection_requests.get()
        || serial > settings.maximum_offered_waves.get()
        || offered_rows > limits.max_samples.get()
        || offered_rows > limits.max_total_shape_rows.get()
    {
        return Err(error(
            "selected probe plan exceeds the original shared budget",
        ));
    }
    let planned_cohorts = order.iter().map(Vec::len).sum::<usize>();
    let preallocated_payload =
        payload_bound(&cases, &order, &pair, templates, unavailable.capacity())?
            .checked_add(
                source_inputs::retained_sources_bytes(&source_inputs)
                    .ok_or_else(|| error("source input retained capacity overflow"))?,
            )
            .ok_or_else(|| error("source input retained capacity overflow"))?
            .checked_add(
                checked_selection
                    .as_ref()
                    .map_or(Some(0), |s| s.retained_payload_bytes())
                    .ok_or_else(|| error("checked selection retained overflow"))?,
            )
            .ok_or_else(|| error("checked selection retained overflow"))?
            .checked_add(
                required_geometry
                    .retained_payload_bytes()
                    .ok_or_else(|| error("probe geometry retained capacity overflow"))?,
            )
            .ok_or_else(|| error("probe geometry retained capacity overflow"))?
            .checked_add(coverage::retained_upper_bound(
                configured_width_limit,
                planned_cohorts,
                context_coverage,
            )?)
            .ok_or_else(|| error("probe coverage retained capacity overflow"))?
            .checked_add(
                population
                    .nonnegative_envelope
                    .as_ref()
                    .and_then(|contract| contract.algorithm_universe.as_ref())
                    .map_or(Some(0), |universe| universe.retained_payload_bytes())
                    .ok_or_else(|| error("algorithm universe retained capacity overflow"))?,
            )
            .ok_or_else(|| error("algorithm universe retained capacity overflow"))?;
    let manifest_budget = population
        .maximum_retained_numeric_bytes
        .checked_sub(external_retained_bytes)
        .ok_or_else(|| error("probe cursor exceeds shared retained capacity"))?
        .checked_sub(preallocated_payload)
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            error("probe population exceeds shared retained capacity before expansion")
        })?;
    let counts = std::array::from_fn::<_, 3, _>(|i| order[i].len());
    let mut phases: [Vec<CohortV2>; 3] = std::array::from_fn(|i| Vec::with_capacity(counts[i]));
    let mut prefix_phases: [Vec<Option<StructuredPrefixCohortV5>>; 3] =
        std::array::from_fn(|i| Vec::with_capacity(counts[i]));
    let mut cohorts = Vec::with_capacity(counts.iter().sum());
    let mut next_seed = 0u64;
    for pass in 0..3 {
        for &case_index in &order[pass] {
            let case = cases
                .get(case_index)
                .ok_or_else(|| error("selected case index differs"))?;
            let ordinal = phases[pass].len();
            phases[pass].push(CohortV2 {
                manifest_case: u32::try_from(ordinal)
                    .map_err(|_| error("probe cohort index overflow"))?,
                repetition: 0,
                requests: vec![
                    CohortRequestV2 {
                        manifest_prompt: original_indices[case.template] as u32,
                        maximum_output: case.maximum_output.get() as u64
                    };
                    case.width
                ],
            });
            prefix_phases[pass].push((!matches!(case.prefix, PrefixKind::Ordinary)).then(|| {
                StructuredPrefixCohortV5 {
                    release_generated: case.release_generated as u64,
                    slots: (0..case.width)
                        .map(|position| prefix_slot(case.prefix, position, &pair).clone())
                        .collect(),
                }
            }));
            cohorts.push(PreparedProbeCohort {
                pass,
                ordinal,
                template: case.template,
                original_template_index: original_indices[case.template],
                width: case.width,
                maximum_output: case.maximum_output,
                suffix_tokens: case.suffix_tokens,
                preset: case.preset,
                prefix: case.prefix,
                route: case.route,
                reset_token_policy: case.reset,
                prefill_chunk: case.explicit_prefill_chunk(),
                native_acquisition: case.acquisition,
                acquisition_key: None,
                seed: next_seed,
            });
            next_seed = next_seed
                .checked_add(case.width as u64)
                .ok_or_else(|| error("probe seed space exhausted"))?;
        }
    }
    let manifest = manifest::freeze(
        templates,
        &original_indices,
        &excluded_template_indices,
        &unavailable,
        input_opportunities.as_ref(),
        checked_selection.as_ref(),
        &cohorts,
        invalidation,
        manifest_budget,
    )?;
    let declaration = StructuredPreparedOwnerBlockDeclarationV8 {
        population,
        cohort_plan: CohortPlanV2 { phases },
        native_prefix_acquisition: None,
        prefix_plan: StructuredPrefixPlanV5 {
            phases: prefix_phases,
        },
        cohort_manifest_payload: manifest,
        maximum_offered_waves: settings.maximum_offered_waves.get(),
    };
    declaration
        .validate()
        .map_err(|e| error(format!("probe original declaration: {e}")))?;
    let audit = PreparedProbePlanAudit {
        required_geometry,
        input_coverage: coverage::inspect(
            &cohorts,
            prompts,
            configured_width_limit,
            context,
            context_coverage,
        )?,
        prepared_prefix_unavailable: unavailable,
        input_opportunities,
        checked_selection,
        excluded_template_indices,
        effective_context: context,
        effective_maximum_rows: width_limit,
        selected_widths: widths,
        planned_cohorts: cohorts.len(),
        planned_requests: requests,
        serial_wave_bound: serial,
        declared_offer_row_bound: offered_rows,
        original_block_offered: declaration.population.schedule.block_offered,
        token_policy_invalidation: invalidation,
        skipped_endpoint_presets: skipped,
        token_ids_examined: discovery.token_ids_examined,
        token_bytes_charged: discovery.token_bytes_charged,
        token_utf8_transitions: discovery.utf8_transitions,
        token_peak_search_states: discovery.peak_search_states,
    };
    let execution = PreparedProbeExecutionPlan {
        cohorts,
        audit,
        source_inputs,
        templates: templates.to_vec(),
        prefill_chunk: chunk,
        prefill_row_ceiling,
    };
    let retained = declaration
        .retained_payload_bytes()
        .and_then(|n| n.checked_add(execution.retained_payload_bytes()?))
        .and_then(|n| n.checked_add(external_retained_bytes))
        .filter(|n| *n <= declaration.population.maximum_retained_numeric_bytes)
        .ok_or_else(|| error("probe plan exceeds the one shared retained budget"))?;
    let _ = retained; // The live adapter charges the execution plan until Drop.
    Ok(PreparedProbePlan {
        declaration,
        limits,
        execution,
        preflight_charge,
        external_retained_bytes,
    })
}

fn prefix_slot(prefix: PrefixKind, position: usize, pair: &PrefixPair) -> &StructuredPrefixSlotV5 {
    match prefix {
        PrefixKind::Clean => &pair.clean,
        PrefixKind::Pending => &pair.pending,
        PrefixKind::Mixed { pending_rows } if position < pending_rows => &pair.pending,
        PrefixKind::Mixed { .. } => &pair.clean,
        PrefixKind::Ordinary => unreachable!("ordinary cohorts have no prefix slots"),
    }
}
