use ferrum_execution::{CompiledProgram, ProgramCompilationError};
use ferrum_runtime::DeviceRuntime;
use ferrum_types::{
    EngineConfig, Result, StartupConcurrencySearch, StartupMemoryPlan, StartupWorkload,
};

pub(super) fn requested(plan: &StartupMemoryPlan) -> bool {
    plan.request.search_serving_concurrency && !plan.request.sequences_is_explicit
}

pub(super) fn upper_bound(engine: &EngineConfig) -> usize {
    // The product validates max_num_seqs <= max_num_batched_tokens. Retain
    // that same public contract so users can replay Auto's effective limits.
    engine
        .output
        .max_sessions
        .min(ferrum_scheduler::ContinuousBatchConfig::default().max_decode_batch)
        .min(engine.batching.max_num_batched_tokens)
        .min(u32::MAX as usize)
}

/// Keep the first verified context and batch limits. Serving and graph warmup
/// must both fit on the exact compilation that will be initialized.
pub(super) fn expand<'a, R: DeviceRuntime>(
    engine: &EngineConfig,
    fitted: &mut StartupMemoryPlan,
    selected: CompiledProgram<'a, R>,
    mut compile: impl FnMut(
        &EngineConfig,
    ) -> std::result::Result<CompiledProgram<'a, R>, ProgramCompilationError>,
) -> Result<CompiledProgram<'a, R>> {
    let original = fitted.selected.max_sequences;
    let admission_bound = upper_bound(engine);
    let upper_bound = selected.serving_concurrency_capture_limit(engine, admission_bound)?;
    if upper_bound < admission_bound {
        fitted.reasons.push(format!(
            "concurrency search limited to {upper_bound}: a wider startup capture exceeds the fitted context of {} tokens",
            fitted.selected.context_tokens
        ));
    }
    let budget = fitted.request.usable_capacity_bytes;
    let mut warmup_peak = selected.startup_capture_warmup_peak_bytes()?;
    let mut lower = original;
    let mut upper = upper_bound;
    let mut best = if let Some(bytes) = warmup_peak.filter(|bytes| *bytes > budget) {
        fitted.reasons.push(format!(
            "concurrency candidate {original}: capture warmup requires {bytes} bytes, usable {budget} bytes"
        ));
        drop(selected);
        // The baseline's single-long and many-shallow probes passed, but its
        // actual capture workload cannot fit. Respect explicitly declared graph
        // widths while looking for a smaller cohort at the same context.
        let required_width = engine
            .backend
            .reusable_execution_capture
            .exact_decode_widths
            .as_ref()
            .and_then(|widths| widths.iter().copied().max())
            .unwrap_or(1);
        lower = required_width.saturating_sub(1);
        upper = original - 1;
        None
    } else {
        Some(selected)
    };
    let mut candidate = if best.is_some() {
        upper
    } else {
        lower + (upper - lower).div_ceil(2)
    };
    let mut compilations = 0;
    let mut may_project_next = true;
    let context_tokens = fitted.selected.context_tokens;
    let chunk_tokens = context_tokens.min(fitted.selected.max_batch_tokens);
    let mut trial_engine = engine.clone();
    while lower < upper {
        let mut projected_next = None;
        trial_engine.scheduler.max_running_requests = candidate;
        compilations += 1;
        match compile(&trial_engine) {
            Ok(trial) => {
                let prefill_peak = trial.startup_peak_bytes(StartupWorkload::Prefill {
                    context_tokens,
                    chunk_tokens,
                })?;
                let decode = |active_sequences| {
                    trial.startup_peak_bytes(StartupWorkload::Decode {
                        context_tokens,
                        active_sequences,
                    })
                };
                let decode_peak = decode(candidate)?;
                let trial_warmup = trial.startup_capture_warmup_peak_bytes()?;
                let largest_peak = prefill_peak.max(decode_peak).max(trial_warmup.unwrap_or(0));
                if largest_peak > budget {
                    let capture = trial_warmup
                        .map(|bytes| format!(", capture warmup {bytes}"))
                        .unwrap_or_default();
                    fitted.reasons.push(format!(
                        "concurrency candidate {candidate}: prefill {prefill_peak}, decode {decode_peak}{capture}, usable {budget} bytes"
                    ));
                }
                if largest_peak <= budget {
                    lower = candidate;
                    fitted.selected.max_sequences = candidate;
                    fitted.context_peak_bytes = prefill_peak;
                    fitted.decode_peak_bytes = decode_peak;
                    warmup_peak = trial_warmup;
                    best = Some(trial);
                } else if prefill_peak <= budget && decode_peak > budget && may_project_next {
                    // Monotone inside this plan; another compiled width may
                    // choose a different layout. Use this projection once as
                    // a guess. Exact trials continue on both sides of it.
                    let mut low = lower;
                    upper = candidate - 1;
                    let mut high = upper;
                    while low < high {
                        let middle = low + (high - low).div_ceil(2);
                        if decode(middle)? <= budget {
                            low = middle;
                        } else {
                            high = middle - 1;
                        }
                    }
                    projected_next = (low > lower).then_some(low);
                    may_project_next = false;
                } else {
                    upper = candidate - 1;
                }
            }
            Err(
                reason @ (ProgramCompilationError::Capacity { .. }
                | ProgramCompilationError::UnavailableProvider(_)),
            ) => {
                // The baseline already compiled against this same provider catalog.
                // Retain the exact reason for rejecting this optional width.
                fitted
                    .reasons
                    .push(format!("concurrency candidate {candidate}: {reason}"));
                upper = candidate - 1;
            }
            Err(ProgramCompilationError::Invalid(error)) => return Err(error),
        }
        candidate = projected_next.unwrap_or_else(|| lower + (upper - lower).div_ceil(2));
    }
    let selected = best.ok_or_else(|| {
        ferrum_types::FerrumError::resource_exhausted(format!(
            "no verified serving concurrency fits the {budget}-byte budget: {}",
            fitted.reasons.join("; ")
        ))
    })?;
    let resident = fitted.selected.max_sequences;
    if resident != original {
        fitted.reasons.push(format!(
            "concurrent sequences: {original} -> {resident}, verified against the compiled serving plan"
        ));
    }
    fitted.serving_concurrency_search = Some(StartupConcurrencySearch {
        upper_bound,
        decode_wave_sequences: resident.min(engine.batching.max_batch_size),
        candidate_compilations: compilations,
        capture_warmup_peak_bytes: warmup_peak,
    });
    Ok(selected)
}
