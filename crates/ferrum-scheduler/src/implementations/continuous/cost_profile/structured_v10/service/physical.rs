//! Share the full existing recipe/statistics/settlement replay, while retaining
//! the source6 open-frontier and natural-terminal population explicitly.
use super::*;
use ferrum_interfaces::execution_cost::HostTerminalExpectationV1;

#[derive(Default)]
pub(super) struct Frontiers(Vec<((String, u64), Frontier)>);
struct Frontier {
    call_id: u64,
    generation: u64,
    generated: u64,
    maximum: u64,
    kv: u32,
    terminal: bool,
}
pub(super) fn validate(
    header: &StructuredServiceHeaderV6,
    opened_at_ns: u64,
    wave: &StructuredServiceWaveV6,
    frontiers: &mut Frontiers,
) -> Result<(StructuredInputV2, u64, u64), CostProfileError> {
    validate_parts(
        &header.fingerprint,
        header.opening.monotonic_ns,
        header.declaration.nonnegative_envelope.as_ref(),
        opened_at_ns,
        wave.ticket,
        wave.fifo,
        wave.issued_at_ns,
        &wave.host_stages,
        wave.independent.as_ref(),
        frontiers,
    )
}
pub(super) fn original_prepared(
    s: &Stages,
    independent: Option<&IndependentAttentionWaveEvidenceWireV2>,
) -> Result<(Prepared, Vec<OfferedRow>), CostProfileError> {
    let fail = || invalid("original service preparation facts differ");
    let exact = s.actual_shape.as_ref().ok_or_else(fail)?;
    let features = exact.numeric_features.as_ref().ok_or_else(fail)?;
    let settled = s
        .structured_evidence
        .as_ref()
        .and_then(|v| v.as_ref().ok())
        .ok_or_else(fail)?;
    if s.rows.is_empty() || s.rows.len() != features.rows.len() || s.rows.len() > 128 {
        return Err(fail());
    }
    let mut rows = Vec::with_capacity(s.rows.len());
    let mut offered = Vec::with_capacity(s.rows.len());
    for (i, (r, n)) in s.rows.iter().zip(&features.rows).enumerate() {
        let work = match r.actual_work {
            RowWork::Decode { kv_tokens } => PreparedWorkV2::Decode { kv_tokens },
            RowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => PreparedWorkV2::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            _ => return Err(fail()),
        };
        let offered_work = match r.actual_work {
            RowWork::Decode { kv_tokens } => OfferedWork::Decode { kv_tokens },
            RowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => OfferedWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            _ => return Err(fail()),
        };
        let context_before = match work {
            PreparedWorkV2::Decode { kv_tokens } => u64::from(kv_tokens),
            PreparedWorkV2::Prefill { offset, .. } => u64::from(offset),
        };
        let frontier = PreparedRowFactsV2 {
            physical_position: i as u32,
            generated_before: n.generated_tokens_before,
            maximum_output: n.maximum_output_tokens,
            context_before,
            work,
        };
        frontier
            .validate()
            .map_err(|e| numeric_error_at(NumericalReplaySite::PhysicalFrontier, e))?;
        rows.push(PreparedRow {
            request_id: r.request_id.clone(),
            owner_incarnation: r.owner_incarnation,
            work_generation: r.work_generation,
            frontier,
        });
        offered.push(OfferedRow {
            request_id: r.request_id.clone(),
            owner: r.owner_incarnation,
            generation: r.work_generation,
            generated: n.generated_tokens_before,
            work: offered_work,
        });
    }
    // This local adapter is never serialized as a Prepared event/forecast.
    let adapted = Prepared {
        exact: exact.clone(),
        selected: s.statistical_evidence.clone().ok_or_else(fail)?,
        selected_independent_attention_v2: independent.cloned(),
        recipe: settled.recipe.clone(),
        owner_facts: serde_json::Value::Null,
        rows,
    };
    Ok((adapted, offered))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_parts(
    fingerprint: &ProfileFingerprint,
    source_opened: u64,
    envelope: Option<&NonNegativeEnvelopeContractV1>,
    opened_at_ns: u64,
    ticket: u64,
    fifo: u64,
    issued_at_ns: u64,
    s: &Stages,
    independent: Option<&IndependentAttentionWaveEvidenceWireV2>,
    frontiers: &mut Frontiers,
) -> Result<(StructuredInputV2, u64, u64), CostProfileError> {
    let fail = || invalid("source6 original call, frontier or host settlement differs");
    if ticket == 0 || fifo == 0 || s.prepare_started_at_ns != Some(issued_at_ns) {
        return Err(fail());
    }
    let (adapted, offered) = original_prepared(s, independent)?;
    let features = adapted.exact.numeric_features.as_ref().ok_or_else(fail)?;
    let settled = s
        .structured_evidence
        .as_ref()
        .and_then(|v| v.as_ref().ok())
        .ok_or_else(fail)?;
    let input = match envelope {
        Some(contract) => prepared::project_service_actual_with_domain(
            &adapted,
            &offered,
            &contract.workload_domain,
        )?
        .with_cost_template_policy(contract.template_policy)
        .map_err(|e| numeric_error_at(NumericalReplaySite::PhysicalTemplatePolicy, e))?,
        None => prepared::project_service_actual(&adapted, &offered)?,
    };
    let (wall, observed) = observation::validate_service_actual(
        fingerprint,
        source_opened,
        opened_at_ns,
        &adapted,
        s,
        independent,
        settled.stage_binding,
    )?;
    let input = if envelope.is_some() {
        input.with_settled_terminal_causes(
            &s.rows
                .iter()
                .enumerate()
                .filter_map(|(p, row)| {
                    row.terminal
                        .as_ref()
                        .map(|terminal| (p as u32, terminal.finish_reason.clone()))
                })
                .collect::<Vec<_>>(),
        )
    } else {
        input.with_settled_completion(
            &s.rows
                .iter()
                .enumerate()
                .filter_map(|(p, row)| row.terminal.as_ref().map(|_| p as u32))
                .collect::<Vec<_>>(),
        )
    }
    .map_err(|e| numeric_error_at(NumericalReplaySite::PhysicalSettledTerminal, e))?;
    for ((r, n), host) in s
        .rows
        .iter()
        .zip(&features.rows)
        .zip(input.physical_host_rows())
    {
        let (start, end) = match r.actual_work {
            RowWork::Decode { kv_tokens } => {
                (kv_tokens, kv_tokens.checked_add(1).ok_or_else(fail)?)
            }
            RowWork::Prefill { offset, count, .. } => {
                (offset, offset.checked_add(count).ok_or_else(fail)?)
            }
            _ => return Err(fail()),
        };
        let emits = host.terminal_expectation != HostTerminalExpectationV1::NoTokenProduced;
        frontiers.advance(
            s.call_id,
            &r.request_id,
            r.owner_incarnation,
            r.work_generation,
            n.generated_tokens_before,
            n.maximum_output_tokens,
            start,
            end,
            emits,
            r.terminal.is_some(),
        )?;
    }
    let input = match envelope {
        Some(contract) => contract
            .project_input(input)
            .map_err(|e| numeric_error_at(NumericalReplaySite::PhysicalEnvelopeProjection, e))?,
        None => input,
    };
    Ok((input, wall, observed))
}

impl Frontiers {
    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .0
            .capacity()
            .checked_mul(std::mem::size_of::<((String, u64), Frontier)>())?;
        for ((request, _), _) in &self.0 {
            bytes = bytes.checked_add(request.capacity())?;
        }
        Some(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn advance(
        &mut self,
        call_id: u64,
        request: &str,
        owner: u64,
        generation: u64,
        generated: u64,
        maximum: u64,
        start: u32,
        end: u32,
        emits: bool,
        terminal: bool,
    ) -> Result<(), CostProfileError> {
        let fail = || invalid("source6 original call frontier differs");
        let position = self.0.binary_search_by(|((id, incarnation), _)| {
            (id.as_str(), *incarnation).cmp(&(request, owner))
        });
        if let Ok(index) = position {
            let prior = &self.0[index].1;
            if prior.terminal
                || prior.generation.checked_add(1) != Some(generation)
                || prior.generated != generated
                || prior.maximum != maximum
                || prior.kv != start
            {
                // Preserve the original rejection and frontier. These fixed-size
                // facts identify which transition failed without retaining a
                // request body or reconstructing any execution authority.
                tracing::warn!(
                    target: "ferrum_scheduler::structured_owner_diagnostics",
                    event = "structured_original_frontier_mismatch_v1",
                    prior_call_id = prior.call_id,
                    actual_call_id = call_id,
                    owner_incarnation = owner,
                    prior_terminal = prior.terminal,
                    expected_generation = ?prior.generation.checked_add(1),
                    actual_generation = generation,
                    expected_generated = prior.generated,
                    actual_generated = generated,
                    expected_maximum = prior.maximum,
                    actual_maximum = maximum,
                    expected_kv = prior.kv,
                    actual_kv = start,
                    completed_kv = end,
                    emits_token = emits,
                    terminal,
                    "Original service frontier continuity rejected"
                );
                return Err(fail());
            }
        }
        let next = Frontier {
            call_id,
            generation,
            generated: generated.checked_add(u64::from(emits)).ok_or_else(fail)?,
            maximum,
            kv: end,
            terminal,
        };
        match position {
            Ok(index) => self.0[index].1 = next,
            Err(index) => self.0.insert(index, ((request.to_owned(), owner), next)),
        }
        Ok(())
    }
}
