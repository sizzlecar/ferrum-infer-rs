//! Conditional finite-offer certificate, not a Source8 execution or a cycle.
use super::*;

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct Bounds {
    minimum: usize,
    maximum: usize,
}

#[derive(Default, Serialize)]
struct Charges {
    requests: usize,
    execution_actions: usize,
    declared_offer_rows: usize,
    serial_token_work: usize,
}
impl Charges {
    fn charge(&mut self, value: work::CaseWork) -> AuditResult<()> {
        self.requests = add(self.requests, value.requests)?;
        self.execution_actions = add(self.execution_actions, value.execution_actions)?;
        self.declared_offer_rows = add(self.declared_offer_rows, value.serial_declared_offer_rows)?;
        self.serial_token_work = add(self.serial_token_work, value.serial_token_work)?;
        Ok(())
    }
    fn accumulate(&mut self, value: &Self) -> AuditResult<()> {
        self.requests = add(self.requests, value.requests)?;
        self.execution_actions = add(self.execution_actions, value.execution_actions)?;
        self.declared_offer_rows = add(self.declared_offer_rows, value.declared_offer_rows)?;
        self.serial_token_work = add(self.serial_token_work, value.serial_token_work)?;
        Ok(())
    }
}

#[derive(Serialize)]
struct Phase {
    phase: usize,
    start_cut: Bounds,
    /// Occurrence ranges are half-open. The whole sequence is frozen in advance.
    first_occurrence: usize,
    floor_fillers_end: usize,
    representatives_end: usize,
    padding_end: usize,
    guaranteed_fresh_cohorts_per_family: Vec<usize>,
    phase_min_offered: usize,
    prefix_after_representatives: Bounds,
    next_cut: Bounds,
    prefix_after_padding: Bounds,
}

#[derive(Default, Serialize)]
struct Flow {
    occurrence_case_indices: Vec<usize>,
    discovery_end: usize,
    discovery_padding_end: usize,
    initial_fit_cut: Bounds,
    phases: Vec<Phase>,
    offered: Bounds,
    work: Charges,
}
impl Flow {
    fn append(
        &mut self,
        index: usize,
        lookup: &impl Fn(usize) -> AuditResult<work::CaseWork>,
    ) -> AuditResult<()> {
        // Re-evaluate the original case work for EVERY fresh occurrence.
        let value = lookup(index)?;
        ensure!(
            value.requests > 0
                && value.declared_offers_minimum > 0
                && value.declared_offers_minimum <= value.declared_offers_upper,
            "finite cohort has invalid work bounds"
        );
        self.offered.minimum = add(self.offered.minimum, value.declared_offers_minimum)?;
        self.offered.maximum = add(self.offered.maximum, value.declared_offers_upper)?;
        self.work.charge(value)?;
        self.occurrence_case_indices.try_reserve(1)?;
        self.occurrence_case_indices.push(index);
        Ok(())
    }
    fn pad(
        &mut self,
        until: usize,
        padding: usize,
        lookup: &impl Fn(usize) -> AuditResult<work::CaseWork>,
    ) -> AuditResult<()> {
        let minimum = lookup(padding)?.declared_offers_minimum;
        ensure!(minimum > 0, "padding has no guaranteed offer progress");
        let count = until.saturating_sub(self.offered.minimum).div_ceil(minimum);
        self.occurrence_case_indices.try_reserve(count)?;
        for _ in 0..count {
            self.append(padding, lookup)?;
        }
        ensure!(
            self.offered.minimum >= until,
            "finite stream ends before latest close"
        );
        Ok(())
    }
}

fn construct(
    representatives: &[Vec<usize>],
    fillers: &[usize],
    padding: usize,
    members: [usize; 3],
    lookup: &impl Fn(usize) -> AuditResult<work::CaseWork>,
) -> AuditResult<Flow> {
    ensure!(
        !representatives.is_empty() && representatives.len() == fillers.len(),
        "finite source family count differs"
    );
    ensure!(members.iter().all(|m| *m > 0), "empty phase floor");
    for (family, original) in representatives.iter().enumerate() {
        ensure!(
            !original.is_empty() && original.contains(&fillers[family]),
            "filler is not an original member of its exact family"
        );
        ensure!(
            original
                .iter()
                .all(|case| representatives[..family].iter().all(|p| !p.contains(case))),
            "finite cohort cannot donate a floor to another family"
        );
    }
    ensure!(
        representatives.iter().any(|p| p.contains(&padding)),
        "padding is not original"
    );
    let mut out = Flow::default();
    for &filler in fillers {
        out.append(filler, lookup)?;
    }
    out.discovery_end = out.occurrence_case_indices.len();
    // A successful discovery is at least offer 1 and no later than its
    // complete guaranteed cohort. Do not borrow its remaining Fit members.
    let mut cut = Bounds {
        minimum: 1,
        maximum: out.offered.maximum,
    };
    out.initial_fit_cut = cut;
    out.pad(cut.maximum, padding, lookup)?;
    out.discovery_padding_end = out.occurrence_case_indices.len();
    for (phase, minimum_members) in members.into_iter().enumerate() {
        ensure!(
            out.offered.minimum >= cut.maximum,
            "phase cohort may precede latest opening"
        );
        let first_occurrence = out.occurrence_case_indices.len();
        let mut guaranteed = Vec::new();
        for (family, original) in representatives.iter().enumerate() {
            let count = minimum_members.saturating_sub(original.len());
            for _ in 0..count {
                out.append(fillers[family], lookup)?;
            }
            guaranteed.push(add(count, original.len())?);
        }
        let floor_fillers_end = out.occurrence_case_indices.len();
        for original in representatives {
            for &case in original {
                out.append(case, lookup)?;
            }
        }
        let representatives_end = out.occurrence_case_indices.len();
        let prefix_after_representatives = out.offered;
        let minimum = out
            .offered
            .maximum
            .checked_sub(cut.minimum)
            .context("finite anchor precedes phase opening")?
            .max(minimum_members);
        let next_cut = Bounds {
            minimum: add(cut.minimum, minimum)?,
            maximum: add(cut.maximum, minimum)?,
        };
        ensure!(
            next_cut.minimum >= out.offered.maximum,
            "phase can freeze before anchors"
        );
        out.pad(next_cut.maximum, padding, lookup)?;
        out.phases.push(Phase {
            phase,
            start_cut: cut,
            first_occurrence,
            floor_fillers_end,
            representatives_end,
            padding_end: out.occurrence_case_indices.len(),
            guaranteed_fresh_cohorts_per_family: guaranteed,
            phase_min_offered: minimum,
            prefix_after_representatives,
            next_cut,
            prefix_after_padding: out.offered,
        });
        cut = next_cut;
    }
    Ok(out)
}

fn padding_precedes(
    index: usize,
    value: work::CaseWork,
    old_index: usize,
    old: work::CaseWork,
) -> AuditResult<bool> {
    let ratio = |a: usize, b: usize| a.checked_mul(b).context("padding ratio overflow");
    let requests = ratio(value.requests, old.declared_offers_minimum)?
        .cmp(&ratio(old.requests, value.declared_offers_minimum)?);
    let actions = ratio(value.execution_actions, old.declared_offers_minimum)?.cmp(&ratio(
        old.execution_actions,
        value.declared_offers_minimum,
    )?);
    Ok((requests, actions, index.cmp(&old_index))
        < (
            std::cmp::Ordering::Equal,
            std::cmp::Ordering::Equal,
            std::cmp::Ordering::Equal,
        ))
}

pub(super) fn evaluate(
    capture: &Capture,
    groups: &[Vec<&PopulationPlan>],
    opportunities: &[CaseOpportunity],
    available: SelectionCapacity,
    source_limit: usize,
) -> AuditResult<Value> {
    let lookup = |index: usize| -> AuditResult<work::CaseWork> {
        let case = capture
            .cases
            .get(index)
            .context("finite case outside capture")?;
        let prompt = *capture
            .prompts
            .get(case.template)
            .context("finite prompt absent")?;
        Ok(work::case_work(
            case,
            prompt,
            capture.chunk,
            capture.row_ceiling,
        )?)
    };
    let mut sources = Vec::new();
    let mut total = Charges::default();
    let mut all_schedules_fit = true;
    for group in groups {
        let mut representatives = Vec::new();
        let mut fillers = Vec::new();
        let mut padding: Option<(usize, work::CaseWork)> = None;
        for member in group {
            // Reuse the exact existing floor checks and filler cost tuple.
            let expanded = expand_cycle(
                &member.key,
                &member.cases,
                add(member.cases.len(), 1)?,
                &capture.cases,
                opportunities,
                &capture.prompts,
                capture.chunk,
                capture.row_ceiling,
            )?;
            fillers.push(*expanded.last().context("finite filler absent")?);
            representatives.push(member.cases.clone());
            for &index in &member.cases {
                let value = lookup(index)?;
                ensure!(value.declared_offers_minimum > 0, "padding minimum is zero");
                if match padding {
                    None => true,
                    Some((old_i, old)) => padding_precedes(index, value, old_i, old)?,
                } {
                    padding = Some((index, value));
                }
            }
        }
        let padding = padding.context("finite padding absent")?.0;
        let original = &group[0].declaration;
        let settings = &original.settings;
        let members = [
            add(settings.max_rank, settings.min_fit_redundancy)?.max(settings.min_phase_samples),
            settings.min_phase_samples,
            settings.min_phase_samples,
        ];
        let flow = construct(&representatives, &fillers, padding, members, &lookup)?;
        let minima = std::array::from_fn(|i| flow.phases[i].phase_min_offered);
        let mut schedule = OwnerBlockScheduleV1::new(1, minima, members)
            .map_err(|reason| error(format!("finite schedule: {reason:?}")))?;
        schedule.prediction_validity = original.schedule.prediction_validity;
        let mut numerical = settings.clone();
        numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
        let validation = schedule.validate(&numerical);
        all_schedules_fit &= validation.is_ok();
        let setup_work = work::setup_for_indices(&capture.cases, &flow.occurrence_case_indices)?;
        let mut setup = Charges::default();
        setup.charge(setup_work)?;
        let mut charges = Charges::default();
        charges.accumulate(&flow.work)?;
        charges.accumulate(&setup)?;
        total.accumulate(&charges)?;
        sources.push(json!({"original_population_indices":group.iter().map(|p|p.original_index).collect::<Vec<_>>(),
            "independent_exact_families":group.iter().map(|p|&p.key).collect::<Vec<_>>(),
            "geometry_representatives":representatives,"filler_case_indices":fillers,"padding_case_index":padding,
            "flow":flow,"schedule":schedule,"derived_max_phase_samples":numerical.max_phase_samples,
            "schedule_within_capacity":validation.is_ok(),"schedule_error":validation.err().map(|e|format!("{e:?}")),
            "setup":setup,"totals":charges}));
    }
    let checks = [
        total.requests <= available.requests,
        total.execution_actions <= available.execution_actions,
        total.declared_offer_rows <= available.declared_offer_rows,
        groups.len() <= source_limit,
        all_schedules_fit,
    ];
    Ok(
        json!({"kind":"onlyconditional_finite_offer_certificate","sources":sources,"totals":total,
        "source_count":groups.len(),"available":{"requests":available.requests,"execution_actions":available.execution_actions,
            "declared_offer_rows":available.declared_offer_rows,"sources":source_limit},
        "fit_checks":{"requests":checks[0],"execution_actions":checks[1],"declared_offer_rows":checks[2],
            "source_count":checks[3],"schedule":checks[4]},"declared_work_and_source_count_fit":checks.into_iter().all(|v|v),
        "limits":["Conditional on original floor-1 cohorts completing and every numerical phase succeeding; no failed-owner restart is financed.",
            "Finite offered-cut certificate only: no periodic fresh_span, SelectedBatch or fictitious cycles are used.",
            "Original raw recipes, live scope closure, unrecorded current branch flags and retained-memory admission remain unverified.",
            "Actual Source8 serialization, independent F/R/Q, cold fallback, expiry/120-second execution and ordinary adoption remain unverified."]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_work(index: usize) -> AuditResult<work::CaseWork> {
        Ok(work::CaseWork {
            declared_offers_minimum: 1,
            declared_offers_upper: if index == 0 { 2 } else { 1 },
            requests: 1,
            execution_actions: 2,
            serial_declared_offer_rows: 2,
            serial_token_work: 2,
        })
    }

    // Exhaust every legal length and nonempty target-member placement for
    // the three variable cohorts. Other-family offers never supply a member.
    fn simulate(flow: &Flow, choices: usize, truncate: bool) -> bool {
        let mut phase = [0usize; 2];
        let mut opening = [None; 2];
        let mut samples = [0usize; 2];
        let mut g = 0;
        let mut choice = choices;
        let end = flow.occurrence_case_indices.len() - usize::from(truncate);
        for (occurrence, &case) in flow.occurrence_case_indices[..end].iter().enumerate() {
            let family = usize::from(case == 2);
            let mask: &[bool] = if case == 0 {
                let selected =
                    [&[true][..], &[true, false], &[false, true], &[true, true]][choice % 4];
                choice /= 4;
                selected
            } else {
                &[true]
            };
            let mut binding = None;
            let mut accepted = None;
            for &member in mask {
                g += 1;
                if member && phase[family] < 3 {
                    if opening[family].is_none() {
                        opening[family] = Some(g); // discovery closes this offer
                    } else {
                        let p = *binding.get_or_insert(phase[family]);
                        if p == phase[family] {
                            samples[family] += 1;
                            accepted = Some(p);
                        }
                    }
                }
                for f in 0..2 {
                    let p = phase[f];
                    if p < 3
                        && opening[f].is_some_and(|s| g - s >= flow.phases[p].phase_min_offered)
                        && samples[f] >= [3, 2, 2][p]
                    {
                        assert!(
                            g >= flow.phases[p].next_cut.minimum
                                && g <= flow.phases[p].next_cut.maximum
                        );
                        phase[f] += 1;
                        opening[f] = Some(g);
                        samples[f] = 0;
                    }
                }
            }
            for p in &flow.phases {
                if (p.floor_fillers_end..p.representatives_end).contains(&occurrence) {
                    assert_eq!(
                        accepted,
                        Some(p.phase),
                        "mandatory representative crossed its intended owner phase"
                    );
                }
            }
        }
        phase == [3, 3]
    }

    #[test]
    fn finite_flow_preserves_anchors_under_variable_offers_and_multi_member_cohorts() {
        let flow = construct(&[vec![0, 1], vec![2]], &[1, 2], 1, [3, 2, 2], &tiny_work).unwrap();
        assert!(flow
            .phases
            .iter()
            .all(|p| p.prefix_after_padding.minimum >= p.next_cut.maximum));
        for choices in 0..64 {
            assert!(simulate(&flow, choices, false));
        }
        assert!(
            !simulate(&flow, 0, true),
            "an upper-bound endpoint alone cannot finish the shortest finite stream"
        );
        assert!(construct(&[vec![0, 1], vec![2]], &[1, 1], 1, [3, 2, 2], &tiny_work).is_err());
        assert!(construct(&[vec![0, 1], vec![1, 2]], &[1, 2], 1, [3, 2, 2], &tiny_work).is_err());
    }

    #[test]
    fn finite_flow_padding_compares_exact_ratios_and_rejects_overflow() {
        let mut a = tiny_work(1).unwrap();
        a.requests = 2;
        a.declared_offers_minimum = 3;
        let mut b = a;
        b.requests = 3;
        b.declared_offers_minimum = 4;
        assert!(padding_precedes(9, a, 1, b).unwrap());
        b = a;
        b.execution_actions += 1;
        assert!(padding_precedes(9, a, 1, b).unwrap());
        assert!(padding_precedes(1, a, 9, a).unwrap());
        b.requests = usize::MAX;
        assert!(padding_precedes(1, a, 9, b).is_err());
    }
}
