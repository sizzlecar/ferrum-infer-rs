use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};

struct InputCase {
    pending: Vec<bool>,
    forced: Vec<bool>,
    length: Vec<bool>,
    product: CostProductOutput,
    algorithm: &'static str,
    kv: u32,
    units: u64,
    domain: HostContentDomainV1,
}
impl Default for InputCase {
    fn default() -> Self {
        Self {
            pending: vec![false; 3],
            forced: vec![true, false, false],
            length: vec![false; 3],
            product: CostProductOutput::FullLogits,
            algorithm: "fixture.discovery",
            kv: 64,
            units: 32,
            domain: HostContentDomainV1::PlainTextGreedyV1,
        }
    }
}
impl InputCase {
    fn input(&self) -> StructuredInputV2 {
        let wave = self.prepared();
        let selected = wave.statistical.as_ref().unwrap();
        StructuredInputV2::from_actual(
            &wave.exact,
            selected,
            selected.structured_capture().unwrap().unwrap(),
        )
        .unwrap()
        .with_settled_completion(
            &self
                .length
                .iter()
                .enumerate()
                .filter_map(|(p, terminal)| terminal.then_some(p as u32))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    fn prepared(&self) -> CanonicalStatisticalWave {
        let rows = self.pending.len() as u32;
        assert_eq!(self.pending.len(), self.forced.len());
        assert_eq!(self.pending.len(), self.length.len());
        let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
        selected
            .kernel(
                SelectedAlgorithmClassV1::new(self.algorithm, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: self.units,
                    padded_units: self.units,
                    inner_units_per_logical_unit: 2,
                    grid: [1, 1, 1],
                    scratch_bytes: 64,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
        let selected = selected.finish().unwrap();
        let mut builder = CanonicalWaveCostBuilder::new_with_structured_statistics(0, self.product);
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id: "fixture.discovery",
                command_index: 0,
                node_index: None,
                command_phase: DeviceCommandPhase::Compute,
                provider: None,
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: rows,
                token_count: u64::from(rows),
                batching_form: "packed",
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&selected),
            })
            .unwrap();
        builder
            .core_readback_route(CoreReadbackRoute::HostSynchronized)
            .unwrap();
        for position in 0..rows as usize {
            builder
                .row(CanonicalCostRow {
                    work: ActualRowWork::Decode { kv_tokens: self.kv },
                    output: CostRowOutput::Decode {
                        requires_full_logits: self.forced[position] || self.pending[position],
                        repetition_tokens: 2,
                        repetition_penalty_bits: 1f32.to_bits(),
                    },
                    host_policy_signature: [3; 32],
                    mask_upload_required: false,
                    host_features: Some(HostCostFeaturesV1 {
                        policy: HostCostPolicyV2 {
                            empirical_content_domain: Some(self.domain),
                            categorical_signature: [4; 32],
                            decoder_text_bytes_per_token: 4,
                            decoder_scratch_bytes_per_token: 8,
                            raw_token_bytes_bound: 4,
                        },
                        state: HostCostStateV1 {
                            generated_tokens_before: 2,
                            maximum_output_tokens: if self.length[position] { 3 } else { 20 },
                            sampling_history_tokens: 2,
                            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                            pending_decoded_utf8: self.pending[position],
                            completion_state_signature: satisfied_completion_cost_signature(),
                        },
                    }),
                })
                .unwrap();
        }
        builder
            .finish_with_captured_structure(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                64,
            )
            .unwrap()
    }
}

fn discover(inputs: &[StructuredInputV2]) -> FrozenDiscovery {
    let mut window = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: inputs.len(),
            maximum_owners: 128,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        20,
    )
    .unwrap();
    for (index, input) in inputs.iter().enumerate() {
        window
            .observe(index as u64 + 1, 23 + index as u64 * 3, input)
            .unwrap();
    }
    window.freeze().unwrap()
}

#[test]
fn discovery_requires_real_full_empty_and_intermediate_challenges_and_keeps_joint_facts() {
    let inputs = [
        InputCase::default().input(),
        InputCase {
            pending: vec![false, true, true],
            length: vec![false, false, true],
            ..Default::default()
        }
        .input(),
        InputCase {
            pending: vec![false, true, false],
            ..Default::default()
        }
        .input(),
    ];
    let frozen = discover(&inputs);
    let coverage = &frozen.scopes()[0].coverage;
    assert_eq!(coverage.pending_eligible_positions, [1, 2]);
    assert_eq!(
        coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::AnySubset]
    );
    assert_eq!(coverage.pending_counts, [0, 1, 2]);
    assert_eq!(coverage.length_counts, [0, 1]);
    assert_eq!(coverage.length_positions, [2]);
    assert_eq!(coverage.joint_counts, [(0, 0), (1, 0), (2, 1)]);
    assert!(!coverage.joint_counts.contains(&(1, 1)));
}

#[test]
fn missing_full_or_intermediate_stays_exact_without_widening_constraints() {
    let empty = InputCase::default().input();
    let first = InputCase {
        pending: vec![false, true, false],
        ..Default::default()
    }
    .input();
    let second = InputCase {
        pending: vec![false, false, true],
        ..Default::default()
    }
    .input();
    let full = InputCase {
        pending: vec![false, true, true],
        ..Default::default()
    }
    .input();
    for inputs in [vec![empty.clone(), first, second], vec![empty, full]] {
        let frozen = discover(&inputs);
        let coverage = &frozen.scopes()[0].coverage;
        assert_eq!(coverage.pending_eligible_positions, [1, 2]);
        assert!(coverage.authorized_pending_constraints.is_empty());
    }
}

#[test]
fn pending_forced_full_route_only_authorizes_nonempty_after_actual_challenges() {
    let inputs = [
        InputCase {
            forced: vec![false; 3],
            pending: vec![true; 3],
            ..Default::default()
        }
        .input(),
        InputCase {
            forced: vec![false; 3],
            pending: vec![true, false, false],
            ..Default::default()
        }
        .input(),
    ];
    let frozen = discover(&inputs);
    let coverage = &frozen.scopes()[0].coverage;
    assert_eq!(coverage.pending_eligible_positions, [0, 1, 2]);
    assert_eq!(
        coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::NonEmptySubset]
    );
    assert_eq!(coverage.pending_counts, [1, 3]);
    // No zero or unobserved count-two population is fabricated. Query-time
    // scope checks will still reject a request that needs the missing count.
    assert_eq!(coverage.joint_counts, [(1, 0), (3, 0)]);
}

#[test]
fn greedy_conditional_empty_never_acquires_unobserved_pending_positions() {
    let input = InputCase {
        forced: vec![false; 3],
        product: CostProductOutput::GreedyToken,
        ..Default::default()
    }
    .input();
    let frozen = discover(&[input]);
    let coverage = &frozen.scopes()[0].coverage;
    assert!(coverage.pending_eligible_positions.is_empty());
    assert!(coverage.pending_positions.is_empty());
    assert_eq!(
        coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::AnySubset]
    );
    assert_eq!(coverage.pending_counts, [0]);
}

#[test]
fn discovery_does_not_select_owners_or_scopes_using_numeric_cost_axes() {
    let a = InputCase::default().input();
    let b = InputCase {
        kv: 128,
        units: 256,
        ..Default::default()
    }
    .input();
    assert_eq!(a.owner(), b.owner());
    assert_ne!(a.regression_axes(), b.regression_axes());
    assert_eq!(discover(&[a]).scopes(), discover(&[b]).scopes());
    // No latency, prediction, fit result, or qualification input is accepted
    // by DiscoveryWindow::observe.
}

#[test]
fn owner_capacity_failure_cannot_drop_a_member_and_resume_the_same_window() {
    let a = InputCase::default().input();
    let b = InputCase {
        algorithm: "fixture.discovery.other",
        ..Default::default()
    }
    .input();
    let mut window = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 2,
            maximum_owners: 1,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        0,
    )
    .unwrap();
    window.observe(1, 1, &a).unwrap();
    assert_eq!(window.observe(2, 2, &b), Err(DiscoveryError::OwnerCapacity));
    assert_eq!(window.observe(2, 3, &a), Err(DiscoveryError::OwnerCapacity));
    assert_eq!(window.freeze().err(), Some(DiscoveryError::OwnerCapacity));
}

#[test]
fn discovery_retained_capacity_includes_the_frozen_scope_and_cannot_be_retried() {
    let input = InputCase::default().input();
    let capacity = std::mem::size_of::<DiscoveryWindow>() + owner_charge() + state_charge();
    let mut too_small = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 1,
            maximum_retained_bytes: capacity - 1,
            maximum_owners: 128,
        },
        0,
    )
    .unwrap();
    assert_eq!(
        too_small.observe(1, 1, &input),
        Err(DiscoveryError::RetainedCapacity)
    );
    assert_eq!(
        too_small.freeze().err(),
        Some(DiscoveryError::RetainedCapacity)
    );
    let mut fits = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 2,
            maximum_retained_bytes: capacity,
            maximum_owners: 128,
        },
        0,
    )
    .unwrap();
    fits.observe(1, 1, &input).unwrap();
    fits.observe(2, 2, &input).unwrap();
    assert_eq!(fits.retained_bytes_upper_bound(), capacity);
    assert_eq!(fits.freeze().unwrap().offered(), 2);
}

#[test]
fn frozen_discovery_requires_new_fifo_samples_and_new_owners_enter_a_new_generation() {
    let a = InputCase::default().input();
    let b = InputCase {
        algorithm: "fixture.discovery.next",
        ..Default::default()
    }
    .input();
    let mut active = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 1,
            maximum_owners: 128,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        20,
    )
    .unwrap();
    active.observe(1, 23, &a).unwrap();
    assert_eq!(active.observe(2, 24, &b), Err(DiscoveryError::WindowClosed));
    let first = active.freeze().unwrap();
    let original = first.scopes().to_vec();
    assert_eq!(first.fifo_bounds(), (20, 23));
    assert!(!first.is_after_discovery(23));
    assert!(first.is_after_discovery(24));
    assert!(!first.contains_owner(&b));
    let mut next = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 1,
            maximum_owners: 128,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        23,
    )
    .unwrap();
    assert_eq!(
        next.observe(1, 23, &b),
        Err(DiscoveryError::TicketOrFifoDiscontinuity)
    );
    assert_eq!(
        next.freeze().err(),
        Some(DiscoveryError::TicketOrFifoDiscontinuity)
    );
    let second = discover(&[b]);
    assert!(!second.contains_owner(&a));
    assert_eq!(first.scopes(), original);
}

#[test]
fn missing_discovery_ticket_cannot_be_hidden_by_a_contiguous_accepted_fifo() {
    let input = InputCase::default().input();
    let mut window = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 2,
            maximum_owners: 128,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        0,
    )
    .unwrap();
    window.observe(1, 1, &input).unwrap();
    assert_eq!(
        window.observe(3, 2, &input),
        Err(DiscoveryError::TicketOrFifoDiscontinuity)
    );
    assert_eq!(
        window.freeze().err(),
        Some(DiscoveryError::TicketOrFifoDiscontinuity)
    );
    let unfinished = DiscoveryWindow::new(
        DiscoveryPolicy {
            offered_waves: 1,
            maximum_owners: 128,
            maximum_retained_bytes: 8 * 1024 * 1024,
        },
        0,
    )
    .unwrap();
    assert_eq!(
        unfinished.freeze().err(),
        Some(DiscoveryError::IncompletePopulation)
    );
}

#[test]
fn maximum_physical_position_is_preserved_without_bitmap_wraparound() {
    let mut case = InputCase {
        pending: vec![false; 128],
        forced: vec![false; 128],
        length: vec![false; 128],
        ..Default::default()
    };
    case.forced[0] = true;
    let empty = case.input();
    case.pending[127] = true;
    case.length[127] = true;
    let frozen = discover(&[empty, case.input()]);
    let coverage = &frozen.scopes()[0].coverage;
    assert_eq!(coverage.pending_positions, [127]);
    assert_eq!(coverage.pending_eligible_positions, [127]);
    assert_eq!(coverage.length_positions, [127]);
    assert_eq!(
        coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::AnySubset]
    );
}

fn installed_case(sampling: PlainTextSamplingRouteV2, pending: u8) -> InputCase {
    InputCase {
        domain: HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
            sampling,
            model_eos: true,
            user_stop: true,
        }),
        pending: (0..3).map(|p| pending & (1 << p) != 0).collect(),
        forced: vec![sampling == PlainTextSamplingRouteV2::FullLogits; 3],
        product: if sampling == PlainTextSamplingRouteV2::FullLogits || pending != 0 {
            CostProductOutput::FullLogits
        } else {
            CostProductOutput::GreedyToken
        },
        ..Default::default()
    }
}

// Continue the producer -> actual numeric input -> independent discovery ->
// bound future recipe pipeline, rather than directly constructing coverage.
fn check_installed_forecast(
    anchor: &InputCase,
    scope: &StructuredScopeV2,
    constraint: HostPendingConstraintV2,
) {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredQueryV2;
    let wave = anchor.prepared();
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let pending = HostPendingSetV2::new_installed(
        &wave.exact,
        recipe,
        &scope.coverage.pending_eligible_positions,
        constraint,
        None,
    )
    .unwrap();
    let query = StructuredQueryV2::from_future(
        &wave.exact,
        selected,
        recipe,
        &HostContentForecastV2::Unresolved(pending),
    )
    .unwrap();
    let demand = query.required_coverage().unwrap();
    assert_eq!(query.owner(), &scope.owner);
    assert_eq!(demand.pending_constraint, Some(constraint));
    assert!(scope
        .coverage
        .authorized_pending_constraints
        .contains(&constraint));
    assert_eq!(
        demand.eligible_pending_positions,
        scope.coverage.pending_eligible_positions
    );
    assert!(demand
        .reachable_joint_counts
        .iter()
        .all(|counts| scope.coverage.joint_counts.contains(counts)));
}

#[test]
fn installed_greedy_discovery_authorizes_the_conditional_empty_future_branch() {
    let case = installed_case(
        PlainTextSamplingRouteV2::Greedy {
            repetition_penalty: false,
        },
        0,
    );
    let frozen = discover(&[case.input()]);
    let scope = &frozen.scopes()[0];
    assert!(scope.coverage.pending_eligible_positions.is_empty());
    assert_eq!(
        scope.coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::AnySubset]
    );
    check_installed_forecast(&case, scope, HostPendingConstraintV2::AnySubset);
}

#[test]
fn installed_full_logits_discovery_keeps_route_with_all_pending_bits_unresolved() {
    let cases = [0, 1, 3, 7].map(|bits| installed_case(PlainTextSamplingRouteV2::FullLogits, bits));
    let frozen = discover(&cases.iter().map(InputCase::input).collect::<Vec<_>>());
    let scope = &frozen.scopes()[0];
    assert_eq!(scope.coverage.pending_eligible_positions, [0, 1, 2]);
    assert_eq!(
        scope.coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::AnySubset]
    );
    check_installed_forecast(&cases[0], scope, HostPendingConstraintV2::AnySubset);
    // Every position has been seen pending in these populations; none can
    // disappear merely to make an incomplete challenge set pass.
    for missing in [vec![0, 1, 2, 4], vec![0, 7], vec![1, 3, 7]] {
        let inputs = missing
            .into_iter()
            .map(|bits| installed_case(PlainTextSamplingRouteV2::FullLogits, bits).input())
            .collect::<Vec<_>>();
        let frozen = discover(&inputs);
        assert_eq!(
            frozen.scopes()[0].coverage.pending_eligible_positions,
            [0, 1, 2]
        );
        assert!(frozen.scopes()[0]
            .coverage
            .authorized_pending_constraints
            .is_empty());
    }
}

#[test]
fn installed_greedy_pending_full_logits_requires_the_nonempty_branch() {
    let cases = [1, 3, 7].map(|bits| {
        installed_case(
            PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: false,
            },
            bits,
        )
    });
    let frozen = discover(&cases.iter().map(InputCase::input).collect::<Vec<_>>());
    let scope = &frozen.scopes()[0];
    assert_eq!(scope.coverage.pending_eligible_positions, [0, 1, 2]);
    assert_eq!(
        scope.coverage.authorized_pending_constraints,
        [HostPendingConstraintV2::NonEmptySubset]
    );
    check_installed_forecast(&cases[0], scope, HostPendingConstraintV2::NonEmptySubset);
}
