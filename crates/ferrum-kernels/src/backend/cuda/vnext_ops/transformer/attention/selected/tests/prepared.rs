use super::*;
use ferrum_interfaces::vnext::*;

fn rows(counts: &[u64]) -> Vec<OperationCostWorkRow> {
    counts
        .iter()
        .map(|&count| OperationCostWorkRow {
            offset: 0,
            count: std::num::NonZeroU64::new(count).unwrap(),
            full_input_tokens: std::num::NonZeroU64::new(count).unwrap(),
        })
        .collect()
}

fn legacy(
    precision: AttentionPrecision,
    projection: AttentionProjection,
    evidence: ProjectionEvidence<'_>,
    rows: &[OperationCostWorkRow],
    packed: bool,
    classes: &PreparedKernelClasses,
) -> SelectedCommandCostEvidenceV1 {
    let tokens = rows.iter().map(|row| row.count.get()).sum();
    let leaves = std::iter::once((tokens, rows.len() as u32, true))
        .take(usize::from(packed))
        .chain(
            rows.iter()
                .filter(|_| !packed)
                .map(|row| (row.count.get(), 1, false)),
        );
    compute_with_classes(
        shape(),
        precision,
        projection,
        evidence,
        leaves,
        tokens,
        rows.len(),
        true,
        SloStructuredCostCapture::HostSettledV1,
        Some(classes),
    )
    .unwrap()
}

// Declaration-only view. The real resource capture and execution guards remain
// in the interfaces/core; this fixture grants no allocation or replay authority.
struct TopologyView {
    operation: OperationId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    values: Vec<ResolvedValueBinding>,
    rows: Vec<OperationCostWorkRow>,
    scratch: ResourceId,
    binding: Option<ResourceId>,
    unavailable: Option<ResourceId>,
}
impl TopologyView {
    fn new(precision: AttentionPrecision, rows: &[OperationCostWorkRow]) -> Self {
        let values = (0..10)
            .map(|ordinal| (ResolvedValueRole::Input, ordinal))
            .chain(std::iter::once((ResolvedValueRole::Output, 0)))
            .map(|(role, ordinal)| {
                let name = format!("gdn.{role:?}.{ordinal}");
                ResolvedValueBinding::new(
                    ProgramValueId::new(name.clone()).unwrap(),
                    role,
                    ordinal,
                    ResolvedTensorSpec::new(
                        vec![1],
                        ElementType::F32,
                        ResolvedTensorLayout::Contiguous,
                    )
                    .unwrap(),
                    if role == ResolvedValueRole::Output {
                        TensorAccess::Write
                    } else {
                        TensorAccess::Read
                    },
                    AliasPolicy::NoAlias,
                    if role == ResolvedValueRole::Input && ordinal >= 8 {
                        BufferUsage::State
                    } else {
                        BufferUsage::Activations
                    },
                    None,
                    ResolvedValueStorage::single(
                        ResourceId::new(name).unwrap(),
                        0,
                        4,
                        ElementType::F32,
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        Self {
            operation: OperationId::new(precision.operation()).unwrap(),
            attributes: BTreeMap::new(),
            values,
            rows: rows.to_vec(),
            scratch: ResourceId::new("gdn.scratch").unwrap(),
            binding: Some(ResourceId::new("gdn.binding").unwrap()),
            unavailable: None,
        }
    }
}
impl ReusableExecutionTopologyView for TopologyView {
    fn operation_id(&self) -> &OperationId {
        &self.operation
    }
    fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        &self.attributes
    }
    fn bindings(&self) -> &[ResolvedValueBinding] {
        &self.values
    }
    fn memory_plan(&self) -> &MemoryPlan {
        panic!("no physical memory plan in numerical fixture")
    }
    fn participant_count(&self) -> usize {
        self.rows.len()
    }
    fn immediate_tokens(&self) -> u64 {
        self.rows.iter().map(|row| row.count.get()).sum()
    }
    fn token_row(&self, index: usize) -> Option<OperationCostWorkRow> {
        self.rows.get(index).copied()
    }
    fn workspace_resource(
        &self,
        workspace: ReusableExecutionWorkspaceAddress,
    ) -> Option<&ResourceId> {
        match workspace {
            ReusableExecutionWorkspaceAddress::Scratch => Some(&self.scratch),
            ReusableExecutionWorkspaceAddress::Binding => self.binding.as_ref(),
            ReusableExecutionWorkspaceAddress::Persistent => None,
        }
    }
    fn resource_reusable_address_scope(
        &self,
        resource: &ResourceId,
    ) -> Result<Option<DeviceReusableAddressScope>, VNextError> {
        Ok((self.unavailable.as_ref() != Some(resource))
            .then_some(DeviceReusableAddressScope::Plan))
    }
}

#[test]
fn prepared_gdn_query_matches_legacy_math_replay_and_topology_for_all_projection_forms() {
    for precision in [
        AttentionPrecision::F32Master,
        AttentionPrecision::F32MasterQ8Projections,
        AttentionPrecision::F32MasterGgufF16Projections,
    ] {
        let quantized = precision.quantizes_projections();
        let (input, output) = matrices(quantized);
        let library = matches!(precision, AttentionPrecision::F32MasterGgufF16Projections);
        let projection = if library {
            AttentionProjection::F16
        } else {
            projection(&input, quantized)
        };
        let evidence = if library {
            ProjectionEvidence::Library(CublasHandleApiIdentity::fixture_identity())
        } else {
            ProjectionEvidence::Native {
                input: &input,
                output: &output,
            }
        };
        let template = PreparedCostTemplate::new(shape(), precision, projection).unwrap();
        let classes = PreparedKernelClasses::new(shape(), precision).unwrap();
        for counts in [
            vec![1],
            vec![7],
            vec![1; 8],
            vec![3, 5],
            vec![1, 7, 3],
            vec![super::super::super::super::native_matrix::MAX_ROWS + 8, 3],
        ] {
            let rows = rows(&counts);
            let tokens = counts.iter().sum();
            for packed in [false, true] {
                if packed && rows.len() == 1 {
                    continue;
                }
                let query = template
                    .query(
                        &rows,
                        tokens,
                        packed,
                        GatedDeltaExecutionCapabilities::recurrent_only(),
                        Some(evidence),
                        true,
                    )
                    .unwrap()
                    .unwrap();
                let selected = query
                    .compute(SloStructuredCostCapture::HostSettledV1)
                    .unwrap();
                let old = legacy(precision, projection, evidence, &rows, packed, &classes);
                assert_eq!(selected, old);
                assert_eq!(selected.algorithm_work(), old.algorithm_work());
                selected
                    .validate_command(tokens, query.dispatches(), query.transfers())
                    .unwrap();
                SelectedReplayAlgorithmTemplateV1::from_selected(
                    &old,
                    tokens,
                    query.dispatches(),
                    query.transfers(),
                )
                .unwrap()
                .validate_binding(&selected)
                .unwrap();
                let view = TopologyView::new(precision, &rows);
                let old_topology = reusable_attention_topology_for_tokens(
                    shape(),
                    GatedDeltaExecutionCapabilities::recurrent_only(),
                    rows.len(),
                    counts.iter().copied(),
                )
                .unwrap();
                assert_eq!(
                    query.topology(&view).unwrap(),
                    ReusableExecutionTopology::Dynamic(old_topology)
                );
                let slots = StateBindingLayout::new(rows.len()).unwrap();
                for index in 0..rows.len() {
                    assert_eq!(
                        query.binding_offset(index).unwrap(),
                        slots.offset(index).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn prepared_gdn_query_requires_current_library_and_keeps_dynamic_rows_and_replay_parameters() {
    let precision = AttentionPrecision::F32MasterGgufF16Projections;
    let prepared = PreparedCostTemplate::new(shape(), precision, AttentionProjection::F16).unwrap();
    let rows = rows(&[3, 5]);
    let capabilities = GatedDeltaExecutionCapabilities::recurrent_only();
    assert!(prepared
        .query(&rows, 8, false, capabilities, None, true)
        .unwrap()
        .is_none());
    let identity = CublasHandleApiIdentity::fixture_identity();
    let changed = CublasHandleApiIdentity::fixture_identity_for_version(130000);
    let query = prepared
        .query(
            &rows,
            8,
            false,
            capabilities,
            Some(ProjectionEvidence::Library(identity)),
            true,
        )
        .unwrap()
        .unwrap();
    let original = query
        .compute(SloStructuredCostCapture::HostSettledV1)
        .unwrap();
    let replay = SelectedReplayAlgorithmTemplateV1::from_selected(
        &original,
        8,
        query.dispatches(),
        query.transfers(),
    )
    .unwrap();
    let fresh = prepared
        .query(
            &rows,
            8,
            false,
            capabilities,
            Some(ProjectionEvidence::Library(changed)),
            true,
        )
        .unwrap()
        .unwrap();
    assert!(replay
        .validate_binding(
            &fresh
                .compute(SloStructuredCostCapture::HostSettledV1)
                .unwrap()
        )
        .is_err());
    assert!(query.compute(SloStructuredCostCapture::Disabled).is_none());
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(8);
    assert!(query
        .append_library(
            &mut builder,
            0,
            changed,
            3,
            shape().qkvzba_features,
            shape().hidden_size
        )
        .is_none());
    assert!(query
        .append_library(
            &mut builder,
            0,
            identity,
            3,
            shape().qkvzba_features + 1,
            shape().hidden_size
        )
        .is_none());
    assert!(query
        .append_library(
            &mut builder,
            1,
            identity,
            3,
            shape().hidden_size,
            shape().value_features + 1
        )
        .is_none());
    let other_rows = self::rows(&[4, 4]);
    let dynamic = prepared
        .query(
            &other_rows,
            8,
            false,
            capabilities,
            Some(ProjectionEvidence::Library(identity)),
            true,
        )
        .unwrap()
        .unwrap()
        .compute(SloStructuredCostCapture::HostSettledV1)
        .unwrap();
    assert!(replay.validate_binding(&dynamic).is_err());
    // Equal aggregate token work is allowed; row-specific replay parameters
    // must still distinguish the two execution sequences.
    let mut changed_shape = shape();
    changed_shape.epsilon *= 2.0;
    let changed_template =
        PreparedCostTemplate::new(changed_shape, precision, AttentionProjection::F16).unwrap();
    let dynamic = changed_template
        .query(
            &rows,
            8,
            false,
            capabilities,
            Some(ProjectionEvidence::Library(identity)),
            true,
        )
        .unwrap()
        .unwrap()
        .compute(SloStructuredCostCapture::HostSettledV1)
        .unwrap();
    assert!(replay.validate_binding(&dynamic).is_err());
}

#[test]
fn prepared_gdn_topology_retains_value_state_workspace_and_query_guards() {
    let precision = AttentionPrecision::F32MasterGgufF16Projections;
    let prepared = PreparedCostTemplate::new(shape(), precision, AttentionProjection::F16).unwrap();
    let rows = rows(&[3, 5]);
    let query = prepared
        .query(
            &rows,
            8,
            false,
            GatedDeltaExecutionCapabilities::recurrent_only(),
            Some(ProjectionEvidence::Library(
                CublasHandleApiIdentity::fixture_identity(),
            )),
            true,
        )
        .unwrap()
        .unwrap();
    let mut view = TopologyView::new(precision, &rows);
    let original = query.topology(&view).unwrap();
    for resource in (0..8)
        .map(|i| ResourceId::new(format!("gdn.Input.{i}")).unwrap())
        .chain([
            ResourceId::new("gdn.Output.0").unwrap(),
            view.scratch.clone(),
            view.binding.clone().unwrap(),
        ])
    {
        view.unavailable = Some(resource);
        assert_eq!(
            query.topology(&view).unwrap(),
            ReusableExecutionTopology::EagerBoundary
        );
    }
    // State addresses are supplied through the required binding workspace.
    // They remain program-bound, exactly as on the original provider path.
    for ordinal in [8, 9] {
        view.unavailable = Some(ResourceId::new(format!("gdn.Input.{ordinal}")).unwrap());
        assert_eq!(query.topology(&view).unwrap(), original);
    }
    view.unavailable = None;
    view.binding = None;
    assert!(query.topology(&view).is_err());
    view.binding = Some(ResourceId::new("gdn.binding").unwrap());
    view.rows[0].offset += 1;
    assert!(query.topology(&view).is_err());
    view.rows[0].offset -= 1;
    let state = view.values.remove(9);
    assert!(query.topology(&view).is_err());
    view.values.insert(9, state);
    view.operation = OperationId::new(AttentionPrecision::F32Master.operation()).unwrap();
    assert!(query.topology(&view).is_err());
}

#[test]
fn prepared_gdn_query_rejects_invalid_population_and_uninstalled_forms() {
    let prepared = PreparedCostTemplate::new(
        shape(),
        AttentionPrecision::F32MasterGgufF16Projections,
        AttentionProjection::F16,
    )
    .unwrap();
    let rows = rows(&[3, 5]);
    let evidence = Some(ProjectionEvidence::Library(
        CublasHandleApiIdentity::fixture_identity(),
    ));
    let capabilities = GatedDeltaExecutionCapabilities::with_chunked_scan(64).unwrap();
    assert!(prepared
        .query(&[], 0, false, capabilities, evidence, true)
        .is_err());
    assert!(prepared
        .query(&rows[..1], 3, true, capabilities, evidence, true)
        .is_err());
    assert!(prepared
        .query(&rows, 7, false, capabilities, evidence, true)
        .is_err());
    assert!(prepared
        .query(&rows, u64::MAX, false, capabilities, evidence, true)
        .is_err());
    let (input, output) = matrices(false);
    assert!(prepared
        .query(
            &rows,
            8,
            false,
            capabilities,
            Some(ProjectionEvidence::Native {
                input: &input,
                output: &output
            }),
            true
        )
        .unwrap()
        .is_none());
    // The installed provider explicitly selects RecurrentScan. Chunk support
    // cannot silently change that preference or introduce a mixed batch form.
    let query = prepared
        .query(&rows, 8, true, capabilities, evidence, true)
        .unwrap()
        .unwrap();
    let recurrent = prepared
        .query(
            &rows,
            8,
            true,
            GatedDeltaExecutionCapabilities::recurrent_only(),
            evidence,
            true,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        query.compute(SloStructuredCostCapture::HostSettledV1),
        recurrent.compute(SloStructuredCostCapture::HostSettledV1)
    );
    let mut digest = GdnTopologyDigest::new(2).unwrap();
    digest
        .push(
            3,
            capabilities
                .select(3, GatedDeltaExecutionPreference::RecurrentScan)
                .unwrap(),
        )
        .unwrap();
    assert!(digest
        .push(
            5,
            capabilities
                .select(5, GatedDeltaExecutionPreference::ChunkedScan)
                .unwrap()
        )
        .is_err());
}

#[test]
#[ignore = "CPU timing diagnostic; release CUDA build, no GPU context"]
fn prepared_gdn_cpu_template_and_selected_projection_cost() {
    use std::{hint::black_box, time::Instant};
    const ITERATIONS: usize = 512;
    for precision in [
        AttentionPrecision::F32Master,
        AttentionPrecision::F32MasterQ8Projections,
        AttentionPrecision::F32MasterGgufF16Projections,
    ] {
        let quantized = precision.quantizes_projections();
        let library = matches!(precision, AttentionPrecision::F32MasterGgufF16Projections);
        let (input, output) = matrices(quantized);
        let projection = if library {
            AttentionProjection::F16
        } else {
            projection(&input, quantized)
        };
        let evidence = if library {
            ProjectionEvidence::Library(CublasHandleApiIdentity::fixture_identity())
        } else {
            ProjectionEvidence::Native {
                input: &input,
                output: &output,
            }
        };
        let prepared = PreparedCostTemplate::new(shape(), precision, projection).unwrap();
        let classes = PreparedKernelClasses::new(shape(), precision).unwrap();
        let capabilities = GatedDeltaExecutionCapabilities::recurrent_only();
        for counts in [vec![1], vec![1; 8], vec![3, 5]] {
            let rows = rows(&counts);
            let tokens = counts.iter().sum();
            let view = TopologyView::new(precision, &rows);
            for packed in [false, true] {
                if packed && rows.len() < 2 {
                    continue;
                }
                let make_legacy = || {
                    let evidence = legacy(
                        precision,
                        projection,
                        black_box(evidence),
                        black_box(&rows),
                        packed,
                        black_box(&classes),
                    );
                    black_box(reusable_attention_address_scope(&view).unwrap());
                    let topology = reusable_attention_topology_for_tokens(
                        shape(),
                        capabilities,
                        rows.len(),
                        counts.iter().copied(),
                    )
                    .unwrap();
                    (evidence, ReusableExecutionTopology::Dynamic(topology))
                };
                let make_prepared = || {
                    let query = black_box(&prepared)
                        .query(
                            black_box(&rows),
                            tokens,
                            packed,
                            capabilities,
                            Some(black_box(evidence)),
                            true,
                        )
                        .unwrap()
                        .unwrap();
                    (
                        query
                            .compute(SloStructuredCostCapture::HostSettledV1)
                            .unwrap(),
                        query.topology(&view).unwrap(),
                    )
                };
                assert_eq!(make_legacy(), make_prepared());
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    black_box(
                        PreparedCostTemplate::new(black_box(shape()), precision, projection)
                            .unwrap(),
                    );
                }
                let preparation = start.elapsed().as_nanos();
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    black_box(make_legacy());
                }
                let legacy_ns = start.elapsed().as_nanos();
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    black_box(make_prepared());
                }
                let prepared_ns = start.elapsed().as_nanos();
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "kind":"gdn_prepared_template_cpu_comparison", "operation":precision.operation(),
                        "counts":counts, "packed":packed, "iterations":ITERATIONS,
                        "template_preparation_total_ns":preparation,
                        "template_preparation_mean_ns":preparation / ITERATIONS as u128,
                        "legacy_static_classes_compute_and_topology_total_ns":legacy_ns,
                        "legacy_static_classes_compute_and_topology_mean_ns":legacy_ns / ITERATIONS as u128,
                        "checked_query_compute_and_topology_total_ns":prepared_ns,
                        "checked_query_compute_and_topology_mean_ns":prepared_ns / ITERATIONS as u128,
                        "scope":"CPU numerical compute and address/topology declarations only; baseline already uses the existing eight static kernel classes and excludes prior route shape/layout/form validation, while prepared includes CheckedGdnQuery and dispatch counting. Both exclude physical capture, route command construction, global graph, cost-model lookup and planner. No end-to-end 2ms claim."
                    })
                );
            }
        }
    }
}
