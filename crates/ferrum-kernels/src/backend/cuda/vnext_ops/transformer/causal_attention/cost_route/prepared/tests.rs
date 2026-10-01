use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::*;

struct Fixture {
    operation: OperationId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    values: Vec<ResolvedValueBinding>,
}

fn value(
    role: ResolvedValueRole,
    ordinal: u32,
    dimensions: Vec<u64>,
    dtype: ElementType,
    weight: bool,
    native: bool,
) -> ResolvedValueBinding {
    let name = format!("fixture.{role:?}.{ordinal}");
    let resource = ResourceId::new(format!("resource.{name}")).unwrap();
    let (weight, storage) = if weight {
        let component = WeightId::new(format!("component.{name}")).unwrap();
        let logical = WeightId::new(format!("weight.{name}")).unwrap();
        let mut physical = dimensions.clone();
        if native {
            *physical.last_mut().unwrap() /= 32;
        }
        let component_spec = WeightComponentSpec {
            id: component.clone(),
            role: if native {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            external_names: vec![name.clone()],
            dimensions: physical,
            encoding: if native {
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: QuantizationFormatId::new("quantization.gguf.q8-0").unwrap(),
                    logical_values_per_block: 32,
                    bytes_per_block: 34,
                })
            } else {
                WeightEncoding::Dense {
                    element_type: dtype,
                }
            },
            required: true,
        };
        let bytes = component_spec.physical_bytes().unwrap();
        let schema = WeightSchema {
            format_id: WeightFormatId::new(if native {
                "weight-format.gguf.native-block"
            } else {
                crate::gguf_f16_projection_materializer::GGUF_F16_PROJECTION_FORMAT_ID
            })
            .unwrap(),
            layout_id: WeightLayoutId::new("layout.prepared-causal.fixture").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![component_spec],
            tensors: vec![WeightTensorSpec {
                id: logical.clone(),
                dimensions: dimensions.clone(),
                logical_element_type: dtype,
                physical_layout: if native {
                    PhysicalWeightLayout::BlockQuantized {
                        blocks: PhysicalWeightComponentBinding {
                            component_id: component.clone(),
                            storage: PhysicalStorageLayout::exact_contiguous(),
                        },
                        block_axis: 1,
                        block_padding: PhysicalWeightPadding::Exact,
                    }
                } else {
                    PhysicalWeightLayout::Dense {
                        component_id: component.clone(),
                    }
                },
                required: true,
            }],
        };
        schema
            .validate(&ModelFamilyId::new("family.prepared-causal.fixture").unwrap())
            .unwrap();
        (
            Some(ResolvedWeightBinding::from_schema(&schema, &logical).unwrap()),
            ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                Some(component),
                resource,
                0,
                bytes,
                if native { ElementType::U8 } else { dtype },
            )
            .unwrap()])
            .unwrap(),
        )
    } else {
        let bytes = dimensions.iter().product::<u64>() * dtype.size_bytes();
        (
            None,
            ResolvedValueStorage::single(resource, 0, bytes, dtype).unwrap(),
        )
    };
    ResolvedValueBinding::new(
        ProgramValueId::new(format!("value.{name}")).unwrap(),
        role,
        ordinal,
        ResolvedTensorSpec::new(dimensions, dtype, ResolvedTensorLayout::Contiguous).unwrap(),
        if role == ResolvedValueRole::Output {
            TensorAccess::Write
        } else {
            TensorAccess::Read
        },
        AliasPolicy::NoAlias,
        if weight.is_some() {
            BufferUsage::Weights
        } else if ordinal == 8 {
            BufferUsage::State
        } else {
            BufferUsage::Activations
        },
        weight,
        storage,
    )
    .unwrap()
}

impl Fixture {
    fn new(native: bool) -> Self {
        let mut attributes = super::super::super::tests::attributes(false);
        for (key, n) in [
            ("hidden_size", 256),
            ("query_heads", 2),
            ("query_features", 256),
            ("query_projection_features", 256),
            ("maximum_context_tokens", 32768),
        ] {
            attributes.insert(AttributeId::new(key).unwrap(), SemanticValue::Unsigned(n));
        }
        let mut values = vec![value(
            ResolvedValueRole::Input,
            0,
            vec![32768, 256],
            ElementType::F32,
            false,
            false,
        )];
        for (ordinal, dims) in [
            (1, vec![256]),
            (2, vec![256, 256]),
            (3, vec![256, 256]),
            (4, vec![256, 256]),
            (5, vec![256, 256]),
            (6, vec![128]),
            (7, vec![128]),
        ] {
            values.push(value(
                ResolvedValueRole::Input,
                ordinal,
                dims,
                ElementType::F16,
                true,
                native && (2..=5).contains(&ordinal),
            ));
        }
        values.push(value(
            ResolvedValueRole::Input,
            8,
            vec![2, 2, 128],
            ElementType::F16,
            false,
            false,
        ));
        values.push(value(
            ResolvedValueRole::Output,
            0,
            vec![32768, 256],
            ElementType::F32,
            false,
            false,
        ));
        Self {
            operation: OperationId::new(if native {
                CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID
            } else {
                CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
            })
            .unwrap(),
            attributes,
            values,
        }
    }
    fn prepare(&self) -> Result<Option<PreparedCostData>, String> {
        PreparedCostData::new(
            &self.operation,
            &self.values,
            &self.attributes,
            CausalAttentionSemantics::Standard,
            CausalPrecision::F32Master,
            #[cfg(feature = "vllm-marlin")]
            MarlinProjectionRuntime {
                multiprocessor_count: 1,
                device_ordinal: 0,
            },
        )
    }
}

// This is the production selected-compute kernel, with current rows rebuilt.
// No runtime/context/stream is constructed and no device is queried.
fn project(
    data: &PreparedCostData,
    counts: &[(u64, u64)],
) -> Option<SelectedCommandCostEvidenceV1> {
    project_with_library(
        data,
        counts,
        Some(super::super::super::super::cublas_api::CublasHandleApiIdentity::fixture_identity()),
    )
}

fn project_with_library(
    data: &PreparedCostData,
    counts: &[(u64, u64)],
    library: Option<super::super::super::super::cublas_api::CublasHandleApiIdentity>,
) -> Option<SelectedCommandCostEvidenceV1> {
    project_using(data, counts, library, true)
}

fn project_using(
    data: &PreparedCostData,
    counts: &[(u64, u64)],
    library: Option<super::super::super::super::cublas_api::CublasHandleApiIdentity>,
    use_template: bool,
) -> Option<SelectedCommandCostEvidenceV1> {
    let policy = AttentionExecutionPolicy::NativeAdaptive;
    let rows = counts
        .iter()
        .map(|&(tokens, context)| selected::Row::new(data.shape, policy, tokens, context, false))
        .collect::<Option<Vec<_>>>()?;
    let projections = if data.rounded {
        selected::ProjectionWork::DenseF16(library?)
    } else {
        selected::ProjectionWork::Native([
            &data.parts[0],
            &data.parts[1],
            &data.parts[2],
            &data.parts[3],
        ])
    };
    if use_template {
        return data
            .template
            .geometry(policy, counts.iter().map(|r| r.0).sum(), rows.len())
            .ok()?
            .finish(&rows, counts.len() > 1, projections)?
            .compute(SloStructuredCostCapture::HostSettledV1);
    }
    selected::compute(
        data.shape,
        CausalPrecision::F32Master,
        data.projection,
        policy,
        projections,
        &rows,
        counts.iter().map(|r| r.0).sum(),
        counts.len() > 1,
        SloStructuredCostCapture::HostSettledV1,
    )
}

#[test]
fn prepared_causal_metadata_keeps_tokens_context_and_algorithm_selection_dynamic() {
    for native in [false, true] {
        let fixture = Fixture::new(native);
        let prepared = fixture.prepare().unwrap().unwrap();
        let mut signatures = std::collections::BTreeSet::new();
        for counts in [
            &[(1, 37)][..],
            &[(1, 38)][..],
            &[(4, 2048)][..],
            &[(2, 4000)][..],
            &[(8, 13000)][..],
            &[(1, 37), (1, 81)][..],
        ] {
            let fresh = fixture.prepare().unwrap().unwrap();
            let reference = project_using(&fresh, counts,
                Some(super::super::super::super::cublas_api::CublasHandleApiIdentity::fixture_identity()), false).unwrap();
            let candidate = project(&prepared, counts).unwrap();
            assert_eq!(candidate, reference);
            assert_eq!(candidate.algorithm_work(), reference.algorithm_work());
            let dispatches = reference
                .algorithm_work()
                .unwrap()
                .unwrap()
                .entries()
                .iter()
                .map(|entry| entry.commands())
                .sum();
            ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1::from_selected(
                &reference,
                counts.iter().map(|r| r.0).sum(),
                dispatches,
                0,
            )
            .unwrap()
            .validate_binding(&candidate)
            .unwrap();
            signatures.insert(*candidate.family_signature());
        }
        let before = project(&prepared, &[(1, 37)]).unwrap();
        let after = project(&prepared, &[(1, 38)]).unwrap();
        assert_ne!(
            before.work(),
            after.work(),
            "current context work must not come from preparation"
        );
        assert!(
            signatures.len() > 1,
            "different selected paths must not share a cached route"
        );
        #[cfg(feature = "vllm-paged-attn-v2")]
        assert_ne!(
            CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::NativeAdaptive,
                prepared.shape,
                4,
                2048
            )
            .unwrap(),
            CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::NativeAdaptive,
                prepared.shape,
                8,
                13000
            )
            .unwrap(),
        );
        assert!(project(&prepared, &[(1, 32769)]).is_none());
        assert!(project(&prepared, &[(0, 1)]).is_none());
    }
}

// Numerical declarations only: this view holds no live resource authority.
// The interface/core fixtures retain the actual capture and submission guards.
struct SelectionView<'a> {
    fixture: &'a Fixture,
    rows: Vec<OperationCostWorkRow>,
    scratch: ResourceId,
    binding: ResourceId,
    unavailable: Option<ResourceId>,
}

impl<'a> SelectionView<'a> {
    fn new(fixture: &'a Fixture, counts: &[(u64, u64)]) -> Self {
        Self {
            fixture,
            rows: counts
                .iter()
                .map(|&(tokens, context)| OperationCostWorkRow {
                    offset: context - tokens,
                    count: std::num::NonZeroU64::new(tokens).unwrap(),
                    full_input_tokens: std::num::NonZeroU64::new(context).unwrap(),
                })
                .collect(),
            scratch: ResourceId::new("selection.scratch").unwrap(),
            binding: ResourceId::new("selection.binding").unwrap(),
            unavailable: None,
        }
    }
}

impl ReusableExecutionTopologyView for SelectionView<'_> {
    fn operation_id(&self) -> &OperationId {
        &self.fixture.operation
    }
    fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        &self.fixture.attributes
    }
    fn bindings(&self) -> &[ResolvedValueBinding] {
        &self.fixture.values
    }
    fn memory_plan(&self) -> &MemoryPlan {
        panic!("numerical topology uses declared scopes, never a fabricated memory plan")
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
            ReusableExecutionWorkspaceAddress::Binding => Some(&self.binding),
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
fn combined_causal_selection_matches_fresh_topology_for_dynamic_rows() {
    for native in [false, true] {
        let fixture = Fixture::new(native);
        let prepared = fixture.prepare().unwrap().unwrap();
        for counts in [
            &[(1, 37)][..],
            &[(1, 512)][..],
            &[(1, 513)][..],
            &[(4, 2048)][..],
            &[(8, 13000)][..],
            &[(1, 37), (2, 81)][..],
        ] {
            let view = SelectionView::new(&fixture, counts);
            let paths = counts
                .iter()
                .map(|&(tokens, context)| {
                    CausalAttentionKernelPath::select(
                        AttentionExecutionPolicy::NativeAdaptive,
                        prepared.shape,
                        tokens,
                        context,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let from_cost = super::super::topology_from_cost_selection(
                &view,
                CausalAttentionSemantics::Standard,
                prepared.shape,
                &paths,
            )
            .unwrap();
            let rows = counts
                .iter()
                .map(|&(tokens, context)| {
                    selected::Row::new(
                        prepared.shape,
                        AttentionExecutionPolicy::NativeAdaptive,
                        tokens,
                        context,
                        false,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let from_template = super::super::topology_from_prepared_selection(
                &view,
                CausalAttentionSemantics::Standard,
                prepared.shape,
                &rows,
            )
            .unwrap();
            let fresh = topology(
                &view,
                AttentionExecutionPolicy::NativeAdaptive,
                CausalAttentionSemantics::Standard,
            )
            .unwrap();
            assert_eq!(from_cost, fresh);
            assert_eq!(from_template, fresh);
            assert!(super::super::topology_from_cost_selection(
                &view,
                CausalAttentionSemantics::Standard,
                prepared.shape,
                &[],
            )
            .is_err());
        }
    }
}

#[test]
fn combined_causal_selection_rechecks_every_captured_address_and_workspace() {
    let fixture = Fixture::new(true);
    let prepared = fixture.prepare().unwrap().unwrap();
    let mut view = SelectionView::new(&fixture, &[(1, 37)]);
    let paths = [CausalAttentionKernelPath::select(
        AttentionExecutionPolicy::NativeAdaptive,
        prepared.shape,
        1,
        37,
    )
    .unwrap()];
    let selected = |view: &SelectionView<'_>| {
        let rows = [selected::Row::new(
            prepared.shape,
            AttentionExecutionPolicy::NativeAdaptive,
            1,
            37,
            false,
        )
        .unwrap()];
        let legacy = super::super::topology_from_cost_selection(
            view,
            CausalAttentionSemantics::Standard,
            prepared.shape,
            &paths,
        )
        .unwrap();
        let candidate = super::super::topology_from_prepared_selection(
            view,
            CausalAttentionSemantics::Standard,
            prepared.shape,
            &rows,
        )
        .unwrap();
        assert_eq!(candidate, legacy);
        candidate
    };
    let known = selected(&view);
    let mut captured = fixture
        .values
        .iter()
        .filter(|value| !(value.role() == ResolvedValueRole::Input && value.ordinal() == 8))
        .flat_map(|value| {
            value
                .storage()
                .components()
                .iter()
                .map(|part| part.resource_id().clone())
        })
        .collect::<Vec<_>>();
    captured.extend([view.scratch.clone(), view.binding.clone()]);
    for resource in captured {
        view.unavailable = Some(resource);
        assert_eq!(selected(&view), ReusableExecutionTopology::EagerBoundary);
        view.unavailable = None;
        assert_eq!(selected(&view), known);
    }
}

#[test]
fn combined_causal_compute_keeps_current_library_identity_out_of_preparation() {
    use super::super::super::super::cublas_api::CublasHandleApiIdentity;
    let fixture = Fixture::new(false);
    let prepared = fixture.prepare().unwrap().unwrap();
    let rows = &[(1, 37), (1, 81)];
    let first = project_with_library(
        &prepared,
        rows,
        Some(CublasHandleApiIdentity::fixture_identity_for_version(
            120900,
        )),
    )
    .unwrap();
    let changed = project_with_library(
        &prepared,
        rows,
        Some(CublasHandleApiIdentity::fixture_identity_for_version(
            130000,
        )),
    )
    .unwrap();
    assert_ne!(first.family_signature(), changed.family_signature());
    assert!(project_with_library(&prepared, rows, None).is_none());
    assert_eq!(project(&prepared, rows).unwrap(), first);
}

#[test]
fn prepared_causal_metadata_revalidates_changed_attributes_weights_and_precision() {
    let mut fixture = Fixture::new(true);
    let old = fixture.prepare().unwrap().unwrap();
    fixture.attributes.insert(
        AttributeId::new("maximum_context_tokens").unwrap(),
        SemanticValue::Unsigned(1024),
    );
    let changed = fixture.prepare().unwrap().unwrap();
    assert_ne!(
        old.shape.maximum_context_tokens,
        changed.shape.maximum_context_tokens
    );
    assert!(project(&old, &[(1, 2048)]).is_some());
    assert!(project(&changed, &[(1, 2048)]).is_none());
    fixture.attributes.insert(
        AttributeId::new("hidden_size").unwrap(),
        SemanticValue::Unsigned(512),
    );
    assert!(fixture.prepare().is_err());
    let mut fixture = Fixture::new(false);
    fixture.values[2] = value(
        ResolvedValueRole::Input,
        2,
        vec![256, 256],
        ElementType::F16,
        true,
        true,
    );
    assert!(
        fixture.prepare().is_err(),
        "RN dense contract cannot reuse native physical layout"
    );
    let fixture = Fixture::new(false);
    assert!(PreparedCostData::new(
        &fixture.operation,
        &fixture.values,
        &fixture.attributes,
        CausalAttentionSemantics::Standard,
        CausalPrecision::F16,
        #[cfg(feature = "vllm-marlin")]
        MarlinProjectionRuntime {
            multiprocessor_count: 1,
            device_ordinal: 0
        },
    )
    .is_err());
}

#[test]
fn prepared_causal_template_matches_legacy_for_batch_alias_and_kernel_abi_variants() {
    use super::super::super::super::cublas_api::CublasHandleApiIdentity;
    let fixture = Fixture::new(false);
    let base = fixture.prepare().unwrap().unwrap().shape;
    for precision in [CausalPrecision::F16, CausalPrecision::F32Master] {
        for (head_dim, sliding, gate, post_norm) in
            [(128, 0, false, false), (256, 1024, true, true)]
        {
            let mut shape = base;
            shape.head_dim = head_dim;
            shape.query_features = shape.query_heads * head_dim;
            shape.kv_features = shape.key_value_heads * head_dim;
            shape.query_projection_features = shape.query_features * if gate { 2 } else { 1 };
            shape.output_gate = gate;
            shape.post_attention_norm = post_norm;
            shape.sliding_window_tokens = sliding;
            let template =
                selected::CostTemplate::new(shape, precision, CausalProjection::F16).unwrap();
            for policy in [
                AttentionExecutionPolicy::Portable,
                AttentionExecutionPolicy::NativeAdaptive,
            ] {
                for counts in [
                    &[(1, 37)][..],
                    &[(1, 512)][..],
                    &[(1, 513)][..],
                    &[(4, 2048)][..],
                    &[(8, 13000)][..],
                    &[(1, 37), (1, 513), (1, 13000)][..],
                    &[(3, 1024), (7, 3072)][..],
                ] {
                    for packed in [false, true]
                        .into_iter()
                        .filter(|packed| !packed || counts.len() > 1)
                    {
                        for inplace in [false, true] {
                            let rows = counts
                                .iter()
                                .map(|&(tokens, end)| {
                                    selected::Row::new(shape, policy, tokens, end, inplace).unwrap()
                                })
                                .collect::<Vec<_>>();
                            let tokens = counts.iter().map(|row| row.0).sum();
                            let projections = selected::ProjectionWork::DenseF16(
                                CublasHandleApiIdentity::fixture_identity(),
                            );
                            let query = template
                                .geometry(policy, tokens, rows.len())
                                .unwrap()
                                .finish(&rows, packed, projections)
                                .unwrap();
                            let candidate = query.compute(SloStructuredCostCapture::HostSettledV1);
                            let legacy = selected::compute(
                                shape,
                                precision,
                                CausalProjection::F16,
                                policy,
                                projections,
                                &rows,
                                tokens,
                                packed,
                                SloStructuredCostCapture::HostSettledV1,
                            );
                            assert!(
                                legacy.is_some(),
                                "fixture must exercise a complete numeric route"
                            );
                            assert_eq!(candidate, legacy, "head={head_dim} policy={policy:?} packed={packed} inplace={inplace}");
                            assert!(query.compute(SloStructuredCostCapture::Disabled).is_none());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_causal_query_rejects_mixed_alias_and_topology_row_substitution() {
    use super::super::super::super::cublas_api::CublasHandleApiIdentity;
    let fixture = Fixture::new(false);
    let prepared = fixture.prepare().unwrap().unwrap();
    let counts = [(1, 37), (1, 81)];
    let policy = AttentionExecutionPolicy::NativeAdaptive;
    let rows = [
        selected::Row::new(prepared.shape, policy, 1, 37, false).unwrap(),
        selected::Row::new(prepared.shape, policy, 1, 81, true).unwrap(),
    ];
    let projections =
        selected::ProjectionWork::DenseF16(CublasHandleApiIdentity::fixture_identity());
    let query = prepared
        .template
        .geometry(policy, 2, 2)
        .unwrap()
        .finish(&rows, true, projections)
        .unwrap();
    assert!(query
        .compute(SloStructuredCostCapture::HostSettledV1)
        .is_none());
    assert!(prepared
        .template
        .geometry(policy, 3, 2)
        .unwrap()
        .finish(&rows, true, projections)
        .is_none());
    let mut view = SelectionView::new(&fixture, &counts);
    view.rows[1].offset += 1;
    view.rows[1].full_input_tokens = std::num::NonZeroU64::new(82).unwrap();
    assert!(super::super::topology_from_prepared_selection(
        &view,
        CausalAttentionSemantics::Standard,
        prepared.shape,
        &rows
    )
    .is_err());
}

#[test]
#[ignore = "CPU timing diagnostic; run in release on the CUDA build host, no GPU context"]
fn prepared_causal_cpu_preparation_and_repeated_selected_projection_cost() {
    use std::{hint::black_box, time::Instant};
    const N: usize = 2048;
    for native in [false, true] {
        let fixture = Fixture::new(native);
        for counts in [vec![(1, 456)], vec![(1, 456); 8], vec![(4, 2048)]] {
            let prepared = fixture.prepare().unwrap().unwrap();
            project(&prepared, &counts).unwrap(); // initialize immutable code identities
            let start = Instant::now();
            for _ in 0..N {
                black_box(fixture.prepare().unwrap().unwrap());
            }
            let preparation = start.elapsed().as_nanos();
            let start = Instant::now();
            for _ in 0..N {
                let fresh = fixture.prepare().unwrap().unwrap();
                black_box(project(&fresh, black_box(&counts)).unwrap());
            }
            let unprepared = start.elapsed().as_nanos();
            let start = Instant::now();
            for _ in 0..N {
                black_box(project(black_box(&prepared), black_box(&counts)).unwrap());
            }
            let reused = start.elapsed().as_nanos();
            let library = Some(
                super::super::super::super::cublas_api::CublasHandleApiIdentity::fixture_identity(),
            );
            assert_eq!(
                project_using(&prepared, &counts, library, false),
                project_using(&prepared, &counts, library, true)
            );
            let start = Instant::now();
            for _ in 0..N {
                black_box(
                    project_using(black_box(&prepared), black_box(&counts), library, false)
                        .unwrap(),
                );
            }
            let untemplated = start.elapsed().as_nanos();
            eprintln!(
                "{}",
                serde_json::json!({"kind":"causal_static_preparation_cpu_comparison", "native":native, "rows":counts.len(), "tokens_per_row":counts[0].0, "context":counts[0].1, "iterations":N, "preparation_total_ns":preparation, "unprepared_plus_selected_compute_total_ns":unprepared, "prepared_selected_compute_total_ns":reused, "untemplated_selected_compute_total_ns":untemplated, "scope":"CPU immutable class/GEMM template plus one dynamic geometry and selected_compute; untemplated reference shares emission and geometry but rebuilds original classes/GEMM layouts; excludes physical range/resource/canonical projection and does not prove end-to-end planner benefit"})
            );
        }
    }
}
