use super::*;

fn prepared(columns: u64) -> (WeightSchema, plan::Prepared, Vec<Vec<WeightId>>) {
    let (mut schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let groups = groups[0]
        .iter()
        .cloned()
        .map(|id| vec![id])
        .collect::<Vec<_>>();
    for component in &mut schema.components {
        component.dimensions = vec![1, columns];
    }
    for tensor in &mut schema.tensors {
        tensor.dimensions = vec![1, columns];
    }
    let mut weights = program.weights().to_vec();
    for weight in &mut weights {
        weight.tensor.dimensions = vec![1, columns];
    }
    let mut blocks = program.blocks().to_vec();
    for node in &mut blocks[0].nodes {
        node.attributes
            .insert(id("in_features"), SemanticValue::Unsigned(columns));
    }
    let program = ModelProgram::new(
        program.family_id().clone(),
        program.inputs().to_vec(),
        blocks,
        program.states().to_vec(),
        weights,
        program.outputs().to_vec(),
    )
    .unwrap();
    let prepared = plan::prepare_program(&schema, &program, &groups).unwrap();
    (schema, prepared, groups)
}

fn archive(schema: &WeightSchema) -> (tempfile::TempDir, SafetensorsArchive) {
    let directory = tempfile::tempdir().unwrap();
    let bytes = schema
        .components
        .iter()
        .map(|spec| {
            (0..spec.physical_bytes().unwrap() / 2)
                .flat_map(|_| 0x3f80_u16.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tensors = schema
        .components
        .iter()
        .zip(&bytes)
        .map(|(spec, bytes)| {
            (
                spec.external_names[0].as_str(),
                TensorView::new(
                    Dtype::BF16,
                    spec.dimensions.iter().map(|&dim| dim as usize).collect(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive)
}

fn materialize_group(
    materializer: &Bf16WholeTensorFp8Materializer,
    source: &dyn WeightComponentSource,
    group: &plan::ConversionGroup,
    host: &HostMemoryAccount,
) {
    drop(
        materializer
            .materialize_components_with_host(
                source,
                &group.sources.iter().collect::<Vec<_>>(),
                &group.outputs.iter().collect::<Vec<_>>(),
                host,
            )
            .unwrap(),
    );
}

#[test]
fn lora_requantizes_the_shared_group_and_caches_adapted_bf16_without_preflight_allocations() {
    use crate::linear_adapter::*;
    use std::sync::Arc;
    let (schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let (_base_directory, source) = archive(&schema);
    let directory = tempfile::tempdir().unwrap();
    let a = [2_f32; 4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = 1_f32.to_le_bytes();
    let mut tensors = Vec::new();
    for module in ["q", "untouched"] {
        tensors.push((
            format!("{module}.lora_A.weight"),
            TensorView::new(Dtype::F32, vec![1, 4], &a).unwrap(),
        ));
        tensors.push((
            format!("{module}.lora_B.weight"),
            TensorView::new(Dtype::F32, vec![1, 1], &b).unwrap(),
        ));
    }
    let file = directory.path().join("adapter.safetensors");
    serialize_to_file(tensors, &None, &file).unwrap();
    let imported = Arc::new(
        ImportedLinearAdapter::open(&LinearAdapterSource::Diffusers {
            file,
            base_model: "fixture/base".into(),
            scaling: LinearAdapterScaling::Unit,
        })
        .unwrap(),
    );
    let bindings = ["q", "untouched"]
        .into_iter()
        .map(|module| LinearAdapterBinding {
            module: module.into(),
            external_weight: format!("{module}.weight"),
            base_dimensions: [1, 4],
            adapter_output_rows: 0..1,
        })
        .collect();
    let host = account(1 << 20);
    let composition = Arc::new(PreparedLinearAdapter::new(imported, bindings, 1.0).unwrap())
        .manage(&host)
        .unwrap();
    let materializer = Bf16WholeTensorFp8Materializer::new_composed_with_id(
        id("materializer.fixture.lora"),
        host.clone(),
        groups.clone(),
        composition.clone(),
    )
    .unwrap()
    .with_host_cache(NonZeroU64::new(65536).unwrap())
    .unwrap();
    let mut prepared = plan::prepare_program(&schema, &program, &groups).unwrap();
    plan::bind_composition(&mut prepared, composition.prepared(), &host).unwrap();
    let before = host.usage().unwrap();
    let lease = host_budget::prepare(
        &host,
        &prepared,
        materializer.host_cache.as_ref(),
        Some(&source),
    )
    .unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, lease.additional_bytes())])
        .unwrap();
    for group in &prepared.conversion_groups {
        let outputs = materializer
            .materialize_components_with_host(
                &source,
                &group.sources.iter().collect::<Vec<_>>(),
                &group.outputs.iter().collect::<Vec<_>>(),
                &scope.account(),
            )
            .unwrap();
        if group.sources == group.outputs {
            assert!(outputs[0]
                .bytes()
                .chunks_exact(2)
                .all(|bytes| bytes == 0x4040_u16.to_le_bytes()));
        } else {
            for (tensor, expected) in schema.tensors[..3].iter().zip([0x7e_u8, 0x71, 0x71]) {
                let execution = prepared
                    .schema
                    .tensors
                    .iter()
                    .find(|candidate| candidate.id == tensor.id)
                    .unwrap();
                let PhysicalWeightLayout::QuantizedTensor {
                    packed_values,
                    scales,
                    ..
                } = &execution.physical_layout
                else {
                    panic!("quantized group");
                };
                let values = outputs
                    .iter()
                    .find(|output| output.component_id() == &packed_values.component_id)
                    .unwrap();
                let scale = outputs
                    .iter()
                    .find(|output| output.component_id() == &scales.component_id)
                    .unwrap();
                assert_eq!(values.bytes(), &[expected; 4]);
                assert_eq!(
                    f32::from_le_bytes(scale.bytes().try_into().unwrap()),
                    3.0 / 448.0
                );
            }
        }
    }
    drop(scope);
    drop(lease);
    let warm = host_budget::prepare(
        &host,
        &prepared,
        materializer.host_cache.as_ref(),
        Some(&source),
    )
    .unwrap();
    assert_eq!(warm.additional_bytes(), 0);
    let committed = host.usage().unwrap().committed_bytes();
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &host);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), committed);
}

#[test]
fn exact_cache_increment_consumes_scope_and_warm_cache_needs_no_second_claim() {
    let (schema, prepared, groups) = prepared(4);
    let (_directory, source) = archive(&schema);
    let aligned =
        HostMemoryAccount::with_allocation_alignment(account(8192).limits(), 128).unwrap();
    let physical = aligned.backing_bytes(8).unwrap();
    let host =
        HostMemoryAccount::with_allocation_alignment(account(3 * physical).limits(), 128).unwrap();
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
        .unwrap()
        .with_host_cache(NonZeroU64::new(24).unwrap())
        .unwrap();
    let budget = |source: Option<&dyn WeightComponentSource>| {
        host_budget::additional(&host, &prepared, materializer.host_cache.as_ref(), source).unwrap()
    };
    assert_eq!(budget(Some(&source)), 3 * physical);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    materialize_group(
        &materializer,
        &source,
        &prepared.conversion_groups[0],
        &host,
    );
    assert_eq!(budget(Some(&source)), 2 * physical);
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, budget(Some(&source)))])
        .unwrap();
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &scope.account());
    }
    drop(scope);
    assert_eq!(host.usage().unwrap().committed_bytes(), 3 * physical);
    assert_eq!(budget(Some(&source)), 0);
    assert_eq!(budget(None), 3 * physical);
    let (_other_directory, other_source) = archive(&schema);
    assert_eq!(budget(Some(&other_source)), 3 * physical);
    assert_eq!(budget(Some(&source)), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 3 * physical);
}

#[test]
fn forecast_counts_warm_groups_evicted_by_earlier_misses_without_mutating_cache() {
    let (schema, prepared, groups) = prepared(4);
    let (_directory, source) = archive(&schema);
    let host = account(24);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
        .unwrap()
        .with_host_cache(NonZeroU64::new(16).unwrap())
        .unwrap();
    for group in &prepared.conversion_groups[1..] {
        materialize_group(&materializer, &source, group, &host);
    }
    // Visiting group zero evicts one; visiting one then evicts two.
    assert_eq!(
        host_budget::additional(
            &host,
            &prepared,
            materializer.host_cache.as_ref(),
            Some(&source)
        )
        .unwrap(),
        24
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 16);
    // A zero-byte scope cannot convert a group but permits the original hits,
    // proving the
    // admission forecast did not actually evict them or change their order.
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 0)])
        .unwrap();
    for group in &prepared.conversion_groups[1..] {
        materialize_group(&materializer, &source, group, &scope.account());
    }
}

#[test]
fn oversized_uncached_groups_budget_one_payload_and_lookup_with_separate_padding() {
    let (schema, prepared, groups) = prepared(lookup::MINIMUM_PACKED_ELEMENTS);
    let (_directory, source) = archive(&schema);
    let host =
        HostMemoryAccount::with_allocation_alignment(account(4 << 20).limits(), 4096).unwrap();
    let payload = host
        .backing_bytes(lookup::MINIMUM_PACKED_ELEMENTS as usize + 4)
        .unwrap();
    let lookup = host.backing_bytes(lookup::MAGNITUDE_ENTRIES).unwrap();
    let uncached = Bf16WholeTensorFp8Materializer::new(host.clone(), groups.clone())
        .unwrap()
        .with_host_cache(NonZeroU64::new(7).unwrap())
        .unwrap();
    assert_eq!(
        host_budget::additional(&host, &prepared, None, Some(&source)).unwrap(),
        payload + lookup
    );
    assert_eq!(
        host_budget::additional(
            &host,
            &prepared,
            uncached.host_cache.as_ref(),
            Some(&source)
        )
        .unwrap(),
        payload + lookup
    );
    let cached = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
        .unwrap()
        .with_host_cache(NonZeroU64::new(prepared.converted_bytes).unwrap())
        .unwrap();
    assert_eq!(
        host_budget::additional(&host, &prepared, cached.host_cache.as_ref(), Some(&source))
            .unwrap(),
        3 * payload + lookup
    );
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 3 * payload + lookup)])
        .unwrap();
    for group in &prepared.conversion_groups {
        materialize_group(&cached, &source, group, &scope.account());
    }
    drop(scope);
    assert_eq!(host.usage().unwrap().committed_bytes(), 3 * payload);
    assert_eq!(
        host_budget::additional(&host, &prepared, cached.host_cache.as_ref(), Some(&source))
            .unwrap(),
        0
    );
}

#[test]
fn invocation_account_rejects_foreign_parent_even_for_cached_payload() {
    let (_directory, source, specs) = source(&[[0x43e0, 0, 0, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(8);
    let foreign = account(8);
    let materializer =
        Bf16WholeTensorFp8Materializer::new(host.clone(), vec![vec![id("weight.cache")]])
            .unwrap()
            .with_host_cache(NonZeroU64::new(8).unwrap())
            .unwrap();
    assert!(materializer
        .materialize_components_with_host(&source, &refs, &outputs, &foreign)
        .is_err());
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 8)])
        .unwrap();
    let view = scope.account();
    drop(
        materializer
            .materialize_components_with_host(&source, &refs, &outputs, &view)
            .unwrap(),
    );
    drop(scope);
    assert!(materializer
        .materialize_components_with_host(&source, &refs, &outputs, &foreign)
        .is_err());
    drop(
        materializer
            .materialize_components_with_host(&source, &refs, &outputs, &host)
            .unwrap(),
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn preparation_pins_warm_cache_through_competing_source_eviction() {
    let (schema, prepared, groups) = prepared(4);
    let (_directory, source) = archive(&schema);
    let (_other_directory, other_source) = archive(&schema);
    let host = account(48);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
        .unwrap()
        .with_host_cache(NonZeroU64::new(24).unwrap())
        .unwrap();
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &host);
    }
    let preparation = host_budget::prepare(
        &host,
        &prepared,
        materializer.host_cache.as_ref(),
        Some(&source),
    )
    .unwrap();
    assert_eq!(preparation.additional_bytes(), 0);
    let cloned = preparation.clone();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 0)])
        .unwrap();
    // A separate plan replaces every LRU entry after the warm query. Exact
    // original owners stay charged and discoverable through the preparation.
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &other_source, group, &host);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 48);
    drop(preparation);
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &scope.account());
    }
    // A later preparation can pin the earlier lease's owners even though the
    // live LRU contains only the other source. There is no second Host charge.
    let next = host_budget::prepare(
        &host,
        &prepared,
        materializer.host_cache.as_ref(),
        Some(&source),
    )
    .unwrap();
    assert_eq!(next.additional_bytes(), 0);
    drop(cloned);
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &scope.account());
    }
    drop(next);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    let group = &prepared.conversion_groups[0];
    assert!(materializer
        .materialize_components_with_host(
            &source,
            &group.sources.iter().collect::<Vec<_>>(),
            &group.outputs.iter().collect::<Vec<_>>(),
            &scope.account(),
        )
        .is_err());
}

#[test]
fn preparation_retains_new_groups_for_replay_after_scope_credit_is_spent() {
    let (schema, prepared, groups) = prepared(4);
    let (_directory, source) = archive(&schema);
    let (_other_directory, other_source) = archive(&schema);
    let host = account(48);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups)
        .unwrap()
        .with_host_cache(NonZeroU64::new(24).unwrap())
        .unwrap();
    let preparation = host_budget::prepare(
        &host,
        &prepared,
        materializer.host_cache.as_ref(),
        Some(&source),
    )
    .unwrap();
    assert_eq!(preparation.additional_bytes(), 24);
    let scope = host
        .reserve_scope(&[(
            HostMemoryClass::WeightBacking,
            preparation.additional_bytes(),
        )])
        .unwrap();
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &scope.account());
    }
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &other_source, group, &host);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 48);
    for group in &prepared.conversion_groups {
        materialize_group(&materializer, &source, group, &scope.account());
    }
    drop(scope);
    assert_eq!(host.usage().unwrap().committed_bytes(), 48);
    drop(preparation);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
}
