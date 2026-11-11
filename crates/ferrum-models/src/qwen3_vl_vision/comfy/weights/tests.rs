//! Owned original visual weights under the caller's actual Host scope. This
//! reuses the existing three-layer BF16 fixture; no second weight generator.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::TensorArchive;
#[path = "../tests/fixture.rs"]
mod fixture;
use fixture::{config, id, limits, Original};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            8 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8 << 20,
                output_bytes: 8 << 20,
                transfer_staging_bytes: 8 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn compile(
    c: &Qwen3VLVisionComfyConfig,
    r: &ReferenceVNextComposition,
) -> CompiledTensorComponent<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let family = TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(Qwen3VLVisionComfyFamily::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let descriptor = r.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        descriptor.total_memory_bytes,
        descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        family,
        r.registry(),
        r.catalog(),
        &policy,
        r.weight_materializers(),
        &c.compile_options().unwrap(),
    )
    .unwrap()
}
fn init(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.visual.weights.{name}")),
        request_id: id(format!("request.visual.weights.{name}")),
        transaction_id: id(format!("transaction.visual.weights.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 256).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn adapter(c: &Qwen3VLVisionComfyConfig) -> (tempfile::TempDir, Qwen3VLVisionComfyWeightAdapter) {
    let (directory, archive) = Original::new(c).archive();
    let source = ComfyCanonicalWeightSource::new(archive).unwrap();
    assert!(source.modules().is_empty());
    (
        directory,
        Qwen3VLVisionComfyWeightAdapter::new(c, Arc::new(source), limits()).unwrap(),
    )
}

#[test]
fn original_bf16_promotion_uses_each_current_scope_and_retires_after_the_last_payload_owner() {
    let c = config(4, 6);
    let (_directory, adapter) = adapter(&c);
    let h = host();
    let component = adapter
        .identity
        .schema()
        .components
        .iter()
        .find(|c| c.dimensions.len() == 1)
        .unwrap();
    let physical = adapter.component_backing_bytes(component, &h).unwrap();
    let scope = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let old = scope.account();
    let source = adapter.weight_source(&old);
    let payload = source.component(component).unwrap();
    assert_eq!(payload.element_type(), ElementType::F32);
    assert!(payload.retained_host_memory().is_some());
    assert_eq!(h.usage().unwrap().committed_bytes(), physical);
    let original = adapter
        .source()
        .archive()
        .tensor(&component.external_names[0])
        .unwrap();
    assert_eq!(original.dtype(), safetensors::Dtype::BF16);
    let expected = original
        .bytes()
        .chunks_exact(2)
        .flat_map(|b| {
            half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))
                .to_f32()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    assert_eq!(payload.bytes(), expected);
    assert_ne!(payload.bytes().as_ptr(), original.bytes().as_ptr());
    scope.close().unwrap();
    // Closing the old authority does not release bytes still owned by this
    // original converted payload, nor allow that view to spend new credit.
    assert_eq!(h.usage().unwrap().committed_bytes(), physical);
    assert!(source.component(component).is_err());
    drop(payload);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let next = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let next_view = next.account();
    let next_source = adapter.weight_source(&next_view);
    let retry = next_source.component(component).unwrap();
    assert_eq!(retry.bytes(), expected);
    drop(retry);
    next.close().unwrap();
    let denied = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical - 1)])
        .unwrap();
    let denied_view = denied.account();
    assert!(adapter
        .weight_source(&denied_view)
        .component(component)
        .is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    denied.close().unwrap();
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn actual_static_plan_group_peak_admits_real_initialization_and_rejects_mismatched_geometry_binding(
) {
    let c = config(4, 6);
    let (_directory, adapter) = adapter(&c);
    let h = host();
    let r =
        ReferenceVNextComposition::create(id("device.reference.visual.weights.groups")).unwrap();
    let compiled = compile(&c, &r);
    let plan = compiled.executable().execution_plan();
    let peak = adapter
        .maximum_group_backing_bytes(compiled.family(), plan, &h)
        .unwrap();
    let placed = plan
        .payload()
        .nodes()
        .iter()
        .flat_map(|n| n.values())
        .filter(|v| v.usage() == BufferUsage::Weights)
        .flat_map(|v| v.storage().components())
        .filter_map(|c| c.component_id().cloned())
        .collect::<BTreeSet<_>>();
    // Identity materialization has one original source per placed group. This
    // is the largest actual promoted payload, not the whole schema's sum.
    assert!(adapter
        .identity
        .component_sources()
        .values()
        .all(|s| s.len() == 1));
    let largest = adapter
        .identity
        .schema()
        .components
        .iter()
        .filter(|c| placed.contains(&c.id))
        .max_by_key(|c| c.physical_bytes().unwrap())
        .unwrap();
    assert_eq!(
        peak,
        h.backing_bytes(largest.physical_bytes().unwrap() as usize)
            .unwrap()
    );
    assert!(peak < adapter.stored_bytes().unwrap() as usize);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let scope = h
        .reserve_scope(&[(HostMemoryClass::WeightBacking, peak)])
        .unwrap();
    let view = scope.account();
    let source = adapter.weight_source(&view);
    let (session, weights) = compiled
        .initialize_shared_weights(
            r.runtime().clone(),
            view.clone(),
            ComponentWeightSource::Stored(&source),
            init("exact_group_peak"),
        )
        .unwrap();
    assert_eq!(
        h.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        0
    );
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("visual weights close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop(weights);
    scope.close().unwrap();
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    // Another valid activation geometry reuses the same original weight
    // adapter, but a plan from that geometry cannot be paired with the old
    // prepared family's fingerprint.
    let alternate = compile(&config(6, 4), &r);
    assert_eq!(
        adapter
            .maximum_group_backing_bytes(
                alternate.family(),
                alternate.executable().execution_plan(),
                &h
            )
            .unwrap(),
        peak
    );
    assert!(adapter
        .maximum_group_backing_bytes(
            compiled.family(),
            alternate.executable().execution_plan(),
            &h
        )
        .is_err());
}

#[test]
fn foreign_component_contracts_and_non_bf16_originals_fail_before_any_promotion() {
    let c = config(4, 6);
    let (_directory, adapter) = adapter(&c);
    let h = host();
    let source = adapter.weight_source(&h);
    let original = &adapter.identity.schema().components[0];
    let mut foreign = original.clone();
    foreign.id = id("component.foreign.visual");
    assert!(source.component(&foreign).is_err());
    let mut dimensions = original.clone();
    dimensions.dimensions[0] += 1;
    assert!(adapter.component_backing_bytes(&dimensions, &h).is_err());
    assert!(source.component(&dimensions).is_err());
    let mut dtype = original.clone();
    dtype.encoding = WeightEncoding::Dense {
        element_type: ElementType::Bf16,
    };
    assert!(adapter.component_backing_bytes(&dtype, &h).is_err());
    assert!(source.component(&dtype).is_err());
    let mut external = original.clone();
    external.external_names[0] = "model.visual.foreign.weight".into();
    assert!(source.component(&external).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);

    // Change one real source tensor's dtype while preserving all other
    // original payload bytes and header shapes. The owned adapter must refuse
    // the source rather than silently treating it as the BF16 declaration.
    let (_source_directory, archive) = Original::new(&c).archive();
    let records = archive
        .tensor_names()
        .map(|name| (name.to_owned(), archive.tensor(name).unwrap()))
        .collect::<Vec<_>>();
    let name = &original.external_names[0];
    let changed = records
        .iter()
        .find(|(n, _)| n == name)
        .unwrap()
        .1
        .bytes()
        .chunks_exact(2)
        .flat_map(|b| {
            half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))
                .to_f32()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("wrong.safetensors");
    let views = records
        .iter()
        .map(|(n, t)| {
            (
                n.as_str(),
                safetensors::tensor::TensorView::new(
                    if n == name {
                        safetensors::Dtype::F32
                    } else {
                        t.dtype()
                    },
                    t.shape().iter().map(|&v| v as usize).collect(),
                    if n == name {
                        changed.as_slice()
                    } else {
                        t.bytes()
                    },
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(views, &None, &file).unwrap();
    let wrong = ComfyCanonicalWeightSource::new(
        SafetensorsArchive::open_single_file(directory.path(), "wrong.safetensors").unwrap(),
    )
    .unwrap();
    assert!(Qwen3VLVisionComfyWeightAdapter::new(&c, Arc::new(wrong), limits()).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
