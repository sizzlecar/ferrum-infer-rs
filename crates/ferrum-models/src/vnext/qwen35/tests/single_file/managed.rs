//! Exercise the sealed managed adapter with the same actual BF16 Qwen fixture.
mod process_mapped;
mod reusable;
#[cfg(feature = "cuda")]
mod reusable_cuda;
use super::*;
use ferrum_execution::{ProgramExecutorConfig, RuntimeComposition};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_runtime::{
    DeviceId, ExecutablePlanView, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits,
    HostMemoryLimits, WeightMaterializerSelection,
};

fn prepare(sources: ProductionModelSourceBundle) -> Arc<PreparedProductionModel> {
    let sources = Arc::new(sources);
    Arc::new(
        crate::vnext::resolve_registered_model_from_sources(&sources)
            .unwrap()
            .into_required()
            .unwrap()
            .define_from_sources(sources)
            .unwrap()
            .prepare(&F16_NUMERICAL_PROFILE_ID.parse().unwrap())
            .unwrap(),
    )
}

fn host(weight_backing: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 18,
                output_bytes: 1 << 18,
                transfer_staging_bytes: 1 << 18,
                weight_backing_bytes: weight_backing,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}

fn component<'a>(prepared: &'a PreparedProductionModel, name: &str) -> &'a WeightComponentSpec {
    prepared
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .find(|component| component.external_names == [name])
        .unwrap()
}

#[test]
fn managed_preparation_preserves_actual_family_and_frozen_source_identities() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let host = host(1 << 18);
    let source = mapped.weight_source(&host);
    let original = prepared.execution_input().unwrap();
    let managed = source.execution_input().unwrap();
    assert!(std::ptr::eq(original.family(), managed.family()));
    assert_eq!(original.original_sources(), managed.original_sources());
    assert_eq!(original.resolved_sources(), managed.resolved_sources());
    assert_eq!(original.facts(), managed.facts());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);

    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ] {
        let spec = component(&prepared, name);
        let payload = managed.weights().component(spec).unwrap();
        let expected = if name.contains("norm") { 1.125 } else { 0.375 };
        assert_eq!(
            payload.bytes(),
            f16::from_f32(expected)
                .to_bits()
                .to_le_bytes()
                .repeat(spec.physical_bytes().unwrap() as usize / 2)
        );
        let bytes = host
            .backing_bytes(spec.physical_bytes().unwrap() as usize)
            .unwrap();
        assert_eq!(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::WeightBacking),
            bytes
        );
        assert_eq!(
            payload
                .retained_host_memory()
                .unwrap()
                .managed_memory_class(),
            Some(HostMemoryClass::WeightBacking)
        );
        drop(payload);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn managed_preparation_uses_existing_scope_and_retains_actual_conversion_owner() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let spec = component(&prepared, "model.language_model.embed_tokens.weight");
    let parent = host(1 << 18);
    let bytes = parent
        .backing_bytes(spec.physical_bytes().unwrap() as usize)
        .unwrap();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, bytes)])
        .unwrap();
    let account = scope.account();
    let source = mapped.weight_source(&account);
    assert_eq!(
        source.component_backing_bytes(spec).unwrap(),
        spec.physical_bytes().unwrap()
    );
    let payload = source.component(spec).unwrap();
    assert!(source.component(spec).is_err());
    let owner = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), bytes);
    assert_eq!(parent.usage().unwrap().committed_bytes(), bytes);
    assert!(source.component(spec).is_err());
    drop(owner);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);

    let too_small = host(bytes - 1);
    let source = mapped.weight_source(&too_small);
    assert!(source.component(spec).is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    let mut foreign = spec.clone();
    foreign.external_names[0] = "lm_head.weight".to_owned();
    assert!(source.component(&foreign).is_err());
    assert!(source.component_backing_bytes(&foreign).is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn managed_preparation_checks_actual_mmap_and_keeps_the_original_mapping_lease() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let archive = mapped.shared_archive();
    assert!(Arc::ptr_eq(&archive, &mapped.shared_archive()));
    let canonical =
        ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource::from_shared_archive(
            archive.clone(),
        )
        .unwrap();
    assert!(std::ptr::eq(canonical.archive(), archive.as_ref()));
    let name = "model.language_model.embed_tokens.weight";
    let original_bytes = canonical.archive().tensor(name).unwrap().bytes().to_vec();
    let spec = component(&prepared, "model.language_model.embed_tokens.weight");
    let mut foreign = std::fs::read(&fixture.selected).unwrap();
    // Replace the pathname atomically, leaving the retained old mmap immutable.
    *foreign.last_mut().unwrap() ^= 1;
    let replacement = fixture.selected.with_extension("replacement");
    std::fs::write(&replacement, foreign).unwrap();
    std::fs::rename(replacement, &fixture.selected).unwrap();
    assert!(prepared.managed_safetensors().is_err());
    let rebound = mapped.with_prepared(Arc::clone(&prepared)).unwrap();
    assert!(Arc::ptr_eq(&archive, &rebound.shared_archive()));
    assert_eq!(
        canonical.archive().tensor(name).unwrap().bytes(),
        original_bytes
    );
    let host = host(1 << 18);
    let source = mapped.weight_source(&host);
    assert_eq!(
        source.component(spec).unwrap().bytes(),
        f16::from_f32(0.375).to_bits().to_le_bytes().repeat(512)
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    let rebound_source = rebound.weight_source(&host);
    assert_eq!(
        rebound_source.component(spec).unwrap().bytes(),
        f16::from_f32(0.375).to_bits().to_le_bytes().repeat(512)
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn managed_preparation_shared_archive_rejects_a_foreign_frozen_source_authority() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let same_bytes_new_authority = prepare(fixture.open().unwrap());
    let error = mapped
        .with_prepared(same_bytes_new_authority)
        .err()
        .expect("a different source Arc cannot borrow the verified map");
    assert!(error.to_string().contains("source authority"));
    assert!(Arc::ptr_eq(
        &mapped.shared_archive(),
        &mapped.shared_archive()
    ));
}

#[test]
fn managed_preparation_directory_index_is_bound_to_frozen_selected_shards() {
    let fixture = Fixture::new();
    let directory = fixture._root.path().join("indexed");
    std::fs::create_dir(&directory).unwrap();
    std::fs::copy(&fixture.selected, directory.join("part.safetensors")).unwrap();
    let archive = SafetensorsArchive::open_single_file(
        fixture.selected.parent().unwrap(),
        "official-bf16.safetensors",
    )
    .unwrap();
    let map = archive
        .tensor_names()
        .map(|name| {
            (
                name.to_owned(),
                Value::String("part.safetensors".to_owned()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let index = serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap();
    std::fs::write(directory.join("model.safetensors.index.json"), &index).unwrap();
    let local = |kind, path: &Path| OriginalModelSource {
        kind,
        location: path.display().to_string(),
        requested_revision: None,
    };
    let sources = crate::vnext::open_registered_product_sources(
        &fixture.semantic,
        &fixture.tokenizer,
        ProductionWeightArtifact::safetensors_directory(&directory),
        OriginalModelSources {
            semantic: local(ModelSourceKind::LocalDirectory, &fixture.semantic),
            tokenizer: local(ModelSourceKind::LocalDirectory, &fixture.tokenizer),
            weights: local(ModelSourceKind::LocalDirectory, &directory),
        },
    )
    .unwrap();
    let prepared = prepare(sources);
    assert!(prepared.managed_safetensors().is_ok());
    let mut changed = index;
    changed.push(b' '); // Same parsed selection, different frozen source bytes.
    std::fs::write(directory.join("model.safetensors.index.json"), changed).unwrap();
    assert!(prepared.managed_safetensors().is_err());
}

#[test]
fn managed_preparation_actual_identity_static_groups_use_peak_not_model_sum() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let host = host(1 << 18);
    let source = mapped.weight_source(&host);
    let (runtime, registry, materializers, materializer, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.cpu.managed-source-budget").unwrap(),
        64 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let composition = RuntimeComposition::new(runtime.clone(), registry, materializers, catalog);
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    engine.backend.enable_reusable_execution = false;
    engine.scheduler.max_running_requests = 2;
    engine.batching.max_num_batched_tokens = 8;
    engine.runtime.max_model_len = Some(128);
    let info = prepared.model_info(engine.model.model_id.clone(), ferrum_types::Device::CPU);
    let config =
        ProgramExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref()).unwrap();
    let compiled = composition
        .compile_model(
            source.execution_input().unwrap(),
            info,
            &engine,
            config,
            WeightMaterializerSelection::exact(materializer),
        )
        .unwrap();
    let plan = compiled.compilation().executable().execution_plan();
    let components = &prepared.family().stored_weight_schema().unwrap().components;
    let allocations = components
        .iter()
        .map(|component| {
            host.backing_bytes(source.component_backing_bytes(component).unwrap() as usize)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let peak = *allocations.iter().max().unwrap();
    assert!(peak < allocations.iter().sum::<usize>());
    assert_eq!(source.maximum_group_backing_bytes(plan).unwrap(), peak);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    // The query's exact bound admits every actual individual conversion.
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, peak)])
        .unwrap();
    let account = scope.account();
    let source = mapped.weight_source(&account);
    for component in components {
        let payload = source.component(component).unwrap();
        assert_eq!(
            payload.bytes(),
            prepared.weights().component(component).unwrap().bytes()
        );
        drop(payload);
    }
    scope.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
