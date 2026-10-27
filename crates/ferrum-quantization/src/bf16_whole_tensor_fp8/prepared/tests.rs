use super::super::tests::{account, fixture};
use super::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn original(root: &Path, schema: &WeightSchema) -> Arc<SafetensorsArchive> {
    let rows = [
        [0x43e0_u16, 0x3f88, 0xbf98, 0x8000],
        [0x4360, 0x3f88, 0xbf98, 0],
        [0x42e0, 0x3f80, 0xbf80, 0x3b00],
        [0x3f80, 0xbf80, 0x8000, 0],
    ];
    let bytes: Vec<Vec<u8>> = rows
        .iter()
        .map(|row| row.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect();
    let views = schema
        .components
        .iter()
        .zip(&bytes)
        .map(|(spec, bytes)| {
            (
                spec.external_names[0].as_str(),
                TensorView::new(Dtype::BF16, vec![1, 4], bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &root.join("original.safetensors")).unwrap();
    Arc::new(SafetensorsArchive::open_single_file(root, "original.safetensors").unwrap())
}

fn exported(
    root: &Path,
    host: HostMemoryAccount,
) -> (
    Arc<SafetensorsArchive>,
    Bf16WholeTensorFp8Materializer,
    plan::Prepared,
    WeightSchema,
    PreparedFp8ExportReceipt,
) {
    let (schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let source = original(root, &schema);
    let materializer = Bf16WholeTensorFp8Materializer::new(host, groups.clone()).unwrap();
    let plan = plan::prepare_program(&schema, &program, &groups).unwrap();
    let receipt = materializer
        .export_plan(&schema, &plan, source.as_ref(), &root.join("prepared"))
        .unwrap();
    (source, materializer, plan, schema, receipt)
}

#[test]
fn prepared_export_is_byte_exact_with_ordered_shared_scale_and_bf16_passthrough() {
    let root = tempfile::tempdir().unwrap();
    let host = account(24); // The shared three-matrix group needs exactly 24B.
    let (source, materializer, plan, schema, receipt) = exported(root.path(), host.clone());
    assert_eq!(receipt.payload_bytes, 32);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let archive =
        PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    archive
        .validate_recipe_parts(&schema, &plan, &materializer.groups)
        .unwrap();
    let wrapper = PreparedFp8TensorArchive::new(source.clone(), archive.clone());
    let exact = ExactBf16TensorSource::new(&wrapper);
    let originals: BTreeMap<_, _> = schema.components.iter().map(|s| (&s.id, s)).collect();
    for output in &plan.schema.components {
        let inputs: Vec<_> = plan.sources[&output.id]
            .iter()
            .map(|id| originals[id])
            .collect();
        let expected = materializer
            .materialize_components(source.as_ref(), &inputs, &[output])
            .unwrap();
        let actual = archive.materialize(&exact, &inputs, &[output]).unwrap();
        assert_eq!(actual[0].bytes(), expected[0].bytes(), "{}", output.id);
        assert_eq!(actual[0].dimensions(), output.dimensions);
        assert!(actual[0].retained_host_memory().is_some());
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    // Known independent conversion vectors, including a tie and signed zero.
    let inputs: Vec<_> = schema.components[..3].iter().collect();
    let derived = plan::derive_group(&inputs).unwrap();
    assert_eq!(
        archive.component(&derived[0]).unwrap().bytes(),
        &[126, 56, 186, 128]
    );
    for scale in &derived[3..] {
        assert_eq!(
            archive.component(scale).unwrap().bytes(),
            &1_f32.to_le_bytes()
        );
    }
    // Original converted BF16 is still an explicit inventory/passthrough request,
    // but prepared materialization itself does not use it.
    assert_eq!(
        exact.component(&schema.components[0]).unwrap().bytes(),
        source.component(&schema.components[0]).unwrap().bytes()
    );
    let native = exact.component(&schema.components[3]).unwrap();
    assert_eq!(native.element_type(), ElementType::Bf16);
    let retained = native.retained_host_memory().unwrap().clone();
    let expected = native.bytes().to_vec();
    drop(native);
    drop(exact);
    drop(wrapper);
    drop(archive);
    drop(source);
    assert_eq!(retained.bytes(), expected);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn prepared_reading_requires_selected_owner_and_preserves_quality_identity() {
    let root = tempfile::tempdir().unwrap();
    let (source, materializer, plan, schema, receipt) = exported(root.path(), account(24));
    let archive =
        PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    let old_descriptor = materializer.descriptor.clone();
    let bound = materializer.bind_prepared(archive.clone()).unwrap();
    assert_eq!(
        bound.descriptor.fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    assert_eq!(
        bound.descriptor.approximate_quality_contract(),
        old_descriptor.approximate_quality_contract()
    );
    assert_ne!(
        bound.descriptor.implementation_fingerprint(),
        old_descriptor.implementation_fingerprint()
    );
    let artifact: serde_json::Value = serde_json::from_slice(
        &quality::artifact(&bound.descriptor, &schema, &plan.schema).unwrap(),
    )
    .unwrap();
    assert_eq!(
        artifact["materializer"]["implementation_fingerprint"],
        bound.descriptor.implementation_fingerprint()
    );
    assert!(bound.with_host_cache(NonZeroU64::new(24).unwrap()).is_err());

    let originals: BTreeMap<_, _> = schema.components.iter().map(|s| (&s.id, s)).collect();
    let output = &plan.schema.components[0];
    let inputs: Vec<_> = plan.sources[&output.id]
        .iter()
        .map(|id| originals[id])
        .collect();
    let wrapper = PreparedFp8TensorArchive::new(source.clone(), archive.clone());
    archive.materialize(&wrapper, &inputs, &[output]).unwrap();
    // Same bytes in a separately mapped instance cannot masquerade as the owner
    // selected in the plan. Raw BF16 cannot trigger a conversion fallback either.
    let other = PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    let other_source = PreparedFp8TensorArchive::new(source.clone(), other);
    assert!(archive
        .materialize(&other_source, &inputs, &[output])
        .is_err());
    assert!(archive
        .materialize(source.as_ref(), &inputs, &[output])
        .is_err());
    let mut wrong = output.clone();
    wrong.dimensions = vec![2, 2];
    assert!(wrapper.component(&wrong).is_err());
    assert!(archive
        .materialize(&wrapper, &inputs, &[output, output])
        .is_err());
    let mut reversed = inputs.clone();
    reversed.reverse();
    if inputs.len() > 1 {
        assert!(archive.materialize(&wrapper, &reversed, &[output]).is_err());
    }
}

#[test]
fn prepared_recipe_rejects_group_order_schema_and_code_identity_mismatch() {
    let root = tempfile::tempdir().unwrap();
    let (_, materializer, plan, schema, receipt) = exported(root.path(), account(24));
    let archive =
        PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    let mut reversed = materializer.groups.clone();
    reversed[0].reverse();
    let (_, program, _) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let other = plan::prepare_program(&schema, &program, &reversed).unwrap();
    assert!(archive
        .validate_recipe_parts(&schema, &other, &reversed)
        .is_err());
    let mut wrong_schema = schema.clone();
    wrong_schema.components[0].external_names[0].push_str(".other");
    assert!(archive.validate_plan(&wrong_schema, &plan).is_err());
    let other_id = Bf16WholeTensorFp8Materializer::new_with_id(
        WeightMaterializerId::new("weight-materializer.other").unwrap(),
        account(24),
        materializer.groups.clone(),
    )
    .unwrap();
    assert!(other_id.bind_prepared(archive).is_err());
}

#[test]
fn prepared_corruption_missing_shard_and_manifest_binding_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let (_, _, _, _, receipt) = exported(root.path(), account(24));
    assert!(PreparedFp8Archive::open(&receipt.manifest_path, &"0".repeat(64)).is_err());
    let bytes = fs::read(&receipt.manifest_path).unwrap();
    let manifest: PreparedFp8Manifest = serde_json::from_slice(&bytes).unwrap();
    let shard = receipt
        .manifest_path
        .parent()
        .unwrap()
        .join(&manifest.shards[0].file);
    let original = fs::read(&shard).unwrap();
    let mut corrupt = original.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    fs::write(&shard, corrupt).unwrap();
    assert!(PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).is_err());
    fs::write(&shard, original).unwrap();
    fs::remove_file(&shard).unwrap();
    assert!(PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).is_err());
    let mut escaped = manifest;
    escaped.shards[0].file = "../outside.safetensors".into();
    let bad = serde_json::to_vec(&escaped).unwrap();
    fs::write(&receipt.manifest_path, &bad).unwrap();
    assert!(PreparedFp8Archive::open(&receipt.manifest_path, &sha(&bad)).is_err());
}

#[test]
fn prepared_export_capacity_failure_publishes_no_manifest_and_releases_backing() {
    let root = tempfile::tempdir().unwrap();
    let (schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let source = original(root.path(), &schema);
    let plan = plan::prepare_program(&schema, &program, &groups).unwrap();
    let small = account(23);
    let materializer = Bf16WholeTensorFp8Materializer::new(small.clone(), groups.clone()).unwrap();
    let destination = root.path().join("too-small");
    let error = materializer
        .export_plan(&schema, &plan, source.as_ref(), &destination)
        .unwrap_err();
    assert!(error.to_string().contains("WeightBacking capacity"));
    assert!(!destination.join(MANIFEST_FILE).exists());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    let host = account(24);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups).unwrap();
    let destination = root.path().join("exact");
    let receipt = materializer
        .export_plan(&schema, &plan, source.as_ref(), &destination)
        .unwrap();
    assert!(receipt.manifest_path.exists());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(materializer
        .export_plan(&schema, &plan, source.as_ref(), &destination)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn prepared_export_late_source_failure_never_publishes_a_valid_archive() {
    struct FailsAfterOne<'a> {
        original: &'a SafetensorsArchive,
        fail: WeightId,
    }
    impl WeightComponentSource for FailsAfterOne<'_> {
        fn component<'a>(
            &'a self,
            spec: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            if spec.id == self.fail {
                return Err(invalid("injected late source failure"));
            }
            self.original.component(spec)
        }
    }
    let root = tempfile::tempdir().unwrap();
    let (schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let source = original(root.path(), &schema);
    let host = account(24);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups.clone()).unwrap();
    let plan = plan::prepare_program(&schema, &program, &groups).unwrap();
    let source = FailsAfterOne {
        original: &source,
        fail: schema.components[3].id.clone(),
    };
    let destination = root.path().join("interrupted");
    let error = materializer
        .export_plan(&schema, &plan, &source, &destination)
        .unwrap_err();
    assert!(error.to_string().contains("injected late source failure"));
    assert!(!destination.join(MANIFEST_FILE).exists());
    assert!(fs::read_dir(&destination).unwrap().any(|entry| entry
        .unwrap()
        .path()
        .extension()
        .is_some_and(|ext| ext == "safetensors")));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn export_resolves_once_and_prepared_initialization_never_reads_original_payloads() {
    struct CountingArchive {
        original: Arc<SafetensorsArchive>,
        calls: std::sync::Mutex<Vec<WeightId>>,
    }
    impl TensorArchive for CountingArchive {
        fn tensor(&self, name: &str) -> ferrum_types::Result<ArchiveTensor<'_>> {
            TensorArchive::tensor(self.original.as_ref(), name)
        }
        fn tensor_count(&self) -> usize {
            self.original.tensor_count()
        }
        fn tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
            TensorArchive::tensor_names(self.original.as_ref())
        }
    }
    impl WeightComponentSource for CountingArchive {
        fn component<'a>(
            &'a self,
            spec: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            self.calls.lock().unwrap().push(spec.id.clone());
            self.original.component(spec)
        }
    }
    let root = tempfile::tempdir().unwrap();
    let (schema, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let source = Arc::new(CountingArchive {
        original: original(root.path(), &schema),
        calls: std::sync::Mutex::new(Vec::new()),
    });
    let host = account(24);
    let converter = Bf16WholeTensorFp8Materializer::new(host.clone(), groups.clone()).unwrap();
    let plan = plan::prepare_program(&schema, &program, &groups).unwrap();
    let receipt = converter
        .export_plan(
            &schema,
            &plan,
            source.as_ref(),
            &root.path().join("prepared"),
        )
        .unwrap();
    let mut calls = source.calls.lock().unwrap().clone();
    calls.sort();
    let mut expected: Vec<_> = schema.components.iter().map(|s| s.id.clone()).collect();
    expected.sort();
    assert_eq!(
        calls, expected,
        "hashing/conversion must share the original resolution, including its read-ahead"
    );
    source.calls.lock().unwrap().clear();
    let archive =
        PreparedFp8Archive::open(&receipt.manifest_path, &receipt.manifest_sha256).unwrap();
    let bound = converter.bind_prepared(archive.clone()).unwrap();
    let wrapper = PreparedFp8TensorArchive::new(source.clone(), archive);
    let exact = ExactBf16TensorSource::new(&wrapper);
    let originals: BTreeMap<_, _> = schema.components.iter().map(|s| (&s.id, s)).collect();
    for _ in 0..2 {
        for output in &plan.schema.components {
            let inputs: Vec<_> = plan.sources[&output.id]
                .iter()
                .map(|id| originals[id])
                .collect();
            let payloads = bound
                .materialize_components(&exact, &inputs, &[output])
                .unwrap();
            assert_eq!(
                payloads[0].bytes().len() as u64,
                output.physical_bytes().unwrap()
            );
        }
    }
    assert!(
        source.calls.lock().unwrap().is_empty(),
        "prepared initialization must not read any original matrix or repeat advice"
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
