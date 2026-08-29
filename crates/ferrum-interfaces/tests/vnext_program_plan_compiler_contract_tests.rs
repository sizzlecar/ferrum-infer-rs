mod vnext_core_contract;

use vnext_core_contract::*;

fn prepared_with_weight_schema(schema: WeightSchema) -> PreparedModelFamily {
    TypedFamilyRegistration::new(FixedSchemaFamily { schema })
        .prepare(&json!({"width": 4}))
        .unwrap()
}

fn compile_options() -> ProgramPlanCompileOptions {
    ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap()
}

#[test]
fn determinism_compile_option_retains_every_operation_output() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut options = compile_options();
    options.retain_all_outputs_for_determinism(&family).unwrap();
    let expected = family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .flat_map(|node| node.outputs.iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(options.completion_retention().values(), &expected);

    let planning = registry.planning();
    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy(4096), &planning, &options)
            .unwrap();
    let retained = compilation
        .executable()
        .execution_plan()
        .payload()
        .retained_completion_values()
        .iter()
        .map(|value| value.value_id().clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(retained, expected);
}

struct PaddedDenseMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl PaddedDenseMaterializer {
    fn new() -> Self {
        Self::with_implementation_fingerprint('9')
    }

    fn with_implementation_fingerprint(fingerprint_byte: char) -> Self {
        Self::with_fidelity(fingerprint_byte, WeightMaterializationFidelity::Exact)
    }

    fn with_fidelity(fingerprint_byte: char, fidelity: WeightMaterializationFidelity) -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.padded-dense"),
                ContractVersion::new(1, 0),
                sha(fingerprint_byte),
                fidelity,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

fn test_approximate_quality_contract() -> ApproximateWeightQualityContract {
    ApproximateWeightQualityContract::new(
        sha('a'),
        sha('b'),
        4,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .unwrap()
}

fn little_endian_u16_sha256(values: &[u16]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn little_endian_u32_sha256(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn test_quality_vector_payload() -> Value {
    let reference_digest = little_endian_u32_sha256(&[1_f32.to_bits()]);
    json!({
        "activation_batches": [1, 4],
        "activation_contract": {"dtype": "F16"},
        "cases": (0..4).map(|case| json!({
            "case_id": format!("numeric-case-{case}"),
            "reference_f32le_sha256": reference_digest
        })).collect::<Vec<_>>(),
        "checkpoint": {
            "id": "checkpoint.synthetic-fp8",
            "repository": "ferrum/synthetic-fp8",
            "revision": "cccccccccccccccccccccccccccccccccccccccc"
        },
        "fixture_id": "synthetic-numeric-vector-v1",
        "generator": {"algorithm": "constant-v1"},
        "reference_contract": {"output_dtype": "F32 little-endian"},
        "schema_version": 1,
        "source_contract": {"values_dtype": "F16"},
        "weight_shapes": [[1, 1], [1, 1]]
    })
}

fn test_numeric_quality_contract() -> ApproximateWeightQualityContract {
    let quality_vector_digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&test_quality_vector_payload()).unwrap())
    );
    ApproximateWeightQualityContract::new(
        sha('a'),
        quality_vector_digest,
        4,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .unwrap()
}

fn test_numeric_quality_artifact(descriptor: &WeightMaterializerDescriptor) -> Vec<u8> {
    let actual = vec![0x3c00_u16];
    let reference = vec![1_f32.to_bits()];
    let cases = (0..4)
        .map(|case| {
            json!({
                "actual_f16_bits": actual,
                "actual_f16le_sha256": little_endian_u16_sha256(&actual),
                "case_id": format!("numeric-case-{case}"),
                "inf_count": 0,
                "nan_count": 0,
                "reference_f32_bits": reference,
                "reference_f32le_sha256": little_endian_u32_sha256(&reference),
                "relative_l2_upper_bound": {"denominator": 1, "numerator": 0}
            })
        })
        .collect::<Vec<_>>();
    let artifact = json!({
        "authority": {
            "id": NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID,
            "implementation_fingerprint": numeric_weight_quality_authority_implementation_fingerprint().unwrap(),
            "version": {"major": 1, "minor": 0}
        },
        "cases": cases,
        "checkpoint": {
            "id": "checkpoint.synthetic-fp8",
            "repository": "ferrum/synthetic-fp8",
            "revision": "cccccccccccccccccccccccccccccccccccccccc"
        },
        "contract": {
            "execution_contract_fingerprint": descriptor.approximate_quality_contract().unwrap().execution_contract_fingerprint(),
            "quality_vector_digest": descriptor.approximate_quality_contract().unwrap().quality_vector_digest()
        },
        "execution": {
            "quantization_format_ids": [],
            "weight_format_id": "weight-format.dense",
            "weight_layout_id": "weight-layout.test.padded-dense"
        },
        "materializer": {
            "fidelity": "approximate",
            "id": descriptor.id(),
            "implementation_fingerprint": descriptor.implementation_fingerprint(),
            "version": descriptor.version()
        },
        "quality_vector_payload": test_quality_vector_payload(),
        "schema_id": NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID,
        "source": {"weight_format_id": "weight-format.dense"}
    });
    serde_json::to_vec(&artifact).unwrap()
}

impl WeightMaterializer for PaddedDenseMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.layout_id = id("weight-layout.test.padded-dense");
        schema.components[0].dimensions = vec![8];
        schema.tensors[0].physical_layout = PhysicalWeightLayout::Stored {
            component: PhysicalWeightComponentBinding {
                component_id: id("weight.component"),
                storage: PhysicalStorageLayout::Contiguous {
                    padding: PhysicalWeightPadding::ZeroFill {
                        padded_dimensions: vec![8],
                    },
                },
            },
        };
        Ok(schema)
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [source_component] = source_components else {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "padded test materializer requires one source component".to_owned(),
            });
        };
        let source_payload = source.component(source_component)?;
        let output_bytes =
            usize::try_from(execution_component.physical_bytes()?).map_err(|_| {
                VNextError::InvalidExecutionPlan {
                    reason: "padded test component exceeds host address space".to_owned(),
                }
            })?;
        let mut bytes = source_payload.bytes().to_vec();
        bytes.resize(output_bytes, 0);
        WeightComponentPayload::from_ordered_sources(
            execution_component,
            execution_component.external_names.clone(),
            source_payload.source_files().to_vec(),
            execution_component.dimensions.clone(),
            execution_component.physical_element_type(),
            bytes,
        )
    }
}

struct LogicalMutationMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl LogicalMutationMaterializer {
    fn new() -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.logical-mutation"),
                ContractVersion::new(1, 0),
                sha('7'),
                WeightMaterializationFidelity::Exact,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

impl WeightMaterializer for LogicalMutationMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.components[0].dimensions = vec![5];
        schema.tensors[0].dimensions = vec![5];
        Ok(schema)
    }

    fn materialize_component<'source>(
        &self,
        _source: &'source dyn WeightComponentSource,
        _source_components: &[&WeightComponentSpec],
        _execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        Err(VNextError::InvalidExecutionPlan {
            reason: "logically invalid test materializer must never initialize weights".to_owned(),
        })
    }
}

#[test]
fn semantic_program_compiles_through_the_registered_provider_authority() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let plan = compilation.executable().execution_plan();
    assert_eq!(plan.payload().nodes().len(), 1);
    assert!(plan.payload().retained_completion_values().is_empty());
    assert_eq!(
        plan.payload().execution_weights().schema(),
        family.weight_schema()
    );
    assert_eq!(
        plan.payload()
            .execution_weights()
            .source_schema_fingerprint(),
        family.weight_schema().fingerprint().unwrap()
    );
    assert_eq!(
        plan.payload()
            .execution_weights()
            .materializer_id()
            .as_str(),
        "weight-materializer.identity"
    );
    assert!(plan
        .payload()
        .execution_weights()
        .approximate_quality_approval()
        .is_none());
    assert!(serde_json::to_value(plan.payload().execution_weights())
        .unwrap()
        .get("approximate_quality_approval")
        .is_none());
    assert_eq!(compilation.node_resolutions().len(), 1);
    assert_eq!(
        compilation
            .value_tensors()
            .get(&id("value.output"))
            .unwrap()
            .dimensions(),
        &[4]
    );

    let node = &plan.payload().nodes()[0];
    let weight = node
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    let resolved_weight = weight
        .weight()
        .expect("provider binding must retain the physical weight contract");
    assert_eq!(resolved_weight.weight_id(), &id("weight.matrix"));
    assert_eq!(
        serde_json::to_value(resolved_weight).unwrap()["format_id"],
        json!("weight-format.dense")
    );
    assert_eq!(resolved_weight.layout_id(), &id("weight-layout.dense"));
    assert_eq!(resolved_weight.components().len(), 1);
    assert_eq!(resolved_weight.components()[0].physical_dimensions(), &[4]);
    assert_eq!(weight.storage().components().len(), 1);
    let component = &weight.storage().components()[0];
    assert!(component
        .resource_id()
        .as_str()
        .starts_with("resource/weight-arena/sha256/"));
    assert_eq!(
        component.offset_bytes() % node.provider_resources().value_alignment_bytes(),
        0
    );
    assert!(registry.estimator_calls.load(Ordering::SeqCst) >= 2);
}

#[test]
fn approximate_materializer_is_not_authorized_by_capability_registration() {
    let family = TestRegistry::new().prepare();
    let materializer_id: WeightMaterializerId = id("weight-materializer.test.padded-dense");
    let descriptor = WeightMaterializerDescriptor::new(
        id("weight-materializer.test.padded-dense"),
        ContractVersion::new(1, 0),
        sha('9'),
        WeightMaterializationFidelity::Approximate,
        BTreeSet::from([id("capability.compute")]),
    )
    .unwrap()
    .with_approximate_quality_contract(test_approximate_quality_contract())
    .unwrap();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer { descriptor })])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut options = compile_options();
    options.require_weight_materializer(materializer_id.clone());

    assert_eq!(
        catalog
            .weight_materializer(&materializer_id)
            .unwrap()
            .fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    assert!(catalog
        .weight_materializer(&materializer_id)
        .unwrap()
        .approximate_quality_contract()
        .is_some());
    assert_eq!(
        serde_json::to_value(catalog.weight_materializer(&materializer_id).unwrap()).unwrap()
            ["fidelity"],
        json!("approximate")
    );
    let error = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        VNextError::WeightMaterializerQualityApprovalRequired { materializer_id }
            if materializer_id == "weight-materializer.test.padded-dense"
    ));
}

#[test]
fn canonical_numeric_artifact_mints_a_live_schema_bound_approval() {
    let family = TestRegistry::new().prepare();
    let descriptor =
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate)
            .descriptor
            .with_approximate_quality_contract(test_numeric_quality_contract())
            .unwrap();
    let artifact = test_numeric_quality_artifact(&descriptor);
    let materializer_id = descriptor.id().clone();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer { descriptor })])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mut options = compile_options();
    options
        .require_weight_materializer_with_numeric_quality_artifact(
            materializer_id.clone(),
            artifact.clone(),
        )
        .unwrap();

    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap();
    let weights = compilation
        .executable()
        .execution_plan()
        .payload()
        .execution_weights();
    assert_eq!(weights.materializer_id(), &materializer_id);
    let approval = weights
        .approximate_quality_approval()
        .expect("approximate plan must retain its verified receipt");
    assert_eq!(approval.authority_id(), NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID);
    assert_eq!(approval.completed_case_count(), 4);
    assert_eq!(approval.relative_l2_max_observed().numerator(), 0);
    assert_eq!(approval.nan_count(), 0);
    assert_eq!(approval.inf_count(), 0);
    assert_eq!(
        approval.source_schema_fingerprint(),
        family.weight_schema().fingerprint().unwrap()
    );
    assert_eq!(
        approval.execution_schema_fingerprint(),
        weights.schema().fingerprint().unwrap()
    );
    assert_eq!(
        approval.artifact_sha256(),
        format!("{:x}", Sha256::digest(&artifact))
    );

    let selection =
        WeightMaterializerSelection::numeric_quality_artifact(materializer_id, artifact).unwrap();
    assert!(selection.has_numeric_quality_artifact());
    assert_eq!(options.weight_materializer_selection(), &selection);
}

#[test]
fn numeric_artifact_is_reusable_but_each_approval_binds_the_live_schema() {
    let first_family = TestRegistry::new().prepare();
    let mut second_source_schema = first_family.weight_schema().clone();
    second_source_schema.components[0].external_names = vec!["alternate-weight.bin".to_owned()];
    let second_family = prepared_with_weight_schema(second_source_schema);
    let descriptor =
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate)
            .descriptor
            .with_approximate_quality_contract(test_numeric_quality_contract())
            .unwrap();
    let selection = WeightMaterializerSelection::numeric_quality_artifact(
        descriptor.id().clone(),
        test_numeric_quality_artifact(&descriptor),
    )
    .unwrap();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer { descriptor })])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();

    let first = materializers
        .select(&first_family, &catalog, &selection)
        .unwrap();
    let second = materializers
        .select(&second_family, &catalog, &selection)
        .unwrap();
    let first_approval = first.plan().approximate_quality_approval().unwrap();
    let second_approval = second.plan().approximate_quality_approval().unwrap();
    assert_eq!(
        first_approval.artifact_sha256(),
        second_approval.artifact_sha256()
    );
    assert_ne!(
        first_approval.source_schema_fingerprint(),
        second_approval.source_schema_fingerprint()
    );
    assert_ne!(
        first_approval.execution_schema_fingerprint(),
        second_approval.execution_schema_fingerprint()
    );
}

#[test]
fn numeric_artifact_parser_is_canonical_bounded_and_recomputes_raw_evidence() {
    let family = TestRegistry::new().prepare();
    let descriptor =
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate)
            .descriptor
            .with_approximate_quality_contract(test_numeric_quality_contract())
            .unwrap();
    let materializer_id = descriptor.id().clone();
    let artifact = test_numeric_quality_artifact(&descriptor);
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer { descriptor })])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();

    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer_id.clone(),
        b"{}".to_vec(),
    )
    .is_err());

    let pretty =
        serde_json::to_vec_pretty(&serde_json::from_slice::<Value>(&artifact).unwrap()).unwrap();
    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer_id.clone(),
        pretty.clone(),
    )
    .is_err());
    assert!(materializers
        .select_with_numeric_quality_artifact(&family, &catalog, &materializer_id, &pretty,)
        .unwrap_err()
        .to_string()
        .contains("canonical compact JSON"));

    let mut unknown = serde_json::from_slice::<Value>(&artifact).unwrap();
    unknown["cases"][0]["untrusted_override"] = json!(true);
    assert!(WeightMaterializerSelection::numeric_quality_artifact(
        materializer_id.clone(),
        serde_json::to_vec(&unknown).unwrap(),
    )
    .is_err());
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&unknown).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("strict schema-valid JSON"));

    let mut changed_raw = serde_json::from_slice::<Value>(&artifact).unwrap();
    changed_raw["cases"][0]["actual_f16_bits"][0] = json!(0_u16);
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&changed_raw).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("raw-vector digest differs"));

    let mut changed_vector = serde_json::from_slice::<Value>(&artifact).unwrap();
    changed_vector["quality_vector_payload"]["fixture_id"] = json!("forged-vector");
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&changed_vector).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("locked quality vector payload"));

    let mut changed_reference = serde_json::from_slice::<Value>(&artifact).unwrap();
    changed_reference["cases"][0]["reference_f32_bits"][0] = json!(0_u32);
    changed_reference["cases"][0]["reference_f32le_sha256"] = json!(little_endian_u32_sha256(&[0]));
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&changed_reference).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("reference differs from the locked quality vector"));

    let mut false_non_finite_count = serde_json::from_slice::<Value>(&artifact).unwrap();
    false_non_finite_count["cases"][0]["actual_f16_bits"][0] = json!(0x7c00_u16);
    false_non_finite_count["cases"][0]["actual_f16le_sha256"] =
        json!(little_endian_u16_sha256(&[0x7c00]));
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&false_non_finite_count).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("reports incorrect NaN or Inf counts"));

    let mut understated = serde_json::from_slice::<Value>(&artifact).unwrap();
    understated["cases"][0]["actual_f16_bits"][0] = json!(0_u16);
    understated["cases"][0]["actual_f16le_sha256"] = json!(little_endian_u16_sha256(&[0]));
    assert!(materializers
        .select_with_numeric_quality_artifact(
            &family,
            &catalog,
            &materializer_id,
            &serde_json::to_vec(&understated).unwrap(),
        )
        .unwrap_err()
        .to_string()
        .contains("understates its relative-L2"));

    let oversized = vec![b'x'; MAX_APPROXIMATE_WEIGHT_QUALITY_ARTIFACT_BYTES + 1];
    assert!(
        WeightMaterializerSelection::numeric_quality_artifact(materializer_id, oversized,).is_err()
    );
}

#[test]
fn approximate_quality_contract_is_policy_metadata_not_an_approval() {
    let contract = test_approximate_quality_contract();
    assert_eq!(contract.execution_contract_fingerprint(), sha('a'));
    assert_eq!(contract.quality_vector_digest(), sha('b'));
    assert_eq!(contract.required_case_count(), 4);
    assert_eq!(
        contract.relative_l2_max(),
        CanonicalRational::new(1, 20).unwrap()
    );
    assert_eq!(contract.nan_count_max(), 0);
    assert_eq!(contract.inf_count_max(), 0);

    let exact = WeightMaterializerDescriptor::new(
        id("weight-materializer.test.exact"),
        ContractVersion::new(1, 0),
        sha('c'),
        WeightMaterializationFidelity::Exact,
        BTreeSet::new(),
    )
    .unwrap();
    assert!(exact
        .with_approximate_quality_contract(contract.clone())
        .is_err());
    assert!(ApproximateWeightQualityContract::new(
        "not-a-digest",
        sha('b'),
        4,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .is_err());
    assert!(ApproximateWeightQualityContract::new(
        sha('a'),
        sha('b'),
        0,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .is_err());
}

#[test]
fn exact_materializer_descriptor_keeps_legacy_wire_and_fingerprint() {
    const LEGACY_JSON: &str = r#"{"fidelity":"exact","id":"weight-materializer.test.padded-dense","implementation_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","required_capabilities":["capability.compute"],"version":{"major":1,"minor":0}}"#;
    const LEGACY_FINGERPRINT: &str =
        "667790b7aded7556b9a628eadc203737f350c8d6fd642f5fa4a26706ecf723b4";

    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer::new())]).unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let descriptor = catalog
        .weight_materializer(&id("weight-materializer.test.padded-dense"))
        .unwrap();
    assert!(descriptor.approximate_quality_contract().is_none());
    assert_eq!(
        serde_json::to_value(descriptor).unwrap(),
        serde_json::from_str::<serde_json::Value>(LEGACY_JSON).unwrap()
    );
    assert_eq!(descriptor.fingerprint().unwrap(), LEGACY_FINGERPRINT);
    assert_eq!(
        serde_json::from_str::<WeightMaterializerDescriptor>(LEGACY_JSON).unwrap(),
        *descriptor
    );
}

#[test]
fn trusted_materializer_changes_physical_plan_memory_and_wire_requires_its_witness() {
    let family = TestRegistry::new().prepare();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer::new())]).unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let identity = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy,
        &registry.planning(),
        &compile_options(),
    )
    .unwrap();
    let mut options = compile_options();
    let materializer_id: WeightMaterializerId = id("weight-materializer.test.padded-dense");
    options.require_weight_materializer(materializer_id.clone());
    let transformed = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy,
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap();
    let plan = transformed.executable().execution_plan();
    assert_eq!(
        plan.payload().execution_weights().materializer_id(),
        &materializer_id
    );
    assert_eq!(
        plan.payload().execution_weights().schema().layout_id,
        id("weight-layout.test.padded-dense")
    );
    assert_eq!(
        plan.payload().memory().static_bytes(),
        identity
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .static_bytes()
            + 16
    );
    plan.validate_against(&family, &catalog, &policy, transformed.node_resolutions())
        .unwrap();

    let wire = plan.to_json().unwrap();
    assert!(ExecutionPlan::from_json_validated(
        &wire,
        &family,
        &catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .is_err());
    let trusted_weights = materializers
        .select_exact(&family, &catalog, &materializer_id)
        .unwrap();
    let mismatched_materializers = WeightMaterializerRegistry::new(vec![Box::new(
        PaddedDenseMaterializer::with_implementation_fingerprint('6'),
    )])
    .unwrap();
    let mismatched_catalog = mismatched_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &mismatched_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let capability_mutated_materializers =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer {
            descriptor: WeightMaterializerDescriptor::new(
                materializer_id.clone(),
                ContractVersion::new(1, 0),
                sha('9'),
                WeightMaterializationFidelity::Exact,
                BTreeSet::new(),
            )
            .unwrap(),
        })])
        .unwrap();
    let capability_mutated_catalog = capability_mutated_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &capability_mutated_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let fidelity_mutated_materializers = WeightMaterializerRegistry::new(vec![Box::new(
        PaddedDenseMaterializer::with_fidelity('9', WeightMaterializationFidelity::Approximate),
    )])
    .unwrap();
    let fidelity_mutated_catalog = fidelity_mutated_materializers
        .augment_catalog(vnext_core_contract::catalog())
        .unwrap();
    assert!(PlanBuildRequest::new(
        &family,
        &fidelity_mutated_catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
    )
    .unwrap()
    .with_execution_weights(trusted_weights.clone())
    .is_err());
    let restored = ExecutionPlan::from_json_validated_with_execution_weights(
        &wire,
        &family,
        &catalog,
        &policy,
        transformed.node_resolutions().to_vec(),
        CompletionRetentionSpec::default(),
        trusted_weights,
    )
    .unwrap();
    assert_eq!(restored, *plan);
}

#[test]
fn materializer_cannot_change_the_prepared_logical_weight_contract() {
    let family = TestRegistry::new().prepare();
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(LogicalMutationMaterializer::new())])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let error = materializers
        .select_exact(
            &family,
            &catalog,
            &id("weight-materializer.test.logical-mutation"),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changes the prepared family's logical tensor contract"),
        "{error}"
    );
}

#[test]
fn dense_binding_in_mixed_checkpoint_does_not_require_the_container_format() {
    let mut schema = TestFamily.weight_schema(&TestConfig { width: 4 }).unwrap();
    schema.format_id = id("weight-format.safetensors.mixed-gptq");
    schema.layout_id = id("weight-layout.synthetic.mixed-gptq");
    let family = prepared_with_weight_schema(schema);
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &compile_options(),
    )
    .unwrap();
    let weight = compilation.executable().execution_plan().payload().nodes()[0]
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    assert_eq!(
        serde_json::to_value(weight.weight().unwrap()).unwrap()["format_id"],
        json!("weight-format.safetensors.mixed-gptq")
    );
}

#[test]
fn quantized_binding_still_requires_its_format_and_abi_before_allocation() {
    let schema = WeightSchema {
        format_id: id("weight-format.synthetic.quantized"),
        layout_id: id("weight-layout.synthetic.quantized"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("weight.component"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["weight.bin".to_owned()],
            dimensions: vec![1],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id("quantization.synthetic.int4"),
                logical_values_per_block: 4,
                bytes_per_block: 2,
            }),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.matrix"),
            dimensions: vec![4],
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(id("weight.component")),
                block_axis: 0,
                block_padding: PhysicalWeightPadding::Exact,
            },
            required: true,
        }],
    };
    let family = prepared_with_weight_schema(schema);
    let catalog = catalog();
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);

    let error = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(4096),
        &registry.planning(),
        &compile_options(),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("weight-format.synthetic.quantized"),
        "{message}"
    );
    assert!(message.contains("quantization.synthetic.int4"), "{message}");
}

#[test]
fn completion_retention_binds_one_typed_output_and_requires_expected_wire_policy() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    assert!(options.retain_completion_value(id("value.output")));

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let plan = compilation.executable().execution_plan();
    let checkpoint = plan.completion_checkpoint(&id("value.output")).unwrap();
    assert_eq!(checkpoint.producer_node_id(), &id("node.main"));
    assert_eq!(checkpoint.output_ordinal(), 0);
    assert_eq!(checkpoint.tensor().dimensions(), &[4]);
    assert_eq!(
        plan.payload().terminal_output_resources(),
        std::slice::from_ref(checkpoint.resource_id())
    );
    let output_descriptor = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id() == checkpoint.resource_id())
        .unwrap();
    assert_eq!(output_descriptor.lifetime(), AllocationLifetime::Step);
    assert!(matches!(
        output_descriptor.demand(),
        DynamicResourceDemand::ActualSequences {
            bytes_per_sequence: 16,
            maximum_sequences: 3,
        }
    ));
    let readback = checkpoint
        .readback_request(3, HostTransferLayout::new(ElementType::F32, 4).unwrap())
        .unwrap();
    assert_eq!(readback.node_id(), checkpoint.producer_node_id());
    assert_eq!(readback.resource_id(), checkpoint.resource_id());
    assert_eq!(readback.participant_index(), 3);
    let work_readback = plan
        .completion_checkpoint_readback_for_work(&id("value.output"), 3, &resource_work(&[2]))
        .unwrap();
    assert_eq!(work_readback.output_layout().element_count(), 4);
    assert!(checkpoint
        .readback_request(0, HostTransferLayout::new(ElementType::U8, 4).unwrap())
        .unwrap_err()
        .to_string()
        .contains("element type differs"));
    assert!(checkpoint
        .readback_request(0, HostTransferLayout::new(ElementType::F32, 5).unwrap())
        .unwrap_err()
        .to_string()
        .contains("exceeds retained activation capacity"));

    let wire = plan.to_json().unwrap();
    assert!(ExecutionPlan::from_json_validated(
        &wire,
        &family,
        &catalog,
        &policy,
        compilation.node_resolutions().to_vec(),
    )
    .is_err());
    let restored = ExecutionPlan::from_json_validated_with_completion_retention(
        &wire,
        &family,
        &catalog,
        &policy,
        compilation.node_resolutions().to_vec(),
        CompletionRetentionSpec::new(BTreeSet::from([id("value.output")])),
    )
    .unwrap();
    assert_eq!(restored, *plan);

    let mut forged = serde_json::from_slice::<Value>(&wire).unwrap();
    forged["payload"]["retained_completion_values"][0]["resource_id"] =
        json!("resource/forged-retained-output");
    rehash_plan_json(&mut forged);
    let forged = serde_json::to_vec(&forged).unwrap();
    assert!(
        ExecutionPlan::from_json_validated_with_completion_retention(
            &forged,
            &family,
            &catalog,
            &policy,
            compilation.node_resolutions().to_vec(),
            CompletionRetentionSpec::new(BTreeSet::from([id("value.output")])),
        )
        .is_err()
    );
}

#[test]
fn completion_retention_rejects_inputs_weights_and_unknown_values_before_planning() {
    for value_id in ["value.input", "value.weight", "value.unknown"] {
        let family = TestRegistry::new().prepare();
        let catalog = catalog();
        let policy = policy(4096);
        let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
        let planning = registry.planning();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.input"),
            ProgramTensorSpec {
                dimensions: vec![4],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.retain_completion_value(id(value_id));

        let error = ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("completion retention must reference a semantic node output"),
            "{value_id}: {error}"
        );
    }
}

#[test]
fn compilation_rejects_missing_or_guessed_product_input_capacity() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::new()).unwrap();
    let error =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap_err();
    assert!(error
        .to_string()
        .contains("every program input requires an explicit canonical tensor capacity"));
}

#[test]
fn compilation_reports_the_exact_tensor_binding_on_signature_mismatch() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::U32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let error =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("input[0] `value.input`"), "{message}");
    assert!(message.contains("dtype=U32"), "{message}");
}

#[test]
fn weight_arena_reaches_provider_alignment_fixed_point() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let registry = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::ArenaAlignment64);
    let planning = registry.planning();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();

    let compilation =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &planning, &options).unwrap();
    let node = &compilation.executable().execution_plan().payload().nodes()[0];
    assert_eq!(node.provider_resources().value_alignment_bytes(), 64);
    assert!(registry.estimator_calls.load(Ordering::SeqCst) >= 3);
    let weight = node
        .values()
        .iter()
        .find(|binding| binding.usage() == BufferUsage::Weights)
        .unwrap();
    assert!(weight
        .storage()
        .components()
        .iter()
        .all(|component| component.offset_bytes() % 64 == 0));
}
