//! Independent LayerNorm fixtures after an actual managed BF16 projection.
//! No dummy weights: the dense prefix either projects two different rows or
//! projects a scalar basis into the exact normalization input under test.

use super::*;
use ferrum_runtime::ModelWeights;
use half::bf16;

#[path = "layer_norm_oracle.rs"]
mod oracle;

const NORMALIZED: &str = "value.tensor.normalized";
const NORM_PROFILE: &str = "reference.tensor.dense-layer-norm.bf16";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NormConfig {
    rows: u64,
    in_features: u64,
    hidden_size: u64,
    epsilon: CanonicalRational,
}

struct DenseNormFamily;

impl ModelFamilyProvider for DenseNormFamily {
    type Config = NormConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.reference.tensor.dense-layer-norm"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tensor.dense-layer-norm")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &NormConfig) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw
            || [config.rows, config.in_features, config.hidden_size].contains(&0)
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: "normalization fixture requires canonical positive dimensions".to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &NormConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.reference.tensor.dense-layer-norm"))
    }

    fn parse_config(&self, raw: &Value) -> Result<NormConfig, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weights(&self, config: &NormConfig) -> Result<ModelWeights, VNextError> {
        TinyTensorFamily(TinyTensorPrecision::Bf16).weights(&TinyTensorConfig {
            rows: config.rows,
            in_features: config.in_features,
            out_features: config.hidden_size,
        })
    }

    fn numerical_profiles(&self, _: &NormConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(NORM_PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(NORMALIZED),
                boundaries: BTreeMap::from([
                    (id(OUTPUT_ID), ElementType::Bf16),
                    (id(NORMALIZED), ElementType::Bf16),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: [DENSE_LINEAR_BF16_OPERATION_ID, LAYER_NORM_BF16_OPERATION_ID]
                    .into_iter()
                    .map(|operation| NumericalOperationContract {
                        operation_id: id(operation),
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    })
                    .collect(),
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        config: &NormConfig,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let dense = TinyTensorFamily(TinyTensorPrecision::Bf16).semantic_program(
            &TinyTensorConfig {
                rows: config.rows,
                in_features: config.in_features,
                out_features: config.hidden_size,
            },
            profile,
        )?;
        let mut blocks = dense.blocks().to_vec();
        blocks[0].nodes.push(ProgramNode {
            id: id("node.tensor.layer-norm"),
            operation_id: id(LAYER_NORM_BF16_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id(OUTPUT_ID)],
            outputs: vec![id(NORMALIZED)],
            attributes: BTreeMap::from([
                (
                    id("hidden_size"),
                    SemanticValue::Unsigned(config.hidden_size),
                ),
                (id("epsilon"), SemanticValue::Rational(config.epsilon)),
            ]),
        });
        ModelProgram::new(
            self.family_id().clone(),
            dense.inputs().to_vec(),
            blocks,
            vec![],
            dense.weights().to_vec(),
            vec![id(NORMALIZED)],
        )
    }

    fn semantic_metadata(&self, _: &NormConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn raw_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

fn fixture(config: &NormConfig, weights: &[u16], host: HostMemoryAccount) -> TensorFixture {
    let directory = tempfile::tempdir().unwrap();
    let weight_bytes = raw_bytes(weights);
    let view = TensorView::new(
        Dtype::BF16,
        vec![config.hidden_size as usize, config.in_features as usize],
        &weight_bytes,
    )
    .unwrap();
    serialize_to_file(
        vec![("linear.weight".to_owned(), view)],
        &None::<HashMap<String, String>>,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = SafetensorsArchive::open(directory.path()).unwrap();
    let family = TypedFamilyRegistration::new(DenseNormFamily)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(NORM_PROFILE).unwrap(),
        )
        .unwrap();
    assert_eq!(family.metadata(), &ModelSemanticMetadata::TensorComponent);
    assert!(family.program().states().is_empty());
    assert!(family.program().checkpoint_inputs().is_none());
    assert!(family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .all(|node| node.work == ProgramNodeWorkSpec::Fixed));
    let composition =
        ReferenceVNextComposition::create(id("device.reference.tensor.layer-norm")).unwrap();
    let policy = TensorPolicy {
        capacity_bytes: composition.runtime().descriptor().total_memory_bytes,
        storage_profiles: composition
            .runtime()
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(INPUT_ID),
        ProgramTensorSpec {
            dimensions: vec![config.rows, config.in_features],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    options.require_weight_materializer(composition.weight_materializer_id().clone());
    options.retain_completion_value(id(NORMALIZED));
    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    let providers = composition
        .registry()
        .bind_plan(compilation.executable())
        .unwrap();
    let resources = initialize_resources_with_weight_bytes(
        Arc::clone(composition.runtime()),
        &family,
        compilation.executable().execution_plan(),
        &source,
        weight_bytes.len() as u64,
    );
    TensorFixture {
        composition,
        compilation,
        providers,
        resources,
        host,
        _source_directory: directory,
    }
}

fn run(config: &NormConfig, inputs: &[u16], weights: &[u16]) -> Vec<u16> {
    let input_bytes = inputs.len() * 2;
    let output_bytes = (config.rows * config.hidden_size * 2) as usize;
    // Only input and final output are Host tensors. The dense intermediate
    // must be part of the compiled device plan, not a detached Host copy.
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            input_bytes + output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                transfer_staging_bytes: 0,
                output_bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let fixture = fixture(config, weights, host.clone());
    let mut executor = executor(&fixture);
    let mut input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, input_bytes)
        .unwrap();
    input.as_mut().copy_from_slice(&raw_bytes(inputs));
    let outputs = executor
        .execute(
            id("run.tensor.dense-layer-norm"),
            id("request.tensor.dense-layer-norm"),
            vec![ManagedHostTensor::new(
                id(INPUT_ID),
                input_spec(vec![config.rows, config.in_features], ElementType::Bf16),
                input.freeze(),
            )
            .unwrap()],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(NORMALIZED));
    assert_eq!(
        outputs[0].tensor().dimensions(),
        [config.rows, config.hidden_size]
    );
    assert_eq!(outputs[0].tensor().element_type(), ElementType::Bf16);
    assert!(host.owns(outputs[0].bytes()));
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.dense_linear_launches, 1);
    assert_eq!(snapshot.layer_norm_launches, 1);
    assert_eq!(
        snapshot.uploaded_bytes,
        (weights.len() * 2 + input_bytes) as u64
    );
    assert_eq!(snapshot.readback_bytes, output_bytes as u64);
    drop(executor);
    close_fixture(fixture);
    let bits = outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    bits
}

fn projection_probe(input: &[u16], epsilon: CanonicalRational) -> Vec<u16> {
    run(
        &NormConfig {
            rows: 1,
            in_features: 1,
            hidden_size: input.len() as u64,
            epsilon,
        },
        &[0x3f80],
        input,
    )
}

#[test]
fn dense_then_layer_norm_keeps_intermediate_bf16_and_managed_lifetimes() {
    let config = NormConfig {
        rows: 2,
        in_features: 3,
        hidden_size: 2,
        epsilon: CanonicalRational::new(3, 4).unwrap(),
    };
    // Independent dense results [[1,2],[2,1]], then mean=3/2,
    // variance=1/4 and sqrt(variance+epsilon)=1, exactly in FP32.
    assert_eq!(
        run(
            &config,
            &bits(&[1., 2., 0., 2., 1., 0.]),
            &bits(&[1., 0., 0., 0., 1., 0.])
        ),
        [0xbf00, 0x3f00, 0x3f00, 0xbf00]
    );
}

#[test]
fn dense_output_rounding_is_a_real_boundary_before_layer_norm() {
    let config = NormConfig {
        rows: 1,
        in_features: 2,
        hidden_size: 2,
        epsilon: CanonicalRational::new(4095, 65536).unwrap(),
    };
    // First dot 1+2^-8 is exactly halfway and rounds to even BF16 1.
    // Second dot is BF16 1+2^-7. Thus mu=1+2^-8, v=2^-16,
    // v+eps=1/16, and outputs are exactly +/- 2^-6. Keeping the first
    // dense dot in FP32 would instead roughly halve both normalized values.
    assert_eq!(
        run(
            &config,
            &[0x3f80, 0x3f80],
            &[0x3f80, 0x3b80, 0x3f80, 0x3c00]
        ),
        [0xbc80, 0x3c80]
    );
}

#[test]
fn layer_norm_centers_shifted_rows_and_places_epsilon_inside_square_root() {
    let eps = CanonicalRational::new(3, 4).unwrap();
    for input in [&[1., 1., 2., 2.][..], &[128., 128., 129., 129.][..]] {
        assert_eq!(
            projection_probe(&bits(input), eps),
            [0xbf00, 0xbf00, 0x3f00, 0x3f00]
        );
    }
    assert_eq!(
        projection_probe(
            &bits(&[-2_f32.powi(-10), 2_f32.powi(-10)]),
            CanonicalRational::new(3, 1 << 20).unwrap()
        ),
        [0xbf00, 0x3f00]
    );
}

#[test]
fn layer_norm_constant_and_nearly_constant_4096_width_have_exact_bits() {
    for width in [1, 3, 4096] {
        assert_eq!(
            projection_probe(
                &vec![0x4300; width],
                CanonicalRational::new(1, 1_000_000).unwrap()
            ),
            vec![0; width]
        );
    }
    let mut input = vec![0x3f80; 4096];
    input[4095] = 0x3f81;
    // mu=1+2^-19, v=4095*2^-38; this epsilon makes v+eps=2^-20.
    // All statistics are exactly representable with the declared pairwise
    // FP32 evaluation. The final 4095/512 rounds to BF16 8.
    let actual = projection_probe(&input, CanonicalRational::new(258049, 1 << 38).unwrap());
    assert_eq!(&actual[..4095], &vec![0xbb00; 4095]);
    assert_eq!(actual[4095], 0x4100);
}

#[test]
fn layer_norm_matches_independent_f64_oracle_with_precomputed_error_bounds() {
    let epsilon = CanonicalRational::new(1, 1_000_000).unwrap();
    let effective_epsilon = (epsilon.numerator() as f64 / epsilon.denominator() as f64) as f32;
    for width in [3, 7, 129, 4096] {
        let input = bits(
            &(0..width)
                .map(|index| ((index * 17 % 29) as i32 - 14) as f32 / 16.0 + 2.0)
                .collect::<Vec<_>>(),
        );
        // The candidate is not executed until the independent oracle has
        // checked its data domain and frozen the input-dependent bounds.
        let reference = oracle::prepare(&input, width, effective_epsilon).unwrap();
        let actual = projection_probe(&input, epsilon);
        reference.check(&actual).unwrap();
    }
    for input in [
        bits(&[128., 129., 128., 130., 129.]),
        bits(&[-65536., 0., 65536.]),
    ] {
        let reference = oracle::prepare(&input, input.len(), effective_epsilon).unwrap();
        let actual = projection_probe(&input, epsilon);
        reference.check(&actual).unwrap();
    }
}

#[test]
fn dense_layer_norm_output_capacity_fails_before_either_compute_launch() {
    let config = NormConfig {
        rows: 2,
        in_features: 3,
        hidden_size: 2,
        epsilon: CanonicalRational::new(3, 4).unwrap(),
    };
    let host = host_account(19, 8); // Required: 12 input + 8 output bytes.
    let fixture = fixture(&config, &bits(&[1., 0., 0., 0., 1., 0.]), host.clone());
    let mut executor = executor(&fixture);
    let mut input = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 12)
        .unwrap();
    input
        .as_mut()
        .copy_from_slice(&raw_bytes(&bits(&[1., 2., 0., 2., 1., 0.])));
    let before = fixture.composition.runtime().snapshot();
    let result = executor.execute(
        id("run.tensor.norm-capacity"),
        id("request.tensor.norm-capacity"),
        vec![ManagedHostTensor::new(
            id(INPUT_ID),
            input_spec(vec![2, 3], ElementType::Bf16),
            input.freeze(),
        )
        .unwrap()],
    );
    assert!(matches!(
        &result,
        Err(TensorExecutionError::Host(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Total,
            requested: 8,
            committed: 12,
            limit: 19,
        }))
    ));
    let after = fixture.composition.runtime().snapshot();
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.dense_linear_launches, 0);
    assert_eq!(after.layer_norm_launches, 0);
    drop((result, executor));
    close_fixture(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
