//! Managed activation graphs. Analytic fixtures exercise the fixed Reference
//! profile's BF16 boundaries; exhaustive numerical qualification lives beside
//! the actual Reference scalar implementation in ferrum-kernels.

use super::*;

const UP: &str = "value.activation.up";
const RAW_GATE: &str = "value.activation.raw_gate";
const GATE: &str = "value.activation.gate";
const FINAL: &str = "value.activation.final";
const PROFILE: &str = "reference.tensor.activations.bf16";

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Graph {
    Silu,
    SiluMul,
    SharedSiluMul,
    TanhResidual,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    width: u64,
    graph: Graph,
}

impl Config {
    fn operations(&self) -> Vec<&'static str> {
        let mut operations = vec![DENSE_LINEAR_BF16_OPERATION_ID];
        operations.extend(match self.graph {
            Graph::Silu => vec![SILU_BF16_OPERATION_ID],
            Graph::SiluMul | Graph::SharedSiluMul => vec![SILU_MUL_BF16_OPERATION_ID],
            Graph::TanhResidual => vec![TANH_BF16_OPERATION_ID, GATED_RESIDUAL_BF16_OPERATION_ID],
        });
        operations
    }

    fn input_specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let spec = |width| ProgramTensorSpec {
            dimensions: vec![self.rows, width],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        };
        let mut inputs = BTreeMap::from([(id(INPUT_ID), spec(1))]);
        if matches!(self.graph, Graph::SiluMul | Graph::TanhResidual) {
            inputs.insert(id(UP), spec(self.width));
        }
        if self.graph == Graph::TanhResidual {
            inputs.insert(id(RAW_GATE), spec(self.width));
        }
        inputs
    }

    fn input_bytes(&self) -> usize {
        self.input_specs()
            .values()
            .map(|s| s.dimensions.iter().product::<u64>() as usize * 2)
            .sum()
    }

    fn output_bytes(&self) -> usize {
        (self.rows * self.width * 2) as usize
    }
}

struct ActivationFamily;

impl ModelFamilyProvider for ActivationFamily {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.reference.tensor.activations"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tensor.activations")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw || config.rows == 0 || config.width == 0 {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: "activation fixture requires canonical positive dimensions".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.reference.tensor.activations"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Config) -> Result<WeightSchema, VNextError> {
        TinyTensorFamily(TinyTensorPrecision::Bf16).weight_schema(&TinyTensorConfig {
            rows: config.rows,
            in_features: 1,
            out_features: config.width,
        })
    }

    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(PROFILE).unwrap();
        let mut boundaries = BTreeMap::from([
            (id(OUTPUT_ID), ElementType::Bf16),
            (id(FINAL), ElementType::Bf16),
        ]);
        if config.graph == Graph::TanhResidual {
            boundaries.insert(id(GATE), ElementType::Bf16);
        }
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(FINAL),
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations: config
                    .operations()
                    .into_iter()
                    .map(|op| NumericalOperationContract {
                        operation_id: id(op),
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
        config: &Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let dense = TinyTensorFamily(TinyTensorPrecision::Bf16).semantic_program(
            &TinyTensorConfig {
                rows: config.rows,
                in_features: 1,
                out_features: config.width,
            },
            profile,
        )?;
        let mut blocks = dense.blocks().to_vec();
        let nodes = &mut blocks[0].nodes;
        let node = |name, operation, inputs: Vec<ProgramValueId>, output| ProgramNode {
            id: id(name),
            operation_id: id(operation),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![id(output)],
            attributes: BTreeMap::new(),
        };
        match config.graph {
            Graph::Silu => nodes.push(node(
                "node.activation.silu",
                SILU_BF16_OPERATION_ID,
                vec![id(OUTPUT_ID)],
                FINAL,
            )),
            Graph::SiluMul | Graph::SharedSiluMul => nodes.push(node(
                "node.activation.silu_mul",
                SILU_MUL_BF16_OPERATION_ID,
                vec![
                    id(OUTPUT_ID),
                    id(if config.graph == Graph::SharedSiluMul {
                        OUTPUT_ID
                    } else {
                        UP
                    }),
                ],
                FINAL,
            )),
            Graph::TanhResidual => {
                nodes.push(node(
                    "node.activation.tanh",
                    TANH_BF16_OPERATION_ID,
                    vec![id(RAW_GATE)],
                    GATE,
                ));
                let mut residual = node(
                    "node.activation.residual",
                    GATED_RESIDUAL_BF16_OPERATION_ID,
                    vec![id(OUTPUT_ID), id(GATE), id(UP)],
                    FINAL,
                );
                residual
                    .attributes
                    .insert(id("broadcast"), SemanticValue::Text("elementwise".into()));
                nodes.push(residual);
            }
        }
        ModelProgram::new(
            self.family_id().clone(),
            config.input_specs().keys().cloned().collect(),
            blocks,
            vec![],
            dense.weights().to_vec(),
            vec![id(FINAL)],
        )
    }

    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn raw(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn account(config: &Config, missing_total: usize, missing_output: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            config.input_bytes() + config.output_bytes() - missing_total,
            HostMemoryClassLimits {
                input_preprocess_bytes: config.input_bytes(),
                transfer_staging_bytes: 0,
                output_bytes: config.output_bytes() - missing_output,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn fixture(config: &Config, weights: &[u16], host: HostMemoryAccount) -> TensorFixture {
    let directory = tempfile::tempdir().unwrap();
    let weight_bytes = raw(weights);
    let view = TensorView::new(Dtype::BF16, vec![config.width as usize, 1], &weight_bytes).unwrap();
    serialize_to_file(
        vec![("linear.weight".to_owned(), view)],
        &None::<HashMap<String, String>>,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = SafetensorsArchive::open(directory.path()).unwrap();
    let family = TypedFamilyRegistration::new(ActivationFamily)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    assert_eq!(family.metadata(), &ModelSemanticMetadata::TensorComponent);
    assert!(family.program().states().is_empty());
    let composition =
        ReferenceVNextComposition::create(id("device.reference.tensor.activations")).unwrap();
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
    let mut options = ProgramPlanCompileOptions::new(config.input_specs()).unwrap();
    options.require_weight_materializer(composition.weight_materializer_id().clone());
    options.retain_completion_value(id(FINAL));
    let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    for node in compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.operation_id().as_str(),
                SILU_BF16_OPERATION_ID | SILU_MUL_BF16_OPERATION_ID | TANH_BF16_OPERATION_ID
            )
        })
    {
        let estimate = node.provider_resources();
        assert_eq!(estimate.value_alignment_bytes(), 16);
        assert!(
            estimate.scratch().is_none()
                && estimate.binding().is_none()
                && estimate.persistent().is_none()
        );
        assert!(
            node.scratch_resource().is_none()
                && node.binding_resource().is_none()
                && node.persistent_resource().is_none()
        );
    }
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

fn inputs(
    config: &Config,
    host: &HostMemoryAccount,
    values: &[(&str, &[u16])],
) -> Vec<ManagedHostTensor> {
    let specs = config.input_specs();
    assert_eq!(values.len(), specs.len());
    values
        .iter()
        .map(|(name, bits)| {
            let value_id: ProgramValueId = id(*name);
            let spec = &specs[&value_id];
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bits.len() * 2)
                .unwrap();
            bytes.as_mut().copy_from_slice(&raw(bits));
            ManagedHostTensor::new(
                value_id,
                input_spec(spec.dimensions.clone(), ElementType::Bf16),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

fn check(bytes: &[u8], expected: &[u16]) {
    assert_eq!(bytes.len(), expected.len() * 2);
    for (index, (bytes, expected)) in bytes.chunks_exact(2).zip(expected).enumerate() {
        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
        if expected & 0x7fff > 0x7f80 {
            assert!(
                actual & 0x7fff > 0x7f80,
                "element {index}: expected NaN, got {actual:#06x}"
            );
        } else {
            assert_eq!(actual, *expected, "element {index}");
        }
    }
}

fn run(config: &Config, weights: &[u16], values: &[(&str, &[u16])], expected: &[u16]) {
    let host = account(config, 0, 0);
    let fixture = fixture(config, weights, host.clone());
    let mut executor = executor(&fixture);
    let input = inputs(config, &host, values);
    let outputs = executor
        .execute(id("run.activation"), id("request.activation"), input)
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(FINAL));
    assert_eq!(
        outputs[0].tensor().dimensions(),
        [config.rows, config.width]
    );
    assert!(host.owns(outputs[0].bytes()));
    check(outputs[0].bytes().bytes(), expected);
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.dense_linear_launches, 1);
    assert_eq!(
        snapshot.silu_launches,
        u64::from(config.graph == Graph::Silu)
    );
    assert_eq!(
        snapshot.silu_mul_launches,
        u64::from(matches!(
            config.graph,
            Graph::SiluMul | Graph::SharedSiluMul
        ))
    );
    assert_eq!(
        snapshot.tanh_launches,
        u64::from(config.graph == Graph::TanhResidual)
    );
    assert_eq!(
        snapshot.gated_residual_launches,
        u64::from(config.graph == Graph::TanhResidual)
    );
    assert_eq!(
        snapshot.uploaded_bytes,
        (weights.len() * 2 + config.input_bytes()) as u64
    );
    assert_eq!(snapshot.readback_bytes, config.output_bytes() as u64);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        config.output_bytes()
    );
    let retained = outputs[0].bytes().clone();
    let view = retained.slice(0..2).unwrap();
    drop((outputs, executor));
    close_fixture(fixture);
    check(retained.bytes(), expected);
    drop(retained);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        config.output_bytes()
    );
    drop(view);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn dense_silu_preserves_fp32_overflow_tiny_tie_and_nonfinite_profile() {
    let config = Config {
        rows: 1,
        width: 7,
        graph: Graph::Silu,
    };
    // Real single-term dense products are exact, including the minimum BF16.
    // Independent analytic exp bounds at 1 give B(SiLU(1))=187/256.
    // expf(90) overflows; x=2^-133 divides by exactly 2 and ties to zero.
    run(
        &config,
        &[0x3f80, 0xc2b4, 1, 0, 0x7f80, 0xff80, 0x7fc1],
        &[(INPUT_ID, &[0x3f80])],
        &[0x3f3b, 0x8000, 0, 0, 0x7f80, 0x7fc0, 0x7fc0],
    );
}

#[test]
fn dense_silu_mul_keeps_the_internal_bf16_rounding_and_typed_specials() {
    let config = Config {
        rows: 2,
        width: 2,
        graph: Graph::SiluMul,
    };
    // B(SiLU(1))=187/256; times 129/128 is 24123/32768, B=188/256.
    // Omitting the internal BF16 round yields 189/256 instead. The last
    // three cells cover -0*Inf, +0*Inf, and +0*(-1) independently.
    run(
        &config,
        &[0x3f80, 0xc2b4],
        &[
            (INPUT_ID, &[0x3f80, 0]),
            (UP, &[0x3f81, 0x7f80, 0x7f80, 0xbf80]),
        ],
        &[0x3f3c, 0x7fc0, 0x7fc0, 0x8000],
    );
}

#[test]
fn dense_value_can_feed_both_read_ports_of_silu_mul() {
    let config = Config {
        rows: 1,
        width: 3,
        graph: Graph::SharedSiluMul,
    };
    run(
        &config,
        &[0x3f80, 0xc2b4, 1],
        &[(INPUT_ID, &[0x3f80])],
        &[0x3f3b, 0, 0],
    );
}

#[test]
fn tanh_then_residual_retains_dense_skip_and_bf16_gate_boundary() {
    let config = Config {
        rows: 2,
        width: 2,
        graph: Graph::TanhResidual,
    };
    // B(tanh(1))=195/256; B((195/256)*(129/128))=197/256.
    // Skipping the tanh BF16 boundary gives 196/256. The second row's
    // nonzero dense skip contributes +/-1, proving it remains live.
    run(
        &config,
        &[0x3f80, 0xbf80],
        &[
            (INPUT_ID, &[0, 0x3f80]),
            (RAW_GATE, &[0x3f80, 0xbf80, 0x7f80, 0xff80]),
            (UP, &[0x3f81, 0x3f81, 0x3f80, 0x3f80]),
        ],
        &[0x3f45, 0xbf45, 0x4000, 0xc000],
    );
}

#[test]
fn activation_output_capacity_fails_before_device_admission_or_upload() {
    let config = Config {
        rows: 2,
        width: 2,
        graph: Graph::TanhResidual,
    };
    for (total_deficit, output_deficit, expected_scope) in [
        (1, 0, HostMemoryPressureScope::Total),
        (
            0,
            1,
            HostMemoryPressureScope::Class(HostMemoryClass::Output),
        ),
    ] {
        let host = account(&config, total_deficit, output_deficit);
        let fixture = fixture(&config, &[0x3f80, 0xbf80], host.clone());
        let mut executor = executor(&fixture);
        let input = inputs(
            &config,
            &host,
            &[
                (INPUT_ID, &[0, 0x3f80]),
                (RAW_GATE, &[0x3f80; 4]),
                (UP, &[0x3f81; 4]),
            ],
        );
        let before = fixture.composition.runtime().snapshot();
        let result = executor.execute(
            id("run.activation-capacity"),
            id("request.activation-capacity"),
            input.clone(),
        );
        assert!(
            matches!(result, Err(TensorExecutionError::Host(HostMemoryError::Capacity { scope, requested, .. }))
            if scope == expected_scope && requested == config.output_bytes())
        );
        let after = fixture.composition.runtime().snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
        assert_eq!(after.readback_bytes, 0);
        assert_eq!(
            (
                after.dense_linear_launches,
                after.silu_launches,
                after.silu_mul_launches,
                after.tanh_launches,
                after.gated_residual_launches
            ),
            (0, 0, 0, 0, 0)
        );
        drop(executor);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            config.input_bytes()
        );
        drop(input);
        close_fixture(fixture);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
