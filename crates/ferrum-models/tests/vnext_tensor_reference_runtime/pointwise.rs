//! Managed BF16 pointwise graphs with real dense projections and independent
//! expected bits. These are component fixtures, not a complete Qwen block.

use super::*;
use ferrum_runtime::ModelWeights;

#[path = "pointwise_oracle.rs"]
mod oracle;

const SCALE: &str = "value.pointwise.scale";
const GATE: &str = "value.pointwise.gate";
const UPDATE: &str = "value.pointwise.update";
const NORMALIZED: &str = "value.pointwise.normalized";
const MODULATED: &str = "value.pointwise.modulated";
const FINAL: &str = "value.pointwise.final";
const PROFILE: &str = "reference.tensor.pointwise.bf16";

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Graph {
    Full,
    Modulation,
    Residual,
    SharedResidual,
}

impl Graph {
    fn has_modulation(self) -> bool {
        matches!(self, Self::Full | Self::Modulation)
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Broadcast {
    Elementwise,
    Row,
}

impl Broadcast {
    fn mode(self) -> RowBroadcast {
        match self {
            Self::Elementwise => RowBroadcast::Elementwise,
            Self::Row => RowBroadcast::Row,
        }
    }
    fn rows(self, rows: u64) -> u64 {
        match self {
            Self::Elementwise => rows,
            Self::Row => 1,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    width: u64,
    graph: Graph,
    broadcast: Broadcast,
}

impl Config {
    fn operations(&self) -> Vec<&'static str> {
        match self.graph {
            Graph::Full => vec![
                DENSE_LINEAR_BF16_OPERATION_ID,
                LAYER_NORM_BF16_OPERATION_ID,
                MODULATION_SCALE_BF16_OPERATION_ID,
                GATED_RESIDUAL_BF16_OPERATION_ID,
            ],
            Graph::Modulation => vec![
                DENSE_LINEAR_BF16_OPERATION_ID,
                MODULATION_SCALE_BF16_OPERATION_ID,
            ],
            Graph::Residual | Graph::SharedResidual => vec![
                DENSE_LINEAR_BF16_OPERATION_ID,
                GATED_RESIDUAL_BF16_OPERATION_ID,
            ],
        }
    }

    fn input_specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let spec = |rows, width| ProgramTensorSpec {
            dimensions: vec![rows, width],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        };
        let mut inputs = BTreeMap::from([(id(INPUT_ID), spec(self.rows, 1))]);
        let parameter_rows = self.broadcast.rows(self.rows);
        if self.graph.has_modulation() {
            inputs.insert(id(SCALE), spec(parameter_rows, self.width));
        }
        if self.graph != Graph::Modulation {
            inputs.insert(id(GATE), spec(parameter_rows, self.width));
        }
        if self.graph == Graph::Residual {
            inputs.insert(id(UPDATE), spec(self.rows, self.width));
        }
        inputs
    }

    fn input_bytes(&self) -> usize {
        self.input_specs()
            .values()
            .map(|spec| spec.dimensions.iter().product::<u64>() as usize * 2)
            .sum()
    }

    fn output_bytes(&self) -> usize {
        (self.rows * self.width * 2) as usize
    }
}

struct PointwiseFamily;

impl ModelFamilyProvider for PointwiseFamily {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.reference.tensor.pointwise"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.reference.tensor.pointwise")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw || config.rows == 0 || config.width == 0 {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: "pointwise fixture needs canonical positive dimensions".into(),
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
        Ok(id("metadata.reference.tensor.pointwise"))
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

    fn weights(&self, config: &Config) -> Result<ModelWeights, VNextError> {
        TinyTensorFamily(TinyTensorPrecision::Bf16).weights(&TinyTensorConfig {
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
        if config.graph == Graph::Full {
            boundaries.insert(id(NORMALIZED), ElementType::Bf16);
            boundaries.insert(id(MODULATED), ElementType::Bf16);
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
        let attributes = || {
            BTreeMap::from([(
                id("broadcast"),
                SemanticValue::Text(config.broadcast.mode().as_str().into()),
            )])
        };
        if config.graph == Graph::Full {
            nodes.push(ProgramNode {
                id: id("node.pointwise.norm"),
                operation_id: id(LAYER_NORM_BF16_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![id(OUTPUT_ID)],
                outputs: vec![id(NORMALIZED)],
                attributes: BTreeMap::from([
                    (id("hidden_size"), SemanticValue::Unsigned(config.width)),
                    (
                        id("epsilon"),
                        SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
                    ),
                ]),
            });
        }
        if config.graph.has_modulation() {
            nodes.push(ProgramNode {
                id: id("node.pointwise.modulation"),
                operation_id: id(MODULATION_SCALE_BF16_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    id(if config.graph == Graph::Full {
                        NORMALIZED
                    } else {
                        OUTPUT_ID
                    }),
                    id(SCALE),
                ],
                outputs: vec![id(if config.graph == Graph::Full {
                    MODULATED
                } else {
                    FINAL
                })],
                attributes: attributes(),
            });
        }
        if config.graph != Graph::Modulation {
            // Dense output is read both by LN and here. Its device lifetime
            // must extend past normalization and modulation, not end at LN.
            nodes.push(ProgramNode {
                id: id("node.pointwise.residual"),
                operation_id: id(GATED_RESIDUAL_BF16_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    id(OUTPUT_ID),
                    id(GATE),
                    id(match config.graph {
                        Graph::Full => MODULATED,
                        Graph::SharedResidual => GATE,
                        _ => UPDATE,
                    }),
                ],
                outputs: vec![id(FINAL)],
                attributes: attributes(),
            });
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
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}
fn decode_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect()
}

fn is_pointwise(node: &PlanNode) -> bool {
    matches!(
        node.operation_id().as_str(),
        MODULATION_SCALE_BF16_OPERATION_ID | GATED_RESIDUAL_BF16_OPERATION_ID
    )
}

fn pointwise_estimator_inputs(fixture: &TensorFixture) -> Vec<String> {
    fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .filter(|node| is_pointwise(node))
        .map(|node| {
            node.provider_resources()
                .estimator_input_fingerprint()
                .to_owned()
        })
        .collect()
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
    let family = TypedFamilyRegistration::new(PointwiseFamily)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    assert_eq!(family.metadata(), &ModelSemanticMetadata::TensorComponent);
    assert!(family.program().states().is_empty());
    assert!(family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .all(|node| node.work == ProgramNodeWorkSpec::Fixed));
    let composition =
        ReferenceVNextComposition::create(id("device.reference.tensor.pointwise")).unwrap();
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
    // These are estimates produced by the real compiler's trusted requests,
    // not a fabricated public estimator request or a second resource ledger.
    for node in compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .filter(|node| is_pointwise(node))
    {
        let resources = node.provider_resources();
        assert_eq!(resources.value_alignment_bytes(), 16);
        assert!(resources.scratch().is_none());
        assert!(resources.binding().is_none());
        assert!(resources.persistent().is_none());
        assert!(node.scratch_resource().is_none());
        assert!(node.binding_resource().is_none());
        assert!(node.persistent_resource().is_none());
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

fn account(total: usize, inputs: usize, outputs: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: inputs,
                transfer_staging_bytes: 0,
                output_bytes: outputs,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn managed_inputs(
    config: &Config,
    host: &HostMemoryAccount,
    data: &[(&str, &[u16])],
) -> Vec<ManagedHostTensor> {
    let specs = config.input_specs();
    assert_eq!(data.len(), specs.len());
    data.iter()
        .map(|(name, values)| {
            let value_id: ProgramValueId = id(*name);
            let spec = &specs[&value_id];
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                .unwrap();
            bytes.as_mut().copy_from_slice(&raw(values));
            ManagedHostTensor::new(
                value_id,
                input_spec(spec.dimensions.clone(), ElementType::Bf16),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect()
}

fn run(config: &Config, weights: &[u16], data: &[(&str, &[u16])], expected: &oracle::Prepared) {
    let input_bytes = config.input_bytes();
    let output_bytes = config.output_bytes();
    // Exact managed Host input + final output budget; no Host intermediates
    // or expanded parameter matrix are allowed by this test's account.
    let host = account(input_bytes + output_bytes, input_bytes, output_bytes);
    let fixture = fixture(config, weights, host.clone());
    let mut executor = executor(&fixture);
    let inputs = managed_inputs(config, &host, data);
    assert_eq!(host.usage().unwrap().committed_bytes(), input_bytes);
    let outputs = executor
        .execute(
            id("run.tensor.pointwise"),
            id("request.tensor.pointwise"),
            inputs,
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value_id(), &id::<ProgramValueId>(FINAL));
    assert_eq!(
        outputs[0].tensor().dimensions(),
        [config.rows, config.width]
    );
    assert_eq!(outputs[0].tensor().element_type(), ElementType::Bf16);
    assert!(host.owns(outputs[0].bytes()));
    expected
        .check(&decode_bytes(outputs[0].bytes().bytes()))
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    let snapshot = fixture.composition.runtime().snapshot();
    assert_eq!(snapshot.dense_linear_launches, 1);
    assert_eq!(
        snapshot.layer_norm_launches,
        u64::from(config.graph == Graph::Full)
    );
    assert_eq!(
        snapshot.modulation_scale_launches,
        u64::from(config.graph.has_modulation())
    );
    assert_eq!(
        snapshot.gated_residual_launches,
        u64::from(config.graph != Graph::Modulation)
    );
    assert_eq!(
        snapshot.uploaded_bytes,
        (weights.len() * 2 + input_bytes) as u64
    );
    assert_eq!(snapshot.readback_bytes, output_bytes as u64);
    let owner = outputs[0].bytes().clone();
    let view = owner.slice(0..2).unwrap();
    drop((outputs, executor));
    close_fixture(fixture);
    expected.check(&decode_bytes(owner.bytes())).unwrap();
    assert_eq!(view.bytes(), &owner.bytes()[..2]);
    drop(owner);
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    drop(view);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

const WEIGHTS: [u16; 4] = [0xbf80, 0xbf80, 0xbf80, 0x4040];
const PROJECTION: [u16; 2] = [0x3f80, 0xbf80];
const DENSE: [u16; 8] = [
    0xbf80, 0xbf80, 0xbf80, 0x4040, 0x3f80, 0x3f80, 0x3f80, 0xc040,
];
const NORM: [u16; 8] = [
    0xbf00, 0xbf00, 0xbf00, 0x3fc0, 0x3f00, 0x3f00, 0x3f00, 0xbfc0,
];
const ROW_SCALE: [u16; 4] = [0xc041, 0xbf80, 0x3f80, 0x3b80];
const ROW_GATE: [u16; 4] = [0x3f7e, 0x3f80, 0x3f00, 0xbf80];

fn full_config(broadcast: Broadcast) -> Config {
    Config {
        rows: 2,
        width: 4,
        graph: Graph::Full,
        broadcast,
    }
}

#[test]
fn dense_norm_modulation_residual_preserves_both_bf16_boundaries_and_skip_lifetime() {
    let config = full_config(Broadcast::Row);
    // Dense gives +/-[-1,-1,-1,3]; mean=0, v=3, eps=1, reciprocal=1/2.
    // Scale[0]=-193/64 makes update=+/-129/128; gate=127/128 makes
    // the rounded product +/-1, cancelling the saved Dense skip to +0.
    // Scale[3]=2^-8 rounds 1+s to 1; gate=-1 exposes any omitted cast.
    let modulation = oracle::modulation(&NORM, &ROW_SCALE, 2, 4, RowBroadcast::Row).unwrap();
    assert_eq!(
        modulation.bits(),
        [0x3f81, 0x8000, 0xbf80, 0x3fc0, 0xbf81, 0, 0x3f80, 0xbfc0]
    );
    let expected = oracle::residual(
        &DENSE,
        &ROW_GATE,
        modulation.bits(),
        2,
        4,
        RowBroadcast::Row,
    )
    .unwrap();
    assert_eq!(
        expected.bits(),
        [0, 0xbf80, 0xbfc0, 0x3fc0, 0, 0x3f80, 0x3fc0, 0xbfc0]
    );
    run(
        &config,
        &WEIGHTS,
        &[
            (INPUT_ID, &PROJECTION),
            (SCALE, &ROW_SCALE),
            (GATE, &ROW_GATE),
        ],
        &expected,
    );
}

#[test]
fn elementwise_parameters_are_distinct_across_rectangular_rows() {
    let config = full_config(Broadcast::Elementwise);
    let scale = [0xc041, 0xbf80, 0x3f80, 0x3b80, 0, 0, 0, 0];
    let gate = [
        0x3f7e, 0x3f80, 0x3f00, 0xbf80, 0x3f80, 0x3f80, 0x3f80, 0x3f80,
    ];
    let modulation = oracle::modulation(&NORM, &scale, 2, 4, RowBroadcast::Elementwise).unwrap();
    let expected = oracle::residual(
        &DENSE,
        &gate,
        modulation.bits(),
        2,
        4,
        RowBroadcast::Elementwise,
    )
    .unwrap();
    assert_eq!(
        expected.bits(),
        [0, 0xbf80, 0xbfc0, 0x3fc0, 0x3fc0, 0x3fc0, 0x3fc0, 0xc090]
    );
    run(
        &config,
        &WEIGHTS,
        &[(INPUT_ID, &PROJECTION), (SCALE, &scale), (GATE, &gate)],
        &expected,
    );
}

#[test]
fn dense_then_modulation_checks_overflow_nonfinite_and_signed_zero_behavior() {
    let config = Config {
        rows: 1,
        width: 7,
        graph: Graph::Modulation,
        broadcast: Broadcast::Row,
    };
    let weights = [0x3fc0, 0, 0x7f7f, 0xbf80, 0x3f80, 0x3f80, 0x3f80];
    let scale = [0x3b80, 0x7f80, 0x3f80, 0xbf80, 0xff80, 0x7fc1, 0xc000];
    let expected = oracle::modulation(&weights, &scale, 1, 7, RowBroadcast::Row).unwrap();
    assert_eq!(
        expected.bits(),
        [0x3fc0, 0x7fc0, 0x7f80, 0x8000, 0xff80, 0x7fc0, 0xbf80]
    );
    run(
        &config,
        &weights,
        &[(INPUT_ID, &[0x3f80]), (SCALE, &scale)],
        &expected,
    );
}

#[test]
fn dense_then_residual_uploads_real_updates_and_preserves_underflow_and_overflow_boundaries() {
    let config = Config {
        rows: 1,
        width: 7,
        graph: Graph::Residual,
        broadcast: Broadcast::Elementwise,
    };
    let weights = [0xbf80, 1, 0xff7f, 0xff7f, 0, 0x3f80, 0];
    let gate = [0x3f7e, 1, 0x3ffe, 0x4000, 0x8000, 0x7f80, 0x4380];
    let update = [0x3f81, 0x3f00, 0x7f01, 0x7f7f, 0x3f80, 0, 0x4380];
    let expected =
        oracle::residual(&weights, &gate, &update, 1, 7, RowBroadcast::Elementwise).unwrap();
    assert_eq!(expected.bits(), [0, 1, 0x7f80, 0x7f80, 0, 0x7fc0, 0x4780]);
    run(
        &config,
        &weights,
        &[(INPUT_ID, &[0x3f80]), (GATE, &gate), (UPDATE, &update)],
        &expected,
    );
}

#[test]
fn four_node_output_capacity_rejects_before_device_admission_or_upload() {
    let config = full_config(Broadcast::Row);
    let inputs = config.input_bytes();
    let outputs = config.output_bytes();
    for (host, scope, committed, limit) in [
        (
            account(inputs + outputs - 1, inputs, outputs),
            HostMemoryPressureScope::Total,
            inputs,
            inputs + outputs - 1,
        ),
        (
            account(inputs + outputs, inputs, outputs - 1),
            HostMemoryPressureScope::Class(HostMemoryClass::Output),
            0,
            outputs - 1,
        ),
    ] {
        let fixture = fixture(&config, &WEIGHTS, host.clone());
        let mut executor = executor(&fixture);
        let input = managed_inputs(
            &config,
            &host,
            &[
                (INPUT_ID, &PROJECTION),
                (SCALE, &ROW_SCALE),
                (GATE, &ROW_GATE),
            ],
        );
        let before = fixture.composition.runtime().snapshot();
        let result = executor.execute(
            id("run.tensor.pointwise-capacity"),
            id("request.tensor.pointwise-capacity"),
            input.clone(),
        );
        assert!(
            matches!(&result, Err(TensorExecutionError::Host(HostMemoryError::Capacity {
            scope: actual_scope, requested, committed: actual_committed, limit: actual_limit,
        })) if *actual_scope == scope && *requested == outputs && *actual_committed == committed && *actual_limit == limit)
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
                after.layer_norm_launches,
                after.modulation_scale_launches,
                after.gated_residual_launches
            ),
            (0, 0, 0, 0)
        );
        drop((result, executor));
        assert_eq!(host.usage().unwrap().committed_bytes(), inputs);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            0
        );
        drop(input);
        close_fixture(fixture);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn one_row_broadcast_modes_compute_the_same_values_but_have_distinct_plan_identities() {
    let weights = [0x3fc0, 0x3f80];
    let scale = [0x3b80, 0x3f80];
    let mut fingerprints = Vec::new();
    let mut estimator_inputs = Vec::new();
    for broadcast in [Broadcast::Row, Broadcast::Elementwise] {
        let config = Config {
            rows: 1,
            width: 2,
            graph: Graph::Modulation,
            broadcast,
        };
        let host = account(
            config.input_bytes() + config.output_bytes(),
            config.input_bytes(),
            config.output_bytes(),
        );
        let fixture = fixture(&config, &weights, host);
        fingerprints.push(
            fixture
                .compilation
                .executable()
                .execution_plan()
                .plan_hash()
                .clone(),
        );
        estimator_inputs.push(pointwise_estimator_inputs(&fixture));
        close_fixture(fixture);
        let expected = oracle::modulation(&weights, &scale, 1, 2, broadcast.mode()).unwrap();
        expected.check(&[0x3fc0, 0x4000]).unwrap();
        run(
            &config,
            &weights,
            &[(INPUT_ID, &[0x3f80]), (SCALE, &scale)],
            &expected,
        );
    }
    assert_ne!(fingerprints[0], fingerprints[1]);
    assert_ne!(estimator_inputs[0], estimator_inputs[1]);
    // A fresh runtime/source with the same semantic mode keeps the identity;
    // the inequality above must not be an incidental runtime-instance effect.
    let config = Config {
        rows: 1,
        width: 2,
        graph: Graph::Modulation,
        broadcast: Broadcast::Row,
    };
    let repeated = fixture(
        &config,
        &weights,
        account(
            config.input_bytes() + config.output_bytes(),
            config.input_bytes(),
            config.output_bytes(),
        ),
    );
    assert_eq!(
        &fingerprints[0],
        repeated
            .compilation
            .executable()
            .execution_plan()
            .plan_hash()
    );
    assert_eq!(estimator_inputs[0], pointwise_estimator_inputs(&repeated));
    close_fixture(repeated);
}

#[test]
fn gate_and_update_can_share_the_same_read_only_device_value() {
    let config = Config {
        rows: 2,
        width: 2,
        graph: Graph::SharedResidual,
        broadcast: Broadcast::Elementwise,
    };
    let weights = [0x3f80, 0xbf80];
    let dense = [0x3f80, 0xbf80, 0xbf80, 0x3f80];
    let shared = [0x3f00, 0x3f80, 0xbf80, 0x4000];
    // Both read ports bind the same program value and managed device region.
    // NoAlias applies to the output, not to immutable input/input overlap.
    let expected =
        oracle::residual(&dense, &shared, &shared, 2, 2, RowBroadcast::Elementwise).unwrap();
    assert_eq!(expected.bits(), [0x3fa0, 0, 0, 0x40a0]);
    run(
        &config,
        &weights,
        &[(INPUT_ID, &PROJECTION), (GATE, &shared)],
        &expected,
    );
}
