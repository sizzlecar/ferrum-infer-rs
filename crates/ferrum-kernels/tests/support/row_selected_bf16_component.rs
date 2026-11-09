//! Real compact and expanded graphs share the same original managed inputs.
//! The independent scalar oracle commits every specified BF16/F32 boundary.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
const X: &str = "value.compact.fixture.x";
const TABLE: &str = "value.compact.fixture.table";
const U: &str = "value.compact.fixture.update";
const SHIFT: &str = "value.compact.fixture.shift";
const SCALE: &str = "value.compact.fixture.scale";
const GATE: &str = "value.compact.fixture.gate";
const SCALED: &str = "value.compact.fixture.scaled";
const MOD: &str = "value.compact.fixture.modulation";
const EXP_MOD: &str = "value.compact.fixture.expanded_modulation";
const ADD: &str = "value.compact.fixture.addcmul";
const EXP_ADD: &str = "value.compact.fixture.expanded_addcmul";
const PROFILE: &str = "fixture.row_selected_bf16.expanded_equivalence.v1";
const ROWS: u64 = 7;
const PARAMETERS: u64 = 4;
const SELECTED: [usize; 7] = [3, 3, 1, 3, 3, 0, 0];
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    width: u64,
}
impl Config {
    fn table_width(&self) -> u64 {
        3 * self.width + 5
    }
    fn shift(&self) -> u64 {
        2
    }
    fn scale(&self) -> u64 {
        self.width + 3
    }
    fn gate(&self) -> u64 {
        2 * self.width + 4
    }
    fn specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        [X, TABLE, U, SHIFT, SCALE, GATE]
            .into_iter()
            .map(|name| {
                (
                    id(name),
                    tensor(if name == TABLE {
                        vec![PARAMETERS, self.table_width()]
                    } else {
                        vec![ROWS, self.width]
                    }),
                )
            })
            .collect()
    }
    fn attrs(&self, kind: RowSelectedBf16Kind) -> BTreeMap<AttributeId, SemanticValue> {
        let mut attrs = BTreeMap::from([
            (id("run_lengths"), SemanticValue::Integers(vec![2, 1, 2, 2])),
            (
                id("parameter_rows"),
                SemanticValue::Integers(vec![3, 1, 3, 0]),
            ),
        ]);
        match kind {
            RowSelectedBf16Kind::Modulation => attrs.extend([
                (id("shift_column"), SemanticValue::Unsigned(self.shift())),
                (id("scale_column"), SemanticValue::Unsigned(self.scale())),
            ]),
            RowSelectedBf16Kind::Addcmul => {
                attrs.insert(id("gate_column"), SemanticValue::Unsigned(self.gate()));
            }
        }
        attrs
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
struct Family {
    id: ModelFamilyId,
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.compact.bf16.fixture")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c: Config = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw || c.width == 0 || c.width > 5376 {
            return Err(invalid("invalid compact fixture geometry"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.compact.bf16.fixture"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(MOD),
                boundaries: [SCALED, MOD, EXP_MOD, ADD, EXP_ADD]
                    .into_iter()
                    .map(|v| (id(v), ElementType::Bf16))
                    .collect(),
                states: vec![],
                kv_storage: vec![],
                operations: [
                    ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
                    ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID,
                    MODULATION_SCALE_BF16_OPERATION_ID,
                    PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
                    ADDCMUL_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|op| NumericalOperationContract {
                    operation_id: id(op),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
                .collect(),
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let node =
            |name: &str, operation: &str, inputs: &[&str], output: &str, attributes| ProgramNode {
                id: id(name),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: inputs.iter().map(|v| id(*v)).collect(),
                outputs: vec![id(output)],
                attributes,
            };
        let broadcast =
            || BTreeMap::from([(id("broadcast"), SemanticValue::Text("elementwise".into()))]);
        ModelProgram::new(
            self.family_id().clone(),
            [X, TABLE, U, SHIFT, SCALE, GATE]
                .into_iter()
                .map(id)
                .collect(),
            vec![ProgramBlock {
                id: "block.compact.fixture".into(),
                nodes: vec![
                    node(
                        "node.compact.mod",
                        ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
                        &[X, TABLE],
                        MOD,
                        c.attrs(RowSelectedBf16Kind::Modulation),
                    ),
                    node(
                        "node.expanded.scale",
                        MODULATION_SCALE_BF16_OPERATION_ID,
                        &[X, SCALE],
                        SCALED,
                        broadcast(),
                    ),
                    node(
                        "node.expanded.shift",
                        PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
                        &[SCALED, SHIFT],
                        EXP_MOD,
                        BTreeMap::new(),
                    ),
                    node(
                        "node.compact.add",
                        ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID,
                        &[X, TABLE, U],
                        ADD,
                        c.attrs(RowSelectedBf16Kind::Addcmul),
                    ),
                    node(
                        "node.expanded.add",
                        ADDCMUL_BF16_OPERATION_ID,
                        &[X, GATE, U],
                        EXP_ADD,
                        broadcast(),
                    ),
                ],
            }],
            vec![],
            vec![],
            [MOD, EXP_MOD, ADD, EXP_ADD].into_iter().map(id).collect(),
        )?
        .with_input_tensor_specs(c.specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn decode(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
// Bit-defined BF16 RNE, independent of the production half conversion helpers.
fn round(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return (bits >> 16) as u16 | 0x0040;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
fn add(a: f32, b: f32) -> f32 {
    (f64::from(a) + f64::from(b)) as f32
}
fn mul(a: f32, b: f32) -> f32 {
    (f64::from(a) * f64::from(b)) as f32
}
fn exact_equal(actual: u16, expected: u16, label: &str, index: usize) {
    if decode(expected).is_nan() {
        assert!(decode(actual).is_nan(), "{label} NaN class at {index}");
    } else {
        assert_eq!(actual, expected, "{label} exact bits at {index}");
    }
}
struct Data {
    inputs: BTreeMap<&'static str, Vec<u16>>,
    modulation: Vec<u16>,
    addcmul: Vec<u16>,
}
fn data(c: &Config, trial: usize) -> Data {
    let h = c.width as usize;
    let w = c.table_width() as usize;
    let mut x = vec![0u16; ROWS as usize * h];
    let mut u = x.clone();
    let mut table = vec![round(f32::NAN); PARAMETERS as usize * w];
    for p in [0usize, 1, 3] {
        for col in 0..h {
            table[p * w + c.shift() as usize + col] = round((p as f32 - 1.) * 0.125);
            table[p * w + c.scale() as usize + col] = round((col % 5) as f32 * 0.0625);
            table[p * w + c.gate() as usize + col] = round((p as f32 + 1.) * 0.25);
        }
    }
    for row in 0..ROWS as usize {
        for col in 0..h {
            let i = row * h + col;
            x[i] = round(1. + ((row + col) % 4) as f32 * 0.25);
            u[i] = round(0.5 + (col % 3) as f32 * 0.125);
        }
    }
    let at = |column: u64, col: usize| 3 * w + column as usize + col;
    table[at(c.scale(), 0)] = round(1. / 256.);
    table[at(c.shift(), 0)] = round(-1.);
    x[0] = round(1.);
    table[w + c.gate() as usize] = round(129. / 128.);
    x[2 * h] = round(1.);
    u[2 * h] = round(132. / 128.);
    if h > 1 {
        table[at(c.scale(), 1)] = round(4. / 128.);
        table[at(c.shift(), 1)] = round(-133. / 128.);
        x[h + 1] = round(129. / 128.);
    }
    if h > 2 {
        table[at(c.gate(), 2)] = 0x7f7f;
        x[3 * h + 2] = 0xff7f;
        u[3 * h + 2] = round(2.);
    }
    if h > 3 {
        table[at(c.gate(), 3)] = round(1.);
        x[4 * h + 3] = 1;
        u[4 * h + 3] = 1;
        table[at(c.scale(), 3)] = round(0.);
        table[at(c.shift(), 3)] = 1;
    }
    if h > 4 {
        table[at(c.shift(), 4)] = round(f32::NAN);
        table[at(c.gate(), 4)] = round(f32::NAN);
    }
    if h > 5 {
        table[at(c.shift(), 5)] = round(f32::INFINITY);
        table[at(c.scale(), 5)] = round(0.);
        x[5] = round(f32::NEG_INFINITY);
    }
    if h > 6 {
        table[at(c.shift(), 6)] = 0x8000;
        table[at(c.scale(), 6)] = round(0.);
        table[at(c.gate(), 6)] = 0x8000;
        x[6] = 0x8000;
        u[6] = round(1.);
    }
    // Only the final request's own table row changes. Row2 remains poison and
    // nonadjacent uses of row3 must stay identical across both executions.
    if trial == 1 {
        for col in 0..h {
            table[c.shift() as usize + col] = round(2.);
            table[c.scale() as usize + col] = round(-0.5);
            table[c.gate() as usize + col] = round(-2.);
        }
    }
    let mut shift = Vec::new();
    let mut scale = Vec::new();
    let mut gate = Vec::new();
    let mut modulation = Vec::new();
    let mut addcmul = Vec::new();
    for (row, &p) in SELECTED.iter().enumerate() {
        for col in 0..h {
            let i = row * h + col;
            let s = table[p * w + c.shift() as usize + col];
            let m = table[p * w + c.scale() as usize + col];
            let g = table[p * w + c.gate() as usize + col];
            shift.push(s);
            scale.push(m);
            gate.push(g);
            let factor = decode(round(add(1., decode(m))));
            let product = decode(round(mul(decode(x[i]), factor)));
            modulation.push(round(add(product, decode(s))));
            addcmul.push(round(add(decode(x[i]), mul(decode(g), decode(u[i])))));
        }
    }
    assert_eq!(
        modulation[0],
        round(0.),
        "factor BF16 boundary counterexample"
    );
    assert_eq!(
        addcmul[2 * h],
        round(2.046875),
        "addcmul must not round product to BF16"
    );
    if h > 1 {
        assert_eq!(
            modulation[h + 1],
            round(0.),
            "product BF16 boundary counterexample"
        );
    }
    if h > 2 {
        assert_eq!(
            addcmul[3 * h + 2],
            round(f32::INFINITY),
            "separate F32 overflow must survive cancellation"
        );
        let fused =
            decode(table[at(c.gate(), 2)]).mul_add(decode(u[3 * h + 2]), decode(x[3 * h + 2]));
        assert!(fused.is_finite(), "single FMA would cancel before overflow");
    }
    Data {
        inputs: BTreeMap::from([
            (X, x),
            (TABLE, table),
            (U, u),
            (SHIFT, shift),
            (SCALE, scale),
            (GATE, gate),
        ]),
        modulation,
        addcmul,
    }
}

pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for width in [1u64, 7, 5376] {
        let c = Config { width };
        let raw = serde_json::to_value(&c).unwrap();
        let family = TypedFamilyRegistration::new(Family {
            id: id("family.compact.bf16.fixture"),
        })
        .prepare_with_profile(&raw, &id(PROFILE))
        .unwrap();
        let mut options = ProgramPlanCompileOptions::new(c.specs()).unwrap();
        for value in [SCALED, MOD, EXP_MOD, ADD, EXP_ADD] {
            options
                .insert_tensor_spec(id(value), tensor(vec![ROWS, width]))
                .unwrap();
        }
        let component = CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            &TensorComponentPolicy::new(
                runtime.descriptor().total_memory_bytes,
                runtime
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap(),
            materializers,
            &options,
        )
        .unwrap();
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                16 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 8 << 20,
                    transfer_staging_bytes: 1 << 20,
                    output_bytes: 4 << 20,
                    weight_backing_bytes: 1 << 20,
                },
            )
            .unwrap(),
            64,
        )
        .unwrap();
        let mut session = component
            .initialize(
                Arc::clone(&runtime),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.compact.init"),
                    request_id: id("request.compact.init"),
                    transaction_id: id("transaction.compact.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut previous: Option<Vec<u16>> = None;
        let mut held = None;
        for trial in 0..2 {
            let expected = data(&c, trial);
            let inputs = expected
                .inputs
                .iter()
                .map(|(name, words)| {
                    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                    let mut backing = host
                        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                        .unwrap();
                    backing.as_mut().copy_from_slice(&bytes);
                    let s = c.specs().remove(&id(*name)).unwrap();
                    ManagedHostTensor::new(
                        id(*name),
                        ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout).unwrap(),
                        backing.freeze(),
                    )
                    .unwrap()
                })
                .collect();
            let outputs = session
                .execute(
                    id(format!("run.compact.{width}.{trial}")),
                    id(format!("request.compact.{width}.{trial}")),
                    inputs,
                )
                .unwrap();
            let words = |name: &str| {
                let output = outputs.iter().find(|o| o.value_id() == &id(name)).unwrap();
                assert_eq!(output.tensor().dimensions(), &[ROWS, width]);
                assert_eq!(output.tensor().element_type(), ElementType::Bf16);
                assert!(host.owns(output.bytes()));
                output
                    .bytes()
                    .bytes()
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            for (new, old, oracle) in [
                (MOD, EXP_MOD, &expected.modulation),
                (ADD, EXP_ADD, &expected.addcmul),
            ] {
                let actual = words(new);
                let expanded = words(old);
                for (i, ((&a, &b), &o)) in actual.iter().zip(&expanded).zip(oracle).enumerate() {
                    exact_equal(a, o, new, i);
                    exact_equal(a, b, "compact vs actual expanded graph", i);
                }
            }
            let actual = words(ADD);
            if let Some(previous) = previous.as_ref() {
                for i in 0..5 * width as usize {
                    exact_equal(actual[i], previous[i], "other request remains unchanged", i);
                }
                assert_ne!(
                    &actual[5 * width as usize..],
                    &previous[5 * width as usize..]
                );
            }
            previous = Some(actual);
            if trial == 1 {
                held = Some(
                    outputs
                        .iter()
                        .find(|o| o.value_id() == &id(MOD))
                        .unwrap()
                        .bytes()
                        .clone(),
                );
            }
            drop(outputs);
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::InputPreprocess),
                0
            );
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|failure| panic!("compact BF16 close: {:?}", failure.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        let output_bytes = host.backing_bytes((ROWS * width * 2) as usize).unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
        drop(held);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
