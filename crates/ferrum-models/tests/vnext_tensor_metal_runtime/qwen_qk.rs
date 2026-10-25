//! Execute the real Qwen Q/K family graph with small deterministic weights.
//! This is component qualification, not a full-model image-quality claim.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentWeightSource, TensorComponentPolicy,
};
use ferrum_models::qwen_image21::{
    QwenImage21Block0QkNormFamily as Qk, QwenImage21Config, QwenImage21QkFrontend, StructureLimits,
};
use transforms::{decode, initialization, oracle, spec, upload_values};

struct Source {
    values: [Vec<f32>; 4],
    bytes: [Vec<u8>; 4],
}
impl Source {
    fn new(hidden: usize, head: usize) -> Self {
        let values: [Vec<f32>; 4] = std::array::from_fn(|branch| {
            (0..if branch < 2 { hidden * hidden } else { head })
                .map(|i| {
                    if branch < 2 {
                        rounds(
                            (((i / hidden) * 11 + (i % hidden) * 17 + branch * 7) % 97) as f32
                                / 128.0
                                - 0.375,
                        )
                    } else {
                        rounds(0.5 + ((i * 3 + branch) % 13) as f32 / 8.0)
                    }
                })
                .collect()
        });
        let bytes = std::array::from_fn(|i| {
            values[i]
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect()
        });
        Self { values, bytes }
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let index = Qk::WEIGHT_NAMES
            .iter()
            .position(|name| component.external_names.iter().any(|n| n == name))
            .ok_or_else(|| invalid("unexpected Qwen Q/K weight"))?;
        WeightComponentPayload::new(
            component,
            Qk::WEIGHT_NAMES[index],
            "generated-small-qwen-qk",
            component.dimensions.clone(),
            component.physical_element_type(),
            self.bytes[index].as_slice(),
        )
    }
}
fn config(head: u64) -> QwenImage21Config {
    let axes = if head == 6 { [2, 2, 2] } else { [32, 32, 64] };
    QwenImage21Config::from_hf_json(&serde_json::to_vec(&serde_json::json!({
        "_class_name": "QwenImage21Transformer2DModel", "num_layers": 1, "num_attention_heads": 2,
        "attention_head_dim": head, "context_in_dim": 8, "in_channels": 4, "out_channels": 4,
        "patch_size": 1, "mlp_ratio": 3, "axes_dims_rope": axes, "causal_condition": true, "eps": 0.000001
    })).unwrap(), StructureLimits::default()).unwrap()
}
fn check_projection(input: &[f32], weight: &[f32], hidden: usize, actual: &[f32]) {
    for (index, &actual) in actual.iter().enumerate() {
        let row = index / hidden;
        let column = index % hidden;
        let (mut sum, mut absolute) = (0.0_f64, 0.0_f64);
        for k in 0..hidden {
            let product = input[row * hidden + k] as f64 * weight[column * hidden + k] as f64;
            sum += product;
            absolute += product.abs();
        }
        let factor = (2 * hidden) as f64 * 2_f64.powi(-24);
        let error = factor / (1.0 - factor) * absolute;
        let reference = (2 * hidden) as f64 * f64::EPSILON;
        let bound = error
            + (sum.abs() + error) / 256.0
            + 2_f64.powi(-134)
            + reference / (1.0 - reference) * absolute;
        assert!(
            actual.is_finite() && (actual as f64 - sum).abs() <= bound,
            "Q/K projection [{row},{column}]: {actual} != {sum}"
        );
    }
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    for head in [6, 128] {
        for frontend in [QwenImage21QkFrontend::Norm, QwenImage21QkFrontend::Rotary] {
            let config = config(head);
            let hidden = config.hidden_size();
            let input = spec(vec![6, hidden], ElementType::Bf16);
            let frequencies = spec(vec![3, head / 2, 2], ElementType::F32);
            let mut specs = BTreeMap::from([(id(Qk::INPUT_ID), input.clone())]);
            if frontend == QwenImage21QkFrontend::Rotary {
                specs.insert(id(Qk::FREQUENCIES_ID), frequencies.clone());
            }
            let prepared = TypedFamilyRegistration::new(Qk::with_frontend(frontend).unwrap())
                .prepare_with_profile(
                    &serde_json::to_value(&config).unwrap(),
                    &id(frontend.metal_profile_id()),
                )
                .unwrap();
            let component = CompiledTensorComponent::compile(
                prepared,
                registry,
                catalog,
                &policy,
                &materializers,
                &ProgramPlanCompileOptions::new(specs).unwrap(),
            )
            .unwrap();
            let source = Source::new(hidden as usize, head as usize);
            let expected_weights = source.values.clone();
            let mut session = component
                .initialize(
                    runtime.clone(),
                    host.clone(),
                    ComponentWeightSource::Stored(&source),
                    initialization("metal.qwen.qk"),
                )
                .unwrap();
            drop(source); // Imported weights must remain valid independently of borrowed payloads.
            let x: Vec<_> = (0..6 * hidden)
                .map(|i| rounds(((i * 53 + i / hidden * 17) % 127) as f32 / 32.0 - 2.0))
                .collect();
            let freq: Vec<_> = (0..3 * head)
                .map(|i| ((i * 19) % 113) as f32 / 73.0 - 0.5)
                .collect();
            let mut previous: Option<Vec<ManagedHostBytes>> = None;
            for repeat in 0..2 {
                let mut inputs = vec![upload_values(host, id(Qk::INPUT_ID), &input, &x)];
                if frontend == QwenImage21QkFrontend::Rotary {
                    inputs.push(upload_values(
                        host,
                        id(Qk::FREQUENCIES_ID),
                        &frequencies,
                        &freq,
                    ));
                }
                let outputs = session
                    .execute(
                        id(format!("run.metal.qwen.qk.{repeat}")),
                        id(format!("request.metal.qwen.qk.{repeat}")),
                        inputs,
                    )
                    .unwrap();
                let get = |name: &str| {
                    decode(
                        outputs
                            .iter()
                            .find(|output| output.value_id().as_str() == name)
                            .unwrap()
                            .bytes()
                            .bytes(),
                    )
                };
                for (branch, (projection, norm, rotary)) in [
                    (Qk::QUERY_ID, Qk::QUERY_NORMALIZED_ID, Qk::QUERY_ROTATED_ID),
                    (Qk::KEY_ID, Qk::KEY_NORMALIZED_ID, Qk::KEY_ROTATED_ID),
                ]
                .into_iter()
                .enumerate()
                {
                    let projected = get(projection);
                    check_projection(&x, &expected_weights[branch], hidden as usize, &projected);
                    let normalized = get(norm);
                    oracle::norm(
                        MetalTensorOperation::HeadRmsNorm,
                        &projected,
                        &expected_weights[branch + 2],
                        head as usize,
                        1e-6,
                        &normalized,
                    );
                    if frontend == QwenImage21QkFrontend::Rotary {
                        oracle::rope(
                            MetalTensorOperation::AdjacentRope,
                            &normalized,
                            &freq,
                            &[],
                            hidden as usize,
                            head as usize,
                            3,
                            &get(rotary),
                        );
                    }
                }
                if let Some(previous) = &previous {
                    assert_eq!(previous.len(), outputs.len());
                    for (a, b) in previous.iter().zip(&outputs) {
                        assert_eq!(a.bytes(), b.bytes().bytes());
                    }
                }
                previous = Some(
                    outputs
                        .iter()
                        .map(|output| output.bytes().clone())
                        .collect(),
                );
            }
            assert!(matches!(
                session.try_release_idle_dynamic_backing().unwrap(),
                IdleDynamicReleaseOutcome::Released(_)
            ));
            assert!(matches!(
                session
                    .close()
                    .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
            assert!(previous
                .as_ref()
                .unwrap()
                .iter()
                .all(|bytes| decode(bytes.bytes()).iter().all(|v| v.is_finite())));
            drop(previous);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            eprintln!("Metal Qwen Q/K {frontend:?}, heads=2 head_dim={head} batch=2 sequence=3: stored weights, every numerical boundary, repeat and retirement passed");
        }
    }
}
