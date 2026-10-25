//! Real Qwen attention graph and weight names, with bounded generated weights.
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentWeightSource, TensorComponentPolicy,
};
use ferrum_models::qwen_image21::{
    QwenImage21AttentionConfig, QwenImage21AttentionKeyRun, QwenImage21AttentionLayout,
    QwenImage21AttentionSegment, QwenImage21Block0AttentionFamily as Family, QwenImage21Config,
    StructureLimits,
};
use transforms::{decode, initialization, oracle, spec, upload_values};

struct Source {
    qk: qwen_qk::Source,
    values: [Vec<f32>; 2],
    bytes: [Vec<u8>; 2],
}
impl Source {
    fn new(hidden: usize, head: usize) -> Self {
        let values: [Vec<f32>; 2] = std::array::from_fn(|branch| {
            (0..hidden * hidden)
                .map(|i| {
                    rounds(((i * 17 + (i / hidden) * 19 + branch * 23) % 61) as f32 / 128.0 - 0.25)
                })
                .collect()
        });
        let bytes = std::array::from_fn(|i| {
            values[i]
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                .collect()
        });
        Self {
            qk: qwen_qk::Source::new(hidden, head),
            values,
            bytes,
        }
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some(index) = Family::WEIGHT_NAMES[4..]
            .iter()
            .position(|name| component.external_names.iter().any(|n| n == name))
        else {
            return self.qk.component(component);
        };
        WeightComponentPayload::new(
            component,
            Family::WEIGHT_NAMES[4 + index],
            "generated-small-qwen-attention",
            component.dimensions.clone(),
            component.physical_element_type(),
            self.bytes[index].as_slice(),
        )
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
    for head in [8, 128] {
        let transformer = QwenImage21Config::from_hf_json(&serde_json::to_vec(&serde_json::json!({
            "_class_name": "QwenImage21Transformer2DModel", "num_layers": 1, "num_attention_heads": 2,
            "attention_head_dim": head, "context_in_dim": 8, "in_channels": 4, "out_channels": 4,
            "patch_size": 1, "mlp_ratio": 3, "axes_dims_rope": [2, 2, head-4], "causal_condition": true, "eps": 0.000001
        })).unwrap(), StructureLimits::default()).unwrap();
        let hidden = transformer.hidden_size();
        let case = attention::Case {
            sequence: 33,
            heads: 2,
            kv_heads: 2,
            width: head as usize,
            segments: vec![(1, false), (32, true)],
            keys: vec![vec![(33, true)], vec![(2, false), (29, true), (2, false)]],
            scale: Family::scale_for_head_dim(head).unwrap(),
        };
        let layout = QwenImage21AttentionLayout::new(
            33,
            case.segments
                .iter()
                .map(|&(n, b)| QwenImage21AttentionSegment::new(n as u32, b))
                .collect(),
            case.keys
                .iter()
                .map(|runs| {
                    runs.iter()
                        .map(|&(n, v)| QwenImage21AttentionKeyRun::new(n as u32, v))
                        .collect()
                })
                .collect(),
        )
        .unwrap();
        let config = QwenImage21AttentionConfig::new(transformer, layout).unwrap();
        assert_eq!(config.attributes().unwrap(), case.attributes());
        let input = spec(vec![66, hidden], ElementType::Bf16);
        let frequencies = spec(vec![33, head / 2, 2], ElementType::F32);
        let prepared = TypedFamilyRegistration::new(Family::new().unwrap())
            .prepare_with_profile(
                &serde_json::to_value(config).unwrap(),
                &id(Family::METAL_PROFILE_ID),
            )
            .unwrap();
        let component = CompiledTensorComponent::compile(
            prepared,
            registry,
            catalog,
            &policy,
            &WeightMaterializerRegistry::identity_only().unwrap(),
            &ProgramPlanCompileOptions::new(BTreeMap::from([
                (id(Family::INPUT_ID), input.clone()),
                (id(Family::FREQUENCIES_ID), frequencies.clone()),
            ]))
            .unwrap(),
        )
        .unwrap();
        let source = Source::new(hidden as usize, head as usize);
        let qk_weights = source.qk.values.clone();
        let vo_weights = source.values.clone();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&source),
                initialization("metal.qwen.attention"),
            )
            .unwrap();
        drop(source);
        let x: Vec<_> = (0..66 * hidden)
            .map(|i| rounds(((i * 53 + i / hidden * 17) % 127) as f32 / 32.0 - 2.0))
            .collect();
        let freq: Vec<_> = (0..33 * head)
            .map(|i| ((i * 19) % 113) as f32 / 73.0 - 0.5)
            .collect();
        let mut previous: Option<Vec<ManagedHostBytes>> = None;
        for repeat in 0..2 {
            let outputs = session
                .execute(
                    id(format!("run.metal.qwen.attention.{repeat}")),
                    id(format!("request.metal.qwen.attention.{repeat}")),
                    vec![
                        upload_values(host, id(Family::INPUT_ID), &input, &x),
                        upload_values(host, id(Family::FREQUENCIES_ID), &frequencies, &freq),
                    ],
                )
                .unwrap();
            assert_eq!(outputs.len(), 9);
            let get = |name: &str| {
                decode(
                    outputs
                        .iter()
                        .find(|o| o.value_id().as_str() == name)
                        .unwrap()
                        .bytes()
                        .bytes(),
                )
            };
            for (branch, (projection, norm, rotary)) in [
                (
                    Family::QUERY_ID,
                    Family::QUERY_NORMALIZED_ID,
                    Family::QUERY_ROTATED_ID,
                ),
                (
                    Family::KEY_ID,
                    Family::KEY_NORMALIZED_ID,
                    Family::KEY_ROTATED_ID,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let projected = get(projection);
                qwen_qk::check_projection(&x, &qk_weights[branch], hidden as usize, &projected);
                let normalized = get(norm);
                oracle::norm(
                    MetalTensorOperation::HeadRmsNorm,
                    &projected,
                    &qk_weights[branch + 2],
                    head as usize,
                    1e-6,
                    &normalized,
                );
                oracle::rope(
                    MetalTensorOperation::AdjacentRope,
                    &normalized,
                    &freq,
                    &[],
                    hidden as usize,
                    head as usize,
                    33,
                    &get(rotary),
                );
            }
            let value = get(Family::VALUE_ID);
            qwen_qk::check_projection(&x, &vo_weights[0], hidden as usize, &value);
            let attended = get(Family::ATTENTION_ID);
            case.check(
                &[
                    get(Family::QUERY_ROTATED_ID),
                    get(Family::KEY_ROTATED_ID),
                    value,
                ],
                &attended,
            );
            qwen_qk::check_projection(
                &attended,
                &vo_weights[1],
                hidden as usize,
                &get(Family::OUTPUT_ID),
            );
            if let Some(previous) = &previous {
                for (a, b) in previous.iter().zip(&outputs) {
                    assert_eq!(a.bytes(), b.bytes().bytes());
                }
            }
            previous = Some(outputs.iter().map(|o| o.bytes().clone()).collect());
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
            .all(|b| decode(b.bytes()).iter().all(|v| v.is_finite())));
        drop(previous);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!("Metal Qwen attention: B=2 S=33 H=2 D={head}, all nine graph boundaries, repeat and retirement passed");
    }
}
