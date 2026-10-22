//! Managed whole-graph composition test, not a full-transformer numerical oracle.
//! Analytic entrance weights produce exact text/image/time intermediates. Two
//! independently invoked existing block families establish the layer chain;
//! the existing F64 LayerNorm oracle checks the final target-only projection.
//! This fixture does not stand in for text encoding or real scheduler inputs.

use super::*;
use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use serde_json::json;

#[path = "../../../tests/vnext_tensor_reference_runtime/layer_norm_oracle.rs"]
mod layer_norm_oracle;
#[path = "../../../examples/support/tensor_component.rs"]
mod shared;

type F = QwenImage21TransformerFamily;
const H: usize = 8;
const TEXT: usize = 2;
const IMAGE: usize = 4;
const SEQUENCE: usize = TEXT + IMAGE;
const BATCH: usize = 2;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}
fn bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}
fn bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

struct Archive {
    _directory: tempfile::TempDir,
    source: SafetensorsArchive,
}
impl Archive {
    fn new(tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>)>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let views = tensors
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::BF16,
                        shape.clone(),
                        bytes,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::tensor::serialize_to_file(
            views,
            &None,
            &directory.path().join("weights.safetensors"),
        )
        .unwrap();
        let weight_map = tensors
            .keys()
            .map(|name| (name.as_str(), "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            directory.path().join("weights.index.json"),
            serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        let source =
            SafetensorsArchive::open_with_index(directory.path(), "weights.index.json").unwrap();
        Self {
            _directory: directory,
            source,
        }
    }
}

struct Fixture {
    config: QwenImage21TransformerConfig,
    full: Archive,
    layers: [Archive; 2],
    input: BTreeMap<ProgramValueId, Vec<u8>>,
    joint: Vec<u16>,
    conditions: [Vec<u16>; 4],
}

impl Fixture {
    fn new() -> Self {
        let transformer = QwenImage21Config::from_hf_json(
            &serde_json::to_vec(&json!({
                "_class_name":"QwenImage21Transformer2DModel", "num_layers":2,
                "num_attention_heads":1, "attention_head_dim":8, "context_in_dim":8,
                "in_channels":8, "out_channels":8, "patch_size":1, "mlp_ratio":3,
                "axes_dims_rope":[2,2,4], "causal_condition":true, "eps":0.75
            }))
            .unwrap(),
            StructureLimits::default(),
        )
        .unwrap();
        let config = QwenImage21TransformerConfig::new(
            transformer,
            QwenImage21T2iLayout::new(TEXT as u32, 2, 2, vec![2, 1]).unwrap(),
        )
        .unwrap();
        let inventory =
            ExpectedWeights::from_config(config.transformer(), StructureLimits::default()).unwrap();
        let tensors = inventory
            .tensors()
            .values()
            .map(|tensor| {
                let shape = tensor
                    .dimensions()
                    .iter()
                    .map(|d| *d as usize)
                    .collect::<Vec<_>>();
                let data = (0..shape.iter().product())
                    .map(|index| bits(weight(tensor.external_name(), &shape, index)))
                    .collect::<Vec<_>>();
                (tensor.external_name().to_owned(), (shape, bytes(&data)))
            })
            .collect::<BTreeMap<_, _>>();
        let full = Archive::new(&tensors);
        let layers = std::array::from_fn(|layer| {
            let selected = Block::WEIGHT_NAMES
                .into_iter()
                .map(|name| {
                    let source = name.replacen(
                        "transformer_blocks.0.",
                        &format!("transformer_blocks.{layer}."),
                        1,
                    );
                    (name.to_owned(), tensors[&source].clone())
                })
                .collect();
            Archive::new(&selected)
        });

        // RMS(+/- .5, eps=.75) is exactly +/- .5. Dense*32 -> +/-16;
        // GELU-tanh saturates to 16 or signed zero, then Dense/16 -> 1 or +0.
        let text_positive = |row: usize, column: usize| match row {
            0 => column % 2 == 0,
            1 => column < 4,
            2 => column % 3 == 0,
            3 => column % 2 == 1, // invalid key, but still an executing query
            _ => unreachable!(),
        };
        let text = (0..BATCH * TEXT * H)
            .map(|i| {
                bits(if text_positive(i / H, i % H) {
                    0.5
                } else {
                    -0.5
                })
            })
            .collect::<Vec<_>>();
        let latent = (0..BATCH * IMAGE * H)
            .map(|i| bits((((i / H) * 3 + (i % H) * 5) % 11) as f32 * 0.25 - 1.25))
            .collect::<Vec<_>>();
        let mut joint = Vec::new();
        for batch in 0..BATCH {
            for row in 0..TEXT {
                joint.extend((0..H).map(|column| {
                    bits(if text_positive(batch * TEXT + row, column) {
                        1.0
                    } else {
                        0.0
                    })
                }));
            }
            joint.extend_from_slice(&latent[batch * IMAGE * H..(batch + 1) * IMAGE * H]);
        }
        // Explicit synthetic feature rows test the [sample0,sample1,t0]
        // selection ABI. They are not claimed to be real sinusoidal features.
        // linear1*8 and two positive saturated BF16 SiLUs yield temb=16,32,8.
        let mut time = vec![0_u16; 3 * 256];
        time[0] = bits(2.0);
        time[256] = bits(4.0);
        time[512] = bits(1.0);
        let mut conditions: [Vec<u16>; 4] = std::array::from_fn(|_| Vec::new());
        for batch in 0..BATCH {
            for row in 0..SEQUENCE {
                let embedding = if row < TEXT {
                    8.0
                } else {
                    16.0 * (batch + 1) as f32
                };
                for (block, destination) in conditions.iter_mut().enumerate() {
                    destination.extend(
                        (0..H).map(|column| bits(embedding * modulation_weight(block, column))),
                    );
                }
            }
        }
        let frequencies = (0..SEQUENCE * H / 2)
            .flat_map(|_| [1.0_f32, 0.0_f32])
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let input = BTreeMap::from([
            (id(F::LATENTS_ID), bytes(&latent)),
            (id(F::TEXT_ID), bytes(&text)),
            (id(F::TIME_FEATURES_ID), bytes(&time)),
            (id(F::FREQUENCIES_ID), frequencies),
        ]);
        Self {
            config,
            full,
            layers,
            input,
            joint,
            conditions,
        }
    }

    fn block_input(&self, hidden: &[u16]) -> BTreeMap<ProgramValueId, Vec<u8>> {
        let mut input = BTreeMap::from([
            (id(Block::INPUT_ID), bytes(hidden)),
            (
                id(Block::FREQUENCIES_ID),
                self.input[&id(F::FREQUENCIES_ID)].clone(),
            ),
        ]);
        for (name, value) in [
            Block::SCALE1_ID,
            Block::GATE1_ID,
            Block::SCALE2_ID,
            Block::GATE2_ID,
        ]
        .into_iter()
        .zip(&self.conditions)
        {
            input.insert(id(name), bytes(value));
        }
        input
    }

    fn target_rows(&self, hidden: &[u16], start: usize) -> Vec<u16> {
        (0..BATCH)
            .flat_map(|batch| {
                hidden[(batch * SEQUENCE + start) * H..(batch * SEQUENCE + start + IMAGE) * H]
                    .iter()
                    .copied()
            })
            .collect()
    }
}

fn modulation_weight(block: usize, column: usize) -> f32 {
    let sign = if column % 2 == 0 { 1.0 } else { -1.0 };
    match block {
        0 => sign / 64.0,
        1 => 1.0 / 16.0,
        2 => -sign / 128.0,
        3 => 0.25,
        _ => unreachable!(),
    }
}

fn weight(name: &str, shape: &[usize], index: usize) -> f32 {
    if name.ends_with("norm_q.weight") || name.ends_with("norm_k.weight") {
        return 1.0;
    }
    if name == "txt_in.text_norm.weight" {
        return 0.0;
    }
    let row = index / shape[1];
    let column = index % shape[1];
    match name {
        "img_in.weight" | "proj_out.weight" => {
            if row == column {
                1.0
            } else {
                0.0
            }
        }
        "txt_in.in_layer.weight" => {
            if row == column {
                32.0
            } else {
                0.0
            }
        }
        "txt_in.out_layer.weight" => {
            if row == column {
                1.0 / 16.0
            } else {
                0.0
            }
        }
        "time_text_embed.timestep_embedder.linear_1.weight" => {
            if row == 0 && column == 0 {
                8.0
            } else {
                0.0
            }
        }
        "time_text_embed.timestep_embedder.linear_2.weight" => {
            if row == 0 && column == 0 {
                1.0
            } else {
                0.0
            }
        }
        "modulation.1.weight" => {
            if column == 0 {
                modulation_weight(row / H, row % H)
            } else {
                0.0
            }
        }
        "norm_out.linear.weight" => 0.0, // final scale zero, final projection identity
        _ => {
            let layer = if name.starts_with("transformer_blocks.0.") {
                0
            } else {
                assert!(name.starts_with("transformer_blocks.1."));
                1
            };
            if name.ends_with("attn.to_q.weight") || name.ends_with("attn.to_k.weight") {
                0.0
            } else if name.ends_with("attn.to_v.weight") {
                if row == column {
                    0.5 * (layer + 1) as f32
                } else {
                    0.0
                }
            } else if name.ends_with("attn.to_out.0.weight") {
                if column == (row + 1 + 2 * layer) % H {
                    (layer + 1) as f32
                } else {
                    0.0
                }
            } else if name.ends_with("img_mlp.gate_layer.weight") {
                if row == column {
                    1.0
                } else {
                    0.0
                }
            } else if name.ends_with("img_mlp.proj.weight") {
                if row == column {
                    0.5
                } else {
                    0.0
                }
            } else if name.ends_with("img_mlp.out.weight") {
                if column == (row + 2 + layer) % H {
                    0.25 * (layer + 1) as f32
                } else {
                    0.0
                }
            } else {
                panic!("unhandled fixture weight {name}")
            }
        }
    }
}

/// Executes genuine admission/managed uploads/readback/retirement, sharing the
/// existing Stored initialization helper. Output ownership outlives device close.
fn execute(
    family: &PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: &SafetensorsArchive,
    input: BTreeMap<ProgramValueId, Vec<u8>>,
) -> BTreeMap<ProgramValueId, Vec<u16>> {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.transformer.execution")).unwrap();
    let runtime = composition.runtime().clone();
    let policy = shared::TensorPolicy {
        capacity_bytes: runtime.descriptor().total_memory_bytes,
        storage_profiles: runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    let ports = plan.payload().tensor_ports().unwrap();
    assert_eq!(ports.outputs().len(), family.program().outputs().len());
    let input_bytes = input.values().map(Vec::len).sum::<usize>();
    let output_bytes = ports
        .outputs()
        .iter()
        .map(|port| port.tensor().minimum_storage_bytes().unwrap() as usize)
        .sum::<usize>();
    let staging_bytes = 1024;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            input_bytes + output_bytes + staging_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                output_bytes,
                transfer_staging_bytes: staging_bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let weight_bytes = family
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .map(|component| component.physical_bytes().unwrap())
        .sum();
    let resources = shared::initialize_resources(
        runtime.clone(),
        family,
        plan,
        source,
        weight_bytes,
        &host,
        staging_bytes,
    )
    .unwrap();
    let outputs = {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            composition.registry().bind_plan(executable).unwrap(),
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .unwrap();
        let managed = ports
            .inputs()
            .iter()
            .map(|port| {
                let bytes = &input[port.value_id()];
                assert_eq!(
                    bytes.len(),
                    port.tensor().minimum_storage_bytes().unwrap() as usize
                );
                let mut owner = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                    .unwrap();
                owner.as_mut().copy_from_slice(bytes);
                ManagedHostTensor::new(
                    port.value_id().clone(),
                    port.tensor().clone(),
                    owner.freeze(),
                )
                .unwrap()
            })
            .collect();
        executor
            .execute(
                id("run.transformer.execution"),
                id("request.transformer.execution"),
                managed,
            )
            .unwrap()
    };
    assert_eq!(host.usage().unwrap().allocated_bytes(), output_bytes);
    match PlanRuntimeResources::close(resources)
        .unwrap_or_else(|_| panic!("managed transformer resource close failed"))
    {
        PlanRuntimeCloseOutcome::Closed(receipt) => {
            assert_eq!(receipt.released_static_resources(), 1)
        }
        PlanRuntimeCloseOutcome::Referenced { strong_count, .. } => {
            panic!("unexpected resource owner count {strong_count}")
        }
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), output_bytes);
    let result = outputs
        .iter()
        .map(|output| {
            let values = output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                .collect();
            (output.value_id().clone(), values)
        })
        .collect();
    drop(outputs);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    result
}

#[test]
fn two_layer_managed_transformer_matches_explicit_t0_and_layer_composition_then_target_only_norm() {
    let fixture = Fixture::new();
    let block_config = fixture.config.block_config().unwrap();
    let block = TypedFamilyRegistration::new(Block::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&block_config).unwrap(),
            &profile_id(Block::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let block_options =
        || ProgramPlanCompileOptions::new(block_config.input_tensor_specs().unwrap()).unwrap();
    // Prepare the composition expectation before executing the full candidate.
    // The full-family remapper, transport ops and entrance graph are not used.
    let layer0 = execute(
        &block,
        block_options(),
        &fixture.layers[0].source,
        fixture.block_input(&fixture.joint),
    );
    let first = &layer0[&id(Block::OUTPUT_ID)];
    assert_ne!(
        first, &fixture.joint,
        "the nonzero residual branches must execute"
    );
    let layer1 = execute(
        &block,
        block_options(),
        &fixture.layers[1].source,
        fixture.block_input(first),
    );
    let last = &layer1[&id(Block::OUTPUT_ID)];
    assert_ne!(last, first, "layer 1 has distinct nonzero weights");
    let targets = fixture.target_rows(last, TEXT);
    let expected = layer_norm_oracle::prepare(&targets, H, 0.75).unwrap();
    // A prefix-inclusive slice has the same byte count but the wrong semantics.
    let wrong_slice = layer_norm_oracle::prepare(&fixture.target_rows(last, 0), H, 0.75).unwrap();
    let first_only =
        layer_norm_oracle::prepare(&fixture.target_rows(first, TEXT), H, 0.75).unwrap();
    let family = TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&fixture.config).unwrap(),
            &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let result = execute(
        &family,
        fixture.config.compile_options().unwrap(),
        &fixture.full.source,
        fixture.input.clone(),
    );
    assert_eq!(result.len(), 1);
    let velocity = &result[&id(F::VELOCITY_ID)];
    assert_eq!(velocity.len(), BATCH * IMAGE * H);
    expected.check(velocity).unwrap();
    assert!(
        wrong_slice.check(velocity).is_err(),
        "the fixture must distinguish prefix leakage from target slicing"
    );
    assert!(
        first_only.check(velocity).is_err(),
        "the fixture must distinguish a skipped second layer"
    );
    assert_ne!(
        &velocity[..IMAGE * H],
        &velocity[IMAGE * H..],
        "batch rows and valid-text geometry are distinct"
    );
}
