use super::super::input_projection as io;
use super::*;
use ferrum_execution::component::ComponentWeightSource;
use ferrum_execution::TensorExecutionOutput;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
mod completed;
mod continuations;
#[cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-sage-vsa-fp4"))]
mod cuda;

fn prefix_isolation(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    p: &FastH3TransformerPackage,
    layout: &FastH3VsaLayout,
    values: &[f32],
) -> Vec<u8> {
    type Prefix = FastH3PrefixAttentionFamily;
    let limits = capacities().attention_core.prefix;
    let (config, source) =
        FastH3PrefixAttentionSource::prepare(layout, values.len() as u64, p.config(), h, limits)
            .unwrap();
    let shape = config.input_shape();
    let inputs = [Prefix::QUERY_ID, Prefix::KEY_ID, Prefix::VALUE_ID]
        .into_iter()
        .map(|name| {
            let bytes = values
                .iter()
                .flat_map(|&value| {
                    std::iter::repeat_n(
                        if name == Prefix::VALUE_ID {
                            bf16::from_f32(value).to_bits()
                        } else {
                            0
                        },
                        (config.padded_rows() * config.columns()) as usize,
                    )
                    .flat_map(u16::to_le_bytes)
                })
                .collect::<Vec<_>>();
            let flat = io::managed(
                h,
                name,
                shape[0] * shape[1],
                shape[2],
                ElementType::Bf16,
                &bytes,
            );
            ManagedHostTensor::new(
                flat.value_id().clone(),
                ResolvedTensorSpec::new(
                    shape.clone(),
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                flat.bytes().clone(),
            )
            .unwrap()
        })
        .collect();
    let out = run(
        r,
        h,
        Prefix::new(limits).unwrap(),
        &config,
        Prefix::PROFILE_ID,
        config.compile_options(limits).unwrap(),
        ComponentWeightSource::Stored(&source),
        inputs,
    );
    out.iter()
        .find(|o| o.value_id().as_str() == Prefix::OUTPUT_ID)
        .unwrap()
        .bytes()
        .bytes()
        .to_vec()
}

fn limits_batch() -> FastH3DenoisingLimits {
    FastH3DenoisingLimits {
        input: FastH3InputLimits {
            maximum_rows: 32768,
            maximum_segments: 32,
            maximum_metadata_bytes: 1 << 20,
            maximum_stored_bytes: 16 << 20,
        },
        backbone: FastH3DitBackboneLimits {
            block: capacities(),
            maximum_layers: 2,
            maximum_plan_nodes: 256,
            maximum_stored_bytes: 16 << 20,
            maximum_metadata_bytes: 8 << 20,
        },
        maximum_plan_nodes: 263,
        maximum_stored_bytes: 32 << 20,
        maximum_metadata_bytes: 16 << 20,
    }
}
fn package() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    package_with(|_| {})
}
fn package_with(
    adjust: impl FnOnce(&mut Fixture),
) -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    let c = config_with_attention(128, 2, 128, 128);
    let mut f = Fixture::with_attention(&c);
    f.metadata.remove("skip_blocks");
    for (name, tensor) in f.packed.clone() {
        f.packed.insert(
            name.replace("transformer_blocks.0", "transformer_blocks.1"),
            tensor,
        );
    }
    for name in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.to_gate_compress",
        "ff.net.0.proj",
        "ff.net.2",
    ] {
        f.dense
            .remove(&format!("transformer_blocks.1.{name}.weight"));
    }
    for (name, (_, _, bytes)) in &mut f.dense {
        for (i, b) in bytes.chunks_exact_mut(2).enumerate() {
            let x = if name.contains("norm") && name.ends_with("weight") {
                1.
            } else {
                ((i * 7 % 17) as f32 - 8.) / 64.
            };
            b.copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
        }
    }
    adjust(&mut f);
    let (dir, p) = f.open_with_config(
        &c,
        FastH3WeightLimits {
            maximum_tensors: 150,
            maximum_source_bytes: 8 << 20,
            ..limits()
        },
    );
    (dir, Arc::new(p.unwrap()))
}
fn run<F: ModelFamilyProvider>(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    f: F,
    c: &F::Config,
    profile: &str,
    options: ProgramPlanCompileOptions,
    source: ComponentWeightSource<'_>,
    inputs: Vec<ManagedHostTensor>,
) -> Vec<TensorExecutionOutput> {
    let mut s = io::session(r, io::prepare(f, c, profile), options, source, h);
    let out = io::execute(&mut s, inputs);
    io::close(s);
    out
}
fn input(
    h: &HostMemoryAccount,
    name: &str,
    rows: u64,
    cols: u64,
    dtype: ElementType,
    salt: usize,
) -> ManagedHostTensor {
    let values = (0..rows * cols).map(|i| ((i as usize * 3 + salt * 11) % 31) as f32 / 16. - 1.);
    let bytes = if dtype == ElementType::F32 {
        values.flat_map(f32::to_le_bytes).collect::<Vec<_>>()
    } else {
        values
            .flat_map(|x| bf16::from_f32(x).to_bits().to_le_bytes())
            .collect()
    };
    io::managed(h, name, rows, cols, dtype, &bytes)
}
fn copy(h: &HostMemoryAccount, output: &TensorExecutionOutput, name: &str) -> ManagedHostTensor {
    let spec = output.tensor();
    io::managed(
        h,
        name,
        spec.dimensions()[0],
        spec.dimensions()[1],
        spec.element_type(),
        output.bytes().bytes(),
    )
}
fn project_pack(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    p: &FastH3TransformerPackage,
    layout: &FastH3PackedLayout,
    inputs: &[Vec<ManagedHostTensor>],
) -> TensorExecutionOutput {
    let l = limits_batch().input;
    let b = inputs.len() as u64;
    let c = FastH3InputProjectionConfig::media_batch_from_package(p, layout, b, l).unwrap();
    let source = FastH3InputProjectionSource::new(p, &c, h, l).unwrap();
    let names = [
        FastH3DenoisingFamily::VIDEO_INPUT_ID,
        FastH3DenoisingFamily::AUDIO_INPUT_ID,
        FastH3DenoisingFamily::TEXT_INPUT_ID,
    ];
    let mut joined = names
        .iter()
        .map(|&name| {
            let parts = inputs
                .iter()
                .map(|v| v.iter().find(|v| v.value_id().as_str() == name).unwrap())
                .collect::<Vec<_>>();
            let shape = parts[0].tensor().dimensions();
            let bytes = parts
                .iter()
                .flat_map(|p| p.bytes().bytes().iter().copied())
                .collect::<Vec<_>>();
            io::managed(
                h,
                name,
                b * shape[0],
                shape[1],
                parts[0].tensor().element_type(),
                &bytes,
            )
        })
        .collect::<Vec<_>>();
    let text = joined.pop().unwrap();
    let projected = run(
        r,
        h,
        FastH3InputProjectionFamily::new(l).unwrap(),
        &c,
        c.phase().profile_id(),
        c.compile_options().unwrap(),
        ComponentWeightSource::Stored(&source),
        joined,
    );
    let packing =
        FastH3InputPackingConfig::from_batched_layout(layout, c.hidden_size(), b, l).unwrap();
    let get = |name| {
        projected
            .iter()
            .find(|o| o.value_id().as_str() == name)
            .unwrap()
    };
    let operands = vec![
        text,
        copy(
            h,
            get(FastH3InputProjectionFamily::VIDEO_OUTPUT_ID),
            FastH3InputPackingFamily::VIDEO_ID,
        ),
        copy(
            h,
            get(FastH3InputProjectionFamily::AUDIO_OUTPUT_ID),
            FastH3InputPackingFamily::AUDIO_ID,
        ),
    ];
    run(
        r,
        h,
        FastH3InputPackingFamily::new(l).unwrap(),
        &packing,
        FastH3InputPackingFamily::PROFILE_ID,
        packing.compile_options().unwrap(),
        ComponentWeightSource::None,
        operands,
    )
    .pop()
    .unwrap()
}

#[test]
fn same_shape_b2_packing_keeps_request_segments_and_original_weight_contract() {
    let (_dir, p) = package();
    let layout = super::super::attention_core::layout();
    let h = host();
    let t = tags();
    let (one, s1, r1) = FastH3DenoisingSource::prepare(
        &p,
        &layout,
        &schedule(),
        &t,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        limits_batch(),
    )
    .unwrap();
    let (two, s2, r2) = FastH3DenoisingSource::prepare_batch(
        &p,
        &layout,
        &schedule(),
        &[&t, &t],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        limits_batch(),
    )
    .unwrap();
    let family = FastH3DenoisingFamily::new(limits_batch()).unwrap();
    // Compare actual schemas, including every map and physical interpretation;
    // no filtering by name/dtype or guessed batch-invariant source hash.
    assert_eq!(family.weights(&one).unwrap(), family.weights(&two).unwrap());
    let prepared = io::prepare(family, &two, two.profile_id());
    for (config, regions) in [(&one, &r1), (&two, &r2)] {
        for chunk_rows in [None, std::num::NonZeroU64::new(64)] {
            let config = config.clone().with_ffn_chunk_rows(chunk_rows);
            let expected = io::prepare(
                FastH3DenoisingFamily::new(limits_batch()).unwrap(),
                &config,
                config.profile_id(),
            );
            let (plan, family) = FastH3DenoisingBatchPlan::prepare_with_family(
                config.clone(),
                limits_batch(),
                regions.clone(),
            )
            .unwrap();
            assert_eq!(family, expected);
            assert_eq!(
                plan.compile_options(),
                &config.compile_options(limits_batch()).unwrap()
            );
        }
    }
    assert!(
        FastH3DenoisingBatchPlan::prepare_with_family(one.clone(), limits_batch(), r2.clone(),)
            .is_err()
    );
    let options = two.compile_options(limits_batch()).unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("stored")
    };
    super::super::ffn::validate_graph_bindings(prepared.program(), schema, &options);
    for c in &schema.components {
        assert_eq!(
            s1.component(c).unwrap().bytes(),
            s2.component(c).unwrap().bytes()
        );
    }
    let r = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.reference.batch_packing").unwrap(),
        64 << 20,
    )
    .unwrap();
    let rows = one.media().rows();
    let widths = one.media().input_widths();
    let members = (0..2)
        .map(|n| {
            vec![
                input(
                    &h,
                    FastH3DenoisingFamily::VIDEO_INPUT_ID,
                    rows[0],
                    widths[0],
                    ElementType::F32,
                    n,
                ),
                input(
                    &h,
                    FastH3DenoisingFamily::AUDIO_INPUT_ID,
                    rows[1],
                    widths[1],
                    ElementType::F32,
                    n + 2,
                ),
                input(
                    &h,
                    FastH3DenoisingFamily::TEXT_INPUT_ID,
                    3,
                    one.backbone().hidden_size(),
                    ElementType::Bf16,
                    n + 4,
                ),
            ]
        })
        .collect::<Vec<_>>();
    let singles = members
        .iter()
        .map(|m| project_pack(&r, &h, &p, layout.packed(), std::slice::from_ref(m)))
        .collect::<Vec<_>>();
    let batch = project_pack(&r, &h, &p, layout.packed(), &members);
    let expected = singles
        .iter()
        .flat_map(|o| o.bytes().bytes().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(batch.bytes().bytes(), expected);
    assert_ne!(singles[0].bytes().bytes(), singles[1].bytes().bytes());
    assert_eq!(
        batch.tensor().dimensions(),
        [2 * layout.packed().sequence_rows() as u64, 128]
    );
    // Real prefix attention, with all keys equiprobable and distinct V per
    // request. A flattened cross-request key domain would mix 1 and -2.
    let prefix = prefix_isolation(&r, &h, &p, &layout, &[1., -2.]);
    let prefix_single = [1., -2.].map(|value| prefix_isolation(&r, &h, &p, &layout, &[value]));
    assert_eq!(prefix, prefix_single.concat());
    for (bytes, value) in prefix_single.iter().zip([1., -2.]) {
        assert!(bytes
            .chunks_exact(2)
            .all(|b| b == bf16::from_f32(value).to_bits().to_le_bytes()));
    }
    let legacy =
        FastH3InputPackingConfig::from_layout(layout.packed(), 128, limits_batch().input).unwrap();
    assert!(serde_json::to_value(&legacy)
        .unwrap()
        .get("batches")
        .is_none());
    drop((batch, singles, members, s1, s2, r1, r2));
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(r.runtime().snapshot().live_allocations, 0);
}

#[test]
fn service_owned_denoising_source_retains_original_package_and_exact_component_contracts() {
    let (_dir, package) = package();
    let h = host();
    let layout = super::super::attention_core::layout();
    let t = tags();
    let (config, owned, _) = OwnedFastH3DenoisingSource::prepare_batch(
        package.clone(),
        &layout,
        &schedule(),
        &[&t],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        limits_batch(),
    )
    .unwrap();
    let (_, borrowed, _) = FastH3DenoisingSource::prepare(
        &package,
        &layout,
        &schedule(),
        &t,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        limits_batch(),
    )
    .unwrap();
    let ModelWeights::Stored(schema) = FastH3DenoisingFamily::new(limits_batch())
        .unwrap()
        .weights(&config)
        .unwrap()
    else {
        panic!("Stored");
    };
    for component in &schema.components {
        let original = borrowed.component(component).unwrap();
        let actual = owned.component(component).unwrap();
        assert_eq!(actual.bytes(), original.bytes());
        assert_eq!(actual.source_files(), original.source_files());
    }
    drop(borrowed);
    let retained = Arc::downgrade(&package);
    drop(package);
    assert!(retained.upgrade().is_some());
    for component in &schema.components {
        assert!(owned
            .component(component)
            .unwrap()
            .retained_host_memory()
            .is_some());
    }
    let mut wrong = schema.components[0].clone();
    wrong.role = if wrong.role == WeightComponentRole::Metadata {
        WeightComponentRole::Values
    } else {
        WeightComponentRole::Metadata
    };
    assert!(owned.component(&wrong).is_err());
    drop(owned);
    assert!(retained.upgrade().is_none());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
}
