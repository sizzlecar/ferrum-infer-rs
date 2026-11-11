//! A complete two-layer Stored CUDA execution and independent scalar oracles.
//! Small geometry qualifies execution/routing; this is not an original-width
//! checkpoint, a Comfy/Torch binary parity result or a serving SLO measurement.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_quantization::comfy_convrot::ComfyQuantizedMatrix;
use std::sync::Arc;
fn ident<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
struct VerificationFamily(PH3ImageTokenRefinerFamily);
impl ModelFamilyProvider for VerificationFamily {
    type Config = PH3ImageTokenRefinerConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(r, c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(r, c)
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.0.parse_config(r)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.0.weights(c)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(c)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self.0.semantic_program(c, p)?;
        let outputs = base
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| n.outputs.clone())
            .collect();
        ModelProgram::new(
            base.family_id().clone(),
            base.inputs().to_vec(),
            base.blocks().to_vec(),
            base.states().to_vec(),
            base.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(base.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(c)
    }
}
struct Capture(BTreeMap<String, Vec<u16>>);
impl Capture {
    fn new(outputs: &[TensorExecutionOutput]) -> Self {
        Self(
            outputs
                .iter()
                .map(|o| {
                    assert_eq!(o.tensor().element_type(), ElementType::Bf16);
                    (
                        o.value_id().as_str().to_owned(),
                        o.bytes()
                            .bytes()
                            .chunks_exact(2)
                            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                            .collect(),
                    )
                })
                .collect(),
        )
    }
    fn get(&self, name: &str) -> &[u16] {
        &self.0[name]
    }
}
fn f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}
fn bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}
fn weight(source: &ComfyCanonicalWeightSource, name: &str) -> Vec<u16> {
    source
        .archive()
        .tensor(name)
        .unwrap()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn equal(name: &str, a: &[u16], e: &[u16]) {
    assert_eq!(a.len(), e.len(), "{name} extent");
    for (index, (&a, &e)) in a.iter().zip(e).enumerate() {
        assert_eq!(a, e, "{name}[{index}]: {} vs {}", f32(a), f32(e));
    }
}
fn ulp(value: f32) -> f64 {
    let value = bf16::from_f32(value.abs());
    if value.to_bits() == 0 {
        return 2f64.powi(-133);
    }
    f64::from(bf16::from_bits(value.to_bits() + 1).to_f32() - value.to_f32())
}
fn norm(
    source: &ComfyCanonicalWeightSource,
    input: &[u16],
    gamma_name: &str,
    width: usize,
    epsilon: CanonicalRational,
    actual: &[u16],
) {
    let gamma = weight(source, gamma_name);
    assert_eq!(gamma.len(), width);
    assert_eq!(input.len(), actual.len());
    let epsilon = epsilon.numerator() as f32 / epsilon.denominator() as f32;
    for (row, (x, y)) in input
        .chunks_exact(width)
        .zip(actual.chunks_exact(width))
        .enumerate()
    {
        let sum = x.iter().map(|&v| f64::from(f32(v)).powi(2)).sum::<f64>();
        let inverse = 1.0_f32 / ((sum / width as f64) as f32 + epsilon).sqrt();
        for (column, ((&x, &g), &actual)) in x.iter().zip(&gamma).zip(y).enumerate() {
            let expected = f32(bits((f32(x) * inverse) * f32(g)));
            let actual = f32(actual);
            assert!(
                actual.is_finite()
                    && (f64::from(actual) - f64::from(expected)).abs() <= ulp(expected),
                "norm {gamma_name} row{row} col{column} independent F64 reduction bound"
            );
        }
    }
}
fn quant(
    source: &ComfyCanonicalWeightSource,
    module: &str,
    input: &[u16],
    fused: bool,
) -> Vec<u16> {
    let matrix = ComfyQuantizedMatrix::from_archive(source.archive(), module).unwrap();
    let width = matrix.columns() * if fused { 2 } else { 1 };
    let mut cache = BTreeMap::<Vec<u16>, Vec<u16>>::new();
    let mut output = Vec::new();
    assert_eq!(input.len() % width, 0);
    for row in input.chunks_exact(width) {
        let values = cache.entry(row.to_vec()).or_insert_with(|| {
            if fused {
                matrix
                    .linear_swiglu_bf16_a8_reference(row, 1)
                    .unwrap()
                    .output_bf16
            } else {
                matrix.linear_bf16_a8_reference(row, 1).unwrap().output_bf16
            }
        });
        output.extend_from_slice(values);
    }
    output
}
/// Independent full F64 softmax. For these finite small-logit operands the
/// declared acceptance is gamma(16*(S+D),2^-24)*max|V| plus final BF16 rounding.
/// This fixture bound is not a universal attention or upstream-binary theorem.
fn attention(c: &PH3ImageTokenRefinerConfig, q: &[u16], k: &[u16], v: &[u16], actual: &[u16]) {
    assert_eq!(q.len(), actual.len());
    let inner = c.inner() as usize;
    let d = c.head_dim as usize;
    let scale = c.attention_scale.numerator() as f64 / c.attention_scale.denominator() as f64;
    let mut start = 0;
    for &length in &c.requests {
        let length = length as usize;
        let end = start + length;
        let factor = 16. * (length + d) as f64 * 2f64.powi(-24);
        let gamma = factor / (1. - factor);
        for row in start..end {
            for head in 0..c.heads as usize {
                let mut weights = (start..end)
                    .map(|key| {
                        (0..d)
                            .map(|channel| {
                                f64::from(f32(q[row * inner + head * d + channel]))
                                    * f64::from(f32(k[key * inner + head * d + channel]))
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator = weights
                    .iter_mut()
                    .map(|w| {
                        *w = (*w - max).exp();
                        *w
                    })
                    .sum::<f64>();
                let mut sums = vec![0.; d];
                let mut maxima = vec![0f64; d];
                for (key, w) in weights.into_iter().enumerate() {
                    for column in 0..d {
                        let value = f64::from(f32(v[(start + key) * inner + head * d + column]));
                        sums[column] += w * value;
                        maxima[column] = maxima[column].max(value.abs());
                    }
                }
                for column in 0..d {
                    let expected = sums[column] / denominator;
                    let actual = f64::from(f32(actual[row * inner + head * d + column]));
                    let bound = gamma * maxima[column]
                        + 2f64.powi(-8) * expected.abs()
                        + 2f64.powi(-134)
                        + f64::EPSILON;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= bound,
                        "attention row{row} head{head} col{column} finite-fixture bound"
                    );
                }
            }
        }
        start = end;
    }
}
fn verify(
    source: &ComfyCanonicalWeightSource,
    c: &PH3ImageTokenRefinerConfig,
    input: &[u16],
    actual: &Capture,
) {
    let rows = c.rows().unwrap() as usize;
    let h = c.hidden as usize;
    let inner = c.inner() as usize;
    let text = c.text_dim as usize;
    let condition = actual.get(program::CONDITION);
    let w = weight(source, "condition_proj.weight");
    let bias = weight(source, "condition_proj.bias");
    let mut expected = Vec::with_capacity(rows * h);
    // Fixture operands are dyadic; at width32 every product and every partial
    // sum is exactly representable in F32. F64 provides an independent target
    // for the BF16 affine boundary without reproducing the provider's SGEMM.
    for row in 0..rows {
        for column in 0..h {
            let dot = (0..text)
                .map(|i| {
                    f64::from(f32(input[row * text + i])) * f64::from(f32(w[column * text + i]))
                })
                .sum::<f64>();
            expected.push(bits((dot + f64::from(f32(bias[column]))) as f32));
        }
    }
    equal("condition BF16 affine", condition, &expected);
    assert!(condition.iter().any(|&v| f32(v) != 0.));
    let mut previous = program::CONDITION.to_owned();
    for layer in 0..c.layers {
        let stage = |name: &str| program::stage(layer, name);
        let weight = |name: &str| format!("token_refiner.blocks.{layer}.{name}");
        norm(
            source,
            actual.get(&previous),
            &weight("norm1.weight"),
            h,
            c.norm_eps,
            actual.get(&stage("norm1")),
        );
        equal(
            "W8 qkv",
            actual.get(&stage("qkv")),
            &quant(
                source,
                &weight("attn.qkv_proj"),
                actual.get(&stage("norm1")),
                false,
            ),
        );
        for (part, name) in ["q", "k", "v"].into_iter().enumerate() {
            let expected = actual
                .get(&stage("qkv"))
                .chunks_exact(3 * inner)
                .flat_map(|row| row[part * inner..(part + 1) * inner].iter().copied())
                .collect::<Vec<_>>();
            equal(name, actual.get(&stage(name)), &expected);
        }
        for name in ["q", "k"] {
            equal(
                "head reshape",
                actual.get(&stage(&format!("{name}_heads"))),
                actual.get(&stage(name)),
            );
            norm(
                source,
                actual.get(&stage(&format!("{name}_heads"))),
                &weight(&format!("attn.{name}_norm.weight")),
                c.head_dim as usize,
                c.qk_eps,
                actual.get(&stage(&format!("{name}_norm_heads"))),
            );
            equal(
                "restored head layout",
                actual.get(&stage(&format!("{name}_norm"))),
                actual.get(&stage(&format!("{name}_norm_heads"))),
            );
        }
        attention(
            c,
            actual.get(&stage("q_norm")),
            actual.get(&stage("k_norm")),
            actual.get(&stage("v")),
            actual.get(&stage("attention")),
        );
        equal(
            "W8 attention output",
            actual.get(&stage("attention_out")),
            &quant(
                source,
                &weight("attn.out_proj"),
                actual.get(&stage("attention")),
                false,
            ),
        );
        let add = |a: &[u16], b: &[u16]| {
            a.iter()
                .zip(b)
                .map(|(&a, &b)| bits(f32(a) + f32(b)))
                .collect::<Vec<_>>()
        };
        equal(
            "attention residual",
            actual.get(&stage("residual1")),
            &add(actual.get(&previous), actual.get(&stage("attention_out"))),
        );
        norm(
            source,
            actual.get(&stage("residual1")),
            &weight("norm2.weight"),
            h,
            c.norm_eps,
            actual.get(&stage("norm2")),
        );
        equal(
            "W8 gate|up",
            actual.get(&stage("gate_up")),
            &quant(
                source,
                &weight("mlp.fc1"),
                actual.get(&stage("norm2")),
                false,
            ),
        );
        equal(
            "fused F32 SwiGLU/W8",
            actual.get(&stage("mlp")),
            &quant(
                source,
                &weight("mlp.fc2"),
                actual.get(&stage("gate_up")),
                true,
            ),
        );
        equal(
            "MLP residual",
            actual.get(&stage("residual2")),
            &add(actual.get(&stage("residual1")), actual.get(&stage("mlp"))),
        );
        assert!(actual
            .get(&stage("attention_out"))
            .iter()
            .any(|&v| f32(v) != 0.));
        assert!(actual.get(&stage("mlp")).iter().any(|&v| f32(v) != 0.));
        assert_ne!(actual.get(&previous), actual.get(&stage("residual1")));
        assert_ne!(
            actual.get(&stage("residual1")),
            actual.get(&stage("residual2"))
        );
        previous = stage("residual2");
    }
    norm(
        source,
        actual.get(&previous),
        "token_refiner.final_norm.weight",
        h,
        c.final_eps,
        actual.get(PH3ImageTokenRefinerFamily::OUTPUT_ID),
    );
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            256 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                transfer_staging_bytes: 64 << 20,
                output_bytes: 64 << 20,
                weight_backing_bytes: 64 << 20,
            },
        )
        .unwrap(),
    )
}
fn input(
    host: &HostMemoryAccount,
    c: &PH3ImageTokenRefinerConfig,
    values: &[u16],
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
        .unwrap();
    for (to, &v) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
        to.copy_from_slice(&v.to_le_bytes());
    }
    ManagedHostTensor::new(
        ident(PH3ImageTokenRefinerFamily::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![c.rows().unwrap(), c.text_dim],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
fn component(
    runtime: &CudaDeviceRuntime,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    c: &PH3ImageTokenRefinerConfig,
) -> CompiledTensorComponent<CudaDeviceRuntime> {
    let family = TypedFamilyRegistration::new(VerificationFamily(
        PH3ImageTokenRefinerFamily::new(limits()).unwrap(),
    ))
    .prepare_with_profile(
        &serde_json::to_value(c).unwrap(),
        &ident(PH3ImageTokenRefinerFamily::CUDA_PROFILE_ID),
    )
    .unwrap();
    let policy = TensorComponentPolicy::new(
        runtime.descriptor().total_memory_bytes,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        family,
        registry,
        catalog,
        &policy,
        materializers,
        &c.compile_options().unwrap(),
    )
    .unwrap()
}
fn init(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: ident(format!("run.refiner.{name}")),
        request_id: ident(format!("request.refiner.{name}")),
        transaction_id: ident(format!("transaction.refiner.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
#[test]
#[ignore = "requires configured CUDA host with admitted two-layer token refiner fixture capacity"]
fn two_stored_w8_refiner_layers_execute_all_branches_and_match_b1_when_packed_as_b2() {
    let c = PH3ImageTokenRefinerConfig::new(&transformer(), vec![512, 516]).unwrap();
    let files = archive(&payloads(&c));
    let checkpoint = PH3ImageTokenRefinerCheckpoint::new(&c, &files.source, limits()).unwrap();
    let values = (0..c.rows().unwrap() as usize)
        .flat_map(|row| {
            (0..c.text_dim as usize).map(move |column| {
                bits(
                    (((row % 7) * 13 + column * 5) % 31) as f32 / 32. - 0.46875
                        + if row >= 512 { 0.125 } else { 0. },
                )
            })
        })
        .collect::<Vec<_>>();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, ident("device.cuda.0")).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 128,
            opaque_memory_profile: "h3-token-refiner-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::AffineLinear,
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::ComfyConvRotW8A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W8A8,
            CudaTensorOperation::ReshapeCopy,
            CudaTensorOperation::RowGather,
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::PlainResidualAdd,
        ],
    )
    .unwrap();
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_bf16_contract().unwrap()),
        Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
        Box::new(comfy_convrot_w8a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap()),
        Box::new(reshape_copy_bf16_contract().unwrap()),
        Box::new(row_gather_bf16_contract().unwrap()),
        Box::new(segmented_attention_bf16_contract().unwrap()),
        Box::new(plain_residual_add_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        ident("provider.engine.cuda.refiner.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(runtime.descriptor().clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let host = host();
    let mut packed = component(runtime.as_ref(), &registry, &catalog, &materializers, &c)
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(checkpoint.source()),
            init("packed"),
        )
        .unwrap();
    let outputs = packed
        .execute(
            ident("run.refiner.packed.execute"),
            ident("request.refiner.packed.execute"),
            vec![input(&host, &c, &values)],
        )
        .unwrap();
    assert!(outputs.iter().all(|o| host.owns(o.bytes())));
    let captured = Capture::new(&outputs);
    verify(&files.source, &c, &values, &captured);
    drop(outputs);
    let first = 512 * c.hidden as usize;
    let mut isolated = c.clone();
    isolated.requests = vec![512];
    let mut single = component(
        runtime.as_ref(),
        &registry,
        &catalog,
        &materializers,
        &isolated,
    )
    .initialize(
        runtime.clone(),
        host.clone(),
        ComponentWeightSource::Stored(checkpoint.source()),
        init("single"),
    )
    .unwrap();
    let outputs = single
        .execute(
            ident("run.refiner.single.execute"),
            ident("request.refiner.single.execute"),
            vec![input(
                &host,
                &isolated,
                &values[..512 * c.text_dim as usize],
            )],
        )
        .unwrap();
    let single_capture = Capture::new(&outputs);
    equal(
        "B1 vs B2 first document",
        single_capture.get(PH3ImageTokenRefinerFamily::OUTPUT_ID),
        &captured.get(PH3ImageTokenRefinerFamily::OUTPUT_ID)[..first],
    );
    drop(outputs);
    assert!(matches!(
        single
            .close()
            .unwrap_or_else(|e| panic!("single close {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let mut changed = values.clone();
    for value in &mut changed[512 * c.text_dim as usize..] {
        *value = bits(f32(*value) + 0.25);
    }
    let outputs = packed
        .execute(
            ident("run.refiner.changed.execute"),
            ident("request.refiner.changed.execute"),
            vec![input(&host, &c, &changed)],
        )
        .unwrap();
    let changed_capture = Capture::new(&outputs);
    equal(
        "perturbed second document isolation",
        &changed_capture.get(PH3ImageTokenRefinerFamily::OUTPUT_ID)[..first],
        &captured.get(PH3ImageTokenRefinerFamily::OUTPUT_ID)[..first],
    );
    assert_ne!(
        &changed_capture.get(PH3ImageTokenRefinerFamily::OUTPUT_ID)[first..],
        &captured.get(PH3ImageTokenRefinerFamily::OUTPUT_ID)[first..]
    );
    drop(outputs);
    assert!(matches!(
        packed
            .close()
            .unwrap_or_else(|e| panic!("packed close {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
