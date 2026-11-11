//! Complete stored CUDA block execution. The independent CPU boundary oracles
//! below do not register a quantized Reference provider or claim Torch parity.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_tensor_providers, CudaTensorOperation,
    },
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
struct VerificationFamily(PH3ImageCurveBlockFamily);
impl ModelFamilyProvider for VerificationFamily {
    type Config = PH3ImageCurveBlockConfig;
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
enum Data {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}
struct Capture(BTreeMap<String, Data>);
impl Capture {
    fn new(outputs: &[TensorExecutionOutput]) -> Self {
        Self(
            outputs
                .iter()
                .map(|output| {
                    let bytes = output.bytes().bytes();
                    let data = match output.tensor().element_type() {
                        ElementType::F32 => Data::F32(
                            bytes
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                .collect(),
                        ),
                        ElementType::Bf16 => Data::Bf16(
                            bytes
                                .chunks_exact(2)
                                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                                .collect(),
                        ),
                        _ => panic!("unexpected block output dtype"),
                    };
                    (output.value_id().as_str().to_owned(), data)
                })
                .collect(),
        )
    }
    fn bf(&self, name: &str) -> &[u16] {
        match &self.0[name] {
            Data::Bf16(values) => values,
            _ => panic!("{name} not BF16"),
        }
    }
    fn f32(&self, name: &str) -> &[f32] {
        match &self.0[name] {
            Data::F32(values) => values,
            _ => panic!("{name} not F32"),
        }
    }
}
fn decode(value: u16) -> f32 {
    bf16::from_bits(value).to_f32()
}
fn encode(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}
fn f32_weight(
    source: &ComfyCanonicalWeightSource,
    c: &PH3ImageCurveBlockConfig,
    name: &str,
) -> Vec<f32> {
    source
        .archive()
        .tensor(&weights::external(c, name))
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn gamma(
    source: &ComfyCanonicalWeightSource,
    c: &PH3ImageCurveBlockConfig,
    name: &str,
) -> Vec<u16> {
    source
        .archive()
        .tensor(&weights::external(c, name))
        .unwrap()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn equal(name: &str, actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len(), "{name} extent");
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual,
            expected,
            "{name}[{i}] actual={} expected={}",
            decode(actual),
            decode(expected)
        );
    }
}
fn ulp(value: f32) -> f64 {
    let value = bf16::from_f32(value.abs());
    let bits = value.to_bits();
    if bits == 0 {
        return 2f64.powi(-133);
    }
    let next = bf16::from_bits(bits + 1).to_f32();
    f64::from(next - value.to_f32())
}
fn normalized(input: &[u16], gamma: &[u16], head: usize, eps: f32) -> Vec<u16> {
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(head) {
        let sum: f64 = row.iter().map(|&v| f64::from(decode(v)).powi(2)).sum();
        let mean = (sum / head as f64) as f32;
        let inverse = 1.0_f32 / (mean + eps).sqrt();
        output.extend(
            row.iter()
                .zip(gamma)
                .map(|(&x, &g)| encode((decode(x) * inverse) * decode(g))),
        );
    }
    output
}
fn norm_compare(name: &str, actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len(), "{name} extent");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(decode(a).is_finite());
        assert!(
            (f64::from(decode(a)) - f64::from(decode(e))).abs() <= ulp(decode(e)),
            "{name}[{i}] exceeds one BF16 neighbor of independent F64-square oracle"
        );
    }
}
fn quant(
    source: &ComfyCanonicalWeightSource,
    c: &PH3ImageCurveBlockConfig,
    module: &str,
    input: &[u16],
    fused: bool,
) -> Vec<u16> {
    let matrix =
        ComfyQuantizedMatrix::from_archive(source.archive(), &weights::external(c, module))
            .unwrap();
    let width = matrix.columns() * (if fused { 2 } else { 1 });
    let mut cache: BTreeMap<Vec<u16>, Vec<u16>> = BTreeMap::new();
    let mut output = Vec::new();
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
fn verify(source: &ComfyCanonicalWeightSource, s: &Scene, basis: &[f32], capture: &Capture) {
    use program::*;
    let c = &s.config;
    let h = c.hidden as usize;
    let rows = c.rows().unwrap() as usize;
    let inner = c.attention_inner_dim() as usize;
    // Contractual F32 dot+bias bound, with the F64 oracle's own rounding term.
    let w = f32_weight(source, c, "adaln_proj.linear.weight");
    let bias = f32_weight(source, c, "adaln_proj.linear.bias");
    let table = capture.f32("value.p_h3_image.curve_block.adaln_f32");
    let k = c.basis as usize;
    let n = 18 * h;
    let u32 = 2f64.powi(-24);
    let u64 = 2f64.powi(-53);
    let g32 = (2 * k) as f64 * u32 / (1. - (2 * k) as f64 * u32);
    let g64 = k as f64 * u64 / (1. - k as f64 * u64);
    for row in 0..c.basis_rows as usize {
        for col in 0..n {
            let mut sum = 0.;
            let mut absolute = 0.;
            for i in 0..k {
                let product = f64::from(basis[row * k + i]) * f64::from(w[col * k + i]);
                sum += product;
                absolute += product.abs();
            }
            let target = sum + f64::from(bias[col]);
            let bound = (1. + u32) * g32 * absolute + u32 * target.abs() + g64 * absolute;
            assert!(
                (f64::from(table[row * n + col]) - target).abs() <= bound + f64::EPSILON,
                "AdaLN F32 error bound row{row} col{col}"
            );
        }
    }
    let table_bf = capture.bf("value.p_h3_image.curve_block.adaln_bf16");
    equal(
        "AdaLN cast",
        table_bf,
        &table.iter().copied().map(encode).collect::<Vec<_>>(),
    );
    equal(
        "AdaLN reshape",
        capture.bf("value.p_h3_image.curve_block.adaln_rows"),
        table_bf,
    );
    let mut parameters = Vec::new();
    for (parameter, &name) in PARAMETERS.iter().enumerate() {
        let mut selected = Vec::new();
        for request in &c.requests {
            for run in request.runs() {
                let start =
                    (run.time_index() as usize * 18 + run.modality() as usize * 6 + parameter) * h;
                for _ in 0..run.length() {
                    selected.extend_from_slice(&table_bf[start..start + h]);
                }
            }
        }
        if !compact_parameters(c) {
            equal(name, capture.bf(name), &selected);
        }
        parameters.push(selected);
    }
    norm_compare(
        "norm1",
        capture.bf(NORM1),
        &normalized(
            &s.x,
            &gamma(source, c, "norm1.weight"),
            h,
            c.norm_eps.numerator() as f32 / c.norm_eps.denominator() as f32,
        ),
    );
    let scale = |input: &[u16], parameter: &[u16]| {
        input
            .iter()
            .zip(parameter)
            .map(|(&x, &scale)| encode(decode(x) * decode(encode(1.0 + decode(scale)))))
            .collect::<Vec<_>>()
    };
    let add = |left: &[u16], right: &[u16]| {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| encode(decode(a) + decode(b)))
            .collect::<Vec<_>>()
    };
    let addcmul = |x: &[u16], g: &[u16], u: &[u16]| {
        x.iter()
            .zip(g)
            .zip(u)
            .map(|((&x, &g), &u)| {
                let product = decode(g) * decode(u);
                encode(decode(x) + product)
            })
            .collect::<Vec<_>>()
    };
    let scaled1 = scale(capture.bf(NORM1), &parameters[1]);
    if !compact_parameters(c) {
        equal("scale1", capture.bf(SCALED1), &scaled1);
    }
    equal("shift1", capture.bf(MOD1), &add(&scaled1, &parameters[0]));
    equal(
        "qkv",
        capture.bf(QKV),
        &quant(source, c, "attn.qkv_proj", capture.bf(MOD1), false),
    );
    for (part, name) in [Q, K, V].into_iter().enumerate() {
        let selected = capture
            .bf(QKV)
            .chunks_exact(3 * inner)
            .flat_map(|row| row[part * inner..(part + 1) * inner].iter().copied())
            .collect::<Vec<_>>();
        equal(name, capture.bf(name), &selected);
    }
    for (input, name, weight) in [(Q, QR, "attn.q_norm.weight"), (K, KR, "attn.k_norm.weight")] {
        let normalized = normalized(
            capture.bf(input),
            &gamma(source, c, weight),
            c.head_dim as usize,
            c.qk_eps.numerator() as f32 / c.qk_eps.denominator() as f32,
        );
        let rot = c.rotary as usize;
        let d = c.head_dim as usize;
        for index in 0..rows * inner {
            let channel = index % d;
            let row = index / inner;
            let mut expected = decode(normalized[index]);
            let bound = if channel < rot {
                let partner = if channel < rot / 2 {
                    index + rot / 2
                } else {
                    index - rot / 2
                };
                let sign = if channel < rot / 2 { -1. } else { 1. };
                let cosine = decode(s.cos[row * rot + channel]);
                let sine = decode(s.sin[row * rot + channel]);
                let left = expected * cosine;
                let right = sign * decode(normalized[partner]) * sine;
                expected = decode(encode(left + right));
                ulp(decode(normalized[index])) * f64::from(cosine.abs())
                    + ulp(decode(normalized[partner])) * f64::from(sine.abs())
                    + ulp(expected)
            } else {
                ulp(expected)
            };
            let actual = decode(capture.bf(name)[index]);
            assert!(
                actual.is_finite() && (f64::from(actual) - f64::from(expected)).abs() <= bound,
                "{name}[{index}] independent norm/rotation bound"
            );
        }
    }
    attention(
        c,
        capture.bf(QR),
        capture.bf(KR),
        capture.bf(V),
        capture.bf(ATTENTION),
    );
    equal(
        "attention output projection",
        capture.bf(ATTENTION_OUT),
        &quant(source, c, "attn.out_proj", capture.bf(ATTENTION), false),
    );
    equal(
        "residual1",
        capture.bf(RESIDUAL1),
        &addcmul(&s.x, &parameters[2], capture.bf(ATTENTION_OUT)),
    );
    norm_compare(
        "norm2",
        capture.bf(NORM2),
        &normalized(
            capture.bf(RESIDUAL1),
            &gamma(source, c, "norm2.weight"),
            h,
            c.norm_eps.numerator() as f32 / c.norm_eps.denominator() as f32,
        ),
    );
    let scaled2 = scale(capture.bf(NORM2), &parameters[4]);
    if !compact_parameters(c) {
        equal("scale2", capture.bf(SCALED2), &scaled2);
    }
    equal("shift2", capture.bf(MOD2), &add(&scaled2, &parameters[3]));
    equal(
        "fc1 gate|up",
        capture.bf(GATE_UP),
        &quant(source, c, "mlp.fc1", capture.bf(MOD2), false),
    );
    equal(
        "F32 SwiGLU fc2",
        capture.bf(MLP),
        &quant(source, c, "mlp.fc2", capture.bf(GATE_UP), true),
    );
    equal(
        "final residual",
        capture.bf(PH3ImageCurveBlockFamily::OUTPUT_ID),
        &addcmul(capture.bf(RESIDUAL1), &parameters[5], capture.bf(MLP)),
    );
    assert_ne!(capture.bf(RESIDUAL1), s.x);
    assert_ne!(
        capture.bf(PH3ImageCurveBlockFamily::OUTPUT_ID),
        capture.bf(RESIDUAL1)
    );
    assert!(parameters[2].windows(2).any(|p| p[0] != p[1]));
}
/// Independent F64 full-softmax target. The finite fixture qualification limit
/// is gamma(16*(S+D),2^-24)*max|V| plus BF16 final-rounding error. This is a
/// declared fixture limit for the native F32 profile, not a uniform vendor-
/// attention theorem, an upstream parity claim or a performance/SLO threshold.
fn attention(c: &PH3ImageCurveBlockConfig, q: &[u16], k: &[u16], v: &[u16], actual: &[u16]) {
    let inner = c.attention_inner_dim() as usize;
    let d = c.head_dim as usize;
    let scale = c.attention_scale.numerator() as f64 / c.attention_scale.denominator() as f64;
    let mut start = 0;
    for request in &c.requests {
        let length = request.rows().unwrap() as usize;
        let end = start + length;
        let factor = 16. * (length + d) as f64 * 2f64.powi(-24);
        let gamma = factor / (1. - factor);
        for row in start..end {
            for head in 0..c.heads as usize {
                let mut logits = Vec::with_capacity(length);
                for key in start..end {
                    let dot: f64 = (0..d)
                        .map(|channel| {
                            f64::from(decode(q[row * inner + head * d + channel]))
                                * f64::from(decode(k[key * inner + head * d + channel]))
                        })
                        .sum();
                    logits.push(dot * scale);
                }
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator: f64 = logits
                    .iter_mut()
                    .map(|x| {
                        *x = (*x - maximum).exp();
                        *x
                    })
                    .sum();
                let mut sums = vec![0.; d];
                let mut maxima = vec![0f64; d];
                for (offset, weight) in logits.into_iter().enumerate() {
                    for channel in 0..d {
                        let value =
                            f64::from(decode(v[(start + offset) * inner + head * d + channel]));
                        sums[channel] += weight * value;
                        maxima[channel] = maxima[channel].max(value.abs());
                    }
                }
                for channel in 0..d {
                    let target = sums[channel] / denominator;
                    let value = decode(actual[row * inner + head * d + channel]);
                    let rounding = 2f64.powi(-8) * target.abs() + 2f64.powi(-134);
                    let bound = gamma * maxima[channel] + rounding + f64::EPSILON;
                    assert!(value.is_finite()&&(f64::from(value)-target).abs()<=bound,"attention row{row} head{head} channel{channel} exceeds finite-fixture bound");
                }
            }
        }
        start = end;
    }
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
fn managed(host: &HostMemoryAccount, id: &str, shape: Vec<u64>, data: &[u16]) -> ManagedHostTensor {
    let mut a = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len() * 2)
        .unwrap();
    for (to, &value) in a.as_mut().chunks_exact_mut(2).zip(data) {
        to.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(
        ident(id),
        ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        a.freeze(),
    )
    .unwrap()
}
#[test]
#[ignore = "requires configured CUDA host with admitted complete-block fixture capacity"]
fn full_stored_curve_block_executes_all_branches_and_isolates_asynchronous_requests() {
    let mut s = scene(false);
    let files = archive(&payloads(&s.config));
    let checkpoint = PH3ImageCurveBlockCheckpoint::new(&s.config, &files.source, limits()).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, ident("device.cuda.0")).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 128,
            opaque_memory_profile: "complete-h3-block-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::CastF32ToBf16,
            CudaTensorOperation::ReshapeCopy,
            CudaTensorOperation::RowGather,
            CudaTensorOperation::BatchedRowConcat,
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::ModulationScale,
            CudaTensorOperation::RowSelectedModulationBf16,
            CudaTensorOperation::RowSelectedAddcmulBf16,
            CudaTensorOperation::PlainResidualAdd,
            CudaTensorOperation::FusedHeadRmsPartialRope,
            CudaTensorOperation::Addcmul,
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::ComfyConvRotW4A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_f32_contract().unwrap()),
        Box::new(cast_f32_to_bf16_contract().unwrap()),
        Box::new(reshape_copy_bf16_contract().unwrap()),
        Box::new(row_gather_bf16_contract().unwrap()),
        Box::new(batched_row_concat_bf16_contract().unwrap()),
        Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
        Box::new(modulation_scale_bf16_contract().unwrap()),
        Box::new(row_selected_modulation_bf16_contract().unwrap()),
        Box::new(row_selected_addcmul_bf16_contract().unwrap()),
        Box::new(plain_residual_add_bf16_contract().unwrap()),
        Box::new(fused_head_rms_partial_rope_bf16_contract().unwrap()),
        Box::new(addcmul_bf16_contract().unwrap()),
        Box::new(segmented_attention_bf16_contract().unwrap()),
        Box::new(comfy_convrot_w4a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        ident("provider.engine.cuda.h3_block.fixture"),
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
    let family = TypedFamilyRegistration::new(VerificationFamily(
        PH3ImageCurveBlockFamily::new(limits()).unwrap(),
    ))
    .prepare_with_profile(
        &serde_json::to_value(&s.config).unwrap(),
        &NumericalProfileId::new(PH3ImageCurveBlockFamily::CUDA_PROFILE_ID).unwrap(),
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
    let component = CompiledTensorComponent::compile(
        family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &s.config.compile_options().unwrap(),
    )
    .unwrap();
    let host = host();
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(checkpoint.source()),
            ComponentInitializationConfig {
                run_id: ident("run.h3_block.init"),
                request_id: ident("request.h3_block.init"),
                transaction_id: ident("transaction.h3_block.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let rows = s.config.rows().unwrap();
    let basis = checkpoint
        .managed_basis(&host, &s.times, PH3ImageCurveLerpArithmetic::SeparateF32)
        .unwrap();
    let basis_values = basis
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let output = session
        .execute(
            ident("run.h3_block.execute"),
            ident("request.h3_block.execute"),
            vec![
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::INPUT_ID,
                    vec![rows, s.config.hidden],
                    &s.x,
                ),
                basis,
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::COSINE_ID,
                    vec![rows, s.config.rotary],
                    &s.cos,
                ),
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::SINE_ID,
                    vec![rows, s.config.rotary],
                    &s.sin,
                ),
            ],
        )
        .unwrap();
    assert!(output.iter().all(|o| host.owns(o.bytes())));
    let captured = Capture::new(&output);
    verify(&files.source, &s, &basis_values, &captured);
    let first = s.config.requests[0].rows().unwrap() as usize * s.config.hidden as usize;
    s.x[first..]
        .iter_mut()
        .for_each(|value| *value = encode(decode(*value) + 0.25));
    s.times[1] = 0.75;
    let basis = checkpoint
        .managed_basis(&host, &s.times, PH3ImageCurveLerpArithmetic::SeparateF32)
        .unwrap();
    let changed = session
        .execute(
            ident("run.h3_block.changed"),
            ident("request.h3_block.changed"),
            vec![
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::INPUT_ID,
                    vec![rows, s.config.hidden],
                    &s.x,
                ),
                basis,
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::COSINE_ID,
                    vec![rows, s.config.rotary],
                    &s.cos,
                ),
                managed(
                    &host,
                    PH3ImageCurveBlockFamily::SINE_ID,
                    vec![rows, s.config.rotary],
                    &s.sin,
                ),
            ],
        )
        .unwrap();
    let changed_capture = Capture::new(&changed);
    equal(
        "request0 isolation",
        &changed_capture.bf(PH3ImageCurveBlockFamily::OUTPUT_ID)[..first],
        &captured.bf(PH3ImageCurveBlockFamily::OUTPUT_ID)[..first],
    );
    assert_ne!(
        &changed_capture.bf(PH3ImageCurveBlockFamily::OUTPUT_ID)[first..],
        &captured.bf(PH3ImageCurveBlockFamily::OUTPUT_ID)[first..]
    );
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((output, changed));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
