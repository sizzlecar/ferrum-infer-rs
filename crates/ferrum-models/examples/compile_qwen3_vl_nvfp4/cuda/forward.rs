use super::*;
use anyhow::{ensure, Result};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_models::qwen3_vl_text::{
    Qwen3VLTextEncoderFamily as Family, Qwen3VLTextNvfp4Checkpoint,
};
use half::bf16;
use std::{fs::OpenOptions, io::Write, time::Instant};
struct Diagnostic<E>(E);
impl<E: std::fmt::Debug> std::fmt::Display for Diagnostic<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

fn inputs(
    checkpoint: &Qwen3VLTextNvfp4Checkpoint,
    tokens: &[u32],
    metadata: &[u8],
    host: &HostMemoryAccount,
) -> Result<Vec<ManagedHostTensor>> {
    let raw: Value = serde_json::from_slice(metadata)?;
    for field in ["image_token_id", "video_token_id"] {
        if let Some(token) = raw.get(field).and_then(Value::as_u64) {
            ensure!(
                !tokens.iter().any(|t| u64::from(*t) == token),
                "visual placeholder requires its actual vision features"
            );
        }
    }
    let text = checkpoint.config().text();
    let a = text.architecture();
    ensure!(
        text.layout().batch() == 1 && tokens.len() as u64 == text.layout().rows(),
        "forward requires one unpadded token sequence"
    );
    let embedding = checkpoint
        .archive()
        .tensor(Family::EMBEDDING_EXTERNAL_NAME)?;
    ensure!(
        embedding.element_type() == Some(ElementType::Bf16)
            && embedding.shape() == [a.vocabulary_size(), a.hidden_size()],
        "original BF16 embedding source differs"
    );
    ensure!(
        tokens.iter().all(|t| u64::from(*t) < a.vocabulary_size()),
        "token outside original vocabulary"
    );
    let names = [Family::EMBEDDINGS_ID, Family::COSINE_ID, Family::SINE_ID];
    let declared = text.input_tensor_specs()?;
    let specs = names
        .map(|name| {
            let t = &declared[&id::<ProgramValueId>(name)];
            ResolvedTensorSpec::new(t.dimensions.clone(), t.element_type, t.layout.clone())
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let requests = specs
        .iter()
        .map(|s| {
            Ok((
                HostMemoryClass::InputPreprocess,
                usize::try_from(s.minimum_storage_bytes()?)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut reservations = host.reserve_many(&requests)?.into_iter();
    let mut embeddings = reservations.next().unwrap().allocate_zeroed()?;
    let mut cosine = reservations.next().unwrap().allocate_zeroed()?;
    let mut sine = reservations.next().unwrap().allocate_zeroed()?;
    let width = usize::try_from(a.hidden_size())?;
    let dim = usize::try_from(a.head_dim())?;
    for (row, &token) in tokens.iter().enumerate() {
        let selected =
            &embedding.bytes()[token as usize * width * 2..(token as usize + 1) * width * 2];
        ensure!(
            selected
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "selected embedding contains nonfinite values"
        );
        embeddings.as_mut()[row * width * 2..(row + 1) * width * 2].copy_from_slice(selected);
    }
    // All three text-only mRoPE axes equal the unpadded token position.
    // This explicitly retains the existing CPU F32/libm -> BF16 profile.
    for pair in 0..dim / 2 {
        let inverse = 1.0f32 / libm::powf(a.rope_theta() as f32, (2 * pair) as f32 / dim as f32);
        for row in 0..tokens.len() {
            let phase = row as f32 * inverse;
            let c = bf16::from_f32(libm::cosf(phase)).to_bits().to_le_bytes();
            let s = bf16::from_f32(libm::sinf(phase)).to_bits().to_le_bytes();
            for column in [pair, pair + dim / 2] {
                let offset = (row * dim + column) * 2;
                cosine.as_mut()[offset..offset + 2].copy_from_slice(&c);
                sine.as_mut()[offset..offset + 2].copy_from_slice(&s);
            }
        }
    }
    names
        .into_iter()
        .zip(specs)
        .zip([embeddings, cosine, sine])
        .map(|((name, spec), bytes)| Ok(ManagedHostTensor::new(id(name), spec, bytes.freeze())?))
        .collect()
}

pub(super) fn run(
    args: &crate::Config,
    image: Option<&crate::image::Prepared>,
    metadata: &[u8],
    checkpoint: Arc<Qwen3VLTextNvfp4Checkpoint>,
    compiled: Arc<CompiledTensorComponent<CudaDeviceRuntime>>,
    runtime: Arc<CudaDeviceRuntime>,
    host: HostMemoryAccount,
) -> Result<Value, Failure> {
    let (input, output_path, presentation, rotary_profile) = if let Some(request) =
        &args.image_forward
    {
        let prepared = image.ok_or_else(|| error("missing checked image presentation"))?;
        let input = prepared.inputs(&checkpoint, &host).map_err(error)?;
        (
            input,
            &request.output,
            json!({"profile":ferrum_models::fast_h3::FastH3ImagePresentation::PROFILE,"input":request,
            "token_ids":prepared.presentation.token_ids(),"text_tags":prepared.presentation.text_tags(),
            "image_rows":prepared.presentation.image_span().visual_run()}),
            ferrum_models::qwen3_vl_text::QWEN3_VL_EDIT_MROPE_PROFILE,
        )
    } else {
        let request = args
            .forward
            .as_ref()
            .ok_or_else(|| error("forward input missing"))?;
        (
            inputs(&checkpoint, &request.token_ids, metadata, &host).map_err(error)?,
            &request.output,
            Value::Null,
            "text_only.f32_libm.to_bf16",
        )
    };
    let source = checkpoint.managed_source(&host).map_err(error)?;
    let start = Instant::now();
    let owners = (runtime.clone(), compiled.clone(), checkpoint.clone());
    let mut session = compiled
        .initialize_retained(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.encoder.forward.init"),
                request_id: id("request.encoder.forward.init"),
                transaction_id: id("transaction.encoder.forward.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 4096).map_err(error)?,
                executor: TensorExecutorConfig::default(),
            },
            owners,
        )
        .map_err(|failure| Failure::retain("encoder weight initialization", failure, ()))?;
    drop(source);
    let initialized_us = start.elapsed().as_micros();
    let started = Instant::now();
    let outputs = match session.session_mut().execute(
        id("run.encoder.forward"),
        id("request.encoder.forward"),
        input,
    ) {
        Ok(output) => output,
        Err(failure) => {
            return Err(Failure::retain(
                "encoder forward",
                Diagnostic(failure),
                session,
            ))
        }
    };
    let forward_us = started.elapsed().as_micros();
    let (_, receipt) = session
        .close()
        .map_err(|failure| Failure::retain("encoder close", failure, ()))?;
    drop(receipt);
    let output = outputs
        .iter()
        .find(|o| o.value_id().as_str() == Family::OUTPUT_ID)
        .ok_or_else(|| error("encoder feature output missing"))?;
    if !output
        .bytes()
        .bytes()
        .chunks_exact(2)
        .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite())
    {
        return Err(error("encoder feature output is nonfinite"));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(error)?;
    file.write_all(output.bytes().bytes()).map_err(error)?;
    file.sync_all().map_err(error)?;
    let report = json!({"output":output_path,"shape":output.tensor().dimensions(),"dtype":"BF16 little-endian","presentation":presentation,
        "sha256":format!("{:x}",Sha256::digest(output.bytes().bytes())),"finite":true,"initialization_us":initialized_us,"forward_us":forward_us,
        "rotary_profile":rotary_profile,"timing_scope":"single managed run; not a baseline comparison"});
    drop(outputs);
    if host.usage().map_err(error)?.allocated_bytes() != 0 {
        return Err(error("encoder Host allocations did not retire"));
    }
    Ok(report)
}
