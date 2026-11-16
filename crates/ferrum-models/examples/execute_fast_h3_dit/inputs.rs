use super::*;
use std::io::Read;
pub(super) fn prepare(
    a: &Config,
    c: &FastH3DitBackboneConfig,
    p: &FastH3TransformerPackage,
    positions: &FastH3PositionPlan,
    host: &HostMemoryAccount,
) -> anyhow::Result<[ManagedHostTensor; 3]> {
    let rows = c.packed_rows();
    let width = c.hidden_size();
    let spec = ResolvedTensorSpec::new(
        vec![rows, width],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )?;
    let mut owner = host.allocate_zeroed(
        HostMemoryClass::InputPreprocess,
        usize::try_from(spec.minimum_storage_bytes()?)?,
    )?;
    let Workload::Backbone { hidden_input, .. } = &a.workload else {
        anyhow::bail!("backbone input loader requires backbone workload");
    };
    match hidden_input {
        HiddenInput::DiagnosticPattern => {
            for (i, dst) in owner.as_mut().chunks_exact_mut(2).enumerate() {
                let value = ((i * 17 % 31) as f32 - 15.) / 64.;
                dst.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        HiddenInput::Bf16File { path, sha256 } => {
            let mut file = std::fs::File::open(path)?;
            anyhow::ensure!(
                file.metadata()?.len() == spec.minimum_storage_bytes()?,
                "hidden file extent differs"
            );
            file.read_exact(owner.as_mut())?;
            anyhow::ensure!(
                format!("{:x}", Sha256::digest(owner.as_ref())) == *sha256,
                "hidden file SHA256 differs"
            );
            anyhow::ensure!(finite(owner.as_ref()), "hidden input is nonfinite");
        }
    }
    let hidden =
        ManagedHostTensor::new(id(FastH3DitBackboneFamily::HIDDEN_ID), spec, owner.freeze())?;
    let (cosine, sine) = rotary(a, c, p, positions, host)?;
    Ok([hidden, cosine, sine])
}
pub(super) fn rotary(
    a: &Config,
    c: &FastH3DitBackboneConfig,
    p: &FastH3TransformerPackage,
    positions: &FastH3PositionPlan,
    host: &HostMemoryAccount,
) -> anyhow::Result<(ManagedHostTensor, ManagedHostTensor)> {
    let rows = c.packed_rows();
    let (cosine, sine) = match a.workload.rotary() {
        Rotary::AnalyticPositions => {
            let (cos, sin) = positions.managed_bf16_frequencies(
                p.config(),
                host,
                id(FastH3DitBackboneFamily::COSINE_ID),
                id(FastH3DitBackboneFamily::SINE_ID),
            )?;
            let [cos, sin] = positions.single_request_bf16_rotary_inputs(p.config(), [cos, sin])?;
            (cos, sin)
        }
        Rotary::DiagnosticIdentity => {
            let shape = ResolvedTensorSpec::new(
                vec![1, rows, c.rotary_dim()],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            let bytes = usize::try_from(shape.minimum_storage_bytes()?)?;
            let mut cos = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            let sin = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            for dst in cos.as_mut().chunks_exact_mut(2) {
                dst.copy_from_slice(&bf16::ONE.to_bits().to_le_bytes());
            }
            (
                ManagedHostTensor::new(
                    id(FastH3DitBackboneFamily::COSINE_ID),
                    shape.clone(),
                    cos.freeze(),
                )?,
                ManagedHostTensor::new(id(FastH3DitBackboneFamily::SINE_ID), shape, sin.freeze())?,
            )
        }
    };
    Ok((cosine, sine))
}

pub(super) fn tensor_file(
    file: &TensorFile,
    spec: ResolvedTensorSpec,
    name: &str,
    host: &HostMemoryAccount,
) -> anyhow::Result<ManagedHostTensor> {
    let mut source = std::fs::File::open(&file.path)?;
    let bytes = spec.minimum_storage_bytes()?;
    anyhow::ensure!(
        source.metadata()?.len() == bytes,
        "input tensor extent differs: {}",
        file.path.display()
    );
    let mut owner = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.try_into()?)?;
    source.read_exact(owner.as_mut())?;
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(owner.as_ref())) == file.sha256,
        "input SHA256 differs: {}",
        file.path.display()
    );
    let all_finite = match spec.element_type() {
        ElementType::Bf16 => finite(owner.as_ref()),
        ElementType::F32 => owner
            .as_ref()
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        _ => false,
    };
    anyhow::ensure!(
        all_finite,
        "input tensor contains nonfinite values or unsupported dtype"
    );
    Ok(ManagedHostTensor::new(id(name), spec, owner.freeze())?)
}

pub(super) fn latent(
    input: &LatentInput,
    spec: ResolvedTensorSpec,
    name: &str,
    host: &HostMemoryAccount,
) -> anyhow::Result<ManagedHostTensor> {
    match input {
        LatentInput::File { path, sha256 } => tensor_file(
            &TensorFile {
                path: path.clone(),
                sha256: sha256.clone(),
            },
            spec,
            name,
            host,
        ),
        LatentInput::DiagnosticZero => {
            let owner = host.allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                usize::try_from(spec.minimum_storage_bytes()?)?,
            )?;
            Ok(ManagedHostTensor::new(id(name), spec, owner.freeze())?)
        }
    }
}

pub(super) fn initial_media(
    a: &Config,
    plan: &FastH3PddPlan,
    layout: &FastH3PackedLayout,
    host: &HostMemoryAccount,
) -> anyhow::Result<([ManagedHostTensor; 2], bool, &'static str)> {
    match &a.workload {
        Workload::Pdd {
            video_reference_and_noise,
            audio_reference_and_noise,
            ..
        } => {
            let requirements = FastH3SamplingHostRequirements::new(plan, layout)?;
            let video = latent(
                video_reference_and_noise,
                requirements.video_tensor().clone(),
                FastH3DenoisingFamily::VIDEO_INPUT_ID,
                host,
            )?;
            let audio = latent(
                audio_reference_and_noise,
                requirements.audio_tensor().clone(),
                FastH3DenoisingFamily::AUDIO_INPUT_ID,
                host,
            )?;
            let diagnostic = matches!(video_reference_and_noise, LatentInput::DiagnosticZero)
                || matches!(audio_reference_and_noise, LatentInput::DiagnosticZero);
            Ok(([video, audio], diagnostic, "explicit_prepared_streams"))
        }
        Workload::PddSeededKeyframe {
            normalized_keyframe,
            seed,
            ..
        } => {
            let reference = tensor_file(
                normalized_keyframe,
                ResolvedTensorSpec::new(
                    vec![
                        1,
                        24,
                        1,
                        a.reference_latent_height as u64,
                        a.reference_latent_width as u64,
                    ],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )?,
                "value.fast_h3.normalized_keyframe",
                host,
            )?;
            let target = FastH3VideoGeometry::new(a.width, a.height, a.frames)?;
            let inputs = prepare_fast_h3_keyframe_media(
                *seed,
                &target,
                layout,
                &reference,
                [
                    id(FastH3DenoisingFamily::VIDEO_INPUT_ID),
                    id(FastH3DenoisingFamily::AUDIO_INPUT_ID),
                ],
                host,
            )?;
            Ok((inputs, false, FAST_H3_INITIAL_MEDIA_PROFILE))
        }
        _ => anyhow::bail!("initial media requires a PDD workload"),
    }
}
