//! Explicit eight-evaluation diagnostic. Original F32 solver states and actual
//! per-step bases are required; a model's BF16 ingress is not a solver state.
use super::{artifacts, data, InputBoundary, OverrideGroup, Trajectory};
use anyhow::{ensure, Context, Result};
#[cfg(feature = "cuda")]
use ferrum_execution::TensorExecutionOutput;
#[cfg(feature = "cuda")]
use ferrum_models::p_h3_image::PH3ImageFinalLayerFamily;
use ferrum_quantization::{SafetensorsArchive, SafetensorsTensor};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes, ManagedHostTensor};
use half::bf16;
use serde::Deserialize;
use serde_json::json;
#[cfg(feature = "cuda")]
use std::{fs::OpenOptions, io::Write};

pub(super) fn validate_options(
    mode: Trajectory,
    boundary: InputBoundary,
    taps: bool,
) -> Result<()> {
    ensure!(
        mode != Trajectory::OriginalEuler || (boundary == InputBoundary::Prepared && !taps),
        "original-euler requires prepared inputs and no hidden taps"
    );
    Ok(())
}

// Separate f32 operations deliberately match CONST.calculate_denoised,
// k_diffusion.to_d and sample_euler. No algebraic simplification or FMA.
fn euler(x: f32, negative_returned: f32, sigma: f32, next: f32) -> Result<f32> {
    ensure!(
        x.is_finite()
            && negative_returned.is_finite()
            && sigma.is_finite()
            && next.is_finite()
            && sigma > 0.0
            && next >= 0.0
            && next < sigma,
        "invalid original Euler scalar"
    );
    let scaled = negative_returned * sigma;
    let denoised = x - scaled;
    let difference = x - denoised;
    let derivative = difference / sigma;
    let dt = next - sigma;
    let delta = derivative * dt;
    let result = x + delta;
    ensure!(result.is_finite(), "nonfinite original Euler successor");
    Ok(result)
}

#[derive(Deserialize)]
struct Contract {
    schema: String,
    solver: String,
    cfg: f64,
    steps: usize,
    original_f32_state: bool,
    routing_unchanged: bool,
}

#[derive(Clone, Copy)]
struct ImageLayout {
    channels: usize,
    height: usize,
    width: usize,
    ph: usize,
    pw: usize,
    elements: usize,
}
impl ImageLayout {
    fn new(shape: [u64; 5], patch: [u64; 3]) -> Result<Self> {
        ensure!(
            shape[0] == 1
                && shape[2] == 1
                && patch[0] == 1
                && shape[1..].iter().all(|v| *v > 0)
                && patch[1] > 0
                && patch[2] > 0
                && shape[3] % patch[1] == 0
                && shape[4] % patch[2] == 0,
            "trajectory requires exact single-frame patch layout"
        );
        let elements = shape
            .iter()
            .try_fold(1u64, |n, v| n.checked_mul(*v))
            .context("image extent overflow")?;
        Ok(Self {
            channels: shape[1].try_into()?,
            height: shape[3].try_into()?,
            width: shape[4].try_into()?,
            ph: patch[1].try_into()?,
            pw: patch[2].try_into()?,
            elements: elements.try_into()?,
        })
    }
    fn nchw(self, packed: usize) -> usize {
        let patch_elements = self.channels * self.ph * self.pw;
        let row = packed / patch_elements;
        let inner = packed % patch_elements;
        let channel = inner / (self.ph * self.pw);
        let y = (row / (self.width / self.pw)) * self.ph + inner / self.pw % self.ph;
        let x = (row % (self.width / self.pw)) * self.pw + inner % self.pw;
        (channel * self.height + y) * self.width + x
    }
}
fn f32_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
}
fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes(
        bytes[2 * i..2 * i + 2].try_into().unwrap(),
    ))
    .to_f32()
}
fn ingress(value: f32) -> Result<f32> {
    let value = bf16::from_f32(value).to_f32();
    ensure!(
        value.is_finite(),
        "trajectory state cannot enter BF16 model precision"
    );
    Ok(value)
}
fn tensor<'a>(
    archive: &'a SafetensorsArchive,
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[u64],
) -> Result<SafetensorsTensor<'a>> {
    let t = archive.tensor(name)?;
    ensure!(
        t.dtype() == dtype && t.shape() == shape,
        "trajectory {name} shape/dtype differs"
    );
    data::finite(&t)?;
    Ok(t)
}

pub(super) struct EulerRun<'a> {
    archive: &'a SafetensorsArchive,
    host: HostMemoryAccount,
    shape: [u64; 5],
    audio_shape: [u64; 4],
    layout: ImageLayout,
    basis_bytes: usize,
    #[cfg(feature = "cuda")]
    sigmas: [f32; 9],
    #[cfg(feature = "cuda")]
    final_state: Option<(ManagedHostBytes, ManagedHostBytes)>,
    steps: Vec<serde_json::Value>,
    input_override: Option<OverrideInputs>,
}
struct OverrideInputs {
    archive: SafetensorsArchive,
    group: OverrideGroup,
    reference_bytes: usize,
    receipt: serde_json::Value,
}
impl<'a> EulerRun<'a> {
    pub(super) fn prepare(
        archive: &'a SafetensorsArchive,
        capture: &data::Capture,
        patch: [u64; 3],
        host: &HostMemoryAccount,
        artifacts: &artifacts::Artifacts,
        max_output: usize,
    ) -> Result<Self> {
        let c: Contract = serde_json::from_value(
            capture
                .provenance
                .get("trajectory")
                .context("missing actual trajectory contract")?
                .clone(),
        )?;
        ensure!(
            c.schema == "ferrum.ph3.original_euler.v1"
                && c.solver == "euler"
                && c.cfg == 1.0
                && c.steps == 8
                && c.original_f32_state
                && c.routing_unchanged,
            "unsupported trajectory origin, CFG, step count or routing"
        );
        let layout = ImageLayout::new(capture.target_shape, patch)?;
        let audio_elements = usize::try_from(capture.audio_shape[1])?
            .checked_mul(4)
            .context("audio extent overflow")?;
        ensure!(
            layout
                .elements
                .checked_mul(4)
                .context("latent bytes overflow")?
                <= max_output,
            "final F32 latent exceeds output budget"
        );
        let sigma = tensor(archive, "trajectory_sigmas", safetensors::Dtype::F32, &[9])?;
        let sigmas: [f32; 9] = std::array::from_fn(|i| f32_at(sigma.bytes(), i));
        ensure!(
            sigmas[0].to_bits() == (capture.sigma as f32).to_bits()
                && sigmas[8] == 0.0
                && sigmas.windows(2).all(|v| v[0] > v[1] && v[1] >= 0.0),
            "trajectory sigma grid differs or is not decreasing to zero"
        );
        let basis0 = archive.tensor("time_basis")?;
        let basis_shape = [8, basis0.shape()[0], basis0.shape()[1]];
        let basis = tensor(
            archive,
            "trajectory_time_basis",
            safetensors::Dtype::F32,
            &basis_shape,
        )?;
        let basis_bytes = basis0.bytes().len();
        ensure!(
            basis.bytes()[..basis_bytes] == *basis0.bytes(),
            "first actual basis differs"
        );
        let mut replay = Vec::with_capacity(16);
        let mut total_mismatches = 0usize;
        for (name, shape, count) in [
            ("image", capture.target_shape.to_vec(), layout.elements),
            ("audio", capture.audio_shape.to_vec(), audio_elements),
        ] {
            let states_shape = [&[9][..], &shape].concat();
            let heads_shape = [&[8][..], &shape].concat();
            let states = tensor(
                archive,
                &format!("trajectory_states_{name}"),
                safetensors::Dtype::F32,
                &states_shape,
            )?;
            let heads = tensor(
                archive,
                &format!("trajectory_returned_{name}"),
                safetensors::Dtype::BF16,
                &heads_shape,
            )?;
            let output = tensor(
                archive,
                &format!("trajectory_output_{name}"),
                safetensors::Dtype::F32,
                &shape,
            )?;
            ensure!(output.bytes() == &states.bytes()[8*count*4..9*count*4],
                "original process_latent_out is not identity for {name}; no implicit latent scaling allowed");
            ensure!(
                &heads.bytes()[..count * 2] == archive.tensor(&format!("returned_{name}"))?.bytes(),
                "first trajectory returned {name} differs from captured first forward"
            );
            for step in 0..8 {
                let mut mismatches = 0usize;
                let mut max_abs = 0f64;
                let mut error_sq = 0f64;
                let mut reference_sq = 0f64;
                for i in 0..count {
                    let actual = euler(
                        f32_at(states.bytes(), step * count + i),
                        bf16_at(heads.bytes(), step * count + i),
                        sigmas[step],
                        sigmas[step + 1],
                    )?;
                    let expected = f32_at(states.bytes(), (step + 1) * count + i);
                    mismatches += usize::from(actual.to_bits() != expected.to_bits());
                    let error = f64::from(actual) - f64::from(expected);
                    max_abs = max_abs.max(error.abs());
                    error_sq += error * error;
                    reference_sq += f64::from(expected).powi(2);
                }
                total_mismatches += mismatches;
                replay.push(json!({"stream":name,"step":step,"elements":count,"bit_mismatches":mismatches,
                    "max_absolute_error":max_abs,"relative_l2":if reference_sq>0.0 {Some((error_sq/reference_sq).sqrt())}else{None},
                    "reference_squared_norm":reference_sq,"error_squared_norm":error_sq}));
            }
        }
        artifacts.json(
            "official-euler-cpu-replay.json",
            &json!({"independent_per_step":true,
            "formula":"D=x-returned_f32*sigma; d=(x-D)/sigma; next=x+d*(next_sigma-sigma)",
            "signed_zero":"bit-exact required","finite":true,"process_latent_out_identity":true,
            "bit_mismatches":total_mismatches,"steps":replay}),
        )?;
        ensure!(
            total_mismatches == 0,
            "official Euler CPU replay differs; GPU execution stopped; inspect replay report"
        );
        let run = Self {
            archive,
            host: host.clone(),
            shape: capture.target_shape,
            audio_shape: capture.audio_shape,
            layout,
            basis_bytes,
            #[cfg(feature = "cuda")]
            sigmas,
            #[cfg(feature = "cuda")]
            final_state: None,
            steps: Vec::with_capacity(8),
            input_override: None,
        };
        run.validate_initial_ports()?;
        Ok(run)
    }

    /// Only a successfully prepared official replay can acquire an override.
    /// Its file has already passed the combined complete-file limit. No
    /// captured production basis or target noise is substituted incidentally.
    pub(super) fn bind_override(
        &mut self,
        archive: SafetensorsArchive,
        group: OverrideGroup,
        file: serde_json::Value,
    ) -> Result<()> {
        ensure!(
            self.input_override.is_none() && self.steps.is_empty(),
            "override must be selected once before execution"
        );
        let image = self.archive.tensor("image_patches")?;
        ensure!(
            image.dtype() == safetensors::Dtype::F32 && image.shape().len() == 2,
            "original image patches must be F32 rank2"
        );
        let reference_bytes = image
            .bytes()
            .len()
            .checked_sub(
                self.layout
                    .elements
                    .checked_mul(4)
                    .context("state size overflows")?,
            )
            .context("image patch extent smaller than target")?;
        let row_bytes = usize::try_from(image.shape()[1])?
            .checked_mul(4)
            .context("patch row size overflows")?;
        ensure!(
            row_bytes > 0 && reference_bytes % row_bytes == 0,
            "reference extent ends inside a patch row"
        );
        let mut records = Vec::new();
        let record = |name: &str,
                      t: &SafetensorsTensor<'_>,
                      selected: &[u8],
                      shape: &[u64],
                      original: &[u8]| {
            json!({
            "tensor":name,"dtype":format!("{:?}",t.dtype()),"source_shape":t.shape(),"source_bytes":t.bytes().len(),
            "source_payload_sha256":artifacts::sha256(t.bytes()),"selected_shape":shape,"selected_bytes":selected.len(),
            "selected_payload_sha256":artifacts::sha256(selected),"replaced_official_payload_sha256":artifacts::sha256(original)})
        };
        match group {
            OverrideGroup::Reference => {
                ensure!(
                    reference_bytes > 0,
                    "reference override requires a real reference prefix"
                );
                let replacement = tensor(
                    &archive,
                    "image_patches",
                    safetensors::Dtype::F32,
                    image.shape(),
                )?;
                let rows = u64::try_from(reference_bytes / row_bytes)?;
                let mut r = record(
                    "image_patches",
                    &replacement,
                    &replacement.bytes()[..reference_bytes],
                    &[rows, image.shape()[1]],
                    &image.bytes()[..reference_bytes],
                );
                r["axis0_range"] = json!({"start":0,"end":rows});
                records.push(r);
            }
            OverrideGroup::Text => {
                let original = self.archive.tensor("text")?;
                ensure!(
                    original.dtype() == safetensors::Dtype::BF16,
                    "original refined text must be BF16"
                );
                let replacement =
                    tensor(&archive, "text", safetensors::Dtype::BF16, original.shape())?;
                records.push(record(
                    "text",
                    &replacement,
                    replacement.bytes(),
                    replacement.shape(),
                    original.bytes(),
                ));
            }
            OverrideGroup::InitialState => {
                for (name, shape) in [
                    ("image", self.shape.to_vec()),
                    ("audio", self.audio_shape.to_vec()),
                ] {
                    let key = format!("initial_{name}_state");
                    let replacement = tensor(&archive, &key, safetensors::Dtype::F32, &shape)?;
                    for i in 0..replacement.bytes().len() / 4 {
                        ingress(f32_at(replacement.bytes(), i))?;
                    }
                    let original = self.archive.tensor(&format!("trajectory_states_{name}"))?;
                    records.push(record(
                        &key,
                        &replacement,
                        replacement.bytes(),
                        replacement.shape(),
                        &original.bytes()[..replacement.bytes().len()],
                    ));
                }
            }
        }
        self.input_override = Some(OverrideInputs {
            archive,
            group,
            reference_bytes,
            receipt: json!({
            "group":group,"file":file,"selected_tensors":records,
            "official_replay_preserved":true,"official_first_ports_validated_before_override":true,
            "basis":"original official trajectory_time_basis unchanged; production basis ignored",
            "scope":"one declared input group replaced for native execution only; no RNG reconstruction or rescaling"}),
        });
        Ok(())
    }
    pub(super) fn override_receipt(&self) -> Option<&serde_json::Value> {
        self.input_override.as_ref().map(|o| &o.receipt)
    }

    fn validate_initial_ports(&self) -> Result<()> {
        let image = self.archive.tensor("trajectory_states_image")?;
        let patches = self.archive.tensor("image_patches")?;
        let offset = patches
            .bytes()
            .len()
            .checked_sub(self.layout.elements * 4)
            .context("image patch extent smaller than target")?
            / 4;
        for i in 0..self.layout.elements {
            ensure!(
                ingress(f32_at(image.bytes(), self.layout.nchw(i)))?.to_bits()
                    == f32_at(patches.bytes(), offset + i).to_bits(),
                "actual initial state differs from first target ingress"
            );
        }
        let audio = self.archive.tensor("trajectory_states_audio")?;
        let patches = self.archive.tensor("audio_patches")?;
        let channels = usize::try_from(self.audio_shape[1])?;
        for i in 0..channels * 4 {
            let index = (i % channels) * 4 + i / channels;
            ensure!(
                ingress(f32_at(audio.bytes(), index))?.to_bits()
                    == f32_at(patches.bytes(), i).to_bits(),
                "actual initial state differs from first audio ingress"
            );
        }
        Ok(())
    }

    fn override_templates(
        &self,
        mut templates: Vec<ManagedHostTensor>,
    ) -> Result<Vec<ManagedHostTensor>> {
        if let Some(o) = self
            .input_override
            .as_ref()
            .filter(|o| o.group == OverrideGroup::Text)
        {
            let text_id = data::input_bindings(InputBoundary::Prepared)
                .iter()
                .find(|(name, _)| *name == "text")
                .context("missing text ABI")?
                .1;
            let index = templates
                .iter()
                .position(|t| t.value_id().as_str() == text_id)
                .context("missing original text template")?;
            let replacement = o.archive.tensor("text")?;
            ensure!(
                replacement.element_type() == Some(templates[index].tensor().element_type())
                    && replacement.shape() == templates[index].tensor().dimensions(),
                "text override differs from actual template"
            );
            let mut bytes = self
                .host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, replacement.bytes().len())?;
            bytes.as_mut().copy_from_slice(replacement.bytes());
            templates[index] = ManagedHostTensor::new(
                templates[index].value_id().clone(),
                templates[index].tensor().clone(),
                bytes.freeze(),
            )?;
        }
        Ok(templates)
    }

    fn initial_state(&self) -> Result<Vec<ManagedHostBytes>> {
        let mut state = Vec::with_capacity(2);
        for (name, bytes) in [
            ("image", self.layout.elements * 4),
            ("audio", usize::try_from(self.audio_shape[1])? * 16),
        ] {
            let source = if let Some(override_inputs) = self
                .input_override
                .as_ref()
                .filter(|o| o.group == OverrideGroup::InitialState)
            {
                override_inputs
                    .archive
                    .tensor(&format!("initial_{name}_state"))?
            } else {
                self.archive.tensor(&format!("trajectory_states_{name}"))?
            };
            let mut owned = self
                .host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            owned.as_mut().copy_from_slice(&source.bytes()[..bytes]);
            state.push(owned.freeze());
        }
        Ok(state)
    }

    fn step_ports(
        &self,
        templates: &[ManagedHostTensor],
        state: &[ManagedHostBytes],
        step: usize,
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            step < 8 && state.len() == 2,
            "invalid trajectory step/state"
        );
        let basis = self.archive.tensor("trajectory_time_basis")?;
        let mut ports = Vec::with_capacity(templates.len());
        for template in templates {
            let name = data::input_bindings(InputBoundary::Prepared)
                .iter()
                .find(|(_, id)| *id == template.value_id().as_str())
                .context("unknown trajectory port")?
                .0;
            if !matches!(name, "image_patches" | "audio_patches" | "time_basis") {
                ports.push(template.clone());
                continue;
            }
            let mut bytes = self
                .host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, template.bytes().len())?;
            match name {
                "time_basis" => bytes.as_mut().copy_from_slice(
                    &basis.bytes()[step * self.basis_bytes..(step + 1) * self.basis_bytes],
                ),
                "image_patches" => {
                    bytes.as_mut().copy_from_slice(template.bytes().bytes());
                    if let Some(override_inputs) = self
                        .input_override
                        .as_ref()
                        .filter(|o| o.group == OverrideGroup::Reference)
                    {
                        let replacement = override_inputs.archive.tensor("image_patches")?;
                        bytes.as_mut()[..override_inputs.reference_bytes].copy_from_slice(
                            &replacement.bytes()[..override_inputs.reference_bytes],
                        );
                    }
                    let offset = bytes.as_ref().len() - self.layout.elements * 4;
                    for i in 0..self.layout.elements {
                        let value = ingress(f32_at(state[0].bytes(), self.layout.nchw(i)))?;
                        bytes.as_mut()[offset + 4 * i..offset + 4 * i + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
                "audio_patches" => {
                    let channels = usize::try_from(self.audio_shape[1])?;
                    for i in 0..channels * 4 {
                        let value =
                            ingress(f32_at(state[1].bytes(), (i % channels) * 4 + i / channels))?;
                        bytes.as_mut()[4 * i..4 * i + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                _ => unreachable!(),
            }
            ports.push(ManagedHostTensor::new(
                template.value_id().clone(),
                template.tensor().clone(),
                bytes.freeze(),
            )?);
        }
        Ok(ports)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn run(
        &mut self,
        templates: Vec<ManagedHostTensor>,
        mut execute: impl FnMut(usize, Vec<ManagedHostTensor>) -> Result<Vec<TensorExecutionOutput>>,
    ) -> Result<()> {
        let templates = self.override_templates(templates)?;
        let mut state = self.initial_state()?;
        for step in 0..8 {
            let ports = self.step_ports(&templates, &state, step)?;
            let outputs = execute(step, ports)?;
            ensure!(
                outputs.len() == 2,
                "trajectory must produce only image/audio velocity"
            );
            let mut successors = Vec::with_capacity(2);
            let mut receipt = Vec::with_capacity(2);
            for (stream, id) in [
                PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID,
                PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID,
            ]
            .iter()
            .enumerate()
            {
                let out = outputs
                    .iter()
                    .find(|v| v.value_id().as_str() == *id)
                    .context("missing velocity")?;
                let count = state[stream].len() / 4;
                let expected = if stream == 0 {
                    vec![
                        (self.layout.height / self.layout.ph * self.layout.width / self.layout.pw)
                            as u64,
                        (self.layout.channels * self.layout.ph * self.layout.pw) as u64,
                    ]
                } else {
                    vec![4, self.audio_shape[1]]
                };
                ensure!(
                    self.host.owns(out.bytes())
                        && out.bytes().class() == HostMemoryClass::Output
                        && out.tensor().element_type() == ferrum_runtime::ElementType::Bf16
                        && out.tensor().dimensions() == expected
                        && out.bytes().len() == count * 2,
                    "velocity owner/shape/precision differs"
                );
                let mut next = self
                    .host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, count * 4)?;
                for i in 0..count {
                    let index = if stream == 0 {
                        self.layout.nchw(i)
                    } else {
                        (i % (count / 4)) * 4 + i / (count / 4)
                    };
                    let negative = -bf16_at(out.bytes().bytes(), i);
                    let value = euler(
                        f32_at(state[stream].bytes(), index),
                        negative,
                        self.sigmas[step],
                        self.sigmas[step + 1],
                    )?;
                    next.as_mut()[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
                }
                receipt.push(json!({"stream":stream,"positive_velocity_bf16_sha256":artifacts::sha256(out.bytes().bytes()),
                    "state_f32_sha256":artifacts::sha256(next.as_ref())}));
                successors.push(next.freeze());
            }
            drop(outputs);
            state = successors;
            self.steps.push(json!({"step":step,"sigma":self.sigmas[step],"next_sigma":self.sigmas[step+1],"streams":receipt}));
        }
        let audio = state.pop().unwrap();
        let image = state.pop().unwrap();
        self.final_state = Some((image, audio));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(super) fn finish(
        mut self,
        artifacts: &artifacts::Artifacts,
        max_metadata: usize,
    ) -> Result<serde_json::Value> {
        let (image, audio) = self
            .final_state
            .take()
            .context("trajectory has no completed final state")?;
        let mut header = serde_json::to_vec(
            &json!({"latent_format_version_0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
            "latent_tensor":{"dtype":"F32","shape":self.shape,"data_offsets":[0,image.len()]}}),
        )?;
        ensure!(
            header.len() <= max_metadata,
            "latent header exceeds metadata budget"
        );
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(artifacts.directory().join("final-target.latent"))?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(image.bytes())?;
        file.sync_all()?;
        let overridden = self.input_override.as_ref().map(|o| &o.receipt);
        Ok(
            json!({"mode":"original_euler","steps":self.steps,"final_image_shape":self.shape,
            "final_image_f32_sha256":artifacts::sha256(image.bytes()),"final_audio_f32_sha256":artifacts::sha256(audio.bytes()),
            "latent_file":"final-target.latent","latent_raw_bytes":image.len(),"scale":"none; stock SaveLatent v0",
            "arithmetic":"separate F32 CONST denoised, to_d and Euler update; original capture CPU replay exact before GPU",
            "input_override":overridden,
            "scope":if overridden.is_some() {"one declared production input group substituted after official replay; all other preparation inputs and actual bases remain official; native DiT and explicit original-order Euler; no service/performance claim"} else {"same original F32 initial state, fixed reference/text/RoPE and actual per-step basis; native DiT and explicit original-order Euler; no service/performance claim"}}),
        )
    }
}

#[cfg(test)]
#[path = "trajectory_tests.rs"]
mod tests;
