//! Bounded streaming of actual production inputs; no tensor reconstruction.
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::p_h3::{
    PH3ImageDitInputObservation, PH3ImageDitInputObserver,
};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use ferrum_runtime::{ElementType, ManagedHostTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct ImageDitInputCaptureOptions {
    pub output_directory: PathBuf,
    /// Cumulative raw stored tensors over all jobs; excludes JSON and PNG.
    pub maximum_payload_bytes: NonZeroUsize,
}
struct Job {
    next_step: usize,
    seed: u64,
    request: String,
    fixed: Value,
}
impl Job {
    fn validate(&self, step: usize, seed: u64, request: &str, fixed: &Value) -> Result<()> {
        ensure!(
            self.next_step == step
                && self.seed == seed
                && self.request == request
                && &self.fixed == fixed,
            "DiT capture skipped/repeated a step or changed fixed inputs/routing"
        );
        Ok(())
    }
}
pub(super) struct DitInputCaptureWriter {
    options: ImageDitInputCaptureOptions,
    reserved: usize,
    jobs: BTreeMap<u64, Job>,
}
impl DitInputCaptureWriter {
    pub(super) fn open(options: ImageDitInputCaptureOptions) -> Result<Self> {
        fs::create_dir(&options.output_directory)?;
        Ok(Self {
            options,
            reserved: 0,
            jobs: BTreeMap::new(),
        })
    }
    fn reserve(&mut self, raw: usize) -> Result<usize> {
        let cumulative = self
            .reserved
            .checked_add(raw)
            .context("DiT cumulative size overflows")?;
        ensure!(
            cumulative <= self.options.maximum_payload_bytes.get(),
            "DiT capture exceeds cumulative raw budget"
        );
        self.reserved = cumulative;
        Ok(cumulative)
    }
}
fn port_name(id: &str) -> Result<&'static str> {
    Ok(match id {
        "value.p_h3_image.dit_forward.image_patches_f32" => "image_patches",
        "value.p_h3_image.dit_forward.audio_rows_f32" => "audio_patches",
        "value.p_h3_image.token_refiner.output" => "text",
        "value.p_h3_image.curve_block.basis" => "time_basis",
        "value.p_h3_image.curve_block.cosine" => "rope_cos",
        "value.p_h3_image.curve_block.sine" => "rope_sin",
        _ => anyhow::bail!("unexpected actual DiT input: {id}"),
    })
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn tensor_record(name: &str, t: &ManagedHostTensor) -> Value {
    json!({"name":name,"value_id":t.value_id(),"dtype":t.tensor().element_type(),
        "shape":t.tensor().dimensions(),"bytes":t.bytes().bytes().len(),"sha256":hash(t.bytes().bytes())})
}
fn dir(options: &ImageDitInputCaptureOptions, job: u64) -> PathBuf {
    options.output_directory.join(format!("job-{job:020}"))
}
fn publish(directory: &Path, name: &str, value: &Value) -> Result<()> {
    let partial = directory.join(format!("{name}.part"));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(&partial, directory.join(name))?;
    fs::remove_file(partial)?;
    Ok(())
}
fn read_record(path: &Path) -> Result<Value> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= 1024 * 1024,
        "capture metadata too large"
    );
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 1024 * 1024, "capture metadata grew");
    Ok(serde_json::from_slice(&bytes)?)
}
impl PH3ImageDitInputObserver for DitInputCaptureWriter {
    fn observe(&mut self, view: PH3ImageDitInputObservation<'_>) -> Result<()> {
        view.payload_bytes()?;
        let mut ports = BTreeMap::new();
        for tensor in view.inputs {
            ensure!(
                ports
                    .insert(port_name(tensor.value_id().as_str())?, tensor)
                    .is_none(),
                "duplicate DiT input port"
            );
        }
        ensure!(ports.len() == 6, "incomplete DiT inputs");
        let image = ports["image_patches"];
        ensure!(
            image.tensor().element_type() == ElementType::F32
                && image.tensor().dimensions().len() == 2
                && view.image_state.tensor().element_type() == ElementType::F32
                && view.audio_state.tensor().element_type() == ElementType::F32,
            "DiT patches and original solver states must be F32"
        );
        // For the supported B1 recipe, image rows are reference then target;
        // patchification is a bijection, so target has the original state byte extent.
        let reference_bytes = image
            .bytes()
            .bytes()
            .len()
            .checked_sub(view.image_state.bytes().bytes().len())
            .context("target exceeds actual image patches")?;
        let row_bytes = usize::try_from(image.tensor().dimensions()[1])?
            .checked_mul(4)
            .context("image row extent overflows")?;
        ensure!(
            row_bytes > 0 && reference_bytes % row_bytes == 0,
            "reference prefix ends inside a row"
        );
        let fixed = json!({"text":tensor_record("text", ports["text"]),
            "rope_cos":tensor_record("rope_cos", ports["rope_cos"]),"rope_sin":tensor_record("rope_sin", ports["rope_sin"]),
            "reference_prefix_bytes":reference_bytes,"reference_prefix_sha256":hash(&image.bytes().bytes()[..reference_bytes]),
            "canonical_config":view.family.canonical_config(),"row_runs":view.row_runs});
        let step = view.step.index();
        let directory = dir(&self.options, view.job);
        if let Some(job) = self.jobs.get(&view.job) {
            job.validate(step, view.seed, view.request.as_str(), &fixed)?;
        } else {
            ensure!(
                step == 0,
                "DiT capture must begin at the original first step"
            );
            fs::create_dir(&directory)?;
        }
        let mut saved = ports
            .iter()
            .filter(|(name, _)| {
                step == 0 || matches!(**name, "image_patches" | "audio_patches" | "time_basis")
            })
            .map(|(name, tensor)| (*name, *tensor))
            .collect::<Vec<_>>();
        if step == 0 {
            saved.push(("initial_image_state", view.image_state));
            saved.push(("initial_audio_state", view.audio_state));
        }
        let raw = saved.iter().try_fold(0usize, |n, (_, t)| {
            n.checked_add(t.bytes().bytes().len())
                .context("DiT raw size overflows")
        })?;
        let cumulative = self.reserve(raw)?;
        let filename = format!("step-{step:04}.safetensors");
        let file_record = write_tensors(&directory.join(&filename), &saved)?;
        let record = json!({"inputs_complete":true,"forward_completed":false,"request":view.request,"job":view.job,"cohort_job":view.cohort_job,
            "seed":view.seed,"width":view.width,"height":view.height,"step_index":step,
            "sigma":view.step.sigma(),"next_sigma":view.step.next_sigma(),"scheduler_timestep":view.step.timestep(),
            "actual_time_inputs":view.times,"actual_row_runs":view.row_runs,
            "family_id":view.family.family_id(),"family_fingerprint":view.family.fingerprint()?,"config_fingerprint":view.family.config_fingerprint(),
            "profile":view.family.numerical_profile(),"fixed":fixed,
            "tensors":saved.iter().map(|(n,t)| tensor_record(n,t)).collect::<Vec<_>>(),
            "file":{"path":filename,"sha256":file_record["sha256"],"bytes":file_record["bytes"]},
            "raw_bytes":raw,"cumulative_raw_bytes":cumulative,"maximum_cumulative_raw_bytes":self.options.maximum_payload_bytes,
            "scope":"original production B1 inputs before actor execution; no reconstruction or time-row deduplication"});
        publish(&directory, &format!("step-{step:04}.json"), &record)?;
        self.jobs.insert(
            view.job,
            Job {
                next_step: step + 1,
                seed: view.seed,
                request: view.request.to_string(),
                fixed,
            },
        );
        Ok(())
    }
}
fn write_tensors(path: &Path, saved: &[(&str, &ManagedHostTensor)]) -> Result<Value> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, tensor) in saved {
        let end = offset
            .checked_add(tensor.bytes().bytes().len())
            .context("DiT offset overflows")?;
        let dtype = match tensor.tensor().element_type() {
            ElementType::F32 => "F32",
            ElementType::Bf16 => "BF16",
            _ => anyhow::bail!("unsupported DiT capture dtype"),
        };
        header.insert(
            name.to_string(),
            json!({"dtype":dtype,"shape":tensor.tensor().dimensions(),"data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&header)?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut digest = Sha256::new();
    {
        let mut out = HashedWriter {
            inner: &mut file,
            digest: &mut digest,
        };
        out.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        out.write_all(&header)?;
        for (_, tensor) in saved {
            out.write_all(tensor.bytes().bytes())?;
        }
    }
    file.sync_all()?;
    Ok(json!({"sha256":format!("{:x}",digest.finalize()),"bytes":file.metadata()?.len()}))
}
struct HashedWriter<'a, W> {
    inner: W,
    digest: &'a mut Sha256,
}
impl<W: Write> Write for HashedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.digest.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
pub(super) fn complete_dit_png(
    options: &ImageDitInputCaptureOptions,
    job: u64,
    seed: u64,
    steps: usize,
    dimensions: PngDimensions,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<Value> {
    ensure!(steps > 0, "DiT capture has no completed steps");
    let directory = dir(options, job);
    let mut records = Vec::with_capacity(steps);
    for step in 0..steps {
        let record = read_record(&directory.join(format!("step-{step:04}.json")))?;
        ensure!(
            record["inputs_complete"] == true
                && record["job"] == job
                && record["seed"] == seed
                && record["step_index"] == step
                && record["width"] == dimensions.width()
                && record["height"] == dimensions.height(),
            "DiT input/PNG identity or step differs"
        );
        records.push(record);
    }
    ensure!(
        !directory.join(format!("step-{steps:04}.json")).exists(),
        "DiT capture has extra steps"
    );
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join("native.png"))?;
    let mut encoded_hash = Sha256::new();
    let mut rgba_hash = Sha256::new();
    let png = write_rgba8_png(
        &mut HashedWriter {
            inner: &mut file,
            digest: &mut encoded_hash,
        },
        dimensions,
        PngLimits::default(),
        |sink| {
            emit(&mut HashedWriter {
                inner: sink,
                digest: &mut rgba_hash,
            })
        },
    )?;
    file.sync_all()?;
    let png = json!({"path":"native.png","width":dimensions.width(),"height":dimensions.height(),
        "encoded_bytes":png.encoded_bytes(),"sha256":format!("{:x}",encoded_hash.finalize()),
        "rgba8_sha256":format!("{:x}",rgba_hash.finalize()),"rgba8_bytes":dimensions.rgba_bytes()});
    publish(
        &directory,
        "manifest.json",
        &json!({"complete":true,"job":job,"seed":seed,"completed_steps":steps,
        "steps":records,"png":png,"budget":"raw input tensors only; metadata/PNG excluded; not extra Host allocation",
        "scope":"same completed production B1 job; capture adds I/O overhead; no performance claim"}),
    )?;
    Ok(json!({"job":job,"manifest":directory.join("manifest.json"),"png":png}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{
        HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
        ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
    };

    #[test]
    fn dit_tensor_file_streams_original_f32_and_bf16_without_owner_or_bit_changes() {
        let temp = tempfile::tempdir().unwrap();
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                64,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 64,
                    output_bytes: 0,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            1,
        )
        .unwrap();
        let tensor = |id: &str, dtype, data: &[u8]| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
                .unwrap();
            bytes.as_mut().copy_from_slice(data);
            ManagedHostTensor::new(
                ProgramValueId::new(id).unwrap(),
                ResolvedTensorSpec::new(vec![1, 2], dtype, ResolvedTensorLayout::Contiguous)
                    .unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        };
        let f32_bytes = [0u8, 0, 0, 128, 0, 0, 160, 63]; // -0.0 and 1.25, exact little endian.
        let bf16_bytes = [128u8, 63, 0, 128]; // 1.0 and -0.0.
        let a = tensor("value.original.a", ElementType::F32, &f32_bytes);
        let b = tensor("value.original.b", ElementType::Bf16, &bf16_bytes);
        let charged = host.usage().unwrap().committed_bytes();
        let path = temp.path().join("actual.safetensors");
        let receipt = write_tensors(&path, &[("initial_state", &a), ("text", &b)]).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(receipt["sha256"], hash(&bytes));
        let archive = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(archive.tensor("initial_state").unwrap().data(), f32_bytes);
        assert_eq!(archive.tensor("text").unwrap().data(), bf16_bytes);
        assert_eq!(archive.tensor("text").unwrap().shape(), [1, 2]);
        assert!(write_tensors(&path, &[("text", &b)]).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), charged);
        drop(a);
        drop(b);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    #[test]
    fn dit_capture_budget_and_fixed_routing_fail_closed_across_jobs() {
        let temp = tempfile::tempdir().unwrap();
        let options = ImageDitInputCaptureOptions {
            output_directory: temp.path().join("capture"),
            maximum_payload_bytes: NonZeroUsize::new(12).unwrap(),
        };
        let mut writer = DitInputCaptureWriter::open(options.clone()).unwrap();
        assert_eq!(writer.reserve(8).unwrap(), 8);
        assert!(writer.reserve(5).is_err());
        assert_eq!(writer.reserve(4).unwrap(), 12);
        assert!(writer.reserve(1).is_err());
        assert!(DitInputCaptureWriter::open(options).is_err());
        let fixed = json!({"reference":"original","row_runs":[{"time_index":2}]});
        let job = Job {
            next_step: 1,
            seed: 42,
            request: "request.test".into(),
            fixed: fixed.clone(),
        };
        job.validate(1, 42, "request.test", &fixed).unwrap();
        assert!(job.validate(0, 42, "request.test", &fixed).is_err());
        assert!(job.validate(2, 42, "request.test", &fixed).is_err());
        assert!(job.validate(1, 43, "request.test", &fixed).is_err());
        assert!(job.validate(1, 42, "request.changed", &fixed).is_err());
        assert!(job
            .validate(1, 42, "request.test", &json!({"reference":"changed"}))
            .is_err());
    }
    #[test]
    fn dit_complete_requires_all_original_steps_and_successful_same_job_png() {
        let temp = tempfile::tempdir().unwrap();
        let options = ImageDitInputCaptureOptions {
            output_directory: temp.path().join("capture"),
            maximum_payload_bytes: NonZeroUsize::new(12).unwrap(),
        };
        fs::create_dir(&options.output_directory).unwrap();
        let d = dir(&options, 7);
        fs::create_dir(&d).unwrap();
        let dimensions = PngDimensions::new(1, 1).unwrap();
        let record = |step| json!({"inputs_complete":true,"forward_completed":false,"job":7,"seed":42,"step_index":step,"width":1,"height":1});
        publish(&d, "step-0000.json", &record(0)).unwrap();
        assert!(complete_dit_png(&options, 7, 42, 2, dimensions, |out| out
            .write_all(&[1, 2, 3, 255]))
        .is_err());
        assert!(!d.join("native.png").exists());
        publish(&d, "step-0001.json", &record(1)).unwrap();
        assert!(complete_dit_png(&options, 7, 43, 2, dimensions, |out| out
            .write_all(&[1, 2, 3, 255]))
        .is_err());
        complete_dit_png(&options, 7, 42, 2, dimensions, |out| {
            out.write_all(&[1, 2, 3, 255])
        })
        .unwrap();
        let manifest = read_record(&d.join("manifest.json")).unwrap();
        assert_eq!(manifest["complete"], true);
        assert_eq!(manifest["steps"].as_array().unwrap().len(), 2);
        assert_eq!(manifest["png"]["rgba8_sha256"], hash(&[1, 2, 3, 255]));
        let failure = dir(&options, 8);
        fs::create_dir(&failure).unwrap();
        let mut r = record(0);
        r["job"] = json!(8);
        publish(&failure, "step-0000.json", &r).unwrap();
        assert!(
            complete_dit_png(&options, 8, 42, 1, dimensions, |_| Err(io::Error::other(
                "pixel failure"
            )))
            .is_err()
        );
        assert!(!failure.join("manifest.json").exists());
    }
}
