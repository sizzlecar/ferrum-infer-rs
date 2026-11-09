//! Final normalized target plus the same job's actual PNG, published together.
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::p_h3::{
    PH3ImageFinalLatentObservation, PH3ImageFinalLatentObserver,
};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

/// The cumulative budget covers only raw latent bytes. PNG encoding retains
/// the ordinary PngLimits; no full image or latent buffer is allocated here.
#[derive(Clone, Debug)]
pub struct ImageFinalLatentCaptureOptions {
    pub output_directory: PathBuf,
    pub maximum_payload_bytes: NonZeroUsize,
}
pub(super) struct FinalLatentCaptureWriter {
    options: ImageFinalLatentCaptureOptions,
    reserved_payload_bytes: usize,
}
impl FinalLatentCaptureWriter {
    pub(super) fn open(options: ImageFinalLatentCaptureOptions) -> Result<Self> {
        fs::create_dir(&options.output_directory)?;
        Ok(Self {
            options,
            reserved_payload_bytes: 0,
        })
    }
    fn capture_with(
        &mut self,
        view: PH3ImageFinalLatentObservation<'_>,
        mut open: impl FnMut(&Path) -> io::Result<File>,
    ) -> Result<()> {
        let payload_bytes = view.payload_bytes()?;
        let cumulative = self
            .reserved_payload_bytes
            .checked_add(payload_bytes)
            .context("final latent cumulative size overflows")?;
        ensure!(
            cumulative <= self.options.maximum_payload_bytes.get(),
            "final latent capture exceeds cumulative raw byte limit"
        );
        let directory = job_directory(&self.options, view.job);
        fs::create_dir(&directory)?;
        self.reserved_payload_bytes = cumulative;
        // Stock SaveLatent v0: the marker prevents legacy 1/0.18215 scaling.
        let mut header = serde_json::to_vec(&json!({
            "latent_format_version_0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
            "latent_tensor":{"dtype":"F32","shape":view.tensor.dimensions(),"data_offsets":[0,payload_bytes]},
        }))?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let prefix = u64::try_from(header.len())?.to_le_bytes();
        let name = "final-target.latent";
        let mut file = open(&directory.join(name))?;
        let mut digest = Sha256::new();
        {
            let mut writer = HashedWriter {
                inner: &mut file,
                digest: &mut digest,
            };
            writer.write_all(&prefix)?;
            writer.write_all(&header)?;
            writer.write_all(view.bytes)?;
        }
        file.sync_all()?;
        let record = json!({
            "latent_complete":true,"role":"completed_denoised_target","request":view.request.as_str(),
            "job":view.job,"seed":view.seed,"width":view.width,"height":view.height,
            "completed_steps":view.completed_steps,"shape":view.tensor.dimensions(),"dtype":"F32",
            "tensor":"latent_tensor","version_marker":"latent_format_version_0",
            "scaling_applied":false,"payload_reordered":false,"byte_order":"little_endian",
            "payload_bytes":payload_bytes,"payload_sha256":format!("{:x}",Sha256::digest(view.bytes)),
            "cumulative_reserved_payload_bytes":cumulative,"maximum_cumulative_raw_bytes":self.options.maximum_payload_bytes,
            "budget":"raw latent bytes only; excludes metadata and PNG; PNG uses ordinary PngLimits",
            "file":{"path":name,"file_bytes":file.metadata()?.len(),"sha256":format!("{:x}",digest.finalize())},
        });
        // This is only a latent receipt. Pair completion is published by the
        // profiler after this same job's PNG has been written successfully.
        publish_json(&directory, "latent.json", &record)
    }
}
impl PH3ImageFinalLatentObserver for FinalLatentCaptureWriter {
    fn observe(&mut self, view: PH3ImageFinalLatentObservation<'_>) -> Result<()> {
        self.capture_with(view, |path| {
            OpenOptions::new().write(true).create_new(true).open(path)
        })
    }
}
fn job_directory(options: &ImageFinalLatentCaptureOptions, job: u64) -> PathBuf {
    options.output_directory.join(format!("job-{job:020}"))
}
fn publish_json(directory: &Path, name: &str, value: &Value) -> Result<()> {
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
struct HashedWriter<'a, W> {
    inner: W,
    digest: &'a mut Sha256,
}
impl<W: Write> Write for HashedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.digest.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub(super) fn complete_png(
    options: &ImageFinalLatentCaptureOptions,
    job: u64,
    seed: u64,
    completed_steps: usize,
    dimensions: PngDimensions,
    limits: PngLimits,
    emit: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<Value> {
    let directory = job_directory(options, job);
    let file = File::open(directory.join("latent.json"))?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 65536,
        "latent receipt exceeds metadata bound"
    );
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 65536,
        "latent receipt grew beyond metadata bound"
    );
    let latent: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        latent["latent_complete"] == true
            && latent["job"] == job
            && latent["seed"] == seed
            && latent["completed_steps"] == completed_steps
            && latent["width"] == dimensions.width()
            && latent["height"] == dimensions.height(),
        "completed PNG identity differs from the captured final target"
    );
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join("native.png"))?;
    let mut file_hash = Sha256::new();
    let mut rgba_hash = Sha256::new();
    let png = {
        let mut encoded = HashedWriter {
            inner: &mut file,
            digest: &mut file_hash,
        };
        write_rgba8_png(&mut encoded, dimensions, limits, |pixels| {
            emit(&mut HashedWriter {
                inner: pixels,
                digest: &mut rgba_hash,
            })
        })?
    };
    file.sync_all()?;
    let png = json!({"path":"native.png","job":job,"width":png.dimensions().width(),
        "height":png.dimensions().height(),"encoded_bytes":png.encoded_bytes(),
        "sha256":format!("{:x}",file_hash.finalize()),"rgba8_bytes":dimensions.rgba_bytes(),
        "rgba8_sha256":format!("{:x}",rgba_hash.finalize())});
    let manifest = json!({"complete":true,"job":job,"latent":latent,"png":png});
    publish_json(&directory, "manifest.json", &manifest)?;
    Ok(json!({"job":job,"manifest":directory.join("manifest.json"),"png":png}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{ElementType, RequestIdentity, ResolvedTensorLayout, ResolvedTensorSpec};
    struct Fixture {
        request: RequestIdentity,
        spec: ResolvedTensorSpec,
        bytes: Vec<u8>,
    }
    impl Fixture {
        fn new() -> Self {
            Self {
                request: RequestIdentity::new("request.final.capture").unwrap(),
                spec: ResolvedTensorSpec::new(
                    vec![1, 24, 1, 1, 1],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes: (0..24)
                    .flat_map(|i| (i as f32 - 12.0).to_le_bytes())
                    .collect(),
            }
        }
        fn view(&self, job: u64) -> PH3ImageFinalLatentObservation<'_> {
            PH3ImageFinalLatentObservation {
                request: &self.request,
                job,
                width: 16,
                height: 16,
                seed: 42,
                completed_steps: 8,
                tensor: &self.spec,
                bytes: &self.bytes,
            }
        }
    }
    fn writer(limit: usize) -> (tempfile::TempDir, FinalLatentCaptureWriter) {
        let root = tempfile::tempdir().unwrap();
        let writer = FinalLatentCaptureWriter::open(ImageFinalLatentCaptureOptions {
            output_directory: root.path().join("capture"),
            maximum_payload_bytes: NonZeroUsize::new(limit).unwrap(),
        })
        .unwrap();
        (root, writer)
    }
    #[test]
    fn final_latent_v0_and_same_job_png_publish_complete_pair_only_after_both_outputs() {
        let fixture = Fixture::new();
        let (_root, mut writer) = writer(192);
        writer.observe(fixture.view(71)).unwrap();
        let directory = job_directory(&writer.options, 71);
        assert!(!directory.join("manifest.json").exists());
        let bytes = fs::read(directory.join("final-target.latent")).unwrap();
        let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(
            safe.tensor("latent_tensor").unwrap().shape(),
            [1, 24, 1, 1, 1]
        );
        assert_eq!(
            safe.tensor("latent_tensor").unwrap().dtype(),
            safetensors::Dtype::F32
        );
        assert_eq!(safe.tensor("latent_tensor").unwrap().data(), fixture.bytes);
        assert_eq!(safe.tensor("latent_format_version_0").unwrap().shape(), [0]);
        let dimensions = PngDimensions::new(16, 16).unwrap();
        assert!(complete_png(
            &writer.options,
            71,
            43,
            8,
            dimensions,
            PngLimits::default(),
            |_| unreachable!()
        )
        .is_err());
        assert!(!directory.join("native.png").exists());
        let rgba = vec![127u8; 16 * 16 * 4];
        let receipt = complete_png(
            &writer.options,
            71,
            42,
            8,
            dimensions,
            PngLimits::default(),
            |sink| sink.write_all(&rgba),
        )
        .unwrap();
        assert_eq!(
            receipt["png"]["rgba8_sha256"],
            format!("{:x}", Sha256::digest(&rgba))
        );
        let actual = fs::read(directory.join("native.png")).unwrap();
        assert_eq!(
            receipt["png"]["sha256"],
            format!("{:x}", Sha256::digest(&actual))
        );
        let manifest: Value =
            serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["complete"], true);
        assert_eq!(manifest["latent"]["job"], manifest["png"]["job"]);
        assert!(writer.observe(fixture.view(71)).is_err());
        assert!(complete_png(
            &writer.options,
            71,
            42,
            8,
            dimensions,
            PngLimits::default(),
            |_| unreachable!()
        )
        .is_err());
    }
    #[test]
    fn final_capture_cumulative_limit_and_io_failures_never_publish_false_completion() {
        let fixture = Fixture::new();
        for (limit, accepted) in [(95, 0), (96, 1), (191, 1), (192, 2)] {
            let (_root, mut writer) = writer(limit);
            for job in 0..accepted {
                writer.observe(fixture.view(job)).unwrap();
            }
            assert!(writer.observe(fixture.view(accepted)).is_err());
            assert_eq!(writer.reserved_payload_bytes, accepted as usize * 96);
            assert!(!job_directory(&writer.options, accepted).exists());
        }
        let (_root, mut writer) = writer(192);
        assert!(FinalLatentCaptureWriter::open(writer.options.clone()).is_err());
        assert!(writer
            .capture_with(fixture.view(71), |path| {
                File::create(path)?;
                File::open(path)
            })
            .is_err());
        assert_eq!(writer.reserved_payload_bytes, 96);
        assert!(!job_directory(&writer.options, 71)
            .join("latent.json")
            .exists());
        writer.observe(fixture.view(72)).unwrap();
        assert!(complete_png(
            &writer.options,
            72,
            42,
            8,
            PngDimensions::new(16, 16).unwrap(),
            PngLimits::default(),
            |_| Err(io::Error::other("pixel sink failure"))
        )
        .is_err());
        assert!(!job_directory(&writer.options, 72)
            .join("manifest.json")
            .exists());
    }
}
