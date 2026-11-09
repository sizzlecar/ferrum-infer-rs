//! Bounded synchronous capture of the main-vision observer's borrowed payload.
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::p_h3::{
    PH3ImageMainVisionObservation, PH3ImageMainVisionObserver, PH3ImageMainVisionPort,
};
use ferrum_runtime::HostMemoryClass;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

/// Optional profiler-only capture. The limit covers cumulative raw RGBA and
/// four-feature payload bytes in the whole directory, excluding metadata.
/// It is not an RSS, GPU memory, or managed Host allocation limit.
#[derive(Debug, Clone)]
pub struct ImageMainVisionCaptureOptions {
    pub output_directory: PathBuf,
    pub maximum_payload_bytes: NonZeroUsize,
}

pub(super) struct MainVisionCaptureWriter {
    options: ImageMainVisionCaptureOptions,
    reserved_payload_bytes: usize,
    next_capture: u64,
}
impl MainVisionCaptureWriter {
    pub(super) fn open(options: ImageMainVisionCaptureOptions) -> Result<Self> {
        fs::create_dir(&options.output_directory).with_context(|| {
            format!(
                "create fresh main vision capture directory {}",
                options.output_directory.display()
            )
        })?;
        Ok(Self {
            options,
            reserved_payload_bytes: 0,
            next_capture: 0,
        })
    }

    fn capture_with(
        &mut self,
        view: PH3ImageMainVisionObservation<'_>,
        mut open_file: impl FnMut(&Path) -> io::Result<File>,
    ) -> Result<()> {
        let payload_bytes = view.payload_bytes()?;
        let cumulative = self
            .reserved_payload_bytes
            .checked_add(payload_bytes)
            .context("cumulative main vision capture payload overflows")?;
        ensure!(
            cumulative <= self.options.maximum_payload_bytes.get(),
            "main vision capture exceeds its directory's cumulative raw payload limit"
        );
        let next = self
            .next_capture
            .checked_add(1)
            .context("capture ordinal overflows")?;
        let directory = self
            .options
            .output_directory
            .join(format!("capture-{:06}", self.next_capture));
        // Refuse the entire next payload before creating its directory. A
        // partially written capture keeps its reservation and is never reused.
        fs::create_dir(&directory)?;
        self.reserved_payload_bytes = cumulative;
        self.next_capture = next;
        let rgba_file = write_payload(&directory, "rgba8.bin", view.rgba.bytes, &mut open_file)?;
        let features = view
            .features
            .iter()
            .map(|feature| {
                let file = write_payload(
                    &directory,
                    feature_filename(feature.port),
                    feature.bytes,
                    &mut open_file,
                )?;
                Ok(json!({
                    "value_id":feature.value_id.as_str(), "tensor":feature.tensor,
                    "host_class":host_class_name(feature.host_class), "byte_order":"little_endian", "file":file,
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = json!({
            "complete":true, "role":PH3ImageMainVisionObservation::ROLE,
            "request":view.request.as_str(), "job":view.job, "grid":view.grid,
            "payload_bytes":payload_bytes, "cumulative_reserved_payload_bytes":cumulative,
            "maximum_cumulative_payload_bytes":self.options.maximum_payload_bytes,
            "rgba":{
                "width":view.rgba.width, "height":view.rgba.height,
                "source_has_alpha":view.rgba.source_has_alpha,
                "host_class":host_class_name(view.rgba.host_class), "layout":"HWC_RGBA8", "file":rgba_file,
            },
            "features":features,
        });
        let partial = directory.join("manifest.json.part");
        let mut file = open_file(&partial)?;
        serde_json::to_writer(&mut file, &manifest)?;
        file.write_all(b"\n")?;
        file.flush()?;
        drop(file);
        // Publish a complete manifest atomically without replacing an existing
        // destination. Both links are in the same fresh local directory.
        fs::hard_link(&partial, directory.join("manifest.json"))?;
        let _ = fs::remove_file(partial);
        Ok(())
    }
}
impl PH3ImageMainVisionObserver for MainVisionCaptureWriter {
    fn observe(&mut self, view: PH3ImageMainVisionObservation<'_>) -> Result<()> {
        self.capture_with(view, |path| {
            OpenOptions::new().create_new(true).write(true).open(path)
        })
    }
}

fn write_payload(
    directory: &Path,
    name: &str,
    bytes: &[u8],
    open_file: &mut impl FnMut(&Path) -> io::Result<File>,
) -> Result<serde_json::Value> {
    open_file(&directory.join(name))?.write_all(bytes)?;
    Ok(json!({
        "path":name, "bytes":bytes.len(), "sha256":format!("{:x}", Sha256::digest(bytes)),
    }))
}
fn feature_filename(port: PH3ImageMainVisionPort) -> &'static str {
    match port {
        PH3ImageMainVisionPort::Main => "vision-main.f32le.bin",
        PH3ImageMainVisionPort::DeepStack0 => "vision-deepstack-0.f32le.bin",
        PH3ImageMainVisionPort::DeepStack1 => "vision-deepstack-1.f32le.bin",
        PH3ImageMainVisionPort::DeepStack2 => "vision-deepstack-2.f32le.bin",
    }
}
fn host_class_name(class: HostMemoryClass) -> &'static str {
    match class {
        HostMemoryClass::InputPreprocess => "input_preprocess",
        HostMemoryClass::TransferStaging => "transfer_staging",
        HostMemoryClass::Output => "output",
        HostMemoryClass::WeightBacking => "weight_backing",
    }
}

#[cfg(test)]
mod tests;
