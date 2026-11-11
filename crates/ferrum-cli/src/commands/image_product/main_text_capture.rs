//! Profiler-only streaming export of the five existing main-text readbacks.
use anyhow::{ensure, Context, Result};
use ferrum_engine::image_generation::p_h3::{
    PH3ImageMainTextObservation, PH3ImageMainTextObserver,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

/// Cumulative raw F32 bytes across the entire fresh directory. Safetensors
/// headers and manifests are metadata; this is not an RSS or Host budget.
#[derive(Debug, Clone)]
pub struct ImageMainTextCaptureOptions {
    pub output_directory: PathBuf,
    pub maximum_payload_bytes: NonZeroUsize,
}

pub(super) struct MainTextCaptureWriter {
    options: ImageMainTextCaptureOptions,
    reserved_payload_bytes: usize,
    next_capture: u64,
}
impl MainTextCaptureWriter {
    pub(super) fn open(options: ImageMainTextCaptureOptions) -> Result<Self> {
        fs::create_dir(&options.output_directory).with_context(|| {
            format!(
                "create fresh main text capture directory {}",
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
        view: PH3ImageMainTextObservation<'_>,
        mut open_file: impl FnMut(&Path) -> io::Result<File>,
    ) -> Result<()> {
        let payload_bytes = view.payload_bytes()?;
        let cumulative = self
            .reserved_payload_bytes
            .checked_add(payload_bytes)
            .context("cumulative main text capture payload overflows")?;
        ensure!(
            cumulative <= self.options.maximum_payload_bytes.get(),
            "main text capture exceeds its directory's cumulative raw payload limit"
        );
        let next = self
            .next_capture
            .checked_add(1)
            .context("capture ordinal overflows")?;
        let directory = self
            .options
            .output_directory
            .join(format!("capture-{:06}", self.next_capture));
        fs::create_dir(&directory)?;
        // Partial writes remain charged and are never retried or overwritten.
        self.reserved_payload_bytes = cumulative;
        self.next_capture = next;
        let shape = [1, 5, view.tokens, PH3ImageMainTextObservation::HIDDEN_SIZE];
        let mut header = serde_json::to_vec(&json!({
            "connector_input": {"dtype":"F32", "shape":shape, "data_offsets":[0,payload_bytes]},
        }))?;
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let prefix = u64::try_from(header.len())?.to_le_bytes();
        let name = "connector_input.safetensors";
        let mut output = open_file(&directory.join(name))?;
        let mut file_hash = Sha256::new();
        let mut raw_hash = Sha256::new();
        output.write_all(&prefix)?;
        output.write_all(&header)?;
        file_hash.update(prefix);
        file_hash.update(&header);
        // For B=1 the existing contiguous [T,H] views are exactly the layers
        // of [1,5,T,H]. No concatenated buffer or intermediate float conversion.
        for tap in &view.taps {
            output.write_all(tap.bytes)?;
            file_hash.update(tap.bytes);
            raw_hash.update(tap.bytes);
        }
        output.flush()?;
        drop(output);
        let manifest = json!({
            "complete":true, "role":PH3ImageMainTextObservation::ROLE,
            "request":view.request.as_str(), "job":view.job,
            "boundary":"raw selected encoder taps before connector standardization",
            "tensor":"connector_input", "dtype":"F32", "byte_order":"little_endian",
            "shape":shape, "layers":PH3ImageMainTextObservation::LAYERS,
            "taps":view.taps.iter().map(|tap| json!({
                "layer":tap.layer, "value_id":tap.value_id.as_str(), "tensor":tap.tensor,
                "host_class":"output", "bytes":tap.bytes.len(),
            })).collect::<Vec<_>>(),
            "payload_bytes":payload_bytes, "raw_sha256":format!("{:x}", raw_hash.finalize()),
            "cumulative_reserved_payload_bytes":cumulative,
            "maximum_cumulative_payload_bytes":self.options.maximum_payload_bytes,
            "file":{"path":name, "sha256":format!("{:x}", file_hash.finalize())},
        });
        let partial = directory.join("manifest.json.part");
        let mut file = open_file(&partial)?;
        serde_json::to_writer(&mut file, &manifest)?;
        file.write_all(b"\n")?;
        file.flush()?;
        drop(file);
        fs::hard_link(&partial, directory.join("manifest.json"))?;
        let _ = fs::remove_file(partial);
        Ok(())
    }
}
impl PH3ImageMainTextObserver for MainTextCaptureWriter {
    fn observe(&mut self, view: PH3ImageMainTextObservation<'_>) -> Result<()> {
        self.capture_with(view, |path| {
            OpenOptions::new().create_new(true).write(true).open(path)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_engine::image_generation::p_h3::PH3ImageMainTextTap;
    use ferrum_runtime::{
        ElementType, HostMemoryClass, ProgramValueId, RequestIdentity, ResolvedTensorLayout,
        ResolvedTensorSpec,
    };

    const TAP_BYTES: usize = 2 * 2560 * 4;
    const PAYLOAD: usize = 5 * TAP_BYTES;
    struct Fixture {
        ids: [ProgramValueId; 5],
        tensor: ResolvedTensorSpec,
        request: RequestIdentity,
        bytes: [Vec<u8>; 5],
    }
    impl Fixture {
        fn new() -> Self {
            Self {
                ids: PH3ImageMainTextObservation::LAYERS.map(|layer| {
                    ferrum_models::qwen3_vl_text::Qwen3VLTextComfyFamily::hidden_state_output_id(
                        layer,
                    )
                    .unwrap()
                }),
                tensor: ResolvedTensorSpec::new(
                    vec![2, 2560],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                request: RequestIdentity::new("request.text.capture.fixture").unwrap(),
                bytes: std::array::from_fn(|tap| {
                    (0..TAP_BYTES / 4)
                        .flat_map(|i| (tap as f32 + i as f32 * 0.125).to_le_bytes())
                        .collect()
                }),
            }
        }
        fn view(&self) -> PH3ImageMainTextObservation<'_> {
            PH3ImageMainTextObservation {
                request: &self.request,
                job: 71,
                tokens: 2,
                taps: std::array::from_fn(|i| PH3ImageMainTextTap {
                    layer: PH3ImageMainTextObservation::LAYERS[i],
                    value_id: &self.ids[i],
                    tensor: &self.tensor,
                    host_class: HostMemoryClass::Output,
                    bytes: &self.bytes[i],
                }),
            }
        }
    }
    fn writer(limit: usize) -> (tempfile::TempDir, MainTextCaptureWriter) {
        let root = tempfile::tempdir().unwrap();
        let writer = MainTextCaptureWriter::open(ImageMainTextCaptureOptions {
            output_directory: root.path().join("captures"),
            maximum_payload_bytes: NonZeroUsize::new(limit).unwrap(),
        })
        .unwrap();
        (root, writer)
    }

    #[test]
    fn text_capture_streams_valid_safetensors_and_reserves_cumulative_bytes() {
        let fixture = Fixture::new();
        for (limit, admitted) in [
            (PAYLOAD - 1, 0),
            (PAYLOAD, 1),
            (2 * PAYLOAD - 1, 1),
            (2 * PAYLOAD, 2),
        ] {
            let (_root, mut writer) = writer(limit);
            for ordinal in 0..admitted {
                writer.observe(fixture.view()).unwrap();
                let dir = writer
                    .options
                    .output_directory
                    .join(format!("capture-{ordinal:06}"));
                let file = fs::read(dir.join("connector_input.safetensors")).unwrap();
                let archive = safetensors::SafeTensors::deserialize(&file).unwrap();
                let tensor = archive.tensor("connector_input").unwrap();
                assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
                assert_eq!(tensor.shape(), [1, 5, 2, 2560]);
                for (chunk, expected) in tensor.data().chunks_exact(TAP_BYTES).zip(&fixture.bytes) {
                    assert_eq!(chunk, expected);
                }
                let manifest: serde_json::Value =
                    serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
                assert_eq!(manifest["complete"], true);
                assert_eq!(
                    manifest["file"]["sha256"],
                    format!("{:x}", Sha256::digest(&file))
                );
                assert_eq!(
                    manifest["raw_sha256"],
                    format!("{:x}", Sha256::digest(tensor.data()))
                );
            }
            assert!(writer.observe(fixture.view()).is_err());
            assert_eq!(writer.reserved_payload_bytes, admitted * PAYLOAD);
            assert!(!writer
                .options
                .output_directory
                .join(format!("capture-{admitted:06}"))
                .exists());
        }
    }

    #[test]
    fn text_capture_io_failure_keeps_reservation_and_cannot_publish_or_overwrite() {
        let fixture = Fixture::new();
        let (_root, mut writer) = writer(PAYLOAD);
        assert!(MainTextCaptureWriter::open(writer.options.clone()).is_err());
        assert!(writer
            .capture_with(fixture.view(), |path| {
                let file = OpenOptions::new().create_new(true).write(true).open(path)?;
                if path.file_name().unwrap() == "connector_input.safetensors" {
                    drop(file);
                    File::open(path)
                } else {
                    Ok(file)
                }
            })
            .is_err());
        assert_eq!(writer.reserved_payload_bytes, PAYLOAD);
        assert!(!writer
            .options
            .output_directory
            .join("capture-000000/manifest.json")
            .exists());
        assert!(writer.observe(fixture.view()).is_err());

        let (_root, mut writer) = self::writer(PAYLOAD);
        assert!(writer
            .capture_with(fixture.view(), |path| {
                if path.file_name().unwrap() == "manifest.json.part" {
                    fs::write(path.with_file_name("manifest.json"), b"original")?;
                }
                OpenOptions::new().create_new(true).write(true).open(path)
            })
            .is_err());
        assert_eq!(
            fs::read(
                writer
                    .options
                    .output_directory
                    .join("capture-000000/manifest.json")
            )
            .unwrap(),
            b"original"
        );
    }
}
