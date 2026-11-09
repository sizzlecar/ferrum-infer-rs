use super::*;
use ferrum_engine::image_generation::p_h3::{PH3ImageMainVisionFeature, PH3ImageMainVisionRgba};
use ferrum_models::qwen3_vl_vision::Qwen3VLVisionGrid;
use ferrum_runtime::{
    ElementType, HostMemoryClass, ProgramValueId, RequestIdentity, ResolvedTensorLayout,
    ResolvedTensorSpec,
};

const PAYLOAD: usize = 4 + 4 * 8;
struct Fixture {
    ids: [ProgramValueId; 4],
    tensor: ResolvedTensorSpec,
    request: RequestIdentity,
    grid: Qwen3VLVisionGrid,
    rgba: [u8; 4],
    features: [[u8; 8]; 4],
}
impl Fixture {
    fn new() -> Self {
        Self {
            ids: PH3ImageMainVisionPort::ALL
                .map(|port| ProgramValueId::new(port.value_id()).unwrap()),
            tensor: ResolvedTensorSpec::new(
                vec![1, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            request: RequestIdentity::new("request.capture.fixture").unwrap(),
            grid: Qwen3VLVisionGrid::new(1, 2, 2).unwrap(),
            rgba: [11, 22, 33, 44],
            features: std::array::from_fn(|index| {
                let mut bytes = [0; 8];
                bytes[..4].copy_from_slice(&(index as f32 + 0.25).to_le_bytes());
                bytes[4..].copy_from_slice(&(-0.0f32).to_le_bytes());
                bytes
            }),
        }
    }
    fn view(&self) -> PH3ImageMainVisionObservation<'_> {
        PH3ImageMainVisionObservation {
            request: &self.request,
            job: 71,
            grid: &self.grid,
            rgba: PH3ImageMainVisionRgba {
                width: 1,
                height: 1,
                source_has_alpha: true,
                host_class: HostMemoryClass::InputPreprocess,
                bytes: &self.rgba,
            },
            features: std::array::from_fn(|index| PH3ImageMainVisionFeature {
                port: PH3ImageMainVisionPort::ALL[index],
                value_id: &self.ids[index],
                tensor: &self.tensor,
                host_class: HostMemoryClass::Output,
                bytes: &self.features[index],
            }),
        }
    }
}
fn writer(limit: usize) -> (tempfile::TempDir, MainVisionCaptureWriter) {
    let parent = tempfile::tempdir().unwrap();
    let writer = MainVisionCaptureWriter::open(ImageMainVisionCaptureOptions {
        output_directory: parent.path().join("captures"),
        maximum_payload_bytes: NonZeroUsize::new(limit).unwrap(),
    })
    .unwrap();
    (parent, writer)
}
fn create_new(path: &Path) -> io::Result<File> {
    OpenOptions::new().create_new(true).write(true).open(path)
}
fn capture(writer: &MainVisionCaptureWriter, ordinal: u64) -> PathBuf {
    writer
        .options
        .output_directory
        .join(format!("capture-{ordinal:06}"))
}

#[test]
fn payloads_and_descriptors_are_exact_and_complete_manifest_is_published_last() {
    let fixture = Fixture::new();
    let (_parent, mut writer) = writer(PAYLOAD);
    let directory = capture(&writer, 0);
    writer
        .capture_with(fixture.view(), |path| {
            if path.file_name().unwrap() == "manifest.json.part" {
                assert!(!directory.join("manifest.json").exists());
                assert_eq!(fs::read(directory.join("rgba8.bin")).unwrap(), fixture.rgba);
                for (port, bytes) in PH3ImageMainVisionPort::ALL.iter().zip(&fixture.features) {
                    assert_eq!(
                        fs::read(directory.join(feature_filename(*port))).unwrap(),
                        bytes
                    );
                }
            }
            create_new(path)
        })
        .unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["complete"], true);
    assert_eq!(manifest["role"], PH3ImageMainVisionObservation::ROLE);
    assert_eq!(manifest["job"], 71);
    assert_eq!(manifest["request"], fixture.request.as_str());
    assert_eq!(
        manifest["grid"],
        json!({"temporal":1, "height":2, "width":2})
    );
    assert_eq!(manifest["payload_bytes"], PAYLOAD);
    assert_eq!(manifest["cumulative_reserved_payload_bytes"], PAYLOAD);
    assert_eq!(manifest["rgba"]["source_has_alpha"], true);
    assert_eq!(manifest["rgba"]["host_class"], "input_preprocess");
    assert_eq!(
        manifest["rgba"]["file"]["sha256"],
        format!("{:x}", Sha256::digest(fixture.rgba))
    );
    assert_eq!(manifest["features"].as_array().unwrap().len(), 4);
    for (index, port) in PH3ImageMainVisionPort::ALL.iter().enumerate() {
        let feature = &manifest["features"][index];
        assert_eq!(feature["value_id"], port.value_id());
        assert_eq!(
            feature["tensor"],
            serde_json::to_value(&fixture.tensor).unwrap()
        );
        assert_eq!(feature["file"]["bytes"], 8);
        assert_eq!(feature["host_class"], "output");
        assert_eq!(
            feature["file"]["sha256"],
            format!("{:x}", Sha256::digest(fixture.features[index]))
        );
    }
}

#[test]
fn cumulative_directory_limit_accepts_exact_first_and_second_capture_and_refuses_one_byte_over() {
    let fixture = Fixture::new();
    for (limit, admitted) in [
        (PAYLOAD - 1, 0),
        (PAYLOAD, 1),
        (2 * PAYLOAD - 1, 1),
        (2 * PAYLOAD, 2),
    ] {
        let (_parent, mut writer) = writer(limit);
        for ordinal in 0..admitted {
            writer.observe(fixture.view()).unwrap();
            assert!(capture(&writer, ordinal).join("manifest.json").is_file());
        }
        assert!(writer.observe(fixture.view()).is_err());
        assert!(
            !capture(&writer, admitted).exists(),
            "refused capture must create no directory or manifest"
        );
        assert_eq!(writer.reserved_payload_bytes, admitted as usize * PAYLOAD);
    }
}

#[test]
fn cumulative_overflow_is_refused_before_creating_a_capture() {
    let fixture = Fixture::new();
    let (_parent, mut writer) = writer(usize::MAX);
    writer.reserved_payload_bytes = usize::MAX - PAYLOAD + 1;
    assert!(writer.observe(fixture.view()).is_err());
    assert!(!capture(&writer, 0).exists());
}

#[test]
fn existing_root_and_capture_directories_are_preserved() {
    let fixture = Fixture::new();
    let (_parent, mut writer) = writer(PAYLOAD);
    let sentinel = writer.options.output_directory.join("keep");
    fs::write(&sentinel, b"original").unwrap();
    assert!(MainVisionCaptureWriter::open(writer.options.clone()).is_err());
    assert_eq!(fs::read(&sentinel).unwrap(), b"original");
    let directory = capture(&writer, 0);
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("keep"), b"child").unwrap();
    assert!(writer.observe(fixture.view()).is_err());
    assert_eq!(fs::read(directory.join("keep")).unwrap(), b"child");
    assert_eq!(writer.reserved_payload_bytes, 0);
}

#[test]
fn write_failure_keeps_partial_reservation_and_never_publishes_a_complete_manifest() {
    let fixture = Fixture::new();
    let (_parent, mut writer) = writer(PAYLOAD);
    let directory = capture(&writer, 0);
    let result = writer.capture_with(fixture.view(), |path| {
        let file = create_new(path)?;
        if path.file_name().unwrap() == "vision-deepstack-0.f32le.bin" {
            drop(file);
            File::open(path) // Actual write_all on this read-only file must fail.
        } else {
            Ok(file)
        }
    });
    assert!(result.is_err());
    assert_eq!(fs::read(directory.join("rgba8.bin")).unwrap(), fixture.rgba);
    assert!(directory.join("vision-main.f32le.bin").exists());
    assert!(!directory.join("manifest.json").exists());
    assert_eq!(writer.reserved_payload_bytes, PAYLOAD);
    assert!(writer.observe(fixture.view()).is_err());
    assert!(!capture(&writer, 1).exists());
}

#[test]
fn complete_manifest_publication_cannot_overwrite_an_existing_file() {
    let fixture = Fixture::new();
    let (_parent, mut writer) = writer(PAYLOAD);
    let directory = capture(&writer, 0);
    assert!(writer
        .capture_with(fixture.view(), |path| {
            if path.file_name().unwrap() == "manifest.json.part" {
                fs::write(directory.join("manifest.json"), b"original")?;
            }
            create_new(path)
        })
        .is_err());
    assert_eq!(
        fs::read(directory.join("manifest.json")).unwrap(),
        b"original"
    );
}
