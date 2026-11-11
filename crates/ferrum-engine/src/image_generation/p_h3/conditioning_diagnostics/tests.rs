use super::*;
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits, ManagedHostBytes,
    ResolvedTensorLayout,
};
use std::{
    io::Write,
    sync::{Arc, Mutex},
};

const PAYLOAD: usize = 4 + 4 * 8;
struct Fixture {
    host: HostMemoryAccount,
    image: ManagedRgba8Image,
    bytes: [ManagedHostBytes; 4],
    ids: [ProgramValueId; 4],
    tensor: ResolvedTensorSpec,
    request: RequestIdentity,
    grid: Qwen3VLVisionGrid,
}
impl Fixture {
    fn new() -> Self {
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                // Each tiny logical allocation needs a rounded 64-byte range
                // plus up to 63 leading bytes for actual pointer alignment.
                5 * 128,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 128,
                    transfer_staging_bytes: 0,
                    output_bytes: 4 * 128,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            64,
        )
        .unwrap();
        let mut rgba = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
            .unwrap();
        rgba.as_mut().copy_from_slice(&[11, 22, 33, 44]);
        let image =
            ManagedRgba8Image::from_rgba8(&host, rgba.freeze(), 1, 1, ImageInputLimits::default())
                .unwrap();
        let bytes = std::array::from_fn(|index| {
            let mut bytes = host.allocate_zeroed(HostMemoryClass::Output, 8).unwrap();
            bytes.as_mut()[..4].copy_from_slice(&(index as f32 + 0.25).to_le_bytes());
            bytes.as_mut()[4..].copy_from_slice(&(-0.0f32).to_le_bytes());
            bytes.freeze()
        });
        Self {
            host,
            image,
            bytes,
            ids: PH3ImageMainVisionPort::ALL
                .map(|port| ProgramValueId::new(port.value_id()).unwrap()),
            tensor: ResolvedTensorSpec::new(
                vec![1, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            request: RequestIdentity::new("request.observer.fixture").unwrap(),
            grid: Qwen3VLVisionGrid::new(1, 2, 2).unwrap(),
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
                host_class: self.image.bytes().class(),
                bytes: self.image.bytes().bytes(),
            },
            features: std::array::from_fn(|index| PH3ImageMainVisionFeature {
                port: PH3ImageMainVisionPort::ALL[index],
                value_id: &self.ids[index],
                tensor: &self.tensor,
                host_class: self.bytes[index].class(),
                bytes: self.bytes[index].bytes(),
            }),
        }
    }
}

struct Capture {
    calls: usize,
    payload: [u8; PAYLOAD],
    used: usize,
    addresses: [usize; 5],
}
impl Default for Capture {
    fn default() -> Self {
        Self {
            calls: 0,
            payload: [0; PAYLOAD],
            used: 0,
            addresses: [0; 5],
        }
    }
}
impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let end = self
            .used
            .checked_add(bytes.len())
            .filter(|end| *end <= self.payload.len())
            .ok_or_else(|| std::io::Error::other("bounded writer full"))?;
        self.payload[self.used..end].copy_from_slice(bytes);
        self.used = end;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
struct Writer(Arc<Mutex<Capture>>);
impl PH3ImageMainVisionObserver for Writer {
    fn observe(&mut self, view: PH3ImageMainVisionObservation<'_>) -> Result<()> {
        let mut capture = self.0.lock().unwrap();
        capture.calls += 1;
        assert_eq!(view.request.as_str(), "request.observer.fixture");
        assert_eq!(view.job, 71);
        assert_eq!(view.grid.height(), 2);
        assert_eq!((view.rgba.width, view.rgba.height), (1, 1));
        assert!(view.rgba.source_has_alpha);
        assert_eq!(view.rgba.host_class, HostMemoryClass::InputPreprocess);
        capture.addresses[0] = view.rgba.bytes.as_ptr() as usize;
        capture.write_all(view.rgba.bytes)?;
        for (index, feature) in view.features.iter().enumerate() {
            assert_eq!(feature.port, PH3ImageMainVisionPort::ALL[index]);
            assert_eq!(feature.value_id.as_str(), feature.port.value_id());
            assert_eq!(feature.tensor.dimensions(), [1, 2]);
            assert_eq!(feature.tensor.element_type(), ElementType::F32);
            assert_eq!(feature.host_class, HostMemoryClass::Output);
            capture.addresses[index + 1] = feature.bytes.as_ptr() as usize;
            capture.write_all(feature.bytes)?;
        }
        Ok(())
    }
}

#[test]
fn complete_borrowed_view_streams_only_logical_bytes_without_retaining_owners() {
    let fixture = Fixture::new();
    let host = fixture.host.clone();
    let capture = Arc::new(Mutex::new(Capture::default()));
    let mut diagnostic = MainVisionDiagnostics::new(
        NonZeroUsize::new(PAYLOAD).unwrap(),
        Box::new(Writer(Arc::clone(&capture))),
    );
    let charged = host.usage().unwrap().committed_bytes();
    assert!(
        charged > PAYLOAD,
        "the fixture exercises actual alignment padding"
    );
    diagnostic.deliver(fixture.view()).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), charged);
    {
        let capture = capture.lock().unwrap();
        assert_eq!((capture.calls, capture.used), (1, PAYLOAD));
        assert_eq!(&capture.payload[..4], fixture.image.bytes().bytes());
        assert_eq!(
            capture.addresses[0],
            fixture.image.bytes().bytes().as_ptr() as usize
        );
        for index in 0..4 {
            assert_eq!(
                &capture.payload[4 + index * 8..12 + index * 8],
                fixture.bytes[index].bytes()
            );
            assert_eq!(
                capture.addresses[index + 1],
                fixture.bytes[index].bytes().as_ptr() as usize
            );
        }
    }
    drop(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    // The observer and its metadata record deliberately remain alive here.
    assert_eq!(capture.lock().unwrap().calls, 1);
}

#[test]
fn one_byte_over_capture_limit_is_refused_before_any_callback() {
    let fixture = Fixture::new();
    let capture = Arc::new(Mutex::new(Capture::default()));
    let mut diagnostic = MainVisionDiagnostics::new(
        NonZeroUsize::new(PAYLOAD - 1).unwrap(),
        Box::new(Writer(Arc::clone(&capture))),
    );
    assert!(diagnostic.deliver(fixture.view()).is_err());
    assert_eq!(capture.lock().unwrap().calls, 0);
}

#[test]
fn disabled_observer_does_not_inspect_or_require_outputs() {
    let fixture = Fixture::new();
    observe_validated_main_vision(
        &mut None,
        &fixture.request,
        71,
        &fixture.grid,
        &fixture.image,
        &[],
    )
    .unwrap();
}

struct FailingObserver {
    panics: bool,
}
impl PH3ImageMainVisionObserver for FailingObserver {
    fn observe(&mut self, _: PH3ImageMainVisionObservation<'_>) -> Result<()> {
        if self.panics {
            panic!("observer fixture panic");
        }
        anyhow::bail!("observer fixture I/O failure")
    }
}

/// Shared with the real ActiveImageStage bridge test. The payload fixture is
/// CPU-only; that test qualifies owner recovery, not a CUDA vision execution.
pub(in crate::image_generation::p_h3) fn observer_failure(panics: bool) -> anyhow::Error {
    let fixture = Fixture::new();
    let host = fixture.host.clone();
    let mut diagnostic = MainVisionDiagnostics::new(
        NonZeroUsize::new(PAYLOAD).unwrap(),
        Box::new(FailingObserver { panics }),
    );
    let error = diagnostic.deliver(fixture.view()).unwrap_err();
    drop(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    error
}

#[test]
fn error_and_panic_become_incomplete_capture_without_retaining_payload_owners() {
    assert!(format!("{:#}", observer_failure(false)).contains("observer fixture I/O failure"));
    assert!(observer_failure(true)
        .to_string()
        .contains("observer panicked"));
}

const TEXT_TAP_BYTES: usize = 2560 * 4;
const TEXT_PAYLOAD: usize = 5 * TEXT_TAP_BYTES;
struct TextFixture {
    host: HostMemoryAccount,
    bytes: [ManagedHostBytes; 5],
    ids: [ProgramValueId; 5],
    tensor: ResolvedTensorSpec,
    request: RequestIdentity,
}
impl TextFixture {
    fn new() -> Self {
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                TEXT_PAYLOAD + 5 * 128,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    transfer_staging_bytes: 0,
                    output_bytes: TEXT_PAYLOAD + 5 * 128,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            64,
        )
        .unwrap();
        let bytes = std::array::from_fn(|tap| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::Output, TEXT_TAP_BYTES)
                .unwrap();
            for word in bytes.as_mut().chunks_exact_mut(4) {
                word.copy_from_slice(&(tap as f32 + 0.25).to_le_bytes());
            }
            bytes.freeze()
        });
        Self {
            host,
            bytes,
            ids: PH3ImageMainTextObservation::LAYERS.map(|layer| {
                ferrum_models::qwen3_vl_text::Qwen3VLTextComfyFamily::hidden_state_output_id(layer)
                    .unwrap()
            }),
            tensor: ResolvedTensorSpec::new(
                vec![1, 2560],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            request: RequestIdentity::new("request.text.observer.fixture").unwrap(),
        }
    }
    fn view(&self) -> PH3ImageMainTextObservation<'_> {
        PH3ImageMainTextObservation {
            request: &self.request,
            job: 71,
            tokens: 1,
            taps: std::array::from_fn(|i| PH3ImageMainTextTap {
                layer: PH3ImageMainTextObservation::LAYERS[i],
                value_id: &self.ids[i],
                tensor: &self.tensor,
                host_class: HostMemoryClass::Output,
                bytes: self.bytes[i].bytes(),
            }),
        }
    }
}
struct TextBorrowChecker {
    addresses: [usize; 5],
    host: HostMemoryAccount,
    charged: usize,
}
impl PH3ImageMainTextObserver for TextBorrowChecker {
    fn observe(&mut self, view: PH3ImageMainTextObservation<'_>) -> Result<()> {
        assert_eq!(view.payload_bytes()?, TEXT_PAYLOAD);
        assert_eq!(self.host.usage()?.committed_bytes(), self.charged);
        for (i, tap) in view.taps.iter().enumerate() {
            assert_eq!(tap.bytes.as_ptr() as usize, self.addresses[i]);
            assert_eq!(
                f32::from_le_bytes(tap.bytes[..4].try_into().unwrap()),
                i as f32 + 0.25
            );
        }
        Ok(())
    }
}
#[test]
fn main_text_borrowed_taps_keep_original_charge_and_validate_order_and_extent() {
    let fixture = TextFixture::new();
    let host = fixture.host.clone();
    let charged = host.usage().unwrap().committed_bytes();
    let mut diagnostic = MainTextDiagnostics::new(
        NonZeroUsize::new(TEXT_PAYLOAD).unwrap(),
        Box::new(TextBorrowChecker {
            addresses: std::array::from_fn(|i| fixture.bytes[i].bytes().as_ptr() as usize),
            host: host.clone(),
            charged,
        }),
    );
    diagnostic.deliver(fixture.view()).unwrap();
    let mut wrong_order = fixture.view();
    wrong_order.taps.swap(0, 1);
    assert!(diagnostic.deliver(wrong_order).is_err());
    let mut wrong_extent = fixture.view();
    wrong_extent.tokens = 2;
    assert!(diagnostic.deliver(wrong_extent).is_err());
    let mut short_budget = MainTextDiagnostics::new(
        NonZeroUsize::new(TEXT_PAYLOAD - 1).unwrap(),
        Box::new(TextFailingObserver { panics: true }),
    );
    assert!(short_budget
        .deliver(fixture.view())
        .unwrap_err()
        .to_string()
        .contains("limit"));
    observe_validated_main_text(&mut None, &fixture.request, 71, &[], 0, &[]).unwrap();
    drop(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
struct TextFailingObserver {
    panics: bool,
}
impl PH3ImageMainTextObserver for TextFailingObserver {
    fn observe(&mut self, _: PH3ImageMainTextObservation<'_>) -> Result<()> {
        if self.panics {
            panic!("text observer fixture panic");
        }
        anyhow::bail!("text observer fixture I/O failure")
    }
}
pub(in crate::image_generation::p_h3) fn text_observer_failure(panics: bool) -> anyhow::Error {
    let fixture = TextFixture::new();
    let host = fixture.host.clone();
    let mut diagnostic = MainTextDiagnostics::new(
        NonZeroUsize::new(TEXT_PAYLOAD).unwrap(),
        Box::new(TextFailingObserver { panics }),
    );
    let error = diagnostic.deliver(fixture.view()).unwrap_err();
    drop(fixture);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    error
}
