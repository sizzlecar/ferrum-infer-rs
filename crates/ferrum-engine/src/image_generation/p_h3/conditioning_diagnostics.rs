//! Optional synchronous views of the already validated main-vision owners.
//! No tensor owner, GPU readback, payload allocation or output sink lives here.
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::{
    qwen3_vl_vision::{comfy::Qwen3VLVisionComfyFamily, Qwen3VLVisionGrid},
    qwen_image21::image_input::ManagedRgba8Image,
};
use ferrum_runtime::{HostMemoryClass, ProgramValueId, RequestIdentity, ResolvedTensorSpec};
use std::{num::NonZeroUsize, panic::AssertUnwindSafe};

/// The only four admitted feature ports. Their order is independent of the
/// runtime's returned output order, and is the main port then DeepStack 0..2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageMainVisionPort {
    Main,
    DeepStack0,
    DeepStack1,
    DeepStack2,
}
impl PH3ImageMainVisionPort {
    pub const ALL: [Self; 4] = [
        Self::Main,
        Self::DeepStack0,
        Self::DeepStack1,
        Self::DeepStack2,
    ];

    pub fn value_id(self) -> &'static str {
        match self {
            Self::Main => Qwen3VLVisionComfyFamily::OUTPUT_ID,
            Self::DeepStack0 => Qwen3VLVisionComfyFamily::DEEPSTACK_IDS[0],
            Self::DeepStack1 => Qwen3VLVisionComfyFamily::DEEPSTACK_IDS[1],
            Self::DeepStack2 => Qwen3VLVisionComfyFamily::DEEPSTACK_IDS[2],
        }
    }
}

/// Actual HWC RGBA8 bytes supplied to main vision, including original alpha.
/// Main vision consumes RGB and ignores alpha; this is not an RGB repack.
pub struct PH3ImageMainVisionRgba<'a> {
    pub width: u32,
    pub height: u32,
    pub source_has_alpha: bool,
    pub host_class: HostMemoryClass,
    pub bytes: &'a [u8],
}

/// Exact logical F32 little-endian bytes, excluding Host alignment padding.
/// These fields cannot retain or clone a ManagedHostBytes/ExecutionOutput owner.
pub struct PH3ImageMainVisionFeature<'a> {
    pub port: PH3ImageMainVisionPort,
    pub value_id: &'a ProgramValueId,
    pub tensor: &'a ResolvedTensorSpec,
    pub host_class: HostMemoryClass,
    pub bytes: &'a [u8],
}

/// One complete main-conditioning-vision boundary. This deliberately does not
/// implement Serialize: a caller's bounded writer streams the borrowed bytes.
pub struct PH3ImageMainVisionObservation<'a> {
    pub request: &'a RequestIdentity,
    pub job: u64,
    pub grid: &'a Qwen3VLVisionGrid,
    pub rgba: PH3ImageMainVisionRgba<'a>,
    pub features: [PH3ImageMainVisionFeature<'a>; 4],
}
impl PH3ImageMainVisionObservation<'_> {
    pub const ROLE: &'static str = "main_conditioning_vision";

    /// Single-callback logical payload extent, not total RSS or retained Host
    /// capacity. The existing owner's backing remains charged independently.
    pub fn payload_bytes(&self) -> Result<usize> {
        self.features
            .iter()
            .try_fold(self.rgba.bytes.len(), |sum, feature| {
                sum.checked_add(feature.bytes.len())
                    .context("main vision diagnostic payload overflows")
            })
    }
}

/// Opt-in composition hook. Consume borrowed views synchronously; buffering,
/// file limits, descriptors and incomplete-artifact cleanup belong to the sink.
/// Capture time is diagnostic time and must not be presented as inference-only.
pub trait PH3ImageMainVisionObserver: Send + Sync {
    fn observe(&mut self, observation: PH3ImageMainVisionObservation<'_>) -> Result<()>;
}

pub(super) struct MainVisionDiagnostics {
    maximum_payload_bytes: NonZeroUsize,
    observer: Box<dyn PH3ImageMainVisionObserver>,
}
impl MainVisionDiagnostics {
    pub(super) fn new(
        maximum_payload_bytes: NonZeroUsize,
        observer: Box<dyn PH3ImageMainVisionObserver>,
    ) -> Self {
        Self {
            maximum_payload_bytes,
            observer,
        }
    }

    fn deliver(&mut self, observation: PH3ImageMainVisionObservation<'_>) -> Result<()> {
        ensure!(
            observation.payload_bytes()? <= self.maximum_payload_bytes.get(),
            "main vision diagnostic exceeds its single-capture logical payload limit"
        );
        // An observer is arbitrary user code. Turn its panic into the same
        // processing error as an I/O failure while the caller still owns active.
        std::panic::catch_unwind(AssertUnwindSafe(|| self.observer.observe(observation)))
            .map_err(|_| anyhow::anyhow!("main vision diagnostic observer panicked"))?
            .context("main vision diagnostic capture is incomplete")
    }
}

/// Called only after actual program/ports, finite F32 values and Host provenance
/// passed the original managed_inputs adapter. Disabled performs no inspection.
pub(super) fn observe_validated_main_vision(
    diagnostics: &mut Option<MainVisionDiagnostics>,
    request: &RequestIdentity,
    job: u64,
    grid: &Qwen3VLVisionGrid,
    image: &ManagedRgba8Image,
    outputs: &[TensorExecutionOutput],
) -> Result<()> {
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };
    ensure!(
        outputs.len() == 4,
        "main vision diagnostic requires four validated outputs"
    );
    let select = |port: PH3ImageMainVisionPort| -> Result<_> {
        let output = outputs
            .iter()
            .find(|output| output.value_id().as_str() == port.value_id())
            .context("main vision diagnostic lacks a validated canonical port")?;
        Ok(PH3ImageMainVisionFeature {
            port,
            value_id: output.value_id(),
            tensor: output.tensor(),
            host_class: output.bytes().class(),
            bytes: output.bytes().bytes(),
        })
    };
    let [main, deep0, deep1, deep2] = PH3ImageMainVisionPort::ALL.map(select);
    diagnostics.deliver(PH3ImageMainVisionObservation {
        request,
        job,
        grid,
        rgba: PH3ImageMainVisionRgba {
            width: image.width(),
            height: image.height(),
            source_has_alpha: image.source_has_alpha(),
            host_class: image.bytes().class(),
            bytes: image.bytes().bytes(),
        },
        features: [main?, deep0?, deep1?, deep2?],
    })
}

/// One raw selected text-encoder readback, before connector standardization.
/// All bytes remain borrowed from the existing charged Host Output owner.
pub struct PH3ImageMainTextTap<'a> {
    pub layer: u64,
    pub value_id: &'a ProgramValueId,
    pub tensor: &'a ResolvedTensorSpec,
    pub host_class: HostMemoryClass,
    pub bytes: &'a [u8],
}

/// The B=1 connector input in layer-major order. This is the original five
/// F32 taps, not the standardized projection input or its zeroed sink row.
pub struct PH3ImageMainTextObservation<'a> {
    pub request: &'a RequestIdentity,
    pub job: u64,
    pub tokens: u64,
    pub taps: [PH3ImageMainTextTap<'a>; 5],
}
impl PH3ImageMainTextObservation<'_> {
    pub const ROLE: &'static str = "main_conditioning_text";
    pub const LAYERS: [u64; 5] = [13, 19, 25, 31, 36];
    pub const HIDDEN_SIZE: u64 = 2560;

    pub fn payload_bytes(&self) -> Result<usize> {
        let tap_bytes = self
            .tokens
            .checked_mul(Self::HIDDEN_SIZE)
            .and_then(|elements| elements.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .context("main text capture size overflows")?;
        ensure!(
            self.tokens > 0,
            "main text capture requires nonempty B=1 taps"
        );
        for (tap, layer) in self.taps.iter().zip(Self::LAYERS) {
            ensure!(
                tap.layer == layer
                    && tap.tensor.element_type() == ferrum_runtime::ElementType::F32
                    && tap.tensor.layout() == &ferrum_runtime::ResolvedTensorLayout::Contiguous
                    && (tap.tensor.dimensions() == [self.tokens, Self::HIDDEN_SIZE]
                        || tap.tensor.dimensions() == [1, self.tokens, Self::HIDDEN_SIZE])
                    && tap.host_class == HostMemoryClass::Output
                    && tap.bytes.len() == tap_bytes,
                "main text capture requires the five ordered contiguous B=1 F32 taps"
            );
        }
        tap_bytes
            .checked_mul(5)
            .context("main text capture payload overflows")
    }
}

pub trait PH3ImageMainTextObserver: Send + Sync {
    fn observe(&mut self, observation: PH3ImageMainTextObservation<'_>) -> Result<()>;
}

pub(super) struct MainTextDiagnostics {
    maximum_payload_bytes: NonZeroUsize,
    observer: Box<dyn PH3ImageMainTextObserver>,
}
impl MainTextDiagnostics {
    pub(super) fn new(
        maximum_payload_bytes: NonZeroUsize,
        observer: Box<dyn PH3ImageMainTextObserver>,
    ) -> Self {
        Self {
            maximum_payload_bytes,
            observer,
        }
    }
    fn deliver(&mut self, observation: PH3ImageMainTextObservation<'_>) -> Result<()> {
        ensure!(
            observation.payload_bytes()? <= self.maximum_payload_bytes.get(),
            "main text diagnostic exceeds its single-capture logical payload limit"
        );
        std::panic::catch_unwind(AssertUnwindSafe(|| self.observer.observe(observation)))
            .map_err(|_| anyhow::anyhow!("main text diagnostic observer panicked"))?
            .context("main text diagnostic capture is incomplete")
    }
}

/// Called after the existing adapter has validated all ports, Host provenance,
/// finite values and shapes and constructed its ordinary standardized input.
/// None performs no inspection and never retains a payload owner.
pub(super) fn observe_validated_main_text(
    diagnostics: &mut Option<MainTextDiagnostics>,
    request: &RequestIdentity,
    job: u64,
    selected_layers: &[u64],
    tokens: u64,
    outputs: &[TensorExecutionOutput],
) -> Result<()> {
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };
    ensure!(
        selected_layers == PH3ImageMainTextObservation::LAYERS && outputs.len() == 5,
        "main text diagnostic requires the five validated connector taps"
    );
    let select = |layer| -> Result<_> {
        let id =
            ferrum_models::qwen3_vl_text::Qwen3VLTextComfyFamily::hidden_state_output_id(layer)?;
        let output = outputs
            .iter()
            .find(|output| output.value_id() == &id)
            .context("main text diagnostic lacks a validated canonical tap")?;
        Ok(PH3ImageMainTextTap {
            layer,
            value_id: output.value_id(),
            tensor: output.tensor(),
            host_class: output.bytes().class(),
            bytes: output.bytes().bytes(),
        })
    };
    let [a, b, c, d, e] = PH3ImageMainTextObservation::LAYERS.map(select);
    diagnostics.deliver(PH3ImageMainTextObservation {
        request,
        job,
        tokens,
        taps: [a?, b?, c?, d?, e?],
    })
}

#[cfg(test)]
pub(super) mod tests;
