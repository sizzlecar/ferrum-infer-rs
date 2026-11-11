use super::ServerError;
use axum::body::Bytes;
use base64::{engine::general_purpose::STANDARD, write::EncoderWriter};
use ferrum_engine::image_generation::service::{
    ImageOutputGate, ImageOutputGateError, ImageOutputLease,
};
use ferrum_models::image_pixels::OwnedImagePixels;
use ferrum_output::{
    png::{encoded_rgba8_png_bytes, write_rgba8_png, PngDimensions, PngLimits},
    OutputBytesLease, OutputSession,
};
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClass, HostMemoryScope, ManagedHostBuffer, ManagedHostBytes,
};
use std::io::{self, Cursor, Write};

const JSON_PREFIX: &[u8] = b"{\"created\":";
const JSON_DATA: &[u8] = b",\"data\":[{\"b64_json\":\"";
const JSON_SUFFIX: &[u8] = b"\"}]}";

/// The backing charge and output credit outlive the response body if a client
/// still holds any Bytes clone/slice. No encoded String or PNG Vec is produced.
struct EncodedImage {
    bytes: ManagedHostBytes,
    _lease: OutputBytesLease,
    _session: OutputSession,
}
impl AsRef<[u8]> for EncodedImage {
    fn as_ref(&self) -> &[u8] {
        self.bytes.bytes()
    }
}

/// Queue metadata owns a small session permit, never a full frame reservation.
/// The scheduler reserves Host encoding backing only for a selected job.
/// Encoded transport credit is acquired after inference has released the worker.
pub(super) struct ResponseGate {
    session: OutputSession,
    host: HostMemoryAccount,
    dimensions: PngDimensions,
    limits: PngLimits,
    scratch_bytes: usize,
    created: i64,
    count: usize,
    physical_bytes: usize,
    /// Original pipeline-owned physical Output, such as a bounded trace. It
    /// cannot be released by draining client responses and is already charged.
    retained_output_bytes: usize,
}

impl ResponseGate {
    pub(super) fn new(
        session: OutputSession,
        host: HostMemoryAccount,
        dimensions: PngDimensions,
        limits: PngLimits,
        scratch_bytes: usize,
        retained_output_bytes: u64,
    ) -> Result<Self, ServerError> {
        let png_bytes = encoded_rgba8_png_bytes(dimensions, limits).map_err(internal)?;
        let created = chrono::Utc::now().timestamp();
        let count = response_bytes(png_bytes, created)
            .ok_or_else(|| capacity("encoded image length exceeds the host address space"))?;
        if count > session.config().max_event_bytes || count > session.config().encoded_bytes_total
        {
            return Err(capacity("encoded image exceeds configured output limit"));
        }
        let physical = physical_response_bytes(&host, scratch_bytes, count).map_err(capacity)?;
        let retained_output_bytes = usize::try_from(retained_output_bytes).map_err(capacity)?;
        let hard_requirement = physical
            .checked_add(retained_output_bytes)
            .ok_or_else(|| capacity("image response and retained Output capacity overflow"))?;
        host.check_scope_capacity(&[(HostMemoryClass::Output, hard_requirement)])
            .map_err(capacity)?;
        Ok(Self {
            session,
            host,
            dimensions,
            limits,
            scratch_bytes,
            created,
            count,
            physical_bytes: physical,
            retained_output_bytes,
        })
    }

    /// Only Host capacity and closure constrain the executable prefix. Holding
    /// Host encoding backing prevents raw outputs from exhausting the bytes
    /// needed to encode them; transport credit must not constrain GPU batches.
    fn try_admit(&self) -> Result<Option<Admission>, ServerError> {
        if self.session.is_closed() {
            return Err(internal("output budget is closed"));
        }
        let demand = [(HostMemoryClass::Output, self.physical_bytes)];
        if let Err(error) = self.host.check_scope_admission(&demand) {
            return if error.is_temporary_capacity() {
                self.wait_for_external_output(&demand)
            } else {
                Err(internal(error))
            };
        }
        let scope = match self.host.reserve_scope(&demand) {
            Ok(scope) => scope,
            Err(ferrum_runtime::HostMemoryError::Capacity { .. }) => {
                return self.wait_for_external_output(&demand)
            }
            Err(error) => return Err(internal(error)),
        };
        Ok(Some(Admission {
            session: self.session.clone(),
            scope,
            dimensions: self.dimensions,
            limits: self.limits,
            scratch_bytes: self.scratch_bytes,
            created: self.created,
            count: self.count,
        }))
    }

    fn wait_for_external_output(
        &self,
        demand: &[(HostMemoryClass, usize)],
    ) -> Result<Option<Admission>, ServerError> {
        // No own frame grant exists yet. Pipeline-retained Output is a fixed
        // baseline, alongside queued Input/retained Weight. Only actual client
        // Output owners may be hypothetically released in this proof.
        self.host
            .check_scope_after_output_release(demand, self.retained_output_bytes)
            .map_err(|error| match error {
                ferrum_runtime::HostMemoryError::Capacity { .. } => capacity(error),
                other => internal(other),
            })?;
        Ok(None)
    }
}

impl ImageOutputGate for ResponseGate {
    fn host_output_bytes(&self, host: &HostMemoryAccount) -> usize {
        if self.host.same_parent(host) {
            self.physical_bytes
        } else {
            0
        }
    }

    fn try_acquire(&self) -> Result<Option<ImageOutputLease>, ImageOutputGateError> {
        self.try_admit()
            .map(|admission| admission.map(ImageOutputLease::new))
            .map_err(|error| match error {
                ServerError::CapacityExceeded(message) => ImageOutputGateError::Capacity(message),
                other => ImageOutputGateError::Failed(format!("{other:?}")),
            })
    }

    fn subscribe_capacity(&self, waker: std::task::Waker) -> ImageOutputLease {
        ImageOutputLease::new((
            self.host.subscribe_capacity(waker.clone()),
            self.session.subscribe_capacity(waker),
        ))
    }
}

/// This Host grant follows the generated output into response encoding.
pub(super) struct Admission {
    session: OutputSession,
    scope: HostMemoryScope,
    dimensions: PngDimensions,
    limits: PngLimits,
    scratch_bytes: usize,
    created: i64,
    count: usize,
}

impl Admission {
    #[cfg(test)]
    pub(super) fn new(
        session: OutputSession,
        host: HostMemoryAccount,
        dimensions: PngDimensions,
        limits: PngLimits,
        scratch_bytes: usize,
    ) -> Result<Self, ServerError> {
        ResponseGate::new(session, host, dimensions, limits, scratch_bytes, 0)?
            .try_admit()?
            .ok_or_else(|| capacity("selected test request has no current output grant"))
    }

    pub(super) fn session(&self) -> OutputSession {
        self.session.clone()
    }

    pub(super) async fn encode(self, pixels: OwnedImagePixels) -> Result<Bytes, ServerError> {
        // The service worker has already completed. Waiting for unread bodies
        // to return transport credit retains bounded Host backing and pixels,
        // but no execution slot. Cancelling this wait drops both owners.
        let lease = self
            .session
            .reserve_encoded(self.count)
            .await
            .map_err(internal)?;
        // A cancelled blocking task retains its grant and pixels until the
        // writer exits, then drops its unpublished encoded owner safely.
        tokio::task::spawn_blocking(move || PreparedEncoding::new(self, lease, pixels)?.finish())
            .await
            .map_err(internal)?
    }
}

struct PreparedEncoding {
    admission: Admission,
    lease: OutputBytesLease,
    pixels: OwnedImagePixels,
    scratch: ManagedHostBuffer,
}

impl PreparedEncoding {
    fn new(
        admission: Admission,
        lease: OutputBytesLease,
        pixels: OwnedImagePixels,
    ) -> Result<Self, ServerError> {
        let host = admission.scope.account();
        let rgba = pixels.view(&host).map_err(internal)?;
        let [batch, _, height, width] = rgba.rgba_dimensions();
        if batch != 1
            || width as u64 != u64::from(admission.dimensions.width())
            || height as u64 != u64::from(admission.dimensions.height())
        {
            return Err(ServerError::InternalError(
                "image response dimensions differ from the admitted single frame".into(),
            ));
        }
        let scratch = host
            .allocate_zeroed(HostMemoryClass::Output, admission.scratch_bytes)
            .map_err(internal)?;
        Ok(Self {
            admission,
            lease,
            pixels,
            scratch,
        })
    }

    fn write(&mut self, sink: &mut dyn Write) -> io::Result<()> {
        let rgba = self
            .pixels
            .view(&self.admission.scope.account())
            .map_err(io::Error::other)?;
        sink.write_all(JSON_PREFIX)?;
        write!(sink, "{}", self.admission.created)?;
        sink.write_all(JSON_DATA)?;
        {
            let mut base64 = EncoderWriter::new(&mut *sink, &STANDARD);
            write_rgba8_png(
                &mut base64,
                self.admission.dimensions,
                self.admission.limits,
                |stream| {
                    rgba.write_frame_rgba8(0, self.scratch.as_mut(), stream)
                        .map_err(io::Error::other)
                },
            )
            .map_err(io::Error::other)?;
            base64.finish()?;
        }
        sink.write_all(JSON_SUFFIX)?;
        Ok(())
    }

    fn finish(mut self) -> Result<Bytes, ServerError> {
        let mut output = self
            .admission
            .scope
            .account()
            .allocate_zeroed(HostMemoryClass::Output, self.admission.count)
            .map_err(internal)?;
        let mut cursor = Cursor::new(output.as_mut());
        self.write(&mut cursor).map_err(internal)?;
        if cursor.position() != self.admission.count as u64 {
            return Err(ServerError::InternalError(
                "encoded image size differs from the admitted exact length".into(),
            ));
        }
        Ok(Bytes::from_owner(EncodedImage {
            bytes: output.freeze(),
            _lease: self.lease,
            _session: self.admission.session,
        }))
    }
}

fn internal(error: impl std::fmt::Display) -> ServerError {
    ServerError::InternalError(format!("image encoding: {error}"))
}
fn capacity(error: impl std::fmt::Display) -> ServerError {
    ServerError::CapacityExceeded(format!("image output capacity: {error}"))
}

fn response_bytes(png_bytes: u64, created: i64) -> Option<usize> {
    // STANDARD base64 has padding and does not escape any JSON string bytes.
    let base64_bytes = png_bytes.div_ceil(3).checked_mul(4)?;
    let timestamp_bytes =
        created.unsigned_abs().checked_ilog10().unwrap_or(0) as u64 + 1 + u64::from(created < 0);
    let bytes = base64_bytes
        .checked_add(timestamp_bytes)?
        .checked_add((JSON_PREFIX.len() + JSON_DATA.len() + JSON_SUFFIX.len()) as u64)?;
    usize::try_from(bytes)
        .ok()
        .filter(|bytes| *bytes <= isize::MAX as usize)
}

fn physical_response_bytes(
    host: &HostMemoryAccount,
    scratch_bytes: usize,
    count: usize,
) -> anyhow::Result<usize> {
    host.backing_bytes(scratch_bytes)?
        .checked_add(host.backing_bytes(count)?)
        .ok_or_else(|| anyhow::anyhow!("encoded image backing exceeds the host address space"))
}

impl super::ImageHttpConfig {
    /// Maximum physical scratch + JSON/base64 backing for one response, using
    /// the actual gate/encoder layout and the longest supported i64 timestamp.
    /// Pure metadata: neither an OutputSession nor a Host grant is acquired.
    pub fn response_backing_bytes(
        &self,
        host: &HostMemoryAccount,
        dimensions: PngDimensions,
    ) -> anyhow::Result<usize> {
        let png = encoded_rgba8_png_bytes(dimensions, self.png_limits)?;
        let count = response_bytes(png, i64::MIN)
            .ok_or_else(|| anyhow::anyhow!("encoded image length exceeds address space"))?;
        physical_response_bytes(host, self.rgba_scratch_bytes, count)
    }
}

#[cfg(test)]
mod tests;
