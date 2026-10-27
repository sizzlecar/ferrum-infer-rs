use super::ServerError;
use axum::body::Bytes;
use base64::{engine::general_purpose::STANDARD, write::EncoderWriter};
use ferrum_models::qwen_image21::pipeline_adapters::QwenImage21RgbaOutput;
use ferrum_output::{
    png::{encoded_rgba8_png_bytes, write_rgba8_png, PngDimensions, PngLimits},
    OutputBytesLease, OutputSession,
};
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClass, HostMemoryScope, ManagedHostBuffer, ManagedHostBytes,
    ResolvedTensorSpec,
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

/// One request's known response bound, reserved before generation submission.
/// Only small metadata is allocated here. Scratch and JSON consume this credit
/// later, while other parent consumers see it as unavailable.
pub(super) struct Admission {
    session: OutputSession,
    scope: HostMemoryScope,
    lease: OutputBytesLease,
    dimensions: PngDimensions,
    limits: PngLimits,
    scratch_bytes: usize,
    created: i64,
    count: usize,
}

impl Admission {
    pub(super) fn new(
        session: OutputSession,
        host: HostMemoryAccount,
        dimensions: PngDimensions,
        limits: PngLimits,
        scratch_bytes: usize,
    ) -> Result<Self, ServerError> {
        let png_bytes = encoded_rgba8_png_bytes(dimensions, limits).map_err(internal)?;
        let created = chrono::Utc::now().timestamp();
        let count = response_bytes(png_bytes, created)
            .ok_or_else(|| capacity("encoded image length exceeds the host address space"))?;
        if count > session.config().max_event_bytes {
            return Err(capacity("encoded image exceeds configured output limit"));
        }
        let physical = host
            .backing_bytes(scratch_bytes)
            .map_err(capacity)?
            .checked_add(host.backing_bytes(count).map_err(capacity)?)
            .ok_or_else(|| capacity("encoded image backing exceeds the host address space"))?;
        let scope = host
            .reserve_scope(&[(HostMemoryClass::Output, physical)])
            .map_err(capacity)?;
        // Do not enqueue a second HTTP waiting room. Admission is immediate
        // and atomic; successful leases survive the response body.
        let lease = session
            .try_reserve_encoded(count)
            .map_err(capacity)?
            .ok_or_else(|| capacity("encoded output budget is currently full"))?;
        Ok(Self {
            session,
            scope,
            lease,
            dimensions,
            limits,
            scratch_bytes,
            created,
            count,
        })
    }

    pub(super) fn session(&self) -> OutputSession {
        self.session.clone()
    }

    pub(super) async fn encode(
        self,
        tensor: ResolvedTensorSpec,
        raw: ManagedHostBytes,
    ) -> Result<Bytes, ServerError> {
        // A cancelled blocking task retains its grant and pixels until the
        // writer exits, then drops its unpublished encoded owner safely.
        tokio::task::spawn_blocking(move || PreparedEncoding::new(self, tensor, raw)?.finish())
            .await
            .map_err(internal)?
    }
}

struct PreparedEncoding {
    admission: Admission,
    tensor: ResolvedTensorSpec,
    raw: ManagedHostBytes,
    scratch: ManagedHostBuffer,
}

impl PreparedEncoding {
    fn new(
        admission: Admission,
        tensor: ResolvedTensorSpec,
        raw: ManagedHostBytes,
    ) -> Result<Self, ServerError> {
        let host = admission.scope.account();
        let rgba = QwenImage21RgbaOutput::new(&host, &tensor, &raw).map_err(internal)?;
        let [batch, _, height, width] = rgba.dimensions();
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
            tensor,
            raw,
            scratch,
        })
    }

    fn write(&mut self, sink: &mut dyn Write) -> io::Result<()> {
        let rgba =
            QwenImage21RgbaOutput::new(&self.admission.scope.account(), &self.tensor, &self.raw)
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
            _lease: self.admission.lease,
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

#[cfg(test)]
mod tests;
