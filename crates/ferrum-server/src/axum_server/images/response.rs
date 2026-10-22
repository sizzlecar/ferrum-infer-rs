use super::ServerError;
use axum::body::Bytes;
use base64::{engine::general_purpose::STANDARD, write::EncoderWriter};
use ferrum_models::qwen_image21::pipeline_adapters::QwenImage21RgbaOutput;
use ferrum_output::{
    png::{write_rgba8_png, PngDimensions, PngLimits},
    OutputBytesLease, OutputSession,
};
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClass, ManagedHostBuffer, ManagedHostBytes, ResolvedTensorSpec,
};
use std::io::{self, Cursor, Write};

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

struct PreparedEncoding {
    session: OutputSession,
    host: HostMemoryAccount,
    tensor: ResolvedTensorSpec,
    raw: ManagedHostBytes,
    scratch: ManagedHostBuffer,
    dimensions: PngDimensions,
    limits: PngLimits,
    created: i64,
    count: usize,
}

impl PreparedEncoding {
    fn new(
        session: OutputSession,
        host: HostMemoryAccount,
        tensor: ResolvedTensorSpec,
        raw: ManagedHostBytes,
        limits: PngLimits,
        scratch_bytes: usize,
    ) -> Result<Self, ServerError> {
        let rgba = QwenImage21RgbaOutput::new(&host, &tensor, &raw).map_err(internal)?;
        let [batch, _, height, width] = rgba.dimensions();
        if batch != 1 {
            return Err(ServerError::InternalError(
                "image response requires exactly one frame".into(),
            ));
        }
        let dimensions = PngDimensions::new(width as u64, height as u64).map_err(internal)?;
        limits.validate(dimensions).map_err(internal)?;
        let scratch = host
            .allocate_zeroed(HostMemoryClass::Output, scratch_bytes)
            .map_err(capacity)?;
        let mut prepared = Self {
            session,
            host,
            tensor,
            raw,
            scratch,
            dimensions,
            limits,
            created: chrono::Utc::now().timestamp(),
            count: 0,
        };
        let mut counter = CountWriter {
            count: 0,
            limit: prepared.session.config().max_event_bytes,
            exceeded: false,
        };
        if let Err(error) = prepared.write(&mut counter) {
            return Err(if counter.exceeded {
                ServerError::CapacityExceeded(
                    "encoded image exceeds configured output limit".into(),
                )
            } else {
                internal(error)
            });
        }
        prepared.count = counter.count;
        Ok(prepared)
    }

    fn write(&mut self, sink: &mut dyn Write) -> io::Result<()> {
        let rgba = QwenImage21RgbaOutput::new(&self.host, &self.tensor, &self.raw)
            .map_err(io::Error::other)?;
        write!(
            sink,
            "{{\"created\":{},\"data\":[{{\"b64_json\":\"",
            self.created
        )?;
        {
            let mut base64 = EncoderWriter::new(&mut *sink, &STANDARD);
            write_rgba8_png(&mut base64, self.dimensions, self.limits, |stream| {
                rgba.write_frame_rgba8(0, self.scratch.as_mut(), stream)
                    .map_err(io::Error::other)
            })
            .map_err(io::Error::other)?;
            base64.finish()?;
        }
        sink.write_all(b"\"}]}")?;
        Ok(())
    }

    fn finish(mut self, lease: OutputBytesLease) -> Result<Bytes, ServerError> {
        let mut output = self
            .host
            .allocate_zeroed(HostMemoryClass::Output, self.count)
            .map_err(capacity)?;
        let mut cursor = Cursor::new(output.as_mut());
        self.write(&mut cursor).map_err(internal)?;
        if cursor.position() != self.count as u64 {
            return Err(ServerError::InternalError(
                "encoded image size changed between passes".into(),
            ));
        }
        Ok(Bytes::from_owner(EncodedImage {
            bytes: output.freeze(),
            _lease: lease,
            _session: self.session,
        }))
    }
}

pub(super) async fn encode(
    session: OutputSession,
    host: HostMemoryAccount,
    tensor: ResolvedTensorSpec,
    raw: ManagedHostBytes,
    limits: PngLimits,
    scratch_bytes: usize,
) -> Result<Bytes, ServerError> {
    // Blocking tasks own all backing and admission while encoding. Cancellation
    // drops the returned owner after the task finishes, not under the writer.
    let prepared = tokio::task::spawn_blocking(move || {
        PreparedEncoding::new(session, host, tensor, raw, limits, scratch_bytes)
    })
    .await
    .map_err(internal)??;
    let lease = prepared
        .session
        .reserve_encoded(prepared.count)
        .await
        .map_err(capacity)?;
    tokio::task::spawn_blocking(move || prepared.finish(lease))
        .await
        .map_err(internal)?
}

fn internal(error: impl std::fmt::Display) -> ServerError {
    ServerError::InternalError(format!("image encoding: {error}"))
}
fn capacity(error: impl std::fmt::Display) -> ServerError {
    ServerError::CapacityExceeded(format!("image output capacity: {error}"))
}

struct CountWriter {
    count: usize,
    limit: usize,
    exceeded: bool,
}
impl Write for CountWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(next) = self
            .count
            .checked_add(bytes.len())
            .filter(|next| *next <= self.limit)
        else {
            self.exceeded = true;
            return Err(io::Error::other(
                "image response size exceeds configured limit",
            ));
        };
        self.count = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
