use super::*;
use axum::{
    body::{Body, Bytes},
    http::header,
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBuffer};
use futures::StreamExt;
use std::io::{self, Read};
use tokio::sync::OwnedSemaphorePermit;

struct Reader {
    reader: Box<dyn Read + Send>,
    // In-flight blocking reads keep this permit after Body cancellation.
    _permit: OwnedSemaphorePermit,
}
pub(in crate::axum_server) async fn get(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Response, ServerError> {
    let videos = available(&state)?.clone();
    // Lookup can reclaim expired records and sync deletion tombstones. Keep
    // disk work off the async executor, and bound artifact owners before lookup.
    tokio::task::spawn_blocking(move || {
        let permit = videos.jobs.admit_reader(&videos.readers)?;
        let artifact = match videos.jobs.artifact(id) {
            None => return Ok(missing()),
            Some(Err(status)) => return Ok((StatusCode::CONFLICT, Json(status)).into_response()),
            Some(Ok(artifact)) => artifact,
        };
        response(
            Box::new(artifact.open().map_err(internal)?),
            artifact.encoded_bytes(),
            permit,
            videos.service.host_account().clone(),
            videos.config.read_buffer_bytes,
            videos.shutdown_signal.subscribe(),
        )
    })
    .await
    .map_err(internal)?
}
fn response(
    source: Box<dyn Read + Send>,
    size: u64,
    permit: OwnedSemaphorePermit,
    host: HostMemoryAccount,
    chunk_bytes: usize,
    mut closing: tokio::sync::watch::Receiver<bool>,
) -> Result<Response, ServerError> {
    let initial = host
        .allocate_zeroed(HostMemoryClass::Output, chunk_bytes)
        .map_err(capacity)?;
    let reader = Reader {
        reader: source,
        _permit: permit,
    };
    let stream = async_stream::try_stream! {
        let mut reader=reader;
        let mut buffer=Some(initial);
        loop {
            if *closing.borrow() { break; }
            // Each yielded Bytes retains its actual allocation. If downstream
            // retains chunks, Host capacity remains charged; allocation failure
            // terminates the download rather than allocating outside admission.
            let mut bytes:ManagedHostBuffer=match buffer.take() { Some(b)=>b, None=>host.allocate_zeroed(HostMemoryClass::Output,chunk_bytes).map_err(io::Error::other)? };
            let work=tokio::task::spawn_blocking(move || {
                let count=reader.reader.read(bytes.as_mut());
                (reader,bytes,count)
            });
            let observed=tokio::select! {
                result=work=>Some(result),
                changed=closing.changed()=> { let _=changed; None }
            };
            let Some(observed)=observed else { break; };
            let (next,bytes,count)=observed.map_err(io::Error::other)?;
            reader=next;
            let count=count?;
            if count==0 { break; }
            let bytes=bytes.freeze().slice(0..count).map_err(io::Error::other)?;
            yield Bytes::from_owner(bytes);
        }
    };
    let body = Body::from_stream(stream.map(|result: Result<Bytes, io::Error>| result));
    Ok((
        [
            (header::CONTENT_TYPE, "video/mp4".to_owned()),
            (header::CONTENT_LENGTH, size.to_string()),
            (header::CACHE_CONTROL, "no-store".to_owned()),
        ],
        body,
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_output::{
        mp4::{
            H264AacFragmentedV1, H264SpeedPreset, Mp4Encoder, Mp4EncoderConfig, Mp4Spool,
            Mp4SpoolLimits, VideoAudioSpec,
        },
        OutputBudget,
    };
    use std::time::Duration;

    #[tokio::test]
    #[ignore = "requires explicitly installed ffmpeg/libx264/AAC on PATH"]
    async fn real_mp4_body_drop_retires_file_but_retained_chunk_keeps_host_credit() {
        let ffmpeg = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|p| p.join("ffmpeg"))
            .find(|p| p.is_file())
            .expect("ffmpeg on PATH");
        let host = super::super::tests::host();
        let spec = VideoAudioSpec {
            width: 32,
            height: 24,
            frames: 4,
            fps_numerator: 24,
            fps_denominator: 1,
            audio_sample_rate: 32000,
            audio_sample_frames: 5400,
        };
        let rgb = host
            .allocate_zeroed(HostMemoryClass::Output, spec.rgb_bytes().unwrap())
            .unwrap()
            .freeze();
        let pcm = host
            .allocate_zeroed(HostMemoryClass::Output, spec.pcm_bytes().unwrap())
            .unwrap()
            .freeze();
        let spool = Mp4Spool::create(
            std::env::temp_dir().join(format!("ferrum-http-mp4-{}", uuid::Uuid::new_v4())),
            Mp4SpoolLimits {
                maximum_bytes: 2 << 20,
                maximum_files: 3,
            },
        )
        .unwrap();
        let output = OutputBudget::new(ferrum_types::OutputConfig {
            max_sessions: 1,
            ..Default::default()
        })
        .unwrap();
        let encoder = Mp4Encoder::new(
            Mp4EncoderConfig {
                ffmpeg,
                threads: 1,
                timeout: Duration::from_secs(10),
                maximum_encoded_bytes: 1 << 20,
                io_buffer_bytes: 4096,
                maximum_stderr_bytes: 4096,
                preset: H264AacFragmentedV1 {
                    h264_speed: H264SpeedPreset::Veryfast,
                    crf: 20,
                    audio_bitrate: 128000,
                },
            },
            spool.clone(),
        )
        .unwrap();
        let artifact = encoder
            .try_admit(output.try_open_session().unwrap(), spec)
            .unwrap()
            .unwrap()
            .start(rgb, pcm)
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let mut expected = [0u8; 17];
        artifact.open().unwrap().read_exact(&mut expected).unwrap();
        let readers = Arc::new(tokio::sync::Semaphore::new(1));
        let (_signal, closing) = tokio::sync::watch::channel(false);
        // Moves the only artifact into the HTTP reader, just as deleting a
        // completed registry record while its download is already active.
        let source = artifact.open().unwrap();
        let size = artifact.receipt().encoded_bytes;
        drop(artifact);
        let response = response(
            Box::new(source),
            size,
            readers.clone().acquire_owned().await.unwrap(),
            host.clone(),
            17,
            closing,
        )
        .unwrap();
        assert_eq!(response.headers()[header::CONTENT_TYPE], "video/mp4");
        let mut stream = response.into_body().into_data_stream();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.as_ref(), expected);
        assert_eq!(readers.available_permits(), 0);
        assert!(spool.usage().reserved_bytes > 0);
        assert_eq!(output.usage().sessions, 1);
        drop(stream);
        let permit = tokio::time::timeout(Duration::from_secs(2), readers.acquire())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(spool.usage().reserved_bytes, 0);
        assert_eq!(output.usage().sessions, 0);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 17);
        drop(first);
        drop(permit);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
