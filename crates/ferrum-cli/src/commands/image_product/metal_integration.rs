//! Actual Metal entrypoints with tiny schema-derived weights; no quality claim.
use super::*;
use anyhow::{ensure, Context};
use base64::Engine;
use ferrum_models::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use std::{path::Path, sync::Arc, time::Duration};
#[path = "metal_integration/assets.rs"]
mod assets;

fn explicit_config(root: &Path) -> PathBuf {
    let mut wire = configuration();
    wire["model"] = json!("tiny-metal");
    wire["metadata"] = json!({"model_index":"model_index.json","text_encoder":"text_encoder/config.json",
        "transformer":"transformer/config.json","vae":"vae/config.json","scheduler":"scheduler/scheduler_config.json",
        "tokenizer":"processor/tokenizer.json","tokenizer_config":"processor/tokenizer_config.json",
        "chat_template":"processor/chat_template.jinja","processor":"processor/processor_config.json"});
    wire["weights"] = json!({"text_encoder":{"kind":"single","directory":"text_encoder","file":"model.safetensors"},
        "transformer":{"kind":"single","directory":"transformer","file":"diffusion_pytorch_model.safetensors"},
        "vae":{"kind":"single","directory":"vae","file":"diffusion_pytorch_model.safetensors"}});
    wire["execution_profile"] = json!("experimental_resident_v1");
    // Editing keeps the model's ordinary 1024² conditioning resolution even
    // though output and weights are tiny; budget that full preparation/encoder.
    wire["backend"] = json!({"kind":"metal","memory":{"kind":"explicit_total","bytes":536870912},
        "bootstrap_memory_bytes":16777216,"stream_memory_bytes":1048576,"maximum_streams":"auto",
        "maximum_in_flight_per_stream":1,"maximum_commands_per_submission":2048,"opaque_memory_profile":"metal.product.fixture.v1",
        "host":{"total_bytes":268435456,"input_preprocess_bytes":201326592,"output_bytes":67108864,"transfer_staging_bytes":0,"weight_backing_bytes":0},
        "pipeline":{"maximum_user_tokens":512,"maximum_width":64,"maximum_height":64,"maximum_batch_size":2,
        "residency":"all_stages","maximum_staging_bytes":1048576,"maximum_initialization_commands":32,"maximum_backing_maintenance_passes":8}});
    wire["serving"] = json!({"maximum_pending":4,"maximum_prompt_bytes":4096,"max_num_seqs":2,
        "request_batch_max_wait_ms":10,"default_width":32,"default_height":32,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
    let path = root.join("metal.json");
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    path
}
fn verify_png(bytes: &[u8]) -> anyhow::Result<()> {
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        1 << 20,
        HostMemoryClassLimits {
            input_preprocess_bytes: 1 << 20,
            output_bytes: 0,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    let mut encoded = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())?;
    encoded.as_mut().copy_from_slice(bytes);
    let encoded = encoded.freeze();
    let image = ManagedRgba8Image::decode_png(&host, &encoded, ImageInputLimits::default())?;
    ensure!(
        (image.width(), image.height()) == (32, 32),
        "unexpected PNG dimensions"
    );
    drop((image, encoded));
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "PNG decoding retained host payloads"
    );
    Ok(())
}
async fn verify_response(response: reqwest::Response) -> anyhow::Result<()> {
    let status = response.status();
    let value: Value = response.json().await?;
    ensure!(status.is_success(), "image HTTP {status}: {value}");
    let encoded = value["data"][0]["b64_json"]
        .as_str()
        .context("missing image payload")?;
    verify_png(&base64::engine::general_purpose::STANDARD.decode(encoded)?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metal_cli_png_and_http_generation_editing_use_real_managed_pipeline() -> anyhow::Result<()>
{
    use ferrum_server::{axum_server::AppState, AxumServer, HttpServer, ServerConfig};
    let package = assets::package();
    let root = package.path();
    let path = explicit_config(root);
    let generated = root.join("generated.png");
    // The ordinary CLI adapter validates metadata, creates the Metal runtime,
    // runs every generation stage, retires sessions and writes its normal PNG.
    execute_run(&path, "cat".into(), 32, 32, 7, &generated)
        .map_err(|e| anyhow::anyhow!("CLI generation: {e:?}"))?;
    verify_png(&fs::read(&generated)?)?;
    let edited = root.join("edited.png");
    execute_run_source_with_image(
        ImageProductSource::ExplicitConfig(path.clone()),
        "blue cat".into(),
        32,
        32,
        1 << 40,
        &edited,
        Some(&generated),
    )
    .map_err(|e| anyhow::anyhow!("CLI editing: {e:?}"))?;
    verify_png(&fs::read(&edited)?)?;
    // Standard directory selection also executes, including automatically
    // derived memory/streams; same model declarations under an arbitrary name.
    let directory_source = ImageProductSource::LocalDirectory {
        path: root.into(),
        options: DirectoryOptions::from_cli(
            0.75,
            std::num::NonZeroUsize::new(1 << 30),
            Some(2),
            "metal",
            None,
            None,
        )?,
    };
    let from_directory = root.join("directory.png");
    execute_run_source(directory_source, "cat".into(), 32, 32, 7, &from_directory)
        .map_err(|e| anyhow::anyhow!("directory generation: {e:?}"))?;
    ensure!(
        fs::read(&generated)? == fs::read(&from_directory)?,
        "configuration route changed output"
    );

    // This is exactly the serving product's asset/factory/worker assembly.
    let prepared = crate::commands::image_product::serve::prepare_http(
        ImageProductSource::ExplicitConfig(path),
        vec!["metal-test".into()],
    )
    .map_err(|e| anyhow::anyhow!("HTTP composition: {e:?}"))?;
    let images = prepared.images;
    ensure!(
        images.service().supports_edits(),
        "editing capability lost at HTTP boundary"
    );
    ensure!(
        images.service().maximum_edit_batch_size() == 2,
        "edit batch ceiling lost"
    );
    let host = images.service().host_account().clone();
    let server = Arc::new(AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    ));
    let reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = reservation.local_addr()?.port();
    drop(reservation);
    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port,
        ..Default::default()
    };
    let task = {
        let server = server.clone();
        tokio::spawn(async move { server.start(&config).await })
    };
    let result = async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        let base = format!("http://127.0.0.1:{port}");
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                if task.is_finished() {
                    anyhow::bail!("HTTP server stopped before readiness");
                }
                if client
                    .get(format!("{base}/health"))
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        let request = |prompt, seed| {
            client
                .post(format!("{base}/v1/images/generations"))
                .json(&json!({"model":"metal-test","prompt":prompt,"size":"32x32","seed":seed}))
                .send()
        };
        let (a, b) = tokio::join!(request("cat", 7_u64), request("blue cat", 1_u64 << 40));
        verify_response(a?).await?;
        verify_response(b?).await?;
        let rejected = client
            .post(format!("{base}/v1/images/generations"))
            .json(&json!({"model":"metal-test","prompt":"cat","size":"33x32"}))
            .send()
            .await?;
        ensure!(
            rejected.status().is_client_error(),
            "invalid dimensions accepted"
        );
        drop(rejected);
        let form = reqwest::multipart::Form::new()
            .text("model", "metal-test")
            .text("prompt", "blue cat")
            .text("size", "32x32")
            .text("seed", "7")
            .part(
                "image",
                reqwest::multipart::Part::bytes(fs::read(&generated)?)
                    .file_name("source.png")
                    .mime_str("image/png")?,
            );
        verify_response(
            client
                .post(format!("{base}/v1/images/edits"))
                .multipart(form)
                .send()
                .await?,
        )
        .await?;
        ensure!(
            images.service().is_accepting(),
            "rejected request stopped pipeline"
        );
        Ok::<_, anyhow::Error>(())
    }
    .await;
    // Always join before propagating assertion/transport failures.
    let stopped = server.finish_stop().await;
    let joined = task.await?;
    let receipt = images
        .take_shutdown()
        .await
        .context("missing shutdown owner")?;
    ensure!(
        !receipt.has_failure(),
        "HTTP worker retained a device failure"
    );
    drop(receipt);
    stopped?;
    joined?;
    result?;
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "HTTP shutdown retained host payloads"
    );
    eprintln!("Metal CLI explicit/directory PNG + edit, concurrent HTTP generations + edit, rejection recovery and shutdown passed (tiny zero weights)");
    Ok(())
}
