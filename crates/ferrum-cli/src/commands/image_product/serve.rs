use super::{ImageProductError, ImageProductSource};
use std::path::Path;

pub async fn execute_serve(
    config_path: &Path,
    host: String,
    port: u16,
    aliases: Vec<String>,
) -> Result<(), ImageProductError> {
    execute_serve_source(
        ImageProductSource::ExplicitConfig(config_path.to_owned()),
        host,
        port,
        aliases,
    )
    .await
}

pub async fn execute_serve_source(
    source: ImageProductSource,
    host: String,
    port: u16,
    aliases: Vec<String>,
) -> Result<(), ImageProductError> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (source, host, port, aliases);
        Err(ImageProductError::Unsupported(
            "image serving currently requires a Ferrum binary built with the cuda feature",
        ))
    }
    #[cfg(feature = "cuda")]
    {
        use ferrum_engine::image_generation::service::{ImageService, ImageServiceConfig};
        use ferrum_output::png::PngLimits;
        use ferrum_server::{
            axum_server::AppState, AxumServer, HttpServer, ImageHttpConfig, ImageHttpService,
            ServerConfig,
        };
        use std::sync::Arc;

        let product = source.prepare()?;
        let serving = product.serving.as_ref().ok_or_else(|| {
            anyhow::anyhow!("image serve requires an explicit serving configuration")
        })?;
        serving.validate()?;
        if aliases.len() > 1 || aliases.first().is_some_and(|name| name.trim().is_empty()) {
            return Err(anyhow::anyhow!(
                "image serve accepts at most one nonempty --served-model-name"
            )
            .into());
        }
        let model = aliases
            .into_iter()
            .next()
            .unwrap_or_else(|| product.model.clone());
        ferrum_server::ServedModelRegistry::try_new(
            model.clone(),
            ferrum_server::ServedModelKind::ImageGeneration,
            vec![model.clone()],
            vec![],
        )
        .map_err(anyhow::Error::from)?;
        let (pipeline, composition) = product.cuda.create_pipeline_with_receipt(product.assets)?;
        let service = ImageService::new(
            pipeline,
            ImageServiceConfig {
                maximum_pending: serving.maximum_pending,
                maximum_prompt_bytes: serving.maximum_prompt_bytes,
                max_num_seqs: serving.max_num_seqs,
                request_batch_max_wait_ms: serving.request_batch_max_wait_ms,
            },
        )
        .map_err(anyhow::Error::from)?;
        let images = ImageHttpService::new(
            model,
            service,
            ImageHttpConfig {
                output: serving.output.clone(),
                default_width: serving.default_width,
                default_height: serving.default_height,
                default_seed: serving.default_seed,
                rgba_scratch_bytes: serving.rgba_scratch_bytes,
                png_limits: PngLimits::default(),
            },
        )?;
        let server = Arc::new(AxumServer::from_state(
            AppState::default().with_images(images.clone()),
        ));
        let server_config = ServerConfig {
            host,
            port,
            ..Default::default()
        };
        tracing::info!(composition=%serde_json::to_string(&composition).map_err(anyhow::Error::from)?, "image CUDA composition and observed conservative memory ceiling");
        tracing::warn!(profile=product.profile, max_num_seqs=serving.max_num_seqs,
            "image HTTP uses configured cold batches; no cross-batch residency or G3 throughput qualification");
        let mut task = {
            let server = server.clone();
            tokio::spawn(async move { server.start(&server_config).await })
        };
        let joined = tokio::select! {
            joined = &mut task => Some(joined),
            _ = crate::commands::serve::serve_shutdown_signal() => None,
        };
        // Never cancel device retirement because an observation timeout elapsed.
        let stop = server.finish_stop().await;
        let joined = match joined {
            Some(result) => result,
            None => task.await,
        };
        let shutdown = images.take_shutdown().await;
        let start = joined
            .map_err(anyhow::Error::from)
            .and_then(|r| r.map_err(anyhow::Error::from));
        let result = start.and(stop.map_err(anyhow::Error::from));
        if shutdown
            .as_ref()
            .is_some_and(|receipt| receipt.has_failure())
            || result.is_err()
        {
            return Err(ImageProductError::Server {
                source: result
                    .err()
                    .unwrap_or_else(|| anyhow::anyhow!("image worker has a retained failure")),
                service: images,
                shutdown,
            });
        }
        drop(shutdown);
        Ok(())
    }
}
