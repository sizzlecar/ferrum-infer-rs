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
    execute_serve_source_with_api_key(source, host, port, aliases, None).await
}

pub(crate) async fn execute_serve_source_with_api_key(
    source: ImageProductSource,
    host: String,
    port: u16,
    aliases: Vec<String>,
    api_key: Option<ferrum_server::ApiKey>,
) -> Result<(), ImageProductError> {
    if let Some(key) = &api_key {
        key.validate().map_err(anyhow::Error::from)?;
    }
    #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
    {
        let _ = (source, host, port, aliases, api_key);
        Err(ImageProductError::Unsupported(
            "image serving requires the cuda feature or the metal feature on macOS",
        ))
    }
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    {
        use ferrum_server::{axum_server::AppState, AxumServer, BoundHttpListener, ServerConfig};
        use std::sync::Arc;

        let listener = BoundHttpListener::bind(&host, port)
            .await
            .map_err(anyhow::Error::from)?;
        let prepared = prepare_http(source, aliases)?;
        let trace = prepared.continuous_trace;
        let images = prepared.images;
        let server = Arc::new(AxumServer::from_state(
            AppState::default().with_images(images.clone()),
        ));
        let server_config = ServerConfig {
            api_key,
            host,
            port,
            ..Default::default()
        };
        tracing::info!(composition=%serde_json::to_string(&prepared.composition).map_err(anyhow::Error::from)?, "image backend composition and device capacity");
        tracing::info!(scheduler=%serde_json::to_string(&prepared.service_config).map_err(anyhow::Error::from)?, "image request scheduling and queue bounds");
        tracing::warn!(profile=prepared.profile, max_num_seqs=prepared.service_config.max_num_seqs,
            "experimental image HTTP uses its declared residency policy; no G3 throughput qualification");
        let mut task = {
            let server = server.clone();
            tokio::spawn(async move {
                server
                    .start_with_listener(&server_config, listener, |_| {})
                    .await
            })
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
            if let Some(trace) = trace {
                trace.export_best_effort();
            }
            return Err(ImageProductError::Server {
                source: result
                    .err()
                    .unwrap_or_else(|| anyhow::anyhow!("image worker has a retained failure")),
                service: images,
                shutdown,
            });
        }
        drop(shutdown);
        if let Some(trace) = trace {
            trace.export_best_effort();
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) struct PreparedHttp {
    pub images: ferrum_server::ImageHttpService,
    profile: &'static str,
    composition: serde_json::Value,
    service_config: ferrum_engine::image_generation::service::ImageServiceConfig,
    continuous_trace: Option<super::trace_artifact::ActiveTraceArtifact>,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn prepare_http(
    source: ImageProductSource,
    aliases: Vec<String>,
) -> Result<PreparedHttp, ImageProductError> {
    use ferrum_engine::image_generation::service::ImageServiceConfig;
    use ferrum_output::png::PngLimits;
    use ferrum_server::{ImageHttpConfig, ImageHttpService};
    let product = source.prepare()?;
    let serving = product
        .serving
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("image serve requires an explicit serving configuration"))?;
    serving.validate()?;
    if aliases.len() > 1 || aliases.first().is_some_and(|name| name.trim().is_empty()) {
        return Err(anyhow::anyhow!(
            "image serve accepts at most one nonempty --served-model-name"
        )
        .into());
    }
    let explicit_alias = !aliases.is_empty();
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
    let (mut pipeline, composition) = product.backend.create(product.assets)?;
    let model = if explicit_alias {
        model
    } else {
        super::composed_model_name(&model, &composition)?
    };
    ferrum_server::ServedModelRegistry::try_new(
        model.clone(),
        ferrum_server::ServedModelKind::ImageGeneration,
        vec![model.clone()],
        vec![],
    )
    .map_err(anyhow::Error::from)?;
    let service_config = ImageServiceConfig {
        batching: serving.batching,
        maximum_pending: serving.maximum_pending,
        maximum_pending_bytes: serving
            .maximum_pending_bytes
            .ok_or_else(|| anyhow::anyhow!("image queue byte budget was not resolved"))?
            .get(),
        maximum_prompt_bytes: serving.maximum_prompt_bytes,
        max_num_seqs: serving.max_num_seqs,
        request_batch_max_wait_ms: serving.request_batch_max_wait_ms,
    };
    let continuous_trace = if let Some(prepared) = product.continuous_trace {
        let context = serde_json::json!({
            "entrypoint":"serve", "composition":&composition,
            "service_config":&service_config
        });
        match pipeline.enable_continuous_trace_artifact(&prepared, &context) {
            Ok(handle) => Some(prepared.bind(handle)),
            Err(source) => {
                if let Err(failure) = pipeline.shutdown() {
                    return Err(ImageProductError::generation_pipeline(failure, pipeline));
                }
                return Err(source.into());
            }
        }
    } else {
        None
    };
    let service = match pipeline.into_service(service_config) {
        Ok(service) => service,
        Err(source) => {
            if let Some(trace) = continuous_trace {
                trace.export_best_effort();
            }
            return Err(source.into());
        }
    };
    let images = match ImageHttpService::new(
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
    ) {
        Ok(images) => images,
        Err(source) => {
            // The original failed constructor retires/drops its service. Do
            // not replace that error with a diagnostic writer failure.
            if let Some(trace) = continuous_trace {
                trace.export_best_effort();
            }
            return Err(source.into());
        }
    };
    Ok(PreparedHttp {
        images,
        profile: product.profile,
        composition,
        service_config,
        continuous_trace,
    })
}
