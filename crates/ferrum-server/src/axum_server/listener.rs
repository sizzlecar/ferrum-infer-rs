//! Own the listening socket from startup validation through HTTP drain.
use super::*;
use std::{io, net::SocketAddr};

/// A reserved HTTP listening socket. Moving it into the server avoids a race
/// between testing a port and binding it again after expensive model startup.
#[derive(Debug)]
pub struct BoundHttpListener {
    listener: tokio::net::TcpListener,
    address: SocketAddr,
}

impl BoundHttpListener {
    /// Bind before constructing an engine. Dropping this value releases the
    /// socket if model startup fails or serving is cancelled before handoff.
    pub async fn bind(host: &str, port: u16) -> ferrum_types::Result<Self> {
        let host = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(host);
        let listener = tokio::net::TcpListener::bind((host, port))
            .await
            .map_err(|error| bind_error(host, port, error))?;
        let address = listener.local_addr().map_err(|error| {
            Error::network(format!("Cannot inspect bound HTTP listener: {error}"))
        })?;
        Ok(Self { listener, address })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }
}

fn bind_error(host: &str, port: u16, error: io::Error) -> Error {
    let hint = match error.kind() {
        io::ErrorKind::AddrInUse => "The address is already in use. Choose a different --port or stop the service using it.",
        io::ErrorKind::PermissionDenied => "Permission to bind was denied. Choose a different --port or check socket permissions. On Windows, also check reserved TCP port ranges (netsh int ipv4 show excludedportrange protocol=tcp).",
        io::ErrorKind::AddrNotAvailable => "The requested host address is not available on this machine. Check --host or use a local interface address.",
        _ => "Check --host and --port and the operating system's network settings.",
    };
    let address = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    Error::network(format!("Cannot listen on {address}: {error}. {hint}"))
}

impl AxumServer {
    /// Serve on the exact socket reserved before engine startup. `on_listening`
    /// runs only after model capabilities, HTTP profiles and routes are ready,
    /// and receives the actual bound address (including an OS-selected port).
    /// The existing start/stop gate and graceful drain also govern this path.
    pub async fn start_with_listener(
        &self,
        config: &ServerConfig,
        listener: BoundHttpListener,
        on_listening: impl FnOnce(SocketAddr) + Send,
    ) -> ferrum_types::Result<()> {
        self.start_inner(config, Some(listener), on_listening).await
    }

    pub(super) async fn start_inner(
        &self,
        config: &ServerConfig,
        listener: Option<BoundHttpListener>,
        on_listening: impl FnOnce(SocketAddr) + Send,
    ) -> ferrum_types::Result<()> {
        // Publish start before stop can decide HTTP is drained. Release this
        // lock before profile assembly, bind and serving so stop can progress.
        let start_guard = self.lifecycle.stop_lock.lock().await;
        if self.lifecycle.shutdown_requested.load(Ordering::Acquire) {
            return Err(Error::internal(
                "cannot start Axum server after shutdown was requested",
            ));
        }
        if self.lifecycle.running.load(Ordering::Acquire) {
            return Err(Error::internal("Axum server is already running"));
        }
        *self
            .lifecycle
            .profile_detail
            .lock()
            .expect("HTTP selected profile detail") = Some(config.profile_detail);
        self.state.validate_lora_execution_support(
            self.state.served_model_registry.adapter_count() > 0,
        )?;
        self.state.validate_image_configuration()?;
        self.lifecycle
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::internal("Axum server is already running"))?;
        let _run_guard = AxumServerRunGuard {
            lifecycle: Arc::clone(&self.lifecycle),
        };
        drop(start_guard);

        let state = self.prepare_profile_state(
            self.state
                .clone()
                .with_request_dump_dir(config.request_dump_dir.clone())
                .with_profile_jsonl(config.profile_jsonl.clone())
                .with_profile_detail(config.profile_detail)
                .with_profile_journal(config.profile_journal.clone())
                .with_memory_profile_jsonl(config.memory_profile_jsonl.clone()),
        )?;
        let app =
            crate::api_key::protect(self.build_router_with_state(state), config.api_key.as_ref())?;
        let listener = match listener {
            Some(listener) => listener,
            None => BoundHttpListener::bind(&config.host, config.port).await?,
        };
        *self
            .lifecycle
            .bound_address
            .lock()
            .expect("HTTP bound address") = Some(listener.local_addr());
        info!("Server listening on {}", listener.local_addr());
        on_listening(listener.local_addr());
        let lifecycle = Arc::clone(&self.lifecycle);
        axum::serve(listener.listener, app)
            .with_graceful_shutdown(async move { lifecycle.wait_for_shutdown().await })
            .await
            .map_err(|error| Error::network(format!("HTTP server failed: {error}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
