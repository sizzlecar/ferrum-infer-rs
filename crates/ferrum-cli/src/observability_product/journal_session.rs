//! Product assembly owns writers; each component receives publication scopes.
use super::*;
use ferrum_bench_core::{
    JsonlFinalReceipt, JsonlJournal, JsonlJournalError, JsonlJournalErrorKind,
    JsonlJournalOpenMode, JsonlJournalOwner, JsonlProducerScope,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct OpenFailure {
    kind: std::io::ErrorKind,
    message: String,
    typed: Option<JsonlJournalError>,
}
impl OpenFailure {
    fn new(error: std::io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
            typed: error
                .get_ref()
                .and_then(|e| e.downcast_ref::<JsonlJournalError>())
                .cloned(),
        }
    }
    fn error(&self) -> std::io::Error {
        match &self.typed {
            Some(error) => std::io::Error::other(error.clone()),
            None => std::io::Error::new(self.kind, self.message.clone()),
        }
    }
}
#[derive(Clone)]
struct Record(Arc<FerrumProfileEvent>);
impl serde::Serialize for Record {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.0.as_ref(), serializer)
    }
}
struct Ready {
    owner: JsonlJournalOwner,
    scope: JsonlProducerScope,
    producer: JsonlJournal<Record>,
    final_receipt: Option<JsonlFinalReceipt>,
}
enum Route {
    Ready(Ready),
    Unavailable(OpenFailure),
}
#[derive(Default)]
struct Errors {
    contract: Option<FerrumError>,
    transport: Option<FerrumError>,
}
impl Errors {
    fn result(&self) -> Result<()> {
        self.contract
            .clone()
            .or_else(|| self.transport.clone())
            .map_or(Ok(()), Err)
    }
}

/// The non-clone assembly authority. Internal sharing only keeps the original
/// owners alive across a cancelled wait on the one registered close operation.
pub struct ProductJournalSession {
    detail: ObservabilityProfileDetail,
    roles: BTreeMap<PathBuf, ProfileOutputRoles>,
    routes: Mutex<BTreeMap<PathBuf, Route>>,
    errors: Mutex<Errors>,
    close: tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<()>>>>,
}
impl ProductJournalSession {
    pub fn prepare(
        config: &ProductObservabilityConfig,
        engine: &ferrum_types::EngineConfig,
    ) -> Arc<Self> {
        let mut paths =
            BTreeMap::<PathBuf, (PathBuf, ProfileOutputRoles, JsonlJournalOpenMode)>::new();
        for (path, role) in [
            (config.profile_jsonl.as_ref(), 0),
            (config.memory_profile_jsonl.as_ref(), 1),
            (config.scheduler_trace_jsonl.as_ref(), 2),
        ] {
            if let Some(path) = path {
                let key =
                    ferrum_bench_core::normalize_jsonl_path(path).unwrap_or_else(|_| path.clone());
                let entry = paths.entry(key).or_insert_with(|| {
                    (
                        path.clone(),
                        ProfileOutputRoles::default(),
                        JsonlJournalOpenMode::Append,
                    )
                });
                match role {
                    0 => entry.1.profile = true,
                    1 => entry.1.memory = true,
                    _ => entry.1.scheduler = true,
                }
            }
        }
        for path in ferrum_engine::EngineJournalScopes::configured_paths(engine) {
            let key =
                ferrum_bench_core::normalize_jsonl_path(&path).unwrap_or_else(|_| path.clone());
            // Preserve the existing product epoch: an Engine trace starts a
            // new artifact, while an independent CLI memory log appends. One
            // canonical route is opened once even when these roles alias.
            paths
                .entry(key)
                .and_modify(|entry| entry.2 = JsonlJournalOpenMode::Truncate)
                .or_insert_with(|| {
                    (
                        path,
                        ProfileOutputRoles::default(),
                        JsonlJournalOpenMode::Truncate,
                    )
                });
        }
        let mut roles = BTreeMap::new();
        let mut routes = BTreeMap::new();
        for (key, (path, role, mode)) in paths {
            let route = match JsonlJournalOwner::open(path, mode, config.journal.into()) {
                Ok(owner) => match owner.producer_scope() {
                    Ok(scope) => {
                        let producer = scope.producer();
                        Route::Ready(Ready {
                            owner,
                            scope,
                            producer,
                            final_receipt: None,
                        })
                    }
                    Err(error) => {
                        Route::Unavailable(OpenFailure::new(std::io::Error::other(error)))
                    }
                },
                Err(error) => Route::Unavailable(OpenFailure::new(error)),
            };
            roles.insert(key.clone(), role);
            routes.insert(key, route);
        }
        let result = Arc::new(Self {
            detail: config.profile_detail,
            roles,
            routes: Mutex::new(routes),
            errors: Mutex::new(Errors::default()),
            close: tokio::sync::Mutex::new(None),
        });
        {
            let routes = result.routes.lock().expect("product journal routes");
            for route in routes.values() {
                if let Route::Unavailable(error) = route {
                    result.record_open(error);
                }
            }
        }
        result
    }
    pub fn result(&self) -> Result<()> {
        self.errors.lock().expect("product journal errors").result()
    }
    pub fn is_closed(&self) -> bool {
        self.routes
            .lock()
            .expect("product journal routes")
            .values()
            .all(|route| match route {
                Route::Unavailable(_) => true,
                Route::Ready(ready) => ready
                    .final_receipt
                    .as_ref()
                    .is_some_and(|receipt| receipt.worker_joined),
            })
    }
    fn record_open(&self, error: &OpenFailure) {
        if let Some(error) = &error.typed {
            self.record(error);
        } else {
            self.transport(error.message.clone());
        }
    }
    fn transport(&self, message: String) {
        if self.detail.requires_complete_evidence() {
            self.errors
                .lock()
                .expect("product journal errors")
                .transport
                .get_or_insert_with(|| FerrumError::io(message));
        } else {
            tracing::warn!(error = %message, "optional product journal delivery degraded");
        }
    }
    fn record(&self, error: &JsonlJournalError) {
        if matches!(
            error.kind(),
            JsonlJournalErrorKind::Config
                | JsonlJournalErrorKind::Encoding
                | JsonlJournalErrorKind::Panic
        ) {
            self.errors
                .lock()
                .expect("product journal errors")
                .contract
                .get_or_insert_with(|| FerrumError::internal(error.to_string()));
        } else {
            self.transport(error.to_string());
        }
    }
    fn component_scopes(
        &self,
        paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<(Vec<JsonlProducerScope>, Vec<(PathBuf, std::io::Error)>)> {
        let routes = self.routes.lock().expect("product journal routes");
        let mut selected = std::collections::BTreeSet::new();
        let mut scopes = Vec::new();
        let mut unavailable = Vec::new();
        for path in paths {
            let key = ferrum_bench_core::normalize_jsonl_path(&path).unwrap_or(path);
            if !selected.insert(key.clone()) {
                continue;
            }
            match routes.get(&key) {
                Some(Route::Ready(ready)) => match ready.owner.producer_scope() {
                    Ok(scope) => scopes.push(scope),
                    Err(error) => unavailable.push((key, std::io::Error::other(error))),
                },
                Some(Route::Unavailable(error)) => unavailable.push((key, error.error())),
                None => {
                    return Err(FerrumError::internal(
                        "journal component requested an unassembled destination",
                    ))
                }
            }
        }
        Ok((scopes, unavailable))
    }
    pub fn engine_scopes(
        &self,
        config: &ferrum_types::EngineConfig,
    ) -> Result<ferrum_engine::EngineJournalScopes> {
        let (scopes, failures) =
            self.component_scopes(ferrum_engine::EngineJournalScopes::configured_paths(config))?;
        let mut result = ferrum_engine::EngineJournalScopes::new(scopes)?;
        for (path, error) in failures {
            result = result.with_open_failure(path, error)?;
        }
        Ok(result)
    }
    pub fn http_scopes(
        &self,
        config: &ferrum_server::ServerConfig,
    ) -> Result<ferrum_server::axum_server::HttpJournalScopes> {
        let (scopes, failures) = self.component_scopes(
            [
                config.profile_jsonl.clone(),
                config.memory_profile_jsonl.clone(),
            ]
            .into_iter()
            .flatten(),
        )?;
        let mut result = ferrum_server::axum_server::HttpJournalScopes::new(scopes)?;
        for (path, error) in failures {
            result = result.with_open_failure(path, error)?;
        }
        Ok(result)
    }
    pub(super) fn write(
        &self,
        events: &[FerrumProfileEvent],
        profile: bool,
        memory: bool,
        scheduler: bool,
    ) -> Result<Vec<PathBuf>> {
        for event in events {
            event.validate().map_err(|e| {
                FerrumError::internal(format!("invalid product observability event: {e}"))
            })?;
        }
        let records: Vec<_> = events
            .iter()
            .cloned()
            .map(|event| Record(Arc::new(event)))
            .collect();
        let routes = self.routes.lock().expect("product journal routes");
        let mut written = Vec::new();
        for (path, roles) in &self.roles {
            let batch: Vec<_> = records
                .iter()
                .filter(|event| {
                    (profile && roles.profile)
                        || (memory && roles.memory && event.0.memory.is_some())
                        || (scheduler && roles.scheduler && event.0.resource.is_some())
                })
                .cloned()
                .collect();
            if batch.is_empty() {
                continue;
            }
            if let Some(Route::Ready(ready)) = routes.get(path) {
                match ready.producer.try_enqueue_batch(batch) {
                    Ok(()) => written.push(path.clone()),
                    Err(error) => self.record(&error),
                }
            }
        }
        self.result()?;
        Ok(written)
    }
    /// A cancelled caller leaves this exact blocking join registered. A live
    /// producer refusal keeps every owner and permits a later explicit retry.
    pub async fn finish(self: &Arc<Self>) -> Result<()> {
        let mut close = self.close.lock().await;
        if close.is_none() {
            let session = Arc::clone(self);
            *close = Some(tokio::task::spawn_blocking(move || {
                session.finish_blocking()
            }));
        }
        let result = match close.as_mut().expect("registered product close").await {
            Ok(result) => result,
            Err(error) => Err(FerrumError::internal(format!(
                "product journal finalizer failed: {error}"
            ))),
        };
        *close = None;
        result
    }
    fn finish_blocking(&self) -> Result<()> {
        let mut routes = self.routes.lock().expect("product journal routes");
        let mut incomplete = None;
        for route in routes.values_mut() {
            let Route::Ready(ready) = route else {
                continue;
            };
            if ready.final_receipt.is_some() {
                continue;
            }
            let receipt = ready.scope.finish();
            if let Err(error) = receipt.result() {
                self.record(&error);
            }
            match ready.owner.close() {
                Ok(final_receipt) => {
                    if let Err(error) = final_receipt.result() {
                        self.record(&error);
                    }
                    if final_receipt.flush.cut.dropped != 0
                        || final_receipt.flush.failed != 0
                        || final_receipt.flush.flushed < final_receipt.flush.cut.accepted
                    {
                        self.transport("product journal has incomplete evidence".to_string());
                    }
                    ready.final_receipt = Some(final_receipt);
                }
                Err(error) => {
                    incomplete.get_or_insert_with(|| {
                        FerrumError::internal(format!(
                            "product journal still has active producer custody: {error}"
                        ))
                    });
                }
            }
        }
        if let Some(error) = incomplete {
            // The owner and any final receipts remain intact. A previous
            // Contract stays primary; incomplete cleanup is still visible.
            if let Some(contract) = self
                .errors
                .lock()
                .expect("product journal errors")
                .contract
                .clone()
            {
                tracing::warn!(%error, "product journal final cleanup remains incomplete after contract failure");
                return Err(contract);
            }
            return Err(error);
        }
        self.result()
    }
    #[cfg(test)]
    pub(super) fn flush(&self) -> Result<()> {
        for route in self.routes.lock().expect("routes").values() {
            if let Route::Ready(ready) = route {
                ready
                    .producer
                    .flush()
                    .result()
                    .map_err(|e| FerrumError::io(e.to_string()))?;
            }
        }
        self.result()
    }
}

/// The command's original engine and journals survive cancellation of a
/// shutdown observer. No path lookup or second finalizer is used on an error.
pub struct ProductEngineLifetime {
    engine: Arc<ProductEngineHandle>,
    journals: Arc<ProductJournalSession>,
    finalization: tokio::sync::Mutex<EngineShutdown>,
    finished: std::sync::atomic::AtomicBool,
}
enum EngineShutdown {
    Idle,
    Running(tokio::task::JoinHandle<Result<()>>),
    Finished(Result<()>),
    Unproved,
}

/// All product consumers share this one original shutdown and terminal wait.
/// The handle delegates execution, but never exposes the unwrapped engine.
struct ProductEngineHandle {
    engine: Arc<dyn ferrum_engine::LlmInferenceEngine + Send + Sync>,
    shutdown: tokio::sync::Mutex<EngineShutdown>,
    completion: tokio::sync::Mutex<EngineShutdown>,
}

impl ProductEngineHandle {
    async fn shutdown_once(&self) -> Result<()> {
        let mut state = self.shutdown.lock().await;
        if matches!(*state, EngineShutdown::Idle) {
            let engine = Arc::clone(&self.engine);
            *state = EngineShutdown::Running(tokio::spawn(async move { engine.shutdown().await }));
        }
        observe_engine_task(&mut state, "shutdown", false).await
    }

    async fn wait_for_original_completion(&self) -> Result<()> {
        // A first retained result does not prove retirement. Preserve that
        // result in shutdown, and observe the same original cleanup separately.
        let _ = self.shutdown_once().await;
        let mut state = self.completion.lock().await;
        if matches!(*state, EngineShutdown::Idle) {
            let engine = Arc::clone(&self.engine);
            *state = EngineShutdown::Running(tokio::spawn(async move {
                engine.wait_for_shutdown_completion().await
            }));
        }
        observe_engine_task(&mut state, "completion observation", true).await
    }
}

async fn observe_engine_task(
    state: &mut EngineShutdown,
    operation: &str,
    terminal_proof_required: bool,
) -> Result<()> {
    if let EngineShutdown::Running(task) = state {
        *state = match task.await {
            Ok(result) => EngineShutdown::Finished(result),
            Err(error) if terminal_proof_required => {
                // A panicked observer did not return the required retirement
                // proof. Keep the original engine and journals in the calling
                // finalization task; never close the writer from this failure.
                tracing::error!(%error, "original engine completion observation lost; cleanup remains unproved");
                EngineShutdown::Unproved
            }
            Err(error) => EngineShutdown::Finished(Err(FerrumError::internal(format!(
                "product engine {operation} task failed: {error}"
            )))),
        };
    }
    match state {
        EngineShutdown::Finished(result) => result.clone(),
        EngineShutdown::Unproved => std::future::pending().await,
        _ => unreachable!("joined original engine lifecycle task"),
    }
}

#[async_trait::async_trait]
impl ferrum_interfaces::InferenceEngine for ProductEngineHandle {
    async fn status(&self) -> ferrum_types::EngineStatus {
        self.engine.status().await
    }
    async fn shutdown(&self) -> Result<()> {
        self.shutdown_once().await
    }
    async fn wait_for_shutdown_completion(&self) -> Result<()> {
        self.wait_for_original_completion().await
    }
    fn config(&self) -> &ferrum_types::EngineConfig {
        self.engine.config()
    }
    fn metrics(&self) -> ferrum_types::EngineMetrics {
        self.engine.metrics()
    }
    async fn health_check(&self) -> ferrum_types::HealthStatus {
        self.engine.health_check().await
    }
    fn cache_metrics_snapshot(&self) -> Option<Value> {
        self.engine.cache_metrics_snapshot()
    }
    fn execution_attribution_snapshot(&self) -> Option<Value> {
        self.engine.execution_attribution_snapshot()
    }
    fn scheduler_slo_snapshot(&self) -> Option<Value> {
        self.engine.scheduler_slo_snapshot()
    }
    fn admission_snapshot(&self) -> Result<Option<ferrum_types::ExecutorAdmissionSnapshot>> {
        self.engine.admission_snapshot()
    }
    fn lora_metrics_snapshot(&self) -> Option<Value> {
        self.engine.lora_metrics_snapshot()
    }
}

#[async_trait::async_trait]
impl ferrum_interfaces::engine::LlmInferenceEngine for ProductEngineHandle {
    fn adapter_execution_support(&self) -> ferrum_interfaces::adapter::AdapterExecutionSupport {
        self.engine.adapter_execution_support()
    }
    fn execution_resource_authority(&self) -> ferrum_types::ExecutionResourceAuthority {
        self.engine.execution_resource_authority()
    }
    fn context_capacity(&self) -> Option<usize> {
        self.engine.context_capacity()
    }
    async fn infer(
        &self,
        request: ferrum_types::InferenceRequest,
    ) -> Result<ferrum_output::InferenceOutput> {
        self.engine.infer(request).await
    }
    async fn infer_stream(
        &self,
        request: ferrum_types::InferenceRequest,
    ) -> Result<ferrum_output::InferenceStream> {
        self.engine.infer_stream(request).await
    }
}

impl ProductEngineLifetime {
    pub fn new(
        engine: Box<dyn ferrum_engine::LlmInferenceEngine + Send + Sync>,
        journals: Arc<ProductJournalSession>,
    ) -> Self {
        Self {
            engine: Arc::new(ProductEngineHandle {
                engine: Arc::from(engine),
                shutdown: tokio::sync::Mutex::new(EngineShutdown::Idle),
                completion: tokio::sync::Mutex::new(EngineShutdown::Idle),
            }),
            journals,
            finalization: tokio::sync::Mutex::new(EngineShutdown::Idle),
            finished: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn engine(&self) -> Arc<dyn ferrum_engine::LlmInferenceEngine + Send + Sync> {
        self.engine.clone()
    }
    pub async fn shutdown(&self) -> Result<()> {
        self.engine.shutdown_once().await
    }
    pub async fn finish(&self) -> Result<()> {
        let first_result = self.shutdown().await;
        let mut finalization = self.finalization.lock().await;
        if matches!(*finalization, EngineShutdown::Idle) {
            let engine = Arc::clone(&self.engine);
            let journals = Arc::clone(&self.journals);
            // This is observation of the original Engine cleanup, not another
            // shutdown invocation. The registered task keeps both authorities
            // alive if its caller stops waiting while resources are Retained.
            *finalization = EngineShutdown::Running(tokio::spawn(async move {
                finish_original_cleanup(engine, journals, first_result).await
            }));
        }
        if let EngineShutdown::Running(task) = &mut *finalization {
            let result = task.await.unwrap_or_else(|error| {
                Err(FerrumError::internal(format!(
                    "product final cleanup observation failed: {error}"
                )))
            });
            *finalization = EngineShutdown::Finished(result);
        }
        let result = match &*finalization {
            EngineShutdown::Finished(result) => result.clone(),
            _ => unreachable!("joined product finalization"),
        };
        let closed = self.journals.is_closed();
        self.finished
            .store(closed, std::sync::atomic::Ordering::Release);
        if !closed {
            // A genuine external producer refusal remains retryable through
            // this same authority. Never reopen a path to manufacture success.
            *finalization = EngineShutdown::Idle;
        }
        result
    }
}

async fn finish_original_cleanup(
    engine: Arc<ProductEngineHandle>,
    journals: Arc<ProductJournalSession>,
    first_result: Result<()>,
) -> Result<()> {
    // Return, including Err, proves the original Engine and its producer
    // scopes actually settled. A retained owner keeps this wait pending.
    let settled = engine.wait_for_original_completion().await;
    let journal_result = journals.finish().await;
    if let Err(error) = &settled {
        if first_result.is_err() {
            tracing::warn!(%error, "original engine cleanup settled with an additional error");
        }
    }
    if let Err(error) = &journal_result {
        if first_result.is_err() || settled.is_err() {
            tracing::warn!(%error, "product journal cleanup failed after engine shutdown error");
        }
    }
    first_result.and(settled).and(journal_result)
}
impl std::ops::Deref for ProductEngineLifetime {
    type Target = dyn ferrum_engine::LlmInferenceEngine + Send + Sync;
    fn deref(&self) -> &Self::Target {
        self.engine.as_ref()
    }
}
impl Drop for ProductEngineLifetime {
    fn drop(&mut self) {
        if self.finished.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let finalization = std::mem::replace(self.finalization.get_mut(), EngineShutdown::Idle);
        let engine = self.engine.clone();
        let journals = self.journals.clone();
        // Cold command cancellation only. An existing shutdown job is joined,
        // never replaced by a second invocation.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let EngineShutdown::Running(task) = finalization {
                    match task.await {
                        Ok(Err(error)) => tracing::warn!(%error, "cancelled product final cleanup reported an error"),
                        Err(error) => tracing::warn!(%error, "cancelled product final cleanup observation failed"),
                        Ok(Ok(())) => {},
                    }
                    return;
                }
                let result = engine.shutdown_once().await;
                if let Err(error) = finish_original_cleanup(engine, journals, result).await {
                    tracing::warn!(%error, "cancelled product command journal cleanup incomplete");
                }
            });
        } else {
            tracing::warn!(
                "product command dropped outside runtime with cleanup custody still present"
            );
        }
    }
}

#[cfg(test)]
mod tests;
