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
    engine: Arc<dyn ferrum_engine::LlmInferenceEngine + Send + Sync>,
    journals: Arc<ProductJournalSession>,
    shutdown: tokio::sync::Mutex<EngineShutdown>,
    finished: std::sync::atomic::AtomicBool,
}
enum EngineShutdown {
    Idle,
    Running(tokio::task::JoinHandle<Result<()>>),
    Finished(Result<()>),
}
impl ProductEngineLifetime {
    pub fn new(
        engine: Box<dyn ferrum_engine::LlmInferenceEngine + Send + Sync>,
        journals: Arc<ProductJournalSession>,
    ) -> Self {
        Self {
            engine: Arc::from(engine),
            journals,
            shutdown: tokio::sync::Mutex::new(EngineShutdown::Idle),
            finished: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn engine(&self) -> Arc<dyn ferrum_engine::LlmInferenceEngine + Send + Sync> {
        self.engine.clone()
    }
    pub async fn shutdown(&self) -> Result<()> {
        let mut state = self.shutdown.lock().await;
        if matches!(*state, EngineShutdown::Idle) {
            let engine = self.engine.clone();
            *state = EngineShutdown::Running(tokio::spawn(async move { engine.shutdown().await }));
        }
        if let EngineShutdown::Running(task) = &mut *state {
            let result = task.await.unwrap_or_else(|error| {
                Err(FerrumError::internal(format!(
                    "product engine shutdown task failed: {error}"
                )))
            });
            *state = EngineShutdown::Finished(result);
        }
        match &*state {
            EngineShutdown::Finished(result) => result.clone(),
            _ => unreachable!("joined engine shutdown"),
        }
    }
    pub async fn finish(&self) -> Result<()> {
        let engine = self.shutdown().await;
        let journals = self.journals.finish().await;
        self.finished.store(
            self.journals.is_closed(),
            std::sync::atomic::Ordering::Release,
        );
        if engine.is_err() {
            if let Err(error) = &journals {
                tracing::warn!(%error, "product journal cleanup failed after engine shutdown error");
            }
        }
        engine.and(journals)
    }
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
        let state = std::mem::replace(self.shutdown.get_mut(), EngineShutdown::Idle);
        let engine = self.engine.clone();
        let journals = self.journals.clone();
        // Cold command cancellation only. An existing shutdown job is joined,
        // never replaced by a second invocation.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let result = match state {
                    EngineShutdown::Idle => engine.shutdown().await,
                    EngineShutdown::Running(task) => task
                        .await
                        .unwrap_or_else(|e| Err(FerrumError::internal(e.to_string()))),
                    EngineShutdown::Finished(result) => result,
                };
                if let Err(error) = result {
                    tracing::warn!(%error, "cancelled product command engine cleanup failed");
                }
                if let Err(error) = journals.finish().await {
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
