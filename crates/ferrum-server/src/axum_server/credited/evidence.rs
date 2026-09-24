//! Finite terminal evidence is kept under the same lease through file output.
use super::*;
use ferrum_interfaces::{
    output_credit::LeasedOutput,
    output_flow::{CreditedExecutionProfile, CreditedPromptEvidence, OutputCompletion},
};

pub(super) struct Observer {
    state: AppState,
    model: String,
    endpoint: &'static str,
    correlation: Option<BenchmarkRequestCorrelation>,
    started: Instant,
}
impl Observer {
    pub(super) fn new(
        state: &AppState,
        model: String,
        endpoint: &'static str,
        correlation: Option<BenchmarkRequestCorrelation>,
    ) -> Option<Self> {
        (state.profile_jsonl.is_some() || state.request_dump_dir.is_some()).then(|| Self {
            state: state.clone(),
            model,
            endpoint,
            correlation,
            started: Instant::now(),
        })
    }
    pub(super) fn spawn(
        self,
        completion: tokio::sync::oneshot::Receiver<LeasedOutput<OutputCompletion>>,
    ) {
        tokio::spawn(async move {
            match completion.await {
                Ok(completion) => {
                    let elapsed = elapsed_us_since(self.started);
                    if let Err(error) = tokio::task::spawn_blocking(move || {
                        self.write(Arc::new(completion), elapsed)
                    })
                    .await
                    {
                        warn!("credited evidence writer task failed: {error}");
                    }
                }
                Err(error) => {
                    warn!("credited completion closed before diagnostic evidence: {error}")
                }
            }
        });
    }
    fn write(self, completion: Arc<LeasedOutput<OutputCompletion>>, elapsed: u64) {
        if let Some(path) = self.state.profile_jsonl.as_ref() {
            let mut attributes = BTreeMap::new();
            extend_benchmark_profile_attributes(&mut attributes, self.correlation.as_ref());
            let preset = self
                .state
                .auto_config
                .as_ref()
                .map(ResolvedFerrumConfig::runtime_env_hash)
                .unwrap_or_else(|| format!("sha256:{}", sha256_hex(b"serve-profile")));
            let result = CreditedExecutionProfile::new(
                completion.clone(),
                ProfileEntrypoint::Serve,
                self.model.clone(),
                self.endpoint,
                self.state.profile_detail,
                elapsed,
                preset,
                attributes,
            )
            .map_err(|e| e.to_string())
            .and_then(|record| {
                ferrum_bench_core::write_jsonl_owned_record(path, record).map_err(|e| e.to_string())
            });
            if let Err(error) = result {
                warn!("failed to write credited request profile: {error}");
            }
        }
        if let Some(root) = self
            .state
            .request_dump_dir
            .as_ref()
            .filter(|_| matches!(completion.payload(), OutputCompletion::Succeeded { .. }))
        {
            let dir = root.join(completion.request_id().to_string());
            let result = std::fs::create_dir_all(&dir)
                .map_err(|e| e.to_string())
                .and_then(|()| {
                    ferrum_bench_core::write_json_owned_record(
                        &dir.join("prompt_token_ids.json"),
                        CreditedPromptEvidence {
                            completion,
                            model: self.model,
                        },
                    )
                    .map_err(|e| e.to_string())
                });
            if let Err(error) = result {
                warn!("failed to write credited prompt evidence: {error}");
            }
        }
    }
}
