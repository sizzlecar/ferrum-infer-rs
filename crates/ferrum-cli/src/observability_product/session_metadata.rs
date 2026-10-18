//! User-supplied labels are session metadata, never physical execution facts.
use super::*;

#[derive(serde::Serialize)]
pub(crate) struct ProductSessionMetadata {
    commit_sha: Option<String>,
    env_hash: String,
    model: String,
    concurrency: u32,
    runtime_flags: Value,
}

impl ProductSessionMetadata {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        commit_sha: Option<String>,
        env_hash: Option<String>,
        model: Option<String>,
        concurrency: Option<u32>,
        runtime_flags_json: Option<String>,
        config: &ferrum_types::ResolvedFerrumConfig,
        model_id: &str,
    ) -> Result<Self> {
        let runtime_flags = match runtime_flags_json {
            Some(json) => {
                let value: Value = serde_json::from_str(&json).map_err(|error| {
                    FerrumError::config(format!("invalid --profile-runtime-flags-json: {error}"))
                })?;
                if !value.is_object() {
                    return Err(FerrumError::config(
                        "--profile-runtime-flags-json must be a JSON object",
                    ));
                }
                value
            }
            None => config.effective_config_document(),
        };
        let env_hash = match env_hash {
            Some(value) if value.starts_with("sha256:") => value,
            Some(_) => {
                return Err(FerrumError::config(
                    "--profile-env-hash must start with sha256:",
                ))
            }
            None => config.runtime_env_hash(),
        };
        Ok(Self {
            commit_sha: commit_sha.filter(|value| !value.trim().is_empty()),
            env_hash,
            model: model
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| model_id.to_owned()),
            concurrency: concurrency
                .filter(|value| *value > 0)
                .unwrap_or_else(|| config.workload_profile.target_concurrency.max(1) as u32),
            runtime_flags,
        })
    }

    pub(crate) fn write(
        &self,
        journals: &ProductJournalSession,
        config: &ProductObservabilityConfig,
    ) -> Result<()> {
        if config.profile_jsonl.is_none() {
            return Ok(());
        }
        let request_id = format!("profile-session-{}", Uuid::new_v4().simple());
        let mut event = actual_base_event(
            config,
            &request_id,
            "product_session_metadata",
            ProfileEventKind::Instant,
            Utc::now(),
        );
        event
            .attributes
            .insert("metadata_scope".into(), json!("product_session"));
        event.attributes.insert(
            "declared_metadata".into(),
            serde_json::to_value(self)
                .map_err(|error| FerrumError::serialization(error.to_string()))?,
        );
        journals.write(&[event], true, false, false)?;
        Ok(())
    }
}
