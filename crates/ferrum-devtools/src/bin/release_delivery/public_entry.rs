//! Check the actual public installer response against the candidate source.
//! Execution of the README pipeline and installed binary checks follow in the
//! release workflow; matching script bytes alone never proves an installation.
use super::public_install::Status;
use clap::Args;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf, time::Duration};

const MAX_SCRIPT_BYTES: usize = 1024 * 1024;

#[derive(Debug, Args)]
pub struct EntryArgs {
    /// Public README installer endpoint, or an explicit loopback test server.
    #[arg(long)]
    pub url: String,
    #[arg(long)]
    pub expected_script: PathBuf,
    /// New receipt; errors preserve the actual response identity when available.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Serialize)]
struct EntryReport {
    schema_version: u32,
    status: Status,
    requested_url: String,
    effective_url: Option<String>,
    expected_sha256: Option<String>,
    actual_sha256: Option<String>,
    response_bytes: Option<usize>,
    http_status: Option<u16>,
    error: Option<String>,
}

async fn observe(args: &EntryArgs, report: &mut EntryReport) -> Result<(), String> {
    let source = fs::read(&args.expected_script).map_err(|error| error.to_string())?;
    if source.is_empty() || source.len() > MAX_SCRIPT_BYTES {
        return Err("candidate installer script is empty or exceeds the script size limit".into());
    }
    report.expected_sha256 = Some(format!("{:x}", Sha256::digest(&source)));
    let url = reqwest::Url::parse(&args.url).map_err(|error| error.to_string())?;
    let loopback = url.scheme() == "http"
        && matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
    if (url.scheme() != "https" && !loopback)
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(
            "installer endpoint requires HTTPS or explicit loopback HTTP, without credentials"
                .into(),
        );
    }
    let client = reqwest::Client::builder()
        .https_only(url.scheme() == "https")
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|error| error.to_string())?;
    let mut response = client
        .get(url)
        .header("Cache-Control", "no-cache")
        .send()
        .await
        .map_err(|error| error.to_string())?;
    report.effective_url = Some(response.url().to_string());
    report.http_status = Some(response.status().as_u16());
    if !response.status().is_success() {
        return Err(format!(
            "public installer returned HTTP {}",
            response.status()
        ));
    }
    if response
        .content_length()
        .is_some_and(|size| size > MAX_SCRIPT_BYTES as u64)
    {
        return Err("public installer response exceeds the script size limit".into());
    }
    let mut actual = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        if chunk.len() > MAX_SCRIPT_BYTES - actual.len() {
            return Err("public installer response exceeds the script size limit".into());
        }
        actual.extend_from_slice(&chunk);
    }
    report.response_bytes = Some(actual.len());
    report.actual_sha256 = Some(format!("{:x}", Sha256::digest(&actual)));
    if actual != source {
        return Err("public installer bytes differ from the release candidate; deploy the matching website script before accepting delivery".into());
    }
    Ok(())
}

pub async fn verify(args: EntryArgs) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .map_err(|error| error.to_string())?;
    let mut report = EntryReport {
        schema_version: 1,
        status: Status::Failed,
        requested_url: args.url.clone(),
        effective_url: None,
        expected_sha256: None,
        actual_sha256: None,
        response_bytes: None,
        http_status: None,
        error: None,
    };
    let result = observe(&args, &mut report).await;
    match &result {
        Ok(()) => report.status = Status::Passed,
        Err(error) => report.error = Some(error.clone()),
    }
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    result
}

#[cfg(test)]
#[path = "public_entry_tests.rs"]
mod tests;
