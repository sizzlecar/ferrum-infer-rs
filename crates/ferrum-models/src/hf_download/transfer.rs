//! Bounded, resumable transfer of one immutable Hub file.

use ferrum_types::{FerrumError, Result};
use futures_util::StreamExt;
use indicatif::ProgressBar;
use reqwest::{header, Client, RequestBuilder, Response, StatusCode};
use std::error::Error;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::Duration;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::time::{sleep_until, timeout_at, Instant};

/// Retry limits for a single file, including HEAD, body transfer and backoff.
/// Existing on-disk partial bytes remain available when attempts are exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadRetryPolicy {
    pub max_attempts: NonZeroU32,
    pub initial_backoff: Duration,
    pub max_elapsed: Duration,
}

impl Default for DownloadRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: NonZeroU32::new(3).unwrap(),
            initial_backoff: Duration::from_secs(1),
            max_elapsed: Duration::from_secs(3600),
        }
    }
}

impl DownloadRetryPolicy {
    pub(super) fn validate(self) -> Result<()> {
        if self.max_elapsed.is_zero() || Instant::now().checked_add(self.max_elapsed).is_none() {
            return Err(FerrumError::config(
                "download max_elapsed must be positive and representable",
            ));
        }
        Ok(())
    }
}

struct TransferFailure {
    error: FerrumError,
    retryable: bool,
}

impl TransferFailure {
    fn terminal(error: FerrumError) -> Self {
        Self {
            error,
            retryable: false,
        }
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self::terminal(FerrumError::model(message))
    }

    fn incomplete(message: impl Into<String>) -> Self {
        Self {
            error: FerrumError::model(message),
            retryable: true,
        }
    }

    fn request(error: reqwest::Error, filename: &str) -> Self {
        let retryable = error.is_timeout()
            || error.is_connect()
            || error.is_body()
            || error.is_decode()
            || (error.is_request() && !error.is_builder() && !error.is_redirect());
        let error = error.without_url();
        let mut message = error.to_string();
        let mut source = error.source();
        while let Some(cause) = source {
            message.push_str(": ");
            message.push_str(&cause.to_string());
            source = cause.source();
        }
        // Reqwest's top-level URL is removed above. Also redact URLs in nested
        // transport error text; redirects can contain signed credentials.
        for scheme in ["https://", "http://", "socks5://", "socks5h://"] {
            while let Some(start) = message.find(scheme) {
                let end = message[start..]
                    .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>'))
                    .map_or(message.len(), |offset| start + offset);
                message.replace_range(start..end, "[redacted URL]");
            }
        }
        Self {
            error: FerrumError::model(format!("Download error for {filename}: {message}")),
            retryable,
        }
    }
}

impl From<std::io::Error> for TransferFailure {
    fn from(error: std::io::Error) -> Self {
        // Local disk/permission failures cannot be repaired by another GET.
        Self::terminal(error.into())
    }
}

pub(super) struct DownloadRetry {
    policy: DownloadRetryPolicy,
    deadline: Instant,
    attempt: u32,
    last_error: Option<String>,
}

impl DownloadRetry {
    pub(super) fn new(policy: DownloadRetryPolicy) -> Result<Self> {
        policy.validate()?;
        let deadline = Instant::now()
            .checked_add(policy.max_elapsed)
            .ok_or_else(|| FerrumError::config("download deadline is not representable"))?;
        Ok(Self {
            policy,
            deadline,
            attempt: 1,
            last_error: None,
        })
    }

    fn exhausted(&self, filename: &str) -> FerrumError {
        let previous = self
            .last_error
            .as_deref()
            .map(|error| format!("; last error: {error}"))
            .unwrap_or_default();
        FerrumError::timeout(format!(
            "Download time budget exhausted for {filename} after {} attempt(s){previous}",
            self.attempt
        ))
    }

    pub(super) fn ensure_time(&self, filename: &str) -> Result<()> {
        if Instant::now() >= self.deadline {
            return Err(self.exhausted(filename));
        }
        Ok(())
    }

    async fn retry(&mut self, failure: TransferFailure, filename: &str) -> Result<()> {
        if !failure.retryable {
            return Err(failure.error);
        }
        self.last_error = Some(failure.error.to_string());
        self.ensure_time(filename)?;
        if self.attempt >= self.policy.max_attempts.get() {
            return Err(FerrumError::model(format!(
                "Download failed for {filename} after {} attempt(s): {}",
                self.attempt, failure.error
            )));
        }
        let factor = 2u32.saturating_pow(self.attempt - 1);
        let backoff = self.policy.initial_backoff.saturating_mul(factor);
        let wake = Instant::now()
            .checked_add(backoff)
            .unwrap_or(self.deadline)
            .min(self.deadline);
        sleep_until(wake).await;
        self.ensure_time(filename)?;
        self.attempt += 1;
        eprintln!(
            "  ↻ Retrying {filename} ({}/{}) after a temporary transfer error",
            self.attempt, self.policy.max_attempts
        );
        Ok(())
    }

    async fn send(
        &self,
        request: RequestBuilder,
        filename: &str,
    ) -> std::result::Result<Response, TransferFailure> {
        self.ensure_time(filename)
            .map_err(TransferFailure::terminal)?;
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        let response = timeout_at(self.deadline, request.timeout(remaining).send())
            .await
            .map_err(|_| TransferFailure::terminal(self.exhausted(filename)))?
            .map_err(|error| TransferFailure::request(error, filename))?;
        self.ensure_time(filename)
            .map_err(TransferFailure::terminal)?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(TransferFailure {
                error: FerrumError::model(format!("Failed to download {filename} ({status})")),
                retryable: matches!(
                    status,
                    StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS
                ) || status.is_server_error(),
            });
        }
        Ok(response)
    }

    pub(super) async fn head(
        &mut self,
        client: &Client,
        url: &str,
        token: Option<&str>,
        filename: &str,
    ) -> Result<Response> {
        loop {
            let mut request = client.head(url);
            if let Some(token) = token {
                request = request.bearer_auth(token);
            }
            match self.send(request, filename).await {
                Ok(response) => return Ok(response),
                Err(failure) => self.retry(failure, filename).await?,
            }
        }
    }
}

/// The byte interval accepted before an incomplete file may be opened for write.
struct ResponsePlan {
    start: u64,
    end_exclusive: Option<u64>,
    total: Option<u64>,
}

impl ResponsePlan {
    fn validate(
        response: &Response,
        requested_start: u64,
        known_total: Option<u64>,
    ) -> std::result::Result<Self, TransferFailure> {
        match response.status() {
            StatusCode::OK => {
                if response.headers().contains_key(header::CONTENT_RANGE) {
                    return Err(TransferFailure::protocol(
                        "Unexpected Content-Range on HTTP 200",
                    ));
                }
                let length = response.content_length();
                if known_total
                    .zip(length)
                    .is_some_and(|(known, length)| known != length)
                {
                    return Err(TransferFailure::protocol(
                        "HTTP 200 length differs from file metadata",
                    ));
                }
                let total = known_total.or(length);
                // RFC 9110 permits ignoring Range. Such a response replaces the
                // incomplete file from byte zero; it must never be appended.
                Ok(Self {
                    start: 0,
                    end_exclusive: total,
                    total,
                })
            }
            StatusCode::PARTIAL_CONTENT => {
                if response
                    .headers()
                    .get_all(header::CONTENT_RANGE)
                    .iter()
                    .count()
                    != 1
                {
                    return Err(TransferFailure::protocol(
                        "HTTP 206 requires exactly one Content-Range",
                    ));
                }
                let range = response
                    .headers()
                    .get(header::CONTENT_RANGE)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| {
                        TransferFailure::protocol("Missing or invalid Content-Range on HTTP 206")
                    })?;
                let (interval, total) = range
                    .strip_prefix("bytes ")
                    .and_then(|range| range.split_once('/'))
                    .ok_or_else(|| {
                        TransferFailure::protocol("Invalid Content-Range unit or syntax")
                    })?;
                let (start, end) = interval
                    .split_once('-')
                    .ok_or_else(|| TransferFailure::protocol("Invalid Content-Range interval"))?;
                let number = |text: &str| -> std::result::Result<u64, TransferFailure> {
                    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
                        return Err(TransferFailure::protocol("Invalid Content-Range number"));
                    }
                    text.parse()
                        .map_err(|_| TransferFailure::protocol("Content-Range number overflow"))
                };
                let (start, end, total) = (number(start)?, number(end)?, number(total)?);
                if start != requested_start
                    || start > end
                    || end >= total
                    || known_total.is_some_and(|known| known != total)
                {
                    return Err(TransferFailure::protocol(
                        "Content-Range disagrees with requested offset or file size",
                    ));
                }
                let length = end - start + 1;
                if response
                    .content_length()
                    .is_some_and(|actual| actual != length)
                {
                    return Err(TransferFailure::protocol(
                        "HTTP 206 body length disagrees with Content-Range",
                    ));
                }
                Ok(Self {
                    start,
                    end_exclusive: Some(end + 1),
                    total: Some(total),
                })
            }
            _ => Err(TransferFailure::protocol(
                "Expected HTTP 200 or 206 for file download",
            )),
        }
    }
}

pub(super) struct FileTransfer<'a> {
    pub client: &'a Client,
    pub url: &'a str,
    pub token: Option<&'a str>,
    pub filename: &'a str,
    pub incomplete_path: &'a Path,
    pub total_size: Option<u64>,
    pub progress: &'a ProgressBar,
}

impl FileTransfer<'_> {
    pub(super) async fn download(mut self, retry: &mut DownloadRetry) -> Result<u64> {
        loop {
            match self.attempt(retry).await {
                Ok(size) => return Ok(size),
                Err(failure) => retry.retry(failure, self.filename).await?,
            }
        }
    }

    async fn attempt(
        &mut self,
        retry: &DownloadRetry,
    ) -> std::result::Result<u64, TransferFailure> {
        retry
            .ensure_time(self.filename)
            .map_err(TransferFailure::terminal)?;
        let existing_size = match fs::metadata(self.incomplete_path).await {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        let resume_from =
            if existing_size > 0 && self.total_size.is_none_or(|total| existing_size < total) {
                existing_size
            } else {
                0
            };
        let mut request = self.client.get(self.url);
        if let Some(token) = self.token {
            request = request.bearer_auth(token);
        }
        if resume_from > 0 {
            request = request.header(header::RANGE, format!("bytes={resume_from}-"));
        }
        let response = retry.send(request, self.filename).await?;
        let plan = ResponsePlan::validate(&response, resume_from, self.total_size)?;
        self.total_size = plan.total;
        if let Some(total) = plan.total {
            self.progress.set_length(total);
        }
        self.progress.set_position(plan.start);

        // Validate status/range/length before truncating or appending any bytes.
        retry
            .ensure_time(self.filename)
            .map_err(TransferFailure::terminal)?;
        let mut file = if plan.start > 0 {
            OpenOptions::new()
                .append(true)
                .open(self.incomplete_path)
                .await?
        } else {
            File::create(self.incomplete_path).await?
        };
        let mut downloaded = plan.start;
        let mut stream = response.bytes_stream();
        loop {
            // Ready chunks can win timeout_at's first poll even at the deadline.
            // Explicitly check between chunks as well as bounding pending reads.
            if let Err(error) = retry.ensure_time(self.filename) {
                file.flush().await?;
                return Err(TransferFailure::terminal(error));
            }
            let next = match timeout_at(retry.deadline, stream.next()).await {
                Ok(next) => next,
                Err(_) => {
                    file.flush().await?;
                    return Err(TransferFailure::terminal(retry.exhausted(self.filename)));
                }
            };
            let Some(chunk) = next else { break };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    // Tokio's file writes may still be buffered. Flush before
                    // another attempt derives its Range from the on-disk size.
                    file.flush().await?;
                    return Err(TransferFailure::request(error, self.filename));
                }
            };
            let next_size = downloaded.checked_add(chunk.len() as u64);
            if next_size.is_none()
                || plan
                    .end_exclusive
                    .zip(next_size)
                    .is_some_and(|(end, next)| next > end)
            {
                // Do not write an offending chunk, and discard this response's
                // earlier bytes so an invalid 206 cannot pollute a valid prefix.
                file.flush().await?;
                file.set_len(plan.start).await?;
                self.progress.set_position(plan.start);
                return Err(TransferFailure::protocol(
                    "Download body exceeds its declared byte range",
                ));
            }
            file.write_all(&chunk).await?;
            downloaded = next_size.unwrap();
            self.progress.set_position(downloaded);
        }
        file.flush().await?;
        drop(file);
        retry
            .ensure_time(self.filename)
            .map_err(TransferFailure::terminal)?;
        let size = fs::metadata(self.incomplete_path).await?.len();
        if size != downloaded {
            return Err(TransferFailure::protocol(
                "Incomplete file size differs from bytes written",
            ));
        }
        // Enforce metadata and Content-Range lengths even for chunked bodies.
        if plan.end_exclusive.is_some_and(|end| size != end)
            || plan.total.is_some_and(|total| size != total)
        {
            return Err(TransferFailure::incomplete(format!(
                "Incomplete download for {}: got {} bytes, expected {}",
                self.filename,
                size,
                plan.total.or(plan.end_exclusive).unwrap()
            )));
        }
        Ok(size)
    }
}
