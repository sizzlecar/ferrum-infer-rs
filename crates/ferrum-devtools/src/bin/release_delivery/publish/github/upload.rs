//! Bounded recovery for one missing, immutable draft asset.
use super::*;
use std::time::Duration;

const MAX_UPLOAD_ATTEMPTS: usize = 3;

enum ObservedAsset {
    Absent,
    Uploaded,
    Starter(Value),
}

enum UploadFailure {
    Ambiguous(String),
    Rejected(String),
}

pub(super) async fn upload_missing_asset(
    api: &GitHub,
    repo: &str,
    accepted: &AcceptedRelease,
    release_id: u64,
    asset: &AcceptedAsset,
    retry_delay: Duration,
) -> Result<(), String> {
    for attempt in 1..=MAX_UPLOAD_ATTEMPTS {
        // Re-observe absence immediately before each POST. An existing starter
        // never grants permission to remove someone else's interrupted upload.
        match observe_asset(api, repo, accepted, release_id, asset, false).await? {
            ObservedAsset::Uploaded => return Ok(()),
            ObservedAsset::Absent => {}
            ObservedAsset::Starter(_) => unreachable!("preflight rejects starters"),
        }
        let failure = match upload_once(api, repo, release_id, asset).await {
            Ok(()) => return Ok(()),
            Err(UploadFailure::Rejected(error)) => return Err(error),
            Err(UploadFailure::Ambiguous(error)) => error,
        };
        eprintln!(
            "{failure}; checking remote state after upload attempt {attempt}/{MAX_UPLOAD_ATTEMPTS}"
        );
        // Even the final attempt may have committed all bytes before losing its
        // response. A fresh, verified complete asset is success without a POST.
        let state = observe_asset(api, repo, accepted, release_id, asset, true).await?;
        if matches!(state, ObservedAsset::Uploaded) {
            return Ok(());
        }
        if attempt == MAX_UPLOAD_ATTEMPTS {
            return Err(format!(
                "{failure}; exhausted {MAX_UPLOAD_ATTEMPTS} upload attempts; remote state retained"
            ));
        }
        if let ObservedAsset::Starter(starter) = state {
            if remove_own_starter(api, repo, accepted, release_id, asset, &starter).await? {
                return Ok(());
            }
        }
        tokio::time::sleep(retry_delay * attempt as u32).await;
    }
    unreachable!("bounded upload loop returns on its final attempt")
}

async fn upload_once(
    api: &GitHub,
    repo: &str,
    release_id: u64,
    asset: &AcceptedAsset,
) -> Result<(), UploadFailure> {
    let bytes = verified_asset_bytes(asset).map_err(UploadFailure::Rejected)?;
    let response = api
        .client
        .post(format!(
            "{}/repos/{repo}/releases/{release_id}/assets",
            api.uploads
        ))
        .query(&[("name", asset.name.as_str())])
        .bearer_auth(&api.token)
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("Accept", "application/vnd.github+json")
        .header("Content-Type", "application/octet-stream")
        .body(bytes)
        .send()
        .await
        .map_err(|error| UploadFailure::Ambiguous(asset_upload_error(&asset.name, error)))?;
    let status = response.status();
    if status != StatusCode::CREATED {
        let request_id = response
            .headers()
            .get("x-github-request-id")
            .and_then(|value| value.to_str().ok())
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 128
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b":-_".contains(&byte))
            });
        let error = format!(
            "asset upload returned HTTP {status} for {}{}",
            asset.name,
            request_id
                .map(|id| format!(" (GitHub request id {id})"))
                .unwrap_or_default(),
        );
        return Err(if matches!(status.as_u16(), 500 | 502 | 503 | 504) {
            UploadFailure::Ambiguous(error)
        } else {
            UploadFailure::Rejected(error)
        });
    }
    // A lost/truncated success response also needs a readback: POST may already
    // have committed the immutable bytes. Never infer success from status alone.
    let remote: Value = response.json().await.map_err(|_| {
        UploadFailure::Ambiguous(format!(
            "asset upload response was unreadable for {}",
            asset.name
        ))
    })?;
    check_remote_asset(api, repo, asset, &remote)
        .await
        .map_err(UploadFailure::Rejected)
}

async fn verify_draft(
    api: &GitHub,
    repo: &str,
    accepted: &AcceptedRelease,
    release_id: u64,
) -> Result<(), String> {
    let release = api
        .get(&format!("/repos/{repo}/releases/{release_id}"))
        .await?
        .ok_or("upload recovery draft disappeared")?;
    let tag = format!("v{}", accepted.version);
    let (observed_id, draft) = validate_release_metadata(&release, &tag, &accepted.notes)?;
    if observed_id != release_id
        || !draft
        || release["target_commitish"].as_str() != Some(accepted.candidate_sha.as_str())
    {
        return Err("upload recovery draft identity or candidate changed".into());
    }
    // Unlike ensure_tag, recovery cannot create a missing tag.
    let reference = api
        .get(&format!("/repos/{repo}/git/ref/tags/{tag}"))
        .await?
        .ok_or("upload recovery tag disappeared")?;
    verify_tag_reference(api, repo, accepted, reference).await
}

async fn observe_asset(
    api: &GitHub,
    repo: &str,
    accepted: &AcceptedRelease,
    release_id: u64,
    asset: &AcceptedAsset,
    attempted_absent: bool,
) -> Result<ObservedAsset, String> {
    verify_draft(api, repo, accepted, release_id).await?;
    let inventory = list_assets(api, repo, release_id).await?;
    let mut ids = BTreeSet::new();
    let mut state = ObservedAsset::Absent;
    for (name, remote) in inventory {
        let id = remote["id"]
            .as_u64()
            .filter(|id| *id > 0)
            .ok_or("upload recovery asset has no valid id")?;
        if !ids.insert(id) {
            return Err("upload recovery inventory repeats an asset id".into());
        }
        let expected = accepted
            .assets
            .iter()
            .find(|expected| expected.name == name)
            .ok_or("upload recovery inventory includes an unaccepted asset")?;
        if name == asset.name && attempted_absent && remote["state"].as_str() == Some("starter") {
            validate_starter(asset, &remote)?;
            state = ObservedAsset::Starter(remote);
        } else {
            check_remote_asset(api, repo, expected, &remote).await?;
            if name == asset.name {
                state = ObservedAsset::Uploaded;
            }
        }
    }
    Ok(state)
}

fn validate_starter(asset: &AcceptedAsset, remote: &Value) -> Result<u64, String> {
    let length = std::fs::metadata(&asset.path)
        .map_err(|_| "accepted asset disappeared")?
        .len();
    let size = remote["size"].as_u64();
    if remote["name"].as_str() != Some(asset.name.as_str())
        || remote["state"].as_str() != Some("starter")
        || remote.get("digest") != Some(&Value::Null)
        || !(size == Some(0) || size == Some(length))
    {
        return Err(format!(
            "upload recovery starter conflicts with accepted asset: {}",
            asset.name
        ));
    }
    remote["id"]
        .as_u64()
        .filter(|id| *id > 0)
        .ok_or_else(|| "upload recovery starter has no valid id".into())
}

/// Returns true if the uncertain upload has completed since the inventory read.
async fn remove_own_starter(
    api: &GitHub,
    repo: &str,
    accepted: &AcceptedRelease,
    release_id: u64,
    asset: &AcceptedAsset,
    observed: &Value,
) -> Result<bool, String> {
    let id = validate_starter(asset, observed)?;
    // The shared publication workflow concurrency serializes our controllers.
    // GitHub DELETE has no conditional state predicate, so also refresh the
    // parent/inventory and the exact asset ID immediately before deletion.
    let latest = observe_asset(api, repo, accepted, release_id, asset, true).await?;
    match latest {
        ObservedAsset::Uploaded => return Ok(true),
        ObservedAsset::Starter(ref current) if validate_starter(asset, current)? == id => {}
        _ => return Err("upload recovery starter changed before cleanup".into()),
    }
    let path = format!("/repos/{repo}/releases/assets/{id}");
    let fresh = api
        .get(&path)
        .await?
        .ok_or("upload recovery starter disappeared before cleanup")?;
    if fresh["id"].as_u64() != Some(id) || fresh["name"].as_str() != Some(asset.name.as_str()) {
        return Err("upload recovery asset identity changed before cleanup".into());
    }
    if fresh["state"].as_str() == Some("uploaded") {
        check_remote_asset(api, repo, asset, &fresh).await?;
        return Ok(true);
    }
    validate_starter(asset, &fresh)?;
    let response = api
        .request(Method::DELETE, &path)
        .send()
        .await
        .map_err(|_| "starter cleanup response unavailable; refusing another upload")?;
    if response.status() != StatusCode::NO_CONTENT {
        return Err(format!(
            "starter cleanup returned HTTP {}; refusing another upload",
            response.status()
        ));
    }
    // An ambiguous or stale deletion must never lead to a replacement POST.
    match observe_asset(api, repo, accepted, release_id, asset, true).await? {
        ObservedAsset::Absent => Ok(false),
        _ => Err("starter cleanup did not leave the accepted asset absent".into()),
    }
}

#[cfg(test)]
#[path = "upload_tests.rs"]
mod tests;
