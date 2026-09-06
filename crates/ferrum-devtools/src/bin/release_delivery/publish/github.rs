//! Reconcile immutable release inputs. Never replace conflicting tags/assets.
use super::{
    http_client, sha256, verified_asset_bytes, AcceptedAsset, AcceptedRelease, PublishArgs,
};
use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Value};
use std::collections::BTreeMap;

struct GitHub {
    client: Client,
    base: String,
    uploads: String,
    token: String,
}
impl GitHub {
    fn new(token: String) -> Result<Self, String> {
        if token.is_empty() {
            return Err("GitHub token environment variable is empty".into());
        }
        Ok(Self {
            client: http_client()?,
            base: "https://api.github.com".into(),
            uploads: "https://uploads.github.com".into(),
            token,
        })
    }
    fn request(&self, method: Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{}", self.base, path))
            .bearer_auth(&self.token)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("Accept", "application/vnd.github+json")
    }
    async fn get(&self, path: &str) -> Result<Option<Value>, String> {
        let response = self
            .request(Method::GET, path)
            .send()
            .await
            .map_err(|_| "GitHub read failed (remote state unknown)")?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(format!(
                "GitHub read returned HTTP {}; not treating this as missing",
                response.status()
            ));
        }
        response
            .json()
            .await
            .map(Some)
            .map_err(|_| "GitHub returned invalid JSON".into())
    }
    async fn write(&self, method: Method, path: &str, body: &Value) -> Result<Value, String> {
        let response = self
            .request(method, path)
            .json(body)
            .send()
            .await
            .map_err(|_| "GitHub write response unavailable; reconcile before retrying")?;
        if !response.status().is_success() {
            return Err(format!(
                "GitHub write returned HTTP {}; reconcile before retrying",
                response.status()
            ));
        }
        response
            .json()
            .await
            .map_err(|_| "GitHub write response was invalid; reconcile before retrying".into())
    }
    async fn raw(&self, path: &str) -> Result<String, String> {
        let response = self
            .request(Method::GET, path)
            .header("Accept", "application/vnd.github.raw+json")
            .send()
            .await
            .map_err(|_| "GitHub contents query failed")?;
        if !response.status().is_success() {
            return Err(format!(
                "GitHub contents query returned HTTP {}",
                response.status()
            ));
        }
        response
            .text()
            .await
            .map_err(|_| "GitHub contents were unreadable".into())
    }
}

pub(super) async fn release_and_tap(
    args: &PublishArgs,
    accepted: &AcceptedRelease,
) -> Result<(), String> {
    // Only this gated controller entry reads credentials. Never persist them.
    let github = GitHub::new(
        std::env::var("GITHUB_TOKEN").map_err(|_| "GITHUB_TOKEN is required for publication")?,
    )?;
    reconcile_release(&github, &args.repo, accepted).await?;
    let tap = GitHub::new(
        std::env::var(&args.tap_token_env)
            .map_err(|_| format!("{} is required for the tap", args.tap_token_env))?,
    )?;
    reconcile_tap(&tap, args, accepted).await
}

async fn ensure_tag(api: &GitHub, repo: &str, accepted: &AcceptedRelease) -> Result<(), String> {
    let tag = format!("v{}", accepted.version);
    let path = format!("/repos/{repo}/git/ref/tags/{tag}");
    let reference = match api.get(&path).await? {
        Some(value) => value,
        None => {
            // Check that the accepted commit actually exists in the target repo.
            let commit = api
                .get(&format!(
                    "/repos/{repo}/git/commits/{}",
                    accepted.candidate_sha
                ))
                .await?
                .ok_or("accepted candidate commit is absent from the release repository")?;
            if commit["sha"].as_str() != Some(accepted.candidate_sha.as_str()) {
                return Err("GitHub candidate commit identity mismatch".into());
            }
            api.write(
                Method::POST,
                &format!("/repos/{repo}/git/refs"),
                &json!({"ref": format!("refs/tags/{tag}"), "sha": accepted.candidate_sha}),
            )
            .await?;
            api.get(&path)
                .await?
                .ok_or("created Git tag is not readable")?
        }
    };
    let mut object = reference["object"].clone();
    // Annotated tags can refer to another tag; detect cycles and bound bad input.
    let mut seen = std::collections::BTreeSet::new();
    loop {
        let sha = object["sha"].as_str().ok_or("Git tag has no object SHA")?;
        match object["type"].as_str() {
            Some("commit") if sha == accepted.candidate_sha => return Ok(()),
            Some("commit") => {
                return Err(
                    "release tag points to a different candidate; refusing to overwrite".into(),
                )
            }
            Some("tag") => {
                if !seen.insert(sha.to_string()) || seen.len() > 32 {
                    return Err("invalid annotated tag chain".into());
                }
                object = api
                    .get(&format!("/repos/{repo}/git/tags/{sha}"))
                    .await?
                    .ok_or("annotated tag object is missing")?["object"]
                    .clone();
            }
            _ => return Err("release tag does not resolve to a commit".into()),
        }
    }
}

async fn list_assets(
    api: &GitHub,
    repo: &str,
    release: u64,
) -> Result<BTreeMap<String, Value>, String> {
    let mut assets = BTreeMap::new();
    let mut page = 1;
    loop {
        let response = api
            .get(&format!(
                "/repos/{repo}/releases/{release}/assets?per_page=100&page={page}"
            ))
            .await?
            .ok_or("GitHub release disappeared during reconciliation")?;
        let rows = response
            .as_array()
            .ok_or("GitHub assets response is not an array")?;
        for asset in rows {
            let name = asset["name"]
                .as_str()
                .ok_or("GitHub asset is missing a name")?;
            if assets.insert(name.into(), asset.clone()).is_some() {
                return Err("GitHub release has duplicate asset names".into());
            }
        }
        if rows.len() < 100 {
            break;
        }
        page += 1;
    }
    Ok(assets)
}

async fn check_remote_asset(
    api: &GitHub,
    repo: &str,
    expected: &AcceptedAsset,
    remote: &Value,
) -> Result<(), String> {
    if remote["name"].as_str() != Some(expected.name.as_str())
        || remote["state"].as_str() != Some("uploaded")
    {
        return Err(format!(
            "remote release asset is incomplete or mismatched: {}",
            expected.name
        ));
    }
    let expected_len = std::fs::metadata(&expected.path)
        .map_err(|_| "accepted asset disappeared")?
        .len();
    if remote["size"].as_u64() != Some(expected_len) {
        return Err(format!("remote asset size conflict: {}", expected.name));
    }
    if let Some(digest) = remote["digest"].as_str() {
        if digest != format!("sha256:{}", expected.sha256) {
            return Err(format!("remote asset checksum conflict: {}", expected.name));
        }
        return Ok(());
    }
    // Older releases may omit GitHub's authoritative digest; compare download bytes.
    let id = remote["id"].as_u64().ok_or("release asset has no id")?;
    let response = api
        .request(Method::GET, &format!("/repos/{repo}/releases/assets/{id}"))
        .header("Accept", "application/octet-stream")
        .send()
        .await
        .map_err(|_| "release asset download failed")?;
    if !response.status().is_success() {
        return Err(format!(
            "release asset download returned HTTP {}",
            response.status()
        ));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|_| "release asset download was incomplete")?;
    if sha256(&bytes) != expected.sha256 {
        return Err(format!("remote asset checksum conflict: {}", expected.name));
    }
    Ok(())
}

async fn reconcile_release(
    api: &GitHub,
    repo: &str,
    accepted: &AcceptedRelease,
) -> Result<(), String> {
    ensure_tag(api, repo, accepted).await?;
    let tag = format!("v{}", accepted.version);
    let release = match api.get(&format!("/repos/{repo}/releases/tags/{tag}")).await? {
        Some(release) => release,
        None => api.write(Method::POST, &format!("/repos/{repo}/releases"), &json!({"tag_name": tag, "target_commitish": accepted.candidate_sha, "name": tag, "body": accepted.notes, "draft": true, "prerelease": false})).await?,
    };
    if release["tag_name"].as_str() != Some(tag.as_str())
        || release["prerelease"].as_bool() != Some(false)
        || release["body"].as_str() != Some(accepted.notes.as_str())
    {
        return Err("existing release metadata conflicts with accepted version/notes".into());
    }
    let id = release["id"]
        .as_u64()
        .ok_or("GitHub release is missing id")?;
    let draft = release["draft"]
        .as_bool()
        .ok_or("GitHub release is missing draft state")?;
    if !draft
        && release["published_at"]
            .as_str()
            .filter(|value| !value.is_empty())
            .is_none()
    {
        return Err("formal GitHub release is missing publication confirmation".into());
    }
    let existing = list_assets(api, repo, id).await?;
    if existing
        .keys()
        .any(|name| !accepted.assets.iter().any(|a| &a.name == name))
    {
        return Err("existing release includes assets outside the accepted inventory".into());
    }
    // Validate all existing assets before adding any missing asset.
    for asset in &accepted.assets {
        if let Some(remote) = existing.get(&asset.name) {
            check_remote_asset(api, repo, asset, remote).await?;
        }
    }
    for asset in &accepted.assets {
        if existing.contains_key(&asset.name) {
            continue;
        }
        if !draft {
            return Err("formal release is missing accepted assets; refusing to mutate a partially published release".into());
        }
        let bytes = verified_asset_bytes(asset)?;
        let response = api
            .client
            .post(format!("{}/repos/{repo}/releases/{id}/assets", api.uploads))
            .query(&[("name", asset.name.as_str())])
            .bearer_auth(&api.token)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("Content-Type", "application/octet-stream")
            .body(bytes)
            .send()
            .await
            .map_err(|_| "asset upload response unavailable; reconcile before retrying")?;
        if !response.status().is_success() {
            return Err(format!(
                "asset upload returned HTTP {}; reconcile before retrying",
                response.status()
            ));
        }
        let remote: Value = response
            .json()
            .await
            .map_err(|_| "asset upload response was invalid; reconcile before retrying")?;
        check_remote_asset(api, repo, asset, &remote).await?;
    }
    let final_assets = list_assets(api, repo, id).await?;
    if final_assets.len() != accepted.assets.len() {
        return Err("release inventory changed during upload".into());
    }
    for asset in &accepted.assets {
        verified_asset_bytes(asset)?;
        check_remote_asset(
            api,
            repo,
            asset,
            final_assets
                .get(&asset.name)
                .ok_or("accepted asset disappeared")?,
        )
        .await?;
    }
    ensure_tag(api, repo, accepted).await?;
    if draft {
        let formal = api
            .write(
                Method::PATCH,
                &format!("/repos/{repo}/releases/{id}"),
                &json!({"draft": false, "prerelease": false}),
            )
            .await?;
        if formal["draft"].as_bool() != Some(false) || formal["published_at"].as_str().is_none() {
            return Err("GitHub has not confirmed formal release publication".into());
        }
    }
    Ok(())
}

async fn reconcile_tap(
    api: &GitHub,
    args: &PublishArgs,
    accepted: &AcceptedRelease,
) -> Result<(), String> {
    let repo = &args.tap_repo;
    let head = api
        .get(&format!("/repos/{repo}/git/ref/heads/{}", args.tap_branch))
        .await?
        .ok_or("tap branch is missing")?;
    let head_sha = head["object"]["sha"]
        .as_str()
        .ok_or("tap branch has no commit SHA")?;
    let commit = api
        .get(&format!("/repos/{repo}/git/commits/{head_sha}"))
        .await?
        .ok_or("tap commit is missing")?;
    let base_tree = commit["tree"]["sha"]
        .as_str()
        .ok_or("tap commit has no tree")?;
    let mut entries = Vec::new();
    for path in ["Formula/ferrum.rb", "Formula/ferrum-cuda.rb"] {
        let original = api
            .raw(&format!("/repos/{repo}/contents/{path}?ref={head_sha}"))
            .await?;
        let replacement = update_formula(&original, &args.repo, accepted)?;
        if original != replacement {
            entries.push(
                json!({"path": path, "mode": "100644", "type": "blob", "content": replacement}),
            );
        }
    }
    if entries.is_empty() {
        return Ok(());
    }
    // One tree/commit updates both formulas, preserving every other path. The
    // non-forced ref update refuses concurrent changes; a retry re-reads them.
    let tree = api
        .write(
            Method::POST,
            &format!("/repos/{repo}/git/trees"),
            &json!({"base_tree": base_tree, "tree": entries}),
        )
        .await?;
    let tree_sha = tree["sha"].as_str().ok_or("new tap tree has no SHA")?;
    let commit = api.write(Method::POST, &format!("/repos/{repo}/git/commits"), &json!({"message": format!("Update Ferrum to v{}", accepted.version), "tree": tree_sha, "parents": [head_sha]})).await?;
    let new_sha = commit["sha"].as_str().ok_or("new tap commit has no SHA")?;
    let updated = api
        .write(
            Method::PATCH,
            &format!("/repos/{repo}/git/refs/heads/{}", args.tap_branch),
            &json!({"sha": new_sha, "force": false}),
        )
        .await?;
    if updated["object"]["sha"].as_str() != Some(new_sha) {
        return Err("tap branch update was not confirmed".into());
    }
    Ok(())
}

fn string_argument(line: &str, method: &str) -> Option<(usize, usize)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with(&format!("{method} \"")) {
        return None;
    }
    let start = line.find('"')? + 1;
    let end = start + line[start..].find('"')?;
    Some((start, end))
}

fn update_formula(source: &str, repo: &str, accepted: &AcceptedRelease) -> Result<String, String> {
    let prefix = format!("https://github.com/{repo}/releases/download/v");
    let version = semver::Version::parse(&accepted.version)
        .map_err(|_| "invalid accepted formula version")?;
    let mut lines: Vec<String> = source.split_inclusive('\n').map(str::to_string).collect();
    let mut changed_urls = 0;
    let mut observed_versions = std::collections::BTreeSet::new();
    for index in 0..lines.len() {
        let Some((start, end)) = string_argument(&lines[index], "url") else {
            continue;
        };
        let url = lines[index][start..end].to_string();
        let Some(tail) = url.strip_prefix(&prefix) else {
            return Err(
                "formula has an unsupported download URL; refusing broad replacement".into(),
            );
        };
        let (old_version, name) = tail
            .split_once('/')
            .ok_or("formula release URL is malformed")?;
        let old =
            semver::Version::parse(old_version).map_err(|_| "formula version is malformed")?;
        if old > version {
            return Err("tap already advertises a newer release; refusing downgrade".into());
        }
        let asset = accepted
            .assets
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| format!("formula archive {name} is absent from the accepted release"))?;
        verified_asset_bytes(asset)?;
        let sha_index = index + 1;
        let (sha_start, sha_end) = lines
            .get(sha_index)
            .and_then(|line| string_argument(line, "sha256"))
            .ok_or("formula URL must have an adjacent sha256 declaration")?;
        if old == version && lines[sha_index][sha_start..sha_end] != asset.sha256 {
            return Err("tap already contains this version with different archive bytes".into());
        }
        observed_versions.insert(old_version.to_string());
        lines[index].replace_range(start..end, &format!("{prefix}{}/{name}", accepted.version));
        lines[sha_index].replace_range(sha_start..sha_end, &asset.sha256);
        changed_urls += 1;
    }
    if changed_urls == 0 {
        return Err("formula contains no recognized release archive".into());
    }
    for line in &mut lines {
        if let Some((start, end)) = string_argument(line, "version") {
            if !observed_versions.contains(&line[start..end]) {
                return Err("formula version does not agree with its download URLs".into());
            }
            line.replace_range(start..end, &accepted.version);
        }
    }
    Ok(lines.concat())
}

#[cfg(test)]
#[path = "github_tests.rs"]
mod tests;
