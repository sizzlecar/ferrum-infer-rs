//! Check immutable HF source selection from the product's own resolution evidence.

use serde_json::Value;
use std::collections::BTreeSet;

pub fn pinned_hf_source(model: &str) -> Result<Option<(&str, &str)>, String> {
    let Some((repo, revision)) = model.rsplit_once('@') else {
        return Ok(None);
    };
    // Evidence can name a file on a worker running a different OS. Recognize
    // POSIX and Windows absolute paths without the replay host's path rules.
    if std::path::Path::new(model).is_absolute()
        || model.starts_with('/')
        || (model
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
            && model.as_bytes().get(1..3) == Some(b":/"))
        || model.starts_with("./")
        || model.starts_with("../")
        || model.contains('\\')
    {
        return Ok(None);
    }
    let parts: Vec<_> = repo.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
        || revision.len() != 40
        || !revision.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err("pinned HF model must be owner/repository@40-hex-commit".into());
    }
    Ok(Some((repo, revision)))
}

/// The snapshot must be the one used by the same process that generated the
/// model observations. A requested model label alone is not resolution evidence.
pub fn verify_pinned_source(model: &str, identity: &Value) -> Result<(), String> {
    let Some((repo, revision)) = pinned_hf_source(model)? else {
        return Ok(());
    };
    if identity["requested_model"] != model {
        return Err(
            "source evidence is missing or requested_model differs from the pinned model".into(),
        );
    }
    if identity["schema_version"] != 1 || identity["resolved_model"] != repo {
        return Err(
            "source identity schema or resolved public model differs from the requested source"
                .into(),
        );
    }
    for role in ["weights", "semantic", "tokenizer"] {
        let original = &identity["original_sources"][role];
        let resolved = &identity["resolved_sources"][role];
        if original["kind"] != "repository"
            || original["location"] != repo
            || original["requested_revision"]
                .as_str()
                .is_none_or(|value| !value.eq_ignore_ascii_case(revision))
            || resolved["canonical_location"] != repo
            || resolved["resolved_revision"]
                .as_str()
                .is_none_or(|value| !value.eq_ignore_ascii_case(revision))
        {
            return Err(format!(
                "{role} source does not resolve the requested repository and immutable revision"
            ));
        }
        let files = resolved["files"]
            .as_array()
            .filter(|files| !files.is_empty())
            .ok_or_else(|| format!("{role} source has no observed file fingerprints"))?;
        let mut names = BTreeSet::new();
        for file in files {
            let name = file["relative_path"]
                .as_str()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| format!("{role} source file is unnamed"))?;
            if !names.insert(name)
                || file["size_bytes"].as_u64().is_none_or(|size| size == 0)
                || file["sha256"].as_str().is_none_or(|digest| {
                    digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
            {
                return Err(format!(
                    "{role} source file fingerprint is missing, invalid, or duplicated"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_sources_require_an_explicit_repository_and_immutable_revision() {
        assert_eq!(pinned_hf_source("org/model").unwrap(), None);
        for local in [
            "/cache/model@revision",
            "C:/cache/model@revision",
            r"C:\cache\model@revision",
            r"\\worker\cache\model@revision",
            "./cache/model@revision",
            "../cache/model@revision",
        ] {
            assert_eq!(pinned_hf_source(local).unwrap(), None, "{local}");
        }
        let revision = "a".repeat(40);
        let model = format!("org/model@{revision}");
        assert_eq!(
            pinned_hf_source(&model).unwrap(),
            Some(("org/model", revision.as_str()))
        );
        for model in [
            "org/model@main",
            "alias@revision",
            "org/../model@revision",
            "org/model@",
        ] {
            assert!(pinned_hf_source(model).is_err(), "{model}");
        }
    }
}
