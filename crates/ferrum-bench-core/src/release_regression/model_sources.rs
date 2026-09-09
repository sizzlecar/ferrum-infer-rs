//! Check immutable HF source selection from the product's own resolution evidence.

use super::types::{GgufSourceProfile, ModelProfile};
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
    verify_identity(model, repo, identity)?;
    for role in ["weights", "semantic", "tokenizer"] {
        verify_repository_role(identity, role, repo, revision, true)?;
    }
    Ok(())
}

pub fn validate_gguf_selection(model: &str, filename: &str) -> Result<(), String> {
    ferrum_types::validate_gguf_filename(filename).map_err(|error| error.to_string())?;
    required_pin(model)?;
    Ok(())
}

fn required_pin(model: &str) -> Result<(&str, &str), String> {
    pinned_hf_source(model)?
        .ok_or_else(|| "GGUF regression sources require owner/repository@40-hex-commit".into())
}

pub fn validate_profile_sources(profile: &ModelProfile) -> Result<(), String> {
    if let Some(gguf) = &profile.gguf {
        validate_gguf_selection(&profile.model, &gguf.filename)?;
        required_pin(&gguf.semantic_source)?;
        required_pin(
            gguf.tokenizer_source
                .as_deref()
                .unwrap_or(&gguf.semantic_source),
        )?;
    }
    Ok(())
}

pub fn verify_profile_source(profile: &ModelProfile, identity: &Value) -> Result<(), String> {
    validate_profile_sources(profile)?;
    match &profile.gguf {
        Some(gguf) => {
            verify_pinned_gguf_source(&profile.model, &gguf.filename, Some(gguf), identity)
        }
        None => verify_pinned_source(&profile.model, identity),
    }
}

/// Standalone probes validate all observed source identities. Prepared tasks
/// additionally require the independently declared metadata repositories/revisions.
/// These expectations check product resolution; they never rewrite its sources.
pub fn verify_pinned_gguf_source(
    model: &str,
    filename: &str,
    expected: Option<&GgufSourceProfile>,
    identity: &Value,
) -> Result<(), String> {
    validate_gguf_selection(model, filename)?;
    let (repo, revision) = required_pin(model)?;
    verify_identity(model, repo, identity)?;
    verify_repository_role(identity, "weights", repo, revision, true)?;
    let files = identity["resolved_sources"]["weights"]["files"]
        .as_array()
        .unwrap();
    if files.len() != 1
        || files[0]["relative_path"]
            .as_str()
            .is_none_or(|name| !name.eq_ignore_ascii_case(filename))
    {
        return Err(
            "weights evidence does not identify the selected repository-relative GGUF file".into(),
        );
    }
    if expected.is_some_and(|expected| expected.filename != filename) {
        return Err("selected GGUF filename differs from the expected profile".into());
    }
    for role in ["semantic", "tokenizer"] {
        let expected_source = expected.map(|expected| {
            if role == "tokenizer" {
                expected
                    .tokenizer_source
                    .as_deref()
                    .unwrap_or(&expected.semantic_source)
            } else {
                &expected.semantic_source
            }
        });
        let observed = &identity["resolved_sources"][role];
        let observed_pin = format!(
            "{}@{}",
            observed["canonical_location"].as_str().unwrap_or(""),
            observed["resolved_revision"].as_str().unwrap_or("")
        );
        let (repo, revision) = required_pin(expected_source.unwrap_or(&observed_pin))?;
        verify_repository_role(identity, role, repo, revision, false)?;
    }
    Ok(())
}

fn verify_identity(model: &str, repo: &str, identity: &Value) -> Result<(), String> {
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
    Ok(())
}

fn verify_repository_role(
    identity: &Value,
    role: &str,
    repo: &str,
    revision: &str,
    require_requested_revision: bool,
) -> Result<(), String> {
    let original = &identity["original_sources"][role];
    let resolved = &identity["resolved_sources"][role];
    if original["kind"] != "repository"
        || original["location"] != repo
        || ((require_requested_revision || !original["requested_revision"].is_null())
            && original["requested_revision"]
                .as_str()
                .is_none_or(|value| !value.eq_ignore_ascii_case(revision)))
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gguf_identity() -> (String, GgufSourceProfile, Value) {
        let model = format!("quantizer/weights@{}", "a".repeat(40));
        let expected = GgufSourceProfile {
            filename: "quantizations/model.gguf".into(),
            semantic_source: format!("author/semantic@{}", "b".repeat(40)),
            tokenizer_source: Some(format!("author/tokenizer@{}", "c".repeat(40))),
        };
        let mut identity = json!({"schema_version":1,"requested_model":model,"resolved_model":"quantizer/weights",
            "original_sources":{},"resolved_sources":{}});
        for (role, source, filename) in [
            ("weights", model.as_str(), expected.filename.as_str()),
            ("semantic", expected.semantic_source.as_str(), "config.json"),
            (
                "tokenizer",
                expected.tokenizer_source.as_deref().unwrap(),
                "tokenizer.json",
            ),
        ] {
            let (repo, revision) = required_pin(source).unwrap();
            identity["original_sources"][role] = json!({"kind":"repository","location":repo,
                "requested_revision":if role=="weights"{Some(revision)}else{None}});
            identity["resolved_sources"][role] = json!({"canonical_location":repo,"resolved_revision":revision,
                "files":[{"relative_path":filename,"size_bytes":16,"sha256":"d".repeat(64)}]});
        }
        (model, expected, identity)
    }

    #[test]
    fn gguf_checks_independent_metadata_pins_and_repository_relative_artifact() {
        let (model, expected, identity) = gguf_identity();
        verify_pinned_gguf_source(&model, &expected.filename, Some(&expected), &identity).unwrap();
        verify_pinned_gguf_source(&model, &expected.filename, None, &identity).unwrap();
        assert!(verify_pinned_source(&model, &identity).is_err());
        for (parent, role, field, value) in [
            (
                "resolved_sources",
                "weights",
                "resolved_revision",
                json!("e".repeat(40)),
            ),
            (
                "original_sources",
                "weights",
                "requested_revision",
                Value::Null,
            ),
            (
                "resolved_sources",
                "semantic",
                "resolved_revision",
                json!("e".repeat(40)),
            ),
            (
                "resolved_sources",
                "tokenizer",
                "canonical_location",
                json!("other/tokenizer"),
            ),
            (
                "original_sources",
                "semantic",
                "location",
                json!("other/semantic"),
            ),
            (
                "original_sources",
                "tokenizer",
                "requested_revision",
                json!("main"),
            ),
            (
                "resolved_sources",
                "weights",
                "files",
                json!([{"relative_path":"model.gguf","size_bytes":16,"sha256":"d".repeat(64)}]),
            ),
            (
                "resolved_sources",
                "weights",
                "files",
                json!([{"relative_path":"other/model.gguf","size_bytes":16,"sha256":"d".repeat(64)}]),
            ),
        ] {
            let mut changed = identity.clone();
            changed[parent][role][field] = value;
            assert!(
                verify_pinned_gguf_source(&model, &expected.filename, Some(&expected), &changed)
                    .is_err(),
                "{parent}/{role}/{field}"
            );
        }
        for role in ["weights", "semantic", "tokenizer"] {
            for files in [
                json!([]),
                json!([{"relative_path":"file","size_bytes":16}]),
                json!([{"relative_path":"file","size_bytes":0,"sha256":"d".repeat(64)}]),
            ] {
                let mut changed = identity.clone();
                changed["resolved_sources"][role]["files"] = files;
                assert!(
                    verify_pinned_gguf_source(&model, &expected.filename, None, &changed).is_err()
                );
            }
        }
    }

    #[test]
    fn shared_metadata_default_and_standalone_observation_do_not_change_expected_pins() {
        let (model, mut expected, mut identity) = gguf_identity();
        expected.tokenizer_source = None;
        identity["original_sources"]["tokenizer"] =
            identity["original_sources"]["semantic"].clone();
        identity["resolved_sources"]["tokenizer"]["canonical_location"] = json!("author/semantic");
        identity["resolved_sources"]["tokenizer"]["resolved_revision"] = json!("b".repeat(40));
        verify_pinned_gguf_source(&model, &expected.filename, Some(&expected), &identity).unwrap();
        identity["resolved_sources"]["semantic"]["resolved_revision"] = json!("e".repeat(40));
        verify_pinned_gguf_source(&model, &expected.filename, None, &identity).unwrap();
        assert!(
            verify_pinned_gguf_source(&model, &expected.filename, Some(&expected), &identity)
                .is_err()
        );
        for bad_model in ["author/model", "author/model@main", "/cache/model@revision"] {
            assert!(validate_gguf_selection(bad_model, "model.gguf").is_err());
        }
        for filename in ["../model.gguf", "model.safetensors", "/model.gguf"] {
            assert!(validate_gguf_selection(&model, filename).is_err());
        }
    }

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
