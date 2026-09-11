use super::HfFileInfo;
use crate::source::cached_weights::{parse_safetensors_index, SAFETENSORS_SINGLE};
use ferrum_types::{FerrumError, Result};
use std::collections::BTreeSet;

pub(super) use crate::source::cached_weights::SAFETENSORS_INDEX;

/// Existing loaders prefer the canonical single file over a sharded index.
pub(super) fn authoritative_index(files: &[HfFileInfo]) -> Option<&HfFileInfo> {
    if files
        .iter()
        .any(|file| is_file(file) && file.path == SAFETENSORS_SINGLE)
    {
        return None;
    }
    files
        .iter()
        .find(|file| is_file(file) && file.path == SAFETENSORS_INDEX)
}

/// Preserve auxiliary model weights needed by non-text repository consumers.
pub(super) fn repository_files(files: &[HfFileInfo]) -> Vec<&HfFileInfo> {
    files
        .iter()
        .filter(|file| is_file(file) && legacy_selected_path(&file.path))
        .collect()
}

pub(super) fn selected_files<'a>(
    files: &'a [HfFileInfo],
    index: Option<&[u8]>,
) -> Result<Vec<&'a HfFileInfo>> {
    if authoritative_index(files).is_none() {
        return Ok(repository_files(files));
    }

    let index = index.ok_or_else(|| {
        FerrumError::model(format!(
            "{SAFETENSORS_INDEX} must be read before selecting sharded weights"
        ))
    })?;
    let shards = indexed_shards(files, index)?;
    let mut selected_paths = BTreeSet::new();
    Ok(files
        .iter()
        .filter(|file| {
            is_file(file)
                && (shards.contains(&file.path)
                    || (!is_weight_path(&file.path) && legacy_selected_path(&file.path)))
                && selected_paths.insert(file.path.as_str())
        })
        .collect())
}

fn indexed_shards(files: &[HfFileInfo], bytes: &[u8]) -> Result<BTreeSet<String>> {
    let shards = parse_safetensors_index(bytes)?;
    let available: BTreeSet<&str> = files
        .iter()
        .filter(|file| is_file(file))
        .map(|file| file.path.as_str())
        .collect();
    for path in &shards {
        if !available.contains(path.as_str()) {
            return Err(FerrumError::model(format!(
                "Shard {path:?} referenced by {SAFETENSORS_INDEX} is absent from the repository file inventory"
            )));
        }
    }
    Ok(shards)
}

fn is_file(file: &HfFileInfo) -> bool {
    file.file_type.as_deref() != Some("directory")
}

pub(super) fn is_weight_path(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
        || [".safetensors", ".pt", ".bin", ".onnx"]
            .iter()
            .any(|extension| path.ends_with(extension))
}

/// Preserve the original ordinary downloader's sidecars and no-index fallback.
fn legacy_selected_path(path: &str) -> bool {
    !path.to_ascii_lowercase().ends_with(".gguf")
        && !path.ends_with(".md")
        && !path.starts_with(".git")
        && (is_weight_path(path)
            || [
                ".json", ".yaml", ".yml", ".jinja", ".model", ".txt", ".png", ".wav",
            ]
            .iter()
            .any(|extension| path.ends_with(extension)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn file(path: &str) -> HfFileInfo {
        HfFileInfo {
            path: path.to_owned(),
            size: Some(1),
            file_type: Some("file".to_owned()),
        }
    }

    fn directory(path: &str) -> HfFileInfo {
        HfFileInfo {
            file_type: Some("directory".to_owned()),
            ..file(path)
        }
    }

    fn paths<'a>(selected: &[&'a HfFileInfo]) -> Vec<&'a str> {
        selected.iter().map(|file| file.path.as_str()).collect()
    }

    #[test]
    fn indexed_selection_keeps_exact_shards_and_existing_sidecars() {
        let shard = "part-01.safetensors";
        let nested = "weights/part-02.safetensors";
        let sidecars = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "tokenizer.model",
            "vocab.txt",
            "config.yaml",
            "config.yml",
            "image.png",
            "audio.wav",
        ];
        let mut files = vec![file(SAFETENSORS_INDEX), file(shard), file(nested)];
        files.extend(sidecars.iter().map(|path| file(path)));
        files.extend([
            file("unused.safetensors"),
            file("original/model.safetensors"),
            file("metal/model.bin"),
            file("alternative.pt"),
            file("alternative.onnx"),
            file("README.md"),
            file(".git/config.json"),
            directory("directory.json"),
        ]);
        let index = serde_json::to_vec(&json!({
            "metadata": {"total_size": 2},
            "weight_map": {
                "a": shard,
                "b": shard,
                "c": nested
            }
        }))
        .unwrap();

        let selected = selected_files(&files, Some(&index)).unwrap();
        let mut expected = vec![SAFETENSORS_INDEX, shard, nested];
        expected.extend(sidecars);
        assert_eq!(paths(&selected), expected);
        assert_eq!(selected.iter().filter(|file| file.path == shard).count(), 1);
    }

    #[test]
    fn index_rejects_malformed_or_incomplete_weight_maps() {
        let files = vec![file(SAFETENSORS_INDEX), file("part.safetensors")];
        for (label, index) in [
            ("malformed JSON", "{"),
            ("missing weight_map", "{}"),
            ("null weight_map", r#"{"weight_map":null}"#),
            ("array weight_map", r#"{"weight_map":[]}"#),
            ("empty weight_map", r#"{"weight_map":{}}"#),
            (
                "mixed string and nonstring entries",
                r#"{"weight_map":{"valid":"part.safetensors","invalid":7}}"#,
            ),
        ] {
            let error = selected_files(&files, Some(index.as_bytes())).unwrap_err();
            assert!(
                error.to_string().contains(SAFETENSORS_INDEX),
                "{label}: {error}"
            );
        }
    }

    #[test]
    fn index_rejects_nonportable_and_url_ambiguous_shard_paths() {
        for path in [
            "",
            "/part.safetensors",
            "../part.safetensors",
            "./part.safetensors",
            "weights/../part.safetensors",
            "weights//part.safetensors",
            "weights/./part.safetensors",
            "weights\\part.safetensors",
            "C:/part.safetensors",
            "C:part.safetensors",
            "//server/part.safetensors",
            "weights/%2e%2e/part.safetensors",
            "part?query.safetensors",
            "part#fragment.safetensors",
            "part\n.safetensors",
            "part\0.safetensors",
            "part.bin",
            "part.safetensors/",
        ] {
            // Presence in the inventory must not bypass path validation.
            let files = vec![file(SAFETENSORS_INDEX), file(path)];
            let index = serde_json::to_vec(&json!({"weight_map": {"weight": path}})).unwrap();
            let error = selected_files(&files, Some(&index)).unwrap_err();
            assert!(
                error.to_string().contains("Invalid shard path"),
                "path {path:?}: {error}"
            );
        }
    }

    #[test]
    fn index_requires_every_reference_to_name_an_inventory_file() {
        let index = br#"{"weight_map":{"weight":"part.safetensors"}}"#;
        for (label, files) in [
            ("missing", vec![file(SAFETENSORS_INDEX)]),
            (
                "directory",
                vec![file(SAFETENSORS_INDEX), directory("part.safetensors")],
            ),
            (
                "different case",
                vec![file(SAFETENSORS_INDEX), file("Part.safetensors")],
            ),
        ] {
            let error = selected_files(&files, Some(index)).unwrap_err();
            assert!(
                error.to_string().contains("part.safetensors")
                    && error.to_string().contains("file inventory"),
                "{label}: {error}"
            );
        }
    }

    #[test]
    fn authoritative_index_requires_bytes_before_weight_selection() {
        let files = vec![file(SAFETENSORS_INDEX), file("part.safetensors")];
        let error = selected_files(&files, None).unwrap_err();
        assert!(error.to_string().contains("must be read"), "{error}");
    }

    #[test]
    fn no_index_and_dual_authority_preserve_existing_fallback() {
        for (label, mut files) in [
            ("no index", vec![]),
            (
                "nested index",
                vec![file("alternate/model.safetensors.index.json")],
            ),
            ("index is a directory", vec![directory(SAFETENSORS_INDEX)]),
            (
                "canonical single plus index",
                vec![file(SAFETENSORS_SINGLE), file(SAFETENSORS_INDEX)],
            ),
        ] {
            files.extend([
                file("custom.safetensors"),
                file("pytorch_model.bin"),
                file("alternative.pt"),
                file("alternative.onnx"),
                file("chat_template.jinja"),
                file("config.json"),
                file("README.md"),
                file("alternative.gguf"),
                file(".git/config.json"),
                directory("folder.json"),
            ]);
            assert!(authoritative_index(&files).is_none(), "{label}");
            let expected: Vec<&str> = files
                .iter()
                .filter(|file| {
                    file.file_type.as_deref() != Some("directory")
                        && !["README.md", "alternative.gguf", ".git/config.json"]
                            .contains(&file.path.as_str())
                })
                .map(|file| file.path.as_str())
                .collect();
            assert_eq!(
                paths(&selected_files(&files, None).unwrap()),
                expected,
                "{label}"
            );
            // Unselected index contents cannot reverse single-file priority.
            assert_eq!(
                paths(&selected_files(&files, Some(b"invalid unused index")).unwrap()),
                expected,
                "{label}"
            );
        }
    }

    #[test]
    fn repository_policy_keeps_auxiliary_weights_beside_a_root_index() {
        let files = vec![
            file(SAFETENSORS_INDEX),
            file("part.safetensors"),
            file("speech_tokenizer/model.safetensors"),
            file("speech_tokenizer/config.json"),
            file("config.json"),
        ];
        assert!(authoritative_index(&files).is_some());
        assert_eq!(
            paths(&repository_files(&files)),
            [
                SAFETENSORS_INDEX,
                "part.safetensors",
                "speech_tokenizer/model.safetensors",
                "speech_tokenizer/config.json",
                "config.json",
            ]
        );
    }

    #[test]
    fn a_directory_named_like_the_single_file_does_not_hide_the_index() {
        let files = vec![
            directory(SAFETENSORS_SINGLE),
            file(SAFETENSORS_INDEX),
            file("part.safetensors"),
        ];
        assert_eq!(
            authoritative_index(&files).map(|file| file.path.as_str()),
            Some(SAFETENSORS_INDEX)
        );
        let selected = selected_files(
            &files,
            Some(br#"{"weight_map":{"weight":"part.safetensors"}}"#),
        )
        .unwrap();
        assert_eq!(paths(&selected), [SAFETENSORS_INDEX, "part.safetensors"]);
    }
}
