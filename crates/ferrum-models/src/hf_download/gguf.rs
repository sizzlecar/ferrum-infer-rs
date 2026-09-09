use super::{selection, FerrumError, HfFileInfo, Result};

pub use ferrum_types::validate_gguf_filename;

pub(super) fn selected_file<'a>(files: &'a [HfFileInfo], filename: &str) -> Result<&'a HfFileInfo> {
    let mut selected = files.iter().filter(|file| {
        file.file_type.as_deref() != Some("directory") && file.path.eq_ignore_ascii_case(filename)
    });
    let first = selected.next().ok_or_else(|| {
        FerrumError::model(format!("GGUF file '{filename}' not found in repository"))
    })?;
    if selected.next().is_some() {
        return Err(FerrumError::model(format!(
            "GGUF filename '{filename}' is ambiguous under case-insensitive matching"
        )));
    }
    validate_gguf_filename(&first.path)?;
    Ok(first)
}

pub(super) fn require_weight_selection(
    files: &[HfFileInfo],
    selected: &[&HfFileInfo],
) -> Result<()> {
    let has_gguf = files.iter().any(|file| {
        file.file_type.as_deref() != Some("directory")
            && file.path.to_ascii_lowercase().ends_with(".gguf")
    });
    if has_gguf
        && !selected
            .iter()
            .any(|file| selection::is_weight_path(&file.path))
    {
        return Err(FerrumError::model(
            "This repository distributes GGUF weights; select one exact file with --gguf-file FILE when using ferrum pull, run or serve",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> HfFileInfo {
        HfFileInfo {
            path: path.into(),
            size: Some(10),
            file_type: Some("file".into()),
        }
    }

    #[test]
    fn exact_file_selection_rejects_ambiguous_names_and_directories() {
        let mut files = vec![
            file("weights/model-Q4_K_M.gguf"),
            file("weights/model-Q8_0.gguf"),
        ];
        assert_eq!(
            selected_file(&files, "weights/model-q4_k_m.gguf")
                .unwrap()
                .path,
            files[0].path
        );
        assert!(selected_file(&files, "missing.gguf").is_err());
        files.push(file("weights/model-q4_k_m.gguf"));
        assert!(selected_file(&files, "weights/model-q4_k_m.gguf").is_err());
        files[2].file_type = Some("directory".into());
        assert!(selected_file(&files, "weights/model-q4_k_m.gguf").is_ok());
    }

    #[test]
    fn only_portable_repository_paths_can_select_a_gguf() {
        for path in ["model.gguf", "weights/model-Q4_K_M.GGUF", "模型/权重.gguf"] {
            validate_gguf_filename(path).unwrap();
        }
        for path in [
            "",
            "../model.gguf",
            "/model.gguf",
            "C:/model.gguf",
            "weights\\model.gguf",
            "weights//model.gguf",
            "./model.gguf",
            "a/../model.gguf",
            "model.safetensors",
            "model%20.gguf",
            "model?.gguf",
            "model\n.gguf",
            "weights./model.gguf",
            "weights /model.gguf",
        ] {
            assert!(validate_gguf_filename(path).is_err(), "{path:?}");
        }
    }

    #[test]
    fn gguf_only_repository_requires_a_file_even_when_sidecars_exist() {
        let mut files = vec![file("model.gguf"), file("config.json")];
        assert!(require_weight_selection(&files, &[&files[1]]).is_err());
        files.push(file("model.safetensors"));
        assert!(require_weight_selection(&files, &[&files[1], &files[2]]).is_ok());
        assert!(require_weight_selection(&[], &[]).is_ok());
    }
}
