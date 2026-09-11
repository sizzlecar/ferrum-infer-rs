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
    crate::source::gguf_selection::validate_standalone_gguf(&first.path)?;
    Ok(first)
}

pub(super) fn automatic_file<'a>(
    files: &'a [HfFileInfo],
    selected: &[&HfFileInfo],
) -> Result<Option<&'a HfFileInfo>> {
    if selected
        .iter()
        .any(|file| selection::is_weight_path(&file.path))
    {
        return Ok(None);
    }
    let filename = crate::source::gguf_selection::select_gguf_file(
        files
            .iter()
            .filter(|file| file.file_type.as_deref() != Some("directory"))
            .map(|file| file.path.as_str()),
        None,
    )?;
    filename
        .map(|filename| selected_file(files, filename))
        .transpose()
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
    fn gguf_selection_preserves_existing_repository_weights_and_sidecars() {
        let mut files = vec![file("model.gguf"), file("config.json")];
        assert_eq!(
            automatic_file(&files, &[&files[1]]).unwrap().unwrap().path,
            "model.gguf"
        );
        files.push(file("model.safetensors"));
        assert!(automatic_file(&files, &[&files[1], &files[2]])
            .unwrap()
            .is_none());
        assert!(automatic_file(&[], &[]).unwrap().is_none());
    }
}
