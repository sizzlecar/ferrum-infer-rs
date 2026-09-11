//! Artifact selection shared by Hub downloads and local cache inspection.
//!
//! Quantization labels select published bytes, not runtime re-quantization.
use ferrum_types::{validate_gguf_filename, FerrumError, Result};

/// Select a standalone model artifact. The default prefers one Q4_K_M file,
/// then an otherwise unambiguous model. Never choose an arbitrary first file.
pub fn select_gguf_file<'a>(
    paths: impl IntoIterator<Item = &'a str>,
    quantization: Option<&str>,
) -> Result<Option<&'a str>> {
    if let Some(quant) = quantization {
        validate_quantization_label(quant)?;
    }
    let mut candidates: Vec<_> = paths
        .into_iter()
        .filter(|path| path.to_ascii_lowercase().ends_with(".gguf"))
        .filter(|path| {
            let name = path.rsplit('/').next().unwrap_or(path);
            !name.to_ascii_lowercase().starts_with("mmproj")
        })
        .collect();
    candidates.sort_unstable();
    if candidates.is_empty() && quantization.is_none() {
        return Ok(None);
    }
    // Validate before selection, including inventory names from remote sources.
    for path in &candidates {
        validate_gguf_filename(path)?;
    }
    let matching: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|path| matches_quantization(path, quantization.unwrap_or("Q4_K_M")))
        .collect();
    let selected = match matching.as_slice() {
        [only] => Some(*only),
        [] if quantization.is_none() && candidates.len() == 1 => Some(candidates[0]),
        _ => None,
    };
    if let Some(path) = selected {
        validate_standalone_gguf(path)?;
        return Ok(Some(path));
    }
    let request = quantization.map_or_else(
        || "No unambiguous default model artifact".to_owned(),
        |quant| format!("No unique model artifact for quantization '{quant}'"),
    );
    let available = candidates
        .iter()
        .map(|path| format!("  {path}"))
        .collect::<Vec<_>>()
        .join("\n");
    Err(FerrumError::model(format!(
        "{request}. Select MODEL:QUANT (for example owner/repo:Q8_0), or use --gguf-file FILE for an exact artifact / an explicit .gguf path. Available files:\n{available}"
    )))
}

pub fn validate_standalone_gguf(path: &str) -> Result<()> {
    validate_gguf_filename(path)?;
    if split_suffix(path).is_some() {
        return Err(FerrumError::unsupported(
            "This model uses split GGUF weights; Ferrum currently requires a standalone GGUF artifact. Select a repository variant distributed as one file.",
        ));
    }
    Ok(())
}

pub fn validate_quantization_label(quant: &str) -> Result<()> {
    if quant.is_empty()
        || !quant
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        || quant.split('-').any(str::is_empty)
    {
        return Err(FerrumError::config(
            "Quantization must be a label such as Q4_K_M, Q8_0 or BF16",
        ));
    }
    Ok(())
}

fn matches_quantization(path: &str, quant: &str) -> bool {
    let stem = &path[..path.len() - 5];
    let stem = split_suffix(path).unwrap_or(stem);
    let lower = stem.to_ascii_lowercase();
    let quant = quant.to_ascii_lowercase();
    lower.strip_suffix(&quant).is_some_and(|prefix| {
        // UD is a distinct published mixed-precision recipe, not the
        // standard Q4_K_M recipe with an interchangeable name.
        !prefix.ends_with("ud-") && (prefix.is_empty() || prefix.ends_with(['-', '.', '/', '_']))
    })
}

// The Hub's conventional multi-part filename is not a complete weight file.
fn split_suffix(path: &str) -> Option<&str> {
    let stem = path.strip_suffix(".gguf").or_else(|| {
        path.to_ascii_lowercase()
            .ends_with(".gguf")
            .then(|| &path[..path.len() - 5])
    })?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (base, part) = left.rsplit_once('-')?;
    (part.len() == 5
        && total.len() == 5
        && part.bytes().all(|byte| byte.is_ascii_digit())
        && total.bytes().all(|byte| byte.is_ascii_digit()))
    .then_some(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_labels_select_the_same_artifact_in_any_inventory_order() {
        let files = [
            "mmproj-F16.gguf",
            "weights/model-Q8_0.gguf",
            "weights/model-Q4_K_M.GGUF",
        ];
        for paths in [files.to_vec(), files.into_iter().rev().collect()] {
            assert_eq!(
                select_gguf_file(paths.iter().copied(), None).unwrap(),
                Some(files[2])
            );
            assert_eq!(
                select_gguf_file(paths.iter().copied(), Some("q4_k_m")).unwrap(),
                Some(files[2])
            );
            assert_eq!(
                select_gguf_file(paths.iter().copied(), Some("Q8_0")).unwrap(),
                Some(files[1])
            );
        }
        assert_eq!(
            select_gguf_file(["one.gguf", "mmproj.gguf"], None).unwrap(),
            Some("one.gguf")
        );
        assert_eq!(
            select_gguf_file(["config.json", "mmproj.gguf"], None).unwrap(),
            None
        );
        let recipes = ["model-UD-Q4_K_M.gguf", "model-Q4_K_M.gguf"];
        assert_eq!(select_gguf_file(recipes, None).unwrap(), Some(recipes[1]));
        assert_eq!(
            select_gguf_file(recipes, Some("ud-q4_k_m")).unwrap(),
            Some(recipes[0])
        );
    }

    #[test]
    fn ambiguity_missing_quantization_and_split_weights_do_not_pick_other_bytes() {
        for (paths, quant) in [
            (vec!["a.gguf", "b.gguf"], None),
            (vec!["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"], None),
            (vec!["model-IQ4_K_M.gguf"], Some("Q4_K_M")),
            (vec!["model-UD-Q4_K_M.gguf"], Some("Q4_K_M")),
            (vec!["model-Q8_0.gguf"], Some("Q4_K_M")),
            (vec!["model-Q4_K_M-00001-of-00002.gguf"], None),
            (
                vec![
                    "model-Q4_K_M-00001-of-00002.gguf",
                    "model-Q4_K_M-00002-of-00002.gguf",
                ],
                Some("Q4_K_M"),
            ),
            (vec!["../model.gguf"], None),
        ] {
            assert!(select_gguf_file(paths, quant).is_err());
        }
    }
}
