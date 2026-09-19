//! Explicit, versioned source recipes for models with independent metadata.
//!
//! Only registered aliases select a recipe. Repository names, architectures,
//! and local filenames do not implicitly activate these choices.

use super::ProductSourceArgs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartupProfile {
    Bonsai2Metal,
}

#[derive(Debug)]
pub(crate) struct ModelRecipe {
    pub requested_model: &'static str,
    pub gguf_file: &'static str,
    pub semantic_source: &'static str,
    pub startup_profile: StartupProfile,
}

impl ModelRecipe {
    /// User-selected roles take precedence; an explicit semantic source also
    /// supplies the tokenizer unless the user independently overrides it.
    pub(super) fn source_args(&self, explicit: &ProductSourceArgs) -> ProductSourceArgs {
        ProductSourceArgs {
            gguf_file: explicit
                .gguf_file
                .clone()
                .or_else(|| Some(self.gguf_file.to_owned())),
            semantic_source: explicit
                .semantic_source
                .clone()
                .or_else(|| Some(self.semantic_source.into())),
            tokenizer_source: explicit.tokenizer_source.clone(),
        }
    }
}

static BONSAI2_27B: ModelRecipe = ModelRecipe {
    requested_model: "prism-ml/Ternary-Bonsai-2-27B-gguf@6ed5e12bf84b7a63069882c91dd9e9218647d17b",
    gguf_file: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    semantic_source: "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    startup_profile: StartupProfile::Bonsai2Metal,
};

pub(crate) fn find(name: &str) -> Option<&'static ModelRecipe> {
    ["bonsai2:27b", "bonsai2:27b-pq2_0"]
        .iter()
        .any(|alias| name.eq_ignore_ascii_case(alias))
        .then_some(&BONSAI2_27B)
}

#[cfg(test)]
mod tests;
