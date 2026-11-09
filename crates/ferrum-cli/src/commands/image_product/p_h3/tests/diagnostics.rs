use super::*;
use ferrum_engine::image_generation::p_h3::{
    PH3ImagePromptExpansionDiagnosticPolicy, PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES,
};

#[test]
fn prompt_expansion_diagnostics_default_disabled_keeps_raw_and_untraced_documents_valid() {
    let directory = tempfile::tempdir().unwrap();
    let original = load(directory.path(), &document()).unwrap();
    assert_eq!(
        original.prompt_expansion_diagnostics,
        PH3ImagePromptExpansionDiagnosticPolicy::Disabled
    );
    let mut wire = document();
    wire["prompt_expansion_diagnostics"] = json!({"mode":"disabled"});
    let explicit = load(directory.path(), &wire).unwrap();
    assert_eq!(
        explicit.prompt_expansion_diagnostics,
        PH3ImagePromptExpansionDiagnosticPolicy::Disabled
    );
    assert!(!explicit.prompt_expansion.is_enabled());
    assert!(explicit.continuous_trace.is_none());
}

#[test]
fn prompt_expansion_diagnostics_require_enabled_expansion_and_the_original_bounded_trace() {
    let directory = tempfile::tempdir().unwrap();
    let policies = [
        (
            json!({"mode":"summary"}),
            PH3ImagePromptExpansionDiagnosticPolicy::Summary,
        ),
        (
            json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":2048}),
            PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
                maximum_utf8_bytes: 2048,
            },
        ),
    ];
    for (policy, expected) in policies {
        let mut wire = enabled_document();
        wire["prompt_expansion_diagnostics"] = policy.clone();
        let error = load(directory.path(), &wire).err().unwrap();
        assert!(format!("{error:#}").contains("require continuous_trace"));
        wire["continuous_trace"] = json!({"path":"observations.jsonl","maximum_bytes":4096});
        for alias in ["public-image-alias", "another-public-alias"] {
            wire["model"] = json!(alias);
            let config = load(directory.path(), &wire).unwrap();
            assert_eq!(config.prompt_expansion_diagnostics, expected);
            assert!(!config.continuous_trace.unwrap().path.exists());
        }
        let mut raw = document();
        raw["prompt_expansion_diagnostics"] = policy;
        raw["continuous_trace"] = wire["continuous_trace"].clone();
        let error = load(directory.path(), &raw).err().unwrap();
        assert!(format!("{error:#}").contains("require enabled native prompt expansion"));
    }
    assert!(!directory.path().join("observations.jsonl").exists());
    assert!(!directory.path().join("weights").exists());
}

#[test]
fn prompt_expansion_diagnostics_reject_invalid_capture_limits_and_unknown_controls_before_loading()
{
    let directory = tempfile::tempdir().unwrap();
    let mut wire = enabled_document();
    wire["continuous_trace"] = json!({"path":"observations.jsonl","maximum_bytes":4096});
    for invalid in [
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":0}),
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES+1}),
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":4097}),
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":-1}),
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":1.5}),
        json!({"mode":"capture_effective_prompt"}),
        json!({"mode":"capture_effective_prompt","maximum_utf8_bytes":2048,"thinking":true}),
        json!({"mode":"summary","maximum_utf8_bytes":2048}),
        json!({"mode":"disabled","unknown":true}),
        json!({"mode":"all_tokens"}),
    ] {
        let diagnostic = invalid.to_string();
        wire["prompt_expansion_diagnostics"] = invalid;
        assert!(
            load(directory.path(), &wire).is_err(),
            "accepted invalid policy: {diagnostic}"
        );
    }
    // Both endpoints are valid; these are capture-byte limits, not a claim that
    // a complete event or request history must fit the remaining trace space.
    for maximum_utf8_bytes in [1, 4096] {
        wire["prompt_expansion_diagnostics"] =
            json!({"mode":"capture_effective_prompt", "maximum_utf8_bytes":maximum_utf8_bytes});
        load(directory.path(), &wire).unwrap();
    }
    assert!(!directory.path().join("observations.jsonl").exists());
}
