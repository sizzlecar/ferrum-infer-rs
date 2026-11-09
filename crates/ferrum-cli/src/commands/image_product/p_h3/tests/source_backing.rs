use super::*;
use ferrum_engine::image_generation::p_h3::PH3ImagePromptExpansionSourceBacking as Policy;

#[test]
fn expansion_source_backing_defaults_fresh_and_selects_explicit_installation_reuse() {
    let directory = tempfile::tempdir().unwrap();
    let wire = enabled_document();
    let config = load(directory.path(), &wire).unwrap();
    let PH3ImagePromptExpansionConfig::Enabled(expansion) = config.prompt_expansion else {
        panic!("enabled fixture")
    };
    assert_eq!(expansion.source_backing, Policy::Fresh);
    assert_eq!(expansion.source_conversion_workers.get(), 1);
    for (name, expected) in [
        ("fresh", Policy::Fresh),
        ("reusable_single_slot", Policy::ReusableSingleSlot),
        ("bounded_streaming", Policy::BoundedStreaming),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["source_backing"] = json!(name);
        let config = load(directory.path(), &wire).unwrap();
        let PH3ImagePromptExpansionConfig::Enabled(expansion) = config.prompt_expansion else {
            panic!("enabled fixture")
        };
        assert_eq!(expansion.source_backing, expected);
    }
    assert!(
        !directory.path().join("weights").exists(),
        "shared run/serve configuration is metadata-only"
    );
}

#[test]
fn expansion_source_conversion_workers_are_bounded_and_never_ignored() {
    let directory = tempfile::tempdir().unwrap();
    for invalid in [json!(0), json!(17), json!(-1), json!(null), json!(true)] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["source_backing"] = json!("bounded_streaming");
        wire["prompt_expansion"]["source_conversion_workers"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
    for workers in [1, 4, 16] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["source_backing"] = json!("bounded_streaming");
        wire["prompt_expansion"]["source_conversion_workers"] = json!(workers);
        let config = load(directory.path(), &wire).unwrap();
        let PH3ImagePromptExpansionConfig::Enabled(expansion) = config.prompt_expansion else {
            panic!("enabled fixture")
        };
        assert_eq!(expansion.source_conversion_workers.get(), workers);
    }
    for backing in ["fresh", "reusable_single_slot"] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["source_backing"] = json!(backing);
        wire["prompt_expansion"]["source_conversion_workers"] = json!(1);
        load(directory.path(), &wire).unwrap();
        wire["prompt_expansion"]["source_conversion_workers"] = json!(4);
        let error = load(directory.path(), &wire).err().unwrap();
        assert!(format!("{error:#}").contains("requires bounded_streaming"));
    }
    let mut wire = enabled_document();
    wire["prompt_expansion"]["weight_cache"] = json!({
        "kind":"process_mapped", "directory":"normalized",
        "maximum_file_bytes":4096,"maximum_metadata_bytes":1024,
    });
    wire["prompt_expansion"]["source_conversion_workers"] = json!(1);
    load(directory.path(), &wire).unwrap();
    wire["prompt_expansion"]["source_conversion_workers"] = json!(4);
    assert!(load(directory.path(), &wire).is_err());
    wire["prompt_expansion"]["source_backing"] = json!("bounded_streaming");
    assert!(load(directory.path(), &wire).is_err());
    let mut disabled = document();
    disabled["prompt_expansion"]["source_conversion_workers"] = json!(1);
    assert!(load(directory.path(), &disabled).is_err());
    assert!(!directory.path().join("normalized").exists());
}

#[test]
fn expansion_source_backing_rejects_unknown_disabled_and_process_mapped_combinations() {
    let directory = tempfile::tempdir().unwrap();
    for invalid in [
        json!("automatic"),
        json!({"kind":"reusable_single_slot"}),
        json!(null),
        json!(1),
    ] {
        let mut wire = enabled_document();
        wire["prompt_expansion"]["source_backing"] = invalid;
        assert!(load(directory.path(), &wire).is_err());
    }
    for policy in ["fresh", "reusable_single_slot", "bounded_streaming"] {
        let mut wire = document();
        wire["prompt_expansion"]["source_backing"] = json!(policy);
        assert!(
            load(directory.path(), &wire).is_err(),
            "disabled expansion cannot discard an explicit source policy"
        );
    }
    let mut wire = enabled_document();
    wire["prompt_expansion"]["weight_cache"] = json!({
        "kind":"process_mapped", "directory":"normalized",
        "maximum_file_bytes":4096,"maximum_metadata_bytes":1024,
    });
    wire["prompt_expansion"]["source_backing"] = json!("fresh");
    load(directory.path(), &wire).unwrap();
    for policy in ["reusable_single_slot", "bounded_streaming"] {
        wire["prompt_expansion"]["source_backing"] = json!(policy);
        let error = load(directory.path(), &wire).err().unwrap();
        assert!(format!("{error:#}").contains("cannot be combined with process-mapped"));
    }
    assert!(!directory.path().join("normalized").exists());
}
