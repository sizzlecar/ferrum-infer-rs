use super::*;

#[test]
fn explicit_test_modules_and_comments_do_not_change_production_ast() {
    let before = r##"pub fn forward() -> &'static str { "#[cfg(test)] mod tests { fake }" }
        #[cfg(test)] mod tests { #[test] fn original() { assert!(true); } }"##;
    let after = r##"// A source comment, not output text.
        pub fn forward() -> &'static str { "#[cfg(test)] mod tests { fake }" }
        #[cfg(test)] mod tests { #[test] fn added_boundary() { assert_eq!(1 + 1, 2); } }"##;
    assert_eq!(rust_validation_only(before, after), Ok(true));
    assert_eq!(
        rust_validation_only(before, &after.replace("fake", "changed output")),
        Ok(false)
    );
    assert_eq!(
        rust_validation_only(before, &after.replace("pub fn forward", "fn forward")),
        Ok(false)
    );
    assert!(rust_validation_only(before, "not valid Rust {").is_err());
}

#[test]
fn complex_cfg_and_other_configuration_do_not_disappear() {
    for attribute in [
        "#[cfg(all(test, feature = \"extra\"))]",
        "#[cfg_attr(test, allow(dead_code))]",
        "#[cfg(feature = \"test\")]",
    ] {
        let before = format!("{attribute} mod conditional {{ fn value() -> u32 {{ 1 }} }}");
        let after = before.replace("{ 1 }", "{ 2 }");
        assert_eq!(
            rust_validation_only(&before, &after),
            Ok(false),
            "{attribute}"
        );
    }
    let before = "mod production { #[cfg(test)] mod tests { fn a() {} } }";
    let after = "mod production { #[cfg(test)] mod tests { fn b() {} } }";
    assert_eq!(
        rust_validation_only(before, after),
        Ok(false),
        "only reviewed top-level cfg(test) modules are removed"
    );
}

fn readme() -> &'static str {
    "# Product\n## Quick Start\n\nInstall Ferrum:\n\n```bash\nbrew tap owner/product\nbrew install ferrum\n```\n\n```bash\nferrum run stable-model\n```\n## Performance Snapshot\n| stable-model | 10 tok/s |\n## Installation\n\nHomebrew:\n\n```bash\nbrew install ferrum\n```\n\nOther installation remains checked.\n"
}

#[test]
fn homebrew_steps_do_not_infer_changed_model_or_performance_promises() {
    let before = readme();
    let after = before
        .replace(
            "Install Ferrum:",
            "Install Ferrum:\n\nOn Homebrew 6, review and trust the formula.",
        )
        .replace(
            "brew install ferrum",
            "brew trust --formula owner/product/ferrum\nbrew install ferrum",
        );
    assert_eq!(homebrew_documentation_only(before, &after), Ok(true));
    for changed in [
        after.replace("stable-model", "new-model"),
        after.replace("10 tok/s", "20 tok/s"),
        after.replace(
            "Other installation remains checked.",
            "Different Cargo flags.",
        ),
    ] {
        assert_eq!(homebrew_documentation_only(before, &changed), Ok(false));
    }
    assert!(homebrew_documentation_only(
        before,
        &after.replace("brew install ferrum", "ferrum run unreviewed-model")
    )
    .is_err());
    assert!(homebrew_documentation_only(
        before,
        &after.replace(
            "brew install ferrum",
            "brew install ferrum; ferrum run another-model"
        )
    )
    .is_err());
    let chinese = before
        .replace("## Quick Start", "## 快速开始")
        .replace("Install Ferrum:", "安装 Ferrum：")
        .replace("## Installation", "## 安装");
    assert_eq!(
        homebrew_documentation_only(
            &chinese,
            &chinese.replace(
                "brew install ferrum",
                "brew trust --formula owner/product/ferrum\nbrew install ferrum"
            )
        ),
        Ok(true)
    );
}

#[test]
fn reviewed_release_module_additions_preserve_every_observability_item() {
    let before = "pub mod stats; pub fn observe(value: u64) -> u64 { value }";
    let after = format!("{before}\npub mod release_regression;\n/// Candidate metadata.\npub mod release_candidate;");
    assert_eq!(bench_release_exports_only(before, &after), Ok(true));
    assert_eq!(bench_release_exports_only(&after, before), Ok(false));
    for changed in [
        after.replace("{ value }", "{ value + 1 }"),
        after.replace("value: u64", "value: u32"),
        after.replace(
            "pub mod release_regression;",
            "#[cfg(feature = \"extra\")] pub mod release_regression;",
        ),
        after.replace(
            "pub mod release_regression;",
            "pub mod release_regression { pub fn execute() {} }",
        ),
        format!("{after} pub mod other;"),
        format!("{after} pub mod release_regression;"),
    ] {
        assert_eq!(bench_release_exports_only(before, &changed), Ok(false));
    }
    let one = format!("{before}\n/// Candidate metadata.\npub mod release_candidate;");
    assert_eq!(bench_release_exports_only(&one, &after), Ok(true));
    assert_eq!(bench_release_exports_only(before, before), Ok(false));
}
