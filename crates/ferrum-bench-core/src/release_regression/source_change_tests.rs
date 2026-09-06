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

fn metal_submission_fixture() -> (String, String) {
    let before = r#"
        pub mod vnext_runtime;
        pub struct MetalBackend;
        struct MetalContext { cmd: Option<Command> }
        impl MetalContext {
            fn encoder(&mut self) { open_encoder(); }
            pub(crate) fn flush(&mut self) { commit_and_wait(); }
        }
        impl Backend for MetalBackend {
            fn gemm() { shader_gemm(); }
        }
        fn shared_queue() { create_queue(); }
    "#;
    let after = before.replace(
        "pub(crate) fn flush(&mut self) { commit_and_wait(); }",
        r#"
        /// Submit the pending legacy command buffer.
        fn submit_and_wait(&mut self) -> Option<&'static metal::CommandBufferRef> { submit(); None }
        pub(crate) fn flush(&mut self) { self.submit_and_wait(); }
        fn flush_checked(&mut self) -> Result<()> { self.submit_and_wait(); Ok(()) }
    "#,
    ) + r#"
        impl MetalBackend {
            pub fn sync_checked(ctx: &mut MetalContext) -> Result<()> { ctx.flush_checked() }
        }
        #[allow(unexpected_cfgs)]
        fn command_buffer_error(cmd: &metal::CommandBufferRef) -> (Option<i64>, Option<String>) { (None, None) }
        fn validate_command_buffer_completion(status: metal::MTLCommandBufferStatus, code: Option<i64>, detail: Option<&str>) -> Result<()> { Ok(()) }
        #[cfg(test)] mod checked_sync_tests { #[test] fn missing_submit_fails() {} }
    "#;
    (before.into(), after)
}

#[test]
fn legacy_submission_proof_keeps_shared_state_and_operator_implementations_intact() {
    let (before, after) = metal_submission_fixture();
    assert_eq!(legacy_metal_submission_only(&before, &after), Ok(true));
    for changed in [
        after.replace("shader_gemm();", "changed_precision();"),
        after.replace("create_queue();", "different_queue();"),
        after.replace("open_encoder();", "different_encoder();"),
        after.replace("cmd: Option<Command>", "cmd: SharedCommand"),
        after.replace(
            "pub mod vnext_runtime;",
            "#[path = \"legacy.rs\"] pub mod vnext_runtime;",
        ),
        after.replace(
            "impl MetalContext {",
            "impl MetalContext { fn other_state_change() {}",
        ),
        after.replace(
            "fn flush_checked(&mut self)",
            "pub fn flush_checked(&mut self)",
        ),
        after.replace(
            "fn flush_checked(&mut self)",
            "#[cfg(feature = \"unchecked\")] fn flush_checked(&mut self)",
        ),
        after.replace(
            "pub(crate) fn flush(&mut self)",
            "pub(crate) fn renamed_flush(&mut self)",
        ),
        after.replace(
            "fn command_buffer_error(cmd:",
            "pub fn command_buffer_error(cmd:",
        ),
    ] {
        assert_eq!(
            legacy_metal_submission_only(&before, &changed),
            Ok(false),
            "{changed}"
        );
    }
    assert!(legacy_metal_submission_only(&before, "invalid Rust {").is_err());
}

#[test]
fn legacy_submission_proof_cannot_ignore_new_traits_or_conditional_implementations() {
    let (before, after) = metal_submission_fixture();
    for changed in [
        after.replace("impl MetalBackend {", "impl Runtime for MetalBackend {"),
        after.replace(
            "impl MetalBackend {",
            "#[cfg(feature = \"alternative\")] impl MetalBackend {",
        ),
        after.replace("pub fn sync_checked", "pub unsafe fn sync_checked"),
        after.clone() + "fn shared_state_change() {}",
    ] {
        assert_eq!(legacy_metal_submission_only(&before, &changed), Ok(false));
    }
    assert_eq!(
        legacy_metal_submission_only("fn other() {}", "fn other() {}"),
        Ok(false)
    );
}

fn ready_observation_fixture() -> (String, String, String) {
    let before = r#"
        fn emit_jsonl_ready(session_id: &str, requested_model: &str, resolved_model: &str, backend: &str) {
            let record = serde_json::json!({
                "event": "ready", "session_id": session_id,
                "requested_model": requested_model, "resolved_model": resolved_model,
                "backend": backend,
            });
            emit_jsonl_record(&record);
        }
        fn execute(one_shot: bool) {
            let model_chat_template = match one_shot { true => Some(load_template()), false => None };
            configure_runtime();
            if one_shot {
                emit_jsonl_ready(&session, &requested, &resolved, &backend);
                generate();
            } else {
                emit_jsonl_ready(&session, &requested, &resolved, &backend);
                interactive();
            }
        }
    "#;
    let after = before
        .replace("backend: &str)", "backend: &str, template: Option<&ModelChatTemplate>,)")
        .replace("\"backend\": backend,", "\"backend\": backend, \"reasoning_protocol\": template.map(ModelChatTemplate::reasoning_capability).unwrap_or_default(),")
        .replace("&resolved, &backend);", "&resolved, &backend, model_chat_template.as_ref(),);");
    let getter = r#"
        use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
        pub struct ModelChatTemplate {
            pub output_protocol: ModelOutputProtocol,
            pub reasoning_protocol: ModelReasoningProtocol,
        }
        impl ModelChatTemplate {
            /// Read the declared scalar capability.
            pub fn reasoning_capability(&self) -> ModelReasoningProtocol {
                if self.output_protocol == ModelOutputProtocol::HarmonyGptOss {
                    ModelReasoningProtocol::ModelGenerated
                } else { self.reasoning_protocol }
            }
        }
    "#;
    (before.into(), after, getter.into())
}

#[test]
fn ready_observation_preserves_the_existing_wire_record_and_all_caller_control_flow() {
    let (before, after, getter) = ready_observation_fixture();
    assert_eq!(
        run_ready_capability_only(&before, &after, &getter),
        Ok(true)
    );
    let with_test = format!("{after} #[cfg(test)] mod tests {{ #[test] fn new_test() {{}} }}");
    assert_eq!(
        run_ready_capability_only(&before, &with_test, &getter),
        Ok(true)
    );
    // The number of calls is not a fixed matrix. Both trees must preserve their
    // actual calls and positions, even for a pre-existing additional path.
    let old_extra = before.replace(
        "interactive();",
        "interactive(); emit_jsonl_ready(&session, &requested, &resolved, &backend);",
    );
    let new_extra = after.replace("interactive();", "interactive(); emit_jsonl_ready(&session, &requested, &resolved, &backend, model_chat_template.as_ref());");
    assert_eq!(
        run_ready_capability_only(&old_extra, &new_extra, &getter),
        Ok(true)
    );
    for changed in [
        after.replace("\"event\": \"ready\"", "\"event\": \"done\""),
        after.replace("\"backend\": backend", "\"backend\": resolved_model"),
        after.replace("&session, &requested", "&requested, &session"),
        after.replace("configure_runtime();", "configure_different_kv();"),
        after.replace("if one_shot {", "if !one_shot {"),
        after.replace("generate();", "emit_jsonl_ready(&session, &requested, &resolved, &backend, model_chat_template.as_ref()); generate();"),
        after.replace("emit_jsonl_record(&record);", "mutate_engine(); emit_jsonl_record(&record);"),
        after.replace("emit_jsonl_record(&record);", "other_sink(&record);"),
        after.replace("&resolved, &backend, model_chat_template.as_ref(),);", "&resolved, &backend, model_chat_template.as_ref(),); changed();"),
    ] {
        assert_eq!(run_ready_capability_only(&before, &changed, &getter), Ok(false), "{changed}");
    }
}

#[test]
fn ready_observation_rejects_unproven_borrowing_and_hidden_record_effects() {
    let (before, after, getter) = ready_observation_fixture();
    for changed in [
        after.replace(
            "model_chat_template.as_ref()",
            "model_chat_template.as_mut()",
        ),
        after.replace(
            "model_chat_template.as_ref()",
            "model_chat_template.take().as_ref()",
        ),
        after.replace("model_chat_template.as_ref()", "other_template.as_ref()"),
        after.replace(
            "model_chat_template.as_ref()",
            "{ mutate_engine(); model_chat_template.as_ref() }",
        ),
        after.replace(
            "template.map(ModelChatTemplate::reasoning_capability)",
            "template.map(ModelChatTemplate::change_configuration)",
        ),
        after.replace(
            "unwrap_or_default()",
            "unwrap_or_else(change_configuration)",
        ),
        after.replace(
            "Option<&ModelChatTemplate>",
            "Option<&mut ModelChatTemplate>",
        ),
        after.replace(
            "\"reasoning_protocol\": template",
            "\"replacement_field\": template",
        ),
    ] {
        assert_eq!(
            run_ready_capability_only(&before, &changed, &getter),
            Ok(false),
            "{changed}"
        );
    }
    for (old, new) in [
        (
            before.replace("Some(load_template())", "custom_wrapper()"),
            after.replace("Some(load_template())", "custom_wrapper()"),
        ),
        (
            before.replace(
                "configure_runtime();",
                "configure_runtime(); let model_chat_template = other();",
            ),
            after.replace(
                "configure_runtime();",
                "configure_runtime(); let model_chat_template = other();",
            ),
        ),
        (
            before.replace("\"backend\": backend", "\"backend\": template"),
            after.replace("\"backend\": backend", "\"backend\": template"),
        ),
        (
            before.replace(
                "\"backend\": backend",
                "\"backend\": macro_value!(template)",
            ),
            after.replace(
                "\"backend\": backend",
                "\"backend\": macro_value!(template)",
            ),
        ),
    ] {
        assert_eq!(run_ready_capability_only(&old, &new, &getter), Ok(false));
    }
}

#[test]
fn ready_observation_checks_the_actual_candidate_getter_and_its_scalar_field_types() {
    let (before, after, getter) = ready_observation_fixture();
    for changed in [
        getter.replace("if self.output_protocol", "mutate_engine(); if self.output_protocol"),
        getter.replace("else { self.reasoning_protocol }", "else { self.update_state() }"),
        getter.replace("else { self.reasoning_protocol }", "else { mutate!(); self.reasoning_protocol }"),
        getter.replace("&self", "&mut self"),
        getter.replace("pub fn reasoning_capability", "#[cfg(feature = \"alternative\")] pub fn reasoning_capability"),
        getter.replace("pub fn reasoning_capability", "pub async fn reasoning_capability"),
        getter.replace("pub output_protocol: ModelOutputProtocol", "pub output_protocol: InteriorMutableProtocol"),
        getter.replace("use ferrum_types::", "use arbitrary_custom_types::"),
        getter.replace("impl ModelChatTemplate", "impl Unreviewed for ModelChatTemplate"),
        format!("{getter} impl ModelChatTemplate {{ pub fn reasoning_capability(&self) -> ModelReasoningProtocol {{ self.reasoning_protocol }} }}"),
        "struct ModelChatTemplate;".into(),
    ] {
        assert_eq!(run_ready_capability_only(&before, &after, &changed), Ok(false), "{changed}");
    }
    // Pure output changes still require protocol/capability checks. The proof
    // establishes execution reach, not that a reported capability is correct.
    let different_observation = getter.replace(
        "else { self.reasoning_protocol }",
        "else { ModelReasoningProtocol::None }",
    );
    assert_eq!(
        run_ready_capability_only(&before, &after, &different_observation),
        Ok(true)
    );
    assert!(run_ready_capability_only(&before, &after, "invalid Rust {").is_err());
    assert_eq!(
        run_ready_capability_only(&before, &before, &getter),
        Ok(false)
    );
}

#[test]
fn legacy_submission_signature_trailing_commas_do_not_hide_parameter_changes() {
    let (before, after) = metal_submission_fixture();
    let multiline = after
        .replace("detail: Option<&str>)", "detail: Option<&str>,)")
        .replace("ctx: &mut MetalContext)", "ctx: &mut MetalContext,)");
    assert_eq!(legacy_metal_submission_only(&before, &multiline), Ok(true));
    assert_eq!(
        legacy_metal_submission_only(
            &before,
            &multiline.replace("detail: Option<&str>,", "detail: Option<&mut str>,")
        ),
        Ok(false)
    );
    assert_eq!(
        legacy_metal_submission_only(
            &before,
            &multiline.replace("ctx: &mut MetalContext,", "ctx: &MetalContext,")
        ),
        Ok(false)
    );
}
