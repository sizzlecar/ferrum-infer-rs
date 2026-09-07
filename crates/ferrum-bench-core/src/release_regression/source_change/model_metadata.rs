//! Recognize immutable template-source wiring without erasing model execution.
use super::test_configuration;
use quote::ToTokens;
use std::collections::BTreeSet;
use syn::{visit_mut::VisitMut, Expr, Item, UseTree};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}
fn expression(source: &str) -> Expr {
    syn::parse_str(source).expect("fixed metadata scope expression")
}
fn same(value: &impl ToTokens, source: &str) -> bool {
    tokens(value) == tokens(&expression(source))
}

fn imports(tree: &UseTree, prefix: &str, output: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => imports(&path.tree, &format!("{prefix}{}::", path.ident), output),
        UseTree::Group(group) => {
            for tree in &group.items {
                imports(tree, prefix, output);
            }
        }
        leaf => {
            output.insert(format!("{prefix}{}", tokens(leaf)));
        }
    }
}

struct Wiring {
    calls: usize,
    valid: bool,
}
impl VisitMut for Wiring {
    fn visit_expr_mut(&mut self, value: &mut Expr) {
        if same(
            value,
            "config.metadata.template.source_file != \"tokenizer_config.json\"",
        ) || same(
            value,
            "!is_hf_template_source(&config.metadata.template.source_file)",
        ) {
            *value = expression("__selected_template_source_invalid");
            return;
        }
        syn::visit_mut::visit_expr_mut(self, value);
    }
    fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
        if same(&call.func, "parse_hf_model_semantic_metadata") {
            self.calls += 1;
            if call.args.len() == 4
                && same(&call.args[2], "sources.chat_template_jinja()")
                && same(&call.args[3], "sources.chat_template_json()")
            {
                call.args.pop();
                call.args.pop();
                call.args.pop_punct();
            } else if call.args.len() != 2 {
                self.valid = false;
            }
        }
        syn::visit_mut::visit_expr_call_mut(self, call);
    }
}

/// Caller restricts this proof to the reviewed production family modules.
/// Only the standalone-source arguments, source-name predicate and its import
/// are normalized. All configuration, weights, programs, state and calls remain
/// byte-for-byte equivalent as parsed Rust tokens, including the first two
/// metadata arguments. Unknown syntax retains full model/operator scope.
pub fn model_template_wiring_only(before: &str, after: &str) -> Result<bool, String> {
    fn retained(source: &str) -> Result<Option<(String, BTreeSet<String>)>, String> {
        let mut file =
            syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
        let mut imported = BTreeSet::new();
        let mut declarations = Vec::new();
        for item in file.items {
            match item {
                Item::Mod(module) if module.attrs.iter().any(test_configuration) => {}
                Item::Use(item)
                    if item.attrs.is_empty()
                        && matches!(item.vis, syn::Visibility::Inherited)
                        && item.leading_colon.is_none() =>
                {
                    imports(&item.tree, "", &mut imported)
                }
                item => declarations.push(item),
            }
        }
        if !imported.contains("super::hf_metadata::parse_hf_model_semantic_metadata") {
            return Ok(None);
        }
        imported.remove("super::hf_metadata::is_hf_template_source");
        file.items = declarations;
        let mut wiring = Wiring {
            calls: 0,
            valid: true,
        };
        wiring.visit_file_mut(&mut file);
        Ok((wiring.valid && wiring.calls > 0).then(|| (tokens(&file), imported)))
    }
    Ok(
        matches!((retained(before)?, retained(after)?), (Some(before), Some(after)) if before == after),
    )
}

const REQUIRED_GPTQ_FORMAT: &str = r#"
    if required_string(quantization, "quant_method")? != "gptq"
        || required_string(quantization, "checkpoint_format")? != "gptq" {
        return Err("quantization_config must describe a GPTQ checkpoint".to_owned());
    }
"#;
const GPTQ_METHOD: &str = r#"
    if required_string(quantization, "quant_method")? != "gptq" {
        return Err("quantization_config must describe a GPTQ checkpoint".to_owned());
    }
"#;
const OPTIONAL_GPTQ_FORMAT: &str = r#"
    for field in ["checkpoint_format", "format"] {
        if quantization.get(field).is_some_and(|value| value.as_str() != Some("gptq")) {
            return Err(format!("quantization_config.{field} must describe a GPTQ v1 checkpoint"));
        }
    }
"#;

/// A source acceptance change can require load tests without implying changed
/// architecture or numerical execution. Prove the exact GPTQ v1 format guards
/// changed, retaining every parsed value, conversion, validation and caller.
/// The caller restricts this proof to the reviewed Qwen3 MoE config module.
pub fn gptq_format_validation_only(before: &str, after: &str) -> Result<bool, String> {
    fn retained(source: &str) -> Result<(String, bool, bool), String> {
        let mut file =
            syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
        file.items.retain(|item| !matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)));
        let required: syn::Stmt = syn::parse_str(REQUIRED_GPTQ_FORMAT).unwrap();
        let optional: syn::Stmt = syn::parse_str(OPTIONAL_GPTQ_FORMAT).unwrap();
        let method: syn::Stmt = syn::parse_str(GPTQ_METHOD).unwrap();
        let mut old = 0;
        let mut new = 0;
        for item in &mut file.items {
            if let Item::Fn(function) = item {
                if function.sig.ident != "parse_quantization" {
                    continue;
                }
                let mut statements = Vec::new();
                for statement in std::mem::take(&mut function.block.stmts) {
                    if tokens(&statement) == tokens(&required) {
                        old += 1;
                        statements.push(method.clone());
                    } else if tokens(&statement) == tokens(&optional) {
                        new += 1;
                    } else {
                        statements.push(statement);
                    }
                }
                function.block.stmts = statements;
            }
        }
        Ok((tokens(&file), old == 1 && new == 0, old == 0 && new == 1))
    }
    let before = retained(before)?;
    let after = retained(after)?;
    Ok(before.1 && after.2 && before.0 == after.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    const BEFORE: &str = r#"
        use super::{hf_metadata::parse_hf_model_semantic_metadata, WeightSource};
        fn prepare(sources: &Sources, config: &Config) {
            let metadata = parse_hf_model_semantic_metadata(&config, tokenizer_bytes)?;
            if config.metadata.template.source_file != "tokenizer_config.json" { reject(); }
            let weights = WeightSource::open(sources.weights())?;
            build_program(weights, 128);
        }
    "#;
    fn after() -> String {
        BEFORE
            .replace("hf_metadata::parse_hf_model_semantic_metadata", "hf_metadata::{is_hf_template_source, parse_hf_model_semantic_metadata}")
            .replace("&config, tokenizer_bytes)", "&config, tokenizer_bytes, sources.chat_template_jinja(), sources.chat_template_json())")
            .replace("config.metadata.template.source_file != \"tokenizer_config.json\"", "!is_hf_template_source(&config.metadata.template.source_file)")
    }
    #[test]
    fn immutable_template_wiring_preserves_every_execution_expression() {
        assert!(model_template_wiring_only(BEFORE, &after()).unwrap());
        for changed in [
            after().replace("weights, 128", "weights, 256"),
            after().replace("sources.weights()", "other.weights()"),
            after().replace("&config, tokenizer_bytes", "&other, tokenizer_bytes"),
            after().replace(
                "sources.chat_template_jinja()",
                "allocate_and_get_template()",
            ),
            after().replace("reject();", "accept();"),
            after().replace("WeightSource", "DifferentWeightSource"),
            after().replace(
                "&config.metadata.template.source_file",
                "&config.metadata.template.template",
            ),
        ] {
            assert!(
                !model_template_wiring_only(BEFORE, &changed).unwrap(),
                "{changed}"
            );
        }
    }
    #[test]
    fn missing_import_unrecognized_arguments_and_macro_changes_remain_conservative() {
        assert!(!model_template_wiring_only("fn prepare() {}", &after()).unwrap());
        assert!(!model_template_wiring_only(
            BEFORE,
            &after().replace("sources.chat_template_json()", "None")
        )
        .unwrap());
        assert!(
            !model_template_wiring_only(BEFORE, &format!("{} operator!(changed);", after()))
                .unwrap()
        );
        assert!(model_template_wiring_only(BEFORE, "fn {").is_err());
    }
    #[test]
    fn gptq_format_acceptance_cannot_hide_changed_parameters_or_execution() {
        let before = format!("fn parse_quantization() {{ {REQUIRED_GPTQ_FORMAT} let bits = read_bits(root); validate(bits)?; Ok(bits) }}");
        let after = format!("fn parse_quantization() {{ {GPTQ_METHOD} {OPTIONAL_GPTQ_FORMAT} let bits = read_bits(root); validate(bits)?; Ok(bits) }}");
        assert!(gptq_format_validation_only(&before, &after).unwrap());
        for changed in [
            after.replace("read_bits(root)", "4"),
            after.replace("validate(bits)?;", ""),
            after.replace("Ok(bits)", "Ok(bits + 1)"),
            after.replace("Some(\"gptq\")", "Some(\"gptq_v2\")"),
            after.replace("value.as_str()", "mutate_and_get_format(value)"),
            format!("{after} fn new_operator() {{}}"),
        ] {
            assert!(
                !gptq_format_validation_only(&before, &changed).unwrap(),
                "{changed}"
            );
        }
        assert!(
            !gptq_format_validation_only(&before, &after.replace(OPTIONAL_GPTQ_FORMAT, ""))
                .unwrap()
        );
    }
}
