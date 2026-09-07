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
}
