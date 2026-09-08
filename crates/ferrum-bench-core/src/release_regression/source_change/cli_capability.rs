//! Closed CLI capability edits; all other production AST tokens are retained.
use super::test_configuration;
use quote::ToTokens;
use syn::{visit_mut::VisitMut, Expr, Item, Meta};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

fn expression(source: &str) -> Expr {
    syn::parse_str(source).expect("fixed CLI scope expression")
}

fn parse(source: &str) -> Result<syn::File, String> {
    let mut file = syn::parse_file(source).map_err(|error| format!("CLI scope parse: {error}"))?;
    file.items.retain(
        |item| !matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)),
    );
    Ok(file)
}

fn function<'a>(file: &'a mut syn::File, name: &str) -> Option<&'a mut syn::ItemFn> {
    let mut matches = file.items.iter_mut().filter_map(|item| match item {
        Item::Fn(item) if item.sig.ident == name => Some(item),
        _ => None,
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn mentions_name(value: &impl ToTokens, name: &str) -> bool {
    // Also reject occurrences inside macros or literals conservatively: an
    // added parameter must not shadow any pre-existing use hidden from visitors.
    tokens(value)
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|word| word == name)
}

const ORIGINAL_MASK: &str =
    "protocol == ModelOutputProtocol::Text && chat_template_options.enable_thinking == Some(false)";
const CAPABILITY_MASK: &str = "protocol == ModelOutputProtocol::Text && chat_template_options.enable_thinking == Some(false) && model_template.is_some_and(|template| template.reasoning_protocol.supports_reasoning())";

struct MaskCondition {
    added: bool,
    found: usize,
}
impl VisitMut for MaskCondition {
    fn visit_expr_if_mut(&mut self, value: &mut syn::ExprIf) {
        let expected = expression(if self.added {
            CAPABILITY_MASK
        } else {
            ORIGINAL_MASK
        });
        if tokens(&value.cond) == tokens(&expected) {
            self.found += 1;
            *value.cond = expression(ORIGINAL_MASK);
        }
        syn::visit_mut::visit_expr_if_mut(self, value);
    }
}

struct MetadataCalls {
    added: bool,
    count: usize,
    bindings: usize,
    option_binding: bool,
    valid: bool,
}
impl VisitMut for MetadataCalls {
    fn visit_pat_ident_mut(&mut self, value: &mut syn::PatIdent) {
        if value.ident == "model_chat_template" {
            self.bindings += 1;
        }
        syn::visit_mut::visit_pat_ident_mut(self, value);
    }

    fn visit_local_mut(&mut self, value: &mut syn::Local) {
        if tokens(&value.pat) == "model_chat_template" {
            // An existing Some arm fixes this local as Option, so the new
            // as_ref() is a borrow rather than an arbitrary method invocation.
            self.option_binding |= value.init.as_ref().is_some_and(|init| {
                matches!(&*init.expr, Expr::Match(value) if value.arms.iter().any(|arm| {
                    matches!(&*arm.body, Expr::Call(call) if tokens(&call.func) == "Some" && call.args.len() == 1)
                }))
            });
        }
        syn::visit_mut::visit_local_mut(self, value);
    }

    fn visit_expr_call_mut(&mut self, value: &mut syn::ExprCall) {
        if tokens(&value.func) == "run_request_metadata" {
            self.count += 1;
            self.valid &=
                value.attrs.is_empty() && value.args.len() == if self.added { 4 } else { 3 };
            if self.added {
                self.valid &= value.args.last().is_some_and(|argument| {
                    tokens(argument) == tokens(&expression("model_chat_template.as_ref()"))
                });
                value.args.pop();
            }
            value.args.pop_punct();
        }
        syn::visit_mut::visit_expr_call_mut(self, value);
    }
}

/// Only add the declared reasoning capability to the existing initial-token
/// mask and pass the existing optional template. No function body is erased.
pub fn cli_run_reasoning_mask_only(before: &str, after: &str) -> Result<bool, String> {
    let mut before = parse(before)?;
    let mut after = parse(after)?;
    let (Some(old), Some(new)) = (
        function(&mut before, "run_request_metadata"),
        function(&mut after, "run_request_metadata"),
    ) else {
        return Ok(false);
    };
    let mut signature: syn::Signature = syn::parse_str("fn run_request_metadata(prompt: &str, chat_template_options: &ChatTemplateOptions, protocol: ModelOutputProtocol) -> HashMap<String, serde_json::Value>").expect("fixed metadata signature");
    signature.inputs.pop_punct();
    old.sig.inputs.pop_punct();
    let parameter: syn::FnArg = syn::parse_str("model_template: Option<&ModelChatTemplate>")
        .expect("fixed template argument");
    if tokens(&old.sig) != tokens(&signature)
        || mentions_name(&old.block, "model_template")
        || new.sig.inputs.len() != 4
        || !new
            .sig
            .inputs
            .last()
            .is_some_and(|value| tokens(value) == tokens(&parameter))
    {
        return Ok(false);
    }
    new.sig.inputs.pop();
    new.sig.inputs.pop_punct();
    let mut old_mask = MaskCondition {
        added: false,
        found: 0,
    };
    let mut new_mask = MaskCondition {
        added: true,
        found: 0,
    };
    old_mask.visit_block_mut(&mut old.block);
    new_mask.visit_block_mut(&mut new.block);
    if old_mask.found != 1 || new_mask.found != 1 {
        return Ok(false);
    }
    let calls = |added| MetadataCalls {
        added,
        count: 0,
        bindings: 0,
        option_binding: false,
        valid: true,
    };
    let mut old_calls = calls(false);
    let mut new_calls = calls(true);
    old_calls.visit_file_mut(&mut before);
    new_calls.visit_file_mut(&mut after);
    Ok(old_calls.valid
        && new_calls.valid
        && old_calls.count > 0
        && old_calls.count == new_calls.count
        && old_calls.bindings == 1
        && new_calls.bindings == 1
        && old_calls.option_binding
        && new_calls.option_binding
        && tokens(&before) == tokens(&after))
}

fn reasoning_help(file: &mut syn::File) -> bool {
    let mut structures = 0;
    let mut fields = [0, 0];
    for item in &mut file.items {
        let Item::Struct(item) = item else { continue };
        if item.ident != "ServeCommand" {
            continue;
        }
        structures += 1;
        for field in &mut item.fields {
            let index = match field.ident.as_ref().map(ToString::to_string).as_deref() {
                Some("enable_thinking") => 0,
                Some("disable_thinking") => 1,
                _ => continue,
            };
            fields[index] += 1;
            if tokens(&field.ty) != "bool" {
                return false;
            }
            for attribute in &field.attrs {
                if attribute.path().is_ident("doc")
                    && !matches!(&attribute.meta, Meta::NameValue(value) if matches!(&value.value, Expr::Lit(value) if matches!(value.lit, syn::Lit::Str(_))))
                {
                    return false;
                }
            }
            field
                .attrs
                .retain(|attribute| !attribute.path().is_ident("doc"));
        }
    }
    structures == 1 && fields == [1, 1]
}

struct Fa2Capability {
    added: bool,
    found: usize,
    valid: bool,
}
impl VisitMut for Fa2Capability {
    fn visit_expr_struct_mut(&mut self, value: &mut syn::ExprStruct) {
        if tokens(&value.path) == "CompiledKernelFeatures" {
            for field in &mut value.fields {
                if tokens(&field.member) != "fa2_direct_ffi" {
                    continue;
                }
                self.found += 1;
                let expected = if self.added {
                    "cfg!(all(unix, feature = \"cuda\"))"
                } else {
                    "cfg!(feature = \"cuda\")"
                };
                self.valid &= field.attrs.is_empty()
                    && field.colon_token.is_some()
                    && tokens(&field.expr) == tokens(&expression(expected));
                field.expr = expression("cfg!(feature = \"cuda\")");
            }
        }
        syn::visit_mut::visit_expr_struct_mut(self, value);
    }
}

/// Accept only reasoning-option help text and the Unix availability boundary
/// of the legacy FA2 direct loader. Every other flag, field and call is retained.
pub fn cli_serve_host_capability_only(before: &str, after: &str) -> Result<bool, String> {
    let mut before = parse(before)?;
    let mut after = parse(after)?;
    if !reasoning_help(&mut before) || !reasoning_help(&mut after) {
        return Ok(false);
    }
    let (Some(old), Some(new)) = (
        function(&mut before, "compiled_kernel_features"),
        function(&mut after, "compiled_kernel_features"),
    ) else {
        return Ok(false);
    };
    let mut old_capability = Fa2Capability {
        added: false,
        found: 0,
        valid: true,
    };
    let mut new_capability = Fa2Capability {
        added: true,
        found: 0,
        valid: true,
    };
    old_capability.visit_block_mut(&mut old.block);
    new_capability.visit_block_mut(&mut new.block);
    Ok(old_capability.valid
        && new_capability.valid
        && old_capability.found == 1
        && new_capability.found == 1
        && tokens(&before) == tokens(&after))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_fixture(added: bool) -> String {
        let parameter = if added {
            ", model_template: Option<&ModelChatTemplate>"
        } else {
            ""
        };
        let argument = if added {
            ", model_chat_template.as_ref()"
        } else {
            ""
        };
        let condition = if added {
            CAPABILITY_MASK
        } else {
            ORIGINAL_MASK
        };
        format!(
            r#"
fn run_request_metadata(prompt: &str, chat_template_options: &ChatTemplateOptions, protocol: ModelOutputProtocol{parameter}) -> HashMap<String, serde_json::Value> {{
    let mut metadata = HashMap::new();
    if !has_unclosed_model_reasoning_block(protocol, prompt) {{
        let mut forbidden = markers(protocol);
        if {condition} {{ forbidden.push(THINK_START_TAG.to_string()); }}
        metadata.insert("initial".to_string(), forbidden);
    }}
    metadata
}}
fn execute() {{
    let model_chat_template = match prepared_model.as_deref() {{ Some(model) => Some(model.template()), None => None }};
    let first = run_request_metadata(&plan.prompt, &options, protocol{argument});
    if interactive {{ consume(run_request_metadata(&plan.prompt, &options, protocol{argument})); }}
    send(first);
}}
#[cfg(test)] mod tests {{ fn fixture() {{}} }}
"#
        )
    }

    #[test]
    fn reasoning_mask_preserves_existing_calls_and_every_other_statement() {
        let before = run_fixture(false);
        let after = run_fixture(true);
        assert!(cli_run_reasoning_mask_only(&before, &after).unwrap());
        for changed in [
            after.replace("metadata.insert", "side_effect(); metadata.insert"),
            after.replace("forbidden.push", "forbidden.clear(); forbidden.push"),
            after.replace(
                "supports_reasoning()",
                "supports_reasoning() || permit_all()",
            ),
            after.replace("model_chat_template.as_ref()", "load_template()"),
            after.replace("&plan.prompt", "rewrite_prompt(&plan.prompt)"),
            after.replace("send(first)", "send_other(first)"),
            after.replace("None => None", "None => load_template()"),
            after.replace("#[cfg(test)]", "#[cfg(all(test, unix))]"),
        ] {
            assert!(!cli_run_reasoning_mask_only(&before, &changed).unwrap());
        }
        assert!(!cli_run_reasoning_mask_only(&before, &before).unwrap());
        assert!(cli_run_reasoning_mask_only("fn invalid(", &after).is_err());
    }

    #[test]
    fn reasoning_mask_rejects_parameter_shadowing_and_unproven_template_borrows() {
        let before = run_fixture(false);
        let after = run_fixture(true);
        for (old, new) in [
            (before.replace("let mut metadata", "observe(model_template); let mut metadata"), after.replace("let mut metadata", "observe(model_template); let mut metadata")),
            (before.replace("let model_chat_template = match prepared_model.as_deref() { Some(model) => Some(model.template()), None => None }", "let model_chat_template = custom_borrower()"), after.replace("let model_chat_template = match prepared_model.as_deref() { Some(model) => Some(model.template()), None => None }", "let model_chat_template = custom_borrower()")),
            (before.replace("if interactive {", "if interactive { let model_chat_template = None;"), after.replace("if interactive {", "if interactive { let model_chat_template = None;")),
        ] {
            assert!(!cli_run_reasoning_mask_only(&old, &new).unwrap());
        }
    }

    fn serve_fixture(added: bool) -> String {
        let help = if added {
            "Standard effort or explicit thinking controls take precedence"
        } else {
            "Used when thinking is omitted"
        };
        let capability = if added {
            "cfg!(all(unix, feature = \"cuda\"))"
        } else {
            "cfg!(feature = \"cuda\")"
        };
        format!(
            r#"
pub struct ServeCommand {{
    #[doc = "{help}"] #[arg(long, conflicts_with = "disable_thinking")] pub enable_thinking: bool,
    #[doc = "{help}"] #[arg(long, conflicts_with = "enable_thinking")] pub disable_thinking: bool,
    #[doc = "Host to bind"] pub host: Option<String>,
}}
fn compiled_kernel_features() -> CompiledKernelFeatures {{
    let native = compiled_native();
    CompiledKernelFeatures {{ cuda: cfg!(feature = "cuda"), fa2_direct_ffi: {capability}, native }}
}}
fn execute() {{ serve(compiled_kernel_features()); }}
#[cfg(test)] mod tests {{ fn fixture() {{}} }}
"#
        )
    }

    #[test]
    fn host_capability_keeps_clap_semantics_other_features_and_production_execution() {
        let before = serve_fixture(false);
        let after = serve_fixture(true);
        assert!(cli_serve_host_capability_only(&before, &after).unwrap());
        for changed in [
            after.replace("conflicts_with", "requires"),
            after.replace("all(unix,", "any(unix,"),
            after.replace("cuda: cfg!(feature = \"cuda\")", "cuda: true"),
            after.replace("compiled_native()", "different_native()"),
            after.replace("let native", "side_effect(); let native"),
            after.replace("Host to bind", "Unrelated option semantics"),
            after.replace(
                "serve(compiled_kernel_features())",
                "serve(other_features())",
            ),
            after.replace("#[cfg(test)]", "#[cfg(all(test, windows))]"),
        ] {
            assert!(!cli_serve_host_capability_only(&before, &changed).unwrap());
        }
        assert!(!cli_serve_host_capability_only(&before, &before).unwrap());
    }
}
