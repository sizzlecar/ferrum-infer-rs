//! Closed AST proofs for declared reasoning metadata; runtime/model code is retained.
use super::test_configuration;
use quote::ToTokens;
use std::collections::BTreeSet;
use syn::{parse::Parser, visit_mut::VisitMut, Expr, Item, UseTree};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}
fn expr_is(value: &Expr, expected: &str) -> bool {
    tokens(value) == tokens(&syn::parse_str::<Expr>(expected).expect("fixed scope expression"))
}
fn import_paths(tree: &UseTree, prefix: &str, output: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => {
            import_paths(&path.tree, &format!("{prefix}{}::", path.ident), output)
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                import_paths(tree, prefix, output);
            }
        }
        leaf => {
            output.insert(format!("{prefix}{}", tokens(leaf)));
        }
    }
}
fn parse(source: &str) -> Result<(syn::File, BTreeSet<String>), String> {
    let mut file = syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
    let mut imports = BTreeSet::new();
    file.items.retain(|item| match item {
        Item::Mod(module) if module.attrs.iter().any(test_configuration) => false,
        Item::Use(item)
            if item.attrs.is_empty()
                && matches!(item.vis, syn::Visibility::Inherited)
                && item.leading_colon.is_none() =>
        {
            import_paths(&item.tree, "", &mut imports);
            false
        }
        _ => true,
    });
    Ok((file, imports))
}

#[derive(Default)]
struct UnknownInitializer {
    count: usize,
    invalid: bool,
}
impl VisitMut for UnknownInitializer {
    fn visit_expr_struct_mut(&mut self, value: &mut syn::ExprStruct) {
        if tokens(&value.path) == "Self" || tokens(&value.path) == "CausalLanguageModelDescriptor" {
            let mut retained = syn::punctuated::Punctuated::new();
            for field in value.fields.clone() {
                if tokens(&field.member) == "reasoning_effort_support" {
                    self.count += 1;
                    self.invalid |= !field.attrs.is_empty()
                        || field.colon_token.is_none()
                        || !expr_is(&field.expr, "ReasoningEffortSupport::Unknown");
                } else {
                    retained.push(field);
                }
            }
            value.fields = retained;
        }
        syn::visit_mut::visit_expr_struct_mut(self, value);
    }
}

/// Caller restricts this to the shared causal descriptor module. The only new
/// state is an explicit Unknown capability; pure accessors do not alter shapes,
/// validation, programs, tensor ownership or constructor evaluation.
pub fn reasoning_descriptor_metadata_only(before: &str, after: &str) -> Result<bool, String> {
    fn retained(source: &str) -> Result<(String, BTreeSet<String>, [usize; 4], bool), String> {
        let (mut file, mut imports) = parse(source)?;
        let imported = imports.remove("ferrum_types::ReasoningEffortSupport");
        let expected_field = syn::Field::parse_named
            .parse_str("reasoning_effort_support: ReasoningEffortSupport")
            .unwrap();
        let expected_setter: syn::ImplItemFn = syn::parse_str("pub fn with_reasoning_effort_support(mut self, support: ReasoningEffortSupport) -> Self { self.reasoning_effort_support = support; self }").unwrap();
        let expected_getter: syn::ImplItemFn = syn::parse_str("pub fn reasoning_effort_support(&self) -> &ReasoningEffortSupport { &self.reasoning_effort_support }").unwrap();
        let mut counts = [0; 4];
        let mut valid = true;
        for item in &mut file.items {
            match item {
                Item::Struct(item) if item.ident == "CausalLanguageModelDescriptor" => {
                    let syn::Fields::Named(fields) = &mut item.fields else {
                        return Ok((String::new(), imports, counts, false));
                    };
                    let mut retained = syn::punctuated::Punctuated::new();
                    for field in fields.named.clone() {
                        if field
                            .ident
                            .as_ref()
                            .is_some_and(|name| name == "reasoning_effort_support")
                        {
                            counts[0] += 1;
                            valid &= tokens(&field) == tokens(&expected_field);
                        } else {
                            retained.push(field);
                        }
                    }
                    fields.named = retained;
                }
                Item::Impl(item) if tokens(&item.self_ty) == "CausalLanguageModelDescriptor" => {
                    let pure_impl = item.trait_.is_none()
                        && item.attrs.is_empty()
                        && item.generics.params.is_empty()
                        && item.generics.where_clause.is_none()
                        && item.unsafety.is_none()
                        && item.defaultness.is_none();
                    let mut retained = Vec::new();
                    for mut method in std::mem::take(&mut item.items) {
                        if let syn::ImplItem::Fn(function) = &mut method {
                            let expected = match function.sig.ident.to_string().as_str() {
                                "with_reasoning_effort_support" => Some((2, &expected_setter)),
                                "reasoning_effort_support" => Some((3, &expected_getter)),
                                _ => None,
                            };
                            if let Some((index, expected)) = expected {
                                counts[index] += 1;
                                valid &= pure_impl && tokens(function) == tokens(expected);
                                continue;
                            }
                            if function.sig.ident == "new" {
                                let mut initializer = UnknownInitializer::default();
                                initializer.visit_block_mut(&mut function.block);
                                counts[1] += initializer.count;
                                valid &=
                                    !initializer.invalid && (initializer.count == 0 || pure_impl);
                            }
                        }
                        retained.push(method);
                    }
                    item.items = retained;
                }
                _ => {}
            }
        }
        valid &= counts == [0; 4] || imported;
        Ok((tokens(&file), imports, counts, valid))
    }
    let before = retained(before)?;
    let after = retained(after)?;
    Ok(before.3
        && after.3
        && before.2 == [0; 4]
        && after.2 == [1; 4]
        && before.0 == after.0
        && before.1 == after.1)
}

fn pure_efforts(value: &Expr) -> bool {
    let Expr::Call(declared) = value else {
        return false;
    };
    if !declared.attrs.is_empty()
        || !expr_is(&declared.func, "ReasoningEffortSupport::Declared")
        || declared.args.len() != 1
    {
        return false;
    }
    let Expr::Call(set) = &declared.args[0] else {
        return false;
    };
    if !set.attrs.is_empty() || !expr_is(&set.func, "BTreeSet::from") || set.args.len() != 1 {
        return false;
    }
    let Expr::Array(values) = &set.args[0] else {
        return false;
    };
    if !values.attrs.is_empty() {
        return false;
    }
    let mut unique = BTreeSet::new();
    values.elems.iter().all(|value| {
        ["None", "Minimal", "Low", "Medium", "High", "XHigh", "Max"]
            .iter()
            .any(|variant| expr_is(value, &format!("ReasoningEffort::{variant}")))
            && unique.insert(tokens(value))
    })
}
fn descriptor_receiver(value: &Expr) -> bool {
    let Expr::MethodCall(protocol) = value else {
        return false;
    };
    let Expr::Try(construct) = &*protocol.receiver else {
        return false;
    };
    let Expr::Call(call) = &*construct.expr else {
        return false;
    };
    protocol.method == "with_output_protocol"
        && protocol.attrs.is_empty()
        && protocol.turbofish.is_none()
        && protocol.args.len() == 1
        && construct.attrs.is_empty()
        && call.attrs.is_empty()
        && expr_is(&call.func, "CausalLanguageModelDescriptor::new")
}
#[derive(Default)]
struct EffortDeclaration {
    count: usize,
    invalid: bool,
}
impl VisitMut for EffortDeclaration {
    fn visit_expr_mut(&mut self, expression: &mut Expr) {
        if let Expr::MethodCall(call) = expression {
            if call.method == "with_reasoning_effort_support" {
                self.count += 1;
                if !call.attrs.is_empty()
                    || call.turbofish.is_some()
                    || call.args.len() != 1
                    || !pure_efforts(&call.args[0])
                    || !descriptor_receiver(&call.receiver)
                {
                    self.invalid = true;
                    return;
                }
                *expression = (*call.receiver).clone();
            }
        }
        syn::visit_mut::visit_expr_mut(self, expression);
    }
}

/// Caller restricts this to a model provider module. The declaration may contain
/// only standard enum values; its constructor and every other expression remain.
/// This proves metadata reach, not that the chosen effort set is correct.
pub fn reasoning_effort_declaration_only(before: &str, after: &str) -> Result<bool, String> {
    fn retained(source: &str) -> Result<(String, BTreeSet<String>, usize, bool), String> {
        let (mut file, mut imports) = parse(source)?;
        let effort_import = imports.remove("ferrum_types::ReasoningEffort");
        let support_import = imports.remove("ferrum_types::ReasoningEffortSupport");
        let mut declaration = EffortDeclaration::default();
        for item in &mut file.items {
            if let Item::Fn(function) = item {
                if function.sig.ident == "production_descriptor" {
                    declaration.visit_block_mut(&mut function.block);
                }
            }
        }
        let valid = !declaration.invalid
            && (declaration.count == 0
                || (effort_import
                    && support_import
                    && imports.contains("std::collections::BTreeSet")));
        Ok((tokens(&file), imports, declaration.count, valid))
    }
    let before = retained(before)?;
    let after = retained(after)?;
    Ok(before.3
        && after.3
        && before.2 == 0
        && after.2 == 1
        && before.0 == after.0
        && before.1 == after.1)
}

/// Only the public reasoning-controls module and its glob re-export may be
/// introduced. Existing exports/types, attributes and all module bodies remain.
pub fn reasoning_controls_exports_only(before: &str, after: &str) -> Result<bool, String> {
    fn retained(source: &str) -> Result<(String, [usize; 2]), String> {
        let mut file =
            syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
        let expected: [Item; 2] = [
            syn::parse_str("pub mod reasoning_controls;").unwrap(),
            syn::parse_str("pub use reasoning_controls::*;").unwrap(),
        ];
        let mut counts = [0; 2];
        file.items.retain(|item| {
            if matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)) {
                return false;
            }
            for (index, expected) in expected.iter().enumerate() {
                if tokens(item) == tokens(expected) {
                    counts[index] += 1;
                    return false;
                }
            }
            true
        });
        Ok((tokens(&file), counts))
    }
    let before = retained(before)?;
    let after = retained(after)?;
    Ok(before.1 == [0; 2] && after.1 == [1; 2] && before.0 == after.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESCRIPTOR: &str = r#"
        use ferrum_types::ModelOutputProtocol;
        pub struct CausalLanguageModelDescriptor {
            hidden_size: usize,
            output_protocol: ModelOutputProtocol,
        }
        impl CausalLanguageModelDescriptor {
            pub fn new(hidden_size: usize) -> Result<Self> {
                validate_shape(hidden_size)?;
                Ok(Self { hidden_size, output_protocol: ModelOutputProtocol::Text, })
            }
            pub fn program(&self) { build_program(self.hidden_size); }
        }
    "#;
    fn descriptor_after() -> String {
        DESCRIPTOR
            .replace("use ferrum_types::ModelOutputProtocol;", "use ferrum_types::{ModelOutputProtocol, ReasoningEffortSupport};")
            .replace("output_protocol: ModelOutputProtocol,", "output_protocol: ModelOutputProtocol, reasoning_effort_support: ReasoningEffortSupport,")
            .replace("output_protocol: ModelOutputProtocol::Text,", "output_protocol: ModelOutputProtocol::Text, reasoning_effort_support: ReasoningEffortSupport::Unknown,")
            .replace("pub fn program", "pub fn with_reasoning_effort_support(mut self, support: ReasoningEffortSupport) -> Self { self.reasoning_effort_support = support; self }\n pub fn reasoning_effort_support(&self) -> &ReasoningEffortSupport { &self.reasoning_effort_support }\n pub fn program")
    }

    #[test]
    fn descriptor_metadata_retains_shape_constructor_and_program() {
        let after = descriptor_after();
        assert_eq!(
            reasoning_descriptor_metadata_only(DESCRIPTOR, &after),
            Ok(true)
        );
        for changed in [
            after.replace("ReasoningEffortSupport::Unknown", "detect_support()"),
            after.replace(
                "self.reasoning_effort_support = support;",
                "self.reasoning_effort_support = support; touch_runtime();",
            ),
            after.replace(
                "{ &self.reasoning_effort_support }",
                "{ trace(); &self.reasoning_effort_support }",
            ),
            after.replace(
                "validate_shape(hidden_size)",
                "validate_shape(hidden_size + 1)",
            ),
            after.replace(
                "build_program(self.hidden_size)",
                "build_program(self.hidden_size * 2)",
            ),
            after.replace("hidden_size: usize", "hidden_size: u32"),
            after.replace(
                "pub fn reasoning_effort_support",
                "#[cfg(feature = \"other\")] pub fn reasoning_effort_support",
            ),
            after.replace(
                "reasoning_effort_support: ReasoningEffortSupport,",
                "pub reasoning_effort_support: ReasoningEffortSupport,",
            ),
        ] {
            assert_eq!(
                reasoning_descriptor_metadata_only(DESCRIPTOR, &changed),
                Ok(false)
            );
        }
        assert_eq!(
            reasoning_descriptor_metadata_only(&after, DESCRIPTOR),
            Ok(false)
        );
        assert!(reasoning_descriptor_metadata_only(DESCRIPTOR, "fn broken(").is_err());
    }

    const PROVIDER: &str = r#"
        use std::collections::BTreeSet;
        use ferrum_types::{DataType, ModelOutputProtocol};
        fn production_descriptor(config: &Config) -> Result<CausalLanguageModelDescriptor> {
            let weights = load_schema(config)?;
            Ok(CausalLanguageModelDescriptor::new(config.shape, weights, DataType::FP16)?
                .with_output_protocol(ModelOutputProtocol::HarmonyGptOss))
        }
        fn program() { build_attention(128); }
    "#;
    fn provider_after(values: &str) -> String {
        PROVIDER
            .replace("DataType, ModelOutputProtocol", "DataType, ModelOutputProtocol, ReasoningEffort, ReasoningEffortSupport")
            .replace(".with_output_protocol(ModelOutputProtocol::HarmonyGptOss)",
                &format!(".with_output_protocol(ModelOutputProtocol::HarmonyGptOss).with_reasoning_effort_support(ReasoningEffortSupport::Declared(BTreeSet::from([{values}])))"))
    }

    #[test]
    fn effort_declaration_is_enum_metadata_and_cannot_erase_side_effects() {
        for values in [
            "ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High,",
            "ReasoningEffort::None",
            "",
        ] {
            assert_eq!(
                reasoning_effort_declaration_only(PROVIDER, &provider_after(values)),
                Ok(true)
            );
        }
        let after = provider_after("ReasoningEffort::Low, ReasoningEffort::High");
        for changed in [
            after.replace("ReasoningEffort::Low", "load_effort()"),
            after.replace(
                "ReasoningEffort::Low",
                "{ touch_runtime(); ReasoningEffort::Low }",
            ),
            after.replace("ReasoningEffort::Low", "ReasoningEffort::NotDeclared"),
            after.replace("ReasoningEffort::Low", "ReasoningEffort::High"),
            after.replace("BTreeSet::from", "custom_set"),
            after.replace("config.shape", "config.shape * 2"),
            after.replace("build_attention(128)", "build_attention(256)"),
            after.replace("load_schema(config)", "load_other_schema(config)"),
            after.replace(
                ".with_reasoning_effort_support(",
                ".with_reasoning_effort_support::<Other>(",
            ),
        ] {
            assert_eq!(
                reasoning_effort_declaration_only(PROVIDER, &changed),
                Ok(false)
            );
        }
        let unrelated_receiver = PROVIDER.replace(
            "CausalLanguageModelDescriptor::new(config.shape, weights, DataType::FP16)?",
            "another_builder()",
        );
        let unrelated_after = after.replace(
            "CausalLanguageModelDescriptor::new(config.shape, weights, DataType::FP16)?",
            "another_builder()",
        );
        assert_eq!(
            reasoning_effort_declaration_only(&unrelated_receiver, &unrelated_after),
            Ok(false)
        );
    }

    #[test]
    fn reasoning_exports_do_not_hide_other_public_contract_changes() {
        let before = "pub mod reasoning; pub use reasoning::*; pub type Size = usize;";
        let after = format!("{before} pub mod reasoning_controls; pub use reasoning_controls::*;");
        assert_eq!(reasoning_controls_exports_only(before, &after), Ok(true));
        for changed in [
            after.replace("usize", "u32"),
            after.replace(
                "pub mod reasoning_controls;",
                "#[cfg(feature = \"other\")] pub mod reasoning_controls;",
            ),
            after.replace(
                "pub mod reasoning_controls;",
                "pub mod reasoning_controls { pub fn execute() {} }",
            ),
            after.replace(
                "pub use reasoning_controls::*;",
                "pub use reasoning_controls::ReasoningEffort;",
            ),
            format!("{after} pub mod other;"),
            format!("{after} pub use reasoning_controls::*;"),
        ] {
            assert_eq!(reasoning_controls_exports_only(before, &changed), Ok(false));
        }
    }
}
