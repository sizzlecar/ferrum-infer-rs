//! A closed proof for the existing CLI ready capability observation, not a
//! generic purity analyzer. Caller supplies immutable run and template sources.
use super::test_configuration;
use quote::{quote, ToTokens};
use syn::{parse::Parse, visit_mut::VisitMut, Expr, Item, Stmt, Token};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}
fn same(value: &impl ToTokens, source: &str) -> bool {
    // Both sides must pass through AST printing: raw macro tokens retain joint
    // punctuation such as `<&`, unlike a parsed Option<&T> parameter/type.
    let expected = syn::parse_str::<Expr>(source)
        .map(|value| tokens(&value))
        .or_else(|_| syn::parse_str::<syn::Type>(source).map(|value| tokens(&value)))
        .or_else(|_| syn::parse_str::<syn::Signature>(source).map(|value| tokens(&value)))
        .or_else(|_| syn::parse_str::<syn::FnArg>(source).map(|value| tokens(&value)))
        .or_else(|_| syn::parse_str::<Stmt>(source).map(|value| tokens(&value)));
    expected.is_ok_and(|expected| tokens(value) == expected)
}
fn scalar(expression: &Expr) -> bool {
    match expression {
        Expr::Path(value) => {
            value.attrs.is_empty()
                && value.qself.is_none()
                && value.path.segments.len() == 2
                && matches!(
                    value.path.segments[0].ident.to_string().as_str(),
                    "ModelOutputProtocol" | "ModelReasoningProtocol"
                )
                && value
                    .path
                    .segments
                    .iter()
                    .all(|segment| matches!(segment.arguments, syn::PathArguments::None))
        }
        Expr::Field(value) => {
            value.attrs.is_empty()
                && same(&value.base, "self")
                && matches!(
                    tokens(&value.member).as_str(),
                    "output_protocol" | "reasoning_protocol"
                )
        }
        Expr::Binary(value) => {
            value.attrs.is_empty()
                && matches!(value.op, syn::BinOp::Eq(_))
                && scalar(&value.left)
                && scalar(&value.right)
        }
        Expr::If(value) => {
            value.attrs.is_empty()
                && scalar(&value.cond)
                && observation(&value.then_branch)
                && value
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, expression)| scalar(expression))
        }
        Expr::Block(value) => {
            value.attrs.is_empty() && value.label.is_none() && observation(&value.block)
        }
        _ => false,
    }
}
fn observation(block: &syn::Block) -> bool {
    matches!(block.stmts.as_slice(), [Stmt::Expr(expression, None)] if scalar(expression))
}
fn imported(tree: &syn::UseTree, path: &[&str]) -> bool {
    match tree {
        syn::UseTree::Path(value) => {
            path.first().is_some_and(|name| value.ident == *name)
                && imported(&value.tree, &path[1..])
        }
        syn::UseTree::Name(value) => path.len() == 1 && value.ident == path[0],
        syn::UseTree::Group(value) => value.items.iter().any(|tree| imported(tree, path)),
        _ => false,
    }
}
fn pure_capability(source: &str) -> syn::Result<bool> {
    let file = syn::parse_file(source)?;
    if ["ModelOutputProtocol", "ModelReasoningProtocol"].iter().any(|name| !file.items.iter().any(|item|
        matches!(item, Item::Use(item) if item.attrs.is_empty() && imported(&item.tree, &["ferrum_types", name])))) { return Ok(false); }
    let fields = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(item) if item.ident == "ModelChatTemplate" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    if fields.len() != 1
        || !fields[0].generics.params.is_empty()
        || fields[0].generics.where_clause.is_some()
        || ["output_protocol", "reasoning_protocol"]
            .iter()
            .any(|name| {
                let expected = if *name == "output_protocol" {
                    "ModelOutputProtocol"
                } else {
                    "ModelReasoningProtocol"
                };
                !fields[0].fields.iter().any(|field| {
                    field.ident.as_ref().is_some_and(|ident| ident == *name)
                        && field.attrs.is_empty()
                        && same(&field.ty, expected)
                })
            })
    {
        return Ok(false);
    }
    let mut methods = Vec::new();
    for item in &file.items {
        if let Item::Impl(item) = item {
            if !same(&item.self_ty, "ModelChatTemplate") {
                continue;
            }
            for method in &item.items {
                if let syn::ImplItem::Fn(method) = method {
                    if method.sig.ident != "reasoning_capability" {
                        continue;
                    }
                    if item.trait_.is_some()
                        || !item.attrs.is_empty()
                        || !item.generics.params.is_empty()
                        || item.generics.where_clause.is_some()
                        || item.unsafety.is_some()
                        || item.defaultness.is_some()
                    {
                        return Ok(false);
                    }
                    methods.push(method);
                }
            }
        }
    }
    Ok(methods.len() == 1
        && matches!(methods[0].vis, syn::Visibility::Public(_))
        && methods[0]
            .attrs
            .iter()
            .all(|attribute| attribute.path().is_ident("doc"))
        && same(
            &methods[0].sig,
            "fn reasoning_capability(&self) -> ModelReasoningProtocol",
        )
        && observation(&methods[0].block))
}
struct Record(Vec<(syn::LitStr, Expr)>);
impl Parse for Record {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        syn::braced!(content in input);
        let mut fields = Vec::new();
        while !content.is_empty() {
            let key = content.parse()?;
            content.parse::<Token![:]>()?;
            fields.push((key, content.parse()?));
            if content.is_empty() {
                break;
            }
            content.parse::<Token![,]>()?;
        }
        Ok(Self(fields))
    }
}
fn ready(file: &mut syn::File) -> Option<&mut syn::ItemFn> {
    let mut functions = file.items.iter_mut().filter_map(|item| match item {
        Item::Fn(function) if function.sig.ident == "emit_jsonl_ready" => Some(function),
        _ => None,
    });
    let function = functions.next()?;
    if functions.next().is_some() {
        return None;
    }
    Some(function)
}
fn record(function: &mut syn::ItemFn) -> Option<&mut syn::Macro> {
    // Keep the sink and the entire emitter body. Only one local flat record
    // followed by the unchanged existing sink is supported; no extra statements.
    if function.block.stmts.len() != 2
        || !same(&function.block.stmts[1], "emit_jsonl_record(&record);")
    {
        return None;
    }
    let Stmt::Local(local) = &mut function.block.stmts[0] else {
        return None;
    };
    if !local.attrs.is_empty() || !same(&local.pat, "record") {
        return None;
    }
    let initializer = local.init.as_mut()?;
    if initializer.diverge.is_some() {
        return None;
    }
    let Expr::Macro(expression) = &mut *initializer.expr else {
        return None;
    };
    (expression.attrs.is_empty() && same(&expression.mac.path, "serde_json::json"))
        .then_some(&mut expression.mac)
}
struct Calls {
    added: bool,
    valid: bool,
    observed: usize,
    bindings: usize,
    option_binding: bool,
}
impl VisitMut for Calls {
    fn visit_pat_ident_mut(&mut self, pattern: &mut syn::PatIdent) {
        if pattern.ident == "model_chat_template" {
            self.bindings += 1;
        }
        syn::visit_mut::visit_pat_ident_mut(self, pattern);
    }
    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        if same(&local.pat, "model_chat_template") {
            // Rust unifies match-arm types: the existing Some constructor
            // fixes this local as Option, rather than an arbitrary as_ref API.
            self.option_binding |= local.init.as_ref().is_some_and(|init| matches!(&*init.expr,
                Expr::Match(value) if value.arms.iter().any(|arm| matches!(&*arm.body, Expr::Call(call) if same(&call.func, "Some") && call.args.len() == 1))));
        }
        syn::visit_mut::visit_local_mut(self, local);
    }
    fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
        if same(&call.func, "emit_jsonl_ready") {
            self.observed += 1;
            self.valid &=
                call.attrs.is_empty() && call.args.len() == if self.added { 5 } else { 4 };
            if self.added {
                self.valid &= call
                    .args
                    .last()
                    .is_some_and(|argument| same(argument, "model_chat_template.as_ref()"));
                call.args.pop();
            }
            call.args.pop_punct();
        }
        syn::visit_mut::visit_expr_call_mut(self, call);
    }
}
/// Normalize only the additional ready observation, then compare every other
/// production AST token. No original field, argument, branch or call disappears.
/// The scalar getter is checked in candidate source; it was not called by the
/// old emitter, so an absent old getter needs no exception or purity assumption.
pub fn run_ready_capability_only(
    before: &str,
    after: &str,
    candidate_template: &str,
) -> Result<bool, String> {
    let prove = || -> syn::Result<bool> {
        if !pure_capability(candidate_template)? {
            return Ok(false);
        }
        let mut before = syn::parse_file(before)?;
        let mut after = syn::parse_file(after)?;
        for file in [&mut before, &mut after] {
            file.items.retain(|item| !matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)));
        }
        let (Some(old), Some(new)) = (ready(&mut before), ready(&mut after)) else {
            return Ok(false);
        };
        if !old.attrs.is_empty() || !matches!(old.vis, syn::Visibility::Inherited)
            || !same(&old.sig, "fn emit_jsonl_ready(session_id: &str, requested_model: &str, resolved_model: &str, backend: &str)") || new.sig.inputs.len() != 5
            || !new.sig.inputs.last().is_some_and(|argument| same(argument, "template: Option<&ModelChatTemplate>")) { return Ok(false); }
        new.sig.inputs.pop();
        old.sig.inputs.pop_punct();
        new.sig.inputs.pop_punct();
        let (Some(old_record), Some(new_record)) = (record(old), record(new)) else {
            return Ok(false);
        };
        let old_fields: Record = syn::parse2(old_record.tokens.clone())?;
        let new_fields: Record = syn::parse2(new_record.tokens.clone())?;
        if old_fields
            .0
            .iter()
            .any(|(key, _)| key.value() == "reasoning_protocol")
            || new_fields.0.len() != old_fields.0.len() + 1
        {
            return Ok(false);
        }
        for ((old_key, old_value), (new_key, new_value)) in old_fields.0.iter().zip(&new_fields.0) {
            // A new parameter must not shadow an old record value, including
            // identifiers hidden inside a pre-existing expression or macro.
            let scalar_value = match old_value {
                Expr::Lit(value) => value.attrs.is_empty(),
                Expr::Path(value) => value.attrs.is_empty() && value.qself.is_none(),
                _ => false,
            };
            if !scalar_value || same(old_value, "template") {
                return Ok(false);
            }
            if tokens(&quote!(#old_key: #old_value)) != tokens(&quote!(#new_key: #new_value)) {
                return Ok(false);
            }
        }
        let (key, value) = new_fields.0.last().expect("new field exists");
        if key.value() != "reasoning_protocol"
            || !same(
                value,
                "template.map(ModelChatTemplate::reasoning_capability).unwrap_or_default()",
            )
        {
            return Ok(false);
        }
        new_record.tokens = old_record.tokens.clone();
        let mut old_calls = Calls {
            added: false,
            valid: true,
            observed: 0,
            bindings: 0,
            option_binding: false,
        };
        let mut new_calls = Calls {
            added: true,
            valid: true,
            observed: 0,
            bindings: 0,
            option_binding: false,
        };
        old_calls.visit_file_mut(&mut before);
        new_calls.visit_file_mut(&mut after);
        Ok(old_calls.valid
            && new_calls.valid
            && old_calls.bindings == 1
            && new_calls.bindings == 1
            && old_calls.option_binding
            && new_calls.option_binding
            && old_calls.observed > 0
            && old_calls.observed == new_calls.observed
            && tokens(&before) == tokens(&after))
    };
    prove().map_err(|error| format!("ready observation scope parse: {error}"))
}
