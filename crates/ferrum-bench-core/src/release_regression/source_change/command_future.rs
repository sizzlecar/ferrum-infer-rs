//! Prove the CLI's command-future allocation change preserves command dispatch.
use quote::ToTokens;
use std::collections::BTreeSet;
use syn::{visit_mut::VisitMut, Expr, Item, Stmt, UseTree};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

fn imports(tree: &UseTree, prefix: &str, result: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => imports(&path.tree, &format!("{prefix}{}::", path.ident), result),
        UseTree::Group(group) => {
            for item in &group.items {
                imports(item, prefix, result);
            }
        }
        leaf => {
            result.insert(format!("{prefix}{}", tokens(leaf)));
        }
    }
}

struct Punctuation;
impl VisitMut for Punctuation {
    fn visit_expr_match_mut(&mut self, value: &mut syn::ExprMatch) {
        for arm in &mut value.arms {
            arm.comma = Some(Default::default());
        }
        syn::visit_mut::visit_expr_match_mut(self, value);
    }
    fn visit_signature_mut(&mut self, value: &mut syn::Signature) {
        value.inputs.pop_punct();
        syn::visit_mut::visit_signature_mut(self, value);
    }
}

fn parse(source: &str) -> Result<(syn::File, BTreeSet<String>), String> {
    let mut file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut imported = BTreeSet::new();
    file.items.retain(|item| match item {
        Item::Mod(item) if item.attrs.iter().any(super::test_configuration) => false,
        Item::Use(item)
            if item.attrs.is_empty()
                && matches!(item.vis, syn::Visibility::Inherited)
                && item.leading_colon.is_none() =>
        {
            imports(&item.tree, "", &mut imported);
            false
        }
        _ => true,
    });
    Punctuation.visit_file_mut(&mut file);
    Ok((file, imported))
}

fn function<'a>(file: &'a mut syn::File, name: &str) -> Option<&'a mut syn::ItemFn> {
    let mut found = file.items.iter_mut().filter_map(|item| match item {
        Item::Fn(item) if item.sig.ident == name => Some(item),
        _ => None,
    });
    let value = found.next()?;
    if found.next().is_some() {
        return None;
    }
    Some(value)
}

fn expr_is(value: &Expr, expected: &str) -> bool {
    tokens(value) == tokens(&syn::parse_str::<Expr>(expected).expect("fixed CLI scope expression"))
}

#[derive(Default)]
struct EscapingControl(bool);
impl VisitMut for EscapingControl {
    fn visit_expr_mut(&mut self, value: &mut Expr) {
        if matches!(
            value,
            Expr::Return(_) | Expr::Try(_) | Expr::Break(_) | Expr::Continue(_) | Expr::Yield(_)
        ) {
            self.0 = true;
        }
        syn::visit_mut::visit_expr_mut(self, value);
    }
    fn visit_macro_mut(&mut self, _: &mut syn::Macro) {
        // Macro expansion can contain return or ?, whose destination changes
        // when the original block moves inside a separate future.
        self.0 = true;
    }
}

fn restore_dispatch(function: &syn::ItemFn) -> Option<Expr> {
    let mut expected: syn::ItemFn = syn::parse_str("fn command_future(command: Commands, config: CliConfig, config_loaded: bool) -> Pin<Box<dyn Future<Output = ferrum_types::Result<()>>>> {} ").unwrap();
    Punctuation.visit_item_fn_mut(&mut expected);
    if !function.attrs.is_empty()
        || tokens(&function.vis) != tokens(&expected.vis)
        || tokens(&function.sig) != tokens(&expected.sig)
    {
        return None;
    }
    let [Stmt::Expr(Expr::Match(dispatch), None)] = function.block.stmts.as_slice() else {
        return None;
    };
    if !expr_is(&dispatch.expr, "command") {
        return None;
    }
    let mut dispatch = dispatch.clone();
    dispatch.expr = Box::new(syn::parse_quote!(cli.command));
    for arm in &mut dispatch.arms {
        let Expr::Call(boxed) = &*arm.body else {
            return None;
        };
        if !boxed.attrs.is_empty() || !expr_is(&boxed.func, "Box::pin") || boxed.args.len() != 1 {
            return None;
        }
        let future = boxed.args[0].clone();
        arm.body = match future {
            Expr::Async(value) if value.attrs.is_empty() && value.capture.is_some() => {
                let mut block = value.block;
                let mut control = EscapingControl::default();
                control.visit_block_mut(&mut block);
                if control.0 {
                    return None;
                }
                Box::new(syn::parse_quote!({ #block }))
            }
            Expr::Call(value) if value.attrs.is_empty() => {
                Box::new(syn::parse_quote!(#value.await))
            }
            _ => return None,
        };
        // A moved async block keeps the same original block, without another
        // scope around its locals or a changed evaluation order.
        if let Expr::Block(value) = &mut *arm.body {
            if let [Stmt::Expr(Expr::Block(inner), None)] = value.block.stmts.as_slice() {
                value.block = inner.block.clone();
            }
        }
    }
    Some(Expr::Match(dispatch))
}

/// Caller restricts this to the CLI main module. Only the allocation of the
/// existing command futures changes; all arms, arguments, startup configuration
/// and error handling must still match the original parsed source.
pub fn boxed_command_dispatch_only(before: &str, after: &str) -> Result<bool, String> {
    let (mut before, old_imports) = parse(before)?;
    let (mut after, mut new_imports) = parse(after)?;
    for name in ["std::future::Future", "std::pin::Pin"] {
        if old_imports.contains(name) || !new_imports.remove(name) {
            return Ok(false);
        }
    }
    if old_imports != new_imports || function(&mut before, "command_future").is_some() {
        return Ok(false);
    }
    let Some(dispatch) = function(&mut after, "command_future").and_then(|f| restore_dispatch(f))
    else {
        return Ok(false);
    };
    after
        .items
        .retain(|item| !matches!(item, Item::Fn(item) if item.sig.ident == "command_future"));
    let Some(original) = function(&mut before, "main") else {
        return Ok(false);
    };
    let [.., Stmt::Local(result), Stmt::Expr(Expr::If(original_if), None)] =
        original.block.stmts.as_slice()
    else {
        return Ok(false);
    };
    if tokens(&result.pat) != "result" {
        return Ok(false);
    }
    let Some(init) = result.init.as_ref() else {
        return Ok(false);
    };
    if init.diverge.is_some() || !matches!(&*init.expr, Expr::Match(_)) {
        return Ok(false);
    }
    let Expr::Let(original_condition) = &*original_if.cond else {
        return Ok(false);
    };
    if !expr_is(&original_condition.expr, "result") {
        return Ok(false);
    }
    let mut restored_result = result.clone();
    restored_result.init.as_mut().unwrap().expr = Box::new(dispatch);
    let Some(current) = function(&mut after, "main") else {
        return Ok(false);
    };
    let Some(Stmt::Expr(Expr::If(current_if), None)) = current.block.stmts.last_mut() else {
        return Ok(false);
    };
    let Expr::Let(condition) = &mut *current_if.cond else {
        return Ok(false);
    };
    if !expr_is(
        &condition.expr,
        "command_future(cli.command, config, config_loaded).await",
    ) {
        return Ok(false);
    }
    condition.expr = Box::new(syn::parse_quote!(result));
    current
        .block
        .stmts
        .insert(current.block.stmts.len() - 1, Stmt::Local(restored_result));
    Ok(tokens(&before) == tokens(&after))
}

#[cfg(test)]
mod tests {
    use super::*;
    const BEFORE: &str = r#"
        use std::process;
        async fn main() {
            let config = load();
            let result = match cli.command {
                Commands::Run(cmd) => run::execute(cmd, config).await,
                Commands::Serve(cmd) => { let ready = check(config_loaded).await; serve::execute(cmd, ready).await }
            };
            if let Err(e) = result { report(e); process::exit(1); }
        }
    "#;
    const AFTER: &str = r#"
        use std::{future::Future, pin::Pin, process};
        async fn main() {
            let config = load();
            if let Err(e) = command_future(cli.command, config, config_loaded).await { report(e); process::exit(1); }
        }
        fn command_future(command: Commands, config: CliConfig, config_loaded: bool) -> Pin<Box<dyn Future<Output = ferrum_types::Result<()>>>> {
            match command {
                Commands::Run(cmd) => Box::pin(run::execute(cmd, config)),
                Commands::Serve(cmd) => Box::pin(async move { let ready = check(config_loaded).await; serve::execute(cmd, ready).await }),
            }
        }
    "#;
    #[test]
    fn allocation_refactor_preserves_commands_arguments_and_error_handling() {
        assert!(boxed_command_dispatch_only(BEFORE, AFTER).unwrap());
        for changed in [
            AFTER.replace("cmd, config)", "cmd, other_config)"),
            AFTER.replace("check(config_loaded)", "check(true)"),
            AFTER.replace("process::exit(1)", "process::exit(0)"),
            AFTER.replace("let config = load()", "let config = different()"),
            AFTER.replace("Box::pin", "different::pin"),
            AFTER.replace("async move", "async"),
            AFTER.replace("let ready = check", "modify_device(); let ready = check"),
        ] {
            assert!(
                !boxed_command_dispatch_only(BEFORE, &changed).unwrap(),
                "{changed}"
            );
        }
    }

    #[test]
    fn moving_a_block_cannot_change_return_or_error_propagation_boundaries() {
        for control in [
            "return Err(fail());",
            "validate()?;",
            "finish_or_return!();",
        ] {
            let move_block = |source: &str| {
                source
                    .replace("async fn main()", "async fn main() -> Result<()>")
                    .replace("let ready = check", &format!("{control} let ready = check"))
                    .replace(
                        "report(e); process::exit(1); }",
                        "report(e); Err(e) } else { Ok(()) }",
                    )
            };
            assert!(
                !boxed_command_dispatch_only(&move_block(BEFORE), &move_block(AFTER)).unwrap(),
                "{control}"
            );
        }
    }
}
