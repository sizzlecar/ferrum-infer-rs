mod vnext_core_contract;

use syn::visit::{self, Visit};
use vnext_core_contract::*;

#[derive(Default)]
struct UnsupportedSuccessVisitor {
    function_stack: Vec<String>,
    unsupported_depth: usize,
    violations: Vec<String>,
    downcasts: Vec<String>,
}

impl UnsupportedSuccessVisitor {
    fn enter(&mut self, name: &syn::Ident) -> bool {
        let unsupported = name.to_string().contains("unsupported");
        self.function_stack.push(name.to_string());
        self.unsupported_depth += usize::from(unsupported);
        unsupported
    }

    fn leave(&mut self, unsupported: bool) {
        self.unsupported_depth -= usize::from(unsupported);
        self.function_stack.pop();
    }

    fn current_function(&self) -> String {
        self.function_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "<module>".to_owned())
    }
}

impl<'ast> Visit<'ast> for UnsupportedSuccessVisitor {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        let unsupported = self.enter(&function.sig.ident);
        visit::visit_item_fn(self, function);
        self.leave(unsupported);
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        let unsupported = self.enter(&function.sig.ident);
        visit::visit_impl_item_fn(self, function);
        self.leave(unsupported);
    }

    fn visit_trait_item_fn(&mut self, function: &'ast syn::TraitItemFn) {
        let unsupported = self.enter(&function.sig.ident);
        visit::visit_trait_item_fn(self, function);
        self.leave(unsupported);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let returns_empty_ok = matches!(call.func.as_ref(), syn::Expr::Path(path)
            if path.path.segments.last().is_some_and(|segment| segment.ident == "Ok"))
            && matches!(call.args.first(), Some(syn::Expr::Tuple(tuple)) if tuple.elems.is_empty())
            && call.args.len() == 1;
        if returns_empty_ok && self.unsupported_depth > 0 {
            self.violations.push(self.current_function());
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "downcast_ref" {
            self.downcasts.push(self.current_function());
        }
        visit::visit_expr_method_call(self, call);
    }
}

fn downcasts_are_panic_boundary_only(path: &std::path::Path, downcasts: &[String]) -> bool {
    downcasts.is_empty()
        || (path.ends_with("resource/static_initialization.rs")
            && downcasts == ["panic_message", "panic_message"])
}

fn vnext_source_files() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/vnext");
    let mut directories = vec![root];
    let mut paths = Vec::new();
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                directories.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
}

#[test]
fn generic_contracts_have_zero_architecture_names() {
    let names = [
        "qwen", "llama", "deepseek", "mistral", "mixtral", "gemma", "chatglm", "internlm",
        "baichuan",
    ];
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap().to_ascii_lowercase();
        for name in names {
            assert!(!source.contains(name), "{} contains {name}", path.display());
        }
    }
}

#[test]
fn silent_success_defaults_are_absent() {
    for path in vnext_source_files() {
        let source = fs::read_to_string(&path).unwrap();
        let syntax = syn::parse_file(&source).unwrap();
        let mut visitor = UnsupportedSuccessVisitor::default();
        visitor.visit_file(&syntax);
        assert!(
            visitor.violations.is_empty(),
            "{} has unsupported functions that silently return Ok(()): {:?}",
            path.display(),
            visitor.violations
        );
        assert!(
            downcasts_are_panic_boundary_only(&path, &visitor.downcasts),
            "{} has non-panic-boundary downcast_ref calls: {:?}",
            path.display(),
            visitor.downcasts
        );
        assert!(!source.contains("std::env::var"));
    }
}

#[test]
fn failure_envelope_wire_limit_precedes_deserialization() {
    let at_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES];
    match FailureEnvelope::decode_untrusted(&at_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(!message.contains("maximum is"));
        }
        other => panic!("equal-to-limit malformed payload hit wrong result: {other:?}"),
    }

    let over_limit = vec![b' '; MAX_FAILURE_ENVELOPE_WIRE_BYTES + 1];
    match FailureEnvelope::decode_untrusted(&over_limit) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted failure envelope");
            assert!(message.contains("maximum is 8192"));
        }
        other => panic!("oversized payload hit wrong result: {other:?}"),
    }
}
