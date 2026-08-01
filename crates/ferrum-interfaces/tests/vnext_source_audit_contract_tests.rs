mod vnext_core_contract;

use syn::visit::{self, Visit};
use vnext_core_contract::*;

#[derive(Default)]
struct UnsupportedSuccessVisitor {
    function_stack: Vec<String>,
    unsupported_depth: usize,
    panic_boundary_depth: usize,
    violations: Vec<String>,
    downcasts: Vec<String>,
}

fn type_path_ends_with(ty: &syn::Type, expected: &str) -> bool {
    matches!(ty, syn::Type::Path(path)
        if path.path.segments.last().is_some_and(|segment| segment.ident == expected))
}

fn is_panic_payload_boundary(signature: &syn::Signature) -> bool {
    let Some(syn::FnArg::Typed(argument)) = signature.inputs.first() else {
        return false;
    };
    let syn::Type::Path(box_type) = argument.ty.as_ref() else {
        return false;
    };
    let Some(box_segment) = box_type.path.segments.last() else {
        return false;
    };
    let syn::PathArguments::AngleBracketed(arguments) = &box_segment.arguments else {
        return false;
    };
    let Some(syn::GenericArgument::Type(syn::Type::TraitObject(payload))) = arguments.args.first()
    else {
        return false;
    };
    let mut bounds = payload
        .bounds
        .iter()
        .filter_map(|bound| match bound {
            syn::TypeParamBound::Trait(bound) => bound
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();
    bounds.sort();
    signature.ident == "panic_message"
        && signature.inputs.len() == 1
        && box_segment.ident == "Box"
        && arguments.args.len() == 1
        && bounds == ["Any", "Send"]
        && matches!(&signature.output, syn::ReturnType::Type(_, output)
            if type_path_ends_with(output, "String"))
}

impl UnsupportedSuccessVisitor {
    fn enter(&mut self, signature: &syn::Signature) -> (bool, bool) {
        let unsupported = signature.ident.to_string().contains("unsupported");
        let panic_boundary = is_panic_payload_boundary(signature);
        self.function_stack.push(signature.ident.to_string());
        self.unsupported_depth += usize::from(unsupported);
        self.panic_boundary_depth += usize::from(panic_boundary);
        (unsupported, panic_boundary)
    }

    fn leave(&mut self, unsupported: bool, panic_boundary: bool) {
        self.unsupported_depth -= usize::from(unsupported);
        self.panic_boundary_depth -= usize::from(panic_boundary);
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
        let (unsupported, panic_boundary) = self.enter(&function.sig);
        visit::visit_item_fn(self, function);
        self.leave(unsupported, panic_boundary);
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        let (unsupported, panic_boundary) = self.enter(&function.sig);
        visit::visit_impl_item_fn(self, function);
        self.leave(unsupported, panic_boundary);
    }

    fn visit_trait_item_fn(&mut self, function: &'ast syn::TraitItemFn) {
        let (unsupported, panic_boundary) = self.enter(&function.sig);
        visit::visit_trait_item_fn(self, function);
        self.leave(unsupported, panic_boundary);
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
        if call.method == "downcast_ref" && self.panic_boundary_depth == 0 {
            self.downcasts.push(self.current_function());
        }
        visit::visit_expr_method_call(self, call);
    }
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
            visitor.downcasts.is_empty(),
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
