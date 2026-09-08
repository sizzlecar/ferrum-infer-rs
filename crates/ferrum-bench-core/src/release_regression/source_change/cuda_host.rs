//! Closed platform guards for optional exports and the Unix diagnostic loader.
//! No kernel/function body is discarded to project a platform's execution code.
use quote::ToTokens;
use std::collections::BTreeSet;
use syn::{parse::Parser, punctuated::Punctuated, Attribute, Item, Meta, Token, UseTree};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}
fn attribute(source: &str) -> Attribute {
    Attribute::parse_outer.parse_str(source).unwrap().remove(0)
}
fn parsed(source: &str) -> Result<syn::File, String> {
    syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))
}

// Only these literal predicates and one flat all(...) are understood. This is
// not a general cfg evaluator; any other expression is left unproven.
fn cfg_terms(attrs: &[Attribute]) -> Option<BTreeSet<String>> {
    if attrs.iter().any(|attr| attr.path().is_ident("cfg_attr")) {
        return None;
    }
    let cfgs: Vec<_> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg"))
        .collect();
    if cfgs.is_empty() {
        return Some(BTreeSet::new());
    }
    if cfgs.len() != 1 {
        return None;
    }
    let condition = cfgs[0].parse_args::<Meta>().ok()?;
    let terms = match condition {
        Meta::List(list) if list.path.is_ident("all") => list
            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .ok()?
            .into_iter()
            .collect(),
        condition => vec![condition],
    };
    let mut result = BTreeSet::new();
    for term in terms {
        let text = tokens(&term);
        let canonical = if matches!(
            text.as_str(),
            "unix" | "feature = \"cuda\"" | "feature = \"candle-cuda-compat\""
        ) {
            text
        } else if let Meta::List(list) = term {
            if !list.path.is_ident("not") {
                return None;
            }
            let nested = tokens(&list.parse_args::<Meta>().ok()?);
            match nested.as_str() {
                "unix" => "not(unix)".into(),
                "target_os = \"windows\"" => "not(windows)".into(),
                _ => return None,
            }
        } else {
            return None;
        };
        if !result.insert(canonical) {
            return None;
        }
    }
    Some(result)
}
fn set(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).into()).collect()
}
fn use_paths(tree: &UseTree, prefix: &str, output: &mut Vec<String>) {
    match tree {
        UseTree::Path(path) => use_paths(&path.tree, &format!("{prefix}{}::", path.ident), output),
        UseTree::Group(group) => {
            for item in &group.items {
                use_paths(item, prefix, output);
            }
        }
        leaf => output.push(format!("{prefix}{}", tokens(leaf))),
    }
}
fn split_use(item: &syn::ItemUse) -> Vec<syn::ItemUse> {
    let mut paths = Vec::new();
    use_paths(&item.tree, "", &mut paths);
    paths
        .into_iter()
        .map(|path| {
            let mut leaf = item.clone();
            leaf.tree = syn::parse_str(&path).expect("flattened parsed import");
            leaf
        })
        .collect()
}
fn replace_cfg(attrs: &mut Vec<Attribute>, replacement: Option<&str>) {
    attrs.retain(|attr| !attr.path().is_ident("cfg"));
    if let Some(replacement) = replacement {
        attrs.push(attribute(replacement));
    }
}
fn optional_guard(attrs: &mut Vec<Attribute>, export: bool, name: &str) -> Option<bool> {
    let (old, canonical) = match (export, name) {
        (false, "nccl_comm") => (set(&[]), None),
        (false, "tp_decode") => (
            set(&["feature = \"candle-cuda-compat\""]),
            Some("#[cfg(feature = \"candle-cuda-compat\")]"),
        ),
        (true, "nccl_comm") => (
            set(&["feature = \"cuda\""]),
            Some("#[cfg(feature = \"cuda\")]"),
        ),
        (true, "tp_decode") => (
            set(&["feature = \"cuda\"", "feature = \"candle-cuda-compat\""]),
            Some("#[cfg(all(feature = \"cuda\", feature = \"candle-cuda-compat\"))]"),
        ),
        _ => return None,
    };
    let mut new = old.clone();
    new.insert("not(windows)".into());
    let actual = cfg_terms(attrs)?;
    if actual != old && actual != new {
        return None;
    }
    replace_cfg(attrs, canonical);
    Some(actual == new)
}

/// Only existing nccl_comm/tp_decode module declarations and CUDA re-exports
/// may add a Windows exclusion. Every other item and every body stays intact.
pub fn cuda_optional_exports_only(before: &str, after: &str) -> Result<bool, String> {
    fn normalized(source: &str) -> Result<Option<(String, Vec<String>, BTreeSet<String>)>, String> {
        let mut file = parsed(source)?;
        let mut items = Vec::new();
        let mut uses = Vec::new();
        let mut guarded = BTreeSet::new();
        let mut seen = BTreeSet::new();
        for mut item in file.items {
            match &mut item {
                Item::Mod(module)
                    if matches!(module.ident.to_string().as_str(), "nccl_comm" | "tp_decode") =>
                {
                    let name = module.ident.to_string();
                    if module.content.is_some()
                        || module.semi.is_none()
                        || !matches!(module.vis, syn::Visibility::Public(_))
                    {
                        return Ok(None);
                    }
                    let Some(windows) = optional_guard(&mut module.attrs, false, &name) else {
                        return Ok(None);
                    };
                    let key = format!("mod:{name}");
                    if !seen.insert(key.clone()) {
                        return Ok(None);
                    }
                    if windows {
                        guarded.insert(key);
                    }
                }
                Item::Use(original) => {
                    for mut leaf in split_use(original) {
                        let path = tokens(&leaf.tree).replace(' ', "");
                        if let Some(name) = path
                            .strip_prefix("backend::cuda::")
                            .filter(|name| matches!(*name, "nccl_comm" | "tp_decode"))
                        {
                            if leaf.leading_colon.is_some()
                                || !matches!(leaf.vis, syn::Visibility::Public(_))
                            {
                                return Ok(None);
                            }
                            let Some(windows) = optional_guard(&mut leaf.attrs, true, name) else {
                                return Ok(None);
                            };
                            let key = format!("use:{name}");
                            if !seen.insert(key.clone()) {
                                return Ok(None);
                            }
                            if windows {
                                guarded.insert(key);
                            }
                        }
                        uses.push(tokens(&leaf));
                    }
                    continue;
                }
                _ => {}
            }
            items.push(item);
        }
        file.items = items;
        uses.sort();
        Ok(Some((tokens(&file), uses, guarded)))
    }
    let (Some(before), Some(after)) = (normalized(before)?, normalized(after)?) else {
        return Ok(false);
    };
    Ok(before.0 == after.0
        && before.1 == after.1
        && before.2 != after.2
        && before.2.is_subset(&after.2))
}

fn unix_guard(attrs: &mut Vec<Attribute>) -> Option<bool> {
    let terms = cfg_terms(attrs)?;
    if !terms.is_empty() && terms != set(&["unix"]) {
        return None;
    }
    replace_cfg(attrs, None);
    Some(!terms.is_empty())
}
fn unsupported_loader(function: &syn::ItemFn) -> bool {
    if cfg_terms(&function.attrs) != Some(set(&["not(unix)"])) {
        return false;
    }
    let [syn::Stmt::Expr(syn::Expr::Call(error), None)] = function.block.stmts.as_slice() else {
        return false;
    };
    if !error.attrs.is_empty() || tokens(&error.func) != "Err" || error.args.len() != 1 {
        return false;
    }
    let syn::Expr::Call(unsupported) = &error.args[0] else {
        return false;
    };
    let expected_call: syn::Expr = syn::parse_str("FerrumError::unsupported").unwrap();
    if !unsupported.attrs.is_empty()
        || tokens(&unsupported.func) != tokens(&expected_call)
        || unsupported.args.len() != 1
    {
        return false;
    }
    let syn::Expr::Lit(literal) = &unsupported.args[0] else {
        return false;
    };
    if !literal.attrs.is_empty()
        || !matches!(&literal.lit, syn::Lit::Str(message) if !message.value().trim().is_empty())
    {
        return false;
    }
    let expected: syn::ItemFn = syn::parse_str(
        "fn load_fa2_shim() -> Result<Fa2Shim> { Err(FerrumError::unsupported(\"unavailable\")) }",
    )
    .unwrap();
    let mut actual = function.clone();
    replace_cfg(&mut actual.attrs, None);
    actual.block = expected.block.clone();
    tokens(&actual) == tokens(&expected)
}

/// Only the Unix loader's existing declarations gain cfg(unix), plus a pure
/// non-Unix unsupported result. Unix dlopen and CUDA execution bodies are kept.
pub fn cuda_diagnostic_loader_only(before: &str, after: &str) -> Result<bool, String> {
    fn normalized(
        source: &str,
    ) -> Result<Option<(String, Vec<String>, BTreeSet<String>, usize)>, String> {
        let mut file = parsed(source)?;
        file.attrs.retain(|attr| !attr.path().is_ident("doc"));
        let mut items = Vec::new();
        let mut uses = Vec::new();
        let mut guarded = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut stubs = 0;
        for mut item in file.items {
            let marked = match &mut item {
                Item::Use(original) => {
                    for mut leaf in split_use(original) {
                        if tokens(&leaf.tree).replace(' ', "") == "std::ffi::CString" {
                            if leaf.leading_colon.is_some()
                                || !matches!(leaf.vis, syn::Visibility::Inherited)
                            {
                                return Ok(None);
                            }
                            let Some(unix) = unix_guard(&mut leaf.attrs) else {
                                return Ok(None);
                            };
                            if !seen.insert("CString".into()) {
                                return Ok(None);
                            }
                            if unix {
                                guarded.insert("CString".into());
                            }
                        }
                        uses.push(tokens(&leaf));
                    }
                    continue;
                }
                Item::ForeignMod(foreign)
                    if foreign.attrs.iter().any(|attr| {
                        tokens(attr) == tokens(&attribute("#[link(name = \"dl\")]"))
                    }) =>
                {
                    let names: Vec<_> = foreign
                        .items
                        .iter()
                        .filter_map(|item| {
                            if let syn::ForeignItem::Fn(function) = item {
                                Some(function.sig.ident.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if names.len() != foreign.items.len()
                        || names.iter().map(String::as_str).collect::<BTreeSet<_>>()
                            != BTreeSet::from(["dlopen", "dlsym", "dlerror"])
                        || names.len() != 3
                    {
                        return Ok(None);
                    }
                    Some(("extern_dl".to_owned(), &mut foreign.attrs))
                }
                Item::Const(value)
                    if matches!(value.ident.to_string().as_str(), "RTLD_NOW" | "RTLD_LOCAL") =>
                {
                    Some((value.ident.to_string(), &mut value.attrs))
                }
                Item::Fn(function)
                    if function.sig.ident == "load_fa2_shim"
                        && cfg_terms(&function.attrs) == Some(set(&["not(unix)"])) =>
                {
                    if !unsupported_loader(function) {
                        return Ok(None);
                    }
                    stubs += 1;
                    continue;
                }
                Item::Fn(function)
                    if matches!(
                        function.sig.ident.to_string().as_str(),
                        "dl_error_string" | "fa2_direct_ffi_shim_path_from_env" | "load_fa2_shim"
                    ) =>
                {
                    Some((function.sig.ident.to_string(), &mut function.attrs))
                }
                _ => None,
            };
            if let Some((name, attrs)) = marked {
                let Some(unix) = unix_guard(attrs) else {
                    return Ok(None);
                };
                if !seen.insert(name.clone()) {
                    return Ok(None);
                }
                if unix {
                    guarded.insert(name);
                }
            }
            items.push(item);
        }
        if seen
            != set(&[
                "CString",
                "extern_dl",
                "RTLD_NOW",
                "RTLD_LOCAL",
                "dl_error_string",
                "fa2_direct_ffi_shim_path_from_env",
                "load_fa2_shim",
            ])
        {
            return Ok(None);
        }
        file.items = items;
        uses.sort();
        Ok(Some((tokens(&file), uses, guarded, stubs)))
    }
    let (Some(before), Some(after)) = (normalized(before)?, normalized(after)?) else {
        return Ok(false);
    };
    Ok(before.0 == after.0
        && before.1 == after.1
        && before.3 == 0
        && after.3 == 1
        && before.2.is_subset(&after.2)
        && after.2.len() == 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPORTS: &str = r#"
        #[cfg(feature = "cuda")]
        pub use backend::cuda::{cublas, nccl_comm};
        #[cfg(all(feature = "cuda", feature = "candle-cuda-compat"))]
        pub use backend::cuda::{cuda_decode, tp_decode};
        pub fn execute(n: usize) -> usize { n * 2 }
    "#;
    const EXPORTS_AFTER: &str = r#"
        #[cfg(feature = "cuda")]
        pub use backend::cuda::cublas;
        #[cfg(all(feature = "cuda", not(target_os = "windows")))]
        pub use backend::cuda::nccl_comm;
        #[cfg(all(feature = "cuda", feature = "candle-cuda-compat"))]
        pub use backend::cuda::cuda_decode;
        #[cfg(all(feature = "cuda", feature = "candle-cuda-compat", not(target_os = "windows"),))]
        pub use backend::cuda::tp_decode;
        pub fn execute(n: usize) -> usize { n * 2 }
    "#;

    #[test]
    fn optional_exports_preserve_unix_feature_gates_and_other_code() {
        assert_eq!(cuda_optional_exports_only(EXPORTS, EXPORTS_AFTER), Ok(true));
        for changed in [
            EXPORTS_AFTER.replace("n * 2", "n * 3"),
            EXPORTS_AFTER.replace("not(target_os = \"windows\")", "not(target_os = \"linux\")"),
            EXPORTS_AFTER.replace("not(target_os = \"windows\")", "any(unix, windows)"),
            EXPORTS_AFTER.replace("feature = \"candle-cuda-compat\", not", "not"),
            EXPORTS_AFTER.replace(
                "pub use backend::cuda::cublas;",
                "pub use backend::cuda::other_kernel;",
            ),
            EXPORTS_AFTER.replace(
                "pub fn execute",
                "#[cfg(not(target_os = \"windows\"))] pub fn execute",
            ),
            EXPORTS_AFTER.replace(
                "pub use backend::cuda::nccl_comm;",
                "#[allow(unused)] pub use backend::cuda::nccl_comm;",
            ),
            format!("{EXPORTS_AFTER} #[cfg(windows)] fn hidden_kernel() {{ side_effect(); }}"),
        ] {
            assert_eq!(cuda_optional_exports_only(EXPORTS, &changed), Ok(false));
        }
        assert_eq!(
            cuda_optional_exports_only(EXPORTS_AFTER, EXPORTS),
            Ok(false)
        );
    }

    #[test]
    fn optional_modules_cannot_hide_bodies_or_unknown_attributes() {
        let before = "pub mod nccl_comm; #[cfg(feature = \"candle-cuda-compat\")] pub mod tp_decode; pub mod paged;";
        let after = "#[cfg(not(target_os = \"windows\"))] pub mod nccl_comm; #[cfg(all(feature = \"candle-cuda-compat\", not(target_os = \"windows\")))] pub mod tp_decode; pub mod paged;";
        assert_eq!(cuda_optional_exports_only(before, after), Ok(true));
        for changed in [
            after.replace("pub mod nccl_comm;", "pub mod nccl_comm { fn kernel() {} }"),
            after.replace("pub mod paged;", "#[cfg(unix)] pub mod paged;"),
            after.replace(
                "pub mod nccl_comm;",
                "#[cfg_attr(unix, path = \"other.rs\")] pub mod nccl_comm;",
            ),
            after.replace("pub mod tp_decode;", "mod tp_decode;"),
        ] {
            assert_eq!(cuda_optional_exports_only(before, &changed), Ok(false));
        }
    }

    const LOADER: &str = r#"
        //! Unix loader diagnostic.
        use std::ffi::{CString, c_char, c_int, c_void, CStr};
        use ferrum_types::{FerrumError, Result};
        struct Fa2Shim { handle: usize }
        #[link(name = "dl")]
        extern "C" {
            fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
            fn dlerror() -> *const c_char;
        }
        const RTLD_NOW: c_int = 2;
        const RTLD_LOCAL: c_int = 0;
        fn dl_error_string() -> String { read_error() }
        fn fa2_direct_ffi_shim_path_from_env() -> Option<String> { read_path() }
        fn load_fa2_shim() -> Result<Fa2Shim> { open_library(RTLD_NOW | RTLD_LOCAL) }
        pub fn execute(n: usize) -> usize { n * 2 }
    "#;
    fn loader_after() -> String {
        let mut after = LOADER
            .replace(
                "//! Unix loader diagnostic.",
                "//! Diagnostic is available on Unix; other platforms return unsupported.",
            )
            .replace(
                "use std::ffi::{CString, c_char, c_int, c_void, CStr};",
                "#[cfg(unix)] use std::ffi::CString; use std::ffi::{c_char, c_int, c_void, CStr};",
            );
        for item in [
            "#[link(name = \"dl\")]",
            "const RTLD_NOW",
            "const RTLD_LOCAL",
            "fn dl_error_string",
            "fn fa2_direct_ffi_shim_path_from_env",
            "fn load_fa2_shim",
        ] {
            after = after.replace(item, &format!("#[cfg(unix)] {item}"));
        }
        after.push_str("#[cfg(not(unix))] fn load_fa2_shim() -> Result<Fa2Shim> { Err(FerrumError::unsupported(\"Diagnostic shims require Unix\")) }");
        after
    }

    #[test]
    fn diagnostic_guard_keeps_dlopen_kernel_bodies_and_exact_unsupported_shape() {
        let after = loader_after();
        assert_eq!(cuda_diagnostic_loader_only(LOADER, &after), Ok(true));
        for changed in [
            after.replace("read_error()", "other_error()"),
            after.replace(
                "open_library(RTLD_NOW | RTLD_LOCAL)",
                "open_library(RTLD_NOW + RTLD_LOCAL)",
            ),
            after.replace("const RTLD_NOW: c_int = 2", "const RTLD_NOW: c_int = 4"),
            after.replace("fn dlsym(", "fn another_symbol("),
            after.replace("n * 2", "n * 3"),
            after.replace("#[cfg(unix)]", "#[cfg(not(target_os = \"windows\"))]"),
            after.replace(
                "Err(FerrumError::unsupported",
                "side_effect(); Err(FerrumError::unsupported",
            ),
            after.replace("\"Diagnostic shims require Unix\"", "compute_message()"),
            after.replace("FerrumError::unsupported", "FerrumError::model"),
            after.replace(
                "#[cfg(not(unix))] fn load_fa2_shim",
                "#[cfg(not(unix))] pub fn load_fa2_shim",
            ),
            after.replace(
                "#[cfg(not(unix))] fn load_fa2_shim",
                "#[cfg(not(unix))] unsafe fn load_fa2_shim",
            ),
            format!("{after} #[cfg(windows)] fn kernel() {{ launch(); }}"),
        ] {
            assert_eq!(cuda_diagnostic_loader_only(LOADER, &changed), Ok(false));
        }
        assert_eq!(cuda_diagnostic_loader_only(&after, LOADER), Ok(false));
        assert!(cuda_diagnostic_loader_only(LOADER, "fn broken(").is_err());
    }
}
