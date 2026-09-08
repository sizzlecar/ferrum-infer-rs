//! Narrow host portability proof for the core CUDA build entrypoint.
//!
//! The complete device compiler command, source/header inventory, cache inputs
//! and dispatch remain in the comparison. Only explicitly described host
//! selection and identity additions are restored to their previous AST.
use quote::ToTokens;
use std::collections::BTreeSet;
use syn::{visit_mut::VisitMut, Block, Expr, Item, Stmt, UseTree};

// These standard expression-list macros permit a trailing comma. Parse only
// their Rust expressions, never expand or reinterpret an unknown macro.
fn macro_arguments(
    value: &syn::Macro,
) -> Option<syn::punctuated::Punctuated<Expr, syn::Token![,]>> {
    let name = value.path.get_ident()?.to_string();
    if !matches!(
        name.as_str(),
        "format" | "println" | "panic" | "assert" | "assert_eq" | "vec"
    ) {
        return None;
    }
    value
        .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
        .ok()
}

struct Punctuation;
impl VisitMut for Punctuation {
    fn visit_item_enum_mut(&mut self, value: &mut syn::ItemEnum) {
        value.variants.pop_punct();
        syn::visit_mut::visit_item_enum_mut(self, value);
    }
    fn visit_expr_array_mut(&mut self, value: &mut syn::ExprArray) {
        value.elems.pop_punct();
        syn::visit_mut::visit_expr_array_mut(self, value);
    }
    fn visit_expr_call_mut(&mut self, value: &mut syn::ExprCall) {
        value.args.pop_punct();
        syn::visit_mut::visit_expr_call_mut(self, value);
    }
    fn visit_expr_method_call_mut(&mut self, value: &mut syn::ExprMethodCall) {
        value.args.pop_punct();
        syn::visit_mut::visit_expr_method_call_mut(self, value);
    }
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
    fn visit_macro_mut(&mut self, value: &mut syn::Macro) {
        if let Some(mut args) = macro_arguments(value) {
            args.pop_punct();
            for arg in &mut args {
                self.visit_expr_mut(arg);
            }
            value.tokens = args.to_token_stream();
        }
    }
}

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

fn block(source: &str) -> Block {
    let mut value =
        syn::parse_str(&format!("{{ {source} }}")).expect("fixed CUDA host scope statements");
    Punctuation.visit_block_mut(&mut value);
    value
}

struct StatementRule {
    after: Vec<Stmt>,
    before: Vec<Stmt>,
    matches: usize,
}

impl StatementRule {
    fn new(after: &str, before: &str) -> Self {
        Self {
            after: block(after).stmts,
            before: block(before).stmts,
            matches: 0,
        }
    }
}

impl VisitMut for StatementRule {
    fn visit_block_mut(&mut self, value: &mut Block) {
        let mut index = 0;
        while index + self.after.len() <= value.stmts.len() {
            if value.stmts[index..index + self.after.len()]
                .iter()
                .zip(&self.after)
                .all(|(a, b)| tokens(a) == tokens(b))
            {
                value
                    .stmts
                    .splice(index..index + self.after.len(), self.before.clone());
                self.matches += 1;
                index += self.before.len();
            } else {
                index += 1;
            }
        }
        syn::visit_mut::visit_block_mut(self, value);
    }
}

struct ExpressionRule {
    after: Expr,
    before: Expr,
    matches: usize,
}

impl VisitMut for ExpressionRule {
    fn visit_expr_mut(&mut self, value: &mut Expr) {
        if tokens(value) == tokens(&self.after) {
            *value = self.before.clone();
            self.matches += 1;
        } else {
            syn::visit_mut::visit_expr_mut(self, value);
        }
    }
    fn visit_macro_mut(&mut self, value: &mut syn::Macro) {
        if let Some(mut args) = macro_arguments(value) {
            for arg in &mut args {
                self.visit_expr_mut(arg);
            }
            value.tokens = args.to_token_stream();
        }
    }
}

fn statements(function: &mut syn::ItemFn, rules: &[(&str, &str)]) -> bool {
    rules.iter().all(|(after, before)| {
        let mut rule = StatementRule::new(after, before);
        rule.visit_block_mut(&mut function.block);
        rule.matches == 1
    })
}

fn expressions(function: &mut syn::ItemFn, rules: &[(&str, &str)]) -> bool {
    rules.iter().all(|(after, before)| {
        let mut rule = ExpressionRule {
            after: syn::parse_str(after).expect("fixed host scope expression"),
            before: syn::parse_str(before).expect("fixed host scope expression"),
            matches: 0,
        };
        Punctuation.visit_expr_mut(&mut rule.after);
        Punctuation.visit_expr_mut(&mut rule.before);
        rule.visit_block_mut(&mut function.block);
        rule.matches == 1
    })
}

fn function<'a>(file: &'a mut syn::File, name: &str) -> Option<&'a mut syn::ItemFn> {
    let mut matching = file.items.iter_mut().filter_map(|item| match item {
        Item::Fn(item) if item.sig.ident == name => Some(item),
        _ => None,
    });
    let found = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    Some(found)
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

const CORE_SELECTION: &str = r#"
    let host_tools = HostTools::current();
    let selected_nvcc = resolve_program(&host_tools.nvcc(cuda_root.as_deref()));
    let nvcc = host_tools.nvcc_invocation(&selected_nvcc)
        .unwrap_or_else(|error| panic!("[core-ptx] invalid NVCC invocation: {error}"));
    let invocation_identity = || {
        (host_tools == HostTools::WindowsMsvc).then(|| {
            let actual = nvcc.canonicalize().unwrap_or_else(|error| {
                panic!("[core-ptx] cannot resolve NVCC invocation {}: {error}", nvcc.display())
            });
            assert_eq!(actual, selected_nvcc,
                "[core-ptx] NVCC invocation must select the recorded compiler");
            sha256_file_fingerprint(&actual)
        })
    };
    let selected_nvcc_identity = invocation_identity();
"#;

fn normalize_core(function: &mut syn::ItemFn) -> bool {
    statements(
        function,
        &[
            (
                CORE_SELECTION,
                r#"let nvcc = cuda_root.as_ref()
            .map(|r| r.join("bin").join("nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"));"#,
            ),
            (
                "let environment_option = HostTools::current().nvcc_environment_option();",
                "",
            ),
            ("flags.extend(environment_option.map(str::to_string));", ""),
            ("command.args(environment_option);", ""),
            (
                r#"assert_eq!(invocation_identity(), selected_nvcc_identity,
            "[core-ptx] NVCC changed before compilation");"#,
                "",
            ),
            (
                r#"assert_eq!(invocation_identity(), selected_nvcc_identity,
            "[core-ptx] NVCC changed during compilation");"#,
                "",
            ),
            (
                r#"if HostTools::current() == HostTools::Unix {
                command.arg("-allow-unsupported-compiler");
            }
            command.args(["-ccbin", ccbin]);"#,
                r#"command.arg("-allow-unsupported-compiler").args(["-ccbin", ccbin]);"#,
            ),
        ],
    )
}

fn normalize_resolver(function: &mut syn::ItemFn) -> bool {
    statements(function, &[
        ("let candidates = HostTools::current().executable_candidates(program);", ""),
    ]) && expressions(function, &[
        ("candidates.iter().find(|candidate| candidate.is_file()).map(PathBuf::as_path).unwrap_or(program)", "program"),
        ("env::var_os(\"PATH\").into_iter().flat_map(|value| env::split_paths(&value).collect::<Vec<_>>()).flat_map(|directory| { candidates.iter().map(move |candidate| directory.join(candidate)) })",
         "env::var_os(\"PATH\").into_iter().flat_map(|value| env::split_paths(&value).collect::<Vec<_>>()).map(|directory| directory.join(program))"),
    ])
}

fn normalize_fingerprint(function: &mut syn::ItemFn) -> bool {
    let expected: syn::ItemFn = syn::parse_quote!(
        fn command_version_fingerprint(program: &Path, version_args: &[&str]) -> String {}
    );
    if tokens(&function.sig) != tokens(&expected.sig) {
        return false;
    }
    function.sig.inputs.pop();
    function.sig.inputs.pop_punct();
    statements(
        function,
        &[(
            r#"let content = fs::read(&resolved)
            .map(|bytes| format!("sha256={:x}", Sha256::digest(bytes)))
            .unwrap_or_else(|_| "sha256=unavailable".to_owned());"#,
            "",
        )],
    ) && expressions(
        function,
        &[
            (
                "std::process::Command::new(&resolved).args(version_args)",
                "std::process::Command::new(&resolved).arg(\"--version\")",
            ),
            (
                r#"format!("program={}:resolved={}:{}:{}:status={:?}:sha256={:x}", program.display(), resolved.display(), metadata, content, output.status.code(), Sha256::digest(&bytes))"#,
                r#"format!("program={}:resolved={}:{}:status={:?}:sha256={:x}", program.display(), resolved.display(), metadata, output.status.code(), Sha256::digest(&bytes))"#,
            ),
            (
                r#"format!("program={}:resolved={}:{}:{}:spawn_error={error}", program.display(), resolved.display(), metadata, content)"#,
                r#"format!("program={}:resolved={}:{}:spawn_error={error}", program.display(), resolved.display(), metadata)"#,
            ),
        ],
    )
}

fn normalize_identity(function: &mut syn::ItemFn) -> bool {
    statements(
        function,
        &[
            (
                r#"let host_tools = HostTools::current();
            let nvcc = resolve_program(&host_tools.nvcc(cuda_root.as_deref()));"#,
                r#"let nvcc = cuda_root.as_ref().map(|root| root.join("bin").join("nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"));"#,
            ),
            (
                r#"let compiler_args: &[&str] = if host_tools == HostTools::WindowsMsvc { &["/Bv"] } else { &["--version"] };
            let archiver_args: &[&str] = if host_tools == HostTools::WindowsMsvc { &["/?"] } else { &["--version"] };"#,
                "",
            ),
            (
                r#"if host_tools == HostTools::WindowsMsvc {
            for key in ["INCLUDE", "LIB", "LIBPATH", "VCToolsInstallDir", "WindowsSdkDir",
                "WindowsSDKVersion", "WindowsSdkVerBinPath", "UniversalCRTSdkDir", "UCRTVersion",] {
                println!("cargo:rerun-if-env-changed={key}");
                lines.push(format!("env.{key}={}", env::var(key).unwrap_or_default()));
            }
        }"#,
                "",
            ),
        ],
    ) && expressions(
        function,
        &[
            (
                "PathBuf::from(host_tools.c_compiler())",
                "PathBuf::from(\"cc\")",
            ),
            (
                "PathBuf::from(host_tools.cpp_compiler())",
                "PathBuf::from(\"c++\")",
            ),
            (
                "command_version_fingerprint(&nvcc, &[\"--version\"])",
                "command_version_fingerprint(&nvcc)",
            ),
            (
                "command_version_fingerprint(&ccbin, compiler_args)",
                "command_version_fingerprint(&ccbin)",
            ),
            (
                "command_version_fingerprint(&cxx, compiler_args)",
                "command_version_fingerprint(&cxx)",
            ),
            (
                "command_version_fingerprint(&PathBuf::from(host_tools.archiver()), archiver_args)",
                "command_version_fingerprint(&PathBuf::from(\"ar\"))",
            ),
        ],
    )
}

const TARGET: &str = r#"let target = env::var("TARGET").expect("TARGET must be set by Cargo");"#;

fn normalize_linking(function: &mut syn::ItemFn) -> bool {
    let resolved = format!(
        r#"{TARGET}
        let resolved_set = NativeOperatorArtifactSetLock::load_and_resolve_for_target(
            &lock_path, Some(&compute_capability), &target)
            .unwrap_or_else(|error| {{ panic!("failed to resolve native operator artifact set {{}}: {{error}}", lock_path.display()) }});"#
    );
    let libraries = format!(
        r#"{TARGET}
        let library_dir = host::cuda_library_directory(cuda_root, &target);
        if library_dir.is_dir() {{ println!("cargo:rustc-link-search=native={{}}", library_dir.display()); }}"#
    );
    if !statements(
        function,
        &[
            (
                &resolved,
                r#"let resolved_set = NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some(&compute_capability))
            .unwrap_or_else(|error| { panic!("failed to resolve native operator artifact set {}: {error}", lock_path.display()) });"#,
            ),
            (
                &libraries,
                r#"let lib64 = cuda_root.join("lib64");
            if lib64.is_dir() { println!("cargo:rustc-link-search=native={}", lib64.display()); }"#,
            ),
        ],
    ) {
        return false;
    }
    struct RuntimeArm {
        expected: String,
        matches: usize,
    }
    impl VisitMut for RuntimeArm {
        fn visit_expr_match_mut(&mut self, value: &mut syn::ExprMatch) {
            value.arms.retain(|arm| {
                if tokens(arm) == self.expected {
                    self.matches += 1;
                    false
                } else {
                    true
                }
            });
            syn::visit_mut::visit_expr_match_mut(self, value);
        }
    }
    let mut expected: syn::ExprMatch = syn::parse_quote!(match library {
        NativeOperatorSystemLibrary::MsvcRuntime => {
            assert_eq!(
                env::var("TARGET").as_deref(),
                Ok("x86_64-pc-windows-msvc"),
                "MSVC runtime libraries require the Windows MSVC target"
            );
            println!("cargo:rustc-link-lib=dylib=msvcrt");
            "msvcprt"
        }
    });
    Punctuation.visit_expr_match_mut(&mut expected);
    let mut visitor = RuntimeArm {
        expected: tokens(&expected.arms[0]),
        matches: 0,
    };
    visitor.visit_block_mut(&mut function.block);
    visitor.matches == 1
}

fn normalize_main(function: &mut syn::ItemFn) -> bool {
    statements(
        function,
        &[(
            r#"
        println!("cargo:rerun-if-changed=build_support/host.rs");
        if env::var_os("CARGO_FEATURE_CUDA").is_some()
            && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            assert_eq!(env::var("TARGET").as_deref(), Ok("x86_64-pc-windows-msvc"),
                "native Windows CUDA requires the x86_64 MSVC target");
            assert!(!env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default()
                .split(',').any(|feature| feature == "crt-static"),
                "native Windows CUDA operators require the dynamic MSVC CRT; remove crt-static");
        }
    "#,
            "",
        )],
    ) && expressions(
        function,
        &[(
            r#"env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")"#,
            r#"env::consts::OS == "macos""#,
        )],
    )
}

fn normalize_link_name(function: &mut syn::ItemFn) -> bool {
    statements(
        function,
        &[(
            r#"
        let target = env::var("TARGET").expect("TARGET must be set by Cargo");
        native_artifact_link_name_for_target(path, NativeOperatorLinkage::Static, &target)
            .unwrap_or_else(|error| panic!("{error}"))
    "#,
            r#"
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_else(|| {
            panic!("native operator artifact has no UTF-8 file name: {}", path.display())
        });
        let Some(stripped) = name.strip_prefix("lib").and_then(|name| name.strip_suffix(".a")) else {
            panic!("static native operator artifact must be named lib<name>.a, got {}", path.display());
        };
        if stripped.is_empty() {
            panic!("static native operator artifact link name is empty: {}", path.display());
        }
        stripped.to_string()
    "#,
        )],
    )
}

// This is a small, closed host-selection contract. In particular, Windows only
// changes executable spelling/CRT discovery; Unix receives no extra NVCC flag.
// Unrecognized methods or side effects require conservative scope again.
const HOST_CONTRACT: &str = r#"
use std::path::{Path, PathBuf};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostTools { Unix, WindowsMsvc }
impl HostTools {
    pub fn current() -> Self {
        if cfg!(windows) { Self::WindowsMsvc } else { Self::Unix }
    }
    pub fn nvcc(self, cuda_root: Option<&Path>) -> PathBuf {
        let name = match self { Self::Unix => "nvcc", Self::WindowsMsvc => "nvcc.exe" };
        cuda_root.map(|root| root.join("bin").join(name)).unwrap_or_else(|| PathBuf::from(name))
    }
    pub fn nvcc_invocation(self, selected: &Path) -> Result<PathBuf, String> {
        if self == Self::Unix { return Ok(selected.to_path_buf()); }
        let selected = selected.to_str().ok_or("Windows NVCC path must be valid Unicode")?.replace('\\', "/");
        let ordinary = if let Some(tail) = selected.strip_prefix("//?/UNC/") {
            format!("//{tail}")
        } else {
            selected.strip_prefix("//?/").unwrap_or(&selected).to_string()
        };
        let bytes = ordinary.as_bytes();
        let drive = bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && &bytes[1..3] == b":/";
        let unc = ordinary.strip_prefix("//").is_some_and(|tail| {
            let mut components = tail.split('/');
            [components.next(), components.next()].iter()
                .all(|part| part.is_some_and(|part| !matches!(part, "" | "." | ".." | "?")))
        });
        if (!drive && !unc) || ordinary.contains(['\0', '\r', '\n'])
            || !ordinary.rsplit('/').next().is_some_and(|name| name.eq_ignore_ascii_case("nvcc.exe")) {
            return Err("Windows NVCC invocation requires an absolute nvcc.exe path".to_string());
        }
        Ok(PathBuf::from(ordinary))
    }
    pub fn c_compiler(self) -> &'static str {
        match self { Self::Unix => "cc", Self::WindowsMsvc => "cl.exe" }
    }
    pub fn cpp_compiler(self) -> &'static str {
        match self { Self::Unix => "c++", Self::WindowsMsvc => "cl.exe" }
    }
    pub fn archiver(self) -> &'static str {
        match self { Self::Unix => "ar", Self::WindowsMsvc => "lib.exe" }
    }
    pub fn nvcc_environment_option(self) -> Option<&'static str> {
        (self == Self::WindowsMsvc).then_some("--use-local-env")
    }
    pub fn executable_candidates(self, program: &Path) -> Vec<PathBuf> {
        let mut candidates = vec![program.to_path_buf()];
        if self == Self::WindowsMsvc && program.extension().is_none() {
            candidates.push(program.with_extension("exe"));
        }
        candidates
    }
}
pub fn cuda_library_directory(root: &Path, target: &str) -> PathBuf {
    if target.ends_with("windows-msvc") { root.join("lib").join("x64") } else { root.join("lib64") }
}
"#;

fn host_contract(source: &str) -> Result<bool, String> {
    fn strip_docs(file: &mut syn::File) {
        file.attrs.retain(|attr| !attr.path().is_ident("doc"));
        for item in &mut file.items {
            match item {
                Item::Enum(item) => item.attrs.retain(|attr| !attr.path().is_ident("doc")),
                Item::Fn(item) => item.attrs.retain(|attr| !attr.path().is_ident("doc")),
                Item::Impl(item) => {
                    item.attrs.retain(|attr| !attr.path().is_ident("doc"));
                    for child in &mut item.items {
                        if let syn::ImplItem::Fn(method) = child {
                            method.attrs.retain(|attr| !attr.path().is_ident("doc"));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let (mut actual, imports) = parse(source)?;
    let (mut expected, expected_imports) = parse(HOST_CONTRACT)?;
    strip_docs(&mut actual);
    strip_docs(&mut expected);
    Ok(imports == expected_imports && tokens(&actual) == tokens(&expected))
}

/// Caller restricts this to kernels/build.rs and obtains `after_host` from the
/// same immutable candidate's build_support/host.rs. A missing or changed host
/// contract cannot be treated as a codegen-preserving portability adjustment.
pub fn core_cuda_build_host_only(
    before: &str,
    after: &str,
    after_host: &str,
) -> Result<bool, String> {
    if !host_contract(after_host)? {
        return Ok(false);
    }
    let (before, before_imports) = parse(before)?;
    let (mut after, mut after_imports) = parse(after)?;
    for added in [
        "host::HostTools",
        "ferrum_native_ops::native_artifact_link_name_for_target",
    ] {
        if before_imports.contains(added) || !after_imports.remove(added) {
            return Ok(false);
        }
    }
    if before_imports != after_imports {
        return Ok(false);
    }
    let expected_host: Item = syn::parse_quote!(
        #[path = "build_support/host.rs"]
        mod host;
    );
    let mut host_modules = 0;
    after.items.retain(|item| {
        if tokens(item) == tokens(&expected_host) {
            host_modules += 1;
            false
        } else {
            true
        }
    });
    if host_modules != 1 {
        return Ok(false);
    }
    // These complete functions are restored through precise local changes;
    // none of their bodies is erased from the comparison.
    for (name, normalize) in [
        (
            "compile_core_ptx",
            normalize_core as fn(&mut syn::ItemFn) -> bool,
        ),
        ("resolve_program", normalize_resolver),
        ("command_version_fingerprint", normalize_fingerprint),
        ("cuda_native_toolchain_identity", normalize_identity),
        ("native_static_link_name", normalize_link_name),
        ("link_native_operator_artifact_set", normalize_linking),
        ("main", normalize_main),
    ] {
        let Some(function) = function(&mut after, name) else {
            return Ok(false);
        };
        if !normalize(function) {
            return Ok(false);
        }
    }
    for name in ["CORE_PTX_KERNELS", "CORE_PTX_HEADERS"] {
        if before
            .items
            .iter()
            .filter(|item| matches!(item, Item::Const(item) if item.ident == name))
            .count()
            != 1
        {
            return Ok(false);
        }
    }
    Ok(tokens(&before) == tokens(&after))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_function(source: &str) -> syn::ItemFn {
        let mut function = syn::parse_str(source).unwrap();
        Punctuation.visit_item_fn_mut(&mut function);
        function
    }

    fn core_fixture() -> (syn::ItemFn, syn::ItemFn) {
        let before = parsed_function(
            r#"
            fn compile_core_ptx(out_dir: &Path, native_build_cache: Option<&CudaNativeBuildCache>) {
                let cuda_root = cuda_root_from_env();
                let nvcc = cuda_root.as_ref().map(|r| r.join("bin").join("nvcc"))
                    .unwrap_or_else(|| PathBuf::from("nvcc"));
                let compute_cap = detect_cuda_compute_cap();
                let mut flags = vec!["-O3".to_string(), "--use_fast_math".to_string()];
                for kernel in CORE_PTX_KERNELS {
                    let signature = core_ptx_signature(kernel, &flags);
                    let mut command = std::process::Command::new(&nvcc);
                    command.arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx").arg("-O3").arg("--use_fast_math");
                    if let Some(ccbin) = &ccbin {
                        command.arg("-allow-unsupported-compiler").args(["-ccbin", ccbin]);
                    }
                    command.arg(kernel);
                    let output = command.output().unwrap();
                    assert!(output.status.success());
                    write_core_ptx_stamp(out_dir, kernel, &signature);
                }
            }
        "#,
        );
        let after = parsed_function(&format!(
            r#"
            fn compile_core_ptx(out_dir: &Path, native_build_cache: Option<&CudaNativeBuildCache>) {{
                let cuda_root = cuda_root_from_env();
                {CORE_SELECTION}
                let compute_cap = detect_cuda_compute_cap();
                let environment_option = HostTools::current().nvcc_environment_option();
                let mut flags = vec!["-O3".to_string(), "--use_fast_math".to_string()];
                flags.extend(environment_option.map(str::to_string));
                for kernel in CORE_PTX_KERNELS {{
                    let signature = core_ptx_signature(kernel, &flags);
                    assert_eq!(invocation_identity(), selected_nvcc_identity,
                        "[core-ptx] NVCC changed before compilation");
                    let mut command = std::process::Command::new(&nvcc);
                    command.arg(format!("--gpu-architecture=sm_{{compute_cap}}"))
                        .arg("--ptx").arg("-O3").arg("--use_fast_math");
                    command.args(environment_option);
                    if let Some(ccbin) = &ccbin {{
                        if HostTools::current() == HostTools::Unix {{ command.arg("-allow-unsupported-compiler"); }}
                        command.args(["-ccbin", ccbin]);
                    }}
                    command.arg(kernel);
                    let output = command.output().unwrap();
                    assert!(output.status.success());
                    assert_eq!(invocation_identity(), selected_nvcc_identity,
                        "[core-ptx] NVCC changed during compilation");
                    write_core_ptx_stamp(out_dir, kernel, &signature);
                }}
            }}
        "#
        ));
        (before, after)
    }

    #[test]
    fn device_arguments_and_execution_are_retained_around_host_adjustments() {
        let (before, after) = core_fixture();
        let mut normalized = after.clone();
        assert!(normalize_core(&mut normalized));
        assert_eq!(tokens(&before), tokens(&normalized));
        let original = tokens(&after);
        for changed in [
            original.replace("--use_fast_math", "--fmad=false"),
            original.replace(
                "--gpu-architecture=sm_{compute_cap}",
                "--gpu-architecture=sm_90",
            ),
            original.replace("command . arg (kernel)", "command . arg (other_kernel)"),
            original.replace(
                "command . output ()",
                "command . arg (\"--ftz=true\") . output ()",
            ),
            original.replace("HostTools :: Unix", "HostTools :: WindowsMsvc"),
            original.replace("let compute_cap", "modify_cuda_inputs(); let compute_cap"),
            original.replace("selected_nvcc_identity ,", "invocation_identity() ,"),
        ] {
            assert_ne!(
                changed, original,
                "fixture mutation must alter actual syntax"
            );
            let mut changed = parsed_function(&changed);
            assert!(!normalize_core(&mut changed) || tokens(&changed) != tokens(&before));
        }
    }

    #[test]
    fn host_selection_requires_unchanged_unix_flags_and_executable_identity() {
        assert!(host_contract(HOST_CONTRACT).unwrap());
        for changed in [
            HOST_CONTRACT.replace(
                "then_some(\"--use-local-env\")",
                "then_some(\"--use_fast_math\")",
            ),
            HOST_CONTRACT.replace(
                "(self == Self::WindowsMsvc).then_some",
                "(self == Self::Unix).then_some",
            ),
            HOST_CONTRACT.replace("Self::Unix => \"nvcc\"", "Self::Unix => \"other-nvcc\""),
            HOST_CONTRACT.replace(
                "return Ok(selected.to_path_buf());",
                "return Ok(PathBuf::from(\"other-nvcc\"));",
            ),
            HOST_CONTRACT.replace(
                "let name = match self",
                "std::env::set_var(\"NVCC_APPEND_FLAGS\", \"--ftz=true\"); let name = match self",
            ),
            format!("{HOST_CONTRACT}\nfn hidden_compile() {{ modify_source(); }}"),
            HOST_CONTRACT.replace("pub enum HostTools", "#[cfg(unix)] pub enum HostTools"),
        ] {
            assert!(!host_contract(&changed).unwrap(), "{changed}");
        }
    }

    #[test]
    fn program_resolution_retains_path_order_and_rejects_an_extra_candidate() {
        let before = parsed_function(
            r#"
            fn resolve_program(program: &Path) -> PathBuf {
                if program.components().count() > 1 {
                    return program.canonicalize().unwrap_or_else(|_| program.to_path_buf());
                }
                env::var_os("PATH").into_iter()
                    .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
                    .map(|directory| directory.join(program))
                    .find(|candidate| candidate.is_file())
                    .and_then(|candidate| candidate.canonicalize().ok())
                    .unwrap_or_else(|| program.to_path_buf())
            }
        "#,
        );
        let after = parsed_function(
            r#"
            fn resolve_program(program: &Path) -> PathBuf {
                let candidates = HostTools::current().executable_candidates(program);
                if program.components().count() > 1 {
                    return candidates.iter().find(|candidate| candidate.is_file())
                        .map(PathBuf::as_path).unwrap_or(program)
                        .canonicalize().unwrap_or_else(|_| program.to_path_buf());
                }
                env::var_os("PATH").into_iter()
                    .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
                    .flat_map(|directory| { candidates.iter().map(move |candidate| directory.join(candidate)) })
                    .find(|candidate| candidate.is_file())
                    .and_then(|candidate| candidate.canonicalize().ok())
                    .unwrap_or_else(|| program.to_path_buf())
            }
        "#,
        );
        let mut normalized = after.clone();
        assert!(normalize_resolver(&mut normalized));
        assert_eq!(tokens(&before), tokens(&normalized));
        let changed = tokens(&after).replace(
            "candidates . iter () . map",
            "candidates . iter () . rev () . map",
        );
        assert_ne!(changed, tokens(&after));
        let mut changed = parsed_function(&changed);
        assert!(!normalize_resolver(&mut changed));
    }
}
