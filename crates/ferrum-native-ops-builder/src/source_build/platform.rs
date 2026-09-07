//! Host-specific command and environment policy for locked native builds.

use super::*;

pub(crate) mod msvc;

pub(crate) const MSVC_TARGET: &str = "x86_64-pc-windows-msvc";
const MSVC_ENVIRONMENT_KEYS: &[&str] = &[
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "WindowsSdkVerBinPath",
];

pub(crate) fn is_msvc(abi: Option<&NativeOperatorHostAbi>) -> bool {
    abi.is_some_and(|abi| abi.compiler_flavor == NativeOperatorCompilerFlavor::Msvc)
}

pub(crate) fn nvcc_environment_option(msvc: bool) -> Option<&'static str> {
    msvc.then_some("--use-local-env")
}

pub(crate) fn is_msvc_compiler(path: &str) -> bool {
    basename(path).eq_ignore_ascii_case("cl.exe")
}

pub(crate) fn nvcc_ccbin_argument(path: &str, msvc: bool) -> Result<String> {
    if !msvc {
        return Ok(path.to_string());
    }
    if !is_msvc_compiler(path) {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC NVCC -ccbin must select cl.exe".to_string(),
        ));
    }
    // NVCC compares this spelling with the ordinary paths in its controlled
    // PATH. Preserve the canonical identity separately and verify this alias
    // against that physical file before executing a source build.
    normalize_windows_path(path)
}

pub(crate) fn validate_nvcc_ccbin_identity(
    compiler: &NativeOperatorToolFileIdentity,
) -> Result<()> {
    let argument = nvcc_ccbin_argument(&compiler.path, true)?;
    let actual = tool_file_identity(Path::new(&argument))?;
    if &actual != compiler {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "NVCC -ccbin invocation does not resolve to its recorded compiler identity: invocation={argument} recorded={}",
            compiler.path
        )));
    }
    Ok(())
}

pub(crate) fn nvcc_program(path: &str, msvc: bool) -> Result<String> {
    if !msvc {
        return Ok(path.to_string());
    }
    if !basename(path).eq_ignore_ascii_case("nvcc.exe") {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC CUDA compilation must select nvcc.exe".to_string(),
        ));
    }
    // NVCC locates nvcc.profile relative to argv[0]. Its Windows profile
    // discovery does not support the verbatim spelling returned by canonicalize.
    // Keep this projection separate from the recorded physical file identity.
    normalize_windows_path(path)
}

pub(crate) fn validate_nvcc_program_identity(nvcc: &NativeOperatorToolFileIdentity) -> Result<()> {
    let program = nvcc_program(&nvcc.path, true)?;
    let actual = tool_file_identity(Path::new(&program))?;
    if &actual != nvcc {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "NVCC invocation does not resolve to its recorded executable identity: invocation={program} recorded={}",
            nvcc.path
        )));
    }
    Ok(())
}

pub(crate) fn validate_host_contract(
    abi: Option<&NativeOperatorHostAbi>,
    target: &str,
    compiler: &str,
    environment: &BTreeMap<String, String>,
) -> Result<()> {
    if let Some(abi) = abi {
        abi.validate()
            .map_err(NativeOperatorBuilderError::Invalid)?;
        if abi.target != target {
            return Err(NativeOperatorBuilderError::Invalid(
                "host ABI target differs from compiler target".to_string(),
            ));
        }
    }
    if is_msvc(abi) {
        if !is_msvc_compiler(compiler) {
            return Err(NativeOperatorBuilderError::Invalid(
                "MSVC host ABI does not bind cl.exe".to_string(),
            ));
        }
        validate_msvc_environment(environment)?;
    } else if !environment.is_empty()
        || is_msvc_compiler(compiler)
        || target.contains("windows-msvc")
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "Windows MSVC compiler requires an explicit host ABI and environment".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

pub(crate) fn parent(path: &str) -> Option<&str> {
    path.rfind(['/', '\\']).map(|index| &path[..index])
}

pub(crate) fn archive_file(plan_name: &str, msvc: bool) -> String {
    if msvc {
        format!(
            "{}.lib",
            plan_name
                .strip_prefix("lib")
                .unwrap_or(plan_name)
                .trim_end_matches(".a")
        )
    } else {
        plan_name.to_string()
    }
}

pub(crate) fn archive_argv(
    archiver: &str,
    archive: &str,
    objects: impl IntoIterator<Item = String>,
    msvc: bool,
) -> Vec<String> {
    let mut argv = if msvc {
        vec![
            archiver.to_string(),
            "/NOLOGO".to_string(),
            format!("/OUT:{archive}"),
        ]
    } else {
        vec![archiver.to_string(), "rcs".to_string(), archive.to_string()]
    };
    argv.extend(objects);
    argv
}

pub(crate) fn capture_msvc_environment() -> Result<BTreeMap<String, String>> {
    let mut environment = BTreeMap::new();
    for key in MSVC_ENVIRONMENT_KEYS {
        let value = std::env::var(key).map_err(|_| NativeOperatorBuilderError::Invalid(format!(
            "MSVC build requires {key}; invoke the builder from the selected x64 VS/SDK developer environment"
        )))?;
        environment.insert((*key).to_string(), value);
    }
    validate_msvc_environment(&environment)?;
    Ok(environment)
}

pub(crate) fn validate_msvc_environment(environment: &BTreeMap<String, String>) -> Result<()> {
    if environment.len() != MSVC_ENVIRONMENT_KEYS.len()
        || MSVC_ENVIRONMENT_KEYS
            .iter()
            .any(|key| !environment.contains_key(*key))
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC environment has missing or undeclared keys".to_string(),
        ));
    }
    for (key, value) in environment {
        if value.is_empty() || value.chars().any(|c| matches!(c, '\0' | '\n' | '\r' | '"')) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "invalid MSVC environment value: {key}"
            )));
        }
        let values: Vec<&str> = if matches!(key.as_str(), "INCLUDE" | "LIB" | "LIBPATH") {
            value.split(';').filter(|part| !part.is_empty()).collect()
        } else {
            vec![value.as_str()]
        };
        if values.is_empty()
            || values
                .iter()
                .any(|path| normalize_windows_path(path).is_err())
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "MSVC {key} must contain absolute Windows paths"
            )));
        }
    }
    Ok(())
}

pub(crate) fn msvc_environment_for_tools(
    tool_paths: [&str; 3],
    recorded: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    if !basename(tool_paths[0]).eq_ignore_ascii_case("nvcc.exe") {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC CUDA build requires nvcc.exe".to_string(),
        ));
    }
    msvc_environment(&tool_paths, tool_paths[1], tool_paths[2], recorded)
}

pub(crate) fn msvc_environment_for_package_tools(
    tool_paths: [&str; 2],
    recorded: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    msvc_environment(&tool_paths, tool_paths[0], tool_paths[1], recorded)
}

fn msvc_environment(
    tool_paths: &[&str],
    compiler: &str,
    archiver: &str,
    recorded: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    validate_msvc_environment(recorded)?;
    if !is_msvc_compiler(compiler) || !basename(archiver).eq_ignore_ascii_case("lib.exe") {
        return Err(NativeOperatorBuilderError::Invalid(
            "MSVC build requires cl.exe and lib.exe".to_string(),
        ));
    }
    let compiler_parent = parent(compiler)
        .ok_or_else(|| NativeOperatorBuilderError::Invalid("cl.exe has no parent".to_string()))?;
    let archiver_parent = parent(archiver).unwrap_or("");
    if normalize_windows_path(compiler_parent)? != normalize_windows_path(archiver_parent)? {
        return Err(NativeOperatorBuilderError::Invalid(
            "cl.exe and lib.exe must come from the same selected MSVC tool directory".to_string(),
        ));
    }
    let mut paths = tool_paths
        .iter()
        .map(|path| {
            parent(path)
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid("MSVC tool has no parent".to_string())
                })
                .and_then(normalize_windows_path)
        })
        .collect::<Result<Vec<_>>>()?;
    let sdk = normalize_windows_path(&recorded["WindowsSdkVerBinPath"])?;
    paths.push(format!("{}/x64", sdk.trim_end_matches('/')));
    paths.push(format!(
        "{}/System32",
        normalize_windows_path(&recorded["SystemRoot"])?
    ));
    paths.sort();
    paths.dedup();
    let mut environment = recorded.clone();
    environment.insert("PATH".to_string(), paths.join(";"));
    environment.insert("VSLANG".to_string(), "1033".to_string());
    environment.insert("SOURCE_DATE_EPOCH".to_string(), "0".to_string());
    environment.insert("TZ".to_string(), "UTC".to_string());
    Ok(environment)
}

pub(crate) fn windows_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    (bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\'))
        || value.starts_with("\\\\")
        || value.starts_with("//")
}

/// Canonical comparison form only; actual argv retains the spelling passed to the compiler.
pub(crate) fn normalize_windows_path(value: &str) -> Result<String> {
    let invalid =
        || NativeOperatorBuilderError::Invalid(format!("invalid absolute Windows path: {value}"));
    if value.chars().any(|c| matches!(c, '\0' | '\n' | '\r')) {
        return Err(invalid());
    }
    let raw = value.replace('\\', "/");
    let raw = raw
        .strip_prefix("//?/UNC/")
        .map(|tail| format!("//{tail}"))
        .unwrap_or_else(|| raw.strip_prefix("//?/").unwrap_or(&raw).to_string());
    let (prefix, tail) =
        if raw.len() >= 3 && raw.as_bytes()[0].is_ascii_alphabetic() && &raw[1..3] == ":/" {
            (raw[..2].to_ascii_uppercase(), &raw[3..])
        } else if let Some(tail) = raw.strip_prefix("//") {
            let mut parts = tail.splitn(3, '/');
            let server = parts.next().unwrap_or("");
            let share = parts.next().unwrap_or("");
            if server.is_empty()
                || share.is_empty()
                || matches!(server, "." | "..")
                || matches!(share, "." | "..")
            {
                return Err(invalid());
            }
            (format!("//{server}/{share}"), parts.next().unwrap_or(""))
        } else {
            return Err(invalid());
        };
    let mut parts = Vec::new();
    for part in tail.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(invalid());
                }
            }
            _ if part.contains(':') => return Err(invalid()),
            _ => parts.push(part),
        }
    }
    Ok(if parts.is_empty() {
        format!("{prefix}/")
    } else {
        format!("{prefix}/{}", parts.join("/"))
    })
}

pub(crate) fn comparison_path(value: &str) -> Result<String> {
    if windows_path(value) {
        normalize_windows_path(value)
    } else {
        normalize_absolute_posix_path_lexically(value, "native path")
    }
}

pub(crate) fn same_path(left: &str, right: &str) -> bool {
    match (comparison_path(left), comparison_path(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub(crate) fn path_is_within(path: &str, root: &str) -> bool {
    match (comparison_path(path), comparison_path(root)) {
        (Ok(path), Ok(root)) => {
            path == root
                || path
                    .strip_prefix(root.trim_end_matches('/'))
                    .is_some_and(|suffix| suffix.starts_with('/'))
        }
        _ => false,
    }
}

pub(super) fn source_relative_path(producer: &str, working_directory: &str) -> Result<String> {
    if windows_path(working_directory) {
        let producer = if windows_path(producer) {
            let producer = normalize_windows_path(producer)?;
            let root = normalize_windows_path(working_directory)?;
            producer
                .strip_prefix(&format!("{}/", root.trim_end_matches('/')))
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "source compiler depfile path escapes its working directory: {producer}"
                    ))
                })?
                .to_string()
        } else {
            if producer.starts_with(['/', '\\']) || producer.contains(':') {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "ambiguous source depfile path: {producer}"
                )));
            }
            producer.replace('\\', "/")
        };
        return normalize_portable_relative_path(Path::new(&producer));
    }
    let producer = Path::new(producer);
    let relative = if producer.is_absolute() {
        producer.strip_prefix(working_directory).map_err(|_| {
            NativeOperatorBuilderError::Invalid(format!(
                "source compiler depfile path escapes its working directory: {}",
                producer.display()
            ))
        })?
    } else {
        producer
    };
    normalize_portable_relative_path(relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_paths_preserve_drive_unc_and_boundary() {
        assert_eq!(
            normalize_windows_path(r"\\?\C:\Program Files\CUDA\include\..\bin").unwrap(),
            "C:/Program Files/CUDA/bin"
        );
        assert_eq!(
            normalize_windows_path(r"\\?\UNC\server\share\SDK").unwrap(),
            "//server/share/SDK"
        );
        assert!(same_path(
            r"\\?\C:\CUDA\include\cuda.h",
            "C:/CUDA/include/cuda.h"
        ));
        assert!(!path_is_within("C:/CUDA-other/cuda.h", "C:/CUDA"));
        assert!(normalize_windows_path("C:relative").is_err());
        assert!(normalize_windows_path("C:/../escape").is_err());
        assert!(normalize_windows_path("C:/file:stream").is_err());
    }

    #[test]
    fn archive_commands_have_distinct_real_tool_contracts() {
        assert_eq!(archive_file("libmarlin.a", true), "marlin.lib");
        assert_eq!(
            archive_argv(
                "lib.exe",
                "C:/out/marlin.lib",
                ["C:/out/a.obj".into()],
                true
            ),
            [
                "lib.exe",
                "/NOLOGO",
                "/OUT:C:/out/marlin.lib",
                "C:/out/a.obj"
            ]
        );
        assert_eq!(
            archive_argv("ar", "/out/libmarlin.a", ["/out/a.o".into()], false),
            ["ar", "rcs", "/out/libmarlin.a", "/out/a.o"]
        );
    }
}
