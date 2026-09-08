//! NVCC's Windows make dependencies include drive colons and native separators.

use super::*;

pub(super) fn is_windows_rule(raw: &str) -> bool {
    let target = raw.trim_start();
    platform::windows_path(target)
        || (target.len() >= 4
            && target.as_bytes()[0].is_ascii_alphabetic()
            && target.as_bytes()[1..3] == *b"\\:")
}

pub(super) fn parse_windows_rule(raw: &str, path: &Path) -> Result<(String, Vec<String>)> {
    parse_windows_rule_with_escaping(raw, path, false)
}

pub(super) fn parse_windows_portable_rule(raw: &str, path: &Path) -> Result<(String, Vec<String>)> {
    parse_windows_rule_with_escaping(raw, path, true)
}

fn parse_windows_rule_with_escaping(
    raw: &str,
    path: &Path,
    portable: bool,
) -> Result<(String, Vec<String>)> {
    let invalid =
        |why: &str| NativeOperatorBuilderError::Invalid(format!("{why}: {}", path.display()));
    if raw.len() > MAX_DEPFILE_BYTES || raw.trim().is_empty() || raw.as_bytes().contains(&0) {
        return Err(invalid("invalid Windows compiler depfile size/content"));
    }
    let normalized = raw.replace("\\\r\n", "").replace("\\\n", "");
    let normalized = normalized.trim_start().trim_end_matches(['\r', '\n']);
    if normalized.contains(['\r', '\n']) {
        return Err(invalid("depfile contains multiple rules"));
    }
    let bytes = normalized.as_bytes();
    let mut delimiter = None;
    for (index, character) in normalized.char_indices() {
        if character != ':' {
            continue;
        }
        let preceding_slashes = bytes[..index]
            .iter()
            .rev()
            .take_while(|&&byte| byte == b'\\')
            .count();
        if preceding_slashes % 2 == 1 {
            continue;
        }
        let drive = index > 0
            && bytes[index - 1].is_ascii_alphabetic()
            && bytes
                .get(index + 1)
                .is_some_and(|byte| matches!(byte, b'/' | b'\\'))
            && (index == 1 || bytes[..index - 1].ends_with(b"\\\\?\\"));
        if !drive {
            delimiter = Some(index);
            break;
        }
    }
    let delimiter = delimiter.ok_or_else(|| invalid("depfile has no target delimiter"))?;
    let target_spelling = &normalized[..delimiter];
    // Native UNC begins with two separators; make-escaped UNC begins with four.
    // A forward-slash UNC target contains no escape marker, so callers parsing our
    // canonical portable representation select make escaping explicitly.
    let escaped_target = portable
        || target_spelling.starts_with(r"\\\\")
        || (target_spelling
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
            && target_spelling.as_bytes().get(1) == Some(&b'\\'));
    let target = words(&normalized[..delimiter], path, 1, escaped_target)?;
    let dependencies = words(
        &normalized[delimiter + 1..],
        path,
        MAX_DEPFILE_DEPENDENCIES,
        escaped_target,
    )?;
    if target.len() != 1 || dependencies.is_empty() {
        return Err(invalid("invalid depfile target/dependency count"));
    }
    Ok((target[0].clone(), dependencies))
}

fn words(value: &str, path: &Path, limit: usize, escaped: bool) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = value.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\\' => match chars.peek().copied() {
                Some(next)
                    if matches!(next, ' ' | '\t' | ':' | '#' | '$')
                        || (escaped && next == '\\') =>
                {
                    word.push(chars.next().expect("peeked escape"));
                }
                Some(_) => word.push('\\'),
                None => {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "incomplete depfile escape: {}",
                        path.display()
                    )))
                }
            },
            c if c.is_whitespace() => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
        if word.len() > MAX_DEPFILE_WORD_BYTES || words.len() > limit {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "depfile word/count exceeds its limit: {}",
                path.display()
            )));
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    if words.len() > limit {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile exceeds word-count limit: {}",
            path.display()
        )));
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvcc_windows_drive_paths_and_continuations_are_dependencies() {
        let raw = "C:\\build\\a.obj: src/kernel.cu \\\r\n C:\\Program\\ Files\\CUDA\\include\\cuda.h C:\\SDK\\new\\type.h\r\n";
        let (target, dependencies) = parse_make_depfile(raw, Path::new("raw.d")).unwrap();
        assert_eq!(target, r"C:\build\a.obj");
        assert_eq!(
            dependencies,
            [
                "src/kernel.cu",
                r"C:\Program Files\CUDA\include\cuda.h",
                r"C:\SDK\new\type.h"
            ]
        );
    }

    #[test]
    fn nvcc_mt_encodes_one_target_without_changing_native_header_paths() {
        // NVCC writes -MT literally but escapes spaces in its header paths.
        // This is the reduced shape of its actual Windows compiler output.
        let dependencies = [
            "kernels/marlin.cu",
            "C:/Program Files/CUDA/bin/../include/cuda_runtime.h",
            "C:/Program Files/Visual Studio/include/vcruntime.h",
        ];
        let rhs = "kernels/marlin.cu \\\r\n    C:/Program\\ Files/CUDA/bin/../include/cuda_runtime.h \\\r\n    C:/Program\\ Files/Visual\\ Studio/include/vcruntime.h\r\n";
        for target in [
            r"C:\build area\对象 #$.obj",
            r"\\?\C:\build area\对象 #$.obj",
        ] {
            assert!(parse_make_depfile(&format!("{target} : {rhs}"), Path::new("raw.d")).is_err());
            let encoded = platform::nvcc_dependency_target(target, true).unwrap();
            let raw = format!("{encoded} : {rhs}");
            let (parsed_target, parsed_dependencies) =
                parse_make_depfile(&raw, Path::new("raw.d")).unwrap();
            assert_eq!(parsed_target, target);
            assert_eq!(parsed_dependencies, dependencies);
            let portable_dependencies = vec![
                "kernels/marlin.cu".to_string(),
                "C:/Program Files/CUDA/include/cuda_runtime.h".to_string(),
                "C:/Program Files/Visual Studio/include/vcruntime.h".to_string(),
            ];
            let portable = serialize_portable_depfile(target, &portable_dependencies).unwrap();
            assert_eq!(
                parse_portable_make_depfile(
                    std::str::from_utf8(&portable).unwrap(),
                    Path::new("portable.d")
                )
                .unwrap(),
                (target.to_string(), portable_dependencies)
            );
        }
    }

    #[test]
    fn nvcc_mt_preserves_native_unc_dependency_roots() {
        let dependency = r"\\sdk\share\include\cuda.h";
        for target in [
            r"\\build\share area\unit #$.obj",
            r"\\?\UNC\build\share area\unit #$.obj",
            "//build/share area/unit #$.obj",
        ] {
            let encoded = platform::nvcc_dependency_target(target, true).unwrap();
            let raw = format!("{encoded} : kernel.cu \\\r\n    {dependency}\r\n");
            let parsed = parse_make_depfile(&raw, Path::new("native.d")).unwrap();
            assert_eq!(
                parsed,
                (
                    target.to_string(),
                    vec!["kernel.cu".to_string(), dependency.to_string()]
                )
            );
            let portable = serialize_portable_depfile(&parsed.0, &parsed.1).unwrap();
            assert_eq!(
                parse_portable_make_depfile(
                    std::str::from_utf8(&portable).unwrap(),
                    Path::new("portable.d")
                )
                .unwrap(),
                parsed
            );
        }
    }

    #[test]
    fn windows_portable_depfile_round_trips_make_escaped_drive_and_slashes() {
        let target = r"C:\build space\a.obj";
        let dependencies = vec![
            "kernel.cu".to_string(),
            r"C:\SDK\include\test.h".to_string(),
        ];
        let bytes = serialize_portable_depfile(target, &dependencies).unwrap();
        assert_eq!(
            parse_make_depfile(
                std::str::from_utf8(&bytes).unwrap(),
                Path::new("portable.d")
            )
            .unwrap(),
            (target.to_string(), dependencies)
        );
    }

    #[test]
    fn native_unc_dependencies_keep_the_network_root_and_round_trip() {
        let dependencies = vec![
            "src/kernel.cu".to_string(),
            r"\\sdk\share\include\cuda.h".to_string(),
        ];
        for target in [
            r"\\build\share\a.obj",
            r"\\?\UNC\build\share\a.obj",
            "//build/share/a.obj",
        ] {
            let raw = format!("{target}: src/kernel.cu \\\r\n {}\r\n", dependencies[1]);
            assert_eq!(
                parse_make_depfile(&raw, Path::new("native.d")).unwrap(),
                (target.to_string(), dependencies.clone())
            );
            let portable = serialize_portable_depfile(target, &dependencies).unwrap();
            assert_eq!(
                parse_portable_make_depfile(
                    std::str::from_utf8(&portable).unwrap(),
                    Path::new("portable.d")
                )
                .unwrap(),
                (target.to_string(), dependencies.clone())
            );
        }
    }

    #[test]
    fn unc_source_bindings_cannot_change_the_server_or_share() {
        let directory = r"\\build\share\source";
        assert_eq!(
            platform::source_relative_path(r"\\?\UNC\build\share\source\kernels\a.cu", directory)
                .unwrap(),
            "kernels/a.cu"
        );
        for path in [
            r"\\other\share\source\kernels\a.cu",
            r"\\build\other\source\kernels\a.cu",
            r"\\build\share\source-other\kernels\a.cu",
        ] {
            assert!(
                platform::source_relative_path(path, directory).is_err(),
                "{path}"
            );
        }
    }

    #[test]
    fn windows_depfile_rejects_missing_terminal_rule_and_extra_rules() {
        for raw in [
            r"C:\build\a.obj C:\SDK\cuda.h",
            "C:/a.obj: a.cu\nC:/b.obj: b.cu",
            "C:/a.obj:",
        ] {
            assert!(
                parse_make_depfile(raw, Path::new("bad.d")).is_err(),
                "{raw}"
            );
        }
    }
}
