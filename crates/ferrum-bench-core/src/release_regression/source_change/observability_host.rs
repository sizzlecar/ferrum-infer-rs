//! Closed host-observability deltas. Retain sink and OS contract obligations;
//! these proofs do not establish runtime correctness or discard function bodies.
use super::test_configuration;
use quote::ToTokens;
use syn::{Attribute, Item, Stmt};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

fn parse(source: &str) -> Result<syn::File, String> {
    let mut file = syn::parse_file(source).map_err(|error| format!("Rust scope parse: {error}"))?;
    file.items.retain(
        |item| !matches!(item, Item::Mod(module) if module.attrs.iter().any(test_configuration)),
    );
    Ok(file)
}

fn statements(source: &str) -> Vec<Stmt> {
    syn::parse_str::<syn::Block>(&format!("{{ {source} }}"))
        .expect("fixed journal initialization contract")
        .stmts
}

const JOURNAL_BEFORE: &str = r#"
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    let file = options.open(&path)?;
    if mode == JsonlJournalOpenMode::Truncate {
        file.set_len(0)?;
    }
"#;
const JOURNAL_AFTER: &str = r#"
    if mode == JsonlJournalOpenMode::Truncate {
        drop(File::create(&path)?);
    }
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
"#;

/// Only restore the exact initialization statements inside open_shared_inner.
/// All ownership checks, queue setup, worker behavior, schemas and error paths
/// remain in the complete file comparison, including their order around opening.
pub fn journal_initialization_only(before: &str, after: &str) -> Result<bool, String> {
    let before = parse(before)?;
    let mut after = parse(after)?;
    let functions: Vec<_> = after
        .items
        .iter_mut()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "open_shared_inner" => Some(function),
            _ => None,
        })
        .collect();
    let [function] = functions.as_slice() else {
        return Ok(false);
    };
    let expected = statements(JOURNAL_AFTER);
    let positions: Vec<_> = function
        .block
        .stmts
        .windows(expected.len())
        .enumerate()
        .filter_map(|(index, window)| {
            window
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| tokens(actual) == tokens(expected))
                .then_some(index)
        })
        .collect();
    let [index] = positions.as_slice() else {
        return Ok(false);
    };
    // Recover the unique mutable function after inspecting the borrowed window.
    let function = functions.into_iter().next().expect("one matching function");
    function
        .block
        .stmts
        .splice(*index..*index + expected.len(), statements(JOURNAL_BEFORE));
    Ok(tokens(&before) == tokens(&after))
}

const WINDOWS_SAMPLE: &str = r#"
    #[cfg(windows)]
    pub fn sample_process_memory() -> Option<ProcessMemorySample> {
        use windows_sys::Win32::System::{
            ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::GetCurrentProcess,
        };
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        let succeeded =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if succeeded == 0 {
            return None;
        }
        let current_bytes = counters.WorkingSetSize as u64;
        Some(ProcessMemorySample {
            current_bytes,
            high_water_bytes: (counters.PeakWorkingSetSize as u64).max(current_bytes),
            source: "windows_process_memory_counters",
        })
    }
"#;

fn restore_cfg(
    file: &mut syn::File,
    name: &str,
    expected: Attribute,
    replacement: Option<Attribute>,
) -> bool {
    let mut matches = 0;
    for item in &mut file.items {
        let Item::Fn(function) = item else {
            continue;
        };
        if function.sig.ident != name {
            continue;
        }
        let mut restored = Vec::new();
        for attribute in std::mem::take(&mut function.attrs) {
            if tokens(&attribute) == tokens(&expected) {
                matches += 1;
                restored.extend(replacement.clone());
            } else {
                restored.push(attribute);
            }
        }
        function.attrs = restored;
    }
    matches == 1
}

/// Recognize the complete Windows resident-memory implementation, then restore
/// only its exact fallback cfg adjustments. Every Unix function and shared
/// observation/snapshot field remains unchanged in the final AST comparison.
pub fn windows_process_memory_only(before: &str, after: &str) -> Result<bool, String> {
    let before = parse(before)?;
    let mut after = parse(after)?;
    let expected: Item = syn::parse_str(WINDOWS_SAMPLE).expect("fixed Windows RSS contract");
    let mut windows_samples = 0;
    after.items.retain(|item| {
        if tokens(item) == tokens(&expected) {
            windows_samples += 1;
            false
        } else {
            true
        }
    });
    if windows_samples != 1
        || !restore_cfg(
            &mut after,
            "current_resident_bytes",
            syn::parse_quote!(#[cfg(all(not(windows), not(target_os = "linux")))]),
            Some(syn::parse_quote!(#[cfg(not(target_os = "linux"))])),
        )
        || !restore_cfg(
            &mut after,
            "high_water_bytes",
            syn::parse_quote!(#[cfg(not(any(unix, windows)))]),
            Some(syn::parse_quote!(#[cfg(not(unix))])),
        )
        || !restore_cfg(
            &mut after,
            "sample_process_memory",
            syn::parse_quote!(#[cfg(not(windows))]),
            None,
        )
        || !restore_cfg(
            &mut after,
            "process_memory_source",
            syn::parse_quote!(#[cfg(not(any(unix, windows)))]),
            Some(syn::parse_quote!(#[cfg(not(unix))])),
        )
    {
        return Ok(false);
    }
    Ok(tokens(&before) == tokens(&after))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn journal(open: &str) -> String {
        format!(
            r#"
            struct Record {{ text: String }}
            fn open_shared_inner(path: PathBuf, mode: JsonlJournalOpenMode, config: Config) -> io::Result<()> {{
                if reuse_live_writer(&path)? {{ return Ok(()); }}
                {open}
                let (sender, receiver) = sync_channel(config.queue_capacity);
                encode_record(&Record {{ text: String::new() }})?;
                file.sync_all()?;
                flush(receiver)?;
                Ok(())
            }}
            "#
        )
    }

    #[test]
    fn journal_accepts_only_the_truncate_then_append_initialization() {
        let before = journal(JOURNAL_BEFORE);
        let after = journal(JOURNAL_AFTER);
        assert!(journal_initialization_only(&before, &after).unwrap());
        let with_tests = format!(
            "{after}\n#[cfg(test)] mod tests {{ fn external_append_and_live_owner() {{}} }}"
        );
        assert!(journal_initialization_only(&before, &with_tests).unwrap());
        assert!(!journal_initialization_only(&before, &before).unwrap());
        assert!(!journal_initialization_only(&after, &before).unwrap());
    }

    #[test]
    fn journal_rejects_changed_ownership_queue_schema_flush_and_error_paths() {
        let before = journal(JOURNAL_BEFORE);
        let after = journal(JOURNAL_AFTER);
        for changed in [
            after.replace(".append(true)", ".append(false)"),
            after.replace("File::create(&path)?", "File::create(&path).ok()"),
            after.replace(".open(&path)?", ".open(&path).unwrap()"),
            after.replace("mode == JsonlJournalOpenMode::Truncate", "true"),
            after.replace("reuse_live_writer(&path)?", "false"),
            after.replace("config.queue_capacity", "config.queue_capacity + 1"),
            after.replace(
                "struct Record { text: String }",
                "struct Record { text: Vec<u8> }",
            ),
            after.replace("flush(receiver)?", "flush(receiver).ok()"),
            after.replace("file.sync_all()?", "file.sync_data()?"),
            after.replace(
                "let (sender, receiver)",
                "unknown_side_effect(); let (sender, receiver)",
            ),
            journal(&format!("{JOURNAL_AFTER}{JOURNAL_AFTER}")),
            format!("{after}\nfn extra_sink() {{ publish(); }}"),
        ] {
            assert_ne!(changed, after);
            assert!(!journal_initialization_only(&before, &changed).unwrap());
        }
    }

    const MEMORY_BEFORE: &str = r#"
        pub struct ProcessMemorySample {
            pub current_bytes: u64,
            pub high_water_bytes: u64,
            pub source: &'static str,
        }
        #[cfg(target_os = "linux")]
        fn current_resident_bytes() -> Option<u64> { linux_resident_bytes() }
        #[cfg(not(target_os = "linux"))]
        fn current_resident_bytes() -> Option<u64> { None }
        #[cfg(unix)]
        fn high_water_bytes() -> Option<u64> { getrusage_bytes() }
        #[cfg(not(unix))]
        fn high_water_bytes() -> Option<u64> { None }
        pub fn sample_process_memory() -> Option<ProcessMemorySample> {
            let high_water = high_water_bytes()?;
            let current = current_resident_bytes().unwrap_or(high_water);
            Some(ProcessMemorySample {
                current_bytes: current,
                high_water_bytes: high_water.max(current),
                source: process_memory_source(),
            })
        }
        #[cfg(not(unix))]
        fn process_memory_source() -> &'static str { "unsupported" }
    "#;

    fn memory_after() -> String {
        let after = MEMORY_BEFORE
            .replace(
                "#[cfg(not(target_os = \"linux\"))]",
                "#[cfg(all(not(windows), not(target_os = \"linux\")))]",
            )
            .replace("#[cfg(not(unix))]", "#[cfg(not(any(unix, windows)))]")
            .replace(
                "pub fn sample_process_memory",
                "#[cfg(not(windows))] pub fn sample_process_memory",
            );
        format!("{after}\n{WINDOWS_SAMPLE}")
    }

    #[test]
    fn memory_accepts_the_complete_windows_sampler_with_unchanged_unix_code() {
        assert!(windows_process_memory_only(MEMORY_BEFORE, &memory_after()).unwrap());
        assert!(!windows_process_memory_only(MEMORY_BEFORE, MEMORY_BEFORE).unwrap());
    }

    #[test]
    fn memory_rejects_wrong_counters_failures_cfgs_and_shared_schema_changes() {
        let after = memory_after();
        for changed in [
            after.replace("counters.WorkingSetSize", "counters.PagefileUsage"),
            after.replace("counters.PeakWorkingSetSize", "counters.PeakPagefileUsage"),
            after.replace("if succeeded == 0", "if succeeded != 0"),
            after.replace("GetCurrentProcess()", "other_process()"),
            after.replace("&mut counters, counters.cb", "&mut counters, 0"),
            after.replace("windows_process_memory_counters", "getrusage_maxrss"),
            after.replace("linux_resident_bytes()", "changed_linux_measurement()"),
            after.replace("getrusage_bytes()", "changed_unix_measurement()"),
            after.replace("pub current_bytes: u64", "pub current_bytes: u32"),
            after.replace("let high_water = high_water_bytes()?", "let high_water = 0"),
            after.replace("#[cfg(windows)]", "#[cfg(any(windows, unix))]"),
            after.replace(
                "let current_bytes = counters",
                "send_telemetry(); let current_bytes = counters",
            ),
            format!("{after}\n{WINDOWS_SAMPLE}"),
        ] {
            assert_ne!(changed, after);
            assert!(!windows_process_memory_only(MEMORY_BEFORE, &changed).unwrap());
        }
        assert!(journal_initialization_only("fn {", "").is_err());
        assert!(windows_process_memory_only("", "fn {").is_err());
    }
}
