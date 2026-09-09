//! Tiny native Rust executables used by the installer integration test's real subprocesses.
use std::{
    env, fs,
    io::{self, BufRead, Write},
    path::Path,
    process,
};

fn main() {
    let argv: Vec<_> = env::args().collect();
    match Path::new(&argv[0]).file_name().unwrap().to_str().unwrap() {
        "uname" => println!(
            "{}",
            env::var(if argv[1] == "-s" {
                "FIXTURE_OS"
            } else {
                "FIXTURE_ARCH"
            })
            .unwrap()
        ),
        "nvidia-smi" => match env::var("FIXTURE_GPU_CAPABILITIES") {
            Ok(value) => println!("{value}"),
            Err(_) => process::exit(1),
        },
        "system_profiler" => println!(
            "{}",
            env::var("FIXTURE_METAL_INFO").unwrap_or_else(|_| {
                r#"{"SPDisplaysDataType":[{"spdisplays_mtlgpufamilysupport":"spdisplays_metal3"}]}"#.into()
            })
        ),
        _ => {
            if argv.get(1).map(String::as_str) == Some("--version-session") {
                for line in io::stdin().lock().lines() {
                    match line.unwrap().as_str() {
                        "version" => {
                            println!("ferrum {}", env!("BOOTSTRAP_FIXTURE_VERSION"));
                            io::stdout().flush().unwrap();
                        }
                        "quit" => return,
                        _ => process::exit(2),
                    }
                }
                return;
            }
            if env::var_os("FIXTURE_MISSING_CUDA").is_some()
                && fs::metadata(Path::new(&argv[0]).parent().unwrap().join("CUDA-BUILD.txt"))
                    .is_ok()
            {
                eprintln!("fixture loader: libcudart.so.12 not found");
                process::exit(127);
            }
            if argv.get(1).map(String::as_str) != Some("--version") {
                process::exit(2);
            }
            println!("ferrum {}", env!("BOOTSTRAP_FIXTURE_VERSION"));
        }
    }
}
