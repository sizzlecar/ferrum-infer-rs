//! Stable Windows entrypoint. Engine upgrades only change current.json.
#[cfg(any(windows, test))]
#[path = "launcher/current.rs"]
mod current;
#[cfg(windows)]
#[path = "launcher/windows.rs"]
mod windows;

fn main() {
    #[cfg(windows)]
    let result = (|| {
        let launcher = std::env::current_exe().map_err(|e| e.to_string())?;
        let root = launcher
            .parent()
            .ok_or("launcher has no installation directory")?;
        let program = current::resolve(root)?;
        windows::run(&program, std::env::args_os().skip(1))
    })();
    #[cfg(not(windows))]
    let result: Result<u32, String> = Err("this launcher requires native Windows".into());
    match result {
        Ok(code) => std::process::exit(code as i32),
        Err(error) => {
            eprintln!("ferrum launcher: {error}");
            std::process::exit(1);
        }
    }
}
