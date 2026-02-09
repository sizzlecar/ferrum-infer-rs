//! Stop command - Stop the running server

use clap::Args;
use colored::*;
use ferrum_types::Result;

#[derive(Args)]
pub struct StopCommand {}

pub async fn execute(_cmd: StopCommand) -> Result<()> {
    let pid_file = std::env::temp_dir().join("ferrum.pid");

    if !pid_file.exists() {
        println!("{}", "No running server found.".yellow());
        return Ok(());
    }

    // Read PID
    let pid_str = std::fs::read_to_string(&pid_file).map_err(|e| {
        ferrum_types::FerrumError::io_str(format!("Failed to read PID file: {}", e))
    })?;

    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|e| ferrum_types::FerrumError::io_str(format!("Invalid PID: {}", e)))?;

    // Send SIGTERM
    #[cfg(unix)]
    {
        use std::process::Command;
        let status = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("{} Server stopped (PID: {})", "✓".green().bold(), pid);
                std::fs::remove_file(&pid_file).ok();
            }
            Ok(_) => {
                println!(
                    "{} Process {} not found (may have already stopped)",
                    "⚠".yellow(),
                    pid
                );
                std::fs::remove_file(&pid_file).ok();
            }
            Err(e) => {
                eprintln!("{} Failed to stop server: {}", "✗".red().bold(), e);
            }
        }
    }

    #[cfg(windows)]
    {
        use std::process::Command;
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("{} Server stopped (PID: {})", "✓".green().bold(), pid);
                std::fs::remove_file(&pid_file).ok();
            }
            _ => {
                println!("{} Process {} not found", "⚠".yellow(), pid);
                std::fs::remove_file(&pid_file).ok();
            }
        }
    }

    Ok(())
}
