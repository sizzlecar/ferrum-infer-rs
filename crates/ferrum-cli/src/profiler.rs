//! Performance Profiler for CLI operations
//!
//! Provides step-by-step timing and debug output for inference operations.

use colored::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A single timing entry
#[derive(Debug, Clone)]
pub struct TimingEntry {
    pub name: String,
    pub duration: Duration,
    pub details: Option<String>,
}

/// Profiler for tracking operation timings
#[derive(Debug)]
pub struct Profiler {
    /// Whether debug mode is enabled
    enabled: bool,
    /// Start time of the profiler
    start_time: Instant,
    /// Current step start time
    current_step_start: Option<Instant>,
    /// Current step name
    current_step_name: Option<String>,
    /// All timing entries
    entries: Vec<TimingEntry>,
    /// Nested level for indentation
    level: usize,
}

impl Profiler {
    /// Create a new profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start_time: Instant::now(),
            current_step_start: None,
            current_step_name: None,
            entries: Vec::new(),
            level: 0,
        }
    }

    /// Start a new step
    pub fn start_step(&mut self, name: &str) {
        // End previous step if any
        self.end_step();

        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!(
                "{}{}  {}...",
                indent,
                "▶".bright_blue(),
                name.bright_white()
            );
        }

        self.current_step_start = Some(Instant::now());
        self.current_step_name = Some(name.to_string());
    }

    /// End the current step
    pub fn end_step(&mut self) {
        if let (Some(start), Some(name)) = (self.current_step_start.take(), self.current_step_name.take()) {
            let duration = start.elapsed();
            
            if self.enabled {
                let indent = "  ".repeat(self.level);
                let duration_str = format_duration_colored(duration);
                println!(
                    "{}{}  {} {}",
                    indent,
                    "✓".bright_green(),
                    name.bright_white(),
                    duration_str
                );
            }

            self.entries.push(TimingEntry {
                name,
                duration,
                details: None,
            });
        }
    }

    /// End the current step with additional details
    pub fn end_step_with_details(&mut self, details: &str) {
        if let (Some(start), Some(name)) = (self.current_step_start.take(), self.current_step_name.take()) {
            let duration = start.elapsed();
            
            if self.enabled {
                let indent = "  ".repeat(self.level);
                let duration_str = format_duration_colored(duration);
                println!(
                    "{}{}  {} {} ({})",
                    indent,
                    "✓".bright_green(),
                    name.bright_white(),
                    duration_str,
                    details.dimmed()
                );
            }

            self.entries.push(TimingEntry {
                name,
                duration,
                details: Some(details.to_string()),
            });
        }
    }

    /// Start a nested section
    pub fn enter_section(&mut self, name: &str) {
        self.end_step();
        
        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!(
                "{}{}  {}",
                indent,
                "┌".bright_cyan(),
                name.bright_cyan().bold()
            );
        }
        self.level += 1;
    }

    /// End a nested section
    pub fn exit_section(&mut self) {
        self.end_step();
        
        if self.level > 0 {
            self.level -= 1;
        }
        
        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!("{}{}  {}", indent, "└".bright_cyan(), "done".dimmed());
        }
    }

    /// Log a debug message
    pub fn debug(&self, message: &str) {
        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!("{}{}  {}", indent, "·".dimmed(), message.dimmed());
        }
    }

    /// Log an info message
    pub fn info(&self, message: &str) {
        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!("{}{}  {}", indent, "ℹ".bright_blue(), message);
        }
    }

    /// Log a warning
    pub fn warn(&self, message: &str) {
        if self.enabled {
            let indent = "  ".repeat(self.level);
            println!("{}{}  {}", indent, "⚠".bright_yellow(), message.yellow());
        }
    }

    /// Log an error
    pub fn error(&self, message: &str) {
        // Always show errors regardless of debug mode
        let indent = "  ".repeat(self.level);
        println!("{}{}  {}", indent, "✗".bright_red(), message.red());
    }

    /// Print summary of all timings
    pub fn print_summary(&mut self) {
        self.end_step();
        
        if !self.enabled || self.entries.is_empty() {
            return;
        }

        let total_duration = self.start_time.elapsed();

        println!();
        println!("{}", "═══ Performance Summary ═══".bright_cyan().bold());
        println!();

        // Find the longest step name for alignment
        let max_name_len = self.entries.iter().map(|e| e.name.len()).max().unwrap_or(20);

        // Print each entry
        let mut total_tracked = Duration::ZERO;
        for entry in &self.entries {
            let name_padded = format!("{:<width$}", entry.name, width = max_name_len);
            let duration_str = format_duration_ms(entry.duration);
            let pct = (entry.duration.as_secs_f64() / total_duration.as_secs_f64() * 100.0) as u32;
            
            let bar_width = (pct as usize).min(30);
            let bar = "█".repeat(bar_width);
            let empty = "░".repeat(30 - bar_width);
            
            let color_bar = if pct > 50 {
                bar.bright_red()
            } else if pct > 20 {
                bar.bright_yellow()
            } else {
                bar.bright_green()
            };

            println!(
                "  {} {} {:>8} {:>3}% {}{}",
                name_padded.white(),
                "│".dimmed(),
                duration_str.bright_white(),
                pct,
                color_bar,
                empty.dimmed()
            );

            if let Some(details) = &entry.details {
                println!(
                    "  {} {} {}",
                    " ".repeat(max_name_len),
                    "│".dimmed(),
                    details.dimmed()
                );
            }

            total_tracked += entry.duration;
        }

        println!();
        println!(
            "  {} {}",
            "Total time:".bright_white().bold(),
            format_duration_ms(total_duration).bright_cyan().bold()
        );

        let untracked = total_duration.saturating_sub(total_tracked);
        if untracked.as_millis() > 10 {
            println!(
                "  {} {}",
                "Untracked:".dimmed(),
                format_duration_ms(untracked).dimmed()
            );
        }

        println!();
    }

    /// Get total elapsed time
    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if profiler is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Global profiler for easy access
pub struct GlobalProfiler {
    profiler: Mutex<Profiler>,
}

impl GlobalProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            profiler: Mutex::new(Profiler::new(enabled)),
        }
    }

    pub fn start_step(&self, name: &str) {
        self.profiler.lock().unwrap().start_step(name);
    }

    pub fn end_step(&self) {
        self.profiler.lock().unwrap().end_step();
    }

    pub fn end_step_with_details(&self, details: &str) {
        self.profiler.lock().unwrap().end_step_with_details(details);
    }

    pub fn enter_section(&self, name: &str) {
        self.profiler.lock().unwrap().enter_section(name);
    }

    pub fn exit_section(&self) {
        self.profiler.lock().unwrap().exit_section();
    }

    pub fn debug(&self, message: &str) {
        self.profiler.lock().unwrap().debug(message);
    }

    pub fn info(&self, message: &str) {
        self.profiler.lock().unwrap().info(message);
    }

    pub fn warn(&self, message: &str) {
        self.profiler.lock().unwrap().warn(message);
    }

    pub fn error(&self, message: &str) {
        self.profiler.lock().unwrap().error(message);
    }

    pub fn print_summary(&self) {
        self.profiler.lock().unwrap().print_summary();
    }
}

/// Format duration with colors based on magnitude
fn format_duration_colored(duration: Duration) -> ColoredString {
    let ms = duration.as_millis();
    let text = format_duration_ms(duration);
    
    if ms > 5000 {
        text.bright_red()
    } else if ms > 1000 {
        text.bright_yellow()
    } else if ms > 100 {
        text.bright_white()
    } else {
        text.bright_green()
    }
}

/// Format duration in milliseconds
fn format_duration_ms(duration: Duration) -> String {
    let ms = duration.as_millis();
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        let minutes = ms / 60_000;
        let seconds = (ms % 60_000) as f64 / 1000.0;
        format!("{}m {:.1}s", minutes, seconds)
    }
}

/// RAII guard for timing a scope
pub struct TimingGuard<'a> {
    profiler: &'a mut Profiler,
    details: Option<String>,
}

impl<'a> TimingGuard<'a> {
    pub fn new(profiler: &'a mut Profiler, name: &str) -> Self {
        profiler.start_step(name);
        Self {
            profiler,
            details: None,
        }
    }

    pub fn set_details(&mut self, details: String) {
        self.details = Some(details);
    }
}

impl<'a> Drop for TimingGuard<'a> {
    fn drop(&mut self) {
        if let Some(details) = self.details.take() {
            self.profiler.end_step_with_details(&details);
        } else {
            self.profiler.end_step();
        }
    }
}

/// Macro for timing a block
#[macro_export]
macro_rules! time_step {
    ($profiler:expr, $name:expr, $block:expr) => {{
        $profiler.start_step($name);
        let result = $block;
        $profiler.end_step();
        result
    }};
}

/// Macro for timing a block with details
#[macro_export]
macro_rules! time_step_with_details {
    ($profiler:expr, $name:expr, $block:expr, $details:expr) => {{
        $profiler.start_step($name);
        let result = $block;
        $profiler.end_step_with_details($details);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = Profiler::new(false); // disabled for test
        profiler.start_step("test step");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_step();
        assert_eq!(profiler.entries.len(), 1);
    }

    #[test]
    fn test_profiler_nested() {
        let mut profiler = Profiler::new(false);
        profiler.enter_section("outer");
        profiler.start_step("inner step");
        profiler.end_step();
        profiler.exit_section();
        assert_eq!(profiler.entries.len(), 1);
    }
}


