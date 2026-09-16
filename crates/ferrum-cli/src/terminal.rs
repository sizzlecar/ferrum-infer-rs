//! One capability-based color policy for command output and diagnostics.

use std::ffi::OsStr;

#[derive(Debug, PartialEq, Eq)]
struct TerminalColors {
    stdout: bool,
    stderr: bool,
}

impl TerminalColors {
    fn detect(
        disabled: bool,
        stdout_supports_ansi: impl FnOnce() -> bool,
        stderr_supports_ansi: impl FnOnce() -> bool,
    ) -> Self {
        Self {
            stdout: !disabled && stdout_supports_ansi(),
            stderr: !disabled && stderr_supports_ansi(),
        }
    }

    fn shared_styles(&self) -> bool {
        // `colored` has one process-wide switch, and Ferrum uses it on both
        // streams. Never leak its escape sequences into a redirected stream.
        self.stdout && self.stderr
    }
}

/// Initialize before parsing commands or emitting diagnostics.
///
/// `console` checks each output handle and enables Windows virtual-terminal
/// processing only when the console API accepts it. A legacy console that
/// cannot enable VT, a redirected stream, or an explicit color opt-out gets
/// plain text. In particular, a TTY alone does not prove ANSI support.
pub fn initialize() {
    let disabled = colors_disabled(
        std::env::var_os("NO_COLOR").as_deref(),
        std::env::var_os("CLICOLOR").as_deref(),
        std::env::var_os("TERM").as_deref(),
    );
    let colors = TerminalColors::detect(
        disabled,
        || console::Term::stdout().features().colors_supported(),
        || console::Term::stderr().features().colors_supported(),
    );
    colored::control::set_override(colors.shared_styles());
    console::set_colors_enabled(colors.stdout);
    console::set_colors_enabled_stderr(colors.stderr);
}

fn colors_disabled(
    no_color: Option<&OsStr>,
    clicolor: Option<&OsStr>,
    term: Option<&OsStr>,
) -> bool {
    no_color.is_some() || clicolor == Some(OsStr::new("0")) || term == Some(OsStr::new("dumb"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_terminals_keep_styles() {
        let colors = TerminalColors::detect(false, || true, || true);
        assert!(colors.stdout);
        assert!(colors.stderr);
        assert!(colors.shared_styles());
    }

    #[test]
    fn unsupported_or_redirected_streams_never_receive_styles() {
        for (stdout, stderr) in [(false, false), (false, true), (true, false)] {
            let colors = TerminalColors::detect(false, || stdout, || stderr);
            assert_eq!(colors.stdout, stdout);
            assert_eq!(colors.stderr, stderr);
            assert!(!colors.shared_styles());
        }
    }

    #[test]
    fn color_opt_out_skips_capability_probes() {
        let colors = TerminalColors::detect(
            true,
            || panic!("stdout must not be probed when colors are disabled"),
            || panic!("stderr must not be probed when colors are disabled"),
        );
        assert!(!colors.stdout);
        assert!(!colors.stderr);
        assert!(!colors.shared_styles());
    }

    #[test]
    fn no_color_clicolor_zero_and_dumb_terminals_disable_styles() {
        assert!(colors_disabled(Some(OsStr::new("1")), None, None));
        assert!(colors_disabled(Some(OsStr::new("")), None, None));
        assert!(colors_disabled(None, Some(OsStr::new("0")), None));
        assert!(colors_disabled(None, None, Some(OsStr::new("dumb"))));
        assert!(!colors_disabled(None, None, None));
        assert!(!colors_disabled(
            None,
            Some(OsStr::new("1")),
            Some(OsStr::new("xterm-256color"))
        ));
    }
}
