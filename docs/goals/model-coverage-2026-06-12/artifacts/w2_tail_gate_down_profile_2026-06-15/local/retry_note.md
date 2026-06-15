2026-06-15 retry note:

- First remote tmux attempt exited before build because the non-login tmux shell
  did not have Rust on PATH (`cargo: command not found`).
- The runner now sources `/root/.cargo/env` and prepends `/root/.cargo/bin`.
- The failed remote output directory was removed before rerunning, so the final
  copied artifact contains the corrected diagnostic attempt.
