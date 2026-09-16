# Terminal output

Ferrum automatically enables styling only when the output terminal supports
ANSI escape sequences. On Windows it first asks the console to enable virtual
terminal processing. If that fails, output stays plain text; changing the
console code page is not required.

`NO_COLOR` disables styling. `CLICOLOR=0` and `TERM=dumb` also request plain text.
Redirected output is plain text, even if `CLICOLOR_FORCE` is inherited from the
environment. Because command messages share a color switch between stdout and
stderr, redirecting either stream disables styling for those messages. Logging
uses stderr's own terminal capability.

For an older Ferrum binary that prints sequences such as `[32m` in a Windows
PowerShell console, disable styling for the current session:

```powershell
Remove-Item Env:CLICOLOR_FORCE -ErrorAction SilentlyContinue
$env:NO_COLOR = '1'
ferrum list
```

This is a terminal presentation setting, not a CUDA or model configuration.
The automatic capability check applies to `list` and the other Ferrum commands,
including console messages from both `run` and `serve`. It does not change model
execution or HTTP response content.
