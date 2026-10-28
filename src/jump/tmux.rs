//! tmux backend.
//!
//! Only applicable when abtop itself runs inside tmux (a `switch-client` needs
//! a tmux context). Maps the agent PID to the owning pane by process descent,
//! then switches client/window/pane. When the PID is in no tmux pane the
//! attempt is `NotApplicable`, letting another backend try.
//!
//! Every tmux call uses `.output()`: an inherited stderr would write over the
//! TUI, and the scroll it causes leaves ratatui's diff out of step with the
//! screen until restart.

use super::{find_pane_target, is_descendant_of, JumpAttempt, TerminalJumper, CLAUDE_RESUME};
use std::process::Command;

pub struct TmuxJumper;

impl TerminalJumper for TmuxJumper {
    fn name(&self) -> &'static str {
        "tmux"
    }

    fn try_jump(&self, pid: u32) -> JumpAttempt {
        if std::env::var("TMUX").is_err() {
            return JumpAttempt::NotApplicable;
        }
        let out = match Command::new("tmux")
            .args([
                "list-panes",
                "-a",
                "-F",
                "#{pane_pid} #{session_name}:#{window_index}.#{pane_index}",
            ])
            .output()
        {
            Ok(o) => o,
            Err(e) => return JumpAttempt::Failed(format!("tmux not runnable ({e})")),
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        let Some(target) = find_pane_target(&stdout, |pane_pid| is_descendant_of(pid, pane_pid))
        else {
            // PID not in any tmux pane — let another adapter try.
            return JumpAttempt::NotApplicable;
        };
        if let Some(session_name) = target.split(':').next() {
            let _ = Command::new("tmux")
                .args(["switch-client", "-t", session_name])
                .output();
        }
        if let Some(window) = target.split('.').next() {
            let _ = Command::new("tmux")
                .args(["select-window", "-t", window])
                .output();
        }
        let _ = Command::new("tmux")
            .args(["select-pane", "-t", &target])
            .output();
        JumpAttempt::Jumped
    }
}

/// Reopen a dead Claude session in its own detached tmux session: type
/// `CLAUDE_RESUME <sid>` into the new session's shell (so the user's PATH and
/// aliases apply and the shell survives the agent exiting), then hand the wait
/// to the tmux server, which switches this client over once Claude is running.
/// The wait runs under `run-shell -b`, so it outlives abtop (`--exit-on-jump`).
/// `NotApplicable` when abtop is not inside tmux.
pub fn resume_claude(cwd: &str, session_id: &str) -> JumpAttempt {
    if std::env::var("TMUX").is_err() {
        return JumpAttempt::NotApplicable;
    }
    // The id is typed into a shell; refuse anything but a plain UUID-like token.
    if !is_shell_safe(session_id) {
        return JumpAttempt::Failed(format!("refusing unsafe session id {session_id:?}"));
    }
    let client = tmux_query(&["display-message", "-p", "#{client_tty}"]).unwrap_or_default();
    let name = free_session_name(session_id);
    let mut args = vec!["new-session", "-d", "-s", &name, "-P", "-F", "#{pane_id}"];
    if std::path::Path::new(cwd).is_dir() {
        args.extend(["-c", cwd]);
    }
    let out = match Command::new("tmux").args(&args).output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return JumpAttempt::Failed(format!(
                "new-session failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            ))
        }
        Err(e) => return JumpAttempt::Failed(format!("tmux not runnable ({e})")),
    };
    let pane = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // The pane's shell, so the wait knows when something else is in front.
    let shell = tmux_query(&[
        "display-message",
        "-p",
        "-t",
        &pane,
        "#{pane_current_command}",
    ])
    .unwrap_or_default();
    let cmd = resume_command(session_id);
    let typed = Command::new("tmux")
        .args(["send-keys", "-t", &pane, "-l", &cmd])
        .output()
        .is_ok_and(|o| o.status.success())
        && Command::new("tmux")
            .args(["send-keys", "-t", &pane, "Enter"])
            .output()
            .is_ok_and(|o| o.status.success());
    if !typed {
        return JumpAttempt::Failed(format!("could not start `{cmd}` in session {name}"));
    }
    let script = switch_when_started_script(&pane, &shell, &client, &name);
    match Command::new("tmux")
        .args(["run-shell", "-b", &script])
        .output()
    {
        Ok(o) if o.status.success() => JumpAttempt::Jumped,
        _ => JumpAttempt::Failed(format!(
            "resumed in tmux session {name}; switch to it manually"
        )),
    }
}

/// Line typed into the new session's shell. The leading `clear` keeps the
/// command itself off the screen.
fn resume_command(session_id: &str) -> String {
    format!("clear; {CLAUDE_RESUME} {session_id}")
}

/// How long the tmux server waits for Claude to come up before switching anyway.
const RESUME_WAIT_TICKS: u32 = 75; // × 0.2s = 15s

/// Shell run by the tmux server: poll until the pane runs something other
/// than its shell (or the leading `clear`), drop the typed command from the
/// scrollback, then switch `client` (or the most recent one) to the session.
fn switch_when_started_script(pane: &str, shell: &str, client: &str, name: &str) -> String {
    // An empty pattern would be a `case` syntax error, so an unsafe shell
    // name just drops out; `''` still covers the not-yet-reported state.
    let idle = if is_shell_safe(shell) {
        format!("{shell}|clear|''")
    } else {
        "clear|''".to_string()
    };
    let client_flag = if !client.is_empty()
        && client
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '/')
    {
        format!("-c {client} ")
    } else {
        String::new()
    };
    format!(
        "i=0; while [ $i -lt {RESUME_WAIT_TICKS} ]; do \
         case \"$(tmux display-message -p -t {pane} '#{{pane_current_command}}')\" in \
         {idle}) i=$((i+1)); sleep 0.2 ;; *) break ;; esac; done; \
         tmux clear-history -t {pane}; tmux switch-client {client_flag}-t '={name}'"
    )
}

/// `claude-<first 8 chars of sid>`, suffixed `-2`, `-3`, ... if taken.
fn free_session_name(session_id: &str) -> String {
    let base = format!("claude-{}", session_id.chars().take(8).collect::<String>());
    let taken = |n: &str| {
        Command::new("tmux")
            .args(["has-session", "-t", &format!("={n}")])
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if !taken(&base) {
        return base;
    }
    (2..)
        .map(|i| format!("{base}-{i}"))
        .find(|n| !taken(n))
        .unwrap_or(base)
}

fn tmux_query(args: &[&str]) -> Option<String> {
    let out = Command::new("tmux").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn is_shell_safe(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_command_allows_skipping_permissions() {
        assert_eq!(
            resume_command("abcd-1234"),
            "clear; claude --allow-dangerously-skip-permissions --resume abcd-1234"
        );
    }

    #[test]
    fn switch_script_waits_for_non_shell_then_switches() {
        let s = switch_when_started_script("%12", "zsh", "/dev/ttys003", "claude-abcd1234");
        assert!(s.contains("-t %12 '#{pane_current_command}'"));
        assert!(s.contains("zsh|clear|''"));
        assert!(s.contains("tmux switch-client -c /dev/ttys003 -t '=claude-abcd1234'"));
    }

    #[test]
    fn switch_script_drops_unsafe_values() {
        let s = switch_when_started_script("%1", "z;sh", "/dev/x'y", "claude-1");
        assert!(s.contains("in clear|'')"));
        assert!(s.contains("tmux switch-client -t '=claude-1'"));
    }
}
