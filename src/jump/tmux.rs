//! tmux backend.
//!
//! Only applicable when abtop itself runs inside tmux (a `switch-client` needs
//! a tmux context). Maps the agent PID to the owning pane by process descent,
//! then switches client/window/pane. When the PID is in no tmux pane the
//! attempt is `NotApplicable`, letting another backend try.

use super::{find_pane_target, is_descendant_of, JumpAttempt, TerminalJumper};
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
                .status();
        }
        if let Some(window) = target.split('.').next() {
            let _ = Command::new("tmux")
                .args(["select-window", "-t", window])
                .status();
        }
        let _ = Command::new("tmux")
            .args(["select-pane", "-t", &target])
            .status();
        JumpAttempt::Jumped
    }
}

/// Reopen a dead Claude session in a new tmux window: `new-window` in the
/// session's cwd, then type `claude --resume <sid>` into the window's shell so
/// the user's PATH and aliases apply and the shell survives the agent exiting.
/// `NotApplicable` when abtop is not inside tmux.
pub fn resume_claude(cwd: &str, session_id: &str) -> JumpAttempt {
    if std::env::var("TMUX").is_err() {
        return JumpAttempt::NotApplicable;
    }
    // The id is typed into a shell; refuse anything but a plain UUID-like token.
    if session_id.is_empty()
        || !session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return JumpAttempt::Failed(format!("refusing unsafe session id {session_id:?}"));
    }
    let mut args = vec!["new-window", "-P", "-F", "#{pane_id}"];
    if std::path::Path::new(cwd).is_dir() {
        args.extend(["-c", cwd]);
    }
    let out = match Command::new("tmux").args(&args).output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return JumpAttempt::Failed(format!(
                "new-window failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            ))
        }
        Err(e) => return JumpAttempt::Failed(format!("tmux not runnable ({e})")),
    };
    let pane = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let cmd = format!("claude --resume {session_id}");
    let typed = Command::new("tmux")
        .args(["send-keys", "-t", &pane, "-l", &cmd])
        .status()
        .is_ok_and(|s| s.success());
    let entered = typed
        && Command::new("tmux")
            .args(["send-keys", "-t", &pane, "Enter"])
            .status()
            .is_ok_and(|s| s.success());
    if entered {
        JumpAttempt::Jumped
    } else {
        JumpAttempt::Failed(format!("could not start `{cmd}` in pane {pane}"))
    }
}
