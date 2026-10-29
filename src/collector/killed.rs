//! Tracks Claude Code sessions that are no longer running, so they can be
//! listed as Stop and resumed later: `/exit`, crashes, SIGKILL,
//! `tmux kill-server`, or a reboot.
//!
//! Every live session is remembered in `~/.cache/abtop/claude_sessions.json`
//! so the record survives abtop restarts and reboots. When a remembered
//! session is no longer running, abtop looks for the user's `/exit` (or
//! `/quit`): first as the session's last entry in `{config-root}/history.jsonl`
//! (current Claude Code), then as the last turn of the transcript (older
//! versions). Either way the record is marked stopped; a clean exit is
//! remembered as `exited` so the row can say so. `x` dismisses a row.
//!
//! Sessions whose cwd is under one of [`TEMP_CWD_ROOTS`] are throwaway and
//! never tracked.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Stopped sessions whose transcript has not changed for this long are
/// forgotten. Matches Claude Code's default `cleanupPeriodDays` (30), after
/// which the transcript is gone and `claude --resume` no longer works.
const KILLED_RETENTION: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// Bytes read from the end of a transcript when looking for a trailing `/exit`.
const TAIL_BYTES: u64 = 64 * 1024;
/// Bytes read from the end of `history.jsonl`, which is shared by all sessions.
const HISTORY_TAIL_BYTES: u64 = 512 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TrackedSession {
    pub session_id: String,
    pub pid: u32,
    pub cwd: String,
    pub started_at: u64,
    pub transcript: PathBuf,
    #[serde(default)]
    pub killed: bool,
    /// Stopped with `/exit` or `/quit` rather than crashed or killed.
    #[serde(default)]
    pub exited: bool,
    /// Seen alive during this abtop run. PID-based checks are only trusted
    /// for these: PIDs loaded from disk may have been reused after a reboot.
    #[serde(skip)]
    pub seen_live: bool,
}

#[derive(Default, Serialize, Deserialize)]
struct StoreFile {
    #[serde(default)]
    sessions: Vec<TrackedSession>,
}

pub(crate) struct KilledTracker {
    /// Persistence file; `None` keeps the tracker in memory only (tests).
    path: Option<PathBuf>,
    tracked: HashMap<String, TrackedSession>,
    loaded: bool,
}

impl KilledTracker {
    pub fn persistent() -> Self {
        Self::with_path(Some(default_path()))
    }

    pub fn with_path(path: Option<PathBuf>) -> Self {
        Self {
            path,
            tracked: HashMap::new(),
            loaded: false,
        }
    }

    fn ensure_loaded(&mut self) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        let Some(path) = &self.path else {
            return;
        };
        let Ok(content) = fs::read_to_string(path) else {
            return;
        };
        let Ok(store) = serde_json::from_str::<StoreFile>(&content) else {
            return;
        };
        for s in store.sessions {
            self.tracked.insert(s.session_id.clone(), s);
        }
    }

    /// Reconcile tracked records with the sessions alive on this tick.
    pub fn update(&mut self, live: Vec<TrackedSession>) {
        self.ensure_loaded();
        let mut changed = false;
        let live_ids: HashSet<String> = live.iter().map(|s| s.session_id.clone()).collect();
        let live_pids: HashSet<u32> = live.iter().map(|s| s.pid).collect();

        for mut s in live.into_iter().filter(|s| !is_temp_cwd(&s.cwd)) {
            s.killed = false;
            s.exited = false;
            s.seen_live = true;
            if self.tracked.get(&s.session_id) != Some(&s) {
                self.tracked.insert(s.session_id.clone(), s);
                changed = true;
            }
        }

        let now = SystemTime::now();
        self.tracked.retain(|sid, s| {
            if is_temp_cwd(&s.cwd) {
                changed = true;
                return false;
            }
            if live_ids.contains(sid) {
                return true;
            }
            if !s.killed {
                changed = true;
                // Same process now runs another sid: `/clear` rotated it.
                let rotated = s.seen_live && live_pids.contains(&s.pid);
                if rotated {
                    return false;
                }
                s.exited = ended_cleanly(s);
                s.killed = true;
            }
            let keep = modified_within(&s.transcript, now, KILLED_RETENTION);
            if !keep {
                changed = true;
            }
            keep
        });

        if changed {
            self.save();
        }
    }

    /// Killed sessions, most recently started first.
    pub fn killed(&mut self) -> Vec<TrackedSession> {
        self.ensure_loaded();
        let mut out: Vec<TrackedSession> = self
            .tracked
            .values()
            .filter(|s| s.killed)
            .cloned()
            .collect();
        out.sort_by_key(|s| std::cmp::Reverse(s.started_at));
        out
    }

    /// Drop a killed record so it no longer shows up. Returns true if removed.
    pub fn forget(&mut self, session_id: &str) -> bool {
        self.ensure_loaded();
        let removed = self.tracked.get(session_id).is_some_and(|s| s.killed)
            && self.tracked.remove(session_id).is_some();
        if removed {
            self.save();
        }
        removed
    }

    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let mut sessions: Vec<TrackedSession> = self.tracked.values().cloned().collect();
        sessions.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        let Ok(json) = serde_json::to_string(&StoreFile { sessions }) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        // Write-then-rename so a concurrent reader never sees a partial file.
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        if fs::write(&tmp, json).is_ok() && fs::rename(&tmp, path).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    }
}

fn default_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".cache"))
        .join("abtop")
        .join("claude_sessions.json")
}

/// Directories whose sessions are throwaway and never tracked. `/private`
/// covers macOS's `/private/tmp` and `$TMPDIR` (`/private/var/folders/...`).
const TEMP_CWD_ROOTS: &[&str] = &["/tmp", "/private"];

fn is_temp_cwd(cwd: &str) -> bool {
    TEMP_CWD_ROOTS.iter().any(|root| {
        cwd.strip_prefix(root)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    })
}

fn modified_within(path: &Path, now: SystemTime, window: Duration) -> bool {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|mtime| now.duration_since(mtime).ok().or(Some(Duration::ZERO)))
        .is_some_and(|age| age <= window)
}

/// True when the user closed the session with `/exit` or `/quit`.
fn ended_cleanly(s: &TrackedSession) -> bool {
    // transcript = {config-root}/projects/{encoded-cwd}/{sid}.jsonl
    let history = s
        .transcript
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|root| root.join("history.jsonl"));
    history.is_some_and(|h| history_ends_with_exit(&h, &s.session_id))
        || transcript_ends_with_exit(&s.transcript)
}

fn is_exit_command(text: &str) -> bool {
    let cmd = text.trim();
    cmd == "/exit" || cmd == "/quit"
}

/// Read the last `max_bytes` of a file as complete lines.
fn tail_lines(path: &Path, max_bytes: u64) -> Option<Vec<String>> {
    let mut file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let start = len.saturating_sub(max_bytes);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    let mut lines: Vec<String> = String::from_utf8_lossy(&buf)
        .lines()
        .map(str::to_string)
        .collect();
    if start > 0 && !lines.is_empty() {
        // The first line is likely cut mid-JSON.
        lines.remove(0);
    }
    Some(lines)
}

/// True when the session's most recent prompt in `history.jsonl` is `/exit`.
fn history_ends_with_exit(history: &Path, session_id: &str) -> bool {
    let Some(lines) = tail_lines(history, HISTORY_TAIL_BYTES) else {
        return false;
    };
    for line in lines.iter().rev() {
        if !line.contains(session_id) {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("sessionId").and_then(|s| s.as_str()) != Some(session_id) {
            continue;
        }
        return v
            .get("display")
            .and_then(|d| d.as_str())
            .is_some_and(is_exit_command);
    }
    false
}

/// True when the last conversational entry in the transcript is the user's
/// `/exit` (or `/quit`) command. Older Claude Code versions logged it there.
fn transcript_ends_with_exit(path: &Path) -> bool {
    let Some(lines) = tail_lines(path, TAIL_BYTES) else {
        return false;
    };
    for line in lines.iter().rev() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("assistant") => return false,
            Some("user") => {}
            _ => continue,
        }
        if v.get("isMeta").and_then(|m| m.as_bool()) == Some(true) {
            continue;
        }
        let Some(content) = v.pointer("/message/content").and_then(|c| c.as_str()) else {
            // Array content (tool_result, rich prompt): a real turn, not /exit.
            return false;
        };
        let trimmed = content.trim_start();
        if trimmed.starts_with("<local-command-") {
            // Output of the slash command itself; keep looking for the command.
            continue;
        }
        return trimmed.starts_with("<command-name>/exit</command-name>")
            || trimmed.starts_with("<command-name>/quit</command-name>");
    }
    false
}

/// Compact "how long ago" label, e.g. `45s`, `12m`, `3h 5m`, `2d 4h`.
pub(crate) fn format_ago(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXIT_TAIL: &str = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"done"}]}}
{"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>Caveat</local-command-caveat>"}}
{"type":"user","message":{"role":"user","content":"<command-name>/exit</command-name>\n<command-message>exit</command-message>\n<command-args></command-args>"}}
{"type":"user","message":{"role":"user","content":"<local-command-stdout>(no content)</local-command-stdout>"}}
{"type":"last-prompt","lastPrompt":"x"}
{"type":"ai-title","aiTitle":"t"}
"#;

    const MID_TURN_TAIL: &str = r#"{"type":"user","message":{"role":"user","content":"fix the bug"}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{}}]}}
"#;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, body).unwrap();
        p
    }

    fn rec(sid: &str, pid: u32, transcript: &Path) -> TrackedSession {
        TrackedSession {
            session_id: sid.to_string(),
            pid,
            cwd: "/home/u/proj".to_string(),
            started_at: 1,
            transcript: transcript.to_path_buf(),
            killed: false,
            exited: false,
            seen_live: false,
        }
    }

    #[test]
    fn exit_command_counts_as_clean() {
        let dir = tempfile::tempdir().unwrap();
        assert!(transcript_ends_with_exit(&write(dir.path(), "a.jsonl", EXIT_TAIL)));
        let quit = EXIT_TAIL.replace("/exit", "/quit");
        assert!(transcript_ends_with_exit(&write(dir.path(), "b.jsonl", &quit)));
    }

    #[test]
    fn mid_turn_or_plain_prompt_is_not_clean() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!transcript_ends_with_exit(&write(dir.path(), "a.jsonl", MID_TURN_TAIL)));
        let prompt = r#"{"type":"user","message":{"role":"user","content":"hello"}}"#;
        assert!(!transcript_ends_with_exit(&write(dir.path(), "b.jsonl", prompt)));
        // An /exit followed by a resumed conversation is no longer clean.
        let resumed = format!("{EXIT_TAIL}{MID_TURN_TAIL}");
        assert!(!transcript_ends_with_exit(&write(dir.path(), "c.jsonl", &resumed)));
        assert!(!transcript_ends_with_exit(&dir.path().join("missing.jsonl")));
    }

    /// Lay out `{root}/projects/-proj/{sid}.jsonl` like a Claude config root.
    fn transcript_in_root(root: &Path, sid: &str, body: &str) -> PathBuf {
        let dir = root.join("projects").join("-proj");
        fs::create_dir_all(&dir).unwrap();
        write(&dir, &format!("{sid}.jsonl"), body)
    }

    fn history_line(sid: &str, display: &str) -> String {
        serde_json::json!({"display": display, "timestamp": 1, "project": "/proj", "sessionId": sid})
            .to_string()
    }

    #[test]
    fn history_exit_counts_as_clean() {
        let dir = tempfile::tempdir().unwrap();
        // Current Claude Code logs `/exit` only in history.jsonl, not the transcript.
        let tp = transcript_in_root(dir.path(), "s1", MID_TURN_TAIL);
        let history = [
            history_line("s1", "reply with just: hi"),
            history_line("s2", "other session"),
            history_line("s1", "/exit"),
            history_line("s2", "/exit"),
            history_line("s3", "/exit"),
            history_line("s3", "resumed and kept going"),
        ]
        .join("\n");
        write(dir.path(), "history.jsonl", &history);

        assert!(history_ends_with_exit(&dir.path().join("history.jsonl"), "s1"));
        assert!(!history_ends_with_exit(&dir.path().join("history.jsonl"), "s3"));
        assert!(!history_ends_with_exit(&dir.path().join("history.jsonl"), "s4"));
        assert!(ended_cleanly(&rec("s1", 1, &tp)));

        let tp3 = transcript_in_root(dir.path(), "s3", MID_TURN_TAIL);
        assert!(!ended_cleanly(&rec("s3", 1, &tp3)));
    }

    #[test]
    fn vanished_session_is_stopped_and_exit_is_remembered() {
        let dir = tempfile::tempdir().unwrap();
        let killed_tp = write(dir.path(), "k.jsonl", MID_TURN_TAIL);
        let exited_tp = write(dir.path(), "e.jsonl", EXIT_TAIL);
        let mut t = KilledTracker::with_path(None);
        t.update(vec![rec("k", 10, &killed_tp), rec("e", 11, &exited_tp)]);
        assert!(t.killed().is_empty());

        t.update(Vec::new());
        let mut stopped: Vec<(String, bool)> = t
            .killed()
            .into_iter()
            .map(|s| (s.session_id, s.exited))
            .collect();
        stopped.sort();
        assert_eq!(
            stopped,
            vec![("e".to_string(), true), ("k".to_string(), false)]
        );

        // Resuming an exited session clears the flag.
        t.update(vec![rec("e", 12, &exited_tp)]);
        assert_eq!(t.killed().len(), 1);
    }

    #[test]
    fn resumed_session_leaves_killed_state() {
        let dir = tempfile::tempdir().unwrap();
        let tp = write(dir.path(), "k.jsonl", MID_TURN_TAIL);
        let mut t = KilledTracker::with_path(None);
        t.update(vec![rec("k", 10, &tp)]);
        t.update(Vec::new());
        assert_eq!(t.killed().len(), 1);
        t.update(vec![rec("k", 20, &tp)]);
        assert!(t.killed().is_empty());
    }

    #[test]
    fn clear_rotation_on_same_pid_is_not_killed() {
        let dir = tempfile::tempdir().unwrap();
        let old = write(dir.path(), "old.jsonl", MID_TURN_TAIL);
        let new = write(dir.path(), "new.jsonl", MID_TURN_TAIL);
        let mut t = KilledTracker::with_path(None);
        t.update(vec![rec("old", 10, &old)]);
        t.update(vec![rec("new", 10, &new)]);
        assert!(t.killed().is_empty());
    }

    #[test]
    fn records_survive_restart_and_forget_persists() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("cache").join("claude_sessions.json");
        let tp = write(dir.path(), "k.jsonl", MID_TURN_TAIL);

        // abtop sees the session alive, then the machine goes down.
        let mut t = KilledTracker::with_path(Some(store.clone()));
        t.update(vec![rec("k", 10, &tp)]);
        drop(t);

        // After reboot a new claude happens to reuse PID 10: the record was
        // not seen live in this run, so it must still be classified Killed.
        let mut t = KilledTracker::with_path(Some(store.clone()));
        t.update(vec![rec("other", 10, &tp)]);
        let killed: Vec<String> = t.killed().into_iter().map(|s| s.session_id).collect();
        assert_eq!(killed, vec!["k".to_string()]);

        assert!(t.forget("k"));
        assert!(!t.forget("other"), "live sessions cannot be forgotten");
        let mut t = KilledTracker::with_path(Some(store));
        t.update(vec![rec("other", 10, &tp)]);
        assert!(t.killed().is_empty());
    }

    #[test]
    fn temp_dir_sessions_are_not_tracked() {
        let dir = tempfile::tempdir().unwrap();
        let tp = write(dir.path(), "k.jsonl", MID_TURN_TAIL);
        let mut tmp = rec("t", 10, &tp);
        tmp.cwd = "/private/tmp/scratch".to_string();
        let mut tmp_root = rec("r", 11, &tp);
        tmp_root.cwd = "/tmp".to_string();
        let mut look_alike = rec("l", 12, &tp);
        look_alike.cwd = "/tmpfoo/proj".to_string();
        let mut t = KilledTracker::with_path(None);
        t.update(vec![tmp, tmp_root, look_alike]);
        t.update(Vec::new());
        let stopped: Vec<String> = t.killed().into_iter().map(|s| s.session_id).collect();
        assert_eq!(stopped, vec!["l".to_string()]);
        assert!(is_temp_cwd("/tmp/a"));
        assert!(is_temp_cwd("/private/var/folders/x/T/proj"));
        assert!(!is_temp_cwd("/privatefoo"));
        assert!(!is_temp_cwd("/Users/me/tmp"));
    }

    #[test]
    fn missing_transcript_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let tp = dir.path().join("gone.jsonl");
        let mut t = KilledTracker::with_path(None);
        t.update(vec![rec("k", 10, &tp)]);
        t.update(Vec::new());
        assert!(t.killed().is_empty());
    }

    #[test]
    fn format_ago_units() {
        assert_eq!(format_ago(5), "5s");
        assert_eq!(format_ago(125), "2m");
        assert_eq!(format_ago(3 * 3600 + 300), "3h 5m");
        assert_eq!(format_ago(2 * 86_400 + 4 * 3600), "2d 4h");
    }
}
