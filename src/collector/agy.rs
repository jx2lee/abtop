//! Antigravity CLI (`agy`) collector.
//!
//! Discovery strategy:
//! 1. `ps` (shared process data) finds running `agy` processes.
//! 2. Each conversation has `presence/<id>.lock`, held open by its process.
//!    The file outlives the process, so only an open descriptor proves that a
//!    PID owns the conversation. This also skips `agy models` and friends.
//! 3. `conversation_summaries.db` gives the run state (running or idle).
//! 4. `brain/<id>/.system_generated/logs/transcript.jsonl` gives tokens, tool
//!    calls and answers. It is the file agy exposes to hooks as `transcriptPath`.
//!
//! 5. `agy -p "/usage"` gives the weekly quota. It makes no model request and
//!    is run by the TUI only, in the background, while a conversation is live.
//! 6. Conversations that stop are remembered in `agy_sessions.json`, as for
//!    Claude Code. One-shot `agy -p` runs are never remembered.
//!
//! Not reported: context window % (agy records neither the window nor, except
//! when it changes, the model).

use super::killed::{self, format_ago, KilledTracker, TrackedSession};
use super::{process, rate_limit};
use crate::model::{
    AgentSession, ChatMessage, ChatRole, ChildProcess, LaunchSurface, RateLimitInfo, SessionStatus,
    ToolCall, MAX_CHAT_MESSAGES,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// Rows fetched from the summaries DB per query.
const MAX_DB_ROWS: u32 = 100;
/// Bytes of new transcript read per tick; the rest is picked up next tick.
const MAX_READ_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TOOL_CALLS: usize = 200;
const MAX_TOKEN_HISTORY: usize = 200;
const MAX_PROMPT_CHARS: usize = 200;
const MAX_CHAT_CHARS: usize = 500;
const MAX_TASK_CHARS: usize = 60;
/// `/usage` starts a whole agy process and asks the server, so keep it rare.
const QUOTA_REFRESH: Duration = Duration::from_secs(5 * 60);
const WEEK_MINUTES: u64 = 7 * 24 * 60;

pub struct AgyCollector {
    root: PathBuf,
    sqlite3_available: Option<bool>,
    /// PID to its open conversations and cwd. Refreshed on slow ticks because
    /// finding them forks `lsof`.
    owners: HashMap<u32, Owner>,
    /// Latest DB rows by conversation id. Kept when a query fails.
    runs: HashMap<String, RunInfo>,
    transcripts: HashMap<String, Transcript>,
    tracker: KilledTracker,
    stopped_transcripts: HashMap<String, Transcript>,
    quota_running: Arc<AtomicBool>,
    quota_attempted: Option<Instant>,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Owner {
    conversations: Vec<String>,
    cwd: String,
}

#[derive(Debug, Default, Clone)]
struct RunInfo {
    status: String,
    title: String,
    top_level: bool,
}

/// State accumulated from a transcript, read incrementally.
#[derive(Debug, Default)]
struct Transcript {
    offset: u64,
    mtime_ms: u64,
    started_at: u64,
    /// `created_at` of the latest finished answer. Steps are stamped when they
    /// begin, so a `last_step_is_answer` transcript's mtime is more accurate.
    last_reply_ms: u64,
    last_step_is_answer: bool,
    /// Whether the transcript alone says the turn is still going. Only used
    /// when the DB has no row for the conversation.
    turn_open: bool,
    /// Label of the tool call the last step is waiting on.
    pending_tool: Option<String>,
    total_input: u64,
    total_output: u64,
    total_cache_read: u64,
    turns: u32,
    model: String,
    initial_prompt: String,
    first_assistant_text: String,
    chat: Vec<ChatMessage>,
    tool_calls: Vec<ToolCall>,
    token_history: Vec<u64>,
}

impl Transcript {
    fn apply_line(&mut self, line: &str) {
        let Ok(step) = serde_json::from_str::<Value>(line) else {
            return;
        };
        let created_at = step["created_at"]
            .as_str()
            .and_then(parse_ts_ms)
            .unwrap_or(0);
        if self.started_at == 0 {
            self.started_at = created_at;
        }
        self.last_step_is_answer = false;
        self.pending_tool = None;
        let content = step["content"].as_str().unwrap_or("");
        match step["type"].as_str() {
            Some("USER_INPUT") => self.apply_user_input(content),
            Some("PLANNER_RESPONSE") => self.apply_response(&step, content, created_at),
            _ => self.turn_open = step["status"].as_str() != Some("ERROR"),
        }
    }

    fn apply_user_input(&mut self, content: &str) {
        self.turn_open = true;
        if let Some(model) = model_from_settings_change(content) {
            self.model = clean_text(model, MAX_TASK_CHARS);
        }
        let prompt = user_request(content);
        if self.initial_prompt.is_empty() {
            self.initial_prompt = clean_text(prompt, MAX_PROMPT_CHARS);
        }
        self.push_chat(ChatRole::User, prompt);
    }

    fn apply_response(&mut self, step: &Value, content: &str, created_at: u64) {
        self.turns += 1;
        let input = step["input_tokens"].as_u64().unwrap_or(0);
        let output = step["output_tokens"].as_u64().unwrap_or(0);
        self.total_input += input;
        self.total_output += output;
        self.total_cache_read += step["cache_read_tokens"].as_u64().unwrap_or(0);
        self.token_history.push(input + output);
        if self.token_history.len() > MAX_TOKEN_HISTORY {
            self.token_history.remove(0);
        }

        let tools = step["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
        for tool in tools {
            let name = clean_text(tool["name"].as_str().unwrap_or("tool"), MAX_TASK_CHARS);
            let arg = tool_arg(&tool["args"]);
            self.pending_tool = Some(tool_label(&name, &arg));
            self.tool_calls.push(ToolCall {
                name,
                arg,
                duration_ms: 0,
            });
        }
        if self.tool_calls.len() > MAX_TOOL_CALLS {
            let excess = self.tool_calls.len() - MAX_TOOL_CALLS;
            self.tool_calls.drain(..excess);
        }

        self.turn_open = !tools.is_empty();
        if content.is_empty() {
            return;
        }
        if self.first_assistant_text.is_empty() {
            self.first_assistant_text = clean_text(content, MAX_PROMPT_CHARS);
        }
        self.push_chat(ChatRole::Assistant, content);
        if tools.is_empty() && step["status"].as_str() == Some("DONE") {
            self.last_step_is_answer = true;
            self.last_reply_ms = self.last_reply_ms.max(created_at);
        }
    }

    fn push_chat(&mut self, role: ChatRole, text: &str) {
        self.chat.push(ChatMessage {
            role,
            text: clean_text(text, MAX_CHAT_CHARS),
        });
        if self.chat.len() > MAX_CHAT_MESSAGES {
            self.chat.remove(0);
        }
    }

    /// Read lines appended since the last call. A trailing partial line stays
    /// unread until it is complete; a shrunk file is parsed from the start.
    fn refresh(&mut self, path: &Path) {
        let Ok(meta) = fs::symlink_metadata(path) else {
            return;
        };
        if !meta.is_file() {
            return;
        }
        if meta.len() < self.offset {
            *self = Self::default();
        }
        self.mtime_ms = modified_ms(&meta);
        if meta.len() == self.offset {
            return;
        }
        let Ok(mut file) = fs::File::open(path) else {
            return;
        };
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut buf = Vec::new();
        if file.take(MAX_READ_BYTES).read_to_end(&mut buf).is_err() {
            return;
        }
        let Some(end) = buf.iter().rposition(|b| *b == b'\n') else {
            return;
        };
        for line in buf[..end].split(|b| *b == b'\n') {
            if let Ok(line) = std::str::from_utf8(line) {
                self.apply_line(line);
            }
        }
        self.offset += end as u64 + 1;
    }

    /// When the agent last finished answering, 0 if it has not.
    fn last_reply(&self) -> u64 {
        if self.last_step_is_answer {
            self.last_reply_ms.max(self.mtime_ms)
        } else {
            self.last_reply_ms
        }
    }
}

impl AgyCollector {
    pub fn new() -> Self {
        Self::with_root(
            dirs::home_dir()
                .unwrap_or_default()
                .join(".gemini/antigravity-cli"),
        )
    }

    fn with_root(root: PathBuf) -> Self {
        Self {
            root,
            sqlite3_available: None,
            owners: HashMap::new(),
            runs: HashMap::new(),
            transcripts: HashMap::new(),
            tracker: KilledTracker::with_path(if cfg!(test) {
                None
            } else {
                Some(killed::store_path("agy_sessions.json"))
            })
            .with_exit_check(ended_with_exit),
            stopped_transcripts: HashMap::new(),
            quota_running: Arc::new(AtomicBool::new(false)),
            quota_attempted: None,
        }
    }

    fn transcript_path(&self, conversation: &str) -> PathBuf {
        self.root
            .join("brain")
            .join(conversation)
            .join(".system_generated/logs/transcript.jsonl")
    }

    fn refresh_owners(&mut self, pids: &[u32], slow_tick: bool) {
        self.owners.retain(|pid, _| pids.contains(pid));
        if !slow_tick && pids.iter().all(|pid| self.owners.contains_key(pid)) {
            return;
        }
        if let Some(found) = find_owners(pids) {
            self.owners = found;
        }
    }

    /// Queried every tick, unlike OpenCode's slow-tick refresh, because the
    /// run state is what tells working from idle. Only forks while an agy
    /// conversation is live.
    fn refresh_runs(&mut self) {
        let db = self.root.join("conversation_summaries.db");
        if fs::symlink_metadata(&db).map_or(true, |m| !m.is_file()) {
            self.runs.clear();
            return;
        }
        let available = *self
            .sqlite3_available
            .get_or_insert_with(|| Command::new("sqlite3").arg("--version").output().is_ok());
        if !available {
            return;
        }
        let sql = format!(
            "SELECT conversation_id, status, title, parent_conversation_id \
             FROM conversation_summaries ORDER BY last_modified_time DESC LIMIT {MAX_DB_ROWS}"
        );
        let Ok(output) = Command::new("sqlite3")
            .args(["-readonly", "-json"])
            .arg(&db)
            .arg(sql)
            .output()
        else {
            return;
        };
        if output.status.success() {
            if let Some(runs) = parse_runs(&String::from_utf8_lossy(&output.stdout)) {
                self.runs = runs;
            }
        }
    }

    /// A process can hold several locks when it runs subagents. Show the
    /// top-level conversation that changed last.
    fn main_conversation(&self, owner: &Owner) -> Option<String> {
        owner
            .conversations
            .iter()
            .max_by_key(|id| {
                let top_level = self.runs.get(*id).is_none_or(|r| r.top_level);
                let mtime = fs::symlink_metadata(self.transcript_path(id))
                    .map(|m| modified_ms(&m))
                    .unwrap_or(0);
                (top_level, mtime)
            })
            .cloned()
    }

    fn lock_started_ms(&self, conversation: &str) -> u64 {
        fs::symlink_metadata(
            self.root
                .join("presence")
                .join(format!("{conversation}.lock")),
        )
        .map(|m| modified_ms(&m))
        .unwrap_or(0)
    }

    fn collect_live(
        &mut self,
        shared: &super::SharedProcessData,
        sessions: &mut Vec<AgentSession>,
        tracked: &mut Vec<TrackedSession>,
    ) {
        let selected: Vec<(u32, Owner, String)> = self
            .owners
            .iter()
            .filter_map(|(pid, owner)| {
                self.main_conversation(owner)
                    .map(|id| (*pid, owner.clone(), id))
            })
            .collect();
        let live: HashSet<&String> = selected.iter().map(|(_, _, id)| id).collect();
        self.transcripts.retain(|id, _| live.contains(id));

        for (pid, owner, id) in &selected {
            let path = self.transcript_path(id);
            self.transcripts
                .entry(id.clone())
                .or_default()
                .refresh(&path);
            let run = self.runs.get(id);
            let transcript = &self.transcripts[id];
            let (status, current_tasks) = derive_status(run.map(|r| r.status.as_str()), transcript);
            let started_at = match transcript.started_at {
                0 => self.lock_started_ms(id),
                t => t,
            };
            let info = shared.process_info.get(pid);
            let mut row = base_row(&self.root, id, &owner.cwd, transcript, run);
            row.pid = *pid;
            row.started_at = started_at;
            row.status = status;
            row.current_tasks = current_tasks;
            row.mem_mb = info.map_or(0, |p| p.rss_kb / 1024);
            row.children = collect_children(*pid, shared);
            sessions.push(row);

            // A run with no prompt yet has nothing to come back to.
            if transcript.started_at > 0 && !info.is_some_and(|p| is_print_mode(&p.command)) {
                tracked.push(TrackedSession {
                    session_id: id.clone(),
                    pid: *pid,
                    cwd: owner.cwd.clone(),
                    started_at,
                    transcript: path,
                    killed: false,
                    exited: false,
                    seen_live: true,
                });
            }
        }
    }

    fn track_stopped(&mut self, tracked: Vec<TrackedSession>) -> Vec<AgentSession> {
        self.tracker.update(tracked);
        let stopped = self.tracker.killed();
        let ids: HashSet<&str> = stopped.iter().map(|s| s.session_id.as_str()).collect();
        self.stopped_transcripts
            .retain(|id, _| ids.contains(id.as_str()));
        stopped
            .iter()
            .filter_map(|rec| self.stopped_row(rec))
            .collect()
    }

    fn stopped_row(&mut self, rec: &TrackedSession) -> Option<AgentSession> {
        if !fs::symlink_metadata(&rec.transcript).is_ok_and(|m| m.is_file()) {
            return None;
        }
        let transcript = self
            .stopped_transcripts
            .entry(rec.session_id.clone())
            .or_default();
        transcript.refresh(&rec.transcript);
        let idle_secs = now_ms().saturating_sub(transcript.mtime_ms) / 1000;
        let mut row = base_row(
            &self.root,
            &rec.session_id,
            &rec.cwd,
            transcript,
            self.runs.get(&rec.session_id),
        );
        // PID 0: the old PID may already belong to an unrelated process.
        row.pid = 0;
        row.started_at = rec.started_at;
        row.status = SessionStatus::Killed;
        row.current_tasks = vec![format!(
            "{} · last active {} ago",
            if rec.exited { "exited" } else { "killed" },
            format_ago(idle_secs)
        )];
        Some(row)
    }
}

impl Default for AgyCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl super::AgentCollector for AgyCollector {
    fn collect(&mut self, shared: &super::SharedProcessData) -> Vec<AgentSession> {
        let pids: Vec<u32> = shared
            .process_info
            .iter()
            .filter(|(_, info)| process::cmd_first_token_has_binary(&info.command, "agy"))
            .map(|(pid, _)| *pid)
            .collect();
        self.refresh_owners(&pids, shared.slow_tick);
        let mut sessions = Vec::new();
        let mut tracked = Vec::new();
        if self.owners.is_empty() {
            self.transcripts.clear();
            self.runs.clear();
        } else {
            self.refresh_runs();
            self.collect_live(shared, &mut sessions, &mut tracked);
        }
        let stopped = self.track_stopped(tracked);
        sessions.extend(stopped);
        sessions
    }

    fn live_rate_limit(&self) -> Option<RateLimitInfo> {
        rate_limit::read_agy_cache()
    }

    fn refresh_quota(&mut self) {
        if self.owners.is_empty()
            || self.quota_running.load(Ordering::Acquire)
            || self
                .quota_attempted
                .is_some_and(|at| at.elapsed() < QUOTA_REFRESH)
        {
            return;
        }
        let now = now_ms() / 1000;
        let cached_at = rate_limit::read_agy_cache().and_then(|rl| rl.updated_at);
        if cached_at.is_some_and(|at| now.saturating_sub(at) < QUOTA_REFRESH.as_secs()) {
            return;
        }
        self.quota_attempted = Some(Instant::now());
        self.quota_running.store(true, Ordering::Release);
        let running = Arc::clone(&self.quota_running);
        let root = self.root.clone();
        std::thread::spawn(move || {
            // `.output()` also keeps agy's stderr off the TUI.
            let output = Command::new("agy")
                .args(["-p", "/usage", "--output-format", "json"])
                .args(["--print-timeout", "30s"])
                .current_dir(&root)
                .stdin(Stdio::null())
                .output();
            if let Ok(output) = output {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(info) = output.status.success().then(|| parse_usage(&stdout, now)) {
                    info.iter().for_each(rate_limit::write_agy_cache);
                }
            }
            running.store(false, Ordering::Release);
        });
    }

    fn forget_killed(&mut self, session_id: &str) -> bool {
        self.stopped_transcripts.remove(session_id);
        self.tracker.forget(session_id)
    }
}

fn base_row(
    root: &Path,
    id: &str,
    cwd: &str,
    transcript: &Transcript,
    run: Option<&RunInfo>,
) -> AgentSession {
    AgentSession {
        agent_cli: "agy",
        launch_surface: LaunchSurface::Cli,
        pid: 0,
        session_id: id.to_string(),
        cwd: cwd.to_string(),
        project_name: process::last_path_segment(cwd).unwrap_or("?").to_string(),
        started_at: transcript.started_at,
        last_reply_ms: transcript.last_reply(),
        status: SessionStatus::Waiting,
        model: if transcript.model.is_empty() {
            "-".to_string()
        } else {
            transcript.model.clone()
        },
        effort: String::new(),
        context_percent: 0.0,
        total_input_tokens: transcript.total_input,
        total_output_tokens: transcript.total_output,
        total_cache_read: transcript.total_cache_read,
        total_cache_create: 0,
        turn_count: transcript.turns,
        current_tasks: vec![],
        mem_mb: 0,
        version: String::new(),
        git_branch: String::new(),
        git_added: 0,
        git_modified: 0,
        token_history: transcript.token_history.clone(),
        context_history: vec![],
        compaction_count: 0,
        context_window: 0,
        subagents: vec![],
        mem_file_count: 0,
        mem_line_count: 0,
        children: vec![],
        initial_prompt: transcript.initial_prompt.clone(),
        session_name: run.map_or_else(String::new, |r| clean_text(&r.title, 80)),
        first_assistant_text: transcript.first_assistant_text.clone(),
        chat_messages: transcript.chat.clone(),
        tool_calls: transcript.tool_calls.clone(),
        pending_since_ms: 0,
        thinking_since_ms: 0,
        file_accesses: vec![],
        config_root: super::abbrev_path(root),
    }
}

/// One-shot runs have no session worth reopening.
fn is_print_mode(command: &str) -> bool {
    command
        .split_whitespace()
        .skip(1)
        .any(|arg| matches!(arg, "-p" | "--print" | "--prompt"))
}

/// True when the user closed the conversation with `/exit` or `/quit`.
/// `history.jsonl` records prompts per workspace, without a conversation id.
fn ended_with_exit(s: &TrackedSession) -> bool {
    // transcript = <root>/brain/<id>/.system_generated/logs/transcript.jsonl
    let Some(root) = s.transcript.ancestors().nth(5) else {
        return false;
    };
    killed::tail_lines(&root.join("history.jsonl"), killed::HISTORY_TAIL_BYTES)
        .is_some_and(|lines| history_ends_with_exit(&lines, &s.cwd))
}

fn history_ends_with_exit(lines: &[String], cwd: &str) -> bool {
    lines
        .iter()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|entry| entry["workspace"].as_str() == Some(cwd))
        .is_some_and(|entry| {
            entry["display"]
                .as_str()
                .is_some_and(killed::is_exit_command)
        })
}

/// Parse `agy -p /usage --output-format json`. Gemini fills the first gauge
/// and the other model family the second.
fn parse_usage(stdout: &str, now_secs: u64) -> Option<RateLimitInfo> {
    let usage: Value = serde_json::from_str(stdout).ok()?;
    let groups = usage.pointer("/command/data/groups")?.as_array()?;
    let mut info = RateLimitInfo {
        source: "agy".to_string(),
        updated_at: Some(now_secs),
        ..Default::default()
    };
    for group in groups {
        let Some((remaining, bucket)) = tightest_bucket(group) else {
            continue;
        };
        let used = Some((1.0 - remaining) * 100.0);
        let resets = bucket["reset_time"]
            .as_str()
            .and_then(parse_ts_ms)
            .map(|ms| ms / 1000);
        let window = (bucket["window"].as_str() == Some("weekly")).then_some(WEEK_MINUTES);
        let is_gemini = group["name"]
            .as_str()
            .is_some_and(|name| name.to_ascii_lowercase().contains("gemini"));
        if is_gemini && info.five_hour_pct.is_none() {
            info.five_hour_pct = used;
            info.five_hour_resets_at = resets;
            info.five_hour_window_minutes = window;
        } else if !is_gemini && info.seven_day_pct.is_none() {
            info.seven_day_pct = used;
            info.seven_day_resets_at = resets;
            info.seven_day_window_minutes = window;
        }
    }
    (info.five_hour_pct.is_some() || info.seven_day_pct.is_some()).then_some(info)
}

/// The group's bucket closest to running out, with its remaining fraction.
/// Proto-derived JSON omits zeros, so a bucket with a reset time but no
/// fraction has nothing left.
fn tightest_bucket(group: &Value) -> Option<(f64, &Value)> {
    group["buckets"]
        .as_array()?
        .iter()
        .filter(|bucket| bucket["reset_time"].is_string())
        .map(|bucket| {
            let remaining = bucket["remaining_fraction"].as_f64().unwrap_or(0.0);
            (remaining.clamp(0.0, 1.0), bucket)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
}

fn derive_status(run_status: Option<&str>, t: &Transcript) -> (SessionStatus, Vec<String>) {
    let running = match run_status {
        Some(s) if !s.is_empty() => !s.ends_with("_IDLE"),
        _ => t.turn_open,
    };
    if !running {
        return (
            SessionStatus::Waiting,
            vec!["waiting for input".to_string()],
        );
    }
    match &t.pending_tool {
        Some(tool) => (SessionStatus::Executing, vec![tool.clone()]),
        None => (SessionStatus::Thinking, vec!["thinking...".to_string()]),
    }
}

fn parse_runs(json: &str) -> Option<HashMap<String, RunInfo>> {
    // sqlite3 -json prints nothing at all for an empty result.
    if json.trim().is_empty() {
        return Some(HashMap::new());
    }
    let rows: Vec<Value> = serde_json::from_str(json).ok()?;
    Some(
        rows.iter()
            .filter_map(|row| {
                let id = row["conversation_id"].as_str()?.to_string();
                let info = RunInfo {
                    status: row["status"].as_str().unwrap_or("").to_string(),
                    title: row["title"].as_str().unwrap_or("").to_string(),
                    top_level: row["parent_conversation_id"]
                        .as_str()
                        .unwrap_or("")
                        .is_empty(),
                };
                Some((id, info))
            })
            .collect(),
    )
}

#[cfg(target_os = "linux")]
fn find_owners(pids: &[u32]) -> Option<HashMap<u32, Owner>> {
    let mut owners = HashMap::new();
    for &pid in pids {
        let conversations: Vec<String> = process::scan_proc_fds(pid)
            .iter()
            .filter_map(|p| conversation_from_lock(&p.to_string_lossy()))
            .collect();
        if !conversations.is_empty() {
            let cwd = fs::read_link(format!("/proc/{pid}/cwd"))
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            owners.insert(pid, Owner { conversations, cwd });
        }
    }
    Some(owners)
}

// Windows has no lsof or /proc to tell which process holds a lock file.
#[cfg(target_os = "windows")]
fn find_owners(_pids: &[u32]) -> Option<HashMap<u32, Owner>> {
    Some(HashMap::new())
}

#[cfg(all(not(target_os = "linux"), not(target_os = "windows")))]
fn find_owners(pids: &[u32]) -> Option<HashMap<u32, Owner>> {
    if pids.is_empty() {
        return Some(HashMap::new());
    }
    let mut command = Command::new("lsof");
    command.args(["-F", "pfn"]);
    for pid in pids {
        command.arg(format!("-p{pid}"));
    }
    let output = command.output().ok()?;
    Some(parse_lsof(&String::from_utf8_lossy(&output.stdout)))
}

/// Parse `lsof -F pfn`, keeping only processes that hold a presence lock.
#[cfg(any(test, all(not(target_os = "linux"), not(target_os = "windows"))))]
fn parse_lsof(stdout: &str) -> HashMap<u32, Owner> {
    let mut owners: HashMap<u32, Owner> = HashMap::new();
    let mut pid = 0;
    let mut fd_is_cwd = false;
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix('p') {
            pid = value.parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix('f') {
            fd_is_cwd = value == "cwd";
        } else if let Some(name) = line.strip_prefix('n') {
            if pid == 0 {
                continue;
            }
            if fd_is_cwd {
                owners.entry(pid).or_default().cwd = name.to_string();
            } else if let Some(id) = conversation_from_lock(name) {
                owners.entry(pid).or_default().conversations.push(id);
            }
        }
    }
    owners.retain(|_, owner| !owner.conversations.is_empty());
    owners
}

/// The id is later joined into a path, so it must be a plain UUID-like token.
fn conversation_from_lock(path: &str) -> Option<String> {
    let id = path
        .split_once("/antigravity-cli/presence/")?
        .1
        .strip_suffix(".lock")?;
    let safe = !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
    safe.then(|| id.to_string())
}

fn collect_children(pid: u32, shared: &super::SharedProcessData) -> Vec<ChildProcess> {
    let mut children = Vec::new();
    let mut stack = shared.children_map.get(&pid).cloned().unwrap_or_default();
    let mut visited = HashSet::new();
    while let Some(child) = stack.pop() {
        if !visited.insert(child) {
            continue;
        }
        if let Some(info) = shared.process_info.get(&child) {
            children.push(ChildProcess {
                pid: child,
                command: info.command.clone(),
                mem_kb: info.rss_kb,
                port: shared.ports.get(&child).and_then(|v| v.first().copied()),
            });
        }
        if let Some(grandchildren) = shared.children_map.get(&child) {
            stack.extend(grandchildren);
        }
    }
    children
}

/// The user's own words, without the metadata agy wraps around them.
fn user_request(content: &str) -> &str {
    content
        .split_once("<USER_REQUEST>")
        .and_then(|(_, rest)| rest.split_once("</USER_REQUEST>"))
        .map_or(content, |(request, _)| request.trim())
}

/// agy records the model only as a note in the user step that changed it.
fn model_from_settings_change(content: &str) -> Option<&str> {
    let (_, rest) = content.split_once("setting `Model Selection` from ")?;
    let (_, rest) = rest.split_once(" to ")?;
    let model = rest
        .split_once(". ")
        .map_or(rest, |(model, _)| model)
        .trim();
    (!model.is_empty()).then_some(model)
}

/// Tool args are JSON-encoded strings; pick the one that says what is touched.
fn tool_arg(args: &Value) -> String {
    const KEYS: &[&str] = &[
        "CommandLine",
        "AbsolutePath",
        "TargetFile",
        "DirectoryPath",
        "SearchPath",
        "Query",
    ];
    let Some(raw) = KEYS.iter().find_map(|key| args[*key].as_str()) else {
        return String::new();
    };
    let decoded = serde_json::from_str::<String>(raw).unwrap_or_else(|_| raw.to_string());
    clean_text(&decoded, MAX_TASK_CHARS)
}

fn tool_label(name: &str, arg: &str) -> String {
    let label = if arg.is_empty() {
        name.to_string()
    } else {
        format!("{name} {arg}")
    };
    label.chars().take(MAX_TASK_CHARS).collect()
}

fn clean_text(text: &str, max_chars: usize) -> String {
    let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    super::sanitize_terminal_text(&super::redact_secrets(&single_line))
        .chars()
        .take(max_chars)
        .collect()
}

fn parse_ts_ms(ts: &str) -> Option<u64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|dt| dt.timestamp_millis().max(0) as u64)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn modified_ms(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: &str = r#"{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","status":"DONE","created_at":"2026-11-09T05:30:01Z","content":"<USER_REQUEST>\nList the files\n</USER_REQUEST>\n<USER_SETTINGS_CHANGE>\nThe user changed setting `Model Selection` from None to Gemini 3.8 Flash (High). No need to comment.\n</USER_SETTINGS_CHANGE>"}"#;
    const TOOL: &str = r#"{"step_index":1,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","created_at":"2026-11-09T05:30:02Z","input_tokens":7030,"cache_read_tokens":10,"output_tokens":442,"tool_calls":[{"name":"run_command","args":{"CommandLine":"\"ls -la\"","toolSummary":"\"Listing\""}}]}"#;
    const ERROR: &str = r#"{"step_index":2,"source":"MODEL","type":"GENERIC","status":"ERROR","created_at":"2026-11-09T05:30:06Z","error":"denied","content":"denied"}"#;
    const ANSWER: &str = r#"{"step_index":3,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","created_at":"2026-11-09T05:30:09Z","input_tokens":100,"cache_read_tokens":0,"output_tokens":5,"content":"OK"}"#;

    fn parsed(lines: &[&str]) -> Transcript {
        let mut transcript = Transcript::default();
        for line in lines {
            transcript.apply_line(line);
        }
        transcript
    }

    #[test]
    fn user_step_gives_prompt_and_model() {
        let t = parsed(&[USER]);
        assert_eq!(t.initial_prompt, "List the files");
        assert_eq!(t.model, "Gemini 3.8 Flash (High)");
        assert_eq!(t.started_at, parse_ts_ms("2026-11-09T05:30:01Z").unwrap());
        assert!(t.turn_open);
    }

    #[test]
    fn response_with_tool_calls_is_not_an_answer() {
        let t = parsed(&[USER, TOOL]);
        assert_eq!(t.last_reply(), 0);
        assert_eq!(t.pending_tool.as_deref(), Some("run_command ls -la"));
        assert_eq!(t.tool_calls[0].arg, "ls -la");
        assert_eq!(
            (t.total_input, t.total_output, t.total_cache_read),
            (7030, 442, 10)
        );
    }

    #[test]
    fn final_answer_sets_last_reply_and_later_steps_keep_it() {
        let mut t = parsed(&[USER, TOOL, ANSWER]);
        let answered = parse_ts_ms("2026-11-09T05:30:09Z").unwrap();
        assert_eq!(t.last_reply(), answered);
        t.mtime_ms = answered + 4_000;
        assert_eq!(t.last_reply(), answered + 4_000, "answer is the last step");
        t.apply_line(USER);
        assert_eq!(
            t.last_reply(),
            answered,
            "mtime now belongs to the new turn"
        );
        assert_eq!(t.turns, 2);
    }

    #[test]
    fn run_status_decides_between_working_and_waiting() {
        let tool = parsed(&[USER, TOOL]);
        let running = Some("CASCADE_RUN_STATUS_RUNNING");
        let idle = Some("CASCADE_RUN_STATUS_IDLE");
        assert_eq!(derive_status(running, &tool).0, SessionStatus::Executing);
        assert_eq!(derive_status(idle, &tool).0, SessionStatus::Waiting);
        assert_eq!(
            derive_status(running, &parsed(&[USER])).0,
            SessionStatus::Thinking
        );
    }

    #[test]
    fn transcript_decides_when_db_has_no_row() {
        assert_eq!(
            derive_status(None, &parsed(&[USER])).0,
            SessionStatus::Thinking
        );
        assert_eq!(
            derive_status(None, &parsed(&[USER, TOOL])).0,
            SessionStatus::Executing
        );
        assert_eq!(
            derive_status(None, &parsed(&[USER, TOOL, ERROR])).0,
            SessionStatus::Waiting
        );
        assert_eq!(
            derive_status(None, &parsed(&[USER, ANSWER])).0,
            SessionStatus::Waiting
        );
    }

    #[test]
    fn lsof_keeps_only_processes_holding_a_presence_lock() {
        let stdout = "p7\nfcwd\nn/work/app\nf56\nn/h/.gemini/antigravity-cli/presence/abc-1.lock\n\
                      p8\nfcwd\nn/work/other\nf3\nn/h/.gemini/antigravity-cli/log/cli.log\n";
        let owners = parse_lsof(stdout);
        assert_eq!(owners.len(), 1);
        assert_eq!(
            owners[&7],
            Owner {
                conversations: vec!["abc-1".to_string()],
                cwd: "/work/app".to_string(),
            }
        );
    }

    #[test]
    fn lock_with_unsafe_id_is_ignored() {
        assert_eq!(
            conversation_from_lock("/h/antigravity-cli/presence/a;b.lock"),
            None
        );
        assert_eq!(
            conversation_from_lock("/h/antigravity-cli/presence/../x.lock"),
            None
        );
    }

    #[test]
    fn secrets_are_redacted_from_tool_args() {
        let args = serde_json::json!({"CommandLine": "\"curl -H ghp_secret123 x\""});
        assert!(!tool_arg(&args).contains("ghp_secret123"));
    }

    #[test]
    fn refresh_waits_for_complete_lines_and_restarts_after_shrink() {
        let path =
            std::env::temp_dir().join(format!("abtop-agy-transcript-{}.jsonl", std::process::id()));
        let (head, tail) = ANSWER.split_at(20);
        fs::write(&path, format!("{USER}\n{head}")).unwrap();
        let mut t = Transcript::default();
        t.refresh(&path);
        assert_eq!(t.turns, 0, "partial line is not parsed");
        fs::write(&path, format!("{USER}\n{head}{tail}\n")).unwrap();
        t.refresh(&path);
        assert_eq!(t.turns, 1);
        t.refresh(&path);
        assert_eq!(t.turns, 1, "unchanged file is not parsed twice");
        fs::write(&path, "\n").unwrap();
        t.refresh(&path);
        assert_eq!(t.turns, 0, "shrunk file resets the state");
        fs::remove_file(&path).unwrap();
    }

    const USAGE: &str = r#"{"conversation_id":"","status":"SUCCESS","command":{"name":"usage","data":{"groups":[
        {"name":"Gemini Models","buckets":[{"id":"gemini-weekly","window":"weekly","remaining_fraction":0.75,"reset_time":"2026-11-16T05:25:16Z"}]},
        {"name":"Claude and GPT models","buckets":[{"id":"3p-weekly","window":"weekly","remaining_fraction":1,"reset_time":"2026-11-16T06:15:35Z"}]}]}}}"#;

    #[test]
    fn usage_maps_model_families_to_gauges() {
        let info = parse_usage(USAGE, 42).unwrap();
        assert_eq!(info.source, "agy");
        assert_eq!(info.updated_at, Some(42));
        assert_eq!(info.five_hour_pct, Some(25.0));
        assert_eq!(info.seven_day_pct, Some(0.0));
        assert_eq!(info.five_hour_window_minutes, Some(WEEK_MINUTES));
        assert_eq!(
            info.seven_day_resets_at,
            parse_ts_ms("2026-11-16T06:15:35Z").map(|ms| ms / 1000)
        );
    }

    #[test]
    fn usage_without_a_fraction_means_exhausted() {
        let json = r#"{"command":{"data":{"groups":[{"name":"Gemini Models","buckets":[{"reset_time":"2026-11-16T05:25:16Z"}]}]}}}"#;
        assert_eq!(parse_usage(json, 0).unwrap().five_hour_pct, Some(100.0));
    }

    #[test]
    fn usage_without_groups_is_ignored() {
        assert!(parse_usage(r#"{"status":"SUCCESS"}"#, 0).is_none());
        assert!(parse_usage("not json", 0).is_none());
    }

    #[test]
    fn exit_is_read_from_the_sessions_own_workspace() {
        let lines: Vec<String> = [
            r#"{"display":"/exit","type":"slash_command","workspace":"/work/a"}"#,
            r#"{"display":"hello","workspace":"/work/b"}"#,
        ]
        .map(String::from)
        .to_vec();
        assert!(history_ends_with_exit(&lines, "/work/a"));
        assert!(!history_ends_with_exit(&lines, "/work/b"));
        assert!(!history_ends_with_exit(&lines, "/work/c"));
    }

    #[test]
    fn print_runs_are_detected_by_flag_only() {
        assert!(is_print_mode("agy -p /usage --output-format json"));
        assert!(is_print_mode("/x/agy --print hi"));
        assert!(!is_print_mode("agy --conversation abc"));
        assert!(!is_print_mode("agy --print-timeout 3s"));
    }

    #[test]
    fn stopped_conversation_is_listed_and_can_be_dismissed() {
        let root = std::env::temp_dir().join(format!("abtop-agy-stop-{}", std::process::id()));
        let mut collector = AgyCollector::with_root(root.clone());
        let transcript = collector.transcript_path("abc");
        fs::create_dir_all(transcript.parent().unwrap()).unwrap();
        fs::write(&transcript, format!("{USER}\n{ANSWER}\n")).unwrap();
        let record = TrackedSession {
            session_id: "abc".to_string(),
            pid: 7,
            cwd: "/work/app".to_string(),
            started_at: 5,
            transcript,
            killed: false,
            exited: false,
            seen_live: true,
        };
        assert!(collector.track_stopped(vec![record]).is_empty());

        let rows = collector.track_stopped(vec![]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, SessionStatus::Killed);
        assert_eq!(rows[0].pid, 0);
        assert_eq!(rows[0].project_name, "app");
        assert!(rows[0].current_tasks[0].starts_with("killed"));

        assert!(crate::collector::AgentCollector::forget_killed(
            &mut collector,
            "abc"
        ));
        assert!(collector.track_stopped(vec![]).is_empty());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn empty_query_result_is_not_a_failure() {
        assert_eq!(parse_runs("").map(|r| r.len()), Some(0));
        assert!(parse_runs("not json").is_none());
    }
}
