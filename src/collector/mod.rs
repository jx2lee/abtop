pub mod agy;
pub mod claude;
pub mod codex;
pub(crate) mod killed;
pub mod mcp;
pub mod opencode;
pub mod process;
pub mod rate_limit;

pub use agy::AgyCollector;
pub use claude::ClaudeCollector;
pub use codex::CodexCollector;
pub use mcp::McpServer;
pub use opencode::OpenCodeCollector;
pub use rate_limit::read_rate_limits;

/// Abbreviate a filesystem path by replacing the home directory prefix with `~`.
pub(crate) fn abbrev_path(path: &std::path::Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(rel) = path.strip_prefix(&home) {
            return format!("~/{}", rel.display());
        }
    }
    path.to_string_lossy().into_owned()
}

/// Redact common secret patterns to avoid displaying credentials in the TUI.
/// Replaces the prefix and all following non-whitespace chars with [REDACTED].
/// Best-effort: covers well-known prefixed tokens, not arbitrary high-entropy strings.
pub(crate) fn redact_secrets(s: &str) -> String {
    const PATTERNS: &[&str] = &[
        // Anthropic / OpenAI / OpenRouter
        "sk-ant-",
        "sk-proj-",
        "sk-or-",
        // Stripe
        "sk_live_",
        "sk_test_",
        "rk_live_",
        "rk_test_",
        // GitHub
        "ghp_",
        "gho_",
        "ghs_",
        "ghr_",
        "ghu_",
        "github_pat_",
        // GitLab
        "glpat-",
        // Slack
        "xoxb-",
        "xoxp-",
        "xoxa-",
        "xoxs-",
        // AWS access key id
        "AKIA",
        "ASIA",
        // Bearer-prefixed headers
        "Bearer ",
    ];
    let mut result = s.to_string();
    for pat in PATTERNS {
        while let Some(pos) = result.find(pat) {
            let end = result[pos..]
                .find(char::is_whitespace)
                .map(|i| pos + i)
                .unwrap_or(result.len());
            result.replace_range(pos..end, "[REDACTED]");
        }
    }
    result
}

/// Strip control characters and Unicode bidi override/isolate marks before
/// transcript text is stored for terminal rendering.
pub(crate) fn sanitize_terminal_text(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !c.is_control()
                && !matches!(
                    *c,
                    '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200E}' | '\u{200F}'
                )
        })
        .collect()
}

use crate::model::{AgentSession, OrphanPort, RateLimitInfo, SessionStatus};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    mpsc::{self, Receiver, Sender},
    Arc,
};
use std::time::{Duration, Instant};

/// Trait for agent-specific session collectors.
/// Implement this to add support for a new AI coding agent.
pub trait AgentCollector {
    /// Return all live sessions for this agent type.
    fn collect(&mut self, shared: &SharedProcessData) -> Vec<AgentSession>;

    /// Return agent-specific rate limit info, if available from session data.
    fn live_rate_limit(&self) -> Option<RateLimitInfo> {
        None
    }

    /// Start a background quota refresh if one is due. Only the TUI calls
    /// this, so headless consumers never spawn the agent.
    fn refresh_quota(&mut self) {}

    /// Return config directories discovered from running agent processes.
    /// Used to feed rate limit lookups across all active config dirs.
    fn discovered_config_dirs(&self) -> Vec<std::path::PathBuf> {
        Vec::new()
    }

    /// Stop listing a Killed session. Returns true if this collector owned it.
    fn forget_killed(&mut self, _session_id: &str) -> bool {
        false
    }
}

/// Process data fetched once per tick and shared across all collectors.
/// Avoids duplicate ps/lsof calls.
pub struct SharedProcessData {
    pub process_info: HashMap<u32, process::ProcInfo>,
    pub children_map: HashMap<u32, Vec<u32>>,
    pub ports: HashMap<u32, Vec<u16>>,
    /// True on slow poll ticks (every 5 ticks ≈ 10s). Collectors should
    /// defer expensive discovery (e.g. /proc reads) to slow ticks.
    pub slow_tick: bool,
    /// PIDs of detected codex mcp-server processes. Populated by
    /// `MultiCollector` after McpDetection runs; CodexCollector
    /// excludes these so a single mcp-server PID isn't double-counted
    /// in the sessions panel.
    pub mcp_server_pids: HashSet<u32>,
    /// Rollout file paths held open by an mcp-server process. The
    /// CodexCollector "recently finished" pass skips these to avoid
    /// PID=0 ghost rows for threads that the mcp-server is still
    /// holding fds for.
    pub mcp_owned_rollouts: HashSet<PathBuf>,
    /// When false, the suppression sets above are empty so the
    /// sessions panel restores upstream behavior. Driven by the user
    /// toggle (Shift+M).
    pub mcp_suppress: bool,
    /// Cached Desktop app-server PID -> open rollout files. This is populated
    /// by a background scanner so slow macOS lsof calls cannot block the TUI.
    pub desktop_rollout_fd_map: HashMap<u32, Vec<PathBuf>>,
}

/// Context window size for this model (e.g. 200K, 1M).
pub(crate) fn context_window_for_model(
    transcript_model: &str,
    configured_model: &str,
    max_context_tokens: u64,
) -> u64 {
    if transcript_model.contains("[1m]")
        || configured_model.contains("[1m]")
        || max_context_tokens > 200_000
    {
        1_000_000
    } else {
        200_000
    }
}

impl SharedProcessData {
    /// Fetch process info. `ports` comes from the [`PortScanner`].
    pub fn fetch(ports: HashMap<u32, Vec<u16>>, slow_tick: bool) -> Self {
        let process_info = process::get_process_info();
        let children_map = process::get_children_map(&process_info);
        Self {
            process_info,
            children_map,
            ports,
            slow_tick,
            mcp_server_pids: HashSet::new(),
            mcp_owned_rollouts: HashSet::new(),
            mcp_suppress: true,
            desktop_rollout_fd_map: HashMap::new(),
        }
    }
}

type PortScanResult = (Vec<u32>, HashMap<u32, Vec<u16>>);

/// Runs the listening-port scan (`lsof` on macOS, about 250ms) off the UI
/// thread. The TUI handles keys on the thread that ticks, so an inline scan
/// on nearly every tick (any new process changes the PID set) is input lag.
/// The first scan runs inline so one-shot snapshots still list ports; later
/// ones run in the background while the previous result is served.
struct PortScanner {
    cached: HashMap<u32, Vec<u16>>,
    /// Sorted PID set when the cached scan started.
    cached_pids: Vec<u32>,
    scanned_once: bool,
    in_flight: bool,
    tx: Sender<PortScanResult>,
    rx: Receiver<PortScanResult>,
}

impl PortScanner {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            cached: HashMap::new(),
            cached_pids: Vec::new(),
            scanned_once: false,
            in_flight: false,
            tx,
            rx,
        }
    }

    /// Ports of live PIDs. `pids` must be sorted. `force` rescans even when
    /// the PID set has not changed.
    fn update(&mut self, pids: &[u32], force: bool) -> HashMap<u32, Vec<u16>> {
        while let Ok((scanned_pids, ports)) = self.rx.try_recv() {
            self.in_flight = false;
            self.cached_pids = scanned_pids;
            self.cached = ports;
        }
        if !self.scanned_once {
            self.scanned_once = true;
            self.cached_pids = pids.to_vec();
            self.cached = process::get_listening_ports();
        } else if !self.in_flight && (force || self.cached_pids != pids) {
            self.in_flight = true;
            let pids = pids.to_vec();
            let tx = self.tx.clone();
            std::thread::spawn(move || {
                let _ = tx.send((pids, process::get_listening_ports()));
            });
        }
        // A PID that exited since the scan no longer owns its ports.
        self.cached
            .iter()
            .filter(|(pid, _)| pids.binary_search(pid).is_ok())
            .map(|(pid, ports)| (*pid, ports.clone()))
            .collect()
    }
}

/// Info about a child process that owns an open port, tracked for orphan detection.
#[derive(Clone)]
struct TrackedPortChild {
    port: u16,
    command: String,
    project_name: String,
}

struct DesktopRolloutScanResult {
    pids: Vec<u32>,
    rollouts: Option<HashMap<u32, Vec<PathBuf>>>,
}

struct DesktopRolloutScanner {
    cached: HashMap<u32, Vec<PathBuf>>,
    cached_pids: Vec<u32>,
    in_flight_pids: Option<Vec<u32>>,
    child_pid: Arc<AtomicU32>,
    last_started: Option<Instant>,
    tx: Sender<DesktopRolloutScanResult>,
    rx: Receiver<DesktopRolloutScanResult>,
}

const DESKTOP_ROLLOUT_SCAN_TIMEOUT: Duration = Duration::from_secs(90);
const DESKTOP_ROLLOUT_RESCAN_INTERVAL: Duration = Duration::from_secs(60);

impl DesktopRolloutScanner {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            cached: HashMap::new(),
            cached_pids: Vec::new(),
            in_flight_pids: None,
            child_pid: Arc::new(AtomicU32::new(0)),
            last_started: None,
            tx,
            rx,
        }
    }

    fn update(&mut self, pids: &[u32]) -> HashMap<u32, Vec<PathBuf>> {
        self.poll_completed();
        if self.should_start(pids) {
            self.start(pids.to_vec());
        }
        self.cached_for(pids)
    }

    fn poll_completed(&mut self) {
        while let Ok(result) = self.rx.try_recv() {
            self.apply_result(result);
        }
    }

    fn apply_result(&mut self, result: DesktopRolloutScanResult) {
        self.in_flight_pids = None;
        if let Some(rollouts) = result.rollouts {
            self.cached_pids = result.pids;
            self.cached = rollouts;
        }
    }

    fn should_start(&self, pids: &[u32]) -> bool {
        if pids.is_empty() || self.in_flight_pids.is_some() {
            return false;
        }
        if self.cached_pids != pids {
            return true;
        }
        self.last_started
            .is_none_or(|started| started.elapsed() >= DESKTOP_ROLLOUT_RESCAN_INTERVAL)
    }

    fn start(&mut self, pids: Vec<u32>) {
        self.in_flight_pids = Some(pids.clone());
        self.last_started = Some(Instant::now());
        let tx = self.tx.clone();
        let child_pid = self.child_pid.clone();
        std::thread::spawn(move || {
            let rollouts = mcp::map_pid_to_rollouts_with_timeout_and_pid_slot(
                &pids,
                DESKTOP_ROLLOUT_SCAN_TIMEOUT,
                Some(child_pid),
            );
            let _ = tx.send(DesktopRolloutScanResult { pids, rollouts });
        });
    }

    fn cached_for(&self, pids: &[u32]) -> HashMap<u32, Vec<PathBuf>> {
        let mut map = HashMap::new();
        for pid in pids {
            if let Some(paths) = self.cached.get(pid) {
                map.insert(*pid, paths.clone());
            }
        }
        map
    }
}

impl Drop for DesktopRolloutScanner {
    fn drop(&mut self) {
        let pid = self.child_pid.swap(0, Ordering::SeqCst);
        mcp::kill_rollout_scan_child(pid);
    }
}

/// Aggregates sessions from multiple collectors (Claude, Codex, etc.)
pub struct MultiCollector {
    collectors: Vec<Box<dyn AgentCollector>>,
    codex_enabled: bool,
    tick_count: u32,
    port_scanner: PortScanner,
    cached_git: HashMap<String, (u32, u32)>,
    /// Port-owning children from previous ticks, keyed by child PID.
    /// Used to detect orphans when a session dies.
    tracked_port_children: HashMap<u32, TrackedPortChild>,
    /// Detected orphan ports (updated each tick).
    pub orphan_ports: Vec<OrphanPort>,
    /// MCP servers (codex mcp-server) detected on the most recent tick.
    pub mcp_servers: Vec<McpServer>,
    /// Whether to hide mcp-server-owned rollouts from the sessions
    /// panel. When `false`, sessions panel reverts to upstream
    /// behavior (mcp-server PIDs and their rollouts appear there too,
    /// with the existing 1-of-N HashMap-overwrite caveat).
    pub mcp_suppress: bool,
    /// Whether Killed sessions are returned from `collect` (toggle: Shift+S).
    pub show_killed: bool,
    desktop_rollout_scanner: DesktopRolloutScanner,
}

/// How often to refresh expensive I/O (in ticks). 5 ticks × 2s = 10s.
const SLOW_POLL_INTERVAL: u32 = 5;

impl MultiCollector {
    /// Build a collector, skipping agents whose identifier is in `hidden`.
    /// Identifiers are matched case-insensitively against each collector's
    /// `agent_cli` name (e.g. `"claude"`, `"codex"`).
    #[cfg(test)]
    pub fn with_hidden(hidden: &[String]) -> Self {
        Self::with_hidden_and_claude_config_dirs(hidden, &[])
    }

    pub fn with_hidden_and_claude_config_dirs(
        hidden: &[String],
        claude_config_dirs: &[PathBuf],
    ) -> Self {
        let is_hidden = |name: &str| hidden.iter().any(|h| h.eq_ignore_ascii_case(name));
        let mut collectors: Vec<Box<dyn AgentCollector>> = Vec::new();
        if !is_hidden("claude") {
            collectors.push(Box::new(ClaudeCollector::with_configured_dirs(
                claude_config_dirs.to_vec(),
            )));
        }
        if !is_hidden("codex") {
            collectors.push(Box::new(CodexCollector::new()));
        }
        if !is_hidden("opencode") {
            collectors.push(Box::new(OpenCodeCollector::new()));
        }
        if !is_hidden("agy") {
            collectors.push(Box::new(AgyCollector::new()));
        }
        let codex_enabled = !is_hidden("codex");
        Self {
            collectors,
            codex_enabled,
            tick_count: SLOW_POLL_INTERVAL, // trigger on first tick
            port_scanner: PortScanner::new(),
            cached_git: HashMap::new(),
            tracked_port_children: HashMap::new(),
            orphan_ports: Vec::new(),
            mcp_servers: Vec::new(),
            mcp_suppress: true,
            show_killed: false,
            desktop_rollout_scanner: DesktopRolloutScanner::new(),
        }
    }

    pub fn set_mcp_suppress(&mut self, on: bool) {
        self.mcp_suppress = on;
    }

    pub fn set_show_killed(&mut self, on: bool) {
        self.show_killed = on;
    }

    /// Stop listing a Killed session. Returns true if any collector owned it.
    pub fn forget_killed(&mut self, session_id: &str) -> bool {
        self.collectors
            .iter_mut()
            .any(|c| c.forget_killed(session_id))
    }

    /// Let collectors whose quota needs a subprocess refresh it in the background.
    pub fn refresh_agent_quotas(&mut self) {
        for collector in &mut self.collectors {
            collector.refresh_quota();
        }
    }

    /// Collect rate limit info from all registered collectors.
    pub fn agent_rate_limits(&self) -> Vec<RateLimitInfo> {
        self.collectors
            .iter()
            .filter_map(|c| c.live_rate_limit())
            .collect()
    }

    /// Return all config directories discovered across all collectors.
    pub fn all_config_dirs(&self) -> Vec<std::path::PathBuf> {
        self.collectors
            .iter()
            .flat_map(|c| c.discovered_config_dirs())
            .collect()
    }

    pub fn collect(&mut self) -> Vec<AgentSession> {
        let slow_tick = self.tick_count >= SLOW_POLL_INTERVAL;
        if slow_tick {
            self.tick_count = 0;
        }
        self.tick_count += 1;

        // Ports: rescan on slow tick or when the PID set changes (PID reuse safety).
        let mut shared = SharedProcessData::fetch(HashMap::new(), slow_tick);
        let mut current_pids: Vec<u32> = shared.process_info.keys().copied().collect();
        current_pids.sort_unstable();
        shared.ports = self.port_scanner.update(&current_pids, slow_tick);

        // Detect MCP servers and stash the suppression sets in `shared`
        // so CodexCollector can avoid double-counting their rollouts.
        let detection = mcp::detect(&shared.process_info);
        self.mcp_servers = detection.servers;
        shared.mcp_suppress = self.mcp_suppress;
        if self.mcp_suppress {
            shared.mcp_server_pids = detection.server_pids;
            shared.mcp_owned_rollouts = detection.owned_rollouts;
        }

        if self.codex_enabled {
            let desktop_pids = CodexCollector::find_codex_desktop_pids_from_shared(
                &shared.process_info,
                &shared.mcp_server_pids,
            );
            shared.desktop_rollout_fd_map = self.desktop_rollout_scanner.update(&desktop_pids);
        }

        let mut all = Vec::new();
        for collector in &mut self.collectors {
            all.extend(collector.collect(&shared));
        }

        // Hide dead sessions: Codex uses pid==0 sentinel, Claude is filtered in collect().
        // Killed sessions are only listed while the Shift+S toggle is on.
        let show_killed = self.show_killed;
        all.retain(|s| match s.status {
            SessionStatus::Done => false,
            SessionStatus::Killed => show_killed,
            _ => true,
        });

        // Git stats: refresh on slow tick; between those, compute only cwds
        // not yet cached so a new session does not show a false clean.
        // Runs after the filter so hidden sessions cost nothing.
        if slow_tick {
            self.cached_git.clear();
        }
        let missing: Vec<&str> = all
            .iter()
            .map(|s| s.cwd.as_str())
            .filter(|cwd| !self.cached_git.contains_key(*cwd))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        self.cached_git.extend(collect_git_stats_for(&missing));
        for s in &mut all {
            if let Some(&(added, modified)) = self.cached_git.get(&s.cwd) {
                s.git_added = added;
                s.git_modified = modified;
            }
        }
        sort_sessions(&mut all);

        // --- Orphan port detection ---
        // 1. Update tracked port children from live sessions
        let mut live_child_pids = std::collections::HashSet::new();
        for s in &all {
            if !matches!(s.status, SessionStatus::Done) {
                for child in &s.children {
                    live_child_pids.insert(child.pid);
                    if let Some(port) = child.port {
                        self.tracked_port_children.insert(
                            child.pid,
                            TrackedPortChild {
                                port,
                                command: child.command.clone(),
                                project_name: s.project_name.clone(),
                            },
                        );
                    }
                }
            }
        }

        // 2. Detect orphans: tracked PIDs that are no longer children of any live session
        //    but are still alive and have an open port
        self.orphan_ports.clear();
        let mut stale_pids = Vec::new();
        for (pid, tracked) in &self.tracked_port_children {
            if live_child_pids.contains(pid) {
                continue; // still owned by a live session
            }
            // Check if process is still alive and still has the port open
            let still_listening = shared
                .ports
                .get(pid)
                .is_some_and(|ports| ports.contains(&tracked.port));
            let still_alive = shared.process_info.contains_key(pid);
            if still_alive && still_listening {
                self.orphan_ports.push(OrphanPort {
                    port: tracked.port,
                    pid: *pid,
                    command: tracked.command.clone(),
                    project_name: tracked.project_name.clone(),
                });
            } else {
                stale_pids.push(*pid);
            }
        }
        // Clean up dead tracked entries
        for pid in stale_pids {
            self.tracked_port_children.remove(&pid);
        }
        self.orphan_ports.sort_by_key(|o| o.port);

        all
    }
}

/// Max `git status` processes run at once.
const GIT_STATS_PARALLELISM: usize = 8;

/// Run `git status` once per cwd, several at a time, so a tick waits for
/// about one `git status` instead of one per session. The TUI handles keys
/// on the same thread, so this wait is input lag.
fn collect_git_stats_for(cwds: &[&str]) -> Vec<(String, (u32, u32))> {
    let mut stats = Vec::with_capacity(cwds.len());
    for chunk in cwds.chunks(GIT_STATS_PARALLELISM) {
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|&cwd| scope.spawn(move || (cwd.to_string(), process::collect_git_stats(cwd))))
                .collect();
            stats.extend(handles.into_iter().filter_map(|h| h.join().ok()));
        });
    }
    stats
}

/// Live sessions first, then Killed ones; within each group, the most recent
/// finished answer first. A session with no answer yet ranks by its start time.
fn sort_sessions(sessions: &mut [AgentSession]) {
    sessions.sort_by_key(|s| {
        let recency = if s.last_reply_ms > 0 {
            s.last_reply_ms
        } else {
            s.started_at
        };
        (
            matches!(s.status, SessionStatus::Killed),
            std::cmp::Reverse(recency),
        )
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort_row(
        id: &str,
        started_at: u64,
        last_reply_ms: u64,
        status: SessionStatus,
    ) -> AgentSession {
        let mut app = crate::app::App::new_with_config(
            crate::theme::Theme::default(),
            &[],
            crate::config::PanelVisibility::default(),
        );
        crate::demo::populate_demo(&mut app);
        let mut s = app.sessions.swap_remove(0);
        s.session_id = id.to_string();
        s.started_at = started_at;
        s.last_reply_ms = last_reply_ms;
        s.status = status;
        s
    }

    #[test]
    fn sort_puts_latest_reply_first_and_killed_last() {
        let mut sessions = vec![
            sort_row("old-start-new-reply", 10, 500, SessionStatus::Waiting),
            sort_row("killed-recent", 90, 900, SessionStatus::Killed),
            sort_row("new-start-old-reply", 80, 200, SessionStatus::Waiting),
            sort_row("no-reply-yet", 300, 0, SessionStatus::Waiting),
            sort_row("killed-old", 95, 100, SessionStatus::Killed),
        ];
        sort_sessions(&mut sessions);
        let order: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(
            order,
            [
                "old-start-new-reply",
                "no-reply-yet",
                "new-start-old-reply",
                "killed-recent",
                "killed-old",
            ]
        );
    }

    #[test]
    fn with_hidden_empty_keeps_all_collectors() {
        let mc = MultiCollector::with_hidden(&[]);
        assert_eq!(mc.collectors.len(), 4);
    }

    #[test]
    fn with_hidden_codex_drops_codex_only() {
        let mc = MultiCollector::with_hidden(&["codex".to_string()]);
        assert_eq!(mc.collectors.len(), 3);
    }

    #[test]
    fn with_hidden_is_case_insensitive() {
        let mc = MultiCollector::with_hidden(&["CODEX".to_string()]);
        assert_eq!(mc.collectors.len(), 3);
        let mc = MultiCollector::with_hidden(&["Claude".to_string()]);
        assert_eq!(mc.collectors.len(), 3);
    }

    #[test]
    fn with_hidden_unknown_names_are_ignored() {
        let mc = MultiCollector::with_hidden(&["kiro".to_string(), "gemini".to_string()]);
        assert_eq!(mc.collectors.len(), 4);
    }

    #[test]
    fn with_hidden_all_agents_yields_empty() {
        let mc = MultiCollector::with_hidden(&[
            "claude".to_string(),
            "codex".to_string(),
            "opencode".to_string(),
            "agy".to_string(),
        ]);
        assert!(mc.collectors.is_empty());
    }

    #[test]
    fn desktop_rollout_scanner_keeps_cache_on_failed_result() {
        let mut scanner = DesktopRolloutScanner::new();
        let mut cached = HashMap::new();
        cached.insert(42, vec![PathBuf::from("/tmp/rollout-live.jsonl")]);

        scanner.apply_result(DesktopRolloutScanResult {
            pids: vec![42],
            rollouts: Some(cached.clone()),
        });
        scanner.apply_result(DesktopRolloutScanResult {
            pids: vec![42],
            rollouts: None,
        });

        assert_eq!(scanner.cached_for(&[42]), cached);
    }

    #[test]
    fn desktop_rollout_scanner_in_flight_guard_blocks_duplicate_scan() {
        let mut scanner = DesktopRolloutScanner::new();
        scanner.in_flight_pids = Some(vec![42]);

        assert!(!scanner.should_start(&[42]));
    }

    #[test]
    fn port_scanner_serves_cache_without_waiting_and_drops_exited_pids() {
        let mut scanner = PortScanner::new();
        // A scan is already running, so `update` must not start another
        // one or wait for it.
        scanner.scanned_once = true;
        scanner.in_flight = true;
        scanner.cached_pids = vec![1, 2];
        scanner.cached = HashMap::from([(1, vec![3000]), (2, vec![4000])]);

        // PID 2 exited after the scan: its port is no longer reported.
        assert_eq!(
            scanner.update(&[1, 3], false),
            HashMap::from([(1, vec![3000])])
        );
        assert!(scanner.in_flight);

        // The background result replaces the cache on the next update.
        scanner
            .tx
            .send((vec![1, 3], HashMap::from([(3, vec![5000])])))
            .unwrap();
        assert_eq!(
            scanner.update(&[1, 3], false),
            HashMap::from([(3, vec![5000])])
        );
        assert!(!scanner.in_flight);
        assert_eq!(scanner.cached_pids, vec![1, 3]);
    }

    #[test]
    fn git_stats_cover_every_cwd_across_chunks() {
        let cwds: Vec<String> = (0..GIT_STATS_PARALLELISM * 2 + 1)
            .map(|i| format!("/nonexistent/abtop-git-stats-{i}"))
            .collect();
        let refs: Vec<&str> = cwds.iter().map(String::as_str).collect();

        let stats: HashMap<String, (u32, u32)> = collect_git_stats_for(&refs).into_iter().collect();

        assert_eq!(stats.len(), cwds.len());
        assert!(stats.values().all(|&s| s == (0, 0)));
    }
}
