//! End-to-end: a real agent loop, a mock provider, and the compression tools.
//!
//! Every other test in `bigctx` exercises a module in isolation. These drive
//! the whole path the way a session does — a model is asked something, it calls
//! `compress`, the tool mutates the shared state, the loop finishes, and the
//! session keeps the result. That is the only way to catch the class of bug
//! that lives *between* modules: a tool writing to a copy of the state, a
//! handle that is cleared too early, a view that is computed before the fold.
//!
//! The provider is a real HTTP server speaking the openai-compat SSE dialect,
//! so the request path (serialization, streaming, tool-call assembly) is the
//! production one.

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::agent::approval;
use crate::agent::bigctx::{BigctxHandle, BigctxSession};
use crate::agent::run_agent;
use crate::agent::{AgentOptions, RunCallbacks, RunRequest};
use crate::providers::{Msg, ResolvedModel};

fn mock_model(port: u16) -> ResolvedModel {
    ResolvedModel {
        provider_name: "mock".into(),
        kind: "openai-compat".into(),
        base_url: format!("http://127.0.0.1:{port}/v1"),
        api_key: Some("sk-x".into()),
        model_id: "m".into(),
        context_window: Some(200_000),
        options: vec![],
    }
}

fn test_opts() -> AgentOptions<'static> {
    AgentOptions {
        max_request_bytes: crate::core::http::MAX_REQUEST_BYTES,
        system: None,
        cwd: std::env::temp_dir(),
        stream: true,
        compact: None,
        reasoning: None,
        hooks: empty_extensions(),
        cache_key: None,
        cache_anchor: None,
        cache_ttl: None,
        goal: None,
    }
}

fn deny_callbacks() -> RunCallbacks<'static> {
    RunCallbacks {
        on_update: Box::leak(Box::new(|_| {})),
        on_approval: Box::leak(Box::new(|_| crate::agent::ApprovalResponse::Deny)),
    }
}

fn empty_extensions() -> &'static crate::agent::ext::Extensions {
    static E: std::sync::OnceLock<crate::agent::ext::Extensions> = std::sync::OnceLock::new();
    E.get_or_init(crate::agent::ext::Extensions::empty)
}

fn read_request(c: &mut std::net::TcpStream) {
    crate::core::testutil::read_request(c)
}

/// One SSE body that answers with a tool call.
fn tool_call_body(call_id: &str, name: &str, arguments: &str) -> String {
    let escaped = serde_json::to_string(arguments).unwrap();
    format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"{call_id}\",\"function\":{{\"name\":\"{name}\",\"arguments\":{escaped}}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

/// One SSE body that answers with plain text.
fn text_body(text: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n\
         data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

fn send_sse(c: &mut std::net::TcpStream, body: &str) {
    let _ = c.write_all(
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    );
}

fn history_with_bulk() -> Vec<Msg> {
    // Enough bulk that folding it is worth a summary, and enough distinct
    // messages that the fold has something to cover.
    let mut history = vec![Msg::user("the task: summarise what we found")];
    for round in 0..8 {
        history.push(Msg::Assistant {
            text: format!("round {round}: reading the auth module in detail"),
            tool_calls: vec![crate::providers::ToolCall {
                id: format!("c{round}"),
                name: "read".into(),
                arguments: json!({"path": "src/auth/verify.rs"}),
            }],
            reasoning: None,
            reasoning_meta: None,
        });
        history.push(Msg::tool_result(
            format!("c{round}"),
            "read",
            "fn verify(token: &str) -> bool { /* a long body */ }\n".repeat(80),
        ));
    }
    history
}

fn bigctx_session(history: &[Msg]) -> BigctxSession {
    let mut config = crate::agent::bigctx::default_config(200_000);
    // the recent window would otherwise protect everything in a short fixture
    config.preserve_recent_messages = 0;
    config.preserve_recent_tokens = 0;
    config.compress.min_range_tokens = 0;

    let scheme = crate::agent::bigctx::IdScheme {
        round: 1,
        first_slot: 0,
    };
    let projection = crate::agent::bigctx::project(history, scheme);
    let mut state = crate::agent::bigctx::create_initial_state();
    let mut next = 1u64;
    crate::agent::bigctx::assign_refs(
        &projection.messages,
        &mut state.message_refs,
        &mut next,
        |_| false,
    )
    .expect("refs");
    state.next_ref_index = next;

    BigctxSession {
        state,
        messages: projection.messages,
        config,
    }
}

/// The whole point: a model calls `compress`, and the block it creates is still
/// there when the loop finishes.
///
/// This is the bug class the unit tests cannot see. The tool writes into the
/// shared handle; the session must read the state back *out* of the handle
/// rather than storing the pre-request snapshot it captured. When that read-back
/// was wrong, every fold was silently discarded at turn end and all 700 unit
/// tests still passed.
#[test]
fn a_compress_call_made_by_the_model_survives_the_turn() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    // What the model asked the tool for, captured from the wire.
    let sent = Arc::new(Mutex::new(Vec::<String>::new()));
    let sent2 = sent.clone();

    let server = std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let n = hits2.fetch_add(1, Ordering::SeqCst);
            let mut c = conn;
            read_request(&mut c);
            match n {
                0 => {
                    // The model folds the middle of the conversation.
                    let args = json!({
                        "content": [{
                            "startId": "m00002",
                            "endId": "m00009",
                            "summary": "The assistant read src/auth/verify.rs eight times and \
                                        found the token check in verify.rs line 42; the decision \
                                        was to keep the 60s connect budget, and the open question \
                                        is whether the retry belongs in the transport layer."
                        }]
                    })
                    .to_string();
                    sent2.lock().unwrap().push(args.clone());
                    send_sse(&mut c, &tool_call_body("call-1", "compress", &args));
                }
                _ => send_sse(&mut c, &text_body("done")),
            }
            if n >= 1 {
                break;
            }
        }
    });

    let history = history_with_bulk();
    let handle = BigctxHandle::new();
    handle.set(bigctx_session(&history));

    let model = mock_model(port);
    let tools = crate::agent::bigctx::bigctx_tools(handle.clone());
    let mut approval_cfg = approval::ApprovalConfig::default();

    let outcome = run_agent(
        RunRequest {
            model: &model,
            tools: &tools,
            prompt: "compress what you have read so far",
            attachments: vec![],
            seed: history.clone(),
            opts: &test_opts(),
        },
        &mut approval_cfg,
        deny_callbacks(),
    )
    .expect("the run must finish");

    server.join().unwrap();

    assert_eq!(hits.load(Ordering::SeqCst), 2, "one tool round then the answer");
    assert_eq!(outcome.final_text, "done");

    // The tool actually ran and produced a block in the shared state.
    let blocks = handle.with(|s| s.state.blocks.clone()).unwrap_or_default();
    assert_eq!(
        blocks.len(),
        1,
        "the compress call must create exactly one block; got {}",
        blocks.len()
    );
    assert_eq!(blocks[0].block_id, "b1");
    assert!(blocks[0].active, "a fresh block is active");
    assert_eq!(
        blocks[0].start_ref.as_deref(),
        Some("m00002"),
        "the block records the ref the model cited"
    );
    assert_eq!(blocks[0].end_ref.as_deref(), Some("m00009"));
    assert!(
        !blocks[0].effective_message_ids.is_empty(),
        "the block must cover the messages it folded"
    );

    // And the tool's reply reached the model as a tool result. The seed already
    // carries eight `read` results, so find the one this tool produced by name.
    let tool_reply = outcome
        .history
        .iter()
        .find_map(|m| match m {
            Msg::ToolResult { name, content, .. } if name == "compress" => Some(content.clone()),
            _ => None,
        })
        .expect("the compress tool result rides the history");
    assert!(
        tool_reply.contains("Compressed:") && tool_reply.contains("b1"),
        "the model must be told which block it created: {tool_reply}"
    );
}

/// A second turn must send the *compressed* view, not the original history.
///
/// This is what makes the whole feature worth anything: the block created in
/// turn one has to shrink the request in turn two. The unit tests check
/// `render`; only this checks that the session asks for it.
#[test]
fn the_next_turn_sends_a_smaller_view_than_the_history() {
    let history = history_with_bulk();

    // Fold everything but the pinned first user message. The state must be the
    // one the session assigned refs to — a fresh state has no refs and every
    // screened range fails to resolve.
    let session = bigctx_session(&history);
    let mut state = session.state.clone();
    let ranges = crate::agent::bigctx::screen(
        &session.messages,
        &session.state,
        &session.config,
    );
    assert!(
        !ranges.compressible.is_empty(),
        "the fixture must offer something to fold"
    );
    let range = &ranges.compressible[0];

    let projection = crate::agent::bigctx::project(
        &history,
        crate::agent::bigctx::IdScheme {
            round: 1,
            first_slot: 0,
        },
    );
    let resolved = crate::agent::bigctx::resolve_boundaries(
        &range.start_ref,
        &range.end_ref,
        &projection.messages,
        &state,
    )
    .expect("the screened range must resolve");

    let protection = crate::agent::bigctx::compute_protection(
        &projection.messages,
        &state,
        &session.config,
    );
    crate::agent::bigctx::apply_compression(
        &mut state,
        &projection.messages,
        &resolved,
        "The assistant read src/auth/verify.rs and found the token check in verify.rs line 42; \
         the decision was to keep the 60s connect budget, and the open question is whether the \
         retry belongs in the transport layer or above it.",
        &protection,
        &session.config,
        None,
        0,
    )
    .expect("the fold must succeed");

    let (view, _) = crate::agent::session::bigctx_view(
        &history,
        Some(&state),
        1,
        Some(200_000),
    );

    assert!(
        view.len() < history.len(),
        "the compressed view must be shorter than the history: {} vs {}",
        view.len(),
        history.len()
    );
    assert!(
        view.iter().any(|m| matches!(m, Msg::Summary { .. })),
        "the folded material must appear as a summary"
    );
    assert_eq!(
        history[0],
        Msg::user("the task: summarise what we found"),
        "the pinned first user message survives"
    );
}

/// The `acp_status` tool must report what the session actually holds.
#[test]
fn status_reports_the_state_the_session_carries() {
    let history = history_with_bulk();
    let session = bigctx_session(&history);
    let handle = BigctxHandle::new();
    handle.set(session);

    let tools = crate::agent::bigctx::bigctx_tools(handle.clone());
    let status = tools
        .iter()
        .find(|t| t.name() == "acp_status")
        .expect("the status tool is registered");

    let mut log = |_| {};
    let out = status.execute(&json!({}), std::path::Path::new("."), &mut log);
    assert!(out.error.is_none(), "{}", out.content);
    assert!(
        out.content.contains("compressible"),
        "status must say what may be folded: {}",
        out.content
    );
    assert!(
        out.content.contains("m0000"),
        "status must quote the refs to cite: {}",
        out.content
    );
}

/// A tool with no session attached must fail loudly rather than silently
/// doing nothing — a model that gets an empty success learns the tool is
/// broken in a much more confusing way.
#[test]
fn the_tools_refuse_to_run_without_a_session() {
    let tools = crate::agent::bigctx::bigctx_tools(BigctxHandle::new());
    let mut log = |_| {};
    for tool in &tools {
        // Arguments each tool would accept, so the refusal comes from the
        // missing session rather than from argument validation.
        let args = match tool.name() {
            "compress" => json!({"content": [{
                "startId": "m00001",
                "endId": "m00002",
                "summary": "a summary long enough to clear the minimum size check"
            }]}),
            "decompress" => json!({"blockId": "b1"}),
            _ => json!({}),
        };
        let out = tool.execute(&args, std::path::Path::new("."), &mut log);
        assert!(
            out.error.is_some(),
            "`{}` must refuse without a session",
            tool.name()
        );
        assert!(
            out.content.contains("No conversation"),
            "`{}` must say why: {}",
            tool.name(),
            out.content
        );
    }
}

/// A session over the band with real foldable mass must produce a nudge, and
/// that nudge must reach the model's view.
///
/// Without this the whole feature is inert: the model is never told it could
/// compress, so it never calls the tool, so no block is ever created.
#[test]
fn a_pressured_session_nudges_and_the_nudge_reaches_the_view() {
    let history = history_with_bulk();
    // The fixture is ~8.6K tokens, so the window has to be well under it for
    // the session to be over the band (nudges start at 45% by default).
    let window = 12_000u64;

    let (view, _state) = crate::agent::session::bigctx_view_with_nudge(
        &history,
        None,
        1,
        Some(window),
    )
    .0;
    let (_view2, nudge) =
        crate::agent::session::bigctx_view_with_nudge(&history, None, 1, Some(window));

    let nudge = nudge.expect("a session over the band with foldable mass must nudge");
    assert!(
        nudge.contains("compress("),
        "the nudge must carry the call shape: {nudge}"
    );
    assert!(
        nudge.contains("KEEP VERBATIM"),
        "the nudge must carry the summary rules"
    );

    // And injecting it changes only the last user message. In this fixture that
    // is also the first one (a single task message at the head), so the check
    // is "the nudge lands on the last user message", not "the head is
    // untouched" — with a longer history the head really would be untouched,
    // and `inject_nudge` walks from the end to guarantee that.
    let mut view = view;
    let before: Vec<Msg> = view.clone();
    crate::agent::session::inject_nudge(&mut view, &nudge);
    let changed: Vec<usize> = before
        .iter()
        .zip(view.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        changed.len(),
        1,
        "exactly one message may change: {changed:?}"
    );
    let last_user_index = view
        .iter()
        .rposition(|m| matches!(m, Msg::User { .. }))
        .expect("a user message exists");
    assert_eq!(
        changed[0], last_user_index,
        "the nudge rides the last user message, not the head"
    );
    assert!(
        view.iter()
            .filter_map(|m| match m {
                Msg::User { text, .. } => Some(text),
                _ => None,
            })
            .any(|t| t.contains("compress(")),
        "the nudge text reached the view"
    );
}

/// Below the band, no nudge — an unthrottled nudge is a fixed cost on every
/// request.
#[test]
fn a_small_session_is_not_nudged() {
    let history = vec![Msg::user("hi"), Msg::assistant("hello")];
    let (_view, nudge) =
        crate::agent::session::bigctx_view_with_nudge(&history, None, 1, Some(1_000_000));
    assert!(nudge.is_none(), "a nearly empty session has nothing to say");
}

/// An empty view has nowhere to put a nudge and must not panic.
#[test]
fn injecting_into_an_empty_view_is_a_no_op() {
    let mut view: Vec<Msg> = Vec::new();
    crate::agent::session::inject_nudge(&mut view, "a nudge");
    assert!(view.is_empty());
}

