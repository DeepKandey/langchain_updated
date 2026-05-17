import os
import json
import logging
import re
import shutil
import threading
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, Optional
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry

# langgraph
from langgraph.graph import StateGraph,START,END

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_ollama import ChatOllama

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# NEW: Token counting helpers
#
# tiktoken uses cl100k_base (GPT-4 / Claude-compatible BPE encoding).
# Gemma uses SentencePiece so counts will be ~10-15% off, but this is
# still far more accurate than character counting and well within the
# safety margin you leave in max_context_tokens.
#
# The encoder is loaded once at module level (it caches internally) so
# repeated calls to count_tokens() are essentially free after the first.
# -----------------------------------------------------------------------
 
_TIKTOKEN_ENCODING = "cl100k_base"
_enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)

def count_tokens(text: str) -> int:
    """Return the approximate token count for *text* using tiktoken BPE."""
    return len(_enc.encode(text))
 
 
def trim_to_token_budget(text: str, token_budget: int) -> str:
    """
    Trim *text* so it fits within *token_budget* tokens.
 
    Uses tiktoken encode/decode so the result is always valid UTF-8 text
    with no mid-codepoint splits (unlike a raw character slice).
    Returns the original string unchanged when it already fits.
    """
    tokens = _enc.encode(text)
    if len(tokens) <= token_budget:
        return text
    return _enc.decode(tokens[:token_budget])
 
 
def select_go_files_within_budget(
    files: dict[str, str],
    token_budget: int,
) -> tuple[dict[str, str], list[str]]:
    """
    NEW: Greedily select whole Go source files until the token budget is exhausted.
 
    Why whole files instead of a raw slice:
    - A file truncated mid-function is nearly useless to the LLM.
    - Knowing "file X was omitted" is actionable; a truncated file is silent data loss.
 
    Files are processed in sorted order (consistent, deterministic).
    The caller should pre-sort by relevance if priority ordering matters.
 
    Returns
    -------
    selected : dict[str, str]
        Files that fit within the budget (whole, unmodified).
    dropped  : list[str]
        Names of files that were excluded because they would exceed the budget.
    """
    selected: dict[str, str] = {}
    dropped: list[str] = []
    used = 0
 
    for name, content in sorted(files.items()):
        cost = count_tokens(content)
        if used + cost <= token_budget:
            selected[name] = content
            used += cost
        else:
            dropped.append(name)
            log.warning(
                "select_go_files_within_budget: dropping '%s' (%d tokens) "
                "— would exceed budget of %d (used so far: %d)",
                name, cost, token_budget, used,
            )
 
    log.info(
        "select_go_files_within_budget: kept %d files (%d tokens used / %d budget), "
        "dropped %d files: %s",
        len(selected), used, token_budget,
        len(dropped), dropped or "none",
    )
    return selected, dropped

# -----------------------------------------------------------------------
# Prompt templates — all LLM prompt content lives here, not in node fns
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """A single system + human prompt pair for one LLM node."""
    system: str
    human: str


@dataclass(frozen=True)
class PromptTemplates:
    """
    All prompt pairs used by the graph, centralised in one place so they can be:
      - swapped in tests without touching node logic
      - loaded from external files at startup
      - reviewed and versioned independently of pipeline code
    """

    summarize_diff: PromptTemplate = field(default_factory=lambda: PromptTemplate(
        system=textwrap.dedent("""\
            You are a senior QA/test automation engineer reviewing a Git diff for a Go codebase.
            Your output will be consumed by a downstream risk analysis step — not read by a human directly.

            Summarize ONLY the changes that affect observable behaviour or testability:
            - New or modified branches, conditions, and control flow paths
            - Validation rule changes (added, removed, or relaxed constraints)
            - Error handling changes (new error types, swallowed errors, changed propagation)
            - API contract changes (new/removed/renamed parameters, changed return shapes)
            - Concurrency, retry, timeout, and backoff changes
            - Any change to default values or fallback behaviour

            Omit:
            - Cosmetic changes (formatting, comments, variable renames with no semantic effect)
            - Test file changes
            - Dependency version bumps unless they change a behaviour your tests would cover

            Format: one bullet per distinct behaviour change. Be specific — name the function,
            field, or code path affected. Do not summarise at file level.
        """),
        human=textwrap.dedent("""\
            ## Git Diff
            {diff_text}

            Produce a concise, test-focused summary of behaviour changes only.
            One bullet per change. Name the specific function or code path affected.
        """)
    ))

    risk_analysis: PromptTemplate = field(default_factory=lambda: PromptTemplate(
        system=textwrap.dedent("""\
            You are a senior Go engineer and test automation specialist conducting a pre-merge
            risk analysis. Your output will directly drive test case generation in the next step.

            For the provided diff and Go source files, produce a structured risk analysis
            covering ALL of the following sections. Do not skip a section even if you believe
            the risk is low — state it explicitly as low and briefly explain why.

            ## Sections (use these exact headings)

            ### Critical paths affected
            Which execution paths are changed? What is the blast radius if they regress?

            ### Edge cases and boundary conditions
            Inputs at or beyond valid limits. Nil/zero/empty values. Off-by-one conditions.
            Unsigned/signed boundary crossings. Empty collections vs nil collections in Go.

            ### Error handling regressions
            Errors that are newly swallowed, newly wrapped, or have changed propagation.
            Functions that previously returned errors now panic, or vice versa.

            ### Backward compatibility risks
            Any change to exported function signatures, struct fields, interface methods,
            or wire formats (JSON tags, protobuf field numbers). Flag any silent breakage.

            ### Concurrency and safety concerns
            New goroutines, channel operations, shared state mutations without locks,
            context cancellation handling, retry storms, timeout changes.

            ### Dependency on external state
            Changes that assume specific database state, environment variables, file system
            layout, or third-party API behaviour that tests would need to stub or mock.

            Be specific: name the function, struct, or line of logic at risk.
            Do not make general statements like "error handling should be tested" —
            say which error path in which function is at risk and why.
        """),
        human=textwrap.dedent("""\
            ## Go Source Files (pre-change context)
            {go_files_text}

            ---

            ## Git Diff (the changes under review)
            {diff_text}

            ---

            Produce the risk analysis using the exact section headings specified.
            Be specific — name functions, types, and code paths.
            Flag low-risk sections explicitly rather than omitting them.
        """)
    ))

    test_generation: PromptTemplate = field(default_factory=lambda: PromptTemplate(
        system=textwrap.dedent("""\
            You are a senior Go engineer specialising in test automation.
            Generate a comprehensive test scenario table for the provided changes.

            ## Output format

            Produce a single markdown table with these exact columns:

            | # | Test scenario | Function / component under test | Input / precondition | Expected outcome | Risk addressed | Priority |
            |---|---|---|---|---|---|---|

            Column definitions:

            - #: Sequential row number.
            - Test scenario: A plain English description of what is being tested and why it matters.
            - Function / component under test: The Go function, method, or API endpoint being exercised.
            - Input / precondition: The specific input values, state, or setup required. Flag external
              dependencies as [requires stub: <what>].
            - Expected outcome: What a manual tester should observe from the OUTSIDE — describe the
              API response, error message shown to the caller, HTTP status code, or observable system
              state. Do NOT mention internal function names, Go error types, or return values.
              Write as if explaining to someone who cannot see the source code.
              Good example: "The request fails with HTTP 400 and a message saying the commit ID is missing."
              Bad example: "fetchCommitDiff returns ErrInvalidCommit."
            - Risk addressed: The exact section heading from the risk analysis this scenario covers.
            - Priority: P0 (regression/critical path), P1 (edge case), P2 (nice to have).

            ## Coverage requirements

            You MUST include at least one scenario for each of the following where applicable:

            1. Happy path — the primary success case for each changed function
            2. Nil / zero-value inputs — especially for pointer receivers and interface params in Go
            3. Empty vs nil slice/map — Go distinguishes these; tests often do not
            4. Error path — every new or modified error return must have a scenario
            5. Boundary conditions — off-by-one, max/min values, empty strings
            6. Concurrent access — if the changed code touches shared state or spawns goroutines
            7. Backward compatibility — if exported signatures changed, confirm callers still compile
            8. Context cancellation — if context.Context is passed, test cancellation and timeout

            ## Constraints

            - Do NOT generate Go test code or implementation examples
            - Do NOT repeat scenarios that are identical in intent even if the wording differs
            - DO flag scenarios that require external dependencies (DB, API, file system) with a
              note in the Input column: [requires stub: <what>]
            - Write ALL expected outcomes in plain English from the caller's perspective —
              no internal function names, no Go type names, no stack traces
            - Base scenarios on the risk analysis provided — every P0 risk must have at least
              one corresponding P0 scenario
        """),
        human=textwrap.dedent("""\
            ## Go Source Files
            {go_files_text}

            ---

            ## Git Diff (recent changes)
            {diff_text}

            ---

            ## Risk Analysis
            {risk_analysis}

            ---

            ## Example of the expected outcome style required

            | # | Test scenario | Function / component under test | Input / precondition | Expected outcome | Risk addressed | Priority |
            |---|---|---|---|---|---|---|
            | 1 | Valid commit with single page of diffs is fetched successfully | fetchCommitDiff | A commit ID with 3 changed files, GitLab returns one page | All 3 file diffs are returned. No error is reported to the caller. | Critical paths affected | P0 |
            | 2 | Commit with no parent fails gracefully | getParentCommitId | A root commit with no parent in the API response | The operation fails with a clear message saying the commit has no parent. No diff is fetched. | Error handling regressions | P0 |
            | 3 | File fetch falls back to current commit when parent file is missing | fetchGoFilesNode | old_path returns 404 from GitLab, new_path returns 200 | The file is fetched from the current commit instead and appears in the output. A warning is logged. [requires stub: GitLab file API] | Critical paths affected | P1 |

            Note: each Expected outcome describes what the caller or observer sees —
            not which internal function returned what value.

            ---

            Generate the test scenario table following this exact style.
            Every P0 risk from the analysis must map to at least one P0 scenario.
            Flag scenarios requiring stubs or mocks in the Input column.
        """)
    ))

# -----------------------------------------------------------------------
# Outlook / Microsoft Graph config
# -----------------------------------------------------------------------
 
@dataclass(frozen=True)
class OutlookConfig:
    """
    Credentials and addressing for Microsoft Graph email delivery.
 
    All four fields are required. Populate them from environment variables
    (recommended) or pass explicit values for testing.
 
    Environment variables (defaults):
        MS_TENANT_ID     - Azure AD tenant ID
        MS_CLIENT_ID     - App registration client ID
        MS_CLIENT_SECRET - App registration client secret
        MS_SENDER        - UPN / SMTP address of the sending mailbox
                           (the app must have Mail.Send permission for this account)
 
    recipient    : destination email address
    subject      : email subject line (can include commit ID at call time)
    """
    tenant_id_env: str = "MS_TENANT_ID"
    client_id_env: str = "MS_CLIENT_ID"
    client_secret_env: str = "MS_CLIENT_SECRET"
    sender_env: str = "MS_SENDER"
    recipient: str = ""
    subject: str = "Diff Review — Risk Analysis & Test Scenarios"

# -----------------------------------------------------------------------
# Microsoft Graph helpers
# -----------------------------------------------------------------------
 
def _ms_graph_token(cfg: OutlookConfig) -> str:
    """
    Fetches an OAuth2 client-credentials token from Azure AD.
 
    Requires the app registration to have the *application* permission
    Mail.Send (not delegated) and admin consent granted.
 
    Raises EnvironmentError for missing env vars, RuntimeError on HTTP failure.
    """
    tenant_id     = os.getenv(cfg.tenant_id_env)
    client_id     = os.getenv(cfg.client_id_env)
    client_secret = os.getenv(cfg.client_secret_env)
 
    missing = [
        name for name, val in [
            (cfg.tenant_id_env, tenant_id),
            (cfg.client_id_env, client_id),
            (cfg.client_secret_env, client_secret),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing Microsoft Graph environment variables: {', '.join(missing)}"
        )
 
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    resp = requests.post(
        url,
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
            "scope":         "https://graph.microsoft.com/.default",
        },
        timeout=(5.0, 15.0),
    )
    if not resp.ok:
        raise RuntimeError(
            f"Failed to obtain MS Graph token: {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()["access_token"]
 
 
def send_outlook_email(
    cfg: OutlookConfig,
    subject: str,
    body_html: str,
) -> None:
    """
    Sends an email via the Microsoft Graph /sendMail endpoint.
 
    Parameters
    ----------
    cfg        : OutlookConfig with credentials and sender/recipient.
    subject    : Subject line (overrides cfg.subject when explicitly provided).
    body_html  : Full HTML body of the email.
 
    Raises RuntimeError if the API call fails.
    """
    sender = os.getenv(cfg.sender_env)
    if not sender:
        raise EnvironmentError(
            f"Missing sender email in environment variable '{cfg.sender_env}'"
        )
 
    token = _ms_graph_token(cfg)
 
    payload = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body_html,
            },
            "toRecipients": [
                {"emailAddress": {"address": cfg.recipient}}
            ],
        },
        "saveToSentItems": True,
    }
 
    url = f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=(5.0, 30.0),
    )
 
    if resp.status_code == 202:
        log.info("Email sent successfully to %s", cfg.recipient)
    else:
        raise RuntimeError(
            f"MS Graph sendMail failed: {resp.status_code} {resp.text[:400]}"
        )
 
 
def _build_email_body(commit_id: str, risk_analysis: str, test_scenarios: str) -> str:
    """
    Converts markdown LLM outputs into a simple HTML email body.
 
    Markdown tables are converted to a monospace block so they render
    readably in Outlook without a full markdown-to-HTML library.
    """
    def md_to_pre(text: str) -> str:
        """Wrap text in a <pre> block with basic HTML escaping."""
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre style='font-family:Consolas,monospace;font-size:13px'>{escaped}</pre>"
 
    return f"""
    <html><body style="font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#222">
    <h2>Diff Review — Commit <code>{commit_id}</code></h2>
    
    <hr/>
    <h3>Risk Analysis</h3>
    {md_to_pre(risk_analysis or "(not produced)")}
    
    <hr/>
    <h3>Test Scenarios</h3>
    {md_to_pre(test_scenarios or "(not produced)")}
    
    <hr/>
    <p style="color:#888;font-size:12px">
    Generated automatically by the GitLab diff review pipeline.
    </p>
    </body></html>
"""
 
# -----------------------------
# Configuration
# ------------------------------
@dataclass(frozen=True)
class DiffFilterConfig:
    patterns_to_remove: tuple[re.Pattern[str],...] = (
            re.compile(r"go\.mod", re.IGNORECASE),
            re.compile(r"go\.sum",re.IGNORECASE),
            re.compile(r"_test\.go$", re.IGNORECASE),
            re.compile(r"\.vscode/", re.IGNORECASE)
        )

@dataclass(frozen=True)
class GitLabConfig:
    project_id: int = 1234
    commit_id: str = ""
    api_base: str = ""
    token_env:str = "TOKEN"

    # work_dir root (per-run dir is created under this)
    work_root: Path = Path(".work")

    # http behaviour
    connect_timeout_s: float= 5.0
    read_timeout_s: float = 30.0
    diff_per_page: int = 100

    # parallelism
    max_file_fetch_workers: int = 8

    # Prompt size guard (rough char budget; tune for your model/context)
    # max_context_chars: int = 128_000

    # CHANGED: renamed from max_context_chars to max_context_tokens.
    # Value is now a token count, not a character count.
    # 120_000 leaves ~8k headroom below a 128k-token context window
    # for system prompts, chat formatting overhead, and the LLM's own reply.
    max_context_tokens: int = 120_000

    # model
    model_name: str = "gemma4:e4b"
    prompts: PromptTemplates = field(default_factory=PromptTemplates)
    diff_filter: DiffFilterConfig = field(default_factory=DiffFilterConfig)

    # Optional — set to None to skip email delivery
    outlook: Optional[OutlookConfig] = None

# Langgraph State(Serializable)
class ReviewState(TypedDict, total=False):
    cfg: GitLabConfig

    # secret
    token:str

    # Git metadata + data
    parent_commit_id:str
    diff_entries: list[dict[str,Any]]
    filtered_diff: list[dict[str,Any]]

    # go file resolution + content
    go_path_pairs: list[tuple[Optional[str],str]]
    fetched_files: dict[str,str]

    # artifacts / formatted context
    run_dir: str
    diff_text:str
    go_files_text: str
    summarized_diff: Optional[str]

    # CHANGED: dropped_go_files added so downstream nodes and logs can
    # report exactly which files were excluded due to the token budget.
    dropped_go_files: list[str]

    # LLM outputs
    risk_analysis: Optional[str]
    test_scenarios: Optional[str]
    
    # control / exit
    exit_reason: Optional[str]

# ----
# Helpers: security + IO + HTTP
# ----
def require_token(env_key:str) -> str:
    token = os.getenv(env_key)
    if not token:
        raise EnvironmentError(f"Missing {env_key} in environment/.env")
    return token

def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    })

    retry = Retry(
          total=4,
          backoff_factor=0.5,
          status_forcelist=(429, 500, 502, 503, 504),
          allowed_methods={"GET"},
          raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://",adapter)
    s.mount("http://",adapter)
    return s

def http_timeout(cfg: GitLabConfig) -> tuple[float,float]:
    return (cfg.connect_timeout_s, cfg.read_timeout_s)

# Per-thread session for parallel file fetching
_thread_local = threading.local()

def _get_thread_session(token: str) -> requests.Session:
    """
    Returns a per-thread session, creating one if it doesn't exist yet.
    This avoids the overhead of creating a new session (connection pool +
    retry adapter) on every fetch_one call while keeping thread safety.
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session(token)
    return _thread_local.session

def _node_session(state: ReviewState) -> requests.Session:
    """
    Builds a fresh authenticated session for a single node invocation.

    Why not reuse across nodes:
    - requests.Session is not serialisable — storing it in ReviewState
      breaks LangGraph checkpointing (SqliteSaver, RedisSaver, etc.)
    - Node invocations are short-lived; the connection pool overhead of
      creating a new session is negligible compared to network round-trips.
    - fetch_go_files_node is the exception — it uses _get_thread_session()
      via threading.local() to share one session per worker thread within
      that single parallel fetch, which IS safe because no checkpointing
      occurs mid-node.
    """
    return make_session(state["token"])

# -----------------------------------------------------------------------
# LLM chain builder — shared by all three LLM nodes
# -----------------------------------------------------------------------

def _build_chain(cfg: GitLabConfig, template: PromptTemplate):
    """
    Constructs a LangChain runnable chain from a PromptTemplate.

    Centralised here so that:
      - all nodes share identical chain-building logic
      - a model provider swap (e.g. ChatOllama -> ChatAnthropic) is
        made in one place rather than three
      - the return type is prompt | model | parser, ready to invoke
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template.system),
        HumanMessagePromptTemplate.from_template(template.human),
    ])
    model = ChatOllama(model=cfg.model_name)
    return prompt | model | StrOutputParser()

def ensure_clean_dir_safe(folder:Path)-> None:
    """ 
    Remove folder content and recreate folder.
    Safety guard prevents deleting outside CWD tree.
    """
    folder = folder.resolve()
    cwd = Path.cwd().resolve()
    if cwd not in folder.parents and folder != cwd:
        raise ValueError(f"Refusing to delete outside current working directory: {folder}")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

def ensure_parent_dir(path:Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def save_json(path:Path, payload: object)-> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# -----
# GitLab API
# ----

def repo_base_url(cfg:GitLabConfig) -> str:
    return f"{cfg.api_base}/projects/{cfg.project_id}/repository"

def get_parent_commit_id(session: requests.Session, cfg:GitLabConfig) -> str:
    url = f"{repo_base_url(cfg)}/commits/{cfg.commit_id}"
    resp = session.get(url, timeout=http_timeout(cfg))
    resp.raise_for_status()
    parents_ids = resp.json().get("parent_ids") or []
    if not parents_ids:
        raise RuntimeError("No parent_ids for commit (unexpected payload or edge case).")
    return parents_ids[0]

def fetch_commit_diff(session: requests.Session, cfg: GitLabConfig) -> list[dict[str, Any]]:
    url = f"{repo_base_url(cfg)}/commits/{cfg.commit_id}/diff"
    results: list[dict[str, Any]] = []
    page = 1

    while True:
        resp = session.get(
            url,
            params={"per_page": cfg.diff_per_page, "page": page},
            timeout=http_timeout(cfg),
        )
        resp.raise_for_status()
        chunk = resp.json()
        if not isinstance(chunk, list):
            raise RuntimeError(f"Unexpected diff payload type: {type(chunk)}")

        results.extend(chunk)

        next_page_raw = (resp.headers.get("X-Next-Page") or "").strip()

        # GitLab returns an empty string for the last page, but guard against
        # unexpected non-integer values (e.g. malformed header, proxy injection)
        # so a bad header doesn't silently truncate results or loop infinitely.
        if not next_page_raw:
            break

        try:
            page = int(next_page_raw)
        except ValueError:
            log.warning(
                "Unexpected X-Next-Page header value %r — stopping pagination "
                "at page %d with %d entries collected so far.",
                next_page_raw,
                page,
                len(results),
            )
            break

    return results

def fetch_raw_file_from_ref(
    session: requests.Session,
    cfg: GitLabConfig,
    file_path: str,
    ref: str
) -> tuple[int, str]:
   encoded = quote(file_path, safe="")
   url = f"{repo_base_url(cfg)}/files/{encoded}/raw"
   resp = session.get(url, params={"ref":ref}, timeout=http_timeout(cfg))
   return resp.status_code,resp.text

# ---
# Diff + File helpers
# ---
def is_go_source_path(path:str) -> bool:
    return (
        path.endswith(".go")
        and not path.endswith("_test.go")
        and not path.endswith("pb.go")
        and "vendor/" not in path
    )

def filter_diff_entries(entries:list[dict[str,Any]], filt:DiffFilterConfig) ->list[dict[str,Any]]:
    out: list[dict[str,Any]] = []
    for e in entries:
        path = (e.get("new_path") or "")
        if any(p.search(path) for p in filt.patterns_to_remove):
            continue
        out.append(e)
    return out

def extract_go_paths(entries: list[dict[str,Any]]) -> list[tuple[Optional[str],str]]:
        pairs: list[tuple[Optional[str],str]] = []
        for e in entries:
            new_path = e.get("new_path")
            if not new_path or not is_go_source_path(new_path):
                continue
            if e.get("deleted_file"):
                continue

            old_path = e.get("old_path")
            if old_path == "/dev/null":
                old_path = None
            pairs.append((old_path,new_path))
        return pairs

def format_gitlab_diff(entries: list[dict[str, Any]]) -> str:
    """
    Formats pre-filtered diff entries into fenced markdown blocks.
    Assumes all entries in the list are relevant — no filtering is performed here.
    """
    blocks: list[str] = []
    for e in entries:
        path = (e.get("new_path") or "")
        diff = (e.get("diff") or "").strip()
        blocks.append(f"### {path}\n```diff\n{diff}\n```")
    return "\n\n".join(blocks)

def format_go_files(files: dict[str,str]) -> str:
    out: list[str] = []
    for name in sorted(files.keys()):
        out.append(f"### File: `{name}`\n```go\n{files[name]}\n```")
    return "\n\n".join(out)


# --- Context budget ---

# -----------------------------------------------------------------------
# CHANGED: Token budget helper
#
# Previously: available_go_budget(cfg, diff_text) -> int  (char count)
# Now:        available_go_token_budget(cfg, diff_text) -> int  (token count)
#
# The function now calls count_tokens() on the actual diff string so the
# budget reflects real tokenization rather than a character approximation.
# The result is a token count consumed by select_go_files_within_budget()
# and trim_to_token_budget(), not a character index.
# -----------------------------------------------------------------------
 
def available_go_token_budget(cfg: GitLabConfig, diff_text: str) -> int:
    """
    Return the number of tokens remaining for Go source files after
    accounting for the diff that will be sent to the LLM.
 
    Called with the actual diff string that will be sent (either the full
    diff or the summarized diff), so the budget is always accurate.
    """
    diff_tokens = count_tokens(diff_text)
    budget = cfg.max_context_tokens - diff_tokens
    if budget <= 0:
        log.warning(
            "Diff alone (%d tokens) meets or exceeds max_context_tokens (%d). "
            "Go files will be excluded entirely.",
            diff_tokens, cfg.max_context_tokens,
        )
        return 0
    log.info(
        "Token budget | diff=%d tokens | remaining for Go files=%d tokens",
        diff_tokens, budget,
    )
    return budget

# ---
# Run directory helpers
# ---
def run_dir_for(cfg: GitLabConfig) -> Path:
    # Per-commit run folder for idempotency and easier debugging
    return cfg.work_root / cfg.commit_id

def paths_for_run(cfg: GitLabConfig) -> dict[str,Path]:
    base = run_dir_for(cfg)
    return {
        "base": base,
        "diff_json": base/"commit_diff.json",
        "raw_files_dir": base / "raw_files",
        "formatted_diff": base / "formatted_gitlab_diff.txt",
        "formatted_go": base / "formatted_golang_files.txt"
    }

# --
# Langgraph nodes
# --
def load_config_node(state: ReviewState) -> dict:
    load_dotenv()
    cfg = state["cfg"]
    token = require_token(cfg.token_env)

    run_paths = paths_for_run(cfg)
    run_paths["base"].mkdir(parents=True,exist_ok=True)

    return {
        "token": token,
        "run_dir": str(run_paths["base"]),
        "summarized_diff": None,
        "fetched_files":{},
        "dropped_go_files": [],   # NEW: initialised here so it's always present in state
        "exit_reason": None
    }

def fetch_commit_metadata_node(state: ReviewState) -> dict:
    cfg = state["cfg"]
    session = _node_session(state)  
    parent = get_parent_commit_id(session,cfg)
    return {"parent_commit_id":parent}

def fetch_diff_node(state:ReviewState) -> dict:
    cfg= state["cfg"]
    session = _node_session(state) 
    diff_entries = fetch_commit_diff(session,cfg)
    return {"diff_entries": diff_entries}

def filter_diff_node(state:ReviewState) -> dict:
    cfg = state["cfg"]
    filtered = filter_diff_entries(state["diff_entries"],cfg.diff_filter)

    run_paths = paths_for_run(cfg)
    save_json(run_paths["diff_json"], filtered)

    log.info("Diff entries %d | Filtered: %d", len(state["diff_entries"]), len(filtered))
    return {"filtered_diff": filtered}

def extract_go_paths_node(state: ReviewState) -> dict:
    pairs = extract_go_paths(state["filtered_diff"])
    log.info("Go source files detected in diff: %d", len(pairs))
    return {"go_path_pairs":pairs}

def route_if_no_go_files(state:ReviewState) -> str:
    if not state.get("go_path_pairs"):
        return "early_exit"
    return "fetch_go_files"

def early_exit_node(state: ReviewState) -> dict:
    return {
        "exit_reason": "No Go source files found in diff after filtering.",
        "test_scenarios": None,     # ← never computed, None is the honest value
        "risk_analysis": None       # ← never computed, None is the honest value
    }


def fetch_go_files_node(state: ReviewState) -> dict:
    """
    Fetch files in parallel:
    - prefer old_path@parent_commit
    - fallback new_path@current_commit
    Store in-memory as fetched_files {path: content}.
    Also save under run_dir/raw_files for auditability.
    """
    cfg = state["cfg"]
    run_paths = paths_for_run(cfg)
    raw_dir = run_paths["raw_files_dir"]
    ensure_clean_dir_safe(raw_dir)

    token = state["token"]
    parent = state["parent_commit_id"]
    pairs = state["go_path_pairs"]
    fetched: dict[str, str] = {}

    def fetch_one(old_path: Optional[str], new_path: str) -> tuple[Optional[str], Optional[str]]:
        session = _get_thread_session(token)
        try:
            # 1) parent version
            if old_path:
                status, text = fetch_raw_file_from_ref(session, cfg, old_path, parent)
                if status == 200:
                    return old_path, text

            # 2) current version fallback
            status, text = fetch_raw_file_from_ref(session, cfg, new_path, cfg.commit_id)
            if status == 200:
                return new_path, text

            log.warning(
                "Could not fetch file — both refs returned non-200. "
                "old_path=%s (parent), new_path=%s (commit %s)",
                old_path, new_path, cfg.commit_id,
            )
            return None, None

        except Exception:
            log.exception(
                "Unexpected error fetching file. old_path=%s new_path=%s",
                old_path, new_path,
            )
            return None, None

    with ThreadPoolExecutor(max_workers=cfg.max_file_fetch_workers) as ex:
        futures = {ex.submit(fetch_one, oldp, newp): (oldp, newp) for oldp, newp in pairs}
        for f in as_completed(futures):
            try:
                path, content = f.result()
            except Exception:
                oldp, newp = futures[f]
                log.exception("Future raised unexpectedly. old_path=%s new_path=%s", oldp, newp)
                continue

            if path and content is not None:
                fetched[path] = content
                out_path = raw_dir / path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")

    log.info("Fetched Go files: %d / %d", len(fetched), len(pairs))
    return {"fetched_files": fetched}

def assemble_context_node(state: ReviewState) -> dict:
    cfg = state["cfg"]
    run_paths = paths_for_run(cfg)

    filtered = state["filtered_diff"]

    formattable : list[dict[str,Any]] = []
    not_formattable : list[dict[str,Any]] = []

    # Ownership of filtering is explicit here, not buried inside a formatting helper.
    # Any entry dropped at this stage is logged so silent loss is impossible.
    
    for e in filtered:
      path = e.get("new_path") or ""
      if is_go_source_path(path) and not e.get("deleted_file"):
          formattable.append(e)
      else:
          not_formattable.append(e)

    if not_formattable:
        log.warning(
            "assemble_context: %d entries were not formattable:",
            len(not_formattable)
        )
        for e in not_formattable:
            log.warning(
                " - path=%s deleted=%s",
                e.get("new_path"),
                e.get("deleted_file")
            )
    diff_text = format_gitlab_diff(formattable)
    go_files_text = format_go_files(state["fetched_files"])

    run_paths["formatted_diff"].write_text(diff_text, encoding="utf-8")
    run_paths["formatted_go"].write_text(go_files_text, encoding="utf-8")

    diff_tokens = count_tokens(diff_text)
    go_tokens = count_tokens(go_files_text)
    log.info(
        "assemble_context: diff=%d tokens | go_files=%d tokens | "
        "combined=%d tokens | budget=%d tokens",
        diff_tokens, go_tokens, diff_tokens + go_tokens, cfg.max_context_tokens,
    )

    return {"diff_text": diff_text, "go_files_text": go_files_text}

def context_size_guard_route(state:ReviewState) -> str:
    cfg = state["cfg"]
    # total = len(state.get("diff_text","")) + len(state.get("go_files_text",""))
    diff_tokens = count_tokens(state.get("diff_text", ""))
    go_tokens   = count_tokens(state.get("go_files_text", ""))
    total_tokens = diff_tokens + go_tokens

    log.info("Context size (tokens): %d | limit: %d", total_tokens, cfg.max_context_tokens)
    if total_tokens > cfg.max_context_tokens:
        return "summarize_diff"
    return "llm_risk_analysis"

def summarize_diff_node(state: ReviewState) -> dict:
    """LLM pass #1 (optional): summarize the diff into test-focused bullets."""
    log.info("Entering summarize_diff_node")
    cfg = state["cfg"]
    chain = _build_chain(cfg, cfg.prompts.summarize_diff)
    summary = chain.invoke({"diff_text": state["diff_text"]})
    return {"summarized_diff": summary}


def llm_risk_analysis_node(state: ReviewState) -> dict:
    """
    LLM pass #2: identify risks, edge cases, and regressions.
    Uses summarized_diff if available (context was too large), else full diff.
    """
    log.info("Entering llm_risk_analysis_node")
    cfg = state["cfg"]

    diff_for_llm = state.get("summarized_diff") or state["diff_text"]
    
    # Budget is derived from the actual diff being sent —
    # if summarization ran, the summary is shorter so go files get more room;
    # if it didn't, the full diff length is used. Either way the sum never
    # exceeds max_context_token.
    # select_go_files_within_budget keeps whole files and logs exactly
    # which files were dropped — no silent mid-file truncation.
    go_token_budget = available_go_token_budget(cfg, diff_for_llm)
    selected_files, dropped = select_go_files_within_budget(
        state["fetched_files"], go_token_budget
    )
    trimmed_go_files = format_go_files(selected_files)

    log.info(
        "Risk analysis input | diff=%d tokens | go_files=%d tokens "
        "(%d files kept, %d dropped)",
        count_tokens(diff_for_llm),
        count_tokens(trimmed_go_files),
        len(selected_files),
        len(dropped),
    )

    chain = _build_chain(cfg, cfg.prompts.risk_analysis)
    log.info("Invoking llm_risk_analysis")
    analysis = chain.invoke({
        "go_files_text": trimmed_go_files,
        "diff_text": diff_for_llm,
    })

    # CHANGED: persist dropped file list to state so save_outputs_node can
    # write it to disk and the caller can see what was excluded.
    return {"risk_analysis": analysis, "dropped_go_files": dropped}


def llm_test_generation_node(state: ReviewState) -> dict:
    """LLM pass #3: generate final test scenarios in tabular format."""
    log.info("Entering llm_test_generation_node")
    cfg = state["cfg"]

    diff_for_llm = state.get("summarized_diff") or state["diff_text"]

    # CHANGED: re-apply the same whole-file selection used in pass 2.
    # This ensures passes 2 and 3 see exactly the same Go file context
    # even if the state dict is re-entered from a checkpoint.
    go_token_budget = available_go_token_budget(cfg, diff_for_llm)
    selected_files, dropped = select_go_files_within_budget(
        state["fetched_files"], go_token_budget
    )
    trimmed_go_files = format_go_files(selected_files)

    log.info(
        "Test generation input | diff=%d tokens | go_files=%d tokens "
        "(%d files kept, %d dropped)",
        count_tokens(diff_for_llm),
        count_tokens(trimmed_go_files),
        len(selected_files),
        len(dropped),
    )

    chain = _build_chain(cfg, cfg.prompts.test_generation)
    resp = chain.invoke({
        "go_files_text": trimmed_go_files,                            
        "diff_text": diff_for_llm,
        "risk_analysis": state.get("risk_analysis") or "",
    })

    return {"test_scenarios": resp}

def save_outputs_node(state: ReviewState) -> dict:
    """Persist final LLM outputs alongside other run artifacts."""
    log.info("Entering save_outputs_node") 
    cfg = state["cfg"]
    base = run_dir_for(cfg)

    if state.get("risk_analysis"):
        (base / "risk_analysis.md").write_text(state["risk_analysis"], encoding="utf-8")
        log.info("Saved risk_analysis.md -> %s", base)
    else:
        log.warning("risk_analysis is empty — risk_analysis.md not written")

    if state.get("test_scenarios"):
        (base / "test_scenarios.md").write_text(state["test_scenarios"], encoding="utf-8")
        log.info("Saved test_scenarios.md -> %s", base)
    else:
        log.warning("test_scenarios is empty — test_scenarios.md not written")

    # NEW: write dropped file list to disk so it's part of the audit trail.
    # An empty list is still written so the absence of the file never means
    # "nothing was dropped" — it always means "this node didn't run".
    dropped = state.get("dropped_go_files") or []
    dropped_path = base / "dropped_go_files.txt"
    dropped_path.write_text(
        "\n".join(dropped) if dropped else "(none)",
        encoding="utf-8",
    )
    if dropped:
        log.warning(
            "save_outputs: %d Go file(s) were excluded from LLM context due to "
            "token budget — see %s",
            len(dropped), dropped_path,
        )
    
    log.info("Run artifacts saved to %s", base)

    return {}

def send_email_node(state: ReviewState) -> dict:
    """
    Sends the LLM outputs (risk analysis + test scenarios) to Outlook via
    the Microsoft Graph API.
 
    Skipped silently when:
    - cfg.outlook is None (email delivery not configured)
    - cfg.outlook.recipient is empty
    - Both risk_analysis and test_scenarios are None/empty
      (e.g. early-exit path — no point sending a blank email)
 
    A send failure is logged as an error but does NOT raise, so the graph
    always completes and local artifacts are always written even if email
    delivery fails.
    """
    log.info("Entering send_email_node")
    cfg = state["cfg"]
 
    if cfg.outlook is None:
        log.info("send_email_node: cfg.outlook is None — skipping email delivery")
        return {}
 
    if not cfg.outlook.recipient:
        log.warning("send_email_node: no recipient configured — skipping email delivery")
        return {}
 
    risk_analysis  = state.get("risk_analysis")  or ""
    test_scenarios = state.get("test_scenarios") or ""
 
    if not risk_analysis and not test_scenarios:
        log.info(
            "send_email_node: both LLM outputs are empty (early-exit path?) "
            "— skipping email delivery"
        )
        return {}
 
    subject = f"{cfg.outlook.subject} [{cfg.commit_id[:8]}]"
    body    = _build_email_body(cfg.commit_id, risk_analysis, test_scenarios)
 
    try:
        send_outlook_email(cfg.outlook, subject=subject, body_html=body)
    except Exception:
        # Non-fatal: log and continue so local artifacts are unaffected.
        log.exception(
            "send_email_node: failed to send email to %s", cfg.outlook.recipient
        )
 
    return {}

# -----------------------------------------------------------------------------
# Graph construction helpers
# -----------------------------------------------------------------------------

def _node_name(fn) -> str:
    """
    Derives a canonical node name from a node function by stripping
    the mandatory '_node' suffix from the function name.

        load_config_node           -> "load_config"
        fetch_commit_metadata_node -> "fetch_commit_metadata"
        llm_risk_analysis_node     -> "llm_risk_analysis"

    Raises ValueError at graph construction time if a function does not
    follow the convention, turning a potential silent runtime KeyError
    into an immediate, descriptive error.
    """
    name = fn.__name__
    if not name.endswith("_node"):
        raise ValueError(
            f"Node function '{name}' does not follow the '_node' suffix "
            f"convention. Rename it to '{name}_node' or update _node_name()."
        )
    return name[:-5]


# -----------------------------------------------------------------------------
# Build & compile graph
# -----------------------------------------------------------------------------

def build_graph():
    g = StateGraph(ReviewState)

    # ------------------------------------------------------------------
    # Node registration
    # Names are derived from function names — never typed as raw strings.
    # Adding a new node means adding it to this list only; the name is
    # never duplicated in the edge declarations below.
    # ------------------------------------------------------------------
    node_fns = [
        load_config_node,
        fetch_commit_metadata_node,
        fetch_diff_node,
        filter_diff_node,
        extract_go_paths_node,
        early_exit_node,
        fetch_go_files_node,
        assemble_context_node,
        summarize_diff_node,
        llm_risk_analysis_node,
        llm_test_generation_node,
        save_outputs_node,
        send_email_node,
    ]
    for fn in node_fns:
        g.add_node(_node_name(fn), fn)

    # ------------------------------------------------------------------
    # Node name aliases
    # Defined once here from the function names themselves.
    # A typo in an alias is a NameError at import time — caught immediately.
    # A missing alias is a NameError at graph construction time — also immediate.
    # Both are far safer than a raw string typo, which only fails at the
    # moment the broken edge is first traversed during a live run.
    # ------------------------------------------------------------------
    LOAD_CONFIG           = _node_name(load_config_node)
    FETCH_COMMIT_METADATA = _node_name(fetch_commit_metadata_node)
    FETCH_DIFF            = _node_name(fetch_diff_node)
    FILTER_DIFF           = _node_name(filter_diff_node)
    EXTRACT_GO_PATHS      = _node_name(extract_go_paths_node)
    EARLY_EXIT            = _node_name(early_exit_node)
    FETCH_GO_FILES        = _node_name(fetch_go_files_node)
    ASSEMBLE_CONTEXT      = _node_name(assemble_context_node)
    SUMMARIZE_DIFF        = _node_name(summarize_diff_node)
    LLM_RISK_ANALYSIS     = _node_name(llm_risk_analysis_node)
    LLM_TEST_GENERATION   = _node_name(llm_test_generation_node)
    SAVE_OUTPUTS          = _node_name(save_outputs_node)
    SEND_EMAIL            = _node_name(send_email_node)   # ← new

    # ------------------------------------------------------------------
    # Linear edges — main pipeline spine
    # ------------------------------------------------------------------
    g.add_edge(START,                 LOAD_CONFIG)
    g.add_edge(LOAD_CONFIG,           FETCH_COMMIT_METADATA)
    g.add_edge(FETCH_COMMIT_METADATA, FETCH_DIFF)
    g.add_edge(FETCH_DIFF,            FILTER_DIFF)
    g.add_edge(FILTER_DIFF,           EXTRACT_GO_PATHS)

    # ------------------------------------------------------------------
    # Conditional branch: no Go source files found -> early exit
    # The dict keys are the string literals returned by the routing
    # function itself (route_if_no_go_files). The values use aliases
    # so at least the destination side is protected from typos.
    # ------------------------------------------------------------------
    g.add_conditional_edges(
        EXTRACT_GO_PATHS,
        route_if_no_go_files,
        {
            "early_exit":     EARLY_EXIT,
            "fetch_go_files": FETCH_GO_FILES,
        },
    )

    g.add_edge(FETCH_GO_FILES, ASSEMBLE_CONTEXT)

    # ------------------------------------------------------------------
    # Conditional branch: context too large -> summarize first
    # ------------------------------------------------------------------
    g.add_conditional_edges(
        ASSEMBLE_CONTEXT,
        context_size_guard_route,
        {
            "summarize_diff":    SUMMARIZE_DIFF,
            "llm_risk_analysis": LLM_RISK_ANALYSIS,
        },
    )

    # ------------------------------------------------------------------
    # LLM pipeline — three sequential passes
    # ------------------------------------------------------------------
    g.add_edge(SUMMARIZE_DIFF,      LLM_RISK_ANALYSIS)   # optional pass feeds into pass 2
    g.add_edge(LLM_RISK_ANALYSIS,   LLM_TEST_GENERATION)
    g.add_edge(LLM_TEST_GENERATION,   SAVE_OUTPUTS)
    g.add_edge(SAVE_OUTPUTS, SEND_EMAIL)
    g.add_edge(SEND_EMAIL, END)

    # ------------------------------------------------------------------
    # Early exit terminal edge
    # ------------------------------------------------------------------
    g.add_edge(EARLY_EXIT, SEND_EMAIL)

    return g.compile()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the GitLab diff review pipeline for a single commit."
    )
    parser.add_argument(
        "--api-base",
        required=True,
        help="GitLab API base URL, e.g. https://gitlab.example.com/api/4",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="GitLab project ID (integer), e.g. 1234",
    )
    parser.add_argument(
        "--commit-id",
        required=True,
        help="Full commit SHA to review. e.g: abc1233dc",
    )
    parser.add_argument(
        "--token-env",
        default="TOKEN",
        help="Name of the environment variable holding the GitLab token"
    )
    parser.add_argument(
        "--model",
        default="gemma4:e4b",
        help="Ollama model name to use (default: gemma4:e4b)"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=120_000,
        help="Maximum token budget for LLM context (default: 120000). "
              "Set to ~90% of your model's context window"
    )
    # Outlook flags (all optional — omit to skip email delivery)
    parser.add_argument("--email-to",    default="",   help="Recipient email address")
    parser.add_argument("--email-subject", default="Diff Review — Risk Analysis & Test Scenarios")
    parser.add_argument("--ms-tenant-env",   default="MS_TENANT_ID")
    parser.add_argument("--ms-client-env",   default="MS_CLIENT_ID")
    parser.add_argument("--ms-secret-env",   default="MS_CLIENT_SECRET")
    parser.add_argument("--ms-sender-env",   default="MS_SENDER")

    args = parser.parse_args()

    # Build OutlookConfig only when a recipient was provided
    outlook_cfg: Optional[OutlookConfig] = None
    if args.email_to:
        outlook_cfg = OutlookConfig(
            tenant_id_env=args.ms_tenant_env,
            client_id_env=args.ms_client_env,
            client_secret_env=args.ms_secret_env,
            sender_env=args.ms_sender_env,
            recipient=args.email_to,
            subject=args.email_subject,
        )

    cfg = GitLabConfig(
        api_base=args.api_base,
        project_id=args.project_id,
        commit_id=args.commit_id,
        token_env=args.token_env,
        model_name=args.model,
        max_context_tokens=args.max_context_tokens,
        outlook=outlook_cfg,
    )
    app = build_graph()
    result = app.invoke({"cfg": cfg})

    exit_reason: Optional[str] = result.get("exit_reason")

    if exit_reason is not None:                         # ← explicit None check, not truthy
        print("\n===== EXIT =====\n")
        print(exit_reason)
    elif result.get("risk_analysis") is not None:       # ← confirms LLM path actually ran
        print("\n===== RISK ANALYSIS =====\n")
        print(result.get("risk_analysis", ""))
        print("\n===== TEST SCENARIOS =====\n")
        print(result.get("test_scenarios", ""))

        # surface dropped files in console output so they are not invisible
        dropped = result.get("dropped_go_files") or []
        if dropped:
            print("\n======== DROPPED FROM LLM CONTEXT (token budget) =====\n")
            print("\n".join(f" - {f}" for f in dropped))
    else:
        # Neither branch produced output — graph may have failed silently
        log.error(
            "Graph completed but produced no output. "
            "exit_reason=%r risk_analysis=%r test_scenarios=%r",
            result.get("exit_reason"),
            result.get("risk_analysis"),
            result.get("test_scenarios"),
        )
    




























    




























