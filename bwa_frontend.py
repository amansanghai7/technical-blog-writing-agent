from __future__ import annotations

import json
import os
import re
import uuid
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app
from langgraph.types import Command

# Blogs output directory — must match BLOGS_DIR in bwa_backend.py.
# Defaults to "." locally; set BLOGS_DIR=/home/ubuntu/blogs on EC2.
BLOGS_DIR = Path(os.getenv("BLOGS_DIR", ".")).resolve()


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(
    graph_app,
    inputs: Any,          # Dict for new runs, Command for HITL resumes
    thread_id: str,
) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress; yields ("updates", delta) for UI display and
    ("final", full_state) once complete. Falls back to invoke() if streaming fails.
    Passes thread_id to the checkpointer so runs are persisted and resumable.
    """
    config = {"configurable": {"thread_id": thread_id}}
    final: Optional[Dict[str, Any]] = None
    try:
        for mode, step in graph_app.stream(inputs, config=config, stream_mode=["updates", "values"]):
            if mode == "updates":
                yield ("updates", step)
            else:
                final = step   # accumulates the latest full-state snapshot
        # Only treat as "final" if the graph actually reached END.
        # When interrupt() fires, the stream also ends but next!=[] — don't yield "final"
        # in that case so the interrupt detection block in the caller can handle it.
        if final is not None:
            try:
                snap = graph_app.get_state(config)
                if snap.next:   # graph paused at interrupt, not at END
                    return
            except Exception:
                pass
            yield ("final", final)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs, config=config)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return (BLOGS_DIR / src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), width='stretch')
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), width='stretch')
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """Returns .md files from BLOGS_DIR, newest first."""
    files = [p for p in BLOGS_DIR.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    _has_interrupted = (
        bool(st.session_state.get("last_thread_id"))
        and not st.session_state.get("run_completed", True)
    )
    resume_btn = st.button(
        "🔄 Resume last run",
        disabled=not _has_interrupted,
        help="Resumes the most recently interrupted run from its last Supabase checkpoint.",
    )
    if st.session_state.get("last_thread_id"):
        tid = st.session_state["last_thread_id"]
        label = "interrupted" if _has_interrupted else "completed"
        st.caption(f"Last thread `{tid[:8]}…` — {label}")

    # Past blogs list
    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,          # old files don't include plan
                    "evidence": [],        # old files don't include evidence
                    "image_specs": [],     # optional (not persisted)
                    "final": md_text,      # markdown body
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "last_thread_id" not in st.session_state:
    st.session_state["last_thread_id"] = None
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None
if "run_completed" not in st.session_state:
    st.session_state["run_completed"] = True
if "phase" not in st.session_state:
    st.session_state["phase"] = "idle"   # idle | awaiting_approval
if "pending_plan" not in st.session_state:
    st.session_state["pending_plan"] = {}
if "pending_replan_count" not in st.session_state:
    st.session_state["pending_replan_count"] = 0

# Layout
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


inputs: Any = None          # declared here so both branches can assign without re-annotating
thread_id: str = ""
status_label: str = ""

if run_btn or resume_btn:
    if run_btn:
        if not topic.strip():
            st.warning("Please enter a topic.")
            st.stop()
        thread_id = str(uuid.uuid4())
        inputs = {
            "topic": topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of.isoformat(),
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
            "replan_count": 0,
        }
        st.session_state["last_thread_id"] = thread_id
        st.session_state["last_inputs"] = inputs
        st.session_state["run_completed"] = False
        st.session_state["phase"] = "idle"
        status_label = "Running graph…"
    else:  # resume_btn
        thread_id = st.session_state["last_thread_id"]
        # Check where the graph actually is before deciding how to resume.
        # Passing last_inputs (original dict with plan=None) would corrupt mid-run state.
        try:
            _rsnap = app.get_state({"configurable": {"thread_id": thread_id}})
            if _rsnap.next and any("plan_review" in str(n) for n in _rsnap.next):
                # Graph is at a HITL interrupt — redirect to plan review UI instead of resuming
                _rpv = _rsnap.values.get("plan")
                _rpv_dict: Dict[str, Any] = (
                    _rpv.model_dump() if _rpv is not None and hasattr(_rpv, "model_dump") else (_rpv or {})
                )
                st.session_state["pending_plan"] = _rpv_dict
                st.session_state["pending_replan_count"] = _rsnap.values.get("replan_count", 0)
                st.session_state["phase"] = "awaiting_approval"
                st.rerun()
        except Exception:
            pass
        # Mid-execution resume: None tells LangGraph to continue from the last checkpoint
        # node without merging/overwriting any existing state fields.
        inputs = None
        status_label = f"Resuming thread {thread_id[:8]}… (continuing from last checkpoint)"

    status = st.status(status_label, expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs, thread_id):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)

            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            st.session_state["run_completed"] = True
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received final state")

    # After stream ends: check if graph paused at plan_review (HITL interrupt)
    if not st.session_state.get("run_completed"):
        try:
            _snap = app.get_state({"configurable": {"thread_id": thread_id}})
            if _snap.next and any("plan_review" in str(n) for n in _snap.next):
                _plan_val = _snap.values.get("plan")
                if hasattr(_plan_val, "model_dump"):
                    _plan_val = _plan_val.model_dump()
                st.session_state["pending_plan"] = _plan_val or {}
                st.session_state["pending_replan_count"] = _snap.values.get("replan_count", 0)
                st.session_state["phase"] = "awaiting_approval"
                status.update(label="⏸️ Waiting for plan approval…", state="running", expanded=False)
        except Exception:
            pass

# ─── Plan Review (HITL) ───────────────────────────────────────────────────────
if st.session_state.get("phase") == "awaiting_approval":
    thread_id = st.session_state.get("last_thread_id", "")
    pending   = st.session_state.get("pending_plan", {})
    replan_n  = st.session_state.get("pending_replan_count", 0)
    max_r     = 2  # must match backend MAX_REPLANS

    st.divider()
    st.warning(
        "⏸️ **Plan ready for your review.** "
        "Approve to begin writing, adjust the details and save edits, or reject to regenerate."
    )

    with st.container(border=True):
        st.subheader(f"📋 {pending.get('blog_title', 'Untitled')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Audience", pending.get("audience", "—"))
        m2.metric("Tone",     pending.get("tone", "—"))
        m3.metric("Kind",     pending.get("blog_kind", "—"))

        tasks = pending.get("tasks", [])
        if tasks:
            st.write(f"**{len(tasks)} sections planned:**")
            for t in tasks:
                flags = " ".join(
                    f"`{label}`"
                    for label, key in [("code", "requires_code"), ("citations", "requires_citations")]
                    if t.get(key)
                )
                st.write(f"- **{t.get('title')}** — ~{t.get('target_words')} words {flags}")

    st.write("**Edit before approving (optional):**")
    new_title    = st.text_input("Blog title", value=pending.get("blog_title", ""),  key="edit_title")
    _ac, _tc     = st.columns(2)
    new_audience = _ac.text_input("Audience",   value=pending.get("audience", ""),   key="edit_audience")
    new_tone     = _tc.text_input("Tone",        value=pending.get("tone", ""),       key="edit_tone")

    _b1, _b2, _b3 = st.columns(3)
    approve_plan = _b1.button("✅ Approve & Write",   type="primary", width='stretch')
    edit_plan    = _b2.button("✏️ Apply Edits & Write",               width='stretch')
    reject_plan  = _b3.button(
        "🔄 Reject & Replan" + (" (limit reached — will approve)" if replan_n >= max_r else ""),
        width='stretch',
    )

    if approve_plan or edit_plan or reject_plan:
        if approve_plan:
            decision = {"action": "approve"}
        elif reject_plan:
            decision = {"action": "reject"}
        else:
            edited = dict(pending)
            edited["blog_title"] = new_title
            edited["audience"]   = new_audience
            edited["tone"]       = new_tone
            decision = {"action": "edit", "edited_plan": edited}

        st.session_state["phase"] = "idle"
        st.session_state["run_completed"] = False

        rev_status   = st.status("Resuming after plan review…", expanded=True)
        rev_progress = st.empty()
        rev_state: Dict[str, Any] = {}
        rev_last_node = None

        for kind, payload in try_stream(app, Command(resume=decision), thread_id):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != rev_last_node:
                    rev_status.write(f"➡️ Node: `{node_name}`")
                    rev_last_node = node_name
                rev_state = extract_latest_state(rev_state, payload)
                rev_progress.json({
                    "sections_done": len(rev_state.get("sections", []) or []),
                    "images":        len(rev_state.get("image_specs", []) or []),
                })
                log(f"[review-resume/{kind}] {json.dumps(payload, default=str)[:800]}")
            elif kind == "final":
                st.session_state["last_out"] = payload
                st.session_state["run_completed"] = True
                rev_status.update(label="✅ Done", state="complete", expanded=False)
                log("[review-resume/final] received final state")

        # Re-check: might have looped back to plan_review after a reject
        if not st.session_state.get("run_completed"):
            try:
                _snap2 = app.get_state({"configurable": {"thread_id": thread_id}})
                if _snap2.next and any("plan_review" in str(n) for n in _snap2.next):
                    _pv2 = _snap2.values.get("plan")
                    if hasattr(_pv2, "model_dump"):
                        _pv2 = _pv2.model_dump()
                    st.session_state["pending_plan"] = _pv2 or {}
                    st.session_state["pending_replan_count"] = _snap2.values.get("replan_count", 0)
                    st.session_state["phase"] = "awaiting_approval"
                    rev_status.update(label="⏸️ New plan ready for review…", state="running", expanded=False)
            except Exception:
                pass

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, width='stretch', hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, BLOGS_DIR / "images")
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = BLOGS_DIR / "images"

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, width='stretch')

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")
