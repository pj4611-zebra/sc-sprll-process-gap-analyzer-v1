"""SPRLL Process Gap Analyzer - Streamlit Frontend (calls FastAPI backend)."""

import os
import re
from datetime import date

import requests as req_lib
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

DISCIPLINES = [
    "Tools-Software Engineering",
    "EMC Cloud-Software Engineering",
]


def build_payload(mode, sprll_numbers, from_date=None, to_date=None, discipline=None):
    if mode == "Enter SPRLL Numbers manually":
        return {"sprll_numbers": sprll_numbers}
    return {
        "from_date": str(from_date),
        "to_date": str(to_date),
        "discipline": discipline,
    }


st.set_page_config(
    page_title="SPRLL Process Gap Analyzer", page_icon="🔍", layout="wide"
)

col_title, col_logo = st.columns([6, 1])
with col_title:
    st.title("SPRLL Process Gap Analyzer")
    st.caption("Release Readiness Gap Analyzer — Quality Engineering")
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), "static", "zebra_logo.jpeg")
    if os.path.exists(logo_path):
        st.image(logo_path, width=110)

st.divider()
st.subheader("Analysis Parameters")

mode = st.radio(
    "Input mode",
    ["Enter SPRLL Numbers manually", "Search by Date & Discipline"],
    horizontal=True,
    label_visibility="collapsed",
)

sprll_numbers = []

if mode == "Enter SPRLL Numbers manually":
    sprll_raw = st.text_input(
        "SPRLL Number(s)",
        placeholder="4212, 3974, 4112",
        help="Separate multiple numbers with commas or spaces. Prefix SPRLL- is added automatically.",
    )
    if sprll_raw:
        parts = re.split(r"[,\s]+", sprll_raw.strip())
        for p in parts:
            p = p.strip()
            if not p:
                continue
            num = re.sub(r"(?i)^sprll-?", "", p)
            if num:
                sprll_numbers.append(f"SPRLL-{num}")

    if sprll_numbers:
        st.caption(f"Will analyze: {', '.join(sprll_numbers)}")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        from_date = st.date_input("From Date", value=date(date.today().year, 1, 1))
    with c2:
        to_date = st.date_input("To Date", value=date.today())
    with c3:
        discipline = st.selectbox("Discipline", DISCIPLINES)

    jql = " AND ".join(
        [
            "project = SPRLL",
            "issuetype = Process",
            "status in (Resolved, Closed)",
            f'"Discipline" = "{discipline}"',
            f'created >= "{from_date}"',
            f'created <= "{to_date}"',
        ]
    )
    with st.expander("JQL Query Preview", expanded=True):
        st.code(jql, language="sql")

st.divider()

analyze_clicked = st.button("Analyze", type="primary", use_container_width=False)

if analyze_clicked:
    payload = build_payload(
        mode=mode,
        sprll_numbers=sprll_numbers,
        from_date=locals().get("from_date"),
        to_date=locals().get("to_date"),
        discipline=locals().get("discipline"),
    )

    try:
        with st.spinner("Calling backend /api/analyze …"):
            resp = req_lib.post(
                f"{BACKEND_URL}/api/analyze", json=payload, timeout=180
            )

        if not resp.ok:
            st.error(f"Backend returned {resp.status_code}: {resp.text[:500]}")
            st.stop()

        result = resp.json()
        issues = result.get("issues", [])
        process_gaps = result.get("process_gaps", [])
        resolved_sprll_numbers = result.get("sprll_numbers", [])

        st.divider()

        if mode == "Search by Date & Discipline":
            st.info(
                f"**Scope:** {from_date} → {to_date} | {discipline} | "
                f"{len(resolved_sprll_numbers)} issue(s)"
            )
        else:
            st.info(
                f"**Scope:** {len(resolved_sprll_numbers)} SPRLL issue(s) analyzed"
            )

        # ── Issues ────────────────────────────────────────────────
        st.subheader(f"Issues Analyzed ({len(issues)})")
        for issue in issues:
            label = (
                f"{issue.get('sprllNumber', issue.get('key', 'SPRLL'))} — "
                f"{issue.get('summary', '')}"
            )
            with st.expander(label, expanded=False):
                if issue.get("error"):
                    st.error(issue["error"])
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Status:** {issue.get('status', '—')}")
                        st.markdown(
                            f"**Discipline:** "
                            f"{issue.get('discipline', issue.get('customfield_12801', '—'))}"
                        )
                        st.markdown(
                            f"**Assignee:** {issue.get('assignee_name', '—')}"
                        )
                    with col_b:
                        missing = issue.get(
                            "missingFields", issue.get("missing_fields", [])
                        )
                        if missing:
                            st.warning(
                                f"**Missing fields ({len(missing)}):** "
                                f"{', '.join(missing)}"
                            )
                        else:
                            st.success("All required fields present")

                        comment_count = issue.get(
                            "matchedCommentCount", issue.get("comment_count", 0)
                        )
                        st.markdown(f"**Assignee Comment Count:** {comment_count}")

                    # Source badge
                    source = issue.get("source", "")
                    if source:
                        st.caption(f"📦 Source: `{source}`")

                    # Description
                    if issue.get("description"):
                        desc = issue["description"]
                        st.markdown("---")
                        st.markdown("**Description:**")
                        st.markdown(
                            desc[:800] + ("…" if len(desc) > 800 else "")
                        )

                    # Assignee comments (raw)
                    assignee_comments = issue.get("assignee_comments", [])
                    if assignee_comments:
                        st.markdown("---")
                        st.markdown(
                            f"**Assignee Comments ({len(assignee_comments)}):**"
                        )
                        for idx, comment_text in enumerate(assignee_comments, 1):
                            st.markdown(
                                f"**Comment {idx}:**\n{comment_text[:600]}"
                                + ("…" if len(comment_text) > 600 else "")
                            )

                    # LLM-generated summary
                    gen_summary = issue.get(
                        "assigneeCommentSummary",
                        issue.get("generated_summary", ""),
                    )
                    if gen_summary:
                        st.markdown("---")
                        st.markdown("**🤖 AI-Generated Summary:**")
                        st.markdown(gen_summary)

        # ── Process Gaps ──────────────────────────────────────────
        st.divider()
        st.subheader("Process Gap Checklist for Release Readiness Review")
        st.caption("AI-generated by backend based on combined SPRLL descriptions")

        for gap in process_gaps:
            st.markdown(
                f"**{gap.get('number', '-')}. {gap.get('title', 'Untitled')}**"
            )
            st.markdown(f"> {gap.get('description', '')}")
            st.write("")

    except Exception as e:
        st.error(f"Analyze failed: {e}")