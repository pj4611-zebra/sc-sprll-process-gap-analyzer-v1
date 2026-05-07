"""SPRLL Process Gap Analyzer - Streamlit Frontend (calls FastAPI backend)."""
import io
import json
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

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
except ImportError:
    Workbook = None

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

DISCIPLINES = [
    "Tools-Software Engineering",
    "EMC Cloud-Software Engineering",
]


def build_payload(mode, sprll_numbers, prompt_option, from_date=None, to_date=None, discipline=None):
    if mode == "Enter SPRLL Numbers manually":
        return {"sprll_numbers": sprll_numbers, "prompt_option": prompt_option}
    return {
        "from_date": str(from_date),
        "to_date": str(to_date),
        "discipline": discipline,
        "prompt_option": prompt_option,
    }


def generate_missing_fields_excel(issues):
    """Generate an Excel report of missing fields per SPRLL issue."""
    if Workbook is None:
        return None

    wb = Workbook()
    ws = wb.active
    ws.title = "Missing Fields Report"

    # Styles
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    error_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    error_font = Font(color="9C0006")
    wrap_alignment = Alignment(wrap_text=True, vertical="top")

    # Headers
    headers = ["SPRLL Number", "Summary", "Status", "Discipline", "Missing Fields", "Remarks"]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = wrap_alignment

    # Data rows
    row = 2
    for issue in issues:
        sprll_num = issue.get("sprllNumber", issue.get("key", ""))
        has_error = issue.get("error")

        ws.cell(row=row, column=1, value=sprll_num).alignment = wrap_alignment

        if has_error:
            # Not a valid SPRLL or fetch failed
            ws.cell(row=row, column=2, value="—").alignment = wrap_alignment
            ws.cell(row=row, column=3, value="—").alignment = wrap_alignment
            ws.cell(row=row, column=4, value="—").alignment = wrap_alignment
            ws.cell(row=row, column=5, value="—").alignment = wrap_alignment
            remark_cell = ws.cell(row=row, column=6, value=f"Not a proper SPRLL number or fetch error: {has_error}")
            remark_cell.alignment = wrap_alignment
            remark_cell.fill = error_fill
            remark_cell.font = error_font
        else:
            summary = issue.get("summary", "—")
            status = issue.get("status", "—")
            discipline = issue.get("discipline", issue.get("customfield_12801", "—"))
            missing = issue.get("missingFields", issue.get("missing_fields", []))
            missing_str = ", ".join(missing) if missing else "None"

            ws.cell(row=row, column=2, value=summary).alignment = wrap_alignment
            ws.cell(row=row, column=3, value=status).alignment = wrap_alignment
            ws.cell(row=row, column=4, value=discipline).alignment = wrap_alignment
            ws.cell(row=row, column=5, value=missing_str).alignment = wrap_alignment

            if missing:
                remark = f"{len(missing)} field(s) missing"
                remark_cell = ws.cell(row=row, column=6, value=remark)
                remark_cell.alignment = wrap_alignment
                remark_cell.fill = error_fill
                remark_cell.font = error_font
            else:
                ws.cell(row=row, column=6, value="All required fields present").alignment = wrap_alignment

        row += 1

    # Column widths
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 30
    ws.column_dimensions["E"].width = 45
    ws.column_dimensions["F"].width = 45

    # Freeze header row
    ws.freeze_panes = "A2"

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


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

# Prompt option is fixed to Option 1 (Senior Analyst)
prompt_option = 1

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
        prompt_option=prompt_option,
        from_date=locals().get("from_date"),
        to_date=locals().get("to_date"),
        discipline=locals().get("discipline"),
    )

    try:
        with st.spinner("Fetching the result from the Vertex AI"):
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
        used_prompt = result.get("prompt_option", prompt_option)

        # Store issues in session state for download
        st.session_state["last_issues"] = issues

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

        # Download missing fields report button
        if Workbook is not None:
            excel_buffer = generate_missing_fields_excel(issues)
            if excel_buffer:
                st.download_button(
                    label="📥 Download Missing Fields Report (Excel)",
                    data=excel_buffer,
                    file_name="sprll_missing_fields_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.warning("Install `openpyxl` to enable Excel download: `pip install openpyxl`")

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

                    source = issue.get("source", "")
                    if source:
                        st.caption(f"📦 Source: `{source}`")

                    if issue.get("description"):
                        desc = issue["description"]
                        st.markdown("---")
                        st.markdown("**Description:**")
                        st.markdown(
                            desc[:800] + ("…" if len(desc) > 800 else "")
                        )

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

                    gen_summary = issue.get(
                        "assigneeCommentSummary",
                        issue.get("generated_summary", ""),
                    )
                    if gen_summary:
                        st.markdown("---")
                        st.markdown("**🤖 AI-Generated Summary:**")
                        # Try to parse JSON summary for structured display
                        try:
                            summary_json = json.loads(gen_summary)
                            if isinstance(summary_json, dict):
                                if "summary" in summary_json:
                                    st.markdown(f"**Summary:** {summary_json['summary']}")
                                if "key_points" in summary_json:
                                    st.markdown("**Key Points:**")
                                    for kp in summary_json["key_points"]:
                                        st.markdown(f"- {kp}")
                                if "key_actions_or_decisions" in summary_json:
                                    st.markdown("**Key Actions/Decisions:**")
                                    for ka in summary_json["key_actions_or_decisions"]:
                                        st.markdown(f"- {ka}")
                                if "evidence" in summary_json:
                                    st.markdown("**Evidence:**")
                                    if isinstance(summary_json["evidence"], list):
                                        for ev in summary_json["evidence"]:
                                            st.markdown(f"- {ev}")
                                    else:
                                        st.markdown(str(summary_json["evidence"]))
                                if "confidence" in summary_json:
                                    st.markdown(f"**Confidence:** {summary_json['confidence']}")
                                if "process_relevance" in summary_json:
                                    st.markdown(f"**Process Relevance:** {summary_json['process_relevance']}")
                                if "limitations" in summary_json and summary_json["limitations"]:
                                    st.markdown(f"**Limitations:** {summary_json['limitations']}")
                            else:
                                st.markdown(gen_summary)
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(gen_summary)

        # ── Process Gaps ──────────────────────────────────────────
        st.divider()
        st.subheader("Process Gap Checklist for Release Readiness Review")
        st.caption("AI-generated based on combined SPRLL descriptions")

        for gap in process_gaps:
            num = gap.get("number", "-")
            title = gap.get("title", "Untitled")
            st.markdown(f"**{num}. {title}**")

            # Process area
            if gap.get("process_area"):
                st.markdown(f"**Process Area:** {gap['process_area']}")

            # Description (Option 1) or gap_description (Option 2)
            desc = gap.get("description") or gap.get("gap_description", "")
            if desc:
                st.markdown(f"> {desc}")

            # Evidence
            if gap.get("evidence"):
                st.markdown(f"**Evidence:** _{gap['evidence']}_")

            # Recommended fix
            if gap.get("recommended_fix"):
                st.markdown(f"**Recommended Fix:** {gap['recommended_fix']}")

            # Related SPRLLs
            if gap.get("related_sprll"):
                sprll_list = ", ".join(gap["related_sprll"])
                st.markdown(f"**🔗 Related SPRLL(s) ({len(gap['related_sprll'])}):** `{sprll_list}`")
                

            st.write("")

    except Exception as e:
        st.error(f"Analyze failed: {e}")