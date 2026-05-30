import base64
import html as _html
import io
import json
import os
import re

import requests as req_lib
import streamlit as st
from datetime import date

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
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────── helpers ───────────────────────────
def esc(value) -> str:
    if value is None:
        return ""
    return _html.escape(str(value))


def esc_multiline(value) -> str:
    return esc(value).replace("\n", "<br>")


def load_css():
    css_path = os.path.join(APP_DIR, "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_data(ttl=3600, show_spinner=False)
def load_discipline_product_map() -> dict:
    """Fetch {discipline: [products]} map from backend, cached for 1 hour."""
    try:
        resp = req_lib.get(f"{BACKEND_URL}/api/discipline-products", timeout=120)
        if resp.ok:
            return resp.json().get("discipline_products", {})
    except Exception:
        pass
    return {}


def build_payload(mode, sprll_numbers, prompt_option, from_date=None, to_date=None, discipline=None, products=None, custom_jql=None):
    if mode == "Enter SPRLL Numbers manually":
        return {"sprll_numbers": sprll_numbers, "prompt_option": prompt_option}
    if custom_jql:
        return {"custom_jql": custom_jql, "prompt_option": prompt_option}
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

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="6366F1", end_color="6366F1", fill_type="solid")
    error_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    error_font = Font(color="9F1239")
    wrap_alignment = Alignment(wrap_text=True, vertical="top")

    headers = ["SPRLL Number", "Summary", "Status", "Discipline", "Missing Fields", "Remarks"]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = wrap_alignment

    row = 2
    for issue in issues:
        sprll_num = issue.get("sprllNumber", issue.get("key", ""))
        has_error = issue.get("error")

        ws.cell(row=row, column=1, value=sprll_num).alignment = wrap_alignment

        if has_error:
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

    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 30
    ws.column_dimensions["E"].width = 45
    ws.column_dimensions["F"].width = 45
    ws.freeze_panes = "A2"

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


# ─────────────────────────── page setup ───────────────────────────
st.set_page_config(
    page_title="SPRLL Release Readiness Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css()


# ─────────────────────────── hero header ───────────────────────────
logo_path = os.path.join(APP_DIR, "static", "zebra_logo.jpeg")
logo_html = ""
if os.path.exists(logo_path):
    try:
        logo_b64 = b64_image(logo_path)
        logo_html = (
            f'<div class="hero-logo">'
            f'<img src="data:image/jpeg;base64,{logo_b64}" '
            f'style="height:46px;display:block;" alt="Zebra"/>'
            f'</div>'
        )
    except Exception:
        logo_html = ""

st.markdown(
    f"""
    <div class="hero-header">
      <div class="hero-content">
        <div>
          <div class="hero-badge"><span class="dot"></span> Quality Engineering · Release Intelligence</div>
          <h1 class="hero-title">SPRLL Release Readiness Analyzer</h1>
          <div class="hero-subtitle">
            AI-powered analysis of release-blocking signals across SPRLL issues —
            uncovering systemic gaps, validating findings, and driving smarter releases.
          </div>
        </div>
        {logo_html}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────── analysis parameters ───────────────────────────
st.markdown('<div class="section-header">Analysis Parameters</div>', unsafe_allow_html=True)

prompt_option = 1  # fixed (Senior Analyst)

mode = st.radio(
    "Input mode",
    ["Enter SPRLL Numbers manually", "Search by Date & Discipline"],
    horizontal=True,
    label_visibility="collapsed",
)

sprll_numbers = []
from_date = None
to_date = None
discipline = None
selected_products = None
jql = None

if mode == "Enter SPRLL Numbers manually":
    sprll_raw = st.text_input(
        "SPRLL Number(s)",
        placeholder="e.g.  4212, 3974, 4112",
        help="Separate multiple numbers with commas or spaces. The 'SPRLL-' prefix is added automatically.",
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
        chips_html = " ".join(f'<span class="sprll-chip">{esc(n)}</span>' for n in sprll_numbers)
        st.markdown(
            f'<div style="margin-top:0.6rem;"><span class="field-label">Will analyze</span>'
            f'<div class="sprll-chips" style="margin-top:0.35rem;">{chips_html}</div></div>',
            unsafe_allow_html=True,
        )
else:
    with st.spinner("Loading disciplines & products from Jira…"):
        disc_prod_map = load_discipline_product_map()

    available_disciplines = sorted(disc_prod_map.keys()) if disc_prod_map else []

    c1, c2, c3 = st.columns(3)
    with c1:
        from_date = st.date_input("From Date", value=date(date.today().year, 1, 1))
    with c2:
        to_date = st.date_input("To Date", value=date.today())
    with c3:
        discipline = st.selectbox(
            "Discipline",
            options=available_disciplines,
            help="Disciplines are loaded live from Jira.",
        )

    available_products = disc_prod_map.get(discipline, []) if discipline else []

    selected_products = st.multiselect(
        "Product (optional)",
        options=available_products,
        placeholder="Select one or more products…" if available_products else "No products for this discipline",
        help="Products are filtered by the selected discipline, loaded live from Jira.",
    )

    disc_clause = f'type in (Process, "Lesson Learned") AND Discipline = "{discipline}"'
    date_filter = f'created >= "{from_date}" AND created <= "{to_date}"'
    status_filter = "status in (Resolved, Closed)"

    if selected_products:
        prod_list = ", ".join(f'"{p}"' for p in selected_products)
        prod_clause = f'type in (Process, "Lesson Learned") AND "Product Selection" in ({prod_list})'
        default_jql = (
            f"project = SPRLL AND "
            f"({disc_clause} OR {prod_clause}) AND "
            f"{status_filter} AND {date_filter}"
        )
    else:
        default_jql = (
            f"project = SPRLL AND {disc_clause} AND "
            f"{status_filter} AND {date_filter}"
        )

    with st.expander("JQL Query Preview", expanded=True):
        jql = st.text_area("Edit the JQL query if needed:", value=default_jql, height=100)

st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
analyze_clicked = st.button("🚀  Run Analysis", type="primary", use_container_width=False)


# ─────────────────────────── analysis output ───────────────────────────
if analyze_clicked:
    payload = build_payload(
        mode=mode,
        sprll_numbers=sprll_numbers,
        prompt_option=prompt_option,
        from_date=from_date,
        to_date=to_date,
        discipline=discipline,
        products=selected_products,
        custom_jql=jql,
    )

    try:
        with st.spinner("Analyzing with Vertex AI · This may take a moment…"):
            resp = req_lib.post(f"{BACKEND_URL}/api/analyze", json=payload, timeout=180)

        if not resp.ok:
            st.error(f"Backend returned {resp.status_code}: {resp.text[:500]}")
            st.stop()

        result = resp.json()
        issues = result.get("issues", [])
        process_gaps = result.get("process_gaps", [])
        resolved_sprll_numbers = result.get("sprll_numbers", [])
        st.session_state["last_issues"] = issues

        # ───── Scope bar ─────
        if mode == "Search by Date & Discipline":
            scope_html = (
                f'<div class="scope-bar">'
                f'<span class="scope-item">📅 <strong>{esc(from_date)}</strong> → <strong>{esc(to_date)}</strong></span>'
                f'<span class="scope-divider">•</span>'
                f'<span class="scope-item">🎯 Discipline: <strong>{esc(discipline)}</strong></span>'
                f'<span class="scope-divider">•</span>'
                f'<span class="scope-item"><strong>{len(resolved_sprll_numbers)}</strong> issue(s) in scope</span>'
                f'</div>'
            )
        else:
            scope_html = (
                f'<div class="scope-bar">'
                f'<span class="scope-item">🎯 <strong>{len(resolved_sprll_numbers)}</strong> SPRLL issue(s) analyzed manually</span>'
                f'</div>'
            )
        st.markdown(scope_html, unsafe_allow_html=True)

        # ───── KPI cards ─────
        total_issues = len(issues)
        total_gaps = len(process_gaps)

        compliant = sum(
            1 for i in issues
            if not i.get("error") and not (i.get("missingFields") or i.get("missing_fields") or [])
        )
        analyzed_non_error = sum(1 for i in issues if not i.get("error"))
        compliance_pct = int(round((compliant / analyzed_non_error) * 100)) if analyzed_non_error else 0

        validations = [g.get("validation") for g in process_gaps if g.get("validation")]
        valid_count = sum(1 for v in validations if (v.get("validation_result") or "").lower() == "valid")
        validated_pct = int(round((valid_count / len(validations)) * 100)) if validations else None

        kpi_html = '<div class="kpi-row">'
        kpi_html += (
            f'<div class="kpi-card primary">'
            f'<div class="kpi-label">Issues Analyzed</div>'
            f'<div class="kpi-value">{total_issues}</div>'
            f'<div class="kpi-sub">SPRLL items in this run</div>'
            f'</div>'
        )
        kpi_html += (
            f'<div class="kpi-card info">'
            f'<div class="kpi-label">Gaps Identified</div>'
            f'<div class="kpi-value">{total_gaps}</div>'
            f'<div class="kpi-sub">AI-generated release findings</div>'
            f'</div>'
        )
        kpi_html += (
            f'<div class="kpi-card success">'
            f'<div class="kpi-label">Field Compliance</div>'
            f'<div class="kpi-value">{compliance_pct}%</div>'
            f'<div class="kpi-sub">{compliant} of {analyzed_non_error} fully populated</div>'
            f'</div>'
        )
        if validated_pct is not None:
            kpi_html += (
                f'<div class="kpi-card warning">'
                f'<div class="kpi-label">Judge Validated</div>'
                f'<div class="kpi-value">{validated_pct}%</div>'
                f'<div class="kpi-sub">{valid_count} of {len(validations)} rated <em>Valid</em></div>'
                f'</div>'
            )
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ───── Tabs ─────
        tab_gaps, tab_issues = st.tabs([
            f"  Release Readiness Findings ({total_gaps})  ",
            f"  Analyzed Issues ({total_issues})  ",
        ])

        # ───── Gaps tab ─────
        with tab_gaps:
            if not process_gaps:
                st.markdown(
                    '<div class="empty-state">No gaps were generated for this scope.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="section-sub">Findings synthesized from combined SPRLL evidence, '
                    'each independently scored by an LLM judge.</div>',
                    unsafe_allow_html=True,
                )

                for idx, gap in enumerate(process_gaps, 1):
                    num = gap.get("number", idx)
                    title = gap.get("title", "Untitled")
                    desc = gap.get("description") or gap.get("gap_description", "")
                    evidence = gap.get("evidence", "")
                    fix = gap.get("recommended_fix", "")
                    lifecycle_phase = gap.get("lifecycle_phase") or gap.get("phase", "")
                    related = gap.get("related_sprll") or []
                    validation = gap.get("validation") or {}

                    meta_badges = []
                    if lifecycle_phase:
                        meta_badges.append(
                            f'<span class="badge badge-purple">⏱ {esc(lifecycle_phase)}</span>'
                        )
                    v_result = validation.get("validation_result", "")
                    v_score = validation.get("validation_score", "")
                    if v_result:
                        badge_class = {
                            "Valid": "badge-success",
                            "Partially Valid": "badge-warning",
                            "Invalid": "badge-danger",
                        }.get(v_result, "badge-neutral")
                        score_txt = f" · {esc(v_score)}/5" if v_score != "" else ""
                        meta_badges.append(
                            f'<span class="badge {badge_class}">⚖ {esc(v_result)}{score_txt}</span>'
                        )

                    keys = []
                    for e in related:
                        if isinstance(e, dict):
                            k = e.get("key", "")
                        else:
                            k = str(e)
                        if k:
                            keys.append(k)
                    chips_html = "".join(f'<span class="sprll-chip">{esc(k)}</span>' for k in keys)

                    card = ['<div class="gap-card">']
                    card.append('<div class="gap-card-header">')
                    card.append(f'<div class="gap-number">{esc(num)}</div>')
                    card.append('<div class="gap-title-wrap">')
                    if meta_badges:
                        card.append(f'<div class="gap-meta">{"".join(meta_badges)}</div>')
                    card.append(f'<div class="gap-title">{esc(title)}</div>')
                    card.append('</div></div>')  # /title-wrap, /header

                    if desc:
                        card.append(f'<div class="gap-desc">{esc_multiline(desc)}</div>')

                    if evidence:
                        card.append(
                            '<div class="gap-section">'
                            '<div class="gap-section-label">📌 Evidence</div>'
                            f'<div class="gap-section-body italic">{esc_multiline(evidence)}</div>'
                            '</div>'
                        )
                    if fix:
                        card.append(
                            '<div class="gap-section">'
                            '<div class="gap-section-label">✅ Recommended Fix</div>'
                            f'<div class="gap-section-body">{esc_multiline(fix)}</div>'
                            '</div>'
                        )
                    if chips_html:
                        card.append(
                            '<div class="gap-section">'
                            f'<div class="gap-section-label">🔗 Related SPRLLs ({len(keys)})</div>'
                            f'<div class="sprll-chips">{chips_html}</div>'
                            '</div>'
                        )

                    card.append('</div>')  # /gap-card
                    st.markdown("".join(card), unsafe_allow_html=True)

                    # Validation drill-down (as Streamlit expander placed right after)
                    if validation:
                        v_color = {
                            "Valid": "🟢",
                            "Partially Valid": "🟡",
                            "Invalid": "🔴",
                        }.get(v_result, "⚪")
                        label = f"{v_color}  LLM Judge Details — {v_result or 'N/A'}"
                        if v_score != "":
                            label += f"  (Score {v_score}/5)"
                        with st.expander(label):
                            v_grid = ['<div class="field-grid">']
                            persona = validation.get("assigned_persona")
                            conf = validation.get("confidence")
                            if persona:
                                v_grid.append(
                                    f'<div class="field-item"><div class="field-label">Persona</div>'
                                    f'<div class="field-value">{esc(persona)}</div></div>'
                                )
                            if conf:
                                v_grid.append(
                                    f'<div class="field-item"><div class="field-label">Confidence</div>'
                                    f'<div class="field-value mono">{esc(conf)}</div></div>'
                                )
                            v_grid.append('</div>')
                            st.markdown("".join(v_grid), unsafe_allow_html=True)

                            if validation.get("reason"):
                                st.markdown(
                                    '<div class="gap-section-label">Reasoning</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(validation["reason"])
                            v_issues = validation.get("identified_issues") or []
                            if v_issues:
                                st.markdown(
                                    '<div class="gap-section-label" style="margin-top:0.8rem;">Identified Issues</div>',
                                    unsafe_allow_html=True,
                                )
                                for vi in v_issues:
                                    st.markdown(f"- {vi}")
                            improved = validation.get("improved_recommendation", "")
                            if improved:
                                st.markdown(
                                    '<div class="gap-section-label" style="margin-top:0.8rem;">Improved Recommendation</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(improved)

        # ───── Issues tab ─────
        with tab_issues:
            if Workbook is not None:
                excel_buffer = generate_missing_fields_excel(issues)
                if excel_buffer:
                    col_dl, _ = st.columns([1, 3])
                    with col_dl:
                        st.download_button(
                            label="📥  Download Missing Fields Report",
                            data=excel_buffer,
                            file_name="sprll_missing_fields_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
            else:
                st.warning("Install `openpyxl` to enable Excel download: `pip install openpyxl`")

            if not issues:
                st.markdown(
                    '<div class="empty-state">No issues to display.</div>',
                    unsafe_allow_html=True,
                )

            for issue in issues:
                sprll_num = issue.get("sprllNumber", issue.get("key", "SPRLL"))
                summary = issue.get("summary", "")
                missing = issue.get("missingFields", issue.get("missing_fields", []))
                has_error = issue.get("error")

                if has_error:
                    status_emoji = "❌"
                elif missing:
                    status_emoji = "⚠️"
                else:
                    status_emoji = "✅"

                label = f"{status_emoji}  {sprll_num}  —  {summary[:120]}"

                with st.expander(label, expanded=False):
                    if has_error:
                        st.error(issue["error"])
                        continue

                    status = issue.get("status", "—")
                    discipline_val = issue.get("discipline", issue.get("customfield_12801", "—"))
                    assignee = issue.get("assignee_name", "—")
                    comment_count = issue.get(
                        "matchedCommentCount", issue.get("comment_count", 0)
                    )

                    grid = ['<div class="field-grid">']
                    grid.append(
                        f'<div class="field-item"><div class="field-label">Status</div>'
                        f'<div class="field-value">{esc(status)}</div></div>'
                    )
                    grid.append(
                        f'<div class="field-item"><div class="field-label">Discipline</div>'
                        f'<div class="field-value">{esc(discipline_val)}</div></div>'
                    )
                    grid.append(
                        f'<div class="field-item"><div class="field-label">Assignee</div>'
                        f'<div class="field-value">{esc(assignee)}</div></div>'
                    )
                    grid.append(
                        f'<div class="field-item"><div class="field-label">Assignee Comments</div>'
                        f'<div class="field-value">{esc(comment_count)}</div></div>'
                    )
                    grid.append('</div>')
                    st.markdown("".join(grid), unsafe_allow_html=True)

                    # compliance badge
                    if missing:
                        bad_html = (
                            f'<div style="margin-top:0.25rem;">'
                            f'<span class="badge badge-warning">⚠ {len(missing)} field(s) missing</span>'
                            f'<div style="margin-top:0.4rem;color:#92400e;font-size:0.88rem;">'
                            f'{esc(", ".join(missing))}</div></div>'
                        )
                    else:
                        bad_html = (
                            '<div style="margin-top:0.25rem;">'
                            '<span class="badge badge-success">✓ All required fields present</span>'
                            '</div>'
                        )
                    st.markdown(bad_html, unsafe_allow_html=True)

                    source = issue.get("source", "")
                    if source:
                        st.markdown(
                            f'<div class="tiny muted" style="margin-top:0.6rem;">📦 Source: '
                            f'<code>{esc(source)}</code></div>',
                            unsafe_allow_html=True,
                        )

                    desc = issue.get("description", "")
                    if desc:
                        st.markdown(
                            '<div class="gap-section-label" style="margin-top:1rem;">Description</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(desc[:800] + ("…" if len(desc) > 800 else ""))

                    assignee_comments = issue.get("assignee_comments", [])
                    if assignee_comments:
                        st.markdown(
                            f'<div class="gap-section-label" style="margin-top:1rem;">'
                            f'Assignee Comments ({len(assignee_comments)})</div>',
                            unsafe_allow_html=True,
                        )
                        for i, comment_text in enumerate(assignee_comments, 1):
                            st.markdown(
                                f"**Comment {i}:**\n\n"
                                + comment_text[:600]
                                + ("…" if len(comment_text) > 600 else "")
                            )

                    gen_summary = issue.get(
                        "assigneeCommentSummary",
                        issue.get("generated_summary", ""),
                    )
                    if gen_summary:
                        st.markdown(
                            '<div class="gap-section-label" style="margin-top:1rem;">🤖 AI Summary</div>',
                            unsafe_allow_html=True,
                        )
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
                                    st.markdown("**Key Actions / Decisions:**")
                                    for ka in summary_json["key_actions_or_decisions"]:
                                        st.markdown(f"- {ka}")
                                if "evidence" in summary_json:
                                    st.markdown("**Evidence:**")
                                    if isinstance(summary_json["evidence"], list):
                                        for ev in summary_json["evidence"]:
                                            st.markdown(f"- _{ev}_")
                                    else:
                                        st.markdown(f"_{summary_json['evidence']}_")
                                if "confidence" in summary_json:
                                    st.markdown(f"**Confidence:** `{summary_json['confidence']}`")
                                if "limitations" in summary_json and summary_json["limitations"]:
                                    st.markdown(f"**Limitations:** {summary_json['limitations']}")
                            else:
                                st.markdown(gen_summary)
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(gen_summary)

    except Exception as e:
        st.error(f"Analysis failed: {e}")
