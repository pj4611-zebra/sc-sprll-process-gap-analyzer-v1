import concurrent.futures
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests as req_lib
from google import genai  # type: ignore
from google.genai.types import GenerateContentConfig  # type: ignore
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .config import get_settings


# =========================
# Field mapping for missing-field checks
# =========================
FIELD_MAPPING = {
    "key": "SPRLL Number",
    "summary": "Summary",
    "description": "Description",
    "customfield_12801": "Discipline",
    "status": "Status",
    "customfield_14501": "LL Type - Primary",
    "customfield_14502": "LL Type - Secondary",
}


# =========================
# Normalisation helpers
# =========================
def _normalize_sprll_number(sprll_number: str) -> str:
    value = (sprll_number or "").strip().upper()
    if not value:
        return value
    if re.match(r"^SPRLL-\d+$", value):
        return value
    num = re.sub(r"^SPRLL-?", "", value)
    if num.isdigit():
        return f"SPRLL-{num}"
    if not value.startswith("SPRLL-"):
        return f"SPRLL-{value}"
    return value


# =========================
# Jira helpers
# =========================
def _jira_headers() -> Dict[str, str]:
    s = get_settings()
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {s.jira_token}",
        "Content-Type": "application/json",
    }


# =========================
# ADF (Atlassian Document Format) → plain text
# =========================
def adf_to_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(item) for item in node)
    if isinstance(node, dict):
        node_type = node.get("type")
        text_value = node.get("text", "")
        content = node.get("content", [])
        if node_type in {"paragraph", "heading"}:
            return adf_to_text(content) + "\n"
        if node_type == "hardBreak":
            return "\n"
        if node_type in {"bulletList", "orderedList"}:
            return adf_to_text(content)
        if node_type == "listItem":
            return f"- {adf_to_text(content)}\n"
        if text_value:
            return text_value
        return adf_to_text(content)
    return str(node)


def normalize_rich_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = adf_to_text(value).strip()
        return text if text else json.dumps(value)
    return "" if value is None else str(value)


# =========================
# Field extraction helpers
# =========================
def extract_field_value(val: Any) -> Any:
    if isinstance(val, dict):
        return val.get("name") or val.get("value")
    if isinstance(val, list):
        return None if len(val) == 0 else str(val)
    return val


def check_missing_fields(issue_key: str, fields: Dict[str, Any]) -> List[str]:
    missing = []
    for field_id, field_name in FIELD_MAPPING.items():
        val = issue_key if field_id == "key" else fields.get(field_id)
        val = extract_field_value(val)
        if val is None or str(val).strip() == "" or str(val).strip().upper() == "NA":
            missing.append(field_name)
    return missing


def _extract_discipline(fields: Dict[str, Any]) -> str:
    discipline_raw = fields.get("customfield_12801")
    if isinstance(discipline_raw, dict):
        return (
            discipline_raw.get("value")
            or discipline_raw.get("name")
            or json.dumps(discipline_raw)
        )
    if isinstance(discipline_raw, list):
        vals = []
        for item in discipline_raw:
            if isinstance(item, dict):
                vals.append(item.get("value") or item.get("name") or str(item))
            else:
                vals.append(str(item))
        return ", ".join(v for v in vals if v)
    return str(discipline_raw) if discipline_raw else ""


def _extract_assignee_name(fields: Dict[str, Any]) -> str:
    assignee = fields.get("assignee") or {}
    return assignee.get("name") or assignee.get("displayName") or ""


def _extract_assignee_comments(
    issue_json: Dict[str, Any], assignee_name: str
) -> List[str]:
    comments: List[str] = []
    issue_fields = issue_json.get("fields", {})
    comment_block = issue_fields.get("comment", {})
    comment_items = (
        comment_block.get("comments", []) if isinstance(comment_block, dict) else []
    )
    for c in comment_items:
        author = c.get("author") or {}
        author_name = author.get("name") or author.get("displayName") or ""
        if assignee_name and author_name == assignee_name:
            body = c.get("body")
            body_text = normalize_rich_text(body)
            if body_text:
                comments.append(body_text)
    return comments


# =========================
# Vertex AI (Gemini) helpers via google-genai SDK
# =========================
@lru_cache
def _get_genai_client() -> genai.Client:
    s = get_settings()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = s.google_application_credentials
    return genai.Client(
        vertexai=True,
        project=s.gcp_project,
        location=s.gcp_location,
    )


# =========================
# Prompt sets for Option 1 and Option 2
# =========================

# --- OPTION 1: Process Gap prompts ---
PROCESS_GAP_PROMPT_1 = """
You are a Senior Quality Engineering Process Gap Analyst with deep expertise in defect prevention, release governance, root-cause analysis, SDLC escape analysis, and continuous improvement.

You will receive combined descriptions from multiple SPRLL records (Lessons Learned from Customer Defects). These records are the ONLY source of truth.
​

Additional Context:
Each SPRLL may contain:
- LL Type-Primary
- LL Type-Secondary

These fields indicate:
- the defect escape category
- the underlying contributing/root cause

Your mission:
1. Analyze all SPRLL records together.
2. Identify the 5 most critical underlying process gaps that allowed issues to escape internal controls and reach the customer.
3. Classify each identified process gap into exactly ONE SDLC phase:
   - Coding Phase
   - Test Phase
   - Requirement Phase
   - Design Review Phase
   - Deployment Phase

Definition of Process Gap:
A process gap is a specific missing, weak, skipped, undefined, ineffective, or unenforced control in the software delivery lifecycle that, if properly implemented, would have prevented the issue or detected it before customer impact.

SDLC Phase Classification Objective:
Determine the EARLIEST reasonable SDLC phase where the issue should ideally have been prevented or detected before reaching the customer.

Core Interpretation Logic:
- LL Type-Primary represents the escape category.
- LL Type-Secondary represents the underlying contributing cause.

MANDATORY PHASE MAPPING RULES:
1. Config Error → Coding Phase
2. Invalid Issue → Test Phase
3. Third Party → Deployment Phase
4. Deployment → Deployment Phase
5. Code Error → Test Phase
6. Requirements → Requirement Phase

If explicit mappings above are available, ALWAYS prioritize them.

Example:
LL Type-Primary = "Test Escape"
LL Type-Secondary = "Code Error"
→ SDLC Phase = "Test Phase"
because the defect escaped during testing before release.

PHASE CLASSIFICATION GUIDANCE:

Coding Phase:
- Logic defects
- Coding mistakes
- Incorrect implementation
- Missed validations
- Security coding flaws
- Incorrect fixes or code changes
- Configuration implementation issues

Test Phase:
- Test escape
- Missing regression coverage
- Regression missed execution
- Missing negative or edge-case testing
- Validation gaps
- Performance/security testing gaps
- Environment/configuration testing misses
- Incomplete test scenarios
- Invalid issue validation misses
- Code error escape during testing

Requirement Phase:
- Requirement ambiguity
- Missing business rules
- Scope misunderstanding
- Feature parity expectations
- Standardization gaps
- Incorrect customer expectation handling
- New feature requests identified as defects

Design Review Phase:
- Architectural weaknesses
- Integration/interface design flaws
- Workflow inconsistencies
- Dependency handling issues
- Scalability limitations
- System behavior caused by design decisions
- Incorrect orchestration between components

Deployment Phase:
- Third-party dependency issues
- Deployment failures
- Environment deployment mismatches
- Release packaging/configuration problems
- Production deployment validation gaps

MANDATORY RULES:
1. Use only facts explicitly present in the input.
2. Do NOT assume missing details or use external knowledge.
3. Every gap MUST be supported by clear evidence from the input.
4. Do NOT produce vague statements such as:
   - Improve testing
   - Improve communication
   - Follow best practices
   - Increase ownership
5. Each gap must point to an exact broken or missing control such as:
   - lifecycle step
   - approval gate
   - checklist item
   - template field
   - review mechanism
   - validation rule
   - monitoring control
   - handoff process
   - release criterion
   - training control
6. Each gap must be unique, non-overlapping, and not a reworded duplicate.
7. Prefer systemic gaps that explain multiple issues over isolated one-off observations.
8. Prioritize gaps with the highest business value if fixed.
9. Be concise, specific, practical, and executive-friendly.
10. If evidence is weak, choose the best-supported interpretation only. Never fabricate.
11. Every identified process gap MUST be classified into exactly ONE SDLC phase.
12. If multiple phases seem possible, choose the EARLIEST preventable phase.
13. Use LL Type-Primary and LL Type-Secondary as primary classification signals whenever available.
14. If LL fields are missing or unclear, infer the SDLC phase using SPRLL evidence and best-supported engineering judgment only.
15. Before finalizing, merge semantically overlapping gaps into a single stronger gap.

IMPACT PRIORITIZATION (use for ranking):
Sort highest to lowest using these factors:
- Severity of customer/business impact
- Likelihood of recurrence
- Failure to detect earlier in lifecycle
- Breadth across modules, teams, releases, or customers
- Ease and value of preventive implementation

ANALYSIS LENSES (use internally only, do not output):
- Requirements completeness and ambiguity
- Design review effectiveness
- Unit / integration / regression coverage
- Negative and edge-case testing
- Environment and configuration parity
- Data validation and error handling
- Monitoring, logging, and alerting
- Release readiness review rigor
- Change management controls
- Deployment validation
- Ownership and handoff clarity
- SOP / checklist compliance
- Risk assessment quality
- Documentation quality
- Training readiness
- Dependency and interface validation

OUTPUT INSTRUCTIONS:
Return ONLY valid JSON.
No markdown.
No explanations.
No comments.
No extra text.

Return exactly 5 objects in a JSON array, sorted by highest impact first.

JSON Schema:
[
  {{
    "number": 1,
    "title": "Short specific gap title (max 10 words)",
    "phase": "Coding Phase | Test Phase | Requirement Phase | Design Review Phase | Deployment Phase",
    "process_area": "Exact affected lifecycle stage, gate, checklist, review, artifact, or control",
    "description": "Precise description of what control was missing, weak, skipped, or unenforced, why it enabled customer escape, and what exact control must now be added or strengthened.",
    "evidence": "Short quote or concise paraphrase from the input directly supporting this gap",
    "recommended_fix": "Concrete action that can be implemented immediately, including where in the process it should be added.",
    "confidence": "High | Medium | Low",
    "related_sprll": [
      {{
        "key": "SPRLL-XXXX",
        "phase": "Coding Phase | Test Phase | Requirement Phase | Design Review Phase | Deployment Phase"
      }}
    ]
  }}
]

RELATED SPRLL RULES:
- "related_sprll" MUST list every SPRLL key from the input whose evidence contributed to this gap.
- For each entry, "phase" is the SDLC phase inferred for THAT specific SPRLL record (may differ from the overall gap phase if multiple phases are represented).
- Use only SPRLL keys explicitly present in the input.

Context:
{combined_descriptions}
""".strip()

# --- Comment Summary prompts ---
COMMENT_SUMMARY_PROMPT_1 = """
You are a Senior Quality Intelligence Analyst specializing in extracting actionable insights from JIRA SPRLL (Lessons Learned) tickets.

You will receive:
1. Assignee name
2. Full JIRA comment history (author + timestamp + comment text)

Your task:
Summarize ONLY the meaningful comments written by the assignee.

Primary Objective:
Extract precise, evidence-based quality information that supports decision-making, lessons learned, process improvement, ownership clarity, and defect prevention.

Definition of Meaningful Comment:
Any assignee comment containing one or more of the following:
- root cause or technical explanation
- defect behavior or limitation
- ownership clarification
- fix status or workaround
- corrective or preventive action
- scope clarification
- dependency or handoff
- validation result
- why request is invalid / not a defect
- next required action
- process learning

Strict Rules:
1. Use ONLY comments authored by the assignee.
2. Ignore comments from all other users.
3. Match assignee names robustly (full name, short name, display variation, case differences).
4. Consider chronology: newer assignee comments override older comments if contradictory.
5. Ignore greetings, acknowledgements, reminders, approvals, status chatter, or text with no actionable value.
6. Do NOT infer beyond what is explicitly written.
7. Do NOT hallucinate missing context.
8. If meaning is unclear or evidence is insufficient, state that clearly.
9. Keep output concise, factual, and management-ready.
10. Preserve critical qualifiers such as "internal only", "invalid request", "covered by existing ticket", "pending other team", etc.

Output Instructions:
Return ONLY valid JSON. No markdown, no extra text.

Schema:
{{
  "assignee": "Exact assignee name",
  "summary": "Concise summary of only the assignee's meaningful comments, highlighting decisions, technical findings, ownership, and required actions.",
  "key_points": [
    "Specific actionable point 1",
    "Specific actionable point 2",
    "Specific actionable point 3"
  ],
  "evidence": [
    "Short direct quote or concise paraphrase from assignee comments"
  ],
  "confidence": "High / Medium / Low"
}}

SPRLL: {sprll_number}
Issue Summary: {issue_summary}
Issue Description: {issue_description}
Assignee: {assignee_name}

Assignee Comments:
{comments_text}
""".strip()

# Comment Summary Prompt 2 (kept for future use but not actively used)
COMMENT_SUMMARY_PROMPT_2 = """
You are an expert quality engineering analyst specializing in extracting actionable insights from Jira comments for SPRLL and process improvement.

Input: ONLY the comments written by the assignee ({assignee_name}) from a Jira SPRLL ticket. These comments are the ONLY source of truth. Do not use any information from the ticket description, root cause, or other fields.

Task:
Analyze the assignee's comments and produce a precise summary focused solely on high-quality, actionable information relevant to process gaps or preventive actions.

Strict Rules (MUST follow):
- Extract only specific, concrete details that can be directly translated into action (e.g., decisions on visibility, documentation, internal profiles, fixes, or invalidations).
- Never include generic statements, opinions, or vague summaries.
- Do not infer, assume, or add any information not explicitly stated in the assignee's comments.
- If the comments lack clear actionable content or contain ambiguity, explicitly state the limitation instead of guessing.
- Keep output concise, factual, and strictly grounded.

Return ONLY the following valid JSON object. No explanations, markdown, or extra text:

{{
  "summary": "concise paragraph (2-4 sentences maximum) containing only high-quality actionable insights from the assignee's comments",
  "key_actions_or_decisions": ["array of specific decisions, clarifications, or actions mentioned. Use empty array [] if none found"],
  "process_relevance": "one short sentence stating relevance to process gaps/lessons learned, or 'No clear process insight from assignee comments' if none",
  "limitations": "any notes on unclear, ambiguous, or missing information (empty string if everything is clear)"
}}

SPRLL: {sprll_number}
Issue Summary: {issue_summary}

Assignee Comments:
{comments_text}
""".strip()


def _build_summary_prompt(
    sprll_number: str,
    issue_summary: str,
    issue_description: str,
    comments_text: str,
    assignee_name: str = "",
    prompt_option: int = 1,
) -> str:
    # Always use prompt option 1
    return COMMENT_SUMMARY_PROMPT_1.format(
        sprll_number=sprll_number,
        issue_summary=issue_summary,
        issue_description=issue_description,
        comments_text=comments_text,
        assignee_name=assignee_name,
    )


def _summarize_with_vertex(
    sprll_number: str,
    issue_summary: str,
    issue_description: str,
    assignee_comments: List[str],
    assignee_name: str = "",
    prompt_option: int = 1,
) -> str:
    if not assignee_comments:
        return "No assignee comments found."

    client = _get_genai_client()
    s = get_settings()

    comments_text = "\n\n".join(f"- {c}" for c in assignee_comments)
    comments_text = comments_text[:12000]
    issue_description = (issue_description or "")[:4000]

    prompt = _build_summary_prompt(
        sprll_number=sprll_number,
        issue_summary=issue_summary,
        issue_description=issue_description,
        comments_text=comments_text,
        assignee_name=assignee_name,
        prompt_option=prompt_option,
    )

    try:
        response = client.models.generate_content(
            model=s.vertex_model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=800,
            ),
        )
        text = response.text
        if text and text.strip():
            return text.strip()
        return (
            "Executive Summary:\n- No summary generated.\n\n"
            "Key Issues:\n- N/A\n\n"
            "Recommended Actions:\n- N/A"
        )
    except Exception as ex:
        return (
            f"Executive Summary:\n- Failed to generate summary.\n\n"
            f"Key Issues:\n- Error: {ex}\n\n"
            f"Recommended Actions:\n- Retry after checking Vertex AI settings."
        )


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        text = text.replace("json", "", 1).strip()
    return text


def generate_process_gaps(
    combined_descriptions: str, prompt_option: int = 1
) -> List[Dict[str, Any]]:
    client = _get_genai_client()
    s = get_settings()

    # Always use PROCESS_GAP_PROMPT_1
    prompt = PROCESS_GAP_PROMPT_1.format(combined_descriptions=combined_descriptions)

    try:
        resp = client.models.generate_content(
            model=s.vertex_model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2000,
            ),
        )
        text = _strip_code_fences(resp.text or "[]")
        gaps = json.loads(text)

        while len(gaps) < 5:
            gaps.append(
                {
                    "number": len(gaps) + 1,
                    "title": f"Quality Gate {len(gaps) + 1}",
                    "process_area": "N/A",
                    "description": "Insufficient evidence to identify additional gap.",
                    "evidence": "N/A",
                    "recommended_fix": "N/A",
                    "related_sprll": [],
                }
            )
        return gaps[:5]
    except Exception as e:
        return [
            {
                "number": i + 1,
                "title": f"Process Gap {i + 1}",
                "description": f"AI generation failed: {type(e).__name__}: {e}",
                "related_sprll": [],
            }
            for i in range(5)
        ]


# =========================
# MongoDB helpers
# =========================
@lru_cache
def _get_mongo_client() -> Optional[MongoClient]:
    s = get_settings()
    try:
        client = MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print("[INFO] MongoDB connection successful.")
        return client
    except PyMongoError as ex:
        print(f"[ERROR] MongoDB connection failed: {ex}")
        return None


def _get_issue_collection():
    s = get_settings()
    client = _get_mongo_client()
    if client is None:
        return None
    return client[s.mongodb_db_name]["issues"]


def _strip_mongo_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _save_to_mongodb(
    key: str,
    summary: str,
    description: str,
    discipline: str,
    status: str,
    assignee_name: str,
    assignee_comments: List[str],
    generated_summary: str,
    missing_fields: List[str],
    ll_type_primary: str = "",
    ll_type_secondary: str = "",
) -> bool:
    collection = _get_issue_collection()
    if collection is None:
        print("[WARN] MongoDB collection not available. Skipping save.")
        return False

    try:
        document = {
            "key": key,
            "summary": summary,
            "description": description,
            "customfield_12801": discipline,
            "status": status,
            "assignee_name": assignee_name,
            "assignee_comments": assignee_comments,
            "generated_summary": generated_summary,
            "comment_count": len(assignee_comments),
            "missing_fields": missing_fields,
            "ll_type_primary": ll_type_primary,
            "ll_type_secondary": ll_type_secondary,
            "processed_at": datetime.now(timezone.utc),
        }

        result = collection.update_one(
            {"key": key},
            {"$set": document},
            upsert=True,
        )

        if result.upserted_id:
            print(f"[INFO] Inserted {key} into MongoDB (ID: {result.upserted_id})")
        else:
            print(f"[INFO] Updated {key} in MongoDB")
        return True

    except PyMongoError as ex:
        print(f"[ERROR] Failed to save {key} to MongoDB: {ex}")
        return False


def _load_from_mongodb(key: str) -> Optional[Dict[str, Any]]:
    collection = _get_issue_collection()
    if collection is None:
        return None
    try:
        doc = collection.find_one({"key": key})
        return _strip_mongo_id(doc) if doc else None
    except PyMongoError as ex:
        print(f"[ERROR] MongoDB read failed for {key}: {ex}")
        return None


# =========================
# Jira search
# =========================
def search_sprll_numbers(from_date: str, to_date: str, discipline: str) -> List[str]:
    s = get_settings()
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

    resp = req_lib.post(
        f"{s.jira_domain}/rest/api/2/search",
        headers=_jira_headers(),
        json={"jql": jql, "fields": ["key", "summary"], "maxResults": 100},
        timeout=30,
    )
    resp.raise_for_status()
    return [i["key"] for i in resp.json().get("issues", [])]


def search_sprll_numbers_by_jql(jql: str) -> List[str]:
    """Search SPRLL numbers using a custom JQL query provided by the user."""
    s = get_settings()
    resp = req_lib.post(
        f"{s.jira_domain}/rest/api/2/search",
        headers=_jira_headers(),
        json={"jql": jql, "fields": ["key", "summary"], "maxResults": 100},
        timeout=30,
    )
    resp.raise_for_status()
    return [i["key"] for i in resp.json().get("issues", [])]


PRODUCT_SELECTION_FIELD_ID = "customfield_21800"
DISCIPLINE_FIELD_ID_PRODUCT = "customfield_12801"


def fetch_discipline_product_map() -> Dict[str, List[str]]:
    """Fetch a mapping of {discipline: [products]} from SPRLL Jira issues dynamically."""
    s = get_settings()
    jql_query = 'project = SPRLL AND issuetype in (Process, "Lesson Learned")'

    start_at = 0
    max_results = 100
    all_issues: List[Dict[str, Any]] = []

    while True:
        resp = req_lib.post(
            f"{s.jira_domain}/rest/api/2/search",
            headers=_jira_headers(),
            json={
                "jql": jql_query,
                "startAt": start_at,
                "maxResults": max_results,
                "fields": ["summary", DISCIPLINE_FIELD_ID_PRODUCT, PRODUCT_SELECTION_FIELD_ID],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        if start_at + max_results >= data.get("total", 0):
            break
        start_at += max_results

    discipline_product_map: Dict[str, set] = {}

    for issue in all_issues:
        fields = issue.get("fields", {})

        discipline_val = fields.get(DISCIPLINE_FIELD_ID_PRODUCT)
        disciplines: List[str] = []
        if isinstance(discipline_val, dict):
            d = discipline_val.get("value") or discipline_val.get("name")
            if d:
                disciplines = [d]
        elif isinstance(discipline_val, list):
            for item in discipline_val:
                d = (item.get("value") or item.get("name")) if isinstance(item, dict) else str(item)
                if d:
                    disciplines.append(d)

        product_val = fields.get(PRODUCT_SELECTION_FIELD_ID)
        prod: Optional[str] = None
        if isinstance(product_val, dict):
            child = product_val.get("child")
            if child and isinstance(child, dict):
                prod = child.get("value")
            else:
                prod = product_val.get("value")

        for disc in disciplines:
            if disc not in discipline_product_map:
                discipline_product_map[disc] = set()
            if prod:
                discipline_product_map[disc].add(prod)

    return {disc: sorted(prods) for disc, prods in sorted(discipline_product_map.items())}


# =========================
# Core: fetch + extract + summarise + store
# =========================
def _extract_custom_field_value(fields: Dict[str, Any], field_id: str) -> str:
    """Extract a string value from a Jira custom field (handles dict/list/str)."""
    raw = fields.get(field_id)
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return raw.get("value") or raw.get("name") or json.dumps(raw)
    if isinstance(raw, list):
        vals = []
        for item in raw:
            if isinstance(item, dict):
                vals.append(item.get("value") or item.get("name") or str(item))
            else:
                vals.append(str(item))
        return ", ".join(v for v in vals if v)
    return str(raw)


def _fetch_and_process_issue(
    sprll_number: str, prompt_option: int = 1
) -> Dict[str, Any]:
    s = get_settings()
    sprll_number = _normalize_sprll_number(sprll_number)
    url = f"{s.jira_domain}/rest/api/2/issue/{sprll_number}"

    try:
        resp = req_lib.get(
            url,
            headers=_jira_headers(),
            params={
                "fields": "summary,description,status,assignee,comment,customfield_12801,customfield_14501,customfield_14502"
            },
            timeout=30,
        )
    except req_lib.RequestException as ex:
        return {
            "key": sprll_number,
            "error": f"Jira request failed: {ex}",
            "missing_fields": [],
        }

    if resp.status_code in (401, 403):
        return {
            "key": sprll_number,
            "error": "Authentication failed. Check Jira token.",
            "missing_fields": [],
        }
    if resp.status_code == 404:
        return {
            "key": sprll_number,
            "error": f"Issue {sprll_number} not found in Jira.",
            "missing_fields": [],
        }

    try:
        resp.raise_for_status()
    except req_lib.HTTPError as ex:
        return {
            "key": sprll_number,
            "error": f"Jira HTTP error: {ex}",
            "missing_fields": [],
        }

    issue_json = resp.json()
    fields = issue_json.get("fields", {})

    issue_key = issue_json.get("key", sprll_number)
    summary = fields.get("summary", "") or ""
    description_raw = fields.get("description")
    description = adf_to_text(description_raw).strip() if description_raw else ""
    discipline = _extract_discipline(fields)
    status_obj = fields.get("status") or {}
    status = (
        status_obj.get("name") if isinstance(status_obj, dict) else str(status_obj)
    ) or ""
    assignee_name = _extract_assignee_name(fields)
    assignee_comments = _extract_assignee_comments(issue_json, assignee_name)
    ll_type_primary = _extract_custom_field_value(fields, "customfield_14501")
    ll_type_secondary = _extract_custom_field_value(fields, "customfield_14502")
    missing_fields = check_missing_fields(issue_key, fields)

    generated_summary = _summarize_with_vertex(
        sprll_number=issue_key,
        issue_summary=summary,
        issue_description=description,
        assignee_comments=assignee_comments,
        assignee_name=assignee_name,
        prompt_option=prompt_option,
    )

    _save_to_mongodb(
        key=issue_key,
        summary=summary,
        description=description,
        discipline=discipline,
        status=status,
        assignee_name=assignee_name,
        assignee_comments=assignee_comments,
        generated_summary=generated_summary,
        missing_fields=missing_fields,
        ll_type_primary=ll_type_primary,
        ll_type_secondary=ll_type_secondary,
    )

    return {
        "key": issue_key,
        "summary": summary,
        "description": description,
        "customfield_12801": discipline,
        "status": status,
        "assignee_name": assignee_name,
        "assignee_comments": assignee_comments,
        "generated_summary": generated_summary,
        "comment_count": len(assignee_comments),
        "missing_fields": missing_fields,
        "ll_type_primary": ll_type_primary,
        "ll_type_secondary": ll_type_secondary,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def get_or_create_issue_document(
    sprll_number: str, prompt_option: int = 1
) -> Dict[str, Any]:
    key = _normalize_sprll_number(sprll_number)

    cached = _load_from_mongodb(key)
    if cached:
        cached["source"] = "mongodb_cache"
        return _ensure_frontend_keys(cached)

    doc = _fetch_and_process_issue(key, prompt_option=prompt_option)
    doc["source"] = "jira_fresh"
    return _ensure_frontend_keys(doc)


def _ensure_frontend_keys(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    out.setdefault("sprllNumber", out.get("key", ""))
    out.setdefault("discipline", out.get("customfield_12801", ""))
    out.setdefault("missingFields", out.get("missing_fields", []))
    out.setdefault("assigneeCommentSummary", out.get("generated_summary", ""))
    out.setdefault("matchedCommentCount", out.get("comment_count", 0))
    out.setdefault("assignee_comments", out.get("assignee_comments", []))
    out.setdefault("assignee_name", out.get("assignee_name", ""))
    out.setdefault("ll_type_primary", out.get("ll_type_primary", ""))
    out.setdefault("ll_type_secondary", out.get("ll_type_secondary", ""))
    return out


# =========================
# Parallel fetch
# =========================
def fetch_issues_parallel(
    sprll_numbers: List[str],
    prompt_option: int = 1,
) -> tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    descriptions: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(get_or_create_issue_document, n, prompt_option): n
            for n in sprll_numbers
        }
        for future in concurrent.futures.as_completed(future_map):
            try:
                issue = future.result()
            except Exception as ex:
                n = future_map[future]
                issue = {
                    "key": n,
                    "sprllNumber": n,
                    "error": str(ex),
                    "missing_fields": [],
                    "missingFields": [],
                }
            issues.append(issue)
            if not issue.get("error") and issue.get("description"):
                key = issue.get("sprllNumber", issue.get("key", ""))
                descriptions.append(f"[{key}]\n{issue['description']}")

    issues.sort(key=lambda x: x.get("sprllNumber", x.get("key", "")))
    return issues, descriptions


# =========================
# Sync endpoint helper
# =========================
def sync_assignee_comments(
    sprll_numbers: List[str], prompt_option: int = 1
) -> Dict[str, Any]:
    collection = _get_issue_collection()
    if collection is not None:
        try:
            collection.create_index("key", unique=True)
        except PyMongoError:
            pass

    stored, failed = 0, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(get_or_create_issue_document, n, prompt_option): n
            for n in sprll_numbers
        }
        for future in concurrent.futures.as_completed(future_map):
            n = future_map[future]
            try:
                doc = future.result()
                if not doc.get("error"):
                    stored += 1
                else:
                    failed.append(
                        {"sprllNumber": n, "error": doc.get("error", "Unknown error")}
                    )
            except Exception as ex:
                failed.append({"sprllNumber": n, "error": str(ex)})

    return {
        "requested": len(sprll_numbers),
        "stored": stored,
        "failedCount": len(failed),
        "failed": failed,
    }