import concurrent.futures
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests as req_lib
import vertexai  # type: ignore
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from vertexai.generative_models import GenerativeModel  # type: ignore

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
    #LLTYPE primary and LLTYpe Secondary , Action taken summary 
}


# =========================
# Normalisation helpers
# =========================
def _normalize_sprll_number(sprll_number: str) -> str:
    """
    Normalize a SPRLL number to the canonical 'SPRLL-XXXX' format.
    Accepts: '4212', 'SPRLL4212', 'sprll-4212', 'SPRLL-4212'.
    """
    value = (sprll_number or "").strip().upper()
    if not value:
        return value
    # Already canonical
    if re.match(r"^SPRLL-\d+$", value):
        return value
    # Strip any leading 'SPRLL-' or 'SPRLL' prefix, then re-add
    num = re.sub(r"^SPRLL-?", "", value)
    if num.isdigit():
        return f"SPRLL-{num}"
    # If it doesn't start with SPRLL-, add it
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
    """
    Convert Jira ADF (Atlassian Document Format) to plain text.
    Handles strings, dict nodes, and lists recursively.
    """
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
    """Convert a potentially ADF/dict/list value to plain text string."""
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
    """Extract discipline from customfield_12801 in various forms."""
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
    """Return the assignee's name (username first, fallback displayName)."""
    assignee = fields.get("assignee") or {}
    return assignee.get("name") or assignee.get("displayName") or ""


def _extract_assignee_comments(
    issue_json: Dict[str, Any], assignee_name: str
) -> List[str]:
    """
    Return plain-text bodies of comments authored by the assignee.
    Matching is done on author.name first, then author.displayName.
    """
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
# Vertex AI (Gemini) helpers
# =========================
@lru_cache
def _get_vertex_model() -> GenerativeModel:
    s = get_settings()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = s.google_application_credentials
    vertexai.init(project=s.gcp_project, location=s.gcp_location)
    return GenerativeModel(s.vertex_model)


def _build_summary_prompt(
    sprll_number: str,
    issue_summary: str,
    issue_description: str,
    comments_text: str,
) -> str:
    return f"""
You are a quality/process analyst.
Summarize assignee comments for one SPRLL issue.
Focus on: completed work, blockers, risks, root causes, and next actions.
Be concise, accurate, and do not invent information.

SPRLL: {sprll_number}
Issue Summary: {issue_summary}
Issue Description: {issue_description}

Assignee Comments:
{comments_text}

Return plain text in this exact structure:
Executive Summary:
- ...

Key Issues:
- ...

Recommended Actions:
- ...
""".strip()


def _summarize_with_vertex(
    sprll_number: str,
    issue_summary: str,
    issue_description: str,
    assignee_comments: List[str],
) -> str:
    """Generate an LLM summary of the assignee's comments."""
    if not assignee_comments:
        return "No assignee comments found."

    model = _get_vertex_model()

    comments_text = "\n\n".join(f"- {c}" for c in assignee_comments)
    # Truncate to stay within token limits
    comments_text = comments_text[:12000]
    issue_description = (issue_description or "")[:4000]

    prompt = _build_summary_prompt(
        sprll_number=sprll_number,
        issue_summary=issue_summary,
        issue_description=issue_description,
        comments_text=comments_text,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 800},
        )
        text = getattr(response, "text", None)
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


def generate_process_gaps(combined_descriptions: str) -> List[Dict[str, Any]]:
    model = _get_vertex_model()

    prompt = f"""
Return ONLY a JSON array of exactly 5 objects.
Each object must contain:
- number (1..5)
- title
- description

Context:
{combined_descriptions}
""".strip()

    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 1200},
        )
        text = _strip_code_fences(getattr(resp, "text", "") or "[]")
        gaps = json.loads(text)

        while len(gaps) < 5:
            gaps.append(
                {
                    "number": len(gaps) + 1,
                    "title": f"Quality Gate {len(gaps) + 1}",
                    "description": "Ensure quality checks are completed.",
                }
            )
        return gaps[:5]
    except Exception as e:
        return [
            {
                "number": i + 1,
                "title": f"Process Gap {i + 1}",
                "description": f"AI generation failed: {type(e).__name__}: {e}",
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
) -> bool:
    """
    Save issue + comments + generated summary to MongoDB.
    Uses upsert so repeated runs update the same document.
    """
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
    """Load a previously-saved issue document from MongoDB."""
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


# =========================
# Core: fetch + extract + summarise + store
# =========================
def _fetch_and_process_issue(sprll_number: str) -> Dict[str, Any]:
    """
    1. Fetch from Jira REST API
    2. Extract fields, assignee comments
    3. Generate LLM summary via Vertex AI
    4. Save everything to MongoDB
    5. Return the complete document
    """
    s = get_settings()
    sprll_number = _normalize_sprll_number(sprll_number)
    url = f"{s.jira_domain}/rest/api/2/issue/{sprll_number}"

    try:
        resp = req_lib.get(
            url,
            headers=_jira_headers(),
            params={
                "fields": "summary,description,status,assignee,comment,customfield_12801"
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

    # --- Extract fields (matching reference script exactly) ---
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

    # --- Extract assignee comments ---
    assignee_comments = _extract_assignee_comments(issue_json, assignee_name)

    # --- Missing fields check ---
    missing_fields = check_missing_fields(issue_key, fields)

    # --- LLM summary ---
    generated_summary = _summarize_with_vertex(
        sprll_number=issue_key,
        issue_summary=summary,
        issue_description=description,
        assignee_comments=assignee_comments,
    )

    # --- Save to MongoDB ---
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
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def get_or_create_issue_document(sprll_number: str) -> Dict[str, Any]:
    """
    Check MongoDB cache first. If found, return cached doc.
    Otherwise fetch from Jira, process, store, and return.
    """
    key = _normalize_sprll_number(sprll_number)

    # Try cache
    cached = _load_from_mongodb(key)
    if cached:
        cached["source"] = "mongodb_cache"
        # Remap for frontend compatibility
        return _ensure_frontend_keys(cached)

    # Fetch fresh
    doc = _fetch_and_process_issue(key)
    doc["source"] = "jira_fresh"
    return _ensure_frontend_keys(doc)


def _ensure_frontend_keys(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure the document has both backend storage keys and frontend display keys.
    The frontend uses 'sprllNumber', 'discipline', 'missingFields', 'assigneeCommentSummary'.
    MongoDB stores 'key', 'customfield_12801', 'missing_fields', 'generated_summary'.
    """
    out = dict(doc)
    # Frontend aliases
    out.setdefault("sprllNumber", out.get("key", ""))
    out.setdefault("discipline", out.get("customfield_12801", ""))
    out.setdefault("missingFields", out.get("missing_fields", []))
    out.setdefault("assigneeCommentSummary", out.get("generated_summary", ""))
    out.setdefault("matchedCommentCount", out.get("comment_count", 0))
    out.setdefault("assignee_comments", out.get("assignee_comments", []))
    out.setdefault("assignee_name", out.get("assignee_name", ""))
    return out


# =========================
# Parallel fetch
# =========================
def fetch_issues_parallel(
    sprll_numbers: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    descriptions: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(get_or_create_issue_document, n): n
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
                descriptions.append(issue["description"])

    issues.sort(key=lambda x: x.get("sprllNumber", x.get("key", "")))
    return issues, descriptions


# =========================
# Sync endpoint helper
# =========================
def sync_assignee_comments(sprll_numbers: List[str]) -> Dict[str, Any]:
    """Process all given SPRLLs, store in MongoDB, return summary."""
    collection = _get_issue_collection()
    if collection is not None:
        try:
            collection.create_index("key", unique=True)
        except PyMongoError:
            pass  # index may already exist

    stored, failed = 0, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(get_or_create_issue_document, n): n
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