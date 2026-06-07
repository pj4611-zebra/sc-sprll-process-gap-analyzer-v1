from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services import (
    analysis_signature,
    delete_phase_gaps_by_run_id,
    fetch_discipline_product_map,
    fetch_issues_parallel,
    find_repeated_gaps_in_phase,
    generate_process_gaps,
    get_gap_dimensions,
    get_gap_insights,
    load_cached_analysis,
    save_cached_analysis,
    save_gaps_to_phase_collections,
    search_issues,
    search_sprll_numbers,
    search_sprll_numbers_by_jql,
    sync_assignee_comments,
)

app = FastAPI(title="SPRLL Process Gap Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    sprll_numbers: List[str] = Field(default_factory=list)
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    discipline: Optional[str] = None
    products: List[str] = Field(default_factory=list)
    custom_jql: Optional[str] = None
    prompt_option: int = Field(default=1, ge=1, le=2)
    force_refresh: bool = False


class CommentSyncRequest(BaseModel):
    sprll_numbers: List[str] = Field(default_factory=list)
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    discipline: Optional[str] = None
    prompt_option: int = Field(default=1, ge=1, le=2)


def resolve_sprll_numbers(payload) -> List[str]:
    if payload.sprll_numbers:
        return payload.sprll_numbers
    if getattr(payload, "custom_jql", None):
        return search_sprll_numbers_by_jql(payload.custom_jql)
    if payload.from_date and payload.to_date and payload.discipline:
        return search_sprll_numbers(
            payload.from_date, payload.to_date, payload.discipline
        )
    raise HTTPException(
        status_code=400,
        detail="Provide sprll_numbers or from_date + to_date + discipline.",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/discipline-products")
def get_discipline_products():
    """Return a mapping of {discipline: [products]} built live from Jira."""
    try:
        mapping = fetch_discipline_product_map()
        return {"discipline_products": mapping}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    sprll_numbers = resolve_sprll_numbers(payload)
    if not sprll_numbers:
        raise HTTPException(status_code=404, detail="No SPRLL issues found.")

    # Always use prompt option 1
    prompt_option = 1

    # ── Cache lookup keyed by the normalized, resolved query ──
    query_meta = {
        "sprll_numbers": sprll_numbers,
        "discipline": payload.discipline,
        "products": payload.products,
        "from_date": payload.from_date,
        "to_date": payload.to_date,
        "custom_jql": payload.custom_jql,
        "prompt_option": prompt_option,
    }
    signature = analysis_signature(query_meta)
    cached = load_cached_analysis(signature)

    if cached and not payload.force_refresh:
        response = dict(cached.get("response") or {})
        response["cached"] = True
        return response

    # On force re-run, drop the previous run's persisted gaps to avoid duplicate
    # recurrence counts in the Database Insights view.
    if cached and payload.force_refresh:
        prev_run_id = (
            (cached.get("response") or {}).get("persistence") or {}
        ).get("analysis_run_id")
        if prev_run_id:
            delete_phase_gaps_by_run_id(prev_run_id)

    issues, descriptions = fetch_issues_parallel(sprll_numbers, prompt_option=prompt_option)
    combined = (
        "\n\n---\n\n".join(descriptions)
        if descriptions
        else "No descriptions available."
    )

    process_gaps = generate_process_gaps(combined, prompt_option=prompt_option)

    # Derive disciplines/products from the analyzed issues (works for all input
    # modes); fall back to the explicit payload values.
    disciplines = {
        i.get("discipline") or i.get("customfield_12801")
        for i in issues
        if (i.get("discipline") or i.get("customfield_12801"))
    }
    if payload.discipline:
        disciplines.add(payload.discipline)
    products = {
        i.get("product") or i.get("customfield_21800")
        for i in issues
        if (i.get("product") or i.get("customfield_21800"))
    }
    products.update(payload.products or [])

    persistence = save_gaps_to_phase_collections(
        process_gaps,
        source_sprll_keys=sprll_numbers,
        sprll_date_range={"from": payload.from_date, "to": payload.to_date},
        disciplines=sorted(d for d in disciplines if d),
        products=sorted(p for p in products if p),
    )

    response = {
        "sprll_numbers": sprll_numbers,
        "issues": issues,
        "process_gaps": process_gaps,
        "prompt_option": prompt_option,
        "persistence": persistence,
        "cached": False,
    }

    # Persist the full result so an identical query is served from the DB next time.
    save_cached_analysis(signature, query_meta, response)
    return response


class RepeatedGapsRequest(BaseModel):
    lifecycle_phase: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    similarity_threshold: Optional[float] = None
    min_cluster_size: int = Field(default=2, ge=2)


@app.post("/api/repeated-gaps")
def repeated_gaps(payload: RepeatedGapsRequest):
    """Return clusters of semantically-similar gaps in a phase within a date window."""
    try:
        clusters = find_repeated_gaps_in_phase(
            lifecycle_phase=payload.lifecycle_phase,
            from_date=payload.from_date,
            to_date=payload.to_date,
            similarity_threshold=payload.similarity_threshold,
            min_cluster_size=payload.min_cluster_size,
        )
        return {
            "lifecycle_phase": payload.lifecycle_phase,
            "from_date": payload.from_date,
            "to_date": payload.to_date,
            "cluster_count": len(clusters),
            "clusters": clusters,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GapInsightsRequest(BaseModel):
    dimension: str = Field(default="discipline")
    value: Optional[str] = None
    lifecycle_phase: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    similarity_threshold: Optional[float] = None
    min_cluster_size: int = Field(default=1, ge=1)
    top_n: int = Field(default=25, ge=1, le=200)
    keyword: Optional[str] = None


@app.post("/api/gap-insights")
def gap_insights(payload: GapInsightsRequest):
    """Most-repeated QA action items, sliced by discipline / product / lifecycle phase."""
    if payload.dimension not in {"discipline", "product", "lifecycle_phase"}:
        raise HTTPException(
            status_code=400,
            detail="dimension must be one of: discipline, product, lifecycle_phase.",
        )
    try:
        clusters = get_gap_insights(
            dimension=payload.dimension,
            value=payload.value,
            from_date=payload.from_date,
            to_date=payload.to_date,
            similarity_threshold=payload.similarity_threshold,
            min_cluster_size=payload.min_cluster_size,
            top_n=payload.top_n,
            lifecycle_phase=payload.lifecycle_phase,
            keyword=payload.keyword,
        )
        issues = []
        if payload.keyword and payload.keyword.strip():
            issues = search_issues(
                payload.keyword,
                discipline=payload.value if payload.dimension == "discipline" else None,
                product=payload.value if payload.dimension == "product" else None,
            )
        return {
            "dimension": payload.dimension,
            "value": payload.value,
            "cluster_count": len(clusters),
            "clusters": clusters,
            "issue_count": len(issues),
            "issues": issues,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gap-dimensions")
def gap_dimensions():
    """Distinct disciplines / products / phases available in the gap collections."""
    try:
        return get_gap_dimensions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/comments/sync")
def comments_sync(payload: CommentSyncRequest):
    sprll_numbers = resolve_sprll_numbers(payload)
    if not sprll_numbers:
        raise HTTPException(status_code=404, detail="No SPRLL issues found.")
    result = sync_assignee_comments(sprll_numbers, prompt_option=1)
    result["sprll_numbers"] = sprll_numbers
    return result