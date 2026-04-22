from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services import (
    fetch_issues_parallel,
    generate_process_gaps,
    search_sprll_numbers,
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
    prompt_option: int = Field(default=1, ge=1, le=2)


class CommentSyncRequest(BaseModel):
    sprll_numbers: List[str] = Field(default_factory=list)
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    discipline: Optional[str] = None
    prompt_option: int = Field(default=1, ge=1, le=2)


def resolve_sprll_numbers(payload) -> List[str]:
    if payload.sprll_numbers:
        return payload.sprll_numbers
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


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    sprll_numbers = resolve_sprll_numbers(payload)
    if not sprll_numbers:
        raise HTTPException(status_code=404, detail="No SPRLL issues found.")

    prompt_option = payload.prompt_option

    issues, descriptions = fetch_issues_parallel(sprll_numbers, prompt_option=prompt_option)
    combined = (
        "\n\n---\n\n".join(descriptions)
        if descriptions
        else "No descriptions available."
    )

    return {
        "sprll_numbers": sprll_numbers,
        "issues": issues,
        "process_gaps": generate_process_gaps(combined, prompt_option=prompt_option),
        "prompt_option": prompt_option,
    }


@app.post("/api/comments/sync")
def comments_sync(payload: CommentSyncRequest):
    sprll_numbers = resolve_sprll_numbers(payload)
    if not sprll_numbers:
        raise HTTPException(status_code=404, detail="No SPRLL issues found.")
    result = sync_assignee_comments(sprll_numbers, prompt_option=payload.prompt_option)
    result["sprll_numbers"] = sprll_numbers
    return result