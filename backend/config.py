from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Jira + Mongo
    jira_domain: str = "https://jira.zebra.com"
    jira_token: str
    mongodb_uri: str
    mongodb_db_name: str = "sprll_analysis"

    # Vertex AI
    gcp_project: str
    gcp_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "text-embedding-004"
    google_application_credentials: str

    # Phase-gap recurrence detection
    similarity_threshold_default: float = 0.85

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Map each LLM-emitted lifecycle_phase value to its dedicated MongoDB collection.
PHASE_TO_COLLECTION: dict[str, str] = {
    "Coding Phase": "gaps_coding_phase",
    "Test Phase": "gaps_test_phase",
    "Requirement Phase": "gaps_requirement_phase",
    "Design Review Phase": "gaps_design_review_phase",
    "Deployment Phase": "gaps_deployment_phase",
    "Documentation Phase": "gaps_documentation_phase",
}