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
    google_application_credentials: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()