from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="POLYMARKET_OVERVIEW_")

    db_path: str = "data/polymarket-overview.sqlite"
    host: str = "127.0.0.1"
    port: int = 8787
    ai_provider: str = "deepseek"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    github_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
