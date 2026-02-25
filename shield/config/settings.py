from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    base_url: str = "http://localhost"
    api_key: str = "mock"
    concurrency: int = 5

    model_config = SettingsConfigDict(env_prefix="SHIELD_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
