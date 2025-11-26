import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NewsVerifyAI"
    debug: bool = True

    # внешние API
    newsapi_key: str | None = None  # NEWSAPI_KEY
    openai_api_key: str | None = None  # OPENAI_API_KEY

    # база
    database_url: str = "sqlite:///./newsverifyai.db"

    class Config:
        env_file = ".env"


settings = Settings()
