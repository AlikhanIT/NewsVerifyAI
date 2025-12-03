import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Database (для совместимости с db.py)
    DATABASE_URL: str = "sqlite:///./newsverify.db"
    database_url: str = ""  # lowercase alias

    # OpenAI / LLM
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    LLM_TIMEOUT: float = 30.0

    # News API
    NEWSAPI_KEY: str = ""
    NEWS_API_BASE: str = "https://newsapi.org/v2"

    # App names / debug — оба варианта для совместимости
    app_name: str = "NewsVerifyAI"
    APP_NAME: str = "NewsVerifyAI"  # allow env var APP_NAME if set
    debug: bool = True
    DEBUG: bool = True  # allow env var DEBUG

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Синхронизируем атрибуты для совместимости с разными частями кода
        if not getattr(self, "database_url", None):
            setattr(self, "database_url", self.DATABASE_URL)
        # DEBUG -> debug
        setattr(self, "debug", bool(getattr(self, "DEBUG", self.debug)))
        # APP_NAME / app_name
        if not getattr(self, "app_name", None):
            setattr(self, "app_name", getattr(self, "APP_NAME", "NewsVerifyAI"))


settings = Settings()

# Вывод диагностики при загрузке
if settings.debug:
    print("\n📋 Configuration loaded:")
    print(f"   app_name: {settings.app_name}")
    print(f"   OPENAI_API_KEY: {'✅ Set' if settings.OPENAI_API_KEY else '❌ Missing'}")
    print(f"   NEWSAPI_KEY: {'✅ Set' if settings.NEWSAPI_KEY else '❌ Missing'}")
    print(f"   DATABASE_URL: {settings.DATABASE_URL}")
    print(f"   database_url (alias): {settings.database_url}")
    print(f"   LLM_TIMEOUT: {settings.LLM_TIMEOUT}s")
    print()
