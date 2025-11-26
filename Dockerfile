# ---------- 1. Базовый образ ----------
FROM python:3.10-slim AS base

# ---------- 2. Рабочая директория ----------
WORKDIR /app

# ---------- 3. Установка системных зависимостей ----------
# Нужны для spaCy и прочих пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- 4. Установка зависимостей ----------
# Сначала только requirements — для кеширования Docker-слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 5. Установка spaCy модели (если надо) ----------
# Если не хочешь — можешь закомментировать
RUN python -m spacy download en_core_web_sm || true

# ---------- 6. Копируем проект ----------
COPY . .

# ---------- 7. Порт ----------
EXPOSE 8000

# ---------- 8. Переменные окружения ----------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------- 9. Запуск приложения ----------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
