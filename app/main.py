from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from .config import settings
from .db import Base, engine, get_db
from .models import CachedResult
from .schemas import CheckRequest, CheckResponse
from .services.verifier import verify_claim
from sqlalchemy.orm import Session

# Создаём таблицы
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name, debug=settings.debug)

# CORS (чтобы фронт открывался без ошибок)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем шаблоны
templates = Jinja2Templates(directory="app/templates")


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name}


# UI – простая HTML морда
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Основной эндпоинт проверки утверждения
@app.post("/check", response_model=CheckResponse)
async def check_claim(
    payload: CheckRequest,
    db: Session = Depends(get_db),
):
    if not payload.text or len(payload.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Текст утверждения слишком короткий.")

    resp = await verify_claim(db=db, payload=payload)
    return resp


# Возможность запуска напрямую (но лучше через uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
