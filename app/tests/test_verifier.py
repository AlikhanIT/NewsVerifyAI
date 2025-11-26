import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.schemas import CheckRequest
from app.services.verifier import verify_claim


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_verify_simple(db_session):
    payload = CheckRequest(text="Company X will launch product Y tomorrow", style="simple")
    resp = await verify_claim(db=db_session, payload=payload)
    assert resp.claim == payload.text
    assert resp.style == "simple"
    assert resp.analysis.status in {"confirmed", "not_found", "uncertain"}
    assert resp.analysis.explanation
