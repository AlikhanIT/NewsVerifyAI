from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from .db import Base


class CachedResult(Base):
    __tablename__ = "cached_results"

    claim_hash = Column(String, primary_key=True, index=True)
    result_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
