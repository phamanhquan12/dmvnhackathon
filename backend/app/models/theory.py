from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base


class TheorySession(Base):
    __tablename__ = "theory_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True) 
    score = Column(Integer)              
    total_questions = Column(Integer)    
    
    details = Column(JSON) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())