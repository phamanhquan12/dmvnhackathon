from uuid import uuid4
from sqlalchemy import Column, Integer, String, UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    employee_id = Column(String, unique=True, index=True) # User_ID (VD: "DENSO_001")
    full_name = Column(String)                            # Name

    # Quan hệ ngược để dễ truy vấn (VD: user.theory_sessions)
    theory_sessions = relationship("TheorySession", back_populates="user", cascade="all, delete-orphan")
    practical_sessions = relationship("PracticalSession", back_populates="user", cascade="all, delete-orphan")
    
    # Learning progress tracking (old system)
    chunk_interactions = relationship("ChunkInteraction", back_populates="user", cascade="all, delete-orphan")
    document_progress = relationship("DocumentProgress", back_populates="user", cascade="all, delete-orphan")
    
    # New pre-generated content progress tracking
    flashcard_progress = relationship("UserFlashcardProgress", back_populates="user", cascade="all, delete-orphan")
    quiz_attempts = relationship("UserQuizAttempt", back_populates="user", cascade="all, delete-orphan")