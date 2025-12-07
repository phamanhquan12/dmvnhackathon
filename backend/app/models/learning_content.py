"""
Pre-generated learning content models.
Flashcards and Quiz sets are generated once per document and stored for tracking.
"""
from uuid import uuid4
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Flashcard(Base):
    """
    Pre-generated flashcard for a document.
    Each flashcard covers specific chunks and is generated once during ingestion.
    """
    __tablename__ = "flashcards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Content
    front = Column(Text, nullable=False)  # Question/prompt
    back = Column(Text, nullable=False)   # Answer
    
    # Which chunks this flashcard covers (for progress tracking)
    chunk_ids = Column(JSON, default=list)  # List of chunk IDs this card covers
    
    # Ordering
    order_index = Column(Integer, default=0)  # Order within the document
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="flashcards")
    user_progress = relationship("UserFlashcardProgress", back_populates="flashcard", cascade="all, delete-orphan")


class QuizSet(Base):
    """
    Pre-generated quiz set for a document.
    Each document has 3-5 quiz sets that together cover all content.
    """
    __tablename__ = "quiz_sets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Set metadata
    set_number = Column(Integer, nullable=False)  # 1, 2, 3, 4, or 5
    title = Column(String, nullable=True)  # Optional title like "Set 1: Introduction"
    
    # Which chunks this quiz set covers
    chunk_ids = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="quiz_sets")
    questions = relationship("QuizQuestion", back_populates="quiz_set", cascade="all, delete-orphan")
    user_attempts = relationship("UserQuizAttempt", back_populates="quiz_set", cascade="all, delete-orphan")


class QuizQuestion(Base):
    """Individual question within a quiz set."""
    __tablename__ = "quiz_questions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    quiz_set_id = Column(UUID(as_uuid=True), ForeignKey("quiz_sets.id", ondelete="CASCADE"), nullable=False)
    
    # Question content
    question = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)  # List of 4 options
    correct_answer = Column(String, nullable=False)  # The correct option text
    explanation = Column(Text, nullable=True)
    
    # Which chunk(s) this question is based on
    chunk_ids = Column(JSON, default=list)
    
    # Ordering
    order_index = Column(Integer, default=0)
    
    # Relationships
    quiz_set = relationship("QuizSet", back_populates="questions")


class UserFlashcardProgress(Base):
    """Track user's progress on individual flashcards."""
    __tablename__ = "user_flashcard_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    flashcard_id = Column(UUID(as_uuid=True), ForeignKey("flashcards.id", ondelete="CASCADE"), nullable=False)
    
    # Progress status
    is_completed = Column(Boolean, default=False)  # User marked as "known"
    review_count = Column(Integer, default=0)  # How many times reviewed
    
    # Spaced repetition support (optional enhancement)
    ease_factor = Column(Integer, default=2)  # 1=Hard, 2=Medium, 3=Easy
    next_review = Column(DateTime(timezone=True), nullable=True)
    
    last_reviewed = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="flashcard_progress")
    flashcard = relationship("Flashcard", back_populates="user_progress")


class UserQuizAttempt(Base):
    """Track user's attempts on quiz sets."""
    __tablename__ = "user_quiz_attempts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    quiz_set_id = Column(UUID(as_uuid=True), ForeignKey("quiz_sets.id", ondelete="CASCADE"), nullable=False)
    
    # Attempt details
    score = Column(Integer, default=0)  # Number correct
    total_questions = Column(Integer, default=0)
    is_passed = Column(Boolean, default=False)  # Score >= 60%
    
    # Detailed answers (for review)
    answers = Column(JSON, default=dict)  # {question_id: user_answer}
    
    attempted_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="quiz_attempts")
    quiz_set = relationship("QuizSet", back_populates="user_attempts")
