"""Learning progress tracking models for Quizlet-like functionality."""
from uuid import uuid4
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean, UUID, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class ChunkInteraction(Base):
    """
    Track user interactions with document chunks.
    This enables Quizlet-like progress tracking - knowing which parts 
    of a document the user has studied/been quizzed on.
    """
    __tablename__ = "chunk_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    chunk_id = Column(String, ForeignKey("document_chunks.id"), nullable=False)
    
    # Interaction type: 'chat', 'quiz', 'flashcard'
    interaction_type = Column(String, nullable=False)
    
    # Was the interaction successful? (for quiz: answered correctly, for flashcard: marked as known)
    was_successful = Column(Boolean, default=False)
    
    # Number of times this chunk was accessed in this interaction type
    interaction_count = Column(Integer, default=1)
    
    # Timestamps
    first_interaction = Column(DateTime(timezone=True), server_default=func.now())
    last_interaction = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chunk_interactions")
    chunk = relationship("DocumentChunk", back_populates="interactions")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'chunk_id', 'interaction_type', name='uix_user_chunk_interaction'),
    )


class DocumentProgress(Base):
    """
    Aggregated progress per user per document.
    Cached/computed values for quick display.
    """
    __tablename__ = "document_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Total chunks in document (cached for quick calculations)
    total_chunks = Column(Integer, default=0)
    
    # Progress metrics
    chunks_studied = Column(Integer, default=0)  # Chunks accessed via chat
    chunks_quizzed = Column(Integer, default=0)  # Chunks tested via quiz
    chunks_flashcarded = Column(Integer, default=0)  # Chunks reviewed via flashcards
    
    # Mastery metrics (chunks with successful interactions)
    chunks_mastered = Column(Integer, default=0)  # Chunks answered correctly in quiz
    
    # Calculated progress percentage (0.0 to 1.0)
    overall_progress = Column(Float, default=0.0)
    
    # Last activity
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="document_progress")
    document = relationship("Document", back_populates="user_progress")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'document_id', name='uix_user_document_progress'),
    )
