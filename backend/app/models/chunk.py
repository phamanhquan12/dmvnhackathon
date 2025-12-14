from uuid import uuid4
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON, UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    document_id = Column(Integer, ForeignKey("documents.id")) # Liên kết với Document
    
    # chunk_id = Column(Integer)       # Chunk_ID (Thứ tự đoạn trong văn bản)
    content = Column(Text)              # Nội dung text
    page_number = Column(Integer)       # Trang số mấy (hoặc giây thứ mấy)

    embedding = mapped_column(Vector(768)) # Vector embedding từ Google Gemini

    document = relationship("Document", back_populates="chunks")
    
    # Learning progress tracking
    interactions = relationship("ChunkInteraction", back_populates="chunk", cascade="all, delete-orphan")
