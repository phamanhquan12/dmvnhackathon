from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base
from enum import Enum

class FileTypeEnum(Enum):
    PDF = "PDF"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    file_type = mapped_column(Enum(FileTypeEnum))  # 'pdf', 'video'
    file_path = Column(String)  
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")