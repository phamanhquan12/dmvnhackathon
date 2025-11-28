from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON, Enum
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base
import enum

class FileTypeEnum(enum.Enum):
    PDF = "PDF"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, nullable=False)           # doc_tittle
    file_type = Column(String)                       # PDF / VIDEO
    file_path = Column(String)                       # Đường dẫn file vật lý
    num_pages = Column(Integer, default=0)           # num_page / duration (seconds)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Quan hệ 1-nhiều với Chunk
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")