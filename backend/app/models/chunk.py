from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base
from app.models.document import Document

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    
    content = Column(Text, nullable=False)  
    
    
    embedding = mapped_column(Vector(768)) 
    
    #SOURCES
    page_number = Column(Integer, nullable=True) #PDF
    start_time = Column(Float, nullable=True)    #VIDEO
    end_time = Column(Float, nullable=True)      

    document = relationship("Document", back_populates="chunks")
