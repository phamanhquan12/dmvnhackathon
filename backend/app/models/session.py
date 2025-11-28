from uuid import uuid4
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON, UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  
from app.core.database import Base



class TheorySession(Base):
    __tablename__ = "theory_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=lambda : str(uuid4()))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id")) # Liên kết với bảng User
    
    score = Column(Float)        # Điểm số (0-10 hoặc 0-100)
    status = Column(String)      # "PASSED" / "FAILED"
    
    # Quan trọng: Lưu chi tiết bài thi (Câu hỏi, Đáp án chọn, Đúng/Sai)
    # Để sau này còn hiện lại cho user xem họ sai ở đâu
    details = Column(JSON, nullable=True) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="theory_sessions")



class PracticalSession(Base):
    __tablename__ = "practical_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=lambda : str(uuid4()))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # score = Column(Float, nullable=True) # Điểm thao tác (nếu có)
    status = Column(String)              # "PASSED" / "FAILED" / "IN_PROGRESS"
    
    # Feedback: Lưu chuỗi JSON hoặc Text mô tả lỗi
    # VD: "Sai bước 2 (Dùng sai kìm). Ôn lại mục An toàn điện."
    feedback = Column(Text, nullable=True) 
    
    # (Tùy chọn) Đường dẫn video quay lại buổi thực hành đó
    # video_record_path = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="practical_sessions")