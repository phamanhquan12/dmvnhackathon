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
    """
    Practical assessment session - tracks a complete SOP video evaluation
    Enhanced for Stage B video-based SOP assessment
    """
    __tablename__ = "practical_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=lambda : str(uuid4()))
    session_code = Column(String(50), unique=True, nullable=True, index=True)  # Human-readable session ID
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # SOP Process info
    process_name = Column(String(255), nullable=True)
    sop_rules_json = Column(JSON, nullable=True)  # Store the rules used for this assessment
    
    # Assessment results
    total_steps = Column(Integer, nullable=True, default=0)
    completed_steps = Column(Integer, nullable=True, default=0)
    score = Column(Float, nullable=True)  # Điểm thao tác (percentage)
    status = Column(String)  # "PASSED" / "FAILED" / "IN_PROGRESS" / "PENDING"
    total_duration = Column(Float, nullable=True)  # Total time in seconds
    
    # Video information
    video_filename = Column(String(500), nullable=True)
    video_path = Column(Text, nullable=True)
    
    # Full report data and feedback
    report_data = Column(JSON, nullable=True)  # Stores complete step_details from SOPEngine
    feedback = Column(Text, nullable=True)  # Lỗi hoặc gợi ý cải thiện

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="practical_sessions")
    step_results = relationship("PracticalStepResult", back_populates="session", cascade="all, delete-orphan")


class PracticalStepResult(Base):
    """
    Individual step result within a practical assessment
    """
    __tablename__ = "practical_step_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("practical_sessions.id"), nullable=False)
    
    # Step identification
    step_id = Column(String(50), nullable=False)
    step_index = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    target_object = Column(String(100), nullable=True)
    
    # Step results
    status = Column(String(50), nullable=False)  # PASSED, FAILED, SKIPPED
    duration = Column(Float, nullable=True)  # Time spent on this step
    timestamp = Column(String(20), nullable=True)  # HH:MM:SS format when completed
    
    # Detection data (optional - for debugging/review)
    detection_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("PracticalSession", back_populates="step_results")