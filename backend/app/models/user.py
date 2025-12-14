from uuid import uuid4
from sqlalchemy import Column, Integer, String, UUID, Enum as SQLEnum, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum
import bcrypt

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    employee_id = Column(String, unique=True, index=True, nullable=True)  # User_ID for regular users
    full_name = Column(String)                                            # Name
    role = Column(SQLEnum(UserRole, values_callable=lambda x: [e.value for e in x]), default=UserRole.USER, nullable=False)  # Role: user or admin
    password_hash = Column(String, nullable=True)                         # Password for admin only
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Quan hệ ngược để dễ truy vấn (VD: user.theory_sessions)
    theory_sessions = relationship("TheorySession", back_populates="user", cascade="all, delete-orphan")
    practical_sessions = relationship("PracticalSession", back_populates="user", cascade="all, delete-orphan")
    
    # Learning progress tracking (old system)
    chunk_interactions = relationship("ChunkInteraction", back_populates="user", cascade="all, delete-orphan")
    document_progress = relationship("DocumentProgress", back_populates="user", cascade="all, delete-orphan")
    
    # New pre-generated content progress tracking
    flashcard_progress = relationship("UserFlashcardProgress", back_populates="user", cascade="all, delete-orphan")
    quiz_attempts = relationship("UserQuizAttempt", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """Hash and set password for admin users"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        if not self.password_hash:
            return False
        try:
            # Try bcrypt verification
            return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
        except (ValueError, TypeError):
            # If password_hash is not a valid bcrypt hash (e.g., plain text from old data)
            # Check if it matches directly (legacy support) and rehash if so
            if self.password_hash == password:
                # Rehash the password for future use
                self.set_password(password)
                return True
            return False