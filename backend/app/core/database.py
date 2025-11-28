from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

class Base(DeclarativeBase):
    pass

def get_engine():
    """Create a new engine instance - avoids event loop binding issues."""
    return create_async_engine(
        settings.database_url,
        echo=True,  # Để False khi chạy production
    )

def get_async_session():
    """Create a new session maker bound to a fresh engine."""
    engine = get_engine()
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

# Keep backward compatibility with existing code
engine = get_engine()
async_session = get_async_session()

# Hàm tiện ích để lấy session (dùng cho cả Script test và App sau này)
async def get_db():
    session_maker = get_async_session()
    async with session_maker() as session:
        yield session


