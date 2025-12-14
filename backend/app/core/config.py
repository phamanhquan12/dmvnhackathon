import os 
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Try multiple .env locations (Docker vs local development)
# In Docker: env vars are passed directly by docker-compose
# Local dev: .env is at project root (one level above backend)
backend_env = Path(__file__).resolve().parent.parent.parent / ".env"  # backend/.env
root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"  # root/.env

# Load from root first (new structure), fallback to backend (old structure)
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
elif backend_env.exists():
    load_dotenv(dotenv_path=backend_env)
else:
    load_dotenv()  # Try default locations

class Settings(BaseSettings):
    PROJECT_NAME: str = "Denso Mind Backend"
    BACKEND_CORS_ORIGINS: list = ["*"]

    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")


    GOOGLE_API_KEY: str = os.getenv("API_KEY")
    # DATABASE_URL: str = os.getenv("CONNECTION_STRING")

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
settings = Settings()

print(root_env)