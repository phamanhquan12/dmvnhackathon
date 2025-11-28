import os 
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
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