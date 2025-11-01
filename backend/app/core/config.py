"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    
    # Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "math_problems"
    
    # Application Settings
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # MCP Server
    MCP_SERVER_URL: str = "http://localhost:8001"
    
    # Guardrails
    ENABLE_INPUT_GUARDRAILS: bool = True
    ENABLE_OUTPUT_GUARDRAILS: bool = True
    MAX_QUESTION_LENGTH: int = 1000
    ALLOWED_TOPICS: str = "mathematics,algebra,calculus,geometry,trigonometry,statistics,probability"
    
    # LLM Settings
    DEFAULT_MODEL: str = "gpt-4"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    
    # DSPy Settings
    DSPY_MODEL: str = "gpt-3.5-turbo"
    DSPY_TEMPERATURE: float = 0.7
    
    # Retrieval Settings
    TOP_K_RESULTS: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
