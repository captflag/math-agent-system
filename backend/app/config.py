"""Application configuration."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Anthropic. Key may also come from the ambient environment
    # (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile),
    # so an empty value here does not necessarily mean "no credentials".
    ANTHROPIC_API_KEY: str = ""
    MODEL: str = "claude-opus-5"
    MAX_TOKENS: int = 8000
    EFFORT: str = "high"          # low | medium | high | xhigh | max
    MAX_TOOL_TURNS: int = 12
    MAX_WEB_SEARCHES: int = 3

    # Self-correction & cascade
    SELF_CORRECT: bool = True         # retry once when the answer isn't machine-verified
    ENABLE_CASCADE: bool = False      # solve with CASCADE_MODEL first, escalate on failure
    CASCADE_MODEL: str = "claude-haiku-4-5"

    # Guardrails
    ENABLE_INPUT_GUARDRAILS: bool = True
    MAX_QUESTION_LENGTH: int = 2000
    RATE_LIMIT_PER_MINUTE: int = 30

    # Artifacts
    PLOTS_DIR: Path = BASE_DIR / "data" / "plots"
    CALIBRATION_PATH: Path = BASE_DIR / "data" / "calibration.json"

    # Tutor sessions
    TUTOR_SESSION_TTL: int = 3600     # seconds

    # Knowledge base
    KB_PATH: Path = BASE_DIR / "data" / "math_kb.json"
    KB_TOP_K: int = 3

    # Feedback
    FEEDBACK_DB: Path = BASE_DIR / "data" / "feedback.sqlite3"
    FEW_SHOT_LIMIT: int = 2       # top-rated solutions injected as exemplars

    # Server
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"


settings = Settings()
