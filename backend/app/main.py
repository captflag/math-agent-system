"""Math Agent v2 — FastAPI application."""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import settings
from app.ratelimit import is_limited, rate_limiter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Math Agent v2",
    description="Agentic math tutor: Claude + local knowledge base + SymPy verification",
    version="2.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if is_limited(request.url.path):
        client = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded — try again in a minute."},
            )
    return await call_next(request)


app.include_router(router, prefix="/api/v1")

settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=settings.PLOTS_DIR), name="plots")

_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if _FRONTEND.exists():
        return FileResponse(_FRONTEND)
    return {"message": "Math Agent v2 API — see /api/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
