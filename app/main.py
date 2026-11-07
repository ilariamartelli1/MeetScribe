from fastapi import FastAPI

from app.config import Settings

settings = Settings()
app = FastAPI(title=settings.app_name)


@app.get(f"{settings.api_prefix}/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
