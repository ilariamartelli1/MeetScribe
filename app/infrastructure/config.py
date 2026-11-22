from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "MeetScribe"
    api_prefix: str = "/api/v1"
