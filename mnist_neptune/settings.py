from pydantic import BaseSettings


class Settings(BaseSettings):
    NEPTUNE_PROJECT_NAME: str
    NEPTUNE_API_TOKEN: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
