from pydantic import BaseSettings, SecretStr


class Settings(BaseSettings):
    NEPTUNE_PROJECT_NAME: str
    NEPTUNE_API_TOKEN: SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
