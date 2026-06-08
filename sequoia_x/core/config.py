"""配置管理模块：通过 pydantic-settings 从环境变量或 .env 文件加载系统配置。"""

import json
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_json_list(value: str | list[str]) -> list[str]:
    """将 JSON 数组字符串解析为 Python 列表。

    支持从环境变量读取的 JSON 字符串（如 '["39277"]'）或直接传入的列表。
    """
    if isinstance(value, list):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"期望 JSON 数组，获取到 {type(parsed).__name__}")
    return [str(item) for item in parsed]


class Settings(BaseSettings):
    """系统配置，从环境变量或 .env 文件加载。

    Attributes:
        db_path: SQLite 数据库路径。
        start_date: 数据回填/查询起始日期。
        wxpusher_token: WxPusher 应用的 AppToken。
        wxpusher_topic_ids: WxPusher 推送的 Topic ID 列表。
    """

    db_path: str = "data/sequoia_v2.db"
    start_date: str = "2024-01-01"
    wxpusher_token: str  # 必填字段，缺失时抛出 ValidationError
    wxpusher_topic_ids: Annotated[
        list[str],
        BeforeValidator(_parse_json_list),
    ] = Field(default=["39277"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """返回全局 Settings 单例。

    首次调用时从环境变量或 .env 文件加载配置。
    若必填字段（wxpusher_token）缺失，抛出 pydantic_core.ValidationError。

    Returns:
        Settings: 全局唯一的配置实例。

    Raises:
        pydantic_core.ValidationError: 当必填字段缺失或字段类型不匹配时抛出。
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
