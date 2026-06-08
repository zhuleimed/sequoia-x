"""WxPusher 通知属性测试。"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from sequoia_x.core.config import Settings
from sequoia_x.notify.wxpusher import WxPusherNotifier


def make_settings(token: str = "AT_test_token_123") -> Settings:
    return Settings(
        db_path="data/test.db",
        start_date="2024-01-01",
        wxpusher_token=token,
        wxpusher_topic_ids=["39277"],
    )


# Feature: sequoia-x-v2, Property 10: WxPusher 消息包含所有选股结果
@given(
    symbols=st.lists(
        st.text(min_size=6, max_size=6, alphabet="0123456789"),
        min_size=1, max_size=10, unique=True,
    )
)
@h_settings(max_examples=50, deadline=None)
def test_notification_contains_all_symbols(symbols: list[str]) -> None:
    """属性 10：send() 发出的消息应包含所有 symbol。"""
    settings = make_settings()
    notifier = WxPusherNotifier(settings)

    with patch("wxpusher.WxPusher.send_message") as mock_send:
        mock_send.return_value = {"code": 1000, "msg": "success"}
        notifier.send(symbols=symbols, strategy_name="TestStrategy")

    call_content = mock_send.call_args.kwargs.get("content", "")
    for symbol in symbols:
        assert symbol in call_content


# Feature: sequoia-x-v2, Property 11: WxPusher 使用配置中的 Token 和 Topic ID
@given(
    token=st.from_regex(r"AT_[a-zA-Z0-9]{20,40}", fullmatch=True)
)
@h_settings(max_examples=50, deadline=None)
def test_notification_uses_config_token(token: str) -> None:
    """属性 11：send() 发出的调用应使用 settings.wxpusher_token 和 wxpusher_topic_ids。"""
    settings = make_settings(token=token)
    notifier = WxPusherNotifier(settings)

    with patch("wxpusher.WxPusher.send_message") as mock_send:
        mock_send.return_value = {"code": 1000, "msg": "success"}
        notifier.send(symbols=["000001"], strategy_name="Test", webhook_key="default")

    assert mock_send.call_args.kwargs.get("token") == token
    assert mock_send.call_args.kwargs.get("topic_ids") == ["39277"]
    assert mock_send.call_args.kwargs.get("content_type") == 1


# Feature: sequoia-x-v2, Property 12: WxPusher 失败时记录 ERROR 日志
@given(fail_code=st.integers(min_value=0, max_value=999).filter(lambda x: x not in (1000,)))
@h_settings(max_examples=50, deadline=None)
def test_push_failure_logs_error(fail_code: int) -> None:
    """属性 12：WxPusher 返回非 1000 时，send() 应记录 ERROR 级别日志，不抛出异常。"""
    import sequoia_x.notify.wxpusher as wxpusher_module

    settings = make_settings()
    notifier = WxPusherNotifier(settings)

    # 直接在该模块的 logger 上挂 handler
    wxpusher_logger = logging.getLogger(wxpusher_module.__name__)
    log_records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _ListHandler(logging.ERROR)
    wxpusher_logger.addHandler(handler)
    try:
        with patch("wxpusher.WxPusher.send_message") as mock_send:
            mock_send.return_value = {"code": fail_code, "msg": "failed"}
            notifier.send(symbols=["000001"], strategy_name="Test")
    finally:
        wxpusher_logger.removeHandler(handler)

    assert any(r.levelno == logging.ERROR for r in log_records)
