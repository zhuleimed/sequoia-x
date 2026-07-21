"""WxPusher 通知模块：将选股结果通过 WxPusher 推送至微信。"""

from datetime import date

from wxpusher import WxPusher

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class WxPusherNotifier:
    """WxPusher 微信推送器。

    将选股结果格式化为纯文本消息并通过 WxPusher 推送到微信。
    """

    def __init__(self, settings: Settings) -> None:
        """
        初始化 WxPusherNotifier。

        Args:
            settings: Settings 实例，提供 WxPusher Token 和 Topic ID 配置。
        """
        self.settings = settings

    @staticmethod
    def _to_xueqiu_code(code: str) -> str:
        """将纯数字代码转为雪球格式：6开头→SH，4/8开头→BJ，其余→SZ。"""
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @staticmethod
    def _get_stock_names(symbols: list[str]) -> dict[str, str]:
        """批量获取股票名称（本地 SQLite 优先，腾讯 API 回退）。"""
        from sequoia_x.core.config import get_settings
        from sequoia_x.data.engine import DataEngine
        return DataEngine(get_settings()).get_stock_names_batch(symbols)

    def _build_message(self, symbols: list[str], strategy_name: str) -> str:
        """构建纯文本推送消息。

        Args:
            symbols: 选股结果代码列表。
            strategy_name: 策略名称。

        Returns:
            格式化的纯文本消息字符串。
        """
        today = date.today().strftime("%Y-%m-%d")
        names = self._get_stock_names(symbols)

        lines: list[str] = []
        lines.append(f"📈 Sequoia-X 选股播报 | {strategy_name}")
        lines.append(f"日期: {today}")
        lines.append(f"选股数量: {len(symbols)}")
        lines.append("")
        lines.append("选股列表:")
        for code in symbols:
            name = names.get(code, code)
            lines.append(f"- {name} ({code})")

        return "\n".join(lines)

    def send(
        self,
        symbols: list[str],
        strategy_name: str,
        webhook_key: str = "default",
    ) -> None:
        """
        将选股结果格式化为纯文本消息并通过 WxPusher 推送。

        WxPusher 根据 topic_ids 路由消息，webhook_key 保留仅用于保持接口兼容性。

        Args:
            symbols: 选股结果代码列表。
            strategy_name: 策略名称，用于消息标题。
            webhook_key: 保留参数，未使用（WxPusher 按 topic 路由）。
        """
        message = self._build_message(symbols, strategy_name)

        try:
            result = WxPusher.send_message(
                content=message,
                token=self.settings.wxpusher_token,
                topic_ids=self.settings.wxpusher_topic_ids,
                content_type=1,  # 1=纯文本, 2=HTML
            )

            # WxPusher 成功响应: {"code": 1000, "msg": "success", ...}
            if result.get("code") == 1000:
                logger.info(
                    f"WxPusher 推送成功 [{strategy_name}]，共 {len(symbols)} 只股票"
                )
            else:
                logger.error(
                    f"WxPusher 推送失败 [{strategy_name}] "
                    f"响应={result}"
                )

        except Exception as exc:
            logger.error(f"WxPusher 推送请求异常 [{strategy_name}]：{exc}")
