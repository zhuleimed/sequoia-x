"""LLM 多维度选股分析引擎。

将策略选出的股票结合大盘趋势、外盘走势、舆论热点、财务面
等维度，通过大模型进行统一分析，输出综合推荐。
"""

import json
from datetime import date
from typing import Optional

import requests

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class MarketAnalyst:
    """LLM 驱动的多维度市场分析器。

    用法：
        analyst = MarketAnalyst(settings)
        report = analyst.analyze(strategies_results)
        # strategies_results = {"均量线突破": ["000001","600519"], ...}
    """

    def __init__(
        self,
        settings: Settings,
        api_key: Optional[str] = None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.settings = settings
        self.api_key = api_key or getattr(settings, "deepseek_api_key", "")
        self.model = model
        self.base_url = base_url

    def analyze(self, strategies_results: dict[str, list[str]]) -> str:
        """执行完整的多维度 LLM 分析。

        Args:
            strategies_results: {策略中文名: [股票代码列表], ...}

        Returns:
            LLM 返回的综合分析报告（Markdown 格式）。
        """
        # 1. 收集市场背景数据
        logger.info("MarketAnalyst: 收集市场背景数据...")
        market_context = self._gather_market_context()

        # 2. 构建 prompt
        prompt = self._build_prompt(strategies_results, market_context)

        # 3. 调用 LLM
        logger.info(f"MarketAnalyst: 调用 DeepSeek API ({self.model})...")
        report = self._call_llm(prompt)

        logger.info("MarketAnalyst: 分析完成")
        return report

    # ── 市场背景收集 ──

    def _gather_market_context(self) -> str:
        """通过 akshare 收集大盘指数和板块信息。"""
        try:
            import akshare as ak
            import pandas as pd
        except ImportError:
            return "（akshare 不可用，跳过实时数据采集）"

        lines = []
        today_str = date.today().strftime("%Y-%m-%d")

        try:
            # 大盘指数
            index_df = ak.stock_zh_index_daily_em(symbol="sh000001")
            last = index_df.iloc[-1]
            prev = index_df.iloc[-2]
            sh_change = (last["close"] - prev["close"]) / prev["close"] * 100
            lines.append(
                f"- 上证指数: {last['close']:.2f} "
                f"({'📈' if sh_change >= 0 else '📉'}) {sh_change:+.2f}%"
            )
        except Exception as e:
            lines.append(f"- 上证指数: 获取失败 ({e})")

        try:
            sz_df = ak.stock_zh_index_daily_em(symbol="sz399001")
            sz_last = sz_df.iloc[-1]
            sz_prev = sz_df.iloc[-2]
            sz_chg = (sz_last["close"] - sz_prev["close"]) / sz_prev["close"] * 100
            lines.append(
                f"- 深证成指: {sz_last['close']:.2f} "
                f"({'📈' if sz_chg >= 0 else '📉'}) {sz_chg:+.2f}%"
            )
        except Exception as e:
            lines.append(f"- 深证成指: 获取失败 ({e})")

        try:
            cy_df = ak.stock_zh_index_daily_em(symbol="sz399006")
            cy_last = cy_df.iloc[-1]
            cy_prev = cy_df.iloc[-2]
            cy_chg = (cy_last["close"] - cy_prev["close"]) / cy_prev["close"] * 100
            lines.append(
                f"- 创业板指: {cy_last['close']:.2f} "
                f"({'📈' if cy_chg >= 0 else '📉'}) {cy_chg:+.2f}%"
            )
        except Exception as e:
            lines.append(f"- 创业板指: 获取失败 ({e})")

        return "\n".join(lines)

    # ── Prompt 构建 ──

    def _build_prompt(
        self,
        strategies_results: dict[str, list[str]],
        market_context: str,
    ) -> str:
        """构建发送给 LLM 的分析 prompt。"""

        # 格式化策略选股结果
        stocks_section = []
        for strategy_name, symbols in strategies_results.items():
            if symbols:
                code_list = ", ".join(symbols)
                stocks_section.append(
                    f"▸ {strategy_name}（{len(symbols)} 只）: {code_list}"
                )
            else:
                stocks_section.append(f"▸ {strategy_name}: 无条件触发")

        stocks_text = "\n".join(stocks_section)

        prompt = f"""你是一位经验丰富的 A 股市场分析师。请对以下策略选出的候选股票进行多维度综合分析。

## 今日市场背景
{market_context}

## 候选股票（由7个量化策略从2981只基础股票池中选出）
{stocks_text}

## 分析要求
请从以下维度对每只候选股进行综合分析（不限于）：

1. **大盘与板块环境**：当前大盘趋势对个股所在板块的影响
2. **美股外盘影响**：隔夜美股三大指数及中概股走势对该股的传导
3. **舆论与热点**：该股近期是否有重大新闻、政策利好或利空
4. **基本面**：根据已知信息分析该股财务健康状况（PE、营收趋势等）
5. **技术面补充**：结合策略信号之外的技术指标进行验证
6. **风险评估**：该股当前的主要风险点

## 输出格式
请严格按照以下格式输出：

```
📊 Sequoia-X AI 综合研判 | {date.today().strftime('%Y-%m-%d')}

### 大盘环境
[对今日大盘的整体判断，1-2句话]

### 个股深度分析

**1. [股票名称 (代码)] — [策略名]**
- 评分: ★★★★☆ (X/5)
- 核心逻辑: [1句话说明为什么推荐]
- 多维分析:
  - 大盘环境: [一句话]
  - 外盘影响: [一句话]
  - 舆情热度: [一句话]
  - 基本面: [一句话]
- 风险提示: [1-2个风险点]

**2. [股票名称 (代码)] — [策略名]**
...

### 综合建议
- 最优先关注: [从所有股票中推荐1-2只]
- 操作建议: [明确的买卖建议，如"可于明早开盘逢低建仓"]
- 风险提醒: [整体市场风险提示]
```

请开始你的分析。注意：分析要简洁有力，每只股票的分析控制在 4-6 句话内。"""
        return prompt

    # ── LLM 调用 ──

    def _call_llm(self, prompt: str) -> str:
        """调用 DeepSeek API。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            logger.info(
                f"DeepSeek API 调用完成: "
                f"输入 {usage.get('prompt_tokens', '?')} tokens / "
                f"输出 {usage.get('completion_tokens', '?')} tokens"
            )
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            return (
                "⚠️ LLM 分析暂时不可用（API 调用失败），"
                "请稍后重试或检查 API Key 是否有效。"
            )
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"DeepSeek API 响应解析失败: {e}")
            return "⚠️ LLM 分析失败（响应解析异常）。"
