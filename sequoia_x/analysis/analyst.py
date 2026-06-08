"""LLM 多维度选股分析引擎。

将策略选出的股票结合实时大盘数据、外盘走势、舆论舆情、
股吧讨论、基本面等维度，通过大模型进行统一分析输出推荐。

核心思路：
  - akshare → 实时行情/财务/板块/资金流数据
  - 网页爬取 → 东方财富股吧/雪球讨论
  - LLM → 仅分析上述实时数据，不依赖模型自带知识
"""

import json
import re
import time
from datetime import date, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 股票名称缓存（避免重复请求）
_STOCK_NAME_CACHE: dict[str, str] = {}


def _get_stock_name(code: str) -> str:
    """通过 baostock 查询股票名称（带缓存）。"""
    if code in _STOCK_NAME_CACHE:
        return _STOCK_NAME_CACHE[code]
    try:
        import baostock as bs
        bs.login()
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        rs = bs.query_stock_basic(code=f"{prefix}.{code}")
        name = code
        while rs.next():
            name = rs.get_row_data()[1]
        bs.logout()
        _STOCK_NAME_CACHE[code] = name
        return name
    except Exception:
        return code


class MarketAnalyst:
    """LLM 驱动的多维度市场分析器（实时数据驱动）。"""

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
        """执行完整的多维分析。

        1. 收集实时大盘/板块/资金流数据
        2. 对每只候选股收集实时行情、股吧热帖、财务数据
        3. 将全部实时数据作为 prompt 上下文 → LLM 分析
        """
        logger.info("MarketAnalyst: 开始实时数据采集...")
        t0 = time.time()

        # 1. 收集全局市场背景
        market = self._gather_market_context()

        # 2. 收集每只股票的详细信息
        all_codes: list[str] = []
        for symbols in strategies_results.values():
            all_codes.extend(symbols)
        all_codes = list(set(all_codes))  # 去重

        stocks_detail = []
        for code in all_codes:
            logger.info(f"  → 采集 {code} ({_get_stock_name(code)})")
            detail = self._gather_stock_detail(code)
            stocks_detail.append(detail)

        t1 = time.time()
        logger.info(f"实时数据采集完成: {len(all_codes)} 只股票, 耗时 {t1-t0:.0f}秒")

        # 3. 构建 prompt（所有数据都是实时采集的）
        prompt = self._build_prompt(strategies_results, market, stocks_detail)

        # 4. 调用 LLM
        logger.info(f"MarketAnalyst: 调用 DeepSeek API ({self.model})...")
        report = self._call_llm(prompt)

        logger.info(f"MarketAnalyst: 分析完成, 总耗时 {time.time()-t0:.0f}秒")
        return report

    # ═══════════════════════════════════════════
    # 采集层 — 全实时数据，无需 LLM 自身知识
    # ═══════════════════════════════════════════

    def _gather_market_context(self) -> dict:
        """采集大盘指数、板块资金流向、隔夜外盘等实时数据。"""
        ctx = {
            "indices": [],
            "sectors": [],
            "fund_flow": [],
            "global_markets": [],
            "errors": [],
        }

        # ── 大盘指数（使用最近交易日数据） ──
        for name, symbol in [
            ("上证指数", "sh000001"),
            ("深证成指", "sz399001"),
            ("创业板指", "sz399006"),
        ]:
            try:
                import akshare as ak
                df = ak.stock_zh_index_daily_em(symbol=symbol)
                if df is not None and len(df) >= 2:
                    last = df.iloc[-1]
                    prev = df.iloc[-2]
                    chg = (last["close"] - prev["close"]) / prev["close"] * 100
                    ctx["indices"].append(
                        f"{name}: {last['close']:.2f} ({chg:+.2f}%) "
                        f"日期:{last.get('date', df.index[-1])}"
                    )
            except Exception as e:
                ctx["errors"].append(f"{name}获取失败: {e}")

        # ── 板块资金流向 ──
        try:
            import akshare as ak
            flow_df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")
            if flow_df is not None and len(flow_df) > 0:
                for _, row in flow_df.head(5).iterrows():
                    ctx["fund_flow"].append(
                        f"{row.get('名称', '?')}: "
                        f"主力净流入{row.get('主力净流入-净额', '?')}亿"
                    )
        except Exception as e:
            ctx["errors"].append(f"资金流: {e}")

        # ── 外盘 ──
        try:
            import akshare as ak
            us = ak.index_global_index(symbol="美国")
            if us is not None and not us.empty:
                for _, r in us.head(3).iterrows():
                    ctx["global_markets"].append(f"{r.get('名称','?')}: {r.get('最新价','?')} ({r.get('涨跌幅','?')})")
        except Exception:
            pass

        return ctx

    def _gather_stock_detail(self, code: str) -> dict:
        """采集单只股票的行情、财务、股吧舆情等数据。

        优先使用 akshare 实时数据，收盘后自动降级为本地 DB 数据。
        """
        detail = {
            "code": code,
            "name": _get_stock_name(code),
            "realtime": "",
            "fundamentals": "",
            "guba_posts": [],
            "news_titles": [],
            "errors": [],
        }

        # ── 实时行情（akshare），收盘后自动降级为本地 DB ──
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot["代码"] == code]
            if not row.empty:
                r = row.iloc[0]
                detail["realtime"] = (
                    f"最新价:{r.get('最新价', '?')} "
                    f"涨跌幅:{r.get('涨跌幅', '?')}% "
                    f"换手率:{r.get('换手率', '?')}% "
                    f"成交量:{r.get('成交量', '?')}手 "
                    f"成交额:{r.get('成交额', '?')}亿 "
                    f"市盈率:{r.get('市盈率-动态', '?')} "
                    f"市净率:{r.get('市净率', '?')}"
                )
        except Exception:
            pass  # 非交易时段无实时数据，切到本地 DB

        # ── 收盘后降级：从本地 SQLite 取最新日线数据 ──
        if not detail["realtime"]:
            try:
                import sqlite3
                import pandas as pd
                conn = sqlite3.connect(self.settings.db_path)
                df = pd.read_sql(
                    "SELECT date, close, volume, turnover FROM stock_daily "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                    conn, params=(code,),
                )
                conn.close()
                if not df.empty:
                    r = df.iloc[0]
                    detail["realtime"] = (
                        f"日期:{r['date']} 收盘价:{r['close']:.2f} "
                        f"成交量:{r['volume']:.0f}手 "
                        f"成交额:{r['turnover']:.2f}亿"
                    )
            except Exception as e:
                detail["errors"].append(f"本地行情: {e}")

        # ── 个股基本面（akshare） ──
        try:
            import akshare as ak
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                items = []
                for _, r in info.iterrows():
                    items.append(f"{r['item']}: {r['value']}")
                detail["fundamentals"] = " | ".join(items[:10])
        except Exception as e:
            detail["errors"].append(f"基本面: {e}")

        # ── 东方财富股吧热帖 ──
        try:
            posts = self._fetch_guba_posts(code)
            detail["guba_posts"] = posts
        except Exception as e:
            detail["errors"].append(f"股吧: {e}")

        # ── 近期新闻 ──
        try:
            import akshare as ak
            news = ak.stock_news_em(symbol=code)
            if news is not None and not news.empty:
                detail["news_titles"] = news["新闻标题"].head(5).tolist()
        except Exception as e:
            detail["errors"].append(f"新闻: {e}")

        return detail

    def _fetch_guba_posts(self, code: str, max_posts: int = 5) -> list[str]:
        """爬取东方财富股吧帖子（实时舆情）。

        使用 East Money 的 API 接口获取帖子列表。
        """
        # 东财股吧API接口
        url = (
            f"https://guba.eastmoney.com/ajax/articallist,{code},1.html"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://guba.eastmoney.com/list,{code}.html",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.encoding = "utf-8"

            posts = []
            # 尝试解析 JSON
            try:
                data = resp.json()
                if isinstance(data, dict) and "art" in data:
                    for item in data["art"][:max_posts]:
                        title = item.get("title", "").strip()
                        reads = item.get("read", "?")
                        comments = item.get("comment", "?")
                        if title:
                            posts.append(f"[阅读{reads} 评论{comments}] {title}")
            except (json.JSONDecodeError, TypeError):
                pass

            if not posts:
                posts.append("（暂无热门讨论）")
            return posts

        except Exception as e:
            logger.debug(f"股吧爬取失败 [{code}]: {e}")
            return ["（股吧数据暂不可用）"]

    # ═══════════════════════════════════════════
    # Prompt 构建 — 全部是实时数据
    # ═══════════════════════════════════════════

    def _build_prompt(
        self,
        strategies_results: dict[str, list[str]],
        market: dict,
        stocks_detail: list[dict],
    ) -> str:
        """构建 prompt，所有数据均为实时采集，LLM 只需分析无需回忆。"""

        today_str = date.today().strftime("%Y-%m-%d")

        # ── 大盘概况 ──
        market_lines = [
            "【A股主要指数】",
            *market.get("indices", []),
            "",
            "【板块资金流向（TOP5）】",
            *market.get("fund_flow", ["（数据暂缺）"]),
            "",
        ]

        # ── 候选股详情 ──
        stock_lines = []
        for sd in stocks_detail:
            stock_lines.append(
                f"\n## {sd['name']} ({sd['code']})"
            )
            stock_lines.append(f"实时行情: {sd['realtime'] or '（暂缺）'}")
            stock_lines.append(f"基本面: {sd['fundamentals'] or '（暂缺）'}")

            # 股吧舆情
            if sd.get("guba_posts"):
                stock_lines.append("--- 股吧热议 ---")
                for p in sd["guba_posts"][:3]:
                    stock_lines.append(f"  • {p}")

            # 新闻
            if sd.get("news_titles"):
                stock_lines.append("--- 近期新闻 ---")
                for n in sd["news_titles"][:3]:
                    stock_lines.append(f"  • {n}")

            # 错误
            if sd.get("errors"):
                stock_lines.append(f"⚠️ 数据采集异常: {'; '.join(sd['errors'])}")

        # ── 策略选股对应关系 ──
        strategy_lines = []
        for sname, symbols in strategies_results.items():
            if symbols:
                names = [f"{_get_stock_name(c)}({c})" for c in symbols]
                strategy_lines.append(f"▸ {sname}: {', '.join(names)}")

        # ── 完整 prompt ──
        prompt = f"""你是一位经验丰富的 A 股市场分析师。以下是 {today_str} 的实时市场数据和候选股票信息，全部是实时采集的，请基于这些数据进行分析，**不要依赖你自己的训练知识**。

## 📊 今日市场实时数据
{" ".join(market_lines)}

## 🎯 候选股票（量化策略选出，附实时数据）
{" ".join(stock_lines)}

## 🔗 策略与股票对应关系
{" ".join(strategy_lines)}

## 分析要求
请对上述候选股票进行多维度综合分析，必须严格基于上面提供的实时数据：

1. **大盘与板块环境** — 调用上面的大盘指数和资金流数据
2. **股吧舆情** — 调用上面的股吧帖子内容，分析散户情绪
3. **实时行情与基本面** — 调用上面的行情/财务数据
4. **综合研判** — 综合所有数据，给出最终推荐

## 输出格式
请严格按照以下格式输出（不要添加额外说明）：

📊 Sequoia-X AI 综合研判 | {today_str}

### 📈 大盘环境
[根据上面提供的实时指数数据和资金流，1-2句话总结]

### 🔍 个股深度分析

**1. [股票名称] ([代码]) — [对应策略名]**
- 综合评分: ⭐X.X/5
- 核心逻辑: [一句话]
- 实时行情分析: [引用上面的行情数据]
- 股吧舆情: [引用上面的帖子，判断散户情绪是多还是空]
- 基本面: [引用上面的财务数据]
- 风险提示: [1-2个]

[对每只有实时数据的候选股重复上述格式]

### 🏆 综合建议
- 最优关注: [1-2只，给出明确理由]
- 操作建议: [明确的买卖建议]
- 风险提醒: [整体风险提示]

请开始分析："""
        return prompt

    # ═══════════════════════════════════════════
    # LLM 调用层
    # ═══════════════════════════════════════════

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
            return "⚠️ LLM 分析暂时不可用（API 调用失败）"
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"DeepSeek API 响应解析失败: {e}")
            return "⚠️ LLM 分析失败（响应解析异常）"
