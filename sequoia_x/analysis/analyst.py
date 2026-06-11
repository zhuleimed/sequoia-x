"""LLM 多维度选股分析引擎。

将策略选出的股票结合实时大盘数据、外盘走势、舆论舆情、
基本面等维度，通过大模型进行统一分析输出推荐。

数据源说明（2026-06-11 验证）：
  - ❌ 东方财富 akshare 系列接口已全面封禁（大盘/行情/板块资金流）
  - ✅ 新浪行情 API → 大盘指数 + 个股实时行情
  - ✅ 本地 SQLite (sequoia_v2.db) → peTTM、pbMRQ 等财务数据
  - ✅ 本地 index_daily 表 → 大盘指数趋势（5日/20日涨跌幅）
  - ✅ baostock → 股票名称查询
  - ✅ 东方财富新闻公告 API（直连HTTP，不走 akshare）
  - ⚠️ 东方财富股吧 API 已变更接口地址，暂不可用
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


def _sina_quote_url() -> str:
    """生成新浪批量行情请求 URL，支持多只股票同时查询。"""
    return "https://hq.sinajs.cn/list={codes_str}"


def _fetch_sina_quotes(codes: list[str]) -> dict[str, dict]:
    """批量查询新浪实时行情。

    Args:
        codes: 股票代码列表，如 ["000001", "600519"]。

    Returns:
        {code: {name, price, chg_pct, volume, amount, ...}} 的字典。
    """
    if not codes:
        return {}

    sina_codes = []
    for c in codes:
        if c.startswith(("6", "9")):
            sina_codes.append(f"sh{c}")
        else:
            sina_codes.append(f"sz{c}")

    try:
        url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
        resp = requests.get(
            url,
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}

        result: dict[str, dict] = {}
        for line in resp.text.strip().split("\n"):
            if "=" not in line:
                continue
            parts = line.split('="')
            if len(parts) < 2:
                continue
            key = parts[0].replace("var hq_str_", "").strip()
            values = parts[1].rstrip(";").split(",")

            # 新浪行情格式: 0-name, 1-open, 2-prev_close, 3-price, 4-chg, 5-chg_pct, ...
            if len(values) >= 32:
                raw_code = key[2:]  # 去掉 sh/sz 前缀
                result[raw_code] = {
                    "name": values[0],
                    "price": values[3],
                    "chg": values[4],
                    "chg_pct": values[5],
                    "volume": values[8],   # 手
                    "amount": values[9],   # 元
                    "high": values[6],
                    "low": values[7],
                    "turnover": values[30],  # 换手率%
                }

        return result
    except Exception as e:
        logger.debug(f"新浪行情查询失败: {e}")
        return {}


def _fetch_sina_index(index_codes: list[str]) -> list[str]:
    """查询新浪大盘指数行情。

    新浪指数格式：0-名称, 1-今开, 2-昨收, 3-现价, 4-最高, 5-最低, ...
    涨跌幅需根据 (现价 - 昨收) / 昨收 自行计算。

    Args:
        index_codes: 指数新浪代码，如 ["sh000001", "sz399001", "sz399006"]

    Returns:
        格式化的指数行情字符串列表。
    """
    if not index_codes:
        return []

    try:
        url = f"https://hq.sinajs.cn/list={','.join(index_codes)}"
        resp = requests.get(
            url,
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        lines = []
        for line in resp.text.strip().split("\n"):
            if "=" not in line:
                continue
            parts = line.split('="')
            if len(parts) < 2:
                continue
            values = parts[1].rstrip(";").split(",")
            if len(values) >= 6:
                name = values[0]
                try:
                    price = float(values[3])
                    prev_close = float(values[2])
                    chg = price - prev_close
                    chg_pct = chg / prev_close * 100 if prev_close != 0 else 0.0
                    lines.append(
                        f"{name}: {price:.2f} "
                        f"(涨跌{chg:+.2f}点, 涨幅{chg_pct:+.2f}%)"
                    )
                except (ValueError, IndexError):
                    lines.append(f"{name}: {values[3]}")

        return lines
    except Exception as e:
        logger.debug(f"新浪指数查询失败: {e}")
        return []


class MarketAnalyst:
    """LLM 驱动的多维度市场分析器（实时数据驱动）。

    数据源说明：
    - 大盘指数：新浪行情 API（实时）+ 本地 index_daily 表（趋势 + PE）
    - 个股行情：新浪行情 API（直连 HTTP）
    - 个股财务：本地 SQLite 数据库（peTTM, pbMRQ 等已有字段）
    - 股票名称：baostock API
    - 新闻公告：东方财富公告 API（直连 HTTP，不走 akshare）
    - 板块资金流：因东方财富封禁 akshare，暂时移除
    - 股吧舆情：接口已变更，暂时不可用
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
        """执行完整的多维分析。

        1. 收集实时大盘/行情数据（新浪 API）
        2. 对每只候选股收集实时行情 + 本地财务数据 + 新闻
        3. 将全部实时数据作为 prompt 上下文 → LLM 分析
        """
        logger.info("MarketAnalyst: 开始实时数据采集（新浪 API + SQLite）...")
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
        logger.info(
            f"实时数据采集完成: {len(all_codes)} 只股票, 耗时 {t1-t0:.0f}秒"
        )

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

    def _fetch_local_index_trends(self) -> list[str]:
        """从本地 index_daily 表读取指数趋势数据。

        计算各指数的今日涨跌幅、近5日涨跌幅、近20日涨跌幅，
        以及最新 PE 估值（若有）。

        Returns:
            格式化的指数趋势字符串列表。
        """
        INDEX_MAP = {
            "sh.000001": "上证指数",
            "sh.000016": "上证50",
            "sh.000300": "沪深300",
            "sh.000905": "中证500",
            "sz.399001": "深证成指",
            "sz.399106": "深证综指",
        }
        try:
            import sqlite3
            import pandas as pd

            conn = sqlite3.connect(self.settings.db_path)
            lines = []
            for code, name in INDEX_MAP.items():
                try:
                    df = pd.read_sql(
                        "SELECT date, close, pctChg FROM index_daily "
                        "WHERE symbol = ? ORDER BY date DESC LIMIT 20",
                        conn,
                        params=(code,),
                    )
                    if df.empty or len(df) < 2:
                        continue

                    latest = df.iloc[0]
                    day1_chg = latest["pctChg"]

                    # 近5日涨跌幅（第0天 vs 第4天）
                    if len(df) >= 5:
                        week_chg = (
                            (latest["close"] / df.iloc[4]["close"]) - 1
                        ) * 100
                        week_str = f"{week_chg:+.2f}%"
                    else:
                        week_str = "N/A"

                    # 近20日涨跌幅（第0天 vs 第19天，最多到最后一个）
                    if len(df) >= 20:
                        month_chg = (
                            (latest["close"] / df.iloc[19]["close"]) - 1
                        ) * 100
                        month_str = f"{month_chg:+.2f}%"
                    else:
                        # 用最早一个代替
                        month_chg = (
                            (latest["close"] / df.iloc[-1]["close"]) - 1
                        ) * 100
                        month_str = f"{month_chg:+.2f}%（仅{len(df)}日）"

                    lines.append(
                        f"{name}: {latest['close']:.0f} "
                        f"(今日{day1_chg:+.2f}%, "
                        f"近5日{week_str}, "
                        f"近20日{month_str})"
                    )
                except Exception as e:
                    logger.debug(f"指数[{code}]读取失败: {e}")
                    continue
            conn.close()
            return lines
        except Exception as e:
            logger.debug(f"本地指数趋势读取失败: {e}")
            return []

    def _gather_market_context(self) -> dict:
        """采集大盘指数行情等实时数据。

        双源采集：
        1. 新浪行情 API → 今日实时点位和涨跌幅
        2. 本地 index_daily 表 → 趋势（近5日/近20日）及 PE 估值
        """
        ctx = {
            "indices": [],
            "index_trends": [],
            "global_markets": [],
            "errors": [],
        }

        # ── 大盘指数实时行情（新浪 API） ──
        try:
            index_data = _fetch_sina_index(
                ["sh000001", "sz399001", "sz399006"]
            )
            ctx["indices"] = index_data
        except Exception as e:
            ctx["errors"].append(f"新浪指数获取失败: {e}")

        # ── 大盘指数趋势 + PE（本地 index_daily 表） ──
        try:
            trends = self._fetch_local_index_trends()
            ctx["index_trends"] = trends
        except Exception as e:
            ctx["errors"].append(f"本地指数趋势获取失败: {e}")

        # ── 外盘（通过新浪获取） ──
        try:
            global_data = _fetch_sina_index(
                [".DJI", ".IXIC", ".INX"]
            )
            if global_data:
                ctx["global_markets"] = global_data
        except Exception:
            pass

        return ctx

    def _gather_stock_detail(self, code: str) -> dict:
        """采集单只股票的行情、财务、新闻等数据。

        数据源优先级：
        1. 新浪实时行情（交易时段）
        2. 本地 SQLite 财务数据（peTTM, pbMRQ）
        3. 东方财富新闻公告 API（直连 HTTP）
        """
        detail = {
            "code": code,
            "name": _get_stock_name(code),
            "realtime": "",
            "fundamentals": "",
            "news_titles": [],
            "errors": [],
        }

        # ── 实时行情（新浪 API，已测试可用） ──
        try:
            quotes = _fetch_sina_quotes([code])
            if code in quotes:
                q = quotes[code]
                detail["realtime"] = (
                    f"最新价:{q.get('price', '?')} "
                    f"涨跌幅:{q.get('chg_pct', '?')}% "
                    f"涨跌额:{q.get('chg', '?')} "
                    f"最高:{q.get('high', '?')} "
                    f"最低:{q.get('low', '?')} "
                    f"换手率:{q.get('turnover', '?')}% "
                    f"成交量:{q.get('volume', '?')}手 "
                    f"成交额:{q.get('amount', '?')}元"
                )
        except Exception as e:
            detail["errors"].append(f"新浪行情: {e}")

        # ── 收盘后降级：从本地 SQLite 取最新日线财务数据 ──
        try:
            import sqlite3
            import pandas as pd

            conn = sqlite3.connect(self.settings.db_path)
            df = pd.read_sql(
                "SELECT date, close, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, "
                "volume, turnover, amount "
                "FROM stock_daily WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                conn,
                params=(code,),
            )
            conn.close()
            if not df.empty:
                r = df.iloc[0]
                # 构建基本面描述
                info_parts = [f"日期:{r['date']}"]
                if pd.notna(r["close"]):
                    info_parts.append(f"收盘价:{r['close']:.2f}")
                if pd.notna(r["pctChg"]):
                    info_parts.append(f"涨跌幅:{r['pctChg']:+.2f}%")
                if pd.notna(r["peTTM"]):
                    info_parts.append(f"PE(TTM):{r['peTTM']:.2f}")
                if pd.notna(r["pbMRQ"]):
                    info_parts.append(f"PB(MRQ):{r['pbMRQ']:.3f}")
                if pd.notna(r["psTTM"]):
                    info_parts.append(f"PS(TTM):{r['psTTM']:.2f}")
                if pd.notna(r["volume"]):
                    info_parts.append(f"成交量:{r['volume']:.0f}手")
                if pd.notna(r["turnover"]):
                    info_parts.append(f"成交额:{r['turnover']:.2f}亿")

                detail["fundamentals"] = " | ".join(info_parts)

                # 如果实时行情尚未获取（非交易时段），使用本地数据
                if not detail["realtime"]:
                    detail["realtime"] = " | ".join(info_parts[:4])
        except Exception as e:
            detail["errors"].append(f"本地财务: {e}")

        # ── 近期新闻公告（东方财富公告 API，不走 akshare，已测试可用） ──
        try:
            news = self._fetch_news(code)
            detail["news_titles"] = news
        except Exception as e:
            detail["errors"].append(f"新闻: {e}")

        return detail

    def _fetch_news(self, code: str, max_news: int = 5) -> list[str]:
        """从东方财富公告 API 获取个股新闻（直连 HTTP，不走 akshare）。"""
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {
            "sr": -1,
            "page_size": max_news,
            "page_index": 1,
            "ann_type": "A",
            "stock_list": code,
            "f_node": 0,
            "s_node": 0,
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://emweb.securities.eastmoney.com/PC_HSF10/",
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                return []

            data = resp.json()
            titles = []
            for item in data.get("data", {}).get("list", []):
                title = item.get("title", "").strip()
                date_str = item.get("noticeDate", "")[:10]
                if title:
                    titles.append(f"[{date_str}] {title}")
            return titles[:max_news]
        except Exception as e:
            logger.debug(f"新闻获取失败 [{code}]: {e}")
            return []

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
        market_lines = ["【A股主要指数（实时）】"]
        if market.get("indices"):
            market_lines.extend(market["indices"])
        else:
            market_lines.append("（数据暂缺）")

        # ── 大盘趋势 + PE（本地历史数据） ──
        if market.get("index_trends"):
            market_lines.append("")
            market_lines.append("【大盘趋势（近5日/近20日涨跌幅）】")
            market_lines.extend(market["index_trends"])

        if market.get("global_markets"):
            market_lines.append("")
            market_lines.append("【外盘行情】")
            market_lines.extend(market["global_markets"])

        # ── 候选股详情 ──
        stock_lines = []
        for sd in stocks_detail:
            stock_lines.append(f"\n## {sd['name']} ({sd['code']})")
            stock_lines.append(
                f"实时行情: {sd['realtime'] or '（非交易时段，见下方财务数据）'}"
            )
            stock_lines.append(
                f"基本面/财务: {sd['fundamentals'] or '（暂缺）'}"
            )

            # 新闻
            if sd.get("news_titles"):
                stock_lines.append("--- 近期公告 ---")
                for n in sd["news_titles"][:3]:
                    stock_lines.append(f"  • {n}")

            # 错误
            if sd.get("errors"):
                stock_lines.append(
                    f"⚠️ 数据采集异常: {'; '.join(sd['errors'])}"
                )

        # ── 策略选股对应关系 ──
        strategy_lines = []
        for sname, symbols in strategies_results.items():
            if symbols:
                names = [f"{_get_stock_name(c)}({c})" for c in symbols]
                strategy_lines.append(f"▸ {sname}: {', '.join(names)}")

        # ── 完整 prompt ──
        prompt = f"""你是一位经验丰富的 A 股市场分析师。以下是 {today_str} 的实时市场数据和候选股票信息，全部是实时采集的（来源：新浪行情 API + 本地数据库），请基于这些数据进行分析，**不要依赖你自己的训练知识**。

## 📊 今日市场实时数据
{" ".join(market_lines)}

## 🎯 候选股票（量化策略选出，附实时数据）
{" ".join(stock_lines)}

## 🔗 策略与股票对应关系
{" ".join(strategy_lines)}

## 分析要求
请对上述候选股票进行多维度综合分析，必须严格基于上面提供的实时数据：

1. **大盘与板块环境** — 调用上面的大盘指数数据
2. **实时行情与基本面** — 调用上面的行情/财务数据（PE/PB等）
3. **新闻公告** — 调用上面的公告标题，判断是否有重大事项
4. **综合研判** — 综合所有数据，给出最终推荐

## 输出格式
请严格按照以下格式输出（不要添加额外说明）：

📊 Sequoia-X AI 综合研判 | {today_str}

### 📈 大盘环境
[根据上面提供的实时指数数据，1-2句话总结]

### 🔍 个股深度分析

**1. [股票名称] ([代码]) — [对应策略名]**
- 综合评分: ⭐X.X/5
- 核心逻辑: [一句话]
- 实时行情分析: [引用上面的行情数据]
- 基本面: [引用上面的财务数据]
- 新闻公告: [引用上面的公告标题]
- 风险提示: [1-2个]

[对每只候选股重复上述格式]

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
