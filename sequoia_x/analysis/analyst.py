"""LLM 多维度选股分析引擎。

将策略选出的股票结合实时大盘数据、外盘走势、舆论舆情、
基本面等维度，通过大模型进行统一分析输出推荐。

数据源说明（2026-06-11 验证）：
  - ❌ 东方财富 akshare 系列接口已全面封禁（大盘/行情/板块资金流）
  - ✅ 知兔 API (zhituapi.com) → 个股实时行情含 PE/PB/市值/60日涨跌幅
  - ✅ 新浪行情 API → 大盘指数实时行情
  - ✅ 本地 index_daily 表 → 大盘指数趋势（5日/20日涨跌幅）
  - ✅ 本地 SQLite (sequoia_v2.db) → peTTM、pbMRQ 等财务数据（兜底）
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

from sequoia_x.core.config import get_settings, Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 股票名称缓存（避免重复请求）
_STOCK_NAME_CACHE: dict[str, str] = {}

# 知兔 API 配置
_ZHITU_BASE = "https://api.zhituapi.com"
_ZHITU_TOKEN = ""  # 从 .env 加载


def _get_zhitu_token() -> str:
    """获取知兔 API token，优先从 .env 读取。"""
    global _ZHITU_TOKEN
    if not _ZHITU_TOKEN:
        try:
            settings = get_settings()
            _ZHITU_TOKEN = getattr(settings, "zhitu_token", "")
        except Exception:
            _ZHITU_TOKEN = ""
    return _ZHITU_TOKEN


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


def _fetch_zhitu_quotes(codes: list[str]) -> dict[str, dict]:
    """批量查询知兔 API 个股实时行情（含 PE/PB/市值等）。

    注意：知兔免费版不支持真正的批量查询，需逐只查询。

    Returns:
        {code: {price, chg_pct, turnover, pe, pb, market_cap, ...}} 的字典。
    """
    token = _get_zhitu_token()
    if not token:
        return {}

    result: dict[str, dict] = {}
    for code in codes:
        try:
            url = f"{_ZHITU_BASE}/hs/real/ssjy/{code}?token={token}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            d = resp.json()
            if not d or "p" not in d:
                continue

            result[code] = {
                "price": d.get("p"),
                "chg_pct": d.get("pc"),         # 涨跌幅%
                "chg": d.get("ud"),              # 涨跌额
                "volume": d.get("v"),            # 成交量(万手)
                "amount": d.get("cje"),          # 成交额(元)
                "amplitude": d.get("zf"),        # 振幅%
                "turnover": d.get("hs"),         # 换手率%
                "pe": d.get("pe"),               # 市盈率
                "pb": d.get("lb"),               # 市净率
                "eps": d.get("fm"),              # 每股收益
                "high": d.get("h"),
                "low": d.get("l"),
                "open": d.get("o"),
                "prev_close": d.get("yc"),
                "total_mv": d.get("sz"),         # 总市值
                "float_mv": d.get("lt"),         # 流通市值
                "chg_60d": d.get("zdf60"),       # 60日涨跌幅%
                "chg_ytd": d.get("zdfnc"),       # 年初至今涨跌幅%
            }
        except Exception as e:
            logger.debug(f"知兔行情查询失败 [{code}]: {e}")
            continue

    return result


def _sina_quote_url() -> str:
    """生成新浪批量行情请求 URL，支持多只股票同时查询。"""
    return "https://hq.sinajs.cn/list={codes_str}"


def _fetch_sina_quotes(codes: list[str]) -> dict[str, dict]:
    """批量查询新浪实时行情（兜底方案，知兔 API 不可用时使用）。

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
    - 大盘指数：新浪行情 API（实时点位）+ 本地 index_daily 表（趋势）
    - 市场情绪：本地 stock_daily 表（涨跌比/中位数/涨停家数）
    - 个股行情 + 财务：知兔 API（PE/PB/市值/60日涨幅，主数据源）
    - 个股行情兜底：新浪行情 API
    - 个股财务兜底：本地 SQLite（peTTM, pbMRQ）
    - 股票名称：baostock API
    - 新闻公告：东方财富公告 API（直连 HTTP）
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

        1. 收集大盘指数 + 市场情绪数据
        2. 对每只候选股采集实时行情 + 财务数据（知兔 API 优先）
        3. 将全部实时数据作为 prompt 上下文 → LLM 分析
        """
        logger.info("MarketAnalyst: 开始实时数据采集（知兔 API + 大盘情绪 + SQLite）...")
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

        计算各指数的今日涨跌幅、近5日涨跌幅、近20日涨跌幅。

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

                    # 近5日涨跌幅
                    if len(df) >= 5:
                        week_chg = (
                            (latest["close"] / df.iloc[4]["close"]) - 1
                        ) * 100
                        week_str = f"{week_chg:+.2f}%"
                    else:
                        week_str = "N/A"

                    # 近20日涨跌幅
                    if len(df) >= 20:
                        month_chg = (
                            (latest["close"] / df.iloc[19]["close"]) - 1
                        ) * 100
                        month_str = f"{month_chg:+.2f}%"
                    else:
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

    def _fetch_market_breadth(self) -> list[str]:
        """从本地 stock_daily 表计算全面的市场情绪与结构指标。

        全部基于本地数据，零网络依赖。包含：
        1. 基础情绪：涨跌比、中位数、涨停/跌停
        2. 板块分层：按代码前缀统计各板块中位数涨幅
        3. 成交量集中度：TOP10 股票成交占比
        4. 涨跌幅分布：各涨跌幅区间的股票数量
        5. 市场估值中位数：全市场 PE/PB 中位数
        6. 日内振幅分布：市场波动烈度
        7. K线形态统计：阳线/阴线/十字星比例
        8. 价格分层表现：不同价位段的涨跌差异
        9. PE/PB四象限：价值/成长/困境分类
        10. 成交额分布：大额交易活跃度

        Returns:
            格式化的市场情绪与结构指标列表。
        """
        try:
            import sqlite3
            import pandas as pd
            import numpy as np

            conn = sqlite3.connect(self.settings.db_path)

            # 获取所有股票的最新日线数据（扩充字段：high, low, open, amount）
            df = pd.read_sql(
                "SELECT symbol, date, open, high, low, close, "
                "pctChg, volume, amount, peTTM, pbMRQ "
                "FROM stock_daily "
                "WHERE (symbol, date) IN ("
                "  SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
                ")",
                conn,
            )
            conn.close()

            if df.empty:
                return []

            latest_date = df["date"].max()
            df_today = df[df["date"] == latest_date].copy()
            if df_today.empty:
                return []

            total = len(df_today)
            pct_chgs = df_today["pctChg"].dropna()
            volumes = df_today["volume"].fillna(0)

            lines = [f"统计日期: {latest_date}", f"样本总数: {total} 只"]

            # ── 1. 基础市场情绪（原有，保留） ──
            up_count = int((pct_chgs > 0).sum())
            down_count = int((pct_chgs < 0).sum())
            flat_count = int((pct_chgs == 0).sum())
            median_chg = pct_chgs.median()
            mean_chg = pct_chgs.mean()
            limit_up = int((pct_chgs >= 9.8).sum())
            limit_down = int((pct_chgs <= -9.8).sum())
            strong_up = int((pct_chgs > 3).sum())
            strong_down = int((pct_chgs < -3).sum())
            ad_ratio = f"{up_count / down_count:.2f}" if down_count > 0 else "∞"

            lines.append("")
            lines.append("【基础情绪】")
            lines.append(f"上涨/下跌/平盘: {up_count}/{down_count}/{flat_count}")
            lines.append(f"涨跌比: {ad_ratio}")
            lines.append(f"涨跌幅中位数: {median_chg:+.2f}% | 均值: {mean_chg:+.2f}%")
            lines.append(f"涨停: {limit_up} | 跌停: {limit_down} | 强势(>3%): {strong_up} | 弱势(<-3%): {strong_down}")

            # ── 2. 板块分层表现（按代码前缀） ──
            def classify_board(symbol: str) -> str:
                if symbol.startswith("688"):
                    return "科创板(688)"
                elif symbol.startswith("300") or symbol.startswith("301"):
                    return "创业板(300)"
                elif symbol.startswith("002"):
                    return "中小板(002)"
                elif symbol.startswith(("00", "001", "003")):
                    return "深主板(00)"
                elif symbol.startswith(("60", "603", "605")):
                    return "沪主板(60)"
                elif symbol.startswith("4") or symbol.startswith("8"):
                    return "北交所(4/8)"
                else:
                    return "其他"

            df_today["board"] = df_today["symbol"].apply(classify_board)
            board_stats = (
                df_today.groupby("board")["pctChg"]
                .agg(["count", "median", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()])
                .rename(columns={"<lambda_0>": "上涨", "<lambda_1>": "下跌"})
            )
            board_stats["涨跌比"] = np.where(
                board_stats["下跌"] > 0,
                (board_stats["上涨"] / board_stats["下跌"]).round(2),
                "∞"
            )

            lines.append("")
            lines.append("【板块分层表现（按代码前缀）】")
            for board_name in ["沪主板(60)", "深主板(00)", "中小板(002)", "创业板(300)", "科创板(688)"]:
                if board_name in board_stats.index:
                    r = board_stats.loc[board_name]
                    lines.append(
                        f"  {board_name}: {int(r['count'])}只 "
                        f"中位数{r['median']:+.2f}% "
                        f"涨跌比{r['涨跌比']}"
                    )

            # ── 3. 成交量集中度 ──
            vol_sorted = volumes.sort_values(ascending=False)
            vol_total = vol_sorted.sum()
            if vol_total > 0:
                top5_pct = vol_sorted.head(5).sum() / vol_total * 100
                top10_pct = vol_sorted.head(10).sum() / vol_total * 100
                top50_pct = vol_sorted.head(50).sum() / vol_total * 100

                lines.append("")
                lines.append("【成交量集中度】")
                lines.append(f"TOP5 成交占比: {top5_pct:.1f}%")
                lines.append(f"TOP10 成交占比: {top10_pct:.1f}%")
                lines.append(f"TOP50 成交占比: {top50_pct:.1f}%")

                # 集中度判断
                if top10_pct > 30:
                    lines.append("  → 高度集中（少数股票主导，警惕流动性风险）")
                elif top10_pct > 20:
                    lines.append("  → 中度集中（行情分化，关注龙头）")
                else:
                    lines.append("  → 较为分散（普涨/普跌格局）")

                lines.append(f"全市场总成交: {vol_total/1e6:.0f}亿手")

            # ── 4. 涨跌幅分布 ──
            bins = [-float("inf"), -9.8, -5, -3, -1, 0, 1, 3, 5, 9.8, float("inf")]
            labels = ["跌停", "-5~-9.8%", "-3~-5%", "-1~-3%", "-1~0%", "0~1%", "1~3%", "3~5%", "5~9.8%", "涨停"]
            df_today["bucket"] = pd.cut(pct_chgs, bins=bins, labels=labels, right=False)
            dist = df_today["bucket"].value_counts()

            lines.append("")
            lines.append("【涨跌幅分布】")
            for lbl in labels:
                cnt = int(dist.get(lbl, 0))
                bar = "█" * min(cnt // 30, 30)
                lines.append(f"  {lbl:>8}: {cnt:>4d}只 {bar}")

            # ── 5. 市场估值中位数 ──
            pe_vals = df_today["peTTM"].dropna()
            pb_vals = df_today["pbMRQ"].dropna()

            # 过滤极端值
            pe_sane = pe_vals[(pe_vals > 0) & (pe_vals < 500)]
            pb_sane = pb_vals[(pb_vals > 0) & (pb_vals < 50)]

            lines.append("")
            lines.append("【市场估值中位数】")
            if not pe_sane.empty:
                lines.append(f"  PE(TTM)中位数: {pe_sane.median():.1f} "
                             f"(正数样本{len(pe_sane)}只)")
            if not pb_sane.empty:
                lines.append(f"  PB(MRQ)中位数: {pb_sane.median():.2f} "
                             f"(正数样本{len(pb_sane)}只)")

            # PE 区间分布
            pe_bins = [0, 10, 20, 30, 50, 100, 500]
            pe_labels = ["<10", "10~20", "20~30", "30~50", "50~100", ">100"]
            pe_dist = pd.cut(pe_sane, bins=pe_bins, labels=pe_labels, right=False).value_counts()
            lines.append("  PE分布: " + " | ".join(
                f"{lbl}:{int(pe_dist.get(lbl,0))}只" for lbl in pe_labels
            ))

            # ═══════════════════════════════════════════════
            # 新增强大模块 6~10
            # ═══════════════════════════════════════════════

            # ── 6. 日内振幅分布（市场波动烈度） ──
            if "high" in df_today.columns and "low" in df_today.columns:
                closes = df_today["close"].replace(0, pd.NA)
                amplitudes = (
                    (df_today["high"] - df_today["low"]) / closes * 100
                ).dropna()

                amp_median = amplitudes.median()
                amp_mean = amplitudes.mean()

                amp_bins = [0, 1, 2, 3, 4, 6, 10, 100]
                amp_labels = ["<1%", "1~2%", "2~3%", "3~4%", "4~6%", "6~10%", ">10%"]
                amp_dist = pd.cut(amplitudes, bins=amp_bins, labels=amp_labels, right=False).value_counts()

                lines.append("")
                lines.append("【日内振幅分布（波动烈度）】")
                lines.append(f"振幅中位数: {amp_median:.1f}% | 均值: {amp_mean:.1f}%")
                for lbl in amp_labels:
                    cnt = int(amp_dist.get(lbl, 0))
                    bar = "█" * min(cnt // 50, 20)
                    lines.append(f"  {lbl:>6}: {cnt:>4d}只 {bar}")

                # 高波动占比
                high_vol = int((amplitudes > 4).sum())
                lines.append(f"高波动(>4%): {high_vol}只 ({high_vol/len(amplitudes)*100:.1f}%)")

            # ── 7. K线形态统计（阳线/阴线/十字星） ──
            if "open" in df_today.columns and "high" in df_today.columns:
                opens = df_today["open"]
                closes = df_today["close"]
                highs = df_today["high"]
                lows = df_today["low"]

                yang = int(((closes > opens) & (pct_chgs > 0)).sum())
                yin = int(((closes < opens) & (pct_chgs < 0)).sum())
                # 十字星：实体 < 振幅的10% 且 有一定振幅
                body = (closes - opens).abs()
                span = (highs - lows).replace(0, pd.NA)
                doji_ratio = body / span
                doji = int(((doji_ratio < 0.1) & (span > closes * 0.005)).sum())

                # 光头光脚阳线：close=high and open=low
                strong_yang = int(((closes == highs) & (opens == lows) & (pct_chgs > 0)).sum())
                strong_yin = int(((closes == lows) & (opens == highs) & (pct_chgs < 0)).sum())

                lines.append("")
                lines.append("【K线形态】")
                lines.append(f"阳线: {yang}只 ({yang/total*100:.0f}%) | "
                             f"阴线: {yin}只 ({yin/total*100:.0f}%)")
                lines.append(f"十字星(窄实体): {doji}只 ({doji/total*100:.0f}%)")
                if strong_yang > 0:
                    lines.append(f"光头光脚阳线(极强): {strong_yang}只")
                if strong_yin > 0:
                    lines.append(f"光头光脚阴线(极弱): {strong_yin}只")

            # ── 8. 价格分层表现 ──
            closes = df_today["close"].dropna()
            price_bins = [0, 5, 10, 20, 50, 100, 500, float("inf")]
            price_labels = ["<5元", "5~10", "10~20", "20~50", "50~100", "100~500", ">500"]
            df_today["price_tier"] = pd.cut(
                closes, bins=price_bins, labels=price_labels, right=False
            )
            price_stats = (
                df_today.groupby("price_tier", observed=True)["pctChg"]
                .agg(["count", "median"])
            )

            lines.append("")
            lines.append("【价格分层表现】")
            for lbl in price_labels:
                if lbl in price_stats.index:
                    r = price_stats.loc[lbl]
                    bar = "█" * min(int(r["count"]) // 50, 20)
                    lines.append(
                        f"  {lbl:>7}: {int(r['count']):>4d}只 "
                        f"中位数{r['median']:+.2f}% {bar}"
                    )

            # ── 9. PE/PB 四象限（价值/成长/困境） ──
            pe_sane_4q = pe_vals[(pe_vals > 0) & (pe_vals < 200)]
            pb_sane_4q = pb_vals[(pb_vals > 0) & (pb_vals < 30)]

            df_4q = df_today.loc[
                pe_sane_4q.index.intersection(pb_sane_4q.index)
            ].copy()
            if not df_4q.empty:
                pe_med = df_4q["peTTM"].median()
                pb_med = df_4q["pbMRQ"].median()

                df_4q["quadrant"] = "其他"
                df_4q.loc[
                    (df_4q["peTTM"] <= pe_med) & (df_4q["pbMRQ"] <= pb_med), "quadrant"
                ] = "深度价值(低PE低PB)"
                df_4q.loc[
                    (df_4q["peTTM"] > pe_med) & (df_4q["pbMRQ"] > pb_med), "quadrant"
                ] = "高成长(高PE高PB)"
                df_4q.loc[
                    (df_4q["peTTM"] <= pe_med) & (df_4q["pbMRQ"] > pb_med), "quadrant"
                ] = "轻资产高估(低PE高PB)"
                df_4q.loc[
                    (df_4q["peTTM"] > pe_med) & (df_4q["pbMRQ"] <= pb_med), "quadrant"
                ] = "困境反转型(高PE低PB)"

                quad_stats = df_4q.groupby("quadrant").agg(
                    数量=("pctChg", "count"),
                    涨幅中位数=("pctChg", "median"),
                ).sort_values("涨幅中位数", ascending=False)

                lines.append("")
                lines.append("【PE/PB四象限（风格归因）】")
                for qname, qr in quad_stats.iterrows():
                    lines.append(
                        f"  {qname}: {int(qr['数量'])}只 "
                        f"中位数{qr['涨幅中位数']:+.2f}%"
                    )

            # ── 10. 成交额分布（大额交易活跃度） ──
            if "amount" in df_today.columns:
                amounts = df_today["amount"].dropna() / 1e8  # 转为亿元
                amt_median = amounts.median()
                amt_total = amounts.sum()

                amt_bins = [0, 0.1, 0.5, 1, 3, 5, 10, 50, float("inf")]
                amt_labels = ["<0.1亿", "0.1~0.5", "0.5~1", "1~3", "3~5", "5~10", "10~50", ">50亿"]
                amt_dist = pd.cut(amounts, bins=amt_bins, labels=amt_labels, right=False).value_counts()

                lines.append("")
                lines.append("【成交额分布】")
                lines.append(f"全市场总成交: {amt_total:.0f}亿 | 中位数: {amt_median:.2f}亿")
                for lbl in amt_labels:
                    cnt = int(amt_dist.get(lbl, 0))
                    pct = cnt / total * 100
                    bar = "█" * min(cnt // 50, 15)
                    lines.append(f"  {lbl:>8}: {cnt:>4d}只 ({pct:.1f}%) {bar}")

                # 大单活跃度
                big_amt = int((amounts > 5).sum())
                lines.append(f"大额交易(>5亿): {big_amt}只 ({big_amt/total*100:.1f}%)")

            return lines

        except Exception as e:
            logger.debug(f"市场情绪数据读取失败: {e}")
            return []

    def _gather_market_context(self) -> dict:
        """采集大盘指数行情等实时数据。

        双源采集：
        1. 新浪行情 API → 今日实时点位和涨跌幅
        2. 本地 index_daily 表 → 趋势（近5日/近20日）
        """
        ctx = {
            "indices": [],
            "index_trends": [],
            "market_breadth": [],
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

        # ── 大盘指数趋势（本地 index_daily 表） ──
        try:
            trends = self._fetch_local_index_trends()
            ctx["index_trends"] = trends
        except Exception as e:
            ctx["errors"].append(f"本地指数趋势获取失败: {e}")

        # ── 市场情绪数据（本地 stock_daily 表，零网络依赖） ──
        try:
            breadth = self._fetch_market_breadth()
            ctx["market_breadth"] = breadth
        except Exception as e:
            ctx["errors"].append(f"市场情绪获取失败: {e}")

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
        1. 知兔 API（实时行情 + PE/PB/市值 + 60日/年初至今涨幅）— 主数据源
        2. 新浪行情 API（兜底）
        3. 本地 SQLite（PE/PB 财务数据兜底）
        4. 东方财富新闻公告 API（直连 HTTP）
        """
        detail = {
            "code": code,
            "name": _get_stock_name(code),
            "realtime": "",
            "fundamentals": "",
            "news_titles": [],
            "errors": [],
        }

        has_zhitu = False

        # ── 主数据源：知兔 API（含行情 + PE/PB/市值） ──
        try:
            quotes = _fetch_zhitu_quotes([code])
            if code in quotes:
                q = quotes[code]
                has_zhitu = True

                # 实时行情部分
                parts = [
                    f"最新价:{q.get('price', '?')}",
                    f"涨跌幅:{q.get('chg_pct', '?')}%",
                    f"涨跌额:{q.get('chg', '?')}",
                    f"换手率:{q.get('turnover', '?')}%",
                    f"最高:{q.get('high', '?')}",
                    f"最低:{q.get('low', '?')}",
                ]
                detail["realtime"] = " | ".join(parts)

                # 财务/估值部分
                fin_parts = []
                if q.get("pe") is not None:
                    fin_parts.append(f"PE:{q['pe']:.2f}")
                if q.get("pb") is not None:
                    fin_parts.append(f"PB:{q['pb']:.3f}")
                if q.get("eps") is not None:
                    fin_parts.append(f"每股收益:{q['eps']:.3f}元")
                if q.get("total_mv"):
                    mv = q["total_mv"]
                    if mv > 1e8:
                        fin_parts.append(f"总市值:{mv/1e8:.0f}亿")
                    else:
                        fin_parts.append(f"总市值:{mv:.0f}")
                if q.get("float_mv"):
                    fmv = q["float_mv"]
                    if fmv > 1e8:
                        fin_parts.append(f"流通市值:{fmv/1e8:.0f}亿")
                if q.get("chg_60d") is not None:
                    fin_parts.append(f"60日涨跌幅:{q['chg_60d']:+.2f}%")
                if q.get("chg_ytd") is not None:
                    fin_parts.append(f"年初至今:{q['chg_ytd']:+.2f}%")

                if fin_parts:
                    detail["fundamentals"] = " | ".join(fin_parts)
        except Exception as e:
            detail["errors"].append(f"知兔行情: {e}")

        # ── 兜底：新浪行情（知兔不可用时） ──
        if not has_zhitu:
            try:
                sina_quotes = _fetch_sina_quotes([code])
                if code in sina_quotes:
                    q = sina_quotes[code]
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

        # ── 财务数据兜底：本地 SQLite（知兔未提供 PE/PB 时） ──
        if not has_zhitu or not detail["fundamentals"]:
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
                    info_parts = [f"日期:{r['date']}"]
                    if pd.notna(r["close"]):
                        info_parts.append(f"收盘价:{r['close']:.2f}")
                    if pd.notna(r["pctChg"]):
                        info_parts.append(f"涨跌幅:{r['pctChg']:+.2f}%")
                    if pd.notna(r["peTTM"]):
                        info_parts.append(f"PE(TTM):{r['peTTM']:.2f}")
                    if pd.notna(r["pbMRQ"]):
                        info_parts.append(f"PB(MRQ):{r['pbMRQ']:.3f}")

                    detail["fundamentals"] = " | ".join(info_parts)

                    # 如果实时行情也缺失，用本地数据顶替
                    if not detail["realtime"]:
                        detail["realtime"] = " | ".join(info_parts[:4])
            except Exception as e:
                detail["errors"].append(f"本地财务: {e}")

        # ── 近期新闻公告（东方财富公告 API，不走 akshare） ──
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

        if market.get("index_trends"):
            market_lines.append("")
            market_lines.append("【大盘趋势（近5日/近20日涨跌幅）】")
            market_lines.extend(market["index_trends"])

        if market.get("market_breadth"):
            market_lines.append("")
            market_lines.append("【市场情绪（全市场统计）】")
            market_lines.extend(market["market_breadth"])

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
                f"估值/财务: {sd['fundamentals'] or '（暂缺）'}"
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
        prompt = f"""你是一位经验丰富的 A 股市场分析师。以下是 {today_str} 的实时市场数据和候选股票信息，全部是实时采集的（来源：知兔 API + 新浪行情 API + 本地数据库），请基于这些数据进行分析，**不要依赖你自己的训练知识**。

## 📊 今日市场实时数据
{" ".join(market_lines)}

## 🎯 候选股票（量化策略选出，附实时数据）
{" ".join(stock_lines)}

## 🔗 策略与股票对应关系
{" ".join(strategy_lines)}

## 分析要求
请对上述候选股票进行多维度综合分析，必须严格基于上面提供的实时数据：

1. **大盘与市场情绪** — 调用上面的指数数据、趋势和全市场涨跌比
2. **实时行情与基本面** — 调用上面的行情/估值数据（PE/PB/60日涨幅等）
3. **新闻公告** — 调用上面的公告标题，判断是否有重大事项
4. **综合研判** — 综合所有数据，给出最终推荐

## 输出格式
请严格按照以下格式输出（不要添加额外说明）：

📊 Sequoia-X AI 综合研判 | {today_str}

### 📈 大盘环境
[根据上面提供的实时指数数据和趋势，1-2句话总结]

### 🔍 个股深度分析

**1. [股票名称] ([代码]) — [对应策略名]**
- 综合评分: ⭐X.X/5
- 核心逻辑: [一句话]
- 实时行情分析: [引用上面的行情数据]
- 基本面与估值: [引用上面的 PE/PB/市值等数据]
- 趋势分析: [引用上面的 60日/年初至今涨跌幅]
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
