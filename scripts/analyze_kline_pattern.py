#!/usr/bin/env python
"""K 线形态全市场对比分析：扫描形态确认日 → 次日开盘买入 → 持有收益统计。

用法:
    # 全形态（默认老鸭头），2026 年至今
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python scripts/analyze_kline_pattern.py --pattern 老鸭头 --start 2026-01-01

输出:
    1. 形态确认日统计（样本数、按月分布）
    2. 对比表：开盘溢价 / 持1天 / 1天涨占比 / 持5天 / 持20天
    3. 与全市场基线对比 + 按季度细分

口径说明:
    - 形态确认日 = T（T 日收盘后形态成立，T+1 日开盘价买入——T+1 模型）
    - 收益 = T+N 日收盘 / T+1 日开盘 - 1
    - 2026-08-03 建立（延续"形态好≠次日涨"研究：全市场 10 万+ 样本实证 1 天涨占比 <50%）

新增形态: 在 PATTERNS 字典加一项 {"func": <函数>, "desc": "..."}
    函数签名: def pattern(group: pd.DataFrame) -> pd.Series  # 返回 bool 掩码（该日形态确认）
    约定: group 为单只股票按日期升序的 DataFrame，含 open/close/high/low/volume/ma5/ma10/ma20
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "sequoia_v2.db"


# ═══════════════════════════════════════════════════════════
# 形态定义库（新增形态在此添加）
# ═══════════════════════════════════════════════════════════

def old_duck_head(g: pd.DataFrame) -> pd.Series:
    """老鸭头形态（均线体系，向量化实现）。

    形态要素（T 日确认）:
      1. 鸭头左耳：T-22~T-3 窗口内出现阶段性高点 H（高于更早 60 日窗口的最高——近期新高）
      2. 鸭头回踩：从 H 回调 3%~25%（H / 回调期最低 L - 1 ∈ [3%, 25%]）
      3. 鸭嘴支撑：回调最低 L ≥ 当前 MA20 × 0.93（近似"回调不破 20 日线"——
         实测 low_20/ma20 中位 0.918，0.93 阈值 ≈ 低点当日不低于其 MA20）
      4. 鸭嘴张开：T 日 close > MA5 > MA10（重新走强）
      5. 回升确认：T 日 close > L × 1.02（已从低点回升 2%+）
      6. 当前仍低于高点（形态未走完）
    """
    close = g["close"]
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    hi_recent = close.rolling(20).max().shift(3)   # 截至 T-3 的 20 日最高（高点位置 ≈ T-22~T-3）
    hi_before = close.rolling(60).max().shift(22)  # T-22 之前的 60 日最高（更早高点）
    low_20 = close.rolling(20).min()               # 回调期最低（截至 T）

    mask = pd.Series(False, index=g.index)
    valid = close.notna() & hi_recent.notna() & hi_before.notna() & low_20.notna() & ma20.notna()

    # 1. 近期新高（H 高于更早高点）
    m1 = valid & (hi_recent > hi_before)
    # 2. 回撤 3%~25%（H 到回调最低）
    drawdown = hi_recent / low_20 - 1
    m2 = m1 & (drawdown >= 0.03) & (drawdown <= 0.25)
    # 3. 回调低点 ≥ 当前 MA20 × 0.93（近似不破 20 日线；实测 2026 数据中位 0.918）
    m3 = m2 & (low_20 >= ma20 * 0.93)
    # 4. 重新走强：close > MA5 > MA10
    m4 = m3 & (close > ma5) & (ma5 > ma10)
    # 5. 已从低点回升 2%+
    m5 = m4 & (close > low_20 * 1.02)
    # 6. 当前仍低于高点（形态未走完）：close < hi_recent（防止已创新高后的样本）
    m6 = m5 & (close < hi_recent)

    mask[m6.fillna(False)] = True
    return mask


def magic_nine(g: pd.DataFrame, direction: str = "bottom", min_gain: float = 0.0) -> pd.Series:
    """神奇九转（TD Sequential 简化版）。

    底部九转: 连续 9 日 close < 各自 4 日前 close（超跌序列，底部反转信号 → 买入候选）
    顶部九转: 连续 9 日 close > 各自 4 日前 close（超涨序列，顶部反转信号 → 理论上应回避）
    第 9 根出现日为确认日（count==9；之后的延续不计，中断即重新计数）。
    min_gain: 严格版附加条件——确认日相对 9 日前收盘的累计涨跌幅下限
              （底: 累计跌幅 ≥ min_gain；顶: 累计涨幅 ≥ min_gain；默认 0 = 原版）
    """
    close = g["close"].values
    n = len(close)
    cond_full = np.zeros(n, dtype=bool)
    if n > 4:
        cond_full[4:] = close[4:] < close[:-4] if direction == "bottom" else close[4:] > close[:-4]
    mask = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if cond_full[i]:
            count += 1
            if count == 9 and i >= 9:
                # 9 日累计涨跌幅检查（严格版）
                if direction == "bottom":
                    ok = (close[i - 9] - close[i]) >= close[i - 9] * min_gain
                else:
                    ok = (close[i] - close[i - 9]) >= close[i - 9] * min_gain
                if ok:
                    mask[i] = True
        else:
            count = 0
    return pd.Series(mask, index=g.index)


PATTERNS = {
    "老鸭头": {
        "func": old_duck_head,
        "desc": "拉升 → 回踩不破20日线 → 再次走强（\"二次启动\"结构）",
        "detail": "① 近期阶段新高 ② 回调3%~25% ③ 回调低点≥MA20×0.93 ④ close>MA5>MA10 ⑤ 已回升≥2% ⑥ 未创新高",
    },
    "神奇九转(底)": {
        "func": lambda g: magic_nine(g, "bottom"),
        "desc": "连续 9 日收跌（TD Sequential 底部九转，超跌反转信号）",
        "detail": "① 连续 9 日 close < 各自 4 日前 close ② 第 9 根出现日为确认日 ③ 中断即重新计数",
    },
    "神奇九转(顶)": {
        "func": lambda g: magic_nine(g, "top"),
        "desc": "连续 9 日收涨（TD Sequential 顶部九转，超涨反转信号）",
        "detail": "① 连续 9 日 close > 各自 4 日前 close ② 第 9 根出现日为确认日 ③ 中断即重新计数",
    },
    "神奇九转(顶·严)": {
        "func": lambda g: magic_nine(g, "top", min_gain=0.05),
        "desc": "连续 9 日收涨 + 9 日累计涨幅≥5%（严格版：要求真正的强势波段）",
        "detail": "① 连续 9 日 close > 各自 4 日前 close ② 确认日相对 9 日前累计涨幅 ≥5% ③ 第 9 根为确认日",
    },
    "神奇九转(底·严)": {
        "func": lambda g: magic_nine(g, "bottom", min_gain=0.05),
        "desc": "连续 9 日收跌 + 9 日累计跌幅≥5%（严格版：要求真正的超跌波段）",
        "detail": "① 连续 9 日 close < 各自 4 日前 close ② 确认日相对 9 日前累计跌幅 ≥5% ③ 第 9 根为确认日",
    },
}


# ═══════════════════════════════════════════════════════════
# 分析主流程
# ═══════════════════════════════════════════════════════════

HOLD_PERIODS = [1, 3, 5, 10, 15]   # 持有期（天），每期输出：绝对收益 + 相对沪深300超额收益


def load_data(start: str) -> pd.DataFrame:
    """加载全市场日线（open/close）+ 沪深300 基准，计算均线与前瞻收益。

    预热：提前 300 天加载（形态需 60+ 日均线历史），统计时按 start 过滤。
    超额收益口径（与股票完全同口径）：
      股票收益 = T+N 收盘 / T+1 开盘 - 1
      指数收益 = 沪深300 T+N 收盘 / T+1 开盘 - 1（同一买入时点、同一结算时点）
    """
    conn = sqlite3.connect(DB_PATH)
    # 预热期：形态窗口最长 60 日 + shift 22 + 20 日均线 ≈ 110 日，300 天余量充足
    df = pd.read_sql(
        f"SELECT symbol, date, open, close FROM stock_daily "
        f"WHERE date >= date('{start}', '-300 days')", conn)
    # 沪深300 指数（超额基准）
    idx = pd.read_sql(
        "SELECT date, open, close FROM index_daily WHERE symbol='sh.000300'", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    # 指数前瞻收益（按指数自身交易日序列）
    idx["buy_open"] = idx["open"].shift(-1)                    # T+1 开盘
    for n in HOLD_PERIODS:
        idx[f"idx_r{n}"] = idx["close"].shift(-(n + 1)) / idx["buy_open"] - 1

    g = df.groupby("symbol", sort=False)
    df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
    df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())

    # 前瞻收益（T+1 开盘买入 → 持 N 天）——先用 g 算完所有列，再统一 merge 指数
    df["buy_open"] = g["open"].shift(-1)                      # T+1 开盘价（买入价）
    for n in HOLD_PERIODS:
        df[f"r{n}"] = g["close"].shift(-(n + 1)) / df["buy_open"] - 1

    # 超额收益：合并指数同期收益（按日期对齐，同口径）
    df = df.merge(idx[["date"] + [f"idx_r{n}" for n in HOLD_PERIODS]], on="date", how="left")
    for n in HOLD_PERIODS:
        df[f"ex{n}"] = df[f"r{n}"] - df[f"idx_r{n}"]

    df["gap"] = df["buy_open"] / df["close"] - 1              # T+1 开盘溢价（vs T 收盘）
    return df


def scan_pattern(df: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
    """全市场扫描形态确认日，返回信号日明细。"""
    pat = PATTERNS[pattern_name]
    signals = []
    for sym, g in df.groupby("symbol", sort=False):
        mask = pat["func"](g)
        if mask.any():
            cols = ["date", "gap"] + [f"r{n}" for n in HOLD_PERIODS] + [f"ex{n}" for n in HOLD_PERIODS]
            hit = g[mask][cols]
            hit.insert(0, "symbol", sym)
            signals.append(hit)
    return pd.concat(signals, ignore_index=True) if signals else pd.DataFrame()


def _per_hold(ser: pd.Series, name: str) -> dict:
    """单个持有期的统计：均值/中位/涨占比/盈亏比。"""
    s = ser.dropna()
    if len(s) == 0:
        return {f"{name}均值": np.nan, f"{name}中位": np.nan, f"{name}涨占比": np.nan, f"{name}盈亏比": np.nan}
    pos = s[s > 0].mean() if (s > 0).any() else np.nan
    neg = s[s < 0].mean() if (s < 0).any() else np.nan
    plr = (pos / abs(neg)) if (pos == pos and neg == neg and neg != 0) else np.nan
    return {
        f"{name}均值": s.mean() * 100,
        f"{name}中位": s.median() * 100,
        f"{name}涨占比": (s > 0).mean() * 100,
        f"{name}盈亏比": plr,
    }


def build_html_report(sig: pd.DataFrame, base: pd.DataFrame, pattern_name: str,
                      start: str, n_stocks: int, n_days_total: int) -> str:
    """生成不等距 HTML 报告（介绍区 + 数据区 + 说明区 + 结论区，§7.4 结构）。

    结论区自动生成：基于超额收益数据归纳 4 条（短期持平/越久越输/分季度环境/总评）。
    """
    pat = PATTERNS[pattern_name]
    s = summary_table(sig, pattern_name)
    b = summary_table(base, "全市场基线(所有日)")
    esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def td(v):  # 数据单元格（百分比格式化）
        return f"<td>{v:+.2f}%</td>" if isinstance(v, float) else f"<td>{v}</td>"

    # 结论区自动生成
    ex1, ex5, ex15 = s["持1天超额"], s["持5天超额"], s["持15天超额"]
    exb15 = b["持15天超额"]
    conds = []
    conds.append(
        f"① 持1-3天超额 {ex1:+.2f}%/{s['持3天超额']:+.2f}% —— "
        f"{'与大盘基本持平，无显著优势' if abs(ex1) < 0.3 else ('短期跑赢大盘' if ex1 > 0 else '短期跑输大盘')}。")
    conds.append(
        f"② 持5-15天超额 {ex5:+.2f}% → {ex15:+.2f}% —— "
        f"{'持有越久越跑输大盘' if ex15 < ex5 else '持有期越长超额越改善'}；"
        f"相对全市场随机买入（{exb15:+.2f}%）{'仍有微弱优势' if ex15 > exb15 else '无相对优势'}。")
    # 季度环境
    sig2 = sig.copy()
    sig2["季度"] = sig2["date"].dt.to_period("Q")
    q_lines = []
    for q, sub in sig2.groupby("季度"):
        qs = summary_table(sub, str(q))
        q_lines.append(f"{q} 持15天超额 {qs['持15天超额']:+.2f}%")
    q_sum = "；".join(q_lines)
    conds.append(f"③ 分季度：{q_sum} —— 跑赢/跑输取决于市场环境。")
    verdict = "总评：该形态净跑输大盘，应回避" if ex15 < 0 else "总评：该形态整体跑赢大盘，可参考"
    conds.append(f"④ {verdict}（超额 {ex15:+.2f}%，样本 {s['样本']:,} 条）。")

    h = []
    h.append("<!DOCTYPE html><html lang='zh'><head><meta charset='UTF-8'>")
    h.append("<title>K线形态对比分析</title>")
    h.append("<style>body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;padding:20px;background:#fafafa;}")
    h.append("table{border-collapse:collapse;width:100%;max-width:1400px;margin:0 auto;background:#fff;")
    h.append("box-shadow:0 2px 8px rgba(0,0,0,.1)}")
    h.append("th,td{border:1px solid #999;padding:6px 8px;font-size:13px;text-align:center}")
    h.append("th{background:#f0f4f8;font-weight:600}")
    h.append(".title-row th{background:#dce6f1;font-size:17px;padding:10px}")
    h.append(".label-col{width:11%;background:#f0f4f8;font-weight:600}")
    h.append(".content-col{text-align:left}")
    h.append(".conclusion td{text-align:left;line-height:1.8}</style></head><body>")
    h.append("<table>")
    # 标题
    h.append('<tr class="title-row"><th colspan="14">K线形态全市场样本对比分析</th></tr>')
    # 一、介绍
    h.append(f'<tr><th class="label-col" colspan="2">K线形态</th><td class="content-col" colspan="12">{esc(pattern_name)}</td></tr>')
    h.append(f'<tr><th class="label-col" colspan="2">定义</th><td class="content-col" colspan="12">{esc(pat["desc"])}</td></tr>')
    h.append(f'<tr><th class="label-col" colspan="2">量化描述</th><td class="content-col" colspan="12">{esc(pat.get("detail", ""))}</td></tr>')
    h.append(f'<tr><th class="label-col" colspan="2">起止日期</th>'
             f'<td class="content-col" colspan="12">{sig["date"].min().date()} ~ {sig["date"].max().date()}'
             f'（全市场 {n_stocks:,} 只，{n_days_total} 交易日，预热 300 天）</td></tr>')
    # 二、数据
    h.append("<tr>" + "".join(f"<th>{c}</th>" for c in
             ["对比对象", "样本", "覆盖股票", "开盘溢价",
              "持1天收益", "持1天超额", "持3天收益", "持3天超额",
              "持5天收益", "持5天超额", "持10天收益", "持10天超额",
              "持15天收益", "持15天超额"]) + "</tr>")
    for row in (s, b):
        cells = [f"<td><b>{esc(row['对比对象'])}</b></td>" if row['对比对象'] == pattern_name else f"<td>{esc(row['对比对象'])}</td>",
                 f"<td>{row['样本']:,}</td>", f"<td>{row['覆盖股票']:,}</td>", f"<td>{row['开盘溢价均值']:+.2f}%</td>"]
        for n in HOLD_PERIODS:
            cells.append(f"<td>{row[f'持{n}天收益']:+.2f}%</td>")
            cells.append(f"<td>{row[f'持{n}天超额']:+.2f}%</td>")
        h.append("<tr>" + "".join(cells) + "</tr>")
    # 三、说明
    h.append('<tr><th class="label-col" colspan="2">说明</th><td class="content-col" colspan="12">'
             "样本=形态确认信号条数；覆盖股票=出现该形态的去重股票数；开盘溢价=T+1开盘÷T收盘−1（均值）；"
             "收益=T+N收盘÷T+1开盘−1（均值）；<b>超额=收益−沪深300同期同口径收益（正=跑赢大盘）</b></td></tr>")
    # 四、结论
    h.append('<tr class="conclusion"><th class="label-col" colspan="2">结论</th><td colspan="12">')
    h.append("<br>".join(conds))
    h.append("</td></tr></table></body></html>")
    return "\n".join(h)


def summary_table(sig: pd.DataFrame, label: str) -> pd.Series:
    """单形态的汇总统计（单表：每持有期 收益 + 超额两列）。

    结构:
      对比对象 / 样本 / 覆盖股票 / 信号期 / 开盘溢价均值 /
      持1/3/5/10/15天 × {收益均值, 超额均值(相对沪深300)}
    """
    n = len(sig)
    if n == 0:
        return pd.Series({"对比对象": label, "样本": 0})
    d = {
        "对比对象": label,
        "样本": n,
        "覆盖股票": sig["symbol"].nunique(),
        "信号期": f"{sig['date'].min().date()}~{sig['date'].max().date()}",
        "开盘溢价均值": sig["gap"].mean() * 100,
    }
    for n_days in HOLD_PERIODS:
        d[f"持{n_days}天收益"] = sig[f"r{n_days}"].dropna().mean() * 100
        d[f"持{n_days}天超额"] = sig[f"ex{n_days}"].dropna().mean() * 100
    return pd.Series(d)


def main():
    ap = argparse.ArgumentParser(description="K 线形态全市场对比分析")
    ap.add_argument("--pattern", default="老鸭头", choices=list(PATTERNS))
    ap.add_argument("--start", default="2026-01-01", help="起始日期（含）")
    args = ap.parse_args()

    print(f"形态: {args.pattern} | 起始: {args.start}")
    print(f"定义: {PATTERNS[args.pattern]['desc']}")
    print("=" * 80)

    df = load_data(args.start)
    print(f"数据: {df['symbol'].nunique()} 只, {df['date'].nunique()} 交易日, {len(df):,} 行")

    # 形态扫描（预热期信号剔除，只统计 start 之后）
    sig = scan_pattern(df, args.pattern)
    sig = sig[sig["date"] >= pd.Timestamp(args.start)]
    if sig.empty:
        print(f"⚠️ 未扫描到任何 {args.pattern} 形态确认日（检查形态定义参数）")
        return
    print(f"形态确认日: {len(sig):,} 条 | 覆盖股票 {sig['symbol'].nunique()} 只 | "
          f"日期 {sig['date'].min().date()} ~ {sig['date'].max().date()}")
    print()

    # 按月分布
    monthly = sig.groupby(sig["date"].dt.to_period("M")).size()
    print("按月出现次数:")
    for m, c in monthly.items():
        print(f"  {m}: {c}")
    print()

    # 对比表：形态 vs 基线（全市场所有交易日的次日开盘买入）——单表
    # 基线只统计 start 之后（预热期数据不参与对比，口径与形态一致）
    base = df[df["date"] >= pd.Timestamp(args.start)].dropna(subset=["r1"]).copy()
    table = pd.concat([
        summary_table(sig, args.pattern),
        summary_table(base, "全市场基线(所有日)"),
    ], axis=1).T
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n════════ 主结果表 ════════")
    print("（收益 = T+N收盘/T+1开盘-1 均值；超额 = 收益 - 沪深300同期同口径收益）")
    print(table.round(2).to_string(index=False))

    # 按季度细分（同口径，紧凑单行）
    print("\n按季度细分:")
    sig["季度"] = sig["date"].dt.to_period("Q")
    for q, sub in sig.groupby("季度"):
        s = summary_table(sub, str(q))
        parts = " | ".join(
            f"持{n}天 {s[f'持{n}天收益']:+.2f}%(超额{s[f'持{n}天超额']:+.2f}%)" for n in HOLD_PERIODS)
        print(f"  {q}: 样本 {s['样本']:>5,} | 覆盖 {s['覆盖股票']:>4,} | 开盘溢价 {s['开盘溢价均值']:+.2f}% | {parts}")

    # HTML 报告（不等距大表：介绍 + 数据 + 说明 + 结论）
    from pathlib import Path
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"kline_pattern_{args.pattern}_{args.start}.html"
    html = build_html_report(sig, base, args.pattern, args.start,
                             df["symbol"].nunique(), df["date"].nunique())
    out_file.write_text(html, encoding="utf-8")
    print(f"\n✅ HTML 报告已生成: {out_file}")


if __name__ == "__main__":
    main()
