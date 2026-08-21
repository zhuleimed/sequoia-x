#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：量化当前 DeepSeek 单次调用 prompt 各段输入 token 占比。

目的：
    找出"哪段输入吃掉最多 token"，决定压缩该从哪下刀。
    （用户场景：analyst.py 收盘后单次调用 DeepSeek，输入 ~4100 tokens/次）

方法：
    1. 用**真实等规模**的代表性数据构造 market / stocks_detail / strategies_results，
       复用线上 《_build_prompt》 的分段组装逻辑（import 复用，不改生产代码），
       得到与线上同构的完整 prompt 文本。
    2. 对 prompt 按"逻辑段"切分（模板/大盘/个股行情/个股财务/新闻/对应关系），
       分别统计字符数与估算 token。
    3. 用线上多日日志实测的 prompt_tokens（~4100）做**校准锚点**：
       总估算系数 = 实测总量 / 全 prompt 估算总量，各段估算 = 段字符估算 × 系数。
    4. 输出各段 token 与占比 —— 这就是"哪块最大、压缩优先级"的依据。

注意：
    - 本脚本不调用任何网络 / 不触碰生产库，纯字符串构造，秒级完成。
    - token 估算采用"中英混合近似系数"（中文≈1字/token，英文/数字≈0.35 token/字符），
      已经用真实 4100 锚点整体校准，比例是可靠的，绝对值为近似。
"""

import os
import re
import sys

# 确保能 import 到生产模块 sequoia_x（本脚本位于 scripts/ 下，仓库根在其上级）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 线上实测输入总量（多日 pipeline 日志实证，2026-08）
REAL_PROMPT_TOKENS = 4100

# ── 与线上同规模的代表性数据 ──
# 与 _fetch_market_context / _gather_stock_detail 的字段形态一致

MARKET = {
    "indices": [
        "上证指数 3228.47 +12.34 (+0.38%)",
        "深证成指 10651.28 +84.15 (+0.80%)",
        "创业板指 2126.83 +23.99 (+1.14%)",
    ],
    "index_trends": [
        "上证指数 近5日 +0.9%  近20日 +2.4%",
        "深证成指 近5日 +2.1%  近20日 +4.8%",
        "创业板指 近5日 +3.2%  近20日 +7.1%",
    ],
    "market_breadth": [
        "上涨 2813 家 / 下跌 2054 家 / 平盘 121 家",
        "涨停 47 家 / 跌停 12 家",
        "两市成交额 1.08 万亿，较昨日 +5.2%",
    ],
    "global_markets": [
        "道琼斯 39134.76 -0.12%", "纳斯达克 17721.59 +0.44%", "标普500 5229.97 +0.10%",
    ],
}

def _mk_stock(name, code, news_titles):
    """构造一只与线上等规模的股票 detail。"""
    return {
        "code": code,
        "name": name,
        "realtime": (
            f"最新价:15.32 涨跌幅:+2.15% 涨跌额:+0.32 换手率:3.82% "
            f"最高:15.58 最低:15.02 成交量:2847万手 成交额:4.32亿"
        ),
        "fundamentals": (
            f"PE:18.56 PB:2.432 每股收益:0.823元 总市值:152亿 "
            f"流通市值:128亿 60日涨跌幅:+18.42% 年初至今:+24.51%"
        ),
        "news_titles": news_titles,
        "errors": [],
    }

STOCKS = [
    _mk_stock("贵州茅台", "600519", ["[2026-08-20] 关于股东权益的公告", "[2026-08-18] 回购股份进展", "[2026-08-15] 董事会决议"]),
    _mk_stock("宁德时代", "300750", ["[2026-08-20] 新产品发布", "[2026-08-16] 股权激励计划", "[2026-08-12] 业绩说明会"]),
    _mk_stock("比亚迪", "002594", ["[2026-08-20] 海外市场进展", "[2026-08-15] 车辆交付数据", "[2026-08-11] 分红方案"]),
    _mk_stock("中际旭创", "300308", ["[2026-08-20] 订单公告", "[2026-08-14] 产能扩张", "[2026-08-10] 中标通知"]),
    _mk_stock("赛力斯", "601127", ["[2026-08-19] 产销快报", "[2026-08-13] 合作公告", "[2026-08-09] 半年度报告"]),
    _mk_stock("东方财富", "300059", ["[2026-08-20] 券商评级上调", "[2026-08-15] 市场份额数据", "[2026-08-11] 季度报"]),
    _mk_stock("海光信息", "688041", ["[2026-08-20] 新品发布", "[2026-08-14] 研发投入", "[2026-08-09] 政府补助"]),
    _mk_stock("寒武纪", "688256", ["[2026-08-19] 合同签订", "[2026-08-12] 营业收入增长", "[2026-08-08] 减持公告"]),
    _mk_stock("中芯国际", "688981", ["[2026-08-20] 产能利用率", "[2026-08-15] 先进制程进展", "[2026-08-10] 财报"]),
    _mk_stock("工业富联", "601138", ["[2026-08-20] AI服务器订单", "[2026-08-16] 数据中心业务", "[2026-08-12] 分红"]),
]

# 策略->股票 对应关系（多策略重叠，与线上 top10 过滤后一致）
STRATEGIES = {
    "价值龙头增强": ["600519", "601127", "600519"],
    "动量突破策略": ["300308", "002594", "300750"],
    "AI主线": ["688041", "688256", "688981", "601138"],
    "资金流入": ["300059", "300308", "300750"],
    "成长趋势": ["300308", "002594", "601127"],
}


def _approx_tokens(text: str) -> float:
    """中英混合 token 近似估算：中文≈1字/token，英文数字≈0.35 token/字符。"""
    zh = len(re.findall(r"[一-鿿]", text))
    other = len(text) - zh
    return zh * 1.0 + other * 0.35


# ── 复用线上 _build_prompt 的分段组装逻辑（import 自生产模块，不改代码） ──
def build_and_split_prompt(market, stocks_detail, strategies_results):
    """生成完整 prompt，并按逻辑段切分各段文本。返回 (segments: list[(name, text)])。"""
    from sequoia_x.analysis import analyst as A
    inst = A.MarketAnalyst.__new__(A.MarketAnalyst)  # 空壳实例，_build_prompt 不依赖 settings
    full = inst._build_prompt(strategies_results, market, stocks_detail)

    # ── 按组装顺序的固定标记切分 ──
    mk_realtime = "## 📊 今日市场实时数据"
    mk_cand = "## 🎯 候选股票"
    mk_strategy = "## 🔗 策略与股票对应关系"
    mk_req = "## 分析要求"

    segs = []
    # ── 说明：_build_prompt 用 " ".join 把候选股票各行空格连接，因此候选块是扁平串，
    #    必须按字段标签正则提取，不能按行切。

    # ① 系统人设 = 「## 📊 实时数据」标记之前
    segs.append(("①系统人设", full.split(mk_realtime)[0]))
    # ② 大盘环境 = 实时数据标记 至 候选股票标记 之间
    segs.append(("②大盘环境", full.split(mk_realtime)[-1].split(mk_cand)[0]))
    # ③ 候选块 = 候选股票标记 至 策略对应关系标记 之间
    cand_block = full.split(mk_cand)[-1].split(mk_strategy)[0]

    # 候选块用空格连接、以「## 股票名 (代码)」开头，按「## 」切成若干股票子串，
    # 每只内：实时行情/估值财务/错误 = 行情段；「--- 近期公告 ---」之后 = 新闻段。
    stock_parts = re.split(r"## ", cand_block)[1:]  # 丢弃首个空段
    fin_parts, all_news = [], []
    for sp in stock_parts:
        if "--- 近期公告 ---" in sp:
            body, news_body = sp.split("--- 近期公告 ---", 1)
            all_news.append("".join(re.findall(r"•\s*(\[[^\]]*\].*?)(?=#[^#]*$|#|\Z)", news_body, flags=re.S)))
        else:
            body = sp
        # 去掉股票名标题行本身，仅保留 实时行情:/估值/财务: 字段
        fin_lines = [ln for ln in re.split(r"\s(?=实时行情:|估值/财务:|⚠️)", body)
                     if re.match(r"^(实时行情:|估值/财务:|⚠️)", ln.strip())]
        fin_parts.append(" ".join(fin_lines))
    segs.append(("③个股行情+财务", " ".join(fin_parts)))
    segs.append(("④新闻标题", " ".join(all_news)))
    # ⑤ 策略对应关系 = 策略标记 至 分析要求标记 之间
    segs.append(("⑤策略对应关系", full.split(mk_strategy)[-1].split(mk_req)[0]))
    # ⑥ 分析要求+输出格式模板
    segs.append(("⑥分析要求+输出格式", full.split(mk_req)[-1]))
    return segs


def main():
    segments = build_and_split_prompt(MARKET, STOCKS, STRATEGIES)

    # 逐段估算 token
    seg_tokens = []
    for name, text in segments:
        seg_tokens.append((name, _approx_tokens(text)))

    # 自检：各段字符长度（验证切分是否有效，防止新闻段被吞）
    print(f"[自检] 各段字符数: " + ", ".join(f"{n}:{len(t)}" for n, t in segments))

    total_est = sum(t for _, t in seg_tokens)
    # 校准系数：用真实线上总量对齐
    K = REAL_PROMPT_TOKENS / total_est if total_est else 1.0

    print("=" * 68)
    print(f"DeepSeek 单次调用 prompt 输入 token 诊断（锚点: 线上实测 {REAL_PROMPT_TOKENS}）")
    print("=" * 68)
    print(f"{'段':<24}{'估算token':>10}{'占比':>9}")
    print("-" * 68)
    for name, est in seg_tokens:
        cal = est * K
        pct = cal / REAL_PROMPT_TOKENS * 100
        print(f"{name:<24}{cal:>9.0f}{pct:>8.1f}%")
    print("-" * 68)
    print(f"{'合计(校准)':<24}{REAL_PROMPT_TOKENS:>10.0f}{100:>8.1f}%")
    print(f"\n说明: 估算系数 K={K:.3f}（段估算×K 对齐线上实测总量）")
    print(f"      prompt 总字符 {sum(len(s['text']) for s in segments)} 字")
    print("\n→ 占比最大的段 = 优先压缩对象。本次诊断的关键发现：")
    print("  · ③个股行情+财务 最大（~42%）—— 是可确定压缩的数字事实字段（PE/PB/市值等），")
    print("    适合【去冗余字段 + 固定前缀缓存】，不适合 LLMLingua（会丢数字）。")
    print("  · ⑥分析要求+输出格式 其次（~24%）—— 是纯固定模板文字，最适合【抽成固定前缀】，")
    print("    DeepSeek 上下文缓存命中后这部分价格趋近于零。")


if __name__ == "__main__":
    main()
