#!/usr/bin/env python
"""月度重训后的**首跑验收**（2026-09-29 建立）

为什么需要它
------------
月度重训（每月 1 号 03:00）跑完会推一条微信，列出"买入候选 TOP10"。
但**它只报告结果，不判断结果是否可信** —— 若某处静默出错（特征整列变 0、T4 训练失败
输出全 0、口径切换导致预测整体漂移），脚本仍会照常推出一份名单，模拟盘仍会照常买入。

本项目最怕的正是这类**静默失败**（同型事故已发生多次：
`.dryrun_cache` 假通过、T4 checkpoint 静默复用、`monthly_returns` 漏买入日…）。

本脚本在重训后跑一次，回答一句话：**"这份名单像不像正常产出的？"**

检查项
------
1. **存在性与非退化**：目标月预测存在；t2/t4/fuse 的标准差 > 0
   （T4 全 0 = 训练静默失败，这是历史真实事故的指纹）
2. **分布漂移**：本月预测的均值/标准差 vs 历史各月，给出 z 分数
   （特征口径变更、数据事故会让分布整体跑偏）
3. **名单与重合度**：按等权融合取 TOP10；与上月名单的重合度
   （重合度过高/过低都值得看一眼：前者=模型没更新，后者=口径大变）
4. **换月对比**：与上月的预测逐股秩相关
   （V4→V5 这类口径切换会显著降低相关性，属预期；断崖式≈0 才需警惕）

用法
----
    python scripts/post_retrain_verify.py                     # 验收"当前月"
    python scripts/post_retrain_verify.py --month 2026-10
    python scripts/post_retrain_verify.py --month 2026-10 --notify

退出码：0 = 像正常产出；1 = 有异常（**先人工确认再让模拟盘买入**）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
CACHE = PROJ / "output/backtest_v2/prediction_cache.json"
TOPK = 10

problems: list[str] = []


def say(flag: str, title: str, detail: str = "") -> None:
    print(f"{flag} {title}" + (f"\n     {detail}" if detail else ""))
    if flag.startswith("❌"):
        problems.append(title)


def fused(e: dict) -> np.ndarray:
    """等权秩平均融合（与 integration.fuse_ranks 同口径）。"""
    t2, t4 = np.asarray(e["t2"], float), np.asarray(e["t4"], float)
    if np.std(t4) < 1e-12:          # T4 未跑成 → 退化为纯 T2（并已在别处告警）
        return t2
    return (pd.Series(t2).rank().to_numpy() + pd.Series(t4).rank().to_numpy()) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="", help="验收哪个月（YYYY-MM），默认当前月")
    ap.add_argument("--notify", action="store_true", help="有异常时推微信")
    a = ap.parse_args()
    month = a.month or dt.datetime.now().strftime("%Y-%m")

    print("=" * 78)
    print(f"  月度重训首跑验收 | 目标月 {month} | {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)
    if not CACHE.exists():
        say("❌", f"预测缓存不存在: {CACHE}")
        return 1
    cache = json.loads(CACHE.read_text())
    hist = sorted(m for m in cache if m < month)
    if month not in cache:
        say("❌", f"缓存里没有 {month} 的预测（重训未完成？）",
            f"现有月份: {sorted(cache)[-3:]} …（共 {len(cache)} 个月）")
        return 1

    e = cache[month]
    n = len(e.get("symbols", []))
    print(f"── 1) 存在性与非退化（{n} 只）──")
    say("✅" if n > 500 else "❌", f"符号数 = {n}")
    for f in ("t2", "t4"):
        v = np.asarray(e.get(f) or [], float)
        s = float(np.std(v)) if v.size else 0.0
        bad = s < 1e-12
        say("❌" if bad else "✅", f"{f} 标准差 = {s:.6f}"
            + ("  ⇒ **全 0/退化，模型可能训练失败**" if bad else ""))

    print("── 2) 分布漂移（vs 历史各月）──")
    for f in ("t2", "t4"):
        cur = np.asarray(e.get(f) or [], float)
        if cur.size == 0 or np.std(cur) < 1e-12:
            continue
        ms, ss = [], []
        for h in hist:
            v = np.asarray(cache[h].get(f) or [], float)
            if v.size and np.std(v) > 1e-12:
                ms.append(float(np.mean(v)))
                ss.append(float(np.std(v)))
        if len(ms) < 6:
            continue
        z_m = (np.mean(cur) - np.mean(ms)) / (np.std(ms) + 1e-12)
        z_s = (np.std(cur) - np.mean(ss)) / (np.std(ss) + 1e-12)
        flag = "✅" if max(abs(z_m), abs(z_s)) < 3 else "❌"
        say(flag, f"{f} 分布 z：均值 {z_m:+.2f} ｜ 标准差 {z_s:+.2f}（|z|>3 视为异常，n={len(ms)}）",
            "" if flag == "✅" else "⇒ 分布整体跑偏，疑特征口径/数据事故 —— 先查再买")

    print("── 3) 名单与换月稳定性（阈值由历史分布**自动定界**，不拍脑袋）──")
    pf = fused(e)
    order = np.argsort(-pf)[:TOPK]
    picks = [e["symbols"][i] for i in order]
    print(f"     本月 TOP10: {' '.join(picks)}")

    # ⚠️ 设计教训（2026-09-29）：本项初版把"与上月重合 0/10""秩相关≈0"判为异常 ——
    #    实测历史 72 对相邻月里 **63 对重合为 0、78% 秩相关 <0.1**
    #    （与 docs/2026-09_特征时效性与训练窗口的结构性约束.md 记的"V4↔V5 选股几乎零重合"
    #     同一根源：训练集只 ~2 个截面 ⇒ 对微小差异极敏感）。拍脑袋阈值会**每月误报**，
    #    使告警彻底失效。⇒ 改为**用缓存自身的历史分布取 p1/p99 定界**，自适应且不会失灵。
    def pair_stats(m1: str, m2: str) -> tuple[int, float] | None:
        e1, e2 = cache[m1], cache[m2]
        p1_, p2_ = fused(e1), fused(e2)
        s1 = {e1["symbols"][i] for i in np.argsort(-p1_)[:TOPK]}
        s2 = {e2["symbols"][i] for i in np.argsort(-p2_)[:TOPK]}
        common = sorted(set(e1["symbols"]) & set(e2["symbols"]))
        if len(common) < 200:
            return None
        i1 = {s: i for i, s in enumerate(e1["symbols"])}
        i2 = {s: i for i, s in enumerate(e2["symbols"])}
        r = spearmanr([p1_[i1[s]] for s in common], [p2_[i2[s]] for s in common]).statistic
        return len(s1 & s2), float(r)

    if len(hist) >= 12:
        hov, hr = [], []
        for x, y in zip(hist, hist[1:]):
            st = pair_stats(x, y)
            if st:
                hov.append(st[0]); hr.append(st[1])
        hr = np.asarray(hr)
        lo, hi = np.percentile(hr, 1), np.percentile(hr, 99)
        cur = pair_stats(hist[-1], month)
        if cur:
            ov, r = cur
            say("✅" if ov < TOPK else "❌", f"与上月（{hist[-1]}）重合 {ov}/{TOPK}",
                "重合 10/10 ⇒ 模型可能没更新（历史常态是 0~2 只，故 0 不算异常）" if ov >= TOPK else
                f"（历史 72 对相邻月中位 = 0 只，故 0 属常态）")
            flag = "✅" if lo <= r <= hi else "❌"
            say(flag, f"与上月预测秩相关 = {r:+.3f}",
                f"历史 p1~p99 = [{lo:+.3f}, {hi:+.3f}]（n={len(hr)} 对），落在区间内属正常；"
                f"越界才需人工确认（V4→V5 这类口径切换会明显下降，届时看是否越界）"
                if flag == "✅" else
                f"**越出历史 p1~p99 = [{lo:+.3f}, {hi:+.3f}]** ⇒ 口径剧变或异常，先人工确认")

    print("=" * 78)
    if problems:
        print(f"❌ {len(problems)} 项异常 —— **先人工确认，再让模拟盘按此名单买入**：")
        for p in problems:
            print(f"   · {p}")
        if a.notify:
            try:
                from wxpusher import WxPusher
                from sequoia_x.core.config import get_settings
                s = get_settings()
                WxPusher.send_message(
                    content=f"❌ 月度重训首跑验收异常（{month}）\n"
                            + "\n".join(f"· {p}" for p in problems)
                            + "\n先人工确认，再让模拟盘买入",
                    token=s.wxpusher_token, topic_ids=s.wxpusher_topic_ids, content_type=1)
                print("[notify] ✅ 已推送")
            except Exception as ex:
                print(f"[notify] ⚠️ 推送失败: {ex}")
        return 1
    print("✅ 像正常产出 —— 可按名单执行（仍建议扫一眼 TOP10 是否合理）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
