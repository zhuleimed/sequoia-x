"""V3b —— 空仓信号补充扫描。

## 为什么要补测

V3 矩阵的诊断发现：**最佳配置剩下的 −26.6% 最大回撤发生在 2019-04-19 ~ 2019-08-09，
整段 77 个交易日 S3 一次都没触发** —— 因为那次是**跟随大盘**的下跌，不是"微盘独立崩盘"。

⇒ 想再降回撤，必须在**另一类信号**上找（半仓、加慢趋势这两条已证伪）。

## 本脚本测什么

在最佳配置（30 只 / 等权 / 硬止损 −12%）上，逐个单独试：

| 信号 | 含义 | 直觉 |
|---|---|---|
| `S3` | 拥挤度（对照，V3 的最优） | 管"微盘独立崩盘" |
| `S1_MA250 / MA120 / MA60` | **组合净值跌破 N 日均线**就空仓 | 管"跟随大盘的下跌"（趋势跟踪） |
| `S4` | 组合已实现波动率过高就空仓 | 管"高波动期" |
| `S3orS1_MA120` | 两者**任一触发**就空仓 | 互补 |
| `S3orS4` | 两者任一触发 | 互补 |

用法：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python strategy_v3b_signals.py [db|ths]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import microcap_data as md
from strategy_v3_matrix import BASE, evaluate, empty_position, log, prepare

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "ths"
N, STOP = 30, -0.12

log("=" * 100)
log(f"V3b 空仓信号补充扫描    数据源: {SOURCE}    配置: {N} 只 / 等权 / 硬止损 −12%")
log("=" * 100)

rows = []
for ma in [250, 120, 60]:
    sel, port, sig, hits = prepare(N, "equal", STOP, sig_ma=ma)
    # 单独用趋势
    k = sig["S1_趋势"].shift(1).fillna(True).astype(float)
    rows.append(evaluate(port, sig, sel, k, hits, f"仅 S1 净值均线 MA{ma}", N, "等权", "−12%"))
    # 单独用波动率
    if ma == 250:
        k4 = sig["S4_波动率"].shift(1).fillna(True).astype(float)
        rows.append(evaluate(port, sig, sel, k4, hits, "仅 S4 波动率", N, "等权", "−12%"))
        k3 = empty_position(sig, "S3")
        rows.append(evaluate(port, sig, sel, k3, hits, "仅 S3 拥挤度", N, "等权", "−12%"))
        k_all = pd.Series(1.0, index=sig.index)      # ⚠️ 对照必须真的是"满仓"，不能复用 k3
        rows.append(evaluate(port, sig, sel, k_all, hits, "不空仓（对照）", N, "等权", "−12%"))
    # 组合：S3 或 趋势（任一触发即撤）
    k3 = empty_position(sig, "S3")
    kor = (k3 * sig["S1_趋势"].shift(1).fillna(True).astype(float)).clip(0, 1)
    rows.append(evaluate(port, sig, sel, kor, hits, f"S3 或 S1(MA{ma}) 任一即撤", N, "等权", "−12%"))
    if ma == 250:
        k4 = sig["S4_波动率"].shift(1).fillna(True).astype(float)
        kor4 = (k3 * k4)
        rows.append(evaluate(port, sig, sel, kor4, hits, "S3 或 S4 任一即撤", N, "等权", "−12%"))
    log(f"  MA{ma} 组跑完")

# 「不空仓」放到最后当基准线（其实没空仓，与 ma 无关，跑一次即可）
log("")
res = pd.DataFrame(rows)
res["_c"] = pd.to_numeric(res["Calmar"], errors="coerce")
res = res.sort_values("_c", ascending=False).drop(columns="_c")
log("=" * 100)
log("结果（越靠前越好）")
log("=" * 100)
log(res.to_string(index=False))
log("")
log("怎么读：")
log("  · 『在市』低 = 多数时间空仓 ⇒ 收益要看清是「在场赚的」还是「运气」（对比在市时间）。")
log("  · 重点看两类信号能不能**互补**：S3 管微盘独立崩盘、S1 管跟随大盘的下跌。")
