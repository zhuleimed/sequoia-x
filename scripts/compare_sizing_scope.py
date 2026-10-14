#!/usr/bin/env python3
"""口径敏感性对照：把【两个口径】的回测产物放一起比（待办 #47b 的常规动作）。

用法：
    python scripts/compare_sizing_scope.py <目录A> <目录B>
  例（同一臂的等权 vs 生产口径）：
    python scripts/compare_sizing_scope.py \
        output/backtest_v2/v6a_B_134  output/backtest_v2/v6a_B_134_prodsz

它回答一个问题：**"这次的结论，是不是由口径决定的？"**
  · 逐配置比「总收益率」：B 更好的配置数 / 差值中位 / 最大正负 3 例
  · 全部配置同向（赢 0/N 或 N/N）⇒ **结论由口径决定**，必须两套口径都报
  · 赢的比例接近一半 ⇒ 结论对口径大致不敏感

⚠️ 它**不能替代第二次回测**：分配方式不同就必须各跑一遍（本工具只把两套结果放一起看）。
   它的价值在于：① 产物自带口径烙印（sizing_manifest.json），不必翻日志反推；
                 ② 两套都在时，翻转一眼可见。
⚠️ 只看「总收益率」：它是最常被拿来下结论、也最容易被口径带偏的那个数（夏普可另看）。
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

KEY = ("风控模式", "TOP_N", "时段")


def load(d: str) -> tuple[dict, dict | None]:
    p = Path(d)
    f = p / "summary_all.csv"
    if not f.exists():
        raise SystemExit(f"❌ 缺 {f} —— 目录写错？或该次回测还没跑完？")
    rows = {(r["风控模式"], r["TOP_N"], r["时段"]): r for r in csv.DictReader(open(f))}
    mf = p / "sizing_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else None   # 老产物没有烙印 ⇒ None
    return rows, man


def scope_name(man: dict | None) -> str:
    if man is None:
        return "未烙印（早于 2026-10-11 的产物）"
    return f"{man.get('sizing_method')} {man.get('sizing_params') or ''}｜{man.get('scope')}"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    da, db = sys.argv[1], sys.argv[2]
    ra, ma = load(da)
    rb, mb = load(db)

    print("=" * 92)
    print("口径敏感性对照（总收益率）")
    print("=" * 92)
    print(f"  A/B 目录 : {da}   ｜   {db}")
    print(f"  A 口径   : {scope_name(ma)}   月份 {ma.get('months') if ma else '?'}")
    print(f"  B 口径   : {scope_name(mb)}   月份 {mb.get('months') if mb else '?'}")
    if ma and mb and ma.get("months") != mb.get("months"):
        print("  ⚠️ 两边月份区间不同 ⇒ **不可直接比**（先确认跑的是同一批月份）")

    keys = sorted(set(ra) & set(rb), key=lambda k: (k[2], k[1], k[0]))
    if not keys:
        raise SystemExit("❌ 两边没有可配对的配置（键 = 风控模式/TOP_N/时段）⇒ 无法对照")
    ds = [(k, float(rb[k]["总收益率"]) - float(ra[k]["总收益率"])) for k in keys]
    vals = [d for _, d in ds]
    nb = sum(1 for d in vals if d > 0)

    med = lambda rows: st.median([float(r["总收益率"]) for r in rows.values()])
    best = lambda rows: max(rows.items(), key=lambda kv: float(kv[1]["总收益率"]))[0]
    print(f"  配对配置 : {len(keys)} 组   ｜  A 中位 {med(ra):+.2%}   B 中位 {med(rb):+.2%}")
    print(f"  A 最优   : {best(ra)[2]} {best(ra)[0]}/TOP{best(ra)[1]}   ｜  "
          f"B 最优: {best(rb)[2]} {best(rb)[0]}/TOP{best(rb)[1]}")
    print()
    print(f"  ★ B − A（逐配置）：B 更好 {nb}/{len(vals)}   差值中位 {st.median(vals):+.2%}"
          f"   范围 {min(vals):+.2%} ~ {max(vals):+.2%}")
    m1 = [k for k in keys if k[0] == "M1"]
    if m1 and any(k[0] == "M0" for k in keys):   # 引擎里 M1≡M0 ⇒ 会把同一个配置数两遍
        print(f"    ⚠️ 其中 {len(m1)} 组是 M1 —— M1≡M0（T1 过滤按设计关闭，见待办 #25）"
              f"⇒ 属**重复计数**，独立配置数要减掉它们。")
    print()
    ordered = sorted(ds, key=lambda kv: -kv[1])            # 差值：从"B 更好"到"B 更差"
    def show(pairs):
        for k, d in pairs:
            print(f"    {d:+.2%}  {k[2]} {k[0]}/TOP{k[1]}")
    if len(ordered) > 6:
        print("  最大正向 3 例（B 明显更好）：")
        show(ordered[:3])
        print("    " + "-" * 40)
        print("  最大负向 3 例（B 明显更差）：")
        show(ordered[-3:])
    else:
        print("  全部配置（按差值从大到小）：")
        show(ordered)
    if len(ordered) < 5:
        print(f"  ⚠️ 只有 {len(ordered)} 组配置 ⇒ 下面的『口径影响』判定仅供参考")
    print()
    ratio = nb / len(vals)
    if ratio in (0.0, 1.0):
        verdict = "🔴 **结论由口径决定**（全部配置同向）⇒ 任何结论都必须同时报两套口径"
    elif ratio <= 0.25 or ratio >= 0.75:
        verdict = "🟡 口径影响很大（绝大多数配置同向）⇒ 结论要附口径说明"
    else:
        verdict = "🟢 结论对口径大致不敏感（输赢接近一半）"
    print(f"  判定：{verdict}")
    print()
    print("  ⚠️ 记住：这只是**点估计**的方向是否翻转；要不要下结论，仍须走按月成对显著性检验")
    print("     （块自助 CI 跨 0 = 分不出高下，本项目已 4 次撞同一面功效墙，见待办 #34/#45/#46/#47b）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
