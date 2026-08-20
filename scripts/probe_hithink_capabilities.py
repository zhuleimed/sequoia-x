#!/usr/bin/env python3
"""零风险探测：同花顺接口能力与限流实证（只读，不动生产数据）。

针对用户三个核心疑问：
  1. ths 是否限流？限流阈值多高？（实测持续高频请求）
  2. 交易日历接口是否可用？（替代 baostock 的 is_trade_day 第 2 层）
  3. 财务接口能否支撑历史估值重建？（B2 路线可行性）
  4. 股票列表接口（替代 baostock get_active_stocks）
  5. 历史 K 线 / 除复权（替代 Tencent/Sina 后备）

用法：
  HITHINK_FINANCE_API_KEY=<key> py312 python scripts/probe_hithink_capabilities.py
只读不写。不打印 API Key。
"""
import json
import os
import sys
import time
import requests

BASE = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")


def req(path, params=None, retries=1):
    url = f"{BASE}{path}"
    t0 = time.time()
    try:
        r = requests.get(url, params=params, headers={"X-api-key": KEY}, timeout=20)
        dt = (time.time() - t0) * 1000
        try:
            return r.json(), r.status_code, dt
        except Exception:
            return {"_raw": r.text[:200]}, r.status_code, dt
    except Exception as e:
        return {"_err": str(e)}, None, (time.time() - t0) * 1000


def main():
    if len(KEY) < 8:
        print("❌ 缺 HITHINK_FINANCE_API_KEY"); sys.exit(1)
    print(f"Key: {KEY[:6]}...  (不打印完整)\n")

    # ── 1. 交易日历 ──
    print("[1] 交易日历 GET /api/a-share/calendar/trading-days")
    j, c, dt = req("/api/a-share/calendar/trading-days")
    if c == 200 and j.get("code") == 0:
        items = j.get("data", {}).get("item", [])
        print(f"    code=0  {len(items)} 个交易日  耗时{dt:.0f}ms")
        print(f"    首: {items[0]['date']}  尾: {items[-1]['date']}  (近一年窗口确认)")
    else:
        print(f"    HTTP={c} {json.dumps(j, ensure_ascii=False)[:200]}")

    # ── 2. 股票列表（含分页） ──
    print("\n[2] 股票列表 GET /api/meta/tickers/list?asset_type=a-share")
    j, c, dt = req("/api/meta/tickers/list", {"asset_type": "a-share", "limit": 1000, "offset": 0})
    if c == 200 and j.get("code") == 0:
        d = j.get("data", {})
        items = d.get("item", [])
        print(f"    code=0 total={d.get('total')} 返回{len(items)} 耗时{dt:.0f}ms")
        print(f"    样例: {[ (i.get('ticker'), i.get('name')) for i in items[:3] ]}")
    else:
        print(f"    HTTP={c} {json.dumps(j, ensure_ascii=False)[:200]}")

    # ── 3. 财务接口（历史区间取数，验证 B2 可行性） ──
    print("\n[3] 财务报表（历史区间, quarterly）GET /api/a-share/financials/income-statements")
    j, c, dt = req("/api/a-share/financials/income-statements",
                   {"thscode": "600519.SH", "period": "quarterly", "limit": 8})
    if c == 200 and j.get("code") == 0:
        items = j.get("data", {}).get("item", [])
        print(f"    code=0  {len(items)} 期  耗时{dt:.0f}ms")
        if items:
            f = items[0]
            print(f"    字段样例: {sorted(f.keys())[:14]}")
            print(f"    首期: report={f.get('report')}  net_profit={f.get('parent_holder_net_profit')}  eps={f.get('basic_eps')}")
    else:
        print(f"    HTTP={c} {json.dumps(j, ensure_ascii=False)[:200]}")

    # ── 4. 财务指标（含 PE 相关） ──
    print("\n[4] 财务指标 GET /api/a-share/financials/indicators")
    j, c, dt = req("/api/a-share/financials/indicators", {"thscode": "600519.SH", "report": "2025-1"})
    if c == 200 and j.get("code") == 0:
        ab = j.get("data", {}).get("abilities", [])
        n = sum(len(a.get("indicators", [])) for a in ab)
        print(f"    code=0 {len(ab)} 能力/共{n}指标 耗时{dt:.0f}ms")
        for a in ab:
            inds = [i.get("name") for i in a.get("indicators", [])][:8]
            print(f"      - {a.get('ability')}: {inds}{'...' if len(a.get('indicators',[]))>8 else ''}")
    else:
        print(f"    HTTP={c} {json.dumps(j, ensure_ascii=False)[:200]}")

    # ── 5. 历史 K 线（区间） ──
    print("\n[5] 历史 K 线 kline")
    j, c, dt = req("/api/a-share/kline/daily", {"thscode": "600519.SH", "start": "20250701", "end": "20250710"})
    if c == 200 and j.get("code") == 0:
        items = j.get("data", {}).get("item", [])
        print(f"    code=0 {len(items)} 根K 耗时{dt:.0f}ms 样例:{items[0] if items else None}")
    else:
        print(f"    HTTP={c} {json.dumps(j, ensure_ascii=False)[:150]}")

    # ── 6. 除复权因子 ──
    print("\n[6] 除复权/公司行动")
    j, c, dt = req("/api/a-share/market/factors", {"thscode": "600519.SH"})
    print(f"    HTTP={c} code={j.get('code') if isinstance(j,dict) else '?'} 耗时{dt:.0f}ms {json.dumps(j,ensure_ascii=False)[:120]}")

    # ── 7. ★限流实证：短时间高频请求，找 QPS 上限 ──
    print("\n[7] ★限流实证：分三档速率持续打估值快照接口，观察 4001")
    ok = c4001 = 0
    # 档位1：串行连发 30 次（无 sleep），看是否触发
    print("    档位A：连环 30 次（无间隔）")
    for i in range(30):
        j, c, dt = req("/api/a-share/valuations/snapshot", {"thscodes": "600519.SH"})
        if isinstance(j, dict) and j.get("code") == 0:
            ok += 1
        elif isinstance(j, dict) and j.get("code") == 4001:
            c4001 += 1
            print(f"      → 第{i+1}次触发 4001（QPS 上限）"); break
    time.sleep(1)
    print(f"    档位A结果: 成功{ok} 触发限流{c4001}")
    ok = c4001 = 0
    # 档位2：模拟月末全市场吞吐（100只/请求 × 52 批，每批前 sleep 0.1）
    print("    档位B：模拟月末吞吐（100只/批 × 20 批, 批间 0.1s）")
    db_codes = None
    try:
        import sqlite3
        conn = sqlite3.connect("data/sequoia_v2.db")
        db_codes = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM stock_daily LIMIT 2000").fetchall()]
        conn.close()
    except Exception as e:
        print(f"    (读DB失败:{e}, 用自造代码)")
    for i in range(20):
        if db_codes:
            batch = db_codes[i*100:(i+1)*100]
            codes = [f"{c}.SZ" if c.startswith(("0","3")) else f"{c}.SH" for c in batch]
        else:
            codes = [f"{i*100+k:06d}.SZ" for k in range(100)]
        j, c, dt = req("/api/a-share/valuations/snapshot", {"thscodes": ",".join(codes)})
        if isinstance(j, dict):
            if j.get("code") == 0:
                ok += 1
            elif j.get("code") in (3001,):
                ok += 1  # 假代码进不了表，但请求本身被处理=不算限流
            elif j.get("code") == 4001:
                c4001 += 1
            else:
                print(f"      → 批{i+1} 其他code={j.get('code')}")
        time.sleep(0.1)
    print(f"    档位B结果: 成功{ok}/20 触发限流{c4001}")

    print("\n✅ 探测完成（只读）")


if __name__ == "__main__":
    main()
