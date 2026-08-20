#!/usr/bin/env python3
"""探测同花顺估值快照接口，并与 DB 内 baostock 历史值对比，验证字段口径能否对齐。

目标（决策前置验证，不动任何生产数据）：
  1. 接口连通性与鉴权（X-api-key）
  2. 返回的 pe_ttm / pb_mrq / ps_ttm / pcf_ttm 与 stock_daily.peTTM 等历史值口径是否对得上
  3. 与 TDX/mootdx 自算口径的差异
  4. 批量请求规模（100/token 上限）与限流行为

用法：
  HITHINK_FINANCE_API_KEY=<key> py312 python scripts/probe_hithink_valuation.py
只读不写，纯探测。不打印 API Key。
"""
import json
import os
import sys
import time
import sqlite3

import requests

BASE = "https://fuyao.aicubes.cn"
KEY = os.environ.get("HITHINK_FINANCE_API_KEY", "")

# 测试股票：茅台(盈利) / 平安银行(低PB) / 万科(亏损负PE) / 工商银行(银行)
TEST = ["600519.SH", "000001.SZ", "000002.SZ", "601398.SH"]


def req(path, params):
    url = f"{BASE}{path}"
    t0 = time.time()
    r = requests.get(url, params=params, headers={"X-api-key": KEY},
                     timeout=20)
    dt = (time.time() - t0) * 1000
    return r, dt


def main():
    if not KEY or len(KEY) < 8:
        print("❌ 缺少 HITHINK_FINANCE_API_KEY 环境变量")
        sys.exit(1)
    print(f"Key 前缀: {KEY[:6]}...（不打印完整 Key）")

    # ── 1. 连通性：估值快照 ──
    r, dt = req("/api/a-share/valuations/snapshot",
                {"thscodes": ",".join(TEST)})
    print(f"\n[1] GET /valuations/snapshot HTTP={r.status_code} 耗时={dt:.0f}ms")
    try:
        j = r.json()
    except Exception:
        print("   非 JSON 响应:", r.text[:300])
        return
    print(f"   信封: code={j.get('code')} message={j.get('message')}")
    if j.get("code") != 0:
        print("   ❌ 鉴权/业务失败，停止。")
        print("   原始:", json.dumps(j, ensure_ascii=False)[:500])
        return

    data = j.get("data", {})
    ts = data.get("timestamp")
    print(f"   data.timestamp(ms)={ts}  ->  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts/1000)) if ts else 'null'}")
    print("   total=", data.get("total"))

    # ── 2. 与 DB baostock 最新历史值对比 ──
    db = os.environ.get("DB_PATH", "data/sequoia_v2.db")
    conn = sqlite3.connect(db)
    print("\n[2] 口径对比：同花顺快照 vs DB baostock（取 DB 最新非空行）")
    print(f"{'代码':<10} {'字段':<10} {'同花顺':>12} {'baostock(最新)':>16} {'差异':>10}")
    print("-" * 60)
    for it in data.get("item", []):
        ticker = it["ticker"]
        row = conn.execute(
            "SELECT date,peTTM,pbMRQ,psTTM,pcfNcfTTM FROM stock_daily "
            "WHERE symbol=? AND peTTM IS NOT NULL ORDER BY date DESC LIMIT 1",
            (ticker,)).fetchone()
        bs_date = row[0] if row else "无"
        pe_b, pb_b, ps_b, pcf_b = (row[1], row[2], row[3], row[4]) if row else (None,)*4
        print(f"\n  {ticker}（对 D日{bs_date}）")
        for src_name, h_val, b_val in [
                ("pe_ttm ", it.get("pe_ttm"),  pe_b),
                ("pe_mrq ", it.get("pe_mrq"),  None),
                ("pb_mrq ", it.get("pb_mrq"),  pb_b),
                ("ps_ttm ", it.get("ps_ttm"),  ps_b),
                ("pcf_ttm", it.get("pcf_ttm"), pcf_b),
        ]:
            diff = ""
            if h_val is not None and b_val not in (None, 0.0):
                diff = f"{(h_val-b_val)/b_val*100:+.1f}%"
            h_s = f"{h_val:.4f}" if isinstance(h_val, (int, float)) else str(h_val)
            b_s = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else str(b_val)
            print(f"    {src_name:<10} {h_s:>12} {b_s:>16} {diff:>10}")

    # ── 3. 批量规模测试：一次带 120 个 token，验证 100 上限 ──
    print("\n[3] 上限测试：一次请求 120 个 token（服务端说默认上限 100）")
    many = [f"{i:06d}.SZ" for i in range(1, 121)]  # 000001..000120
    r2, dt2 = req("/api/a-share/valuations/snapshot", {"thscodes": ",".join(many)})
    try:
        j2 = r2.json()
        print(f"    HTTP={r2.status_code} code={j2.get('code')} message={j2.get('message')} "
              f"-> {len(j2.get('data',{}).get('item',[]))} 条 耗时={dt2:.0f}ms")
    except Exception as e:
        print("   解析失败:", r2.text[:200], e)

    # ── 4. 限流粗测：连续 3 次真实股票请求观察是否被拒 ──
    # 用 DB 里的真实代码（前 300 个不重复），避免 3001 假阳性
    db_codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol LIMIT 300").fetchall()]
    if db_codes:
        print("\n[4] 连续 3 次请求测限流（用 DB 真实代码，各批 ～100 个）")
        step = 100
        for k in range(3):
            batch = db_codes[k*step:(k+1)*step]
            codes = [f"{c}.SZ" if c.startswith(("0", "3")) else f"{c}.SH" for c in batch]
            r3, dt3 = req("/api/a-share/valuations/snapshot", {"thscodes": ",".join(codes)})
            try:
                j3 = r3.json()
                n = len(j3.get("data", {}).get("item", [])) if j3.get("data") else -1
                print(f"    第{k+1}次: HTTP={r3.status_code} code={j3.get('code')} 返回{n}/{len(batch)}条 耗时={dt3:.0f}ms")
            except Exception as e:
                print(f"    第{k+1}次: HTTP={r3.status_code} 解析失败 {r3.text[:150]} {e}")
            time.sleep(0.4)
    else:
        print("\n[4] DB 无股票代码，跳过限流测试")

    conn.close()
    print("\n✅ 探测完成（只读，未写任何数据）")


if __name__ == "__main__":
    main()
