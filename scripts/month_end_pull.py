#!/usr/bin/env python3
"""月末扩展维度提前拉取 — cron 每天 19:00 触发, 非月末最后交易日零成本退出

设计（用户 2026-08-07 确认）:
  - 避开 18:10-18:45 OHLCV 日线同步窗口（接口当日数据未就绪 + 资源冲突）
  - 19:00 开始, 有 24-48h 缓冲窗口吸收封禁/限频失败（1 号 0 点重训前从容补采）
  - 判断逻辑: 今天 = 本月最后交易日才执行（用交易日历, 非简单日期）
  - 与 1 号 Step0 互补: 月末强制全量刷新(--refresh-days 0), 1号只补缺失+failed清单

2026-08-07 月末自动链（用户要求"不能人工启动"）:
  拉取完成 → 覆盖率检查(parquet 文件计数) → 训练缓存自动重建(121/88+80 维, ~2-6h)
  → 自检(维度+采样日覆盖) → 单月干跑验证(build_prediction_cache, 临时输出)
  → 微信推送完成/失败; 9/1 00:00 v2_monthly_retrain 轮询等待缓存就绪

用法(cron):
  0 19 * * 1-5 cd <project> && py312 python scripts/month_end_pull.py >> logs/month_end_pull_$(date +%Y%m).log 2>&1
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 交易日历来源（akshare 新浪, 免费）; 失败时回退: 周一~周五直接放行（近似）
def get_trade_dates():
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    return set(df["trade_date"].astype(str).tolist())


def is_last_trade_day(today: date, trade_dates: set) -> bool:
    """今天是否本月最后交易日"""
    if today.strftime("%Y-%m-%d") not in trade_dates:
        return False  # 今天不是交易日
    month = today.strftime("%Y-%m")
    month_dates = sorted(d for d in trade_dates if d.startswith(month))
    return month_dates and month_dates[-1] == today.strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════
#  月末自动链（2026-08-07, 用户要求全自动无人值守）
# ════════════════════════════════════════════════════════════

def _notify(title: str, body: str) -> None:
    """wxpusher 微信推送（完成/失败告警, 不阻断流程）。"""
    try:
        from wxpusher import WxPusher
        from sequoia_x.core.config import get_settings
        s = get_settings()
        WxPusher.send_message(content=f"{title}\n{body}", token=s.wxpusher_token,
                              topic_ids=s.wxpusher_topic_ids, content_type=1)
        print(f"[notify] ✅ 已推送: {title}")
    except Exception as e:
        print(f"[notify] ⚠️ 推送失败: {e}")


def _extra_coverage_ok() -> tuple[bool, str]:
    """覆盖率检查: 7 类 parquet 文件数 / 5206 ≥ 90%（文件计数, 含存量+本次; 非 manifest 单次计数）。"""
    total = 5206
    low = []
    for subset in ("fund_flow", "finance", "holders", "consensus", "news", "xdxr", "forecast"):
        d = PROJECT_DIR / "data/extra_features" / subset
        n = len(list(d.glob("*.parquet"))) if d.exists() else 0
        cov = n / total * 100
        if cov < 90:
            low.append(f"{subset}={cov:.0f}%({n})")
    if low:
        return False, "覆盖率不足: " + ", ".join(low)
    return True, "7 类全部 ≥90%"


def _verify_caches() -> tuple[bool, str]:
    """自检: 训练缓存存在 + 维度正确(121/88 树模型, 80 T4) + 采样日覆盖到月末。"""
    import json as _json
    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.labels import _dataset_cache_path, resolve_sample_end
    cfg = get_config()
    cfg.sample_end = resolve_sample_end(cfg)  # DB 最后交易日, 与重建/重训同口径
    symbols = _json.loads((PROJECT_DIR / "output/backtest_v2/.stock_pool.json").read_text())
    include_extra = bool(getattr(cfg, "extra_features", False))
    exp_tree = 121 if include_extra else 88
    msgs, bad = [], []
    for name, ms, extra, want in [("树模型", True, include_extra, exp_tree),
                                  ("T4", False, False, 80)]:
        d, _ = _dataset_cache_path(cfg, symbols, ms, extra)
        if not (d / "metadata.json").exists():
            bad.append(f"{name}缓存缺失:{d.name}")
            continue
        m = _json.loads((d / "metadata.json").read_text())
        xdim = m["X_shape"][2]
        if xdim != want:
            bad.append(f"{name}维度错误:{xdim}≠{want}")
            continue
        dates = _json.loads((d / "dates.json").read_text())
        msgs.append(f"{name}={xdim}维,{len(set(dates))}日,止于{dates[-1]}")
    return (not bad), "; ".join(msgs + bad)


def auto_rebuild_and_verify(today: date) -> bool:
    """自动链: 覆盖率检查 → 缓存重建 → 自检 → 单月干跑验证。

    Returns: True=全链通过（9/1 重训可直接运行）; False=某环失败（已微信告警）。
    """
    print(f"[{today}] ═══ 自动链开始 ═══")

    # 1. 覆盖率检查（坏数据不入缓存）
    print(f"[{today}] ① 覆盖率检查...")
    ok, msg = _extra_coverage_ok()
    if not ok:
        _notify("⚠️ 月末自动链中止: " + msg,
                "9/1 00:00 重训将轮询等待缓存就绪（12h 后失败告警）。请查看 logs/month_end_pull_*.log")
        print(f"[{today}] ❌ {msg}")
        return False
    print(f"[{today}] ✅ {msg}")

    # 2. 训练缓存重建（121/88 + 80 维并行, 2-6h; include_extra 从 config 读）
    print(f"[{today}] ② 重建训练数据集缓存（预计 2-6h, 期间 9/1 重训轮询等待）...")
    r = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts/rebuild_dataset_cache.py")],
                       cwd=str(PROJECT_DIR), timeout=10 * 3600)
    if r.returncode != 0:
        _notify("❌ 月末缓存重建失败", "9/1 重训将轮询等待后失败; 请查看 logs/month_end_pull_*.log 排查")
        print(f"[{today}] ❌ 缓存重建失败 exit={r.returncode}")
        return False
    print(f"[{today}] ✅ 缓存重建完成")

    # 3. 自检（维度 + 采样日覆盖）
    ok, msg = _verify_caches()
    if not ok:
        _notify("❌ 月末缓存自检失败", msg)
        print(f"[{today}] ❌ 自检失败: {msg}")
        return False
    print(f"[{today}] ✅ 自检: {msg}")

    # 4. 单月干跑验证（临时输出, 不污染生产 prediction_cache.json）
    month = today.strftime("%Y-%m")
    print(f"[{today}] ④ 单月干跑验证（{month}, build_prediction_cache, 约 30-60min）...")
    dry = PROJECT_DIR / "output/backtest_v2/.dryrun_cache.json"
    r = subprocess.run(
        [sys.executable, "-u", str(PROJECT_DIR / "scripts/build_prediction_cache.py"),
         "--start-month", month, "--end-month", month, "--skip-t4",
         "--output", str(dry)],
        cwd=str(PROJECT_DIR), timeout=4 * 3600)
    if r.returncode != 0 or not dry.exists():
        _notify("❌ 月末干跑验证失败", "121/88 维预测链路未验证通过, 9/1 重训前需人工排查")
        print(f"[{today}] ❌ 干跑验证失败 exit={r.returncode}")
        return False
    print(f"[{today}] ✅ 干跑验证通过")

    _notify("✅ 月末扩展维度全链完成",
            f"{month} 数据拉取 + 缓存重建 + 自检 + 干跑全部通过, 9/1 00:00 重训可直接运行")
    print(f"[{today}] ═══ 自动链全部完成 ═══")
    return True


def main():
    today = date.today()
    try:
        trade_dates = get_trade_dates()
        if not is_last_trade_day(today, trade_dates):
            print(f"[{today}] 非本月最后交易日, 跳过")
            return
    except Exception as e:
        # 日历接口失败 → 放行（宁可多拉也不漏拉, 断点续跑保证幂等）
        if today.weekday() >= 5:
            print(f"[{today}] 日历获取失败({e}) 且为周末, 跳过")
            return
        print(f"[{today}] 日历获取失败({e}), 回退放行(工作日)")
        pass

    print(f"[{today}] ★ 本月最后交易日, 启动扩展维度全量刷新")
    # 主采集: 7 类扩展维度（--refresh-days 0 = 强制全量; 2026-08-07 修复:
    #   原 40 天阈值 > 月末间隔 ~30 天 → 永远判定"新鲜"跳过 → 数据停留在筑基快照）
    cmd = [sys.executable, str(PROJECT_DIR / "scripts/collect_extra_features.py"),
           "--codes", str(PROJECT_DIR / "scripts/all_a_codes.txt"),
           "--data", "fund_flow,finance,holders,consensus,xdxr,news,forecast",
           "--refresh-days", "0"]
    r = subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=12 * 3600)
    print(f"[{today}] 扩展维度拉取结束, exit={r.returncode}")

    # 财联社快讯备源(独立脚本, 快讯流→按股票归档; 月末拉最近 3 天即当月新闻面补充)
    r2 = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts/pull_cls_news.py")],
                        cwd=str(PROJECT_DIR), timeout=3600)
    print(f"[{today}] 财联社快讯拉取结束, exit={r2.returncode}")

    if r.returncode != 0 or r2.returncode != 0:
        _notify("⚠️ 月末扩展维度拉取失败",
                f"collect exit={r.returncode}, news_cls exit={r2.returncode} → 自动链中止, 需人工介入")
        sys.exit(1)

    ok = auto_rebuild_and_verify(today)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
