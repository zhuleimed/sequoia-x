#!/usr/bin/env python3
"""月末扩展维度提前拉取 — cron 每天 19:00 触发, 非月末最后交易日零成本退出

设计（用户 2026-08-07 确认）:
  - 避开 18:10-18:45 OHLCV 日线同步窗口（接口当日数据未就绪 + 资源冲突）
  - 19:00 开始, 有 24-48h 缓冲窗口吸收封禁/限频失败（1 号 0 点重训前从容补采）
  - 判断逻辑: 今天 = 本月最后交易日才执行（用交易日历, 非简单日期）
  - 与 1 号 Step0 互补: 月末拉全量刷新(>40天), 1号只补缺失+failed清单

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

    print(f"[{today}] ★ 本月最后交易日, 启动扩展维度全量刷新(>40天)")
    # 主采集: 8 类扩展维度(含 forecast 业绩预告 + news_cls 由下方单独拉)
    cmd = [sys.executable, str(PROJECT_DIR / "scripts/collect_extra_features.py"),
           "--codes", str(PROJECT_DIR / "scripts/all_a_codes.txt"),
           "--data", "fund_flow,finance,holders,consensus,xdxr,news,forecast",
           "--refresh-days", "40"]
    r = subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=6 * 3600)
    print(f"[{today}] 扩展维度拉取结束, exit={r.returncode}")

    # 财联社快讯备源(独立脚本, 快讯流→按股票归档; 月末拉最近 3 天即当月新闻面补充)
    r2 = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts/pull_cls_news.py")],
                        cwd=str(PROJECT_DIR), timeout=3600)
    print(f"[{today}] 财联社快讯拉取结束, exit={r2.returncode}")
    sys.exit(r.returncode if r.returncode else r2.returncode)


if __name__ == "__main__":
    main()
