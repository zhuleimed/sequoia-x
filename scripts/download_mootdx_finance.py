#!/usr/bin/env python3
"""下载全部 mootdx（通达信）财务数据到 parquet 文件。

数据源：通达信 gpcwYYYYMMDD.zip（147 期，1988-2026）
存储：data/extra_features/mootdx_finance/gpcwYYYYMMDD.parquet
特性：断点续跑、进度日志、全零列检测

用法：
    python scripts/download_mootdx_finance.py              # 全部下载
    python scripts/download_mootdx_finance.py --dry-run     # 仅列出不下载
    python scripts/download_mootdx_finance.py --status      # 查看进度
    python scripts/download_mootdx_finance.py --limit 10    # 只下载最近 N 期
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "extra_features" / "mootdx_finance"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"

# 小于此值的文件视为空文件
MIN_FILE_SIZE = 500


def get_logger():
    """简易日志。"""
    import logging
    logger = logging.getLogger("mootdx_download")
    logger.setLevel(logging.DEBUG)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(
        LOG_DIR / f"mootdx_finance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# 列名去重
# ---------------------------------------------------------------------------
def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """处理重复列名：给重复列加 _2, _3 后缀。"""
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# 下载逻辑
# ---------------------------------------------------------------------------
def get_file_list():
    """从通达信获取全部财务文件列表。"""
    from mootdx.financial.financial import FinancialList

    fl = FinancialList()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    tmpfile.close()

    try:
        result = fl.content(downdir=tmpfile.name)
        result.close()

        with open(tmpfile.name, 'r', encoding='utf-8') as fp:
            content = fp.read()

        lines = content.strip().split('\n')
        files = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                files.append({
                    'filename': parts[0],
                    'hash': parts[1],
                    'filesize': int(parts[2]),
                })

        files.sort(key=lambda x: x['filename'])
        return files
    finally:
        os.unlink(tmpfile.name)


def download_and_parse(filename: str, tmpdir: str):
    """下载并解析一个财务数据文件 → DataFrame（index=code, 列名已去重）。"""
    from mootdx.financial.financial import Financial

    f = Financial()
    filepath = os.path.join(tmpdir, filename)

    # 检查缓存
    if os.path.exists(filepath) and os.path.getsize(filepath) >= MIN_FILE_SIZE:
        logger.debug(f"  使用缓存: {filename} ({os.path.getsize(filepath)} bytes)")
    else:
        if os.path.exists(filepath):
            os.remove(filepath)
        try:
            result = f.content(filename=filename, downdir=tmpdir, filesize=0)
            result.close()
        except Exception as e:
            logger.error(f"  下载失败: {filename} - {e}")
            return None

    if not os.path.exists(filepath) or os.path.getsize(filepath) < MIN_FILE_SIZE:
        logger.debug(f"  跳过空文件: {filename} ({os.path.getsize(filepath)} bytes)")
        return None

    # 解析
    try:
        with open(filepath, 'rb') as fp:
            data = f.parse(download_file=fp)
    except Exception as e:
        logger.error(f"  解析失败: {filename} - {e}")
        os.remove(filepath)
        return None

    if data is None or len(data) == 0:
        logger.warning(f"  解析结果为空: {filename}")
        return None

    # 转 DataFrame + 去重列名
    try:
        df = f.to_df(data, header='zh')
        df = _dedupe_columns(df)
    except Exception as e:
        logger.error(f"  to_df 失败: {filename} - {e}")
        return None

    return df


# ---------------------------------------------------------------------------
# 保存为 parquet
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, report_date: int):
    """将一期数据保存为 parquet 文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"gpcw{report_date}.parquet"

    # 清理 NaN 和 Inf（parquet 对极端值友好）
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()

    # 确保 report_date 列存在
    if 'report_date' not in df_clean.columns:
        df_clean['report_date'] = report_date

    df_clean.to_parquet(out_path, index=True, compression='zstd')
    file_size = os.path.getsize(out_path)
    return out_path, file_size


def get_completed_periods():
    """查询已下载的报告期列表（从 parquet 文件）。"""
    if not OUTPUT_DIR.exists():
        return set()
    completed = set()
    for f in OUTPUT_DIR.glob("gpcw*.parquet"):
        try:
            report_date = int(f.stem.replace('gpcw', ''))
            completed.add(report_date)
        except ValueError:
            pass
    return completed


# ---------------------------------------------------------------------------
# 清单管理
# ---------------------------------------------------------------------------
def save_manifest(files: list[dict], completed: set, zero_columns: dict):
    """保存下载清单。"""
    manifest = {
        'updated_at': datetime.now().isoformat(),
        'total_periods': len(files),
        'completed_periods': len(completed),
        'earliest_period': files[0]['filename'] if files else None,
        'latest_period': files[-1]['filename'] if files else None,
        'total_fields': 0,
        'zero_columns_count': len(zero_columns),
        'zero_columns': zero_columns,
        'download_size_bytes': sum(f['filesize'] for f in files),
        'periods': [
            {
                'filename': f['filename'],
                'report_date': int(f['filename'].replace('gpcw', '').replace('.zip', '')),
                'filesize': f['filesize'],
                'downloaded': int(f['filename'].replace('gpcw', '').replace('.zip', '')) in completed,
            }
            for f in files
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 列分析工具
# ---------------------------------------------------------------------------
def _guess_field_group(col_name: str) -> str:
    """根据列名猜测所属分组（用于分析报告）。"""
    name = col_name.lower()
    if any(kw in name for kw in ['每股', '净资产收益率', 'eps', '加权']):
        return '每股指标'
    if any(kw in name for kw in ['货币资金', '应收', '应付', '存货', '资产', '负债', '权益',
                                   '商誉', '固定资产', '流动比率', '速动比率', '现金比率',
                                   '产权', '周转', '天数']):
        return '资产负债表/营运'
    if any(kw in name for kw in ['营业收入', '营业成本', '销售费用', '管理费用',
                                   '财务费用', '利润', '净利润', '所得税', 'ebit']):
        return '利润表'
    if any(kw in name for kw in ['现金流', '折旧', '摊销']):
        return '现金流量表'
    if any(kw in name for kw in ['增长率', '同比']):
        return '成长能力'
    if any(kw in name for kw in ['利率', '比率', '利润率', '毛利率', '净利率', '报酬率']):
        return '财务比率'
    if any(kw in name for kw in ['股东', '机构', '持股', '流通']):
        return '股东/机构'
    if any(kw in name for kw in ['预告', '快报']):
        return '业绩预告/快报'
    if name.startswith('col'):
        return '未命名'
    return '其他'


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="下载 mootdx 全部财务数据到 parquet 文件")
    parser.add_argument("--dry-run", action="store_true", help="仅列出文件列表")
    parser.add_argument("--status", action="store_true", help="查看当前进度")
    parser.add_argument("--limit", type=int, default=0, help="只下载最近 N 期（0=全部）")
    parser.add_argument("--analyze", action="store_true", help="完成后输出列分析报告")
    args = parser.parse_args()

    # ---------- 导入 mootdx ----------
    try:
        from mootdx.financial.financial import FinancialList, Financial
    except ImportError:
        logger.error("mootdx 未安装！请先: pip install mootdx")
        sys.exit(1)

    # ---------- 获取文件列表 ----------
    logger.info("正在获取财务文件列表...")
    try:
        all_files = get_file_list()
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        sys.exit(1)

    # 过滤空文件
    files = [f for f in all_files if f['filesize'] >= MIN_FILE_SIZE]
    skipped_empty = len(all_files) - len(files)
    logger.info(
        f"获取到 {len(all_files)} 期（{all_files[0]['filename']} ~ {all_files[-1]['filename']}），"
        f"有效 {len(files)} 期，跳过 {skipped_empty} 个空文件"
    )

    if args.dry_run:
        total_size = sum(f['filesize'] for f in files)
        logger.info(f"总大小: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"最近5期: {[f['filename'] for f in files[-5:]]}")
        logger.info(f"最早5期: {[f['filename'] for f in files[:5]]}")
        return

    # ---------- 进度查询 ----------
    completed = get_completed_periods()
    if args.status:
        logger.info(f"下载进度: {len(completed)}/{len(files)} 期已完成")
        if completed:
            dates = sorted(completed)
            logger.info(f"  范围: {min(dates)} ~ {max(dates)}")
        recent = [f for f in files[-5:] if int(f['filename'].replace('gpcw', '').replace('.zip', '')) in completed]
        logger.info(f"  最近5期: {len(recent)}/5 已完成")
        return

    # ---------- 确定待下载列表 ----------
    pending = [
        f for f in files
        if int(f['filename'].replace('gpcw', '').replace('.zip', '')) not in completed
    ]

    if args.limit > 0:
        pending = pending[-args.limit:]

    if not pending:
        logger.info("全部已下载！")
        return

    total_size = sum(f['filesize'] for f in pending)
    logger.info(
        f"已完成 {len(completed)}/{len(files)} 期，"
        f"待下载 {len(pending)} 期，预计 {total_size / 1024 / 1024:.1f} MB"
    )

    # ---------- 临时下载目录 ----------
    tmpdir = os.path.join(tempfile.gettempdir(), 'mootdx_finance_download')
    os.makedirs(tmpdir, exist_ok=True)

    # ---------- 逐期下载 ----------
    start_time = time.time()
    success_count = 0
    fail_count = 0
    zero_counts = {}  # 列名 → 全零出现次数
    total_downloaded = 0

    for i, finfo in enumerate(pending):
        filename = finfo['filename']
        report_date = int(filename.replace('gpcw', '').replace('.zip', ''))

        # 进度信息
        elapsed = time.time() - start_time
        if i > 0 and success_count > 0:
            avg_time = elapsed / i
            eta_seconds = avg_time * (len(pending) - i)
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "计算中..."

        logger.info(
            f"[{i+1}/{len(pending)}] {filename} "
            f"({finfo['filesize']/1024:.0f}KB) "
            f"| ✓{success_count} ✗{fail_count} | ETA {eta_str}"
        )

        # 下载 + 解析
        t0 = time.time()
        df = download_and_parse(filename, tmpdir)
        if df is None:
            fail_count += 1
            logger.warning(f"  ✗ 失败/空文件，继续下一期")
            continue

        dt = time.time() - t0
        logger.info(f"  下载解析: {dt:.1f}s, shape={df.shape}")

        # 输出验证（铁律五）
        if df.shape[0] < 50:
            logger.warning(f"  ⚠ 仅 {df.shape[0]} 只股票，跳过")
            fail_count += 1
            continue

        nonzero_cols = sum(
            1 for c in df.columns
            if c not in ('code', 'report_date', 'symbol')
            and (df[c] != 0).any()
        )
        logger.info(f"  有效字段: {nonzero_cols}/{len(df.columns)}，股票数: {df.shape[0]}")

        # 保存 parquet
        t0 = time.time()
        try:
            out_path, file_size = save_parquet(df, report_date)
            dt2 = time.time() - t0
            logger.info(f"  写入: {out_path.name} ({file_size/1024:.0f}KB) in {dt2:.1f}s")
        except Exception as e:
            logger.error(f"  写入 parquet 失败: {e}")
            fail_count += 1
            continue

        # 统计全零列（跨期对比用）
        for c in df.columns:
            if c not in ('code', 'report_date', 'symbol'):
                if not df[c].any():
                    zero_counts[c] = zero_counts.get(c, 0) + 1

        success_count += 1
        completed.add(report_date)
        total_downloaded += 1

    # ---------- 完成 ----------
    total_elapsed = time.time() - start_time
    logger.info(f"{'='*60}")
    logger.info(f"下载完成！")
    logger.info(f"  成功: {success_count} 期")
    logger.info(f"  失败: {fail_count} 期")
    logger.info(f"  总耗时: {str(timedelta(seconds=int(total_elapsed)))}")
    logger.info(f"  数据目录: {OUTPUT_DIR}")

    # 全零列（在所有已下载期中全为零的比例 >95%）
    if total_downloaded > 0:
        always_zero = [
            (c, cnt) for c, cnt in sorted(zero_counts.items(),
                                           key=lambda x: -x[1])
            if cnt >= total_downloaded * 0.95
        ]
        if always_zero:
            logger.info(f"  全零列（≥95%期）: {len(always_zero)} 个")
            for c, cnt in always_zero[:30]:
                logger.info(f"    {c}: {cnt}/{total_downloaded} 期全零")

    # 保存清单
    save_manifest(all_files, completed,
                  [c for c, _ in zero_counts.items()])

    # 列分析报告
    if args.analyze and total_downloaded > 0:
        logger.info(f"\n{'='*60}")
        logger.info("列分析报告:")
        # 加载最新一期的 parquet 做分析
        latest_parquet = sorted(OUTPUT_DIR.glob("gpcw*.parquet"))[-1]
        df_latest = pd.read_parquet(latest_parquet)
        groups = {}
        for c in df_latest.columns:
            if c in ('code', 'report_date', 'symbol'):
                continue
            g = _guess_field_group(c)
            groups.setdefault(g, []).append(c)
        for g, cols in sorted(groups.items()):
            logger.info(f"  {g}: {len(cols)} 字段")
            if len(cols) <= 10:
                logger.info(f"    {cols}")

    # 清理旧缓存（保留最近10期）
    keep_files = {f['filename'] for f in files[-10:]}
    for fname in os.listdir(tmpdir):
        if fname not in keep_files:
            try:
                os.remove(os.path.join(tmpdir, fname))
            except OSError:
                pass

    logger.info(f"\n临时文件已清理（保留最近10期缓存）")
    logger.info(f"完成！")


if __name__ == '__main__':
    main()
