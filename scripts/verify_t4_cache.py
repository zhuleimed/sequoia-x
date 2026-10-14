#!/usr/bin/env python3
"""T4 checkpoint 缓存的**体检命令** —— 把"缓存该不该复用"从"等出事才发现"变成"随时可查"。

背景（2026-10-08 彻底方案，见 `docs/特征维度变更操作规程.md` §9.5）：
  待办 #11（"缓存 key 粒度不足"）年内复发至少 4 次，每次都是"发现一个轴 → 往文件名再加标签"。
  改为：checkpoint 名带 **spec 指纹**（`..._h{hash}.keras`），旁边落一份自描述的
  `{name}.spec.json`；加载前由 `deep_lstm._checkpoint_spec_ok()` 逐字段校验，不符则报警重训。

本脚本**只读**，做结构性体检：

  ✅ 自洽        模型 + spec 都在，且**文件名里的 hash == spec 内容算出的 hash**
  🔴 命名不符    文件名 hash 与 spec 内容对不上 ⇒ **文件被改过名/被替换**（最危险，加载时会被拒）
  ⚠️ 无 spec     只有模型没有 spec ⇒ 旧产物，**无法判定口径**，下次加载会被丢弃重训
  ⚠️ 孤立 spec   只有 spec 没有模型
  ⚠️ 重复        多份模型指向同一 spec（同一口径存了多份）

用法：
    python scripts/verify_t4_cache.py                     # 体检默认目录
    python scripts/verify_t4_cache.py --dir <模型目录>
    python scripts/verify_t4_cache.py --quiet             # 只打印汇总
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_DIR = "data/models/v2_selection"


def spec_hash_of(spec: dict) -> str:
    """与 deep_lstm._spec_fingerprint() 必须一致（同一算法，独立实现以便交叉验证）。"""
    return hashlib.md5(
        json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:12]


def hash_in_filename(name: str) -> str | None:
    """从 `..._h{hash}.keras` 里抽出 hash。"""
    stem = name[:-len(".keras")] if name.endswith(".keras") else name
    for part in reversed(stem.split("_")):
        if part.startswith("h") and len(part) == 13:
            return part[1:]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="T4 checkpoint 缓存体检（只读）")
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"模型目录（默认 {DEFAULT_DIR}）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总")
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.exists():
        print(f"❌ 目录不存在: {d}")
        return 2

    models = sorted(p for p in d.glob("t4_checkpoint_*.keras"))
    self_ok, bad_name, no_spec = [], [], []
    hash_groups = defaultdict(list)

    for m in models:
        sp = Path(str(m) + ".spec.json")
        if not sp.exists():
            no_spec.append(m)
            continue
        try:
            spec = json.loads(sp.read_text())
        except Exception as e:
            bad_name.append((m, f"spec 解析失败: {e}"))
            continue
        calc = spec_hash_of(spec)
        fname = hash_in_filename(m.name)
        if fname != calc:
            bad_name.append((m, f"文件名 hash={fname} ≠ spec 内容 hash={calc}"))
            continue
        self_ok.append(m)
        hash_groups[calc].append(m)

    specs_only = sorted(f for f in d.glob("t4_checkpoint_*.keras.spec.json")
                        if not Path(str(f)[:-len(".spec.json")]).exists())
    dups = {h: v for h, v in hash_groups.items() if len(v) > 1}

    print("=" * 72)
    print(f"T4 checkpoint 体检  目录={d}")
    print(f"时间 {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)
    print(f"  模型文件总数        : {len(models)}")
    print(f"  ✅ 自洽（名实相符） : {len(self_ok)}")
    print(f"  🔴 命名不符         : {len(bad_name)}")
    print(f"  ⚠️ 无 spec（旧产物）: {len(no_spec)}")
    print(f"  ⚠️ 孤立 spec        : {len(specs_only)}")
    print(f"  ⚠️ 重复口径         : {len(dups)} 组")

    if not a.quiet:
        if bad_name:
            print("\n🔴 命名不符（文件被改过名/替换 → 加载时会被拒并重训）：")
            for m, why in bad_name[:20]:
                print(f"    {m.name}\n        {why}")
            if len(bad_name) > 20:
                print(f"    …另有 {len(bad_name) - 20} 个")
        if no_spec:
            print(f"\n⚠️ 无 spec 的旧产物（前 10 个，共 {len(no_spec)}）：")
            for m in no_spec[:10]:
                print(f"    {m.name}")
            print("    处理建议（先确认再执行）：")
            print(f'      mkdir -p {d}/_t4_ckpt_no_spec_archive && \\')
            print(f'        mv {d}/t4_checkpoint_*.keras {d}/_t4_ckpt_no_spec_archive/ 2>/dev/null')
            print("      注：这些文件不会被新逻辑复用（名字里没有 spec 指纹），只会占磁盘。")
        if specs_only:
            print(f"\n⚠️ 孤立 spec（前 10 个，共 {len(specs_only)}）：")
            for f in specs_only[:10]:
                print(f"    {f.name}")
        if dups:
            print(f"\n⚠️ 重复口径（同一 spec 多份文件）：")
            for h, v in list(dups.items())[:10]:
                print(f"    h{h}: {len(v)} 份 → {[p.name for p in v[:3]]}")

    print("\n【怎么读】")
    print("  · **🔴 命名不符是唯一的危险信号** —— 说明文件名与内容对不上（被人改名/替换过）。")
    print("    新逻辑加载时会拒绝它并重训；若这个数字 > 0，请查是谁动了文件。")
    print("  · 无 spec / 孤立 spec 都只是「历史遗留占空间」，不影响正确性。")
    print("  · 本命令只读，不会删任何文件。")

    return 1 if bad_name else 0


if __name__ == "__main__":
    sys.exit(main())
