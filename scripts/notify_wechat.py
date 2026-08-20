#!/usr/bin/env python3
"""独立微信推送脚本：经 WxPusher 推送消息到手机微信（不依赖 daemon auto-push）。

2026-08-20: 用于 V4 后台迁移链结束时推送结果到用户手机。
Token/Topic 从 .env 读取（与项目 WxPusher 同一配置）。

用法：
  py312 python scripts/notify_wechat.py "<消息内容>"
"""
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def load_env(file: Path) -> None:
    """极简 .env 解析（避免依赖 dotenv）。"""
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    if len(sys.argv) < 2:
        print("用法: notify_wechat.py '<消息>'")
        sys.exit(2)
    msg = sys.argv[1]
    load_env(PROJ / ".env")
    token = os.environ.get("WXPUSHER_TOKEN", "")
    topic_ids = os.environ.get("WXPUSHER_TOPIC_IDS", "[]")
    if not token:
        print("❌ 缺 WXPUSHER_TOKEN (.env)")
        sys.exit(1)
    try:
        import json
        topic_ids_l = json.loads(topic_ids)
    except Exception:
        topic_ids_l = [39277]
    from wxpusher import WxPusher
    resp = WxPusher.send_message(
        content=msg,
        token=token,
        topic_ids=topic_ids_l,
        summary="V4迁移进度",
        content_type=1,
    )
    ok = resp.get("code") == 1000
    print(f"{'✅' if ok else '⚠️'} 微信推送 {resp}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
