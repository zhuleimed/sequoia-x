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


def main():
    """独立 wxpusher 推送（与生产 _notify 同源: get_settings() 加载 token/topic）。

    2026-08-20: 复用项目标准配置路径，而非手动 parse .env——
    与 month_end_pull._notify / v2_monthly_retrain 荐股推送同源一致。
    """
    if len(sys.argv) < 2:
        print("用法: notify_wechat.py '<消息>'")
        sys.exit(2)
    msg = sys.argv[1]
    try:
        from sequoia_x.core.config import get_settings
        from wxpusher import WxPusher
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)
    try:
        s = get_settings()
        resp = WxPusher.send_message(
            content=msg,
            token=s.wxpusher_token,
            topic_ids=s.wxpusher_topic_ids,
            summary="V4迁移进度",
            content_type=1,
        )
    except Exception as e:
        print(f"⚠️ 推送失败(不阻断): {e}")
        sys.exit(1)
    ok = resp.get("code") == 1000
    print(f"{'✅' if ok else '⚠️'} 微信推送 code={resp.get('code')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
