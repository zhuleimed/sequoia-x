#!/usr/bin/env python3
"""mootdx 免费数据源客户端 — 已修复内置服务器失效问题

⚠️ 2026-08-07 实测：mootdx 内置 HQ_HOSTS（银河系）K线已失效（仅 finance 可用），
   必须手动指定有效服务器。本文件内置实测可达的服务器列表，自动切换。

用法:
    from mootdx_client import get_client
    client = get_client()          # 自动挑选可达服务器
    client.bars(symbol='600519', frequency=9, offset=800)   # 日K
    client.quotes(symbol='600519')                          # 实时行情
    client.finance(symbol='600519')                         # 财务快照
    client.xdxr(symbol='600519')                            # 除权除息
"""
import logging

from mootdx.quotes import Quotes

logger = logging.getLogger(__name__)

# 实测可达的行情服务器（2026-08-07 验证 K 线可用）
SERVER_POOL = [
    ("180.153.18.170", 7709),  # 上海电信主站Z1
    ("180.153.18.172", 80),    # 上海电信主站Z80
    ("202.108.253.139", 80),   # 北京联通主站Z80
    ("60.191.117.167", 7709),  # 杭州电信主站J1
    ("115.238.56.198", 7709),  # 杭州电信主站J2
]

# 财务服务器（finance/xdxr 专用，7709 群内实测可用）
FIN_SERVER = ("110.41.147.114", 7709)  # 深圳双线主站1


def _try_server(addr, port, market='std'):
    """连接并验证 K 线可用性"""
    try:
        c = Quotes.factory(market=market, server=(addr, port), timeout=8)
        bars = c.bars(symbol='600519', frequency=9, offset=1)
        if bars is not None and len(bars) > 0:
            return c
        c.close()
    except Exception as e:
        logger.warning(f"服务器 {addr}:{port} 验证失败: {e}")
    return None


def get_client(market='std'):
    """获取可用的 mootdx 客户端（自动在服务器池中挑选）。

    Returns:
        Quotes 实例；全部不可达时返回 None。
    """
    # 1. 优先逐台验证 K 线服务器
    for addr, port in SERVER_POOL:
        client = _try_server(addr, port, market)
        if client:
            logger.info(f"使用行情服务器: {addr}:{port}")
            return client
    # 2. 兜底：财务服务器（finance 可用但 K 线可能空）
    try:
        c = Quotes.factory(market=market, server=FIN_SERVER, timeout=8)
        logger.info(f"使用财务服务器(仅finance/xdxr): {FIN_SERVER[0]}:{FIN_SERVER[1]}")
        return c
    except Exception as e:
        logger.error(f"全部服务器连接失败: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_client()
    if client:
        print("连接成功，测试 600519:")
        print(client.quotes(symbol="600519").head(1).to_string())
        print(client.bars(symbol="600519", frequency=9, offset=3).head(3).to_string())
        print(client.finance(symbol="600519").head(1).to_string())
    else:
        print("连接失败")
