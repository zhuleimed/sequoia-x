"""数据模块。

- DataEngine（查询层）：行情数据读写，提供 get_ohlcv / get_base_stock_pool 等查询接口。
- DataSync（同步层）：baostock → SQLite 全量/增量数据同步管线。
"""
