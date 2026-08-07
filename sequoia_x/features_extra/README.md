# features_extra — 扩展维度特征模块（2026-08-07 实现完成）

**88+33=121 维拼接已实现（2026-08-07 16:45）**：`cfg.extra_features=True` 启用（V2Config，
默认 False=88 维行为不变）。接入点：`model_selection_v2/features.py`（_extract_per_day_features
加 extra_matrix 参数）+ `labels.py`（缓存 hash 含 extra_features）+ `scripts/build_prediction_cache.py`
（训练缓存目录动态化 + 预测特征拼接 + incomplete 过滤）。8 月首次月度重训前启用。

**归属规则（用户明确要求）**：020_TDX 为探索验证项目，**一切生产代码以 sequoia-x 为准**。

## 已实现

### `build_extra_features.py`（主模块, 30-33 维特征）

接口（与 model_selection_v2/features.py 88 维对齐）:
```python
build_extra_features(dates, close, code, groups=None)
  → (features_df, coverage)
    features_df: (n_days, N) 稠密 float32, 缺失 fillna(0)
    coverage:    {特征名: 非零比例}
```

| 数据面 | 特征(前缀) | 维数 | 对齐方式(防 look-ahead) |
|--------|-----------|:---:|------------------------|
| fund_flow | ff_main_ratio/5d/天数/20d累计/动量/xbig | 6 | 日频直接对齐 |
| finance | fin_roe/gp_margin/np_margin/debt/yoy/chg/cf_quality/eps/bps | 10 | 报告期→法定披露日(Q1:4/30 H1:8/31 Q3:10/31 年报:次年4/30) asof |
| holders | hd_num_chg/hd_avg_mcap | 2 | 公告日 asof |
| consensus | cs_buy_ratio/org_num/pred_pe/aim_dev/aim_spread | 5 | **快照: 文件采集日后生效**(回测早期无值, 树模型忽略) |
| news | nw_cnt_5d/nw_cnt_20d/nw_src_div | 3 | 发布时间对齐 |
| xdxr | xd_yield/div_cnt_3y/song_cnt_3y | 3 | 事件日滚动365D |
| forecast | fc_type_12m/max_chg/cnt/freshness | 4 | 发布日 asof |
| **合计** | | **33** | |

### 验证（2026-08-07 实测）
- 600519: ROE 16.8%/毛利率 90.4%/负债率 12.4%/BPS 211 ✅（与财务数据一致）
- 000001: 买入占比 54.5%/目标价偏离 -4.2% ✅
- 披露日重复(年报与Q1同日)已处理(drop_duplicates)
- consensus 快照从采集日生效(覆盖率按日增长, 3-6月后形成历史序列可算预测修正)

## 缺口处置机制（build_extra_with_flag / scan_incomplete, 已实现）

```python
build_extra_with_flag(dates, close, code) → (features_df, incomplete, coverage)
scan_incomplete(codes, get_close)         → {code: incomplete}   # 批量, 训练前调用
```

**规则**（2026-08-07 用户确认）:
- 关键数据面 fund_flow/finance/holders（每只股票理应都有）: 覆盖 <5% → incomplete
- 语义可缺失面 consensus(46%无研报)/forecast(无预告)/news/xdxr(次新无分红): 缺失不影响
- 窗口限制(fund_flow 120天 vs 250天窗口=48%) ≠ 缺失, 阈值 5% 区分
- incomplete 股票: 训练集剔除(复用"有效股票"过滤) / 预测不产生信号

验证(实测): 600519→False | fund_flow缺失→True | consensus无覆盖→False

## 规划（后续迭代）
- sentiment.py: news 词表情绪（第二档）
