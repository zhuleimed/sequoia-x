#!/usr/bin/env python3
"""akshare 免费数据采集器 — 为 sequoia-x 88 维扩充提供基础数据

数据落盘结构（parquet, 按股票分文件, 支持断点续跑）:
    <OUT>/fund_flow/{code}.parquet   资金流向(120天历史+每日增量)
    <OUT>/finance/{code}.parquet     财务摘要(1998起全历史, 季频)
    <OUT>/holders/{code}.parquet     股东人数(62期历史, 季频)
    <OUT>/reports/{code}.parquet     券商研报+盈利预测(近期)

设计原则(铁律):
    1. 断点续跑: 每只股票完成后立即写盘; 启动时扫描已有文件跳过
    2. 限速: 并发<=8, 每请求间隔>=0.15s, 东财限频保护
    3. 日志: 进度百分比 + 每类数据规模 + ETA + 完成自检
    4. 低干扰: nice 已由调用方设置; CPU 占用低(网络IO为主)

用法:
    python3 collector.py [--codes codes.txt] [--data fund_flow,finance,holders,reports]
                         [--out /path/to/extra_features] [--workers 8] [--limit N]
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ── 参数 ──────────────────────────────────────────────────
DEFAULT_OUT = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/extra_features"
DEFAULT_POOL = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/output/backtest_v2/.stock_pool.json"
REQUEST_INTERVAL = 0.15  # 秒, 东财限频保护
SUBSETS = ("fund_flow", "finance", "holders", "consensus", "news", "xdxr", "forecast")


def code_market(code: str) -> str:
    """600519 -> sh / 000001 -> sz / 8xxxxx/4xxxxx -> bj"""
    if code.startswith(("60", "68", "90")):
        return "sh"
    if code.startswith(("00", "30", "20")):
        return "sz"
    return "bj"


def load_codes(pool_path: str, limit: int = None):
    if pool_path.endswith(".json"):
        data = json.load(open(pool_path))
        codes = data if isinstance(data, list) else list(data)
    else:  # txt: 每行一个代码
        codes = [l.strip() for l in open(pool_path) if l.strip()]
    return codes[:limit] if limit else codes


def fetch_fund_flow(code: str) -> pd.DataFrame:
    """主力资金流向(东财 push2his 直连, ~120天) — 替换 akshare 层

    直连优势: 自带 Referer/UA + 请求间隔>=1s 限流(东财风控阈值 ~5次/s),
    比 akshare 内部高频调用稳定得多。
    字段验证: 主力净额 = 大单净额 + 超大单净额 (f52 = f55 + f56)。
    """
    import requests
    market = "1" if code_market(code) == "sh" else "0"
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           f"?lmt=0&klt=101&fields1=f1,f2,f3,f7"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           f"&secid={market}.{code}")
    headers = {
        "Referer": "https://quote.eastmoney.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data")
    if not data or not data.get("klines"):
        return None
    rows = []
    for line in data["klines"]:
        p = line.split(",")
        if len(p) < 14:
            continue
        rows.append({
            "日期": p[0],
            "收盘价": p[11],
            "涨跌幅": p[12],
            "主力净流入-净额": p[1],
            "小单净流入-净额": p[2],
            "中单净流入-净额": p[3],
            "大单净流入-净额": p[4],
            "超大单净流入-净额": p[5],
            "主力净流入-净占比": p[6],
            "小单净流入-净占比": p[7],
            "中单净流入-净占比": p[8],
            "大单净流入-净占比": p[9],
            "超大单净流入-净占比": p[10],
        })
    df = pd.DataFrame(rows)
    df.insert(0, "code", code)
    # 东财限流: 请求间隔 >=1s (风控阈值 ~5/s, 并发4时每秒<=4次安全)
    time.sleep(1.0)
    return df


def fetch_finance(code: str) -> pd.DataFrame:
    """财务摘要(同花顺, 全历史) — 失败时降级 mootdx finance() 快照(最新期)

    降级链: 同花顺 akshare(102期全历史) → mootdx finance()(最新期34字段快照)
    """
    import akshare as ak
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is None or len(df) == 0:
            raise ValueError("空返回")
        df.insert(0, "code", code)
        # 清洗: 同花顺混合列(数值/百分比字符串/False)会破坏 parquet 类型推断,
        # 全量转字符串 + 统一空值, 保证可序列化
        return df.astype(str).replace({"False": "", "nan": "", "None": ""})
    except Exception:
        # ── 降级: mootdx 财务快照（最新期, 字段对齐 akshare 中文列名）──
        client = _get_mootdx()
        fin = client.finance(symbol=code)
        if fin is None or len(fin) == 0:
            return None
        r = fin.iloc[-1]
        return pd.DataFrame([{
            "code": code,
            "报告期": str(r.get("updated_date", "")),
            "净利润": r.get("jinglirun"),
            "营业总收入": r.get("zhuyingshouru"),
            "基本每股收益": r.get("jinglirun") / r.get("zongguben") if r.get("zongguben") else None,
            "每股净资产": r.get("meigujingzichan"),
            "净资产收益率": r.get("jinglirun") / r.get("jingzichan") * 100 if r.get("jingzichan") else None,
            "每股经营现金流": r.get("jingyingxianjinliu") / r.get("zongguben") if r.get("zongguben") else None,
            "资产负债率": r.get("liudongfuzhai", 0) / r.get("zongzichan", 0) * 100 if r.get("zongzichan") else None,
            "(降级源)": "mootdx",
        }])


# 东财直连通用头(各子域风控独立, 统一限流 >=1s/请求)
_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}


def _em_get(url: str, timeout=20) -> dict:
    """东财直连 GET + 限流(>=1s), 返回 JSON dict"""
    import requests
    resp = requests.get(url, headers=_EM_HEADERS, timeout=timeout)
    resp.raise_for_status()
    time.sleep(1.0)  # 东财 IP 风控: 每请求间隔 >=1s
    return resp.json()


def fetch_holders(code: str) -> pd.DataFrame:
    """股东户数(东财 datacenter-web 直连, 全历史分页)"""
    rows = []
    page = 1
    while page <= 100:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
               f"?reportName=RPT_HOLDERNUM_DET&columns=ALL&pageSize=100"
               f"&pageNumber={page}&filter=(SECURITY_CODE%3D%22{code}%22)"
               f"&source=WEB&client=WEB")
        d = _em_get(url)
        data = (d.get("result") or {}).get("data") or []
        if not data:
            break
        for r in data:
            rows.append({
                "股东户数统计截止日": r.get("END_DATE", "")[:10],
                "区间涨跌幅": r.get("INTERVAL_CHRATE"),
                "股东户数-本次": r.get("HOLDER_NUM"),
                "股东户数-上次": r.get("PRE_HOLDER_NUM"),
                "股东户数-增减": r.get("HOLDER_NUM_CHANGE"),
                "股东户数-增减比例": r.get("HOLDER_NUM_RATIO"),
                "户均持股市值": r.get("AVG_MARKET_CAP"),
                "户均持股数量": r.get("AVG_HOLD_NUM"),
                "总市值": r.get("TOTAL_MARKET_CAP"),
                "总股本": r.get("TOTAL_A_SHARES"),
                "股本变动": r.get("CHANGE_SHARES"),
                "股本变动原因": r.get("CHANGE_REASON"),
                "股东户数公告日期": r.get("HOLD_NOTICE_DATE", "")[:10],
            })
        if len(data) < 100:
            break
        page += 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.insert(0, "code", code)
    return df


def fetch_reports(code: str) -> pd.DataFrame:
    """券商研报+盈利预测(东财 reportapi 直连) — 2026-08-07 弃用(ADR V3-14)

    原因: reportapi 当天全程封禁(串行也失败); consensus 快照已覆盖其核心价值
    (评级分布/三年EPS预测/目标价); 预测修正由 consensus 月度快照积累实现。
    保留代码供参考, 不再进入月度流程 --data 列表。
    """
    rows = []
    for page in range(1, 50):
        url = ("https://reportapi.eastmoney.com/report/list"
               f"?pageSize=100&industryCode=*&industry=*&rating=*&ratingChange=*"
               f"&beginTime=2016-01-01&endTime={time.strftime('%Y-%m-%d')}"
               f"&pageNo={page}&fields=&qType=0&orgCode=&code={code}&rcode=")
        d = _em_get(url)
        data = d.get("data") or []
        if not data:
            break
        for r in data:
            rows.append({
                "报告名称": r.get("title"),
                "股票代码": r.get("stockCode"),
                "股票简称": r.get("stockName"),
                "机构": r.get("orgSName") or r.get("orgName"),
                "东财评级": r.get("emRatingName"),
                "评级变化": r.get("ratingChange"),
                "日期": str(r.get("publishDate", ""))[:10],
                "2026-盈利预测-收益": r.get("predictThisYearEps"),
                "2026-盈利预测-市盈率": r.get("predictThisYearPe"),
                "2027-盈利预测-收益": r.get("predictNextYearEps"),
                "2027-盈利预测-市盈率": r.get("predictNextYearPe"),
                "2028-盈利预测-收益": r.get("predictNextTwoYearEps"),
                "2028-盈利预测-市盈率": r.get("predictNextTwoYearPe"),
                "目标价": r.get("indvAimPriceT"),
                "行业": r.get("industryName"),
                "作者": r.get("author"),
                "报告PDF链接": r.get("encodeUrl"),
            })
        if len(data) < 100:
            break
        page += 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.insert(0, "code", code)
    return df


def fetch_consensus(code: str) -> pd.DataFrame:
    """一致预期快照 — 报表优先 + reportapi 兜底（2026-08-07 修正）

    2026-08-07 抽查发现: RPT_WEB_RESPREDICT 报表仅收录 2814 只(54%),
    大量 2025-2026 年有活跃研报的股票(688089/688199/600444 等)不在其中。
    → 报表无记录时, 用 reportapi 逐条研报聚合兜底(评级分布/最新预测EPS/目标价)。
    """
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
           "?reportName=RPT_WEB_RESPREDICT&columns=ALL&pageSize=1&pageNumber=1"
           f"&filter=(SECURITY_CODE%3D%22{code}%22)&source=WEB&client=WEB")
    d = _em_get(url)
    data = (d.get("result") or {}).get("data") or []
    if data:
        r = data[0]
        return pd.DataFrame([{
            "code": code,
            "机构数": r.get("RATING_ORG_NUM"),
            "买入数": r.get("RATING_BUY_NUM"),
            "增持数": r.get("RATING_ADD_NUM"),
            "中性数": r.get("RATING_NEUTRAL_NUM"),
            "减持数": r.get("RATING_REDUCE_NUM"),
            "卖出数": r.get("RATING_SALE_NUM"),
            "Y1实际EPS": r.get("EPS1"),
            "Y1年度": r.get("YEAR1"),
            "Y2预测EPS": r.get("EPS2"),
            "Y2年度": r.get("YEAR2"),
            "Y3预测EPS": r.get("EPS3"),
            "Y3年度": r.get("YEAR3"),
            "Y4预测EPS": r.get("EPS4"),
            "Y4年度": r.get("YEAR4"),
            "目标价上限": r.get("DEC_AIMPRICEMAX"),
            "目标价下限": r.get("DEC_AIMPRICEMIN"),
        }])

    # ── 报表无记录 → reportapi 逐条研报聚合兜底(近 30 月) ──
    rows = []
    for page in range(1, 20):
        url = ("https://reportapi.eastmoney.com/report/list"
               f"?pageSize=100&industryCode=*&industry=*&rating=*&ratingChange=*"
               f"&beginTime=2024-01-01&endTime={time.strftime('%Y-%m-%d')}"
               f"&pageNo={page}&fields=&qType=0&orgCode=&code={code}&rcode=")
        d = _em_get(url)
        data = d.get("data") or []
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
    if not rows:
        return None  # 确实无研报
    orgs = {r.get("orgCode") for r in rows if r.get("orgCode")}
    buy = sum(1 for r in rows if r.get("emRatingName") == "买入")
    add = sum(1 for r in rows if r.get("emRatingName") == "增持")
    neu = sum(1 for r in rows if r.get("emRatingName") == "中性")
    red = sum(1 for r in rows if r.get("emRatingName") == "减持")
    sale = sum(1 for r in rows if r.get("emRatingName") == "卖出")
    latest = rows[0]  # reportapi 按时间倒序
    return pd.DataFrame([{
        "code": code,
        "机构数": len(orgs),
        "买入数": buy,
        "增持数": add,
        "中性数": neu,
        "减持数": red,
        "卖出数": sale,
        "Y1实际EPS": latest.get("actualLastYearEps"),
        "Y1年度": "",
        "Y2预测EPS": latest.get("predictThisYearEps"),
        "Y2年度": "",
        "Y3预测EPS": latest.get("predictNextYearEps"),
        "Y3年度": "",
        "Y4预测EPS": latest.get("predictNextTwoYearEps"),
        "Y4年度": "",
        "目标价上限": latest.get("indvAimPriceT"),
        "目标价下限": latest.get("indvAimPriceL"),
        "(兜底源)": "reportapi",
    }])


def fetch_news(code: str) -> pd.DataFrame:
    """个股新闻(东财 search-api-web 直连, JSONP 剥壳, 单页100条≈近1月)

    数量信号只需近期活跃度, 单页 100 条足够(≈1个月); 全量 559 条需 12 页
    → 全市场 5206 只将耗时 4h+, 单页方案 ~22 分钟(4并发)。
    """
    import requests
    param = (f'{{"uid":"","keyword":"{code}","type":["cmsArticleWebOld"],'
             f'"client":"web","clientType":"web","clientVersion":"curr",'
             f'"param":{{"cmsArticleWebOld":{{"searchScope":"default","sort":"default",'
             f'"pageIndex":1,"pageSize":100,"preTag":"<em>","postTag":"</em>"}}}}}}')
    url = ("https://search-api-web.eastmoney.com/search/jsonp"
           f"?cb=callback&param={requests.utils.quote(param)}")
    resp = requests.get(url, headers=_EM_HEADERS, timeout=20)
    resp.raise_for_status()
    text = resp.text
    start, end = text.find("("), text.rfind(")")
    d = json.loads(text[start + 1:end])
    items = (d.get("result") or {}).get("cmsArticleWebOld") or []
    if not items:
        return None
    rows = [{
        "新闻标题": it.get("title", "").replace("<em>", "").replace("</em>", ""),
        "新闻内容": it.get("content", "").replace("<em>", "").replace("</em>", ""),
        "发布时间": it.get("date"),
        "文章来源": it.get("mediaName"),
        "新闻链接": it.get("url"),
    } for it in items]
    df = pd.DataFrame(rows)
    df.insert(0, "code", code)
    time.sleep(1.0)  # 东财限流
    return df


def fetch_forecast(code: str) -> pd.DataFrame:
    """业绩预告(baostock, 全历史事件数据) — 2026-08-07 新增第7类

    字段: profitForcastExpPubDate(预告发布日期, 天然披露日=asof对齐零成本)
          profitForcastType(略增/预增/预减/扭亏/首亏...) → 事件类型因子
          profitForcastChgPctUp/Dwn(净利润增幅上下限) → 预增信号
    注意: baostock 阻塞式 API, 单进程强制(见 CONCURRENCY_LIMIT)
    """
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login 失败: {lg.error_msg}")
    try:
        bs_code = f"{code_market(code)}.{code}"
        rs = bs.query_forecast_report(bs_code, start_date="1998-01-01", end_date="2026-12-31")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None  # 无业绩预告记录(业绩稳定公司正常现象)
        df = pd.DataFrame(rows, columns=rs.fields)
        if "code" in df.columns:
            df = df.rename(columns={"code": "bs_code"})  # baostock 自带 code(sh.600519)
        df.insert(0, "code", code)
        return df
    finally:
        bs.logout()


# mootdx 客户端(延迟加载, 复用修复版服务器配置)
_MOOTDX_CLIENT = None


def _get_mootdx():
    global _MOOTDX_CLIENT
    if _MOOTDX_CLIENT is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mootdx_client import get_client
        _MOOTDX_CLIENT = get_client()
    return _MOOTDX_CLIENT


def fetch_xdxr(code: str) -> pd.DataFrame:
    """除权除息历史(mootdx, 稳定源) — 分红/送转/配股全历史"""
    client = _get_mootdx()
    df = client.xdxr(symbol=code)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.insert(0, "code", code)
    return df


FETCHERS = {
    "fund_flow": fetch_fund_flow,
    "finance": fetch_finance,
    "holders": fetch_holders,
    "reports": fetch_reports,
    "consensus": fetch_consensus,
    "news": fetch_news,
    "xdxr": fetch_xdxr,
    "forecast": fetch_forecast,
}


def _done_path(out, subset, code):
    return os.path.join(out, subset, f"{code}.parquet")


def already_skip(out, subset, codes, refresh_days):
    """计算需要采集的股票列表(断点续跑 + 新鲜度)。

    - 文件不存在 → 需要采集
    - 文件存在且年龄 < refresh_days → 跳过(数据还新鲜)
    - 文件存在但年龄 >= refresh_days → 重拉(覆盖)
    """
    d = os.path.join(out, subset)
    os.makedirs(d, exist_ok=True)
    todo, skip = [], 0
    for c in codes:
        p = _done_path(out, subset, c)
        if os.path.exists(p):
            if refresh_days is not None and (time.time() - os.path.getmtime(p)) >= refresh_days * 86400:
                todo.append(c)  # 过期, 重拉
            else:
                skip += 1
        else:
            todo.append(c)
    if skip:
        print(f"  [{subset}] 跳过新鲜文件 {skip} 只", flush=True)
    return todo


def collect_subset(subset, codes, out, workers, refresh_days=None):
    """采集一类数据, 支持断点续跑+新鲜度。返回 (新增, 成功, 失败)"""
    fn = FETCHERS[subset]
    todo = already_skip(out, subset, codes, refresh_days)
    if not todo:
        return 0, 0, 0

    ok = fail = skip_empty = 0
    t0 = time.time()
    total = len(todo)
    lock = __import__("threading").Lock()

    # 解析类错误(数据源本身问题, 重试无意义, 直接跳过); 连接类错误(限频/抖动, 重试)
    PARSE_ERRORS = (AttributeError, ValueError, TypeError, KeyError, IndexError)

    def work(code):
        """带重试(指数退避)的采集任务。连接类错误重试; 解析类错误直接跳过;
        空返回(无覆盖/无记录) = 正常现象, 记 skip 不进 failed 清单。"""
        nonlocal ok, fail, skip_empty
        for attempt in range(4):  # 连接错误最多重试 4 次
            try:
                df = fn(code)
                if df is not None and len(df) > 0:
                    df.to_parquet(_done_path(out, subset, code), index=False)
                    with lock:
                        ok += 1
                        return code, True
                with lock:
                    skip_empty += 1
                    return code, "empty"  # 空返回 = 数据不存在(如无研报覆盖), 正常
            except PARSE_ERRORS as e:
                if isinstance(e, json.JSONDecodeError):
                    # 服务端偶发返回损坏 JSON(baostock 业绩预告实测) → 重试而非跳过
                    if attempt < 3:
                        time.sleep(2 ** attempt * 1.5)
                        continue
                with lock:
                    fail += 1
                    print(f"    ! {code} 解析失败(跳过): "
                          f"{traceback.format_exc().strip().splitlines()[-1][:120]}", flush=True)
                return code, False
            except Exception:
                if attempt < 3:
                    time.sleep(2 ** attempt * 1.5)  # 1.5s / 3s / 6s 退避
                else:
                    print(f"    ! {code} 重试4次仍失败: "
                          f"{traceback.format_exc().strip().splitlines()[-1][:150]}", flush=True)
        with lock:
            fail += 1
            return code, False

    failed_codes = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            code, ok_flag = f.result()
            if ok_flag is False:
                failed_codes.append(code)
            if i % 50 == 0 or i == total:
                el = time.time() - t0
                eta = el / i * (total - i)
                print(f"  [{subset}] {i}/{total} ({100*i/total:.0f}%) "
                      f"成功{ok} 失败{fail} 空跳过{skip_empty} 耗时{el/60:.1f}min ETA {eta/60:.1f}min",
                      flush=True)

    # ── 失败留痕: failed_{subset}.txt + manifest.json 汇总 ──
    if failed_codes:
        fp = os.path.join(out, f"failed_{subset}.txt")
        with open(fp, "w") as f:
            f.write("\n".join(failed_codes))
        print(f"  [{subset}] 失败 {len(failed_codes)} 只 → 已记录 {fp}", flush=True)
        print(f"  [{subset}] 下月补采: collector --codes {fp}", flush=True)
    if subset == "fund_flow" and failed_codes:
        print(f"  [fund_flow] 💡 DDE 降级指引(实盘当日资金流, 摆脱东财):", flush=True)
        print(f"    python3 scripts/dde_calculator.py --codes {fp} "
              f"--start <日期> --end <日期> --out data/extra_features/dde", flush=True)
    return len(todo), ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=DEFAULT_POOL)
    ap.add_argument("--data", default=",".join(SUBSETS))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="只采前 N 只(调试)")
    ap.add_argument("--refresh-days", type=int, default=None,
                    help="文件年龄超过 N 天才重拉(月度刷新用); 缺省=补采模式(缺失才拉)")
    args = ap.parse_args()

    codes = load_codes(args.codes, args.limit)
    subsets = [s for s in args.data.split(",") if s in FETCHERS]
    mode = f"刷新(>{args.refresh_days}天)" if args.refresh_days is not None else "补采(仅缺失)"
    print(f"股票池 {len(codes)} 只 | 类别 {subsets} | 模式 {mode} | 并发 {args.workers} | 输出 {args.out}")

    # 2026-08-07 实测教训: 不同数据面风控差异大, 按子域固化并发上限
    #   mootdx(xdxr): 并发>1 时服务器拒绝(返回空) → 必须串行
    #   push2his(fund_flow)/reportapi(reports): 高频触发 IP 时间性封禁 → 预防性串行
    #   baostock(forecast): 阻塞式全局连接 API → 必须单进程
    #   datacenter-web(holders)/search-api(news): 4 并发实测 OK
    CONCURRENCY_LIMIT = {"fund_flow": 1, "reports": 1, "xdxr": 1, "forecast": 1}

    summary = {}
    for s in subsets:
        w = CONCURRENCY_LIMIT.get(s, args.workers)
        print(f"\n===== 开始采集: {s} (并发 {w}) =====", flush=True)
        added, ok, fail = collect_subset(s, codes, args.out, w, args.refresh_days)
        print(f"===== {s} 完成: 新增{added} 成功{ok} 失败{fail} =====", flush=True)
        summary[s] = {"attempted": added, "success": ok, "fail": fail}

    # manifest: 运行摘要(供人工核对 + 自动化检查)
    manifest = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "codes_source": args.codes,
        "subsets": summary,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n运行摘要已写入 {os.path.join(args.out, 'manifest.json')}")

    # 自检: 统计落盘文件
    print("\n===== 落盘自检 =====")
    for s in subsets:
        d = os.path.join(args.out, s)
        n = len([f for f in os.listdir(d) if f.endswith(".parquet")]) if os.path.isdir(d) else 0
        print(f"  {s}: {n} 只股票")
    print("ALL DONE")


if __name__ == "__main__":
    main()
