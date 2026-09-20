# -*- coding: utf-8 -*-
"""
生成自包含 HTML 数据分析报告（含内联 SVG 图表）。
读取 detection_data.csv，计算统计指标并输出 detection_report.html。
"""
import os
import csv
import math
import html

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "detection_data.csv")
OUT_PATH = os.path.join(HERE, "detection_report.html")

# ---------- 读取数据 ----------
rows = []
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        r["circles"] = int(r["circles"])
        r["fallen"] = int(r["fallen"])
        r["correct"] = (r["correct"].strip().lower() == "true")
        rows.append(r)

# ---------- 统计 ----------
def g(rows, pred):  # predicate
    return [r for r in rows if pred(r)]

tidy = g(rows, lambda r: r["expected"] == "整齐")
messy = g(rows, lambda r: r["expected"] == "不整齐")
tp = sum(1 for r in tidy if r["actual"] == "整齐")
fn = sum(1 for r in tidy if r["actual"] == "不整齐")
fp = sum(1 for r in messy if r["actual"] == "整齐")
tn = sum(1 for r in messy if r["actual"] == "不整齐")

N = len(rows)
n_correct = tp + tn
n_wrong = N - n_correct
acc = n_correct / N
tidy_acc = tp / len(tidy)          # 整齐类召回
messy_acc = tn / len(messy)        # 不整齐类召回
precision = tp / (tp + fp)
recall = tidy_acc
f1 = 2 * precision * recall / (precision + recall)

# 误判类型
miss_messy = fp        # 漏检不整齐（不整齐判成整齐）
false_alarm = fn       # 误报不整齐（整齐判成不整齐）

# ---------- 圆形数量分布（用于箱线图） ----------
tidy_circles = sorted(r["circles"] for r in tidy)
messy_circles = sorted(r["circles"] for r in messy)

def quartiles(xs):
    n = len(xs)
    def q(p):
        k = (n - 1) * p
        lo = int(math.floor(k)); hi = int(math.ceil(k))
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return q(0.0), q(0.25), q(0.5), q(0.75), q(1.0)

t_min, t_q1, t_med, t_q3, t_max = quartiles(tidy_circles)
m_min, m_q1, m_med, m_q3, m_max = quartiles(messy_circles)

def fmt_pct(x):
    return f"{x*100:.1f}%"

# ---------- SVG 工具 ----------
def donut_segment(cx, cy, r, ir, a0, a1):
    """角度以度为单位，0°=右，顺时针为正（SVG y 向下）。返回 path d。"""
    def pt(ang, rad):
        a = math.radians(ang)
        return (cx + rad * math.cos(a), cy + rad * math.sin(a))
    large = 1 if (a1 - a0) > 180 else 0
    x0o, y0o = pt(a0, r)
    x1o, y1o = pt(a1, r)
    x1i, y1i = pt(a1, ir)
    x0i, y0i = pt(a0, ir)
    return (f"M {x0o:.3f} {y0o:.3f} "
            f"A {r:.3f} {r:.3f} 0 {large} 1 {x1o:.3f} {y1o:.3f} "
            f"L {x1i:.3f} {y1i:.3f} "
            f"A {ir:.3f} {ir:.3f} 0 {large} 0 {x0i:.3f} {y0i:.3f} Z")

def build_donut(slices):
    """slices: list of (label, count, color). 返回 SVG 内容与图例 HTML。"""
    cx, cy, r, ir = 140, 140, 120, 66
    total = sum(s[1] for s in slices)
    svg = []
    legend = []
    a = -90.0
    for label, count, color in slices:
        frac = count / total if total else 0
        sweep = 360.0 * frac
        d = donut_segment(cx, cy, r, ir, a, a + sweep)
        svg.append(f'<path d="{d}" fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>')
        mid = a + sweep / 2.0
        # 标签放在环外侧（半径 r+24）
        lx = cx + (r + 26) * math.cos(math.radians(mid))
        ly = cy + (r + 26) * math.sin(math.radians(mid))
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" '
                   f'class="donut-label">{count}</text>')
        legend.append(f'<span class="lg-item"><span class="sw" style="background:{color}"></span>'
                      f'{html.escape(label)} <b>{count}</b> · {fmt_pct(frac)}</span>')
        a += sweep
    # 中心文字
    svg.append(f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" class="donut-center-num">{n_correct}/{N}</text>')
    svg.append(f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" class="donut-center-sub">判定正确</text>')
    return "".join(svg), "".join(legend)

# 混淆矩阵颜色（语义化）
CM = [
    # (期望, 实际, 数量, 占比, 颜色, 描述)
    ("整齐", "整齐", tp, tp / len(tidy), "var(--good)", "正确判整齐"),
    ("整齐", "不整齐", fn, fn / len(tidy), "var(--warning)", "误报不整齐"),
    ("不整齐", "整齐", fp, fp / len(messy), "var(--critical)", "漏检不整齐"),
    ("不整齐", "不整齐", tn, tn / len(messy), "var(--good)", "正确判不整齐"),
]

# ---------- 逐类准确率柱状图 ----------
BAR = [
    ("总体准确率", acc, "var(--blue)"),
    ("整齐类", tidy_acc, "var(--aqua)"),
    ("不整齐类", messy_acc, "var(--orange)"),
]

def build_bars(items, height=220):
    """items: list of (label, value 0-1, color). 返回 SVG。"""
    w = 420
    top = 20
    plot_h = height - top - 40
    baseline = top + plot_h
    bw = 66
    gap = 40
    total_w = gap + len(items) * (bw + gap)
    x0 = (w - total_w) / 2 + gap
    svg = []
    svg.append(f'<line x1="0" y1="{baseline}" x2="{w}" y2="{baseline}" stroke="var(--baseline)" stroke-width="1.5"/>')
    # 参考刻度 0/25/50/75/100%
    for pct in (0, 25, 50, 75, 100):
        y = baseline - plot_h * pct / 100
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" stroke="var(--gridline)" stroke-width="1"/>')
        svg.append(f'<text x="0" y="{y - 4:.1f}" class="tick">{pct}%</text>')
    for i, (label, val, color) in enumerate(items):
        x = x0 + i * (bw + gap)
        bh = plot_h * val
        svg.append(f'<rect x="{x:.1f}" y="{baseline - bh:.1f}" width="{bw}" height="{bh:.1f}" rx="4" fill="{color}"/>')
        svg.append(f'<text x="{x + bw/2:.1f}" y="{baseline - bh - 8:.1f}" text-anchor="middle" class="bar-val">{fmt_pct(val)}</text>')
        svg.append(f'<text x="{x + bw/2:.1f}" y="{baseline + 18:.1f}" text-anchor="middle" class="bar-label">{label}</text>')
    return "".join(svg)

# ---------- 箱线图（圆形检测数量） ----------
def build_boxplot():
    w, height = 420, 260
    top, bottom = 20, height - 46
    # y 轴映射（数据范围 55 ~ 165）
    ymin, ymax = 55, 170
    def Y(v):
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    svg = []
    # 网格与刻度
    for v in range(60, 170, 20):
        y = Y(v)
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" stroke="var(--gridline)" stroke-width="1"/>')
        svg.append(f'<text x="0" y="{y - 4:.1f}" class="tick">{v}</text>')
    groups = [("整齐", tidy_circles, "var(--aqua)"), ("不整齐", messy_circles, "var(--orange)")]
    cx = [w * 0.28, w * 0.72]
    for i, (label, xs, color) in enumerate(groups):
        x = cx[i]
        lo, q1, med, q3, hi = quartiles(xs)
        bw = 46
        # 散点（加少量抖动）
        import random
        rnd = random.Random(i + 42)
        for v in xs:
            jx = x + rnd.uniform(-bw * 0.36, bw * 0.36)
            svg.append(f'<circle cx="{jx:.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}" opacity="0.55"/>')
        # 箱体
        svg.append(f'<rect x="{x - bw/2:.1f}" y="{Y(q3):.1f}" width="{bw}" height="{Y(q1)-Y(q3):.1f}" '
                   f'fill="{color}" opacity="0.22" stroke="{color}" stroke-width="1.5" rx="3"/>')
        # 中位线
        svg.append(f'<line x1="{x - bw/2:.1f}" y1="{Y(med):.1f}" x2="{x + bw/2:.1f}" y2="{Y(med):.1f}" '
                   f'stroke="{color}" stroke-width="2.5"/>')
        # 须
        for v in (lo, hi):
            svg.append(f'<line x1="{x:.1f}" y1="{Y(v):.1f}" x2="{x:.1f}" y2="{Y(q1 if v==lo else q3):.1f}" '
                       f'stroke="{color}" stroke-width="1.2"/>')
        svg.append(f'<line x1="{x - bw*0.3:.1f}" y1="{Y(lo):.1f}" x2="{x + bw*0.3:.1f}" y2="{Y(lo):.1f}" stroke="{color}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x - bw*0.3:.1f}" y1="{Y(hi):.1f}" x2="{x + bw*0.3:.1f}" y2="{Y(hi):.1f}" stroke="{color}" stroke-width="1.5"/>')
        # 中位数值标注
        svg.append(f'<text x="{x:.1f}" y="{Y(q3) - 8:.1f}" text-anchor="middle" class="bar-label">中位 {med:.0f}</text>')
        svg.append(f'<text x="{x:.1f}" y="{height - 14:.1f}" text-anchor="middle" class="bar-label">{label}（n={len(xs)}）</text>')
    return "".join(svg)

# ---------- 生成 HTML ----------
donut_slices = [
    ("正确判整齐", tp, "var(--aqua)"),
    ("正确判不整齐", tn, "var(--blue)"),
    ("漏检不整齐", fp, "var(--critical)"),
    ("误报不整齐", fn, "var(--warning)"),
]
donut_svg, donut_legend = build_donut(donut_slices)
bars_svg = build_bars(BAR)
box_svg = build_boxplot()

# 全量表
table_rows_html = []
for r in rows:
    cls = "ok" if r["correct"] else "bad"
    mark = "✓" if r["correct"] else "✗"
    table_rows_html.append(
        f'<tr class="{cls}"><td>{r["file"]}</td><td>{r["expected"]}</td><td>{r["actual"]}</td>'
        f'<td>{mark}</td><td>{r["circles"]}</td><td>{r["fallen"]}</td><td class="reason">{html.escape(r["reason"])}</td></tr>'
    )

def cm_cell(exp, act, cnt, frac, color, desc):
    return (f'<div class="cm-cell" style="--cell:{color}">'
            f'<div class="cm-num">{cnt}</div>'
            f'<div class="cm-frac">{fmt_pct(frac)}</div>'
            f'<div class="cm-desc">{desc}</div></div>')

html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>枪头整齐度检测 · 数据分析报告</title>
<style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#898781;
  --gridline:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --yellow:#eda100; --red:#e34948;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --serious:#ec835a; }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{ --page:#0d0d0d; --surface:#1a1a19; --text:#ffffff; --text2:#c3c2b7;
    --muted:#898781; --gridline:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
    --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --yellow:#c98500; --red:#e66767;
    --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --serious:#ec835a; }} }}
:root[data-theme="dark"] {{ --page:#0d0d0d; --surface:#1a1a19; --text:#ffffff; --text2:#c3c2b7;
  --muted:#898781; --gridline:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --yellow:#c98500; --red:#e66767;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --serious:#ec835a; }}

* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.55; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 64px; }}
header {{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; }}
h1 {{ font-size:22px; margin:0; font-weight:650; }}
.sub {{ color:var(--text2); font-size:13px; margin-top:4px; }}
.theme-btn {{ font-size:12px; padding:6px 12px; border:1px solid var(--border); background:var(--surface);
  color:var(--text2); border-radius:8px; cursor:pointer; }}
h2 {{ font-size:15px; margin:40px 0 14px; font-weight:650; }}
p.lead {{ color:var(--text2); font-size:13.5px; margin:0 0 8px; }}

/* KPI */
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:22px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }}
.kpi .v {{ font-size:30px; font-weight:700; letter-spacing:-.5px; }}
.kpi .l {{ color:var(--text2); font-size:12.5px; margin-top:4px; }}
.kpi.neg .v {{ color:var(--critical); }}
.kpi.pos .v {{ color:var(--good); }}

/* 图表卡片 */
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:20px 22px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
svg {{ width:100%; height:auto; display:block; }}
.tick {{ fill:var(--muted); font-size:10px; }}
.bar-label {{ fill:var(--text2); font-size:12px; }}
.bar-val {{ fill:var(--text); font-size:13px; font-weight:600; }}
.donut-label {{ fill:var(--text); font-size:13px; font-weight:600; }}
.donut-center-num {{ fill:var(--text); font-size:30px; font-weight:700; }}
.donut-center-sub {{ fill:var(--text2); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:6px; }}
.lg-item {{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:var(--text2); }}
.lg-item b {{ color:var(--text); }}
.sw {{ width:11px; height:11px; border-radius:3px; display:inline-block; flex:none; }}

/* 混淆矩阵 */
.cm {{ display:grid; grid-template-columns:auto 1fr 1fr; gap:8px; }}
.cm-h {{ font-size:12px; color:var(--text2); text-align:center; padding:4px; }}
.cm-rowlab {{ font-size:12px; color:var(--text2); display:flex; align-items:center; justify-content:flex-end; padding-right:4px; }}
.cm-cell {{ background:var(--surface); background:color-mix(in srgb, var(--cell) 16%, var(--surface));
  border:1px solid var(--cell); border-radius:10px; text-align:center; padding:18px 8px; }}
.cm-num {{ font-size:26px; font-weight:700; color:var(--text); }}
.cm-frac {{ font-size:12px; color:var(--text2); margin-top:2px; }}
.cm-desc {{ font-size:11px; margin-top:6px; color:var(--text2); }}

/* 表 */
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th,td {{ padding:7px 9px; text-align:left; border-bottom:1px solid var(--gridline); }}
th {{ color:var(--text2); font-weight:600; position:sticky; top:0; background:var(--surface); }}
tr.ok td {{ background:rgba(12,163,12,.05); }}
tr.bad td {{ background:rgba(208,59,59,.07); }}
td.reason {{ color:var(--text2); }}
.notice {{ font-size:12.5px; color:var(--text2); background:var(--surface);
  border:1px solid var(--border); border-radius:12px; padding:14px 18px; margin-top:16px; }}
.notice b {{ color:var(--text); }}
ul {{ margin:8px 0 0; padding-left:20px; }}
li {{ margin:3px 0; }}
.attribution {{ font-size:12.5px; color:var(--text2); background:var(--surface);
  border:1px solid var(--border); border-radius:10px; padding:9px 14px; margin:20px 0 6px; }}
.attribution b {{ color:var(--text); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="attribution">作者：<b>Claude（AI 助手）</b> · 所做工作：编写测试脚本并生成报告</div>
  <header>
    <div>
      <h1>枪头整齐度检测 · 数据分析报告</h1>
      <div class="sub">基于 {N} 张测试图片（整齐 {len(tidy)} 张 / 不整齐 {len(messy)} 张）的离线回归结果</div>
    </div>
    <button class="theme-btn" onclick="toggleTheme()">切换明暗</button>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="v">{fmt_pct(acc)}</div><div class="l">总体准确率（{n_correct}/{N}）</div></div>
    <div class="kpi pos"><div class="v">{fmt_pct(tidy_acc)}</div><div class="l">整齐类准确率（{tp}/{len(tidy)}）</div></div>
    <div class="kpi neg"><div class="v">{fmt_pct(messy_acc)}</div><div class="l">不整齐类准确率（{tn}/{len(messy)}）</div></div>
    <div class="kpi neg"><div class="v">{n_wrong}</div><div class="l">误判总数（漏检 {miss_messy} + 误报 {false_alarm}）</div></div>
  </div>

  <h2>1 · 混淆矩阵</h2>
  <p class="lead">行 = 真实状态，列 = 检测判定。绿色为判定正确，红/橙为误判。</p>
  <div class="card">
    <div class="cm">
      <div></div><div class="cm-h">判定「整齐」</div><div class="cm-h">判定「不整齐」</div>
      <div class="cm-rowlab">真实「整齐」</div>
      {cm_cell("整齐","整齐",tp,tp/len(tidy),"var(--good)","正确 · TP")}
      {cm_cell("整齐","不整齐",fn,fn/len(tidy),"var(--warning)","误报 · FN")}
      <div class="cm-rowlab">真实「不整齐」</div>
      {cm_cell("不整齐","整齐",fp,fp/len(messy),"var(--critical)","漏检 · FP")}
      {cm_cell("不整齐","不整齐",tn,tn/len(messy),"var(--good)","正确 · TN")}
    </div>
  </div>

  <h2>2 · 判定结果分布（饼图）</h2>
  <p class="lead">中心数字为判定正确的样本占比。漏检不整齐是最主要的错误来源。</p>
  <div class="card grid2">
    <svg viewBox="0 0 280 280" role="img" aria-label="判定结果分布饼图">{donut_svg}</svg>
    <div class="legend" style="align-content:center;">{donut_legend}</div>
  </div>

  <h2>3 · 逐类准确率</h2>
  <p class="lead">检测器对「整齐」很敏感（{fmt_pct(tidy_acc)}），但对「不整齐」漏检严重（仅 {fmt_pct(messy_acc)}）。</p>
  <div class="card"><svg viewBox="0 0 420 220" role="img" aria-label="逐类准确率柱状图">{bars_svg}</svg></div>

  <h2>4 · 圆形检测数量分布</h2>
  <p class="lead">整齐与不整齐图像的「圆形数」高度重叠，说明仅凭圆形数量/对齐无法区分不整齐 —— 这是漏检的根因。</p>
  <div class="card"><svg viewBox="0 0 420 260" role="img" aria-label="圆形数量箱线图">{box_svg}</svg></div>

  <h2>5 · 逐图明细</h2>
  <div class="card" style="padding:8px 16px 16px; max-height:460px; overflow:auto;">
    <table>
      <thead><tr><th>文件</th><th>期望</th><th>实际</th><th>结果</th><th>圆形数</th><th>散落数</th><th>说明</th></tr></thead>
      <tbody>{''.join(table_rows_html)}</tbody>
    </table>
  </div>

  <div class="notice">
    <b>结论：</b>
    <ul>
      <li>总体准确率 <b>{fmt_pct(acc)}</b>，整齐类召回 <b>{fmt_pct(tidy_acc)}</b>，但不整齐类召回仅 <b>{fmt_pct(messy_acc)}</b>。</li>
      <li>{n_wrong} 个误判中，<b>{miss_messy} 个是「漏检不整齐」</b>（把不整齐误判为整齐，占比 {fmt_pct(miss_messy/n_wrong)}）—— 若用于自动控制，意味着系统会放过不整齐的枪头直接进入下一步，是最需要改进的方向。</li>
      <li>仅 {false_alarm} 个「误报不整齐」（整齐被判为不整齐），后果只是多震荡一轮，代价较小。</li>
      <li>漏检样本的检测结果均为「圆形齐整、无散落」，说明现有圆检测无法刻画这类不整齐形态（如枪头倾斜/错位但俯视仍为圆），建议补充相邻间距均匀性检查或角度一致性特征。</li>
    </ul>
  </div>
</div>
<script>
function toggleTheme() {{
  const r = document.documentElement;
  r.setAttribute('data-theme', r.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}}
</script>
</body>
</html>"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html_doc)

print("已生成报告:", OUT_PATH)
print(f"  样本 {N} | 准确率 {fmt_pct(acc)} | 整齐召回 {fmt_pct(tidy_acc)} | 不整齐召回 {fmt_pct(messy_acc)}")
print(f"  混淆矩阵: TP={tp} FN={fn} FP={fp} TN={tn}")
