"""
Static HTML report generator.

Reads analysis outputs from a topic_analysis/ directory and generates a
self-contained HTML file (images embedded as base64).

Usage:
    from modules.reporters.html_report import generate_html_report

    generate_html_report(
        channel_id="UCxxx",
        channel_title="My Channel",
        report_dir="./reports/UCxxx",          # summary.json lives here
        topic_dir="./reports/UCxxx/topic_analysis",
        output_path="./reports/UCxxx/report.html",
    )
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _b64_img(path: Path) -> str | None:
    """Return a data-URI string for the image at path, or None if missing."""
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "png")
    return f"data:image/{mime};base64,{data}"


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_metrics(topic_dir: Path) -> dict:
    """Load key numbers from video_metrics.csv."""
    p = topic_dir / "video_metrics.csv"
    if not p.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(p)
        return {
            "video_count": len(df),
            "median_halflife": round(df["half_life"].median(), 3),
            "median_active_window": round(df["active_window"].median(), 3),
            "fastest": df.nsmallest(3, "half_life")[["title", "half_life"]].to_dict("records"),
            "slowest": df.nlargest(3, "half_life")[["title", "half_life"]].to_dict("records"),
        }
    except Exception:
        return {}


def _img_section(label: str, b64: str | None) -> str:
    if not b64:
        return f'<p class="missing">[{label} not available]</p>'
    return f'<img src="{b64}" alt="{label}" class="chart-img">'


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #f5f4f0; --surface: #fff; --border: #1a1a1a; --border-light: #d0cec9;
  --text: #1a1a1a; --muted: #888; --accent: #d4601a;
  --mono: 'DM Mono', monospace; --serif: 'Noto Serif TC', serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--serif);
       font-weight: 300; line-height: 1.8; }
.wrap { max-width: 860px; margin: 0 auto; padding: 0 2rem 4rem; }
/* masthead */
.masthead { border-bottom: 3px solid var(--border); padding: 1.5rem 0 1rem; }
.masthead-row { display: flex; align-items: baseline; justify-content: space-between; }
.masthead h1 { font-size: 42px; font-weight: 700; letter-spacing: -0.02em; }
.masthead-meta { font-family: var(--mono); font-size: 10px; color: var(--muted);
                 text-align: right; line-height: 1.6; }
hr.rule { border: none; border-top: 1px solid var(--border); margin-top: .75rem; }
/* sections */
.section { border-top: 2px solid var(--border); margin-top: 2.5rem; padding-top: 1.5rem; }
.section-label { font-family: var(--mono); font-size: 10px; letter-spacing: .12em;
                 text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
.section h2 { font-size: 22px; font-weight: 600; margin-bottom: 1rem; }
/* stat strip */
.stats { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1.2rem 0; }
.stat { border-left: 3px solid var(--accent); padding-left: .8rem; }
.stat-num { font-size: 28px; font-weight: 700; line-height: 1; }
.stat-label { font-family: var(--mono); font-size: 10px; color: var(--muted); }
/* charts */
.chart-img { width: 100%; border: 1px solid var(--border-light);
             border-radius: 2px; margin: 1rem 0; }
.missing { color: var(--muted); font-style: italic; font-size: 13px; margin: .5rem 0; }
/* table */
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 1rem 0; }
th { text-align: left; border-bottom: 2px solid var(--border);
     font-family: var(--mono); font-size: 10px; letter-spacing: .1em;
     text-transform: uppercase; padding: .4rem .6rem; }
td { border-bottom: 1px solid var(--border-light); padding: .4rem .6rem;
     vertical-align: top; }
"""

_FONTS = (
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Noto+Serif+TC:wght@300;400;600&family=DM+Mono:wght@300;400"
    '&display=swap" rel="stylesheet">'
)


def _render_anomaly_table(rows: list[dict], col: str, label: str) -> str:
    if not rows:
        return ""
    ths = "<tr><th>影片標題</th><th>" + label + "</th></tr>"
    tds = "".join(
        f"<tr><td>{r['title'][:60]}</td><td>{r[col]:.3f}</td></tr>"
        for r in rows
    )
    return f"<table>{ths}{tds}</table>"


# ── public API ────────────────────────────────────────────────────────────────

def generate_html_report(
    channel_id: str,
    channel_title: str,
    report_dir: str | Path,
    topic_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Generate a self-contained HTML analysis report.

    Parameters
    ----------
    channel_id     : YouTube channel ID
    channel_title  : Human-readable channel name
    report_dir     : Directory containing summary.json
    topic_dir      : Directory containing PNGs and CSVs from topic analysis
    output_path    : Where to write the output HTML file

    Returns the path to the written file.
    """
    report_dir = Path(report_dir)
    topic_dir = Path(topic_dir)
    output_path = Path(output_path)

    summary = _load_json(report_dir / "summary.json")
    metrics = _load_metrics(topic_dir)

    # Load images
    img_time = _b64_img(topic_dir / "time_analysis_overview.png")
    img_halflife = _b64_img(topic_dir / "halflife_analysis.png")
    img_cumulative = _b64_img(topic_dir / "cumulative_curves_comparison.png")
    img_similarity = _b64_img(topic_dir / "topic_similarity_matrix.png")
    img_topic_dist = _b64_img(topic_dir / "topic_distribution_comparison.png")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    last_updated = summary.get("last_updated", "—")[:10]
    total_videos = summary.get("stats", {}).get("total_videos", "—")
    total_comments = summary.get("stats", {}).get("total_comments_analyzed", "—")

    # ── stat strip ────────────────────────────────────────────────────────────
    stat_items = [
        (str(total_videos), "影片數"),
        (str(total_comments), "已分析留言"),
    ]
    if metrics.get("median_halflife") is not None:
        stat_items.append((f"{metrics['median_halflife']}天", "留言半衰期中位數"))
    if metrics.get("median_active_window") is not None:
        stat_items.append((f"{metrics['median_active_window']}天", "活躍窗口中位數"))

    stats_html = "".join(
        f'<div class="stat"><div class="stat-num">{n}</div>'
        f'<div class="stat-label">{l}</div></div>'
        for n, l in stat_items
    )

    # ── anomaly tables ────────────────────────────────────────────────────────
    fastest_tbl = _render_anomaly_table(
        metrics.get("fastest", []), "half_life", "半衰期 (天)"
    )
    slowest_tbl = _render_anomaly_table(
        metrics.get("slowest", []), "half_life", "半衰期 (天)"
    )

    # ── assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{channel_title} — 分析報告</title>
{_FONTS}
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

  <!-- masthead -->
  <div class="masthead">
    <div class="masthead-row">
      <h1>{channel_title}</h1>
      <div class="masthead-meta">
        頻道 ID: {channel_id}<br>
        資料截止: {last_updated}<br>
        產生時間: {now_str}
      </div>
    </div>
    <hr class="rule">
  </div>

  <!-- overview stats -->
  <div class="section">
    <div class="section-label">概覽</div>
    <h2>數據摘要</h2>
    <div class="stats">{stats_html}</div>
  </div>

  <!-- section 1: comment timing -->
  <div class="section">
    <div class="section-label">Section 01</div>
    <h2>留言時間分析 — 觀眾反應速度</h2>
    <p>分析各影片留言的時間分布，了解粉絲核心群的即時反應模式。</p>
    {_img_section("時間分析概覽", img_time)}
    {_img_section("累積曲線比較", img_cumulative)}
  </div>

  <!-- section 2: identity gap -->
  <div class="section">
    <div class="section-label">Section 02</div>
    <h2>身份落差分析 — 標題 vs 留言主題</h2>
    <p>以 BERTopic 分別對影片標題（Internal Identity）和觀眾留言（External Identity）
       建立主題模型，透過餘弦相似度矩陣量化兩者的語義距離。</p>
    {_img_section("主題分布比較", img_topic_dist)}
    {_img_section("相似度矩陣", img_similarity)}
  </div>

  <!-- section 3: halflife -->
  <div class="section">
    <div class="section-label">Section 03</div>
    <h2>留言半衰期分析</h2>
    <p>計算每部影片的「留言半衰期」（達到 50% 留言量所需天數）與活躍窗口，
       找出互動模式的異常個案。</p>
    {_img_section("半衰期分析", img_halflife)}

    <h3 style="margin-top:1.2rem;font-size:15px;">反應最快的影片</h3>
    {fastest_tbl or '<p class="missing">資料不足</p>'}

    <h3 style="margin-top:1.2rem;font-size:15px;">討論持續最久的影片</h3>
    {slowest_tbl or '<p class="missing">資料不足</p>'}
  </div>

</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  [html] written {output_path}")
    return output_path
