# YouTube Channel Analysis Tool

Fetch transcripts and comments from a YouTube channel, analyse audience profiles and brand positioning with a local LLM (Ollama), and run BERTopic-based topic modelling to measure the gap between a creator's self-presentation and audience expectations.
Results are displayed in a browser-based web UI and saved as Markdown / HTML reports.

## Features

| Feature | Description |
|---------|-------------|
| Audience profile analysis | Infer audience age, interests, pain points, and language patterns from comments |
| Brand positioning analysis | Extract content themes, communication style, and value propositions from transcripts |
| Comment timing analysis | Time distribution, cumulative curves, and per-video halflife for all comments |
| Identity gap analysis | BERTopic topic modelling on titles vs. comments to quantify creator–audience divergence |
| HTML report generation | Self-contained report page (images embedded as base64) from analysis outputs |
| Incremental updates | Fetch only videos published since the last run; cached data is reused automatically |
| Channel management | View, re-analyse, or delete any previously analysed channel from the UI |
| Real-time progress | Long-running tasks stream progress via SSE |

---

## Prerequisites

### 1. Ollama (local LLM)

Install Ollama from [ollama.com](https://ollama.com), then pull a model:
```bash
ollama pull gemma3:12b
```

### 2. YouTube Data API v3 key

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a project → Enable **YouTube Data API v3** → Create an API key

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

> Topic analysis (`comment_time`, `identity_gap`) additionally requires `bertopic`, `sentence-transformers`, and `plotly` — all listed in `requirements.txt`.

### 4. `.env` file

```
YOUTUBE_API_KEY=your_key_here
LOCAL_LLM_URL=http://localhost:11434/v1   # optional, this is the default
```

---

## Quick Start

**Windows:** double-click `start.bat`

**Manual:**
```bash
# Make sure Ollama is already running, then:
python server.py
```

Open `http://localhost:5000` in your browser.

---

## Usage

### Analysing a new channel

1. Select an Ollama model from the dropdown
2. Enter a channel handle (e.g. `@ChannelName`) or channel ID
3. Set the maximum number of videos to analyse
4. Optionally skip audience or brand analysis
5. Click **開始抓取並分析**

### Managing existing channels

Select a channel from the dropdown, then use the action buttons:

| Button | What it does |
|--------|-------------|
| 查看結果 | Load and display the last analysis results |
| 補抓並分析 | Fetch videos published since the last run, then re-analyse |
| 直接分析 | Re-run analysis on existing cached data (no new fetch) |
| 僅補抓 | Fetch new videos and save to cache, without running analysis |
| 主題分析 | Run topic modelling + timing analysis (see below) |
| 刪除 | Delete all data and reports for this channel |

### Topic analysis

Topic analysis is available from the UI via the **主題分析** button on the channel management panel, or can be triggered programmatically:

```python
from modules.analysis.comment_time import analyze_comment_timing
from modules.analysis.identity_gap import IdentityGapAnalyzer, VideoLevelAnalyzer
from modules.reporters.html_report import generate_html_report

CHANNEL_ID  = "UCxxx"
DATA_DIR    = "./data"
REPORT_DIR  = f"./reports/{CHANNEL_ID}"
TOPIC_DIR   = f"{REPORT_DIR}/topic_analysis"

# Comment timing: halflife, cumulative curves
stats = analyze_comment_timing(CHANNEL_ID, DATA_DIR, TOPIC_DIR)

# Stage 1 – macro identity gap (full channel)
gap = IdentityGapAnalyzer(CHANNEL_ID, DATA_DIR)
gap.run(TOPIC_DIR, n_topics_title=10, n_topics_comment=15)

# Stage 2 – per-video micro analysis (optional)
micro = VideoLevelAnalyzer(CHANNEL_ID, DATA_DIR)
micro.run(["videoId1", "videoId2"], output_dir=f"{TOPIC_DIR}/micro")

# Generate self-contained HTML report
generate_html_report(
    channel_id=CHANNEL_ID,
    channel_title="頻道名稱",
    report_dir=REPORT_DIR,
    topic_dir=TOPIC_DIR,
    output_path=f"{REPORT_DIR}/report.html",
)
```

---

## Output files

### LLM analysis (`reports/<channel_id>/`)

| File | Description |
|------|-------------|
| `summary.json` | Channel metadata and stats used by the web UI |
| `audience_report.md` | Audience profile report |
| `brand_report.md` | Brand positioning report |
| `comments.csv` | All fetched comments |
| `transcripts.csv` | All fetched transcripts |

### Topic analysis (`reports/<channel_id>/topic_analysis/`)

| File | Description |
|------|-------------|
| `time_analysis_overview.png` | Comment time distribution (4-panel) |
| `cumulative_curves_comparison.png` | Per-video cumulative curves + average |
| `halflife_analysis.png` | Halflife distribution and scatter plots |
| `video_metrics.csv` | Per-video halflife and active-window metrics |
| `topic_distribution_comparison.png` | Title vs. comment topic distribution |
| `topic_similarity_matrix.png` | Cross-space cosine similarity heatmap |
| `title_topics_barchart.html` | Interactive title topic chart (BERTopic) |
| `comment_topics_barchart.html` | Interactive comment topic chart (BERTopic) |
| `title_topic_assignments.csv` | Per-video topic assignment |
| `comment_topic_assignments_sample.csv` | Comment topic assignment (first 1000) |
| `micro/<video_id>_topics.html` | Per-video interactive chart (Stage 2) |
| `micro/video_comparison.csv` | Stage 2 per-video summary metrics |

### Raw data cache (`data/raw/`)

```
data/raw/
├── videos/<channel_id>/video_list.json
├── comments/<channel_id>/<video_id>.json
└── transcripts/<channel_id>/<video_id>.json
```

Raw data and reports are excluded from git (see `.gitignore`).

---

## `.env` reference

| Variable | Default | Description |
|----------|---------|-------------|
| `YOUTUBE_API_KEY` | — | YouTube Data API v3 key **(required)** |
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | Ollama API base URL |
| `MAX_VIDEOS` | `20` | Default maximum videos to fetch |
| `MAX_COMMENTS_PER_VIDEO` | `100` | Maximum comments per video |
| `TRANSCRIPT_LANGUAGES` | `ja,zh-Hant,zh-Hans,en` | Language priority (comma-separated) |
| `DATA_DIR` | `./data` | Raw data cache directory |
| `REPORTS_DIR` | `./reports` | Report output directory |

---

## Project structure

```
├── server.py                        # Flask backend + API endpoints
├── index.html                       # Web UI frontend
├── start.bat                        # Windows launcher (Ollama + Flask + browser)
├── config/
│   └── settings.py                  # Settings dataclass + .env loader
└── modules/
    ├── fetcher.py                    # Fetch orchestration (full + incremental)
    ├── analyzer.py                   # LLM analysis orchestration + report writing
    ├── analysis/
    │   ├── audience.py               # Audience profile LLM prompt + parser
    │   ├── brand.py                  # Brand positioning LLM prompt + parser
    │   ├── comment_time.py           # Comment timing analysis (halflife, curves)
    │   └── identity_gap.py           # BERTopic identity-gap analysis (Stage 1 + 2)
    ├── collectors/
    │   ├── youtube_api.py            # YouTube Data API client
    │   ├── transcript.py             # Transcript fetcher
    │   └── comments.py              # Comment fetcher
    ├── llm_providers/
    │   ├── base.py                   # BaseLLMClient interface
    │   └── local_llm.py             # Ollama client
    ├── reporters/
    │   ├── markdown.py               # Markdown report writer
    │   ├── csv_reporter.py           # CSV export writer
    │   └── html_report.py            # Static HTML report generator
    └── storage/
        ├── file_store.py             # JSON / CSV read-write
        └── cache.py                  # Cache helpers
```

---

## Troubleshooting

**Model dropdown shows "無可用模型"**
Ollama is not running or has no models installed. Run `ollama pull gemma3:12b` and restart the server.

**Analysis fails with quota error**
The YouTube Data API has a daily quota. Wait until the next day or create a new API key.

**Prompt truncation / slow analysis**
Use a model with a larger context window, or reduce the number of videos analysed.

**Chinese / Japanese characters appear as gibberish**
Run `chcp 65001` in your terminal before starting, or use `start.bat` which sets this automatically.

**BERTopic / topic analysis fails**
Ensure all optional dependencies are installed: `pip install bertopic sentence-transformers plotly`.
The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) is downloaded automatically on first run.
