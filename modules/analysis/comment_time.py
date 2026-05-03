"""
Comment timing analysis: time distribution, cumulative curves, per-video halflife.

Usage:
    from modules.analysis.comment_time import analyze_comment_timing

    stats = analyze_comment_timing(
        channel_id="UCxxx",
        data_dir="./data",
        output_dir="./reports/UCxxx/topic_analysis",
    )
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── data loader ────────────────────────────────────────────────────────────────

def load_merged_df(channel_id: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load video metadata + comments from data/raw/<channel_id>/ and merge into
    a single DataFrame with a `days_since_publish` column.

    Returns (df_merged, df_videos).
    """
    videos_path = data_dir / "raw" / "videos" / channel_id / "video_list.json"
    comments_dir = data_dir / "raw" / "comments" / channel_id

    with open(videos_path, encoding="utf-8") as f:
        videos = json.load(f)
    df_videos = pd.DataFrame(videos)
    df_videos["published_at"] = pd.to_datetime(df_videos["published_at"], utc=True)

    all_comments: list[dict] = []
    for p in sorted(comments_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for c in data:
                c.setdefault("video_id", p.stem)
            all_comments.extend(data)

    df_comments = pd.DataFrame(all_comments)
    df_comments["published_at"] = pd.to_datetime(df_comments["published_at"], utc=True)

    df_merged = df_comments.merge(
        df_videos[["video_id", "published_at", "title", "view_count"]],
        on="video_id",
        suffixes=("_comment", "_video"),
    )
    df_merged["days_since_publish"] = (
        df_merged["published_at_comment"] - df_merged["published_at_video"]
    ).dt.total_seconds() / 86400

    return df_merged, df_videos


# ── internal plot helpers ──────────────────────────────────────────────────────

def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_time_overview(df_merged: pd.DataFrame, output_dir: Path) -> None:
    _setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(df_merged["days_since_publish"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("距離發布天數")
    axes[0, 0].set_ylabel("評論數")
    axes[0, 0].set_title("所有影片的評論時間分布")
    median_all = df_merged["days_since_publish"].median()
    axes[0, 0].axvline(median_all, color="red", linestyle="--",
                       label=f"中位數: {median_all:.2f}天")
    axes[0, 0].legend()

    early = df_merged[df_merged["days_since_publish"] <= 7]
    axes[0, 1].hist(early["days_since_publish"], bins=30, edgecolor="black",
                    alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("距離發布天數")
    axes[0, 1].set_ylabel("評論數")
    axes[0, 1].set_title("前7天的評論分布")
    median_early = early["days_since_publish"].median()
    axes[0, 1].axvline(median_early, color="red", linestyle="--",
                       label=f"中位數: {median_early:.2f}天")
    axes[0, 1].legend()

    sorted_days = np.sort(df_merged["days_since_publish"])
    cumulative = np.arange(1, len(sorted_days) + 1) / len(sorted_days) * 100
    axes[1, 0].plot(sorted_days, cumulative, linewidth=2)
    axes[1, 0].set_xlabel("距離發布天數")
    axes[1, 0].set_ylabel("累積評論百分比 (%)")
    axes[1, 0].set_title("評論累積分布")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(50, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(90, color="orange", linestyle="--", alpha=0.5)
    day_50 = float(sorted_days[np.argmin(np.abs(cumulative - 50))])
    day_90 = float(sorted_days[np.argmin(np.abs(cumulative - 90))])
    axes[1, 0].text(day_50, 55, f"50%在{day_50:.1f}天內", fontsize=9)
    axes[1, 0].text(day_90, 85, f"90%在{day_90:.1f}天內", fontsize=9)

    video_stats = df_merged.groupby("video_id").agg(
        days_median=("days_since_publish", "median"),
        published_at_video=("published_at_video", "first"),
    ).reset_index().sort_values("published_at_video")
    axes[1, 1].scatter(video_stats["published_at_video"],
                       video_stats["days_median"], alpha=0.6)
    axes[1, 1].set_xlabel("影片發布時間")
    axes[1, 1].set_ylabel("評論中位數時間(天)")
    axes[1, 1].set_title("不同時期影片的觀眾反應速度")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "time_analysis_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [time] written time_analysis_overview.png")


def _compute_video_metrics(df_merged: pd.DataFrame,
                            df_videos: pd.DataFrame) -> pd.DataFrame:
    records = []
    for video_id in df_merged["video_id"].unique():
        vc = df_merged[df_merged["video_id"] == video_id].copy()
        vc = vc.sort_values("days_since_publish")
        vc["cumulative_pct"] = (
            np.arange(1, len(vc) + 1) / len(vc) * 100
        )

        half_life = float(
            vc[vc["cumulative_pct"] >= 50].iloc[0]["days_since_publish"]
        ) if len(vc) > 0 else float("nan")

        try:
            t_10 = vc[vc["cumulative_pct"] >= 10].iloc[0]["days_since_publish"]
            t_90 = vc[vc["cumulative_pct"] >= 90].iloc[0]["days_since_publish"]
            active_window = float(t_90 - t_10)
        except IndexError:
            active_window = float("nan")

        vi = df_videos[df_videos["video_id"] == video_id]
        if vi.empty:
            continue
        vi = vi.iloc[0]

        records.append({
            "video_id": video_id,
            "title": vi["title"],
            "view_count": vi.get("view_count", 0),
            "published_at": vi["published_at"],
            "comment_count": len(vc),
            "half_life": half_life,
            "active_window": active_window,
            "comments_data": vc[["days_since_publish", "cumulative_pct"]].values,
        })

    return pd.DataFrame(records)


def _plot_cumulative_curves(df_merged: pd.DataFrame,
                             df_metrics: pd.DataFrame,
                             output_dir: Path) -> None:
    _setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for _, row in df_metrics.iterrows():
        d = row["comments_data"]
        axes[0].plot(d[:, 0], d[:, 1], alpha=0.3, linewidth=0.8, color="gray")

    max_day = df_merged["days_since_publish"].max()
    time_pts = np.linspace(0, max_day, 100)
    avg_cum = []
    for t in time_pts:
        pcts = []
        for _, row in df_metrics.iterrows():
            d = row["comments_data"]
            if len(d) > 0:
                p = d[d[:, 0] <= t, 1]
                pcts.append(p[-1] if len(p) > 0 else 0.0)
        avg_cum.append(float(np.mean(pcts)) if pcts else 0.0)

    axes[0].plot(time_pts, avg_cum, color="red", linewidth=3,
                 label="平均曲線", zorder=10)
    axes[0].set_xlabel("距離發布天數")
    axes[0].set_ylabel("累積評論百分比 (%)")
    axes[0].set_title("所有影片的累積評論曲線")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, 10)

    for _, row in df_metrics.iterrows():
        d = row["comments_data"]
        d24 = d[d[:, 0] <= 1]
        if len(d24) > 0:
            axes[1].plot(d24[:, 0] * 24, d24[:, 1], alpha=0.3,
                         linewidth=0.8, color="gray")

    time_pts_24h = np.linspace(0, 1, 100)
    avg_cum_24h = []
    for t in time_pts_24h:
        pcts = []
        for _, row in df_metrics.iterrows():
            d = row["comments_data"]
            if len(d) > 0:
                p = d[d[:, 0] <= t, 1]
                pcts.append(p[-1] if len(p) > 0 else 0.0)
        avg_cum_24h.append(float(np.mean(pcts)) if pcts else 0.0)

    axes[1].plot(time_pts_24h * 24, avg_cum_24h, color="red", linewidth=3,
                 label="平均曲線", zorder=10)
    axes[1].set_xlabel("距離發布小時數")
    axes[1].set_ylabel("累積評論百分比 (%)")
    axes[1].set_title("前24小時的累積曲線(聚焦)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_curves_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [time] written cumulative_curves_comparison.png")


def _plot_halflife_analysis(df_metrics: pd.DataFrame, output_dir: Path) -> None:
    _setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    hl = df_metrics["half_life"].dropna()
    axes[0, 0].hist(hl, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(hl.median(), color="red", linestyle="--",
                       label=f"中位數: {hl.median():.3f}天")
    axes[0, 0].set_xlabel("半衰期 (天)")
    axes[0, 0].set_ylabel("影片數")
    axes[0, 0].set_title("半衰期分布")
    axes[0, 0].legend()

    axes[0, 1].scatter(df_metrics["published_at"], df_metrics["half_life"], alpha=0.6)
    axes[0, 1].set_xlabel("影片發布時間")
    axes[0, 1].set_ylabel("半衰期 (天)")
    axes[0, 1].set_title("半衰期隨時間變化")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(df_metrics["view_count"], df_metrics["half_life"], alpha=0.6)
    axes[1, 0].set_xlabel("觀看數")
    axes[1, 0].set_ylabel("半衰期 (天)")
    axes[1, 0].set_title("半衰期 vs 觀看數")
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    aw = df_metrics["active_window"].dropna()
    axes[1, 1].hist(aw, bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[1, 1].axvline(aw.median(), color="red", linestyle="--",
                       label=f"中位數: {aw.median():.3f}天")
    axes[1, 1].set_xlabel("活躍窗口 (10%-90%的時間, 天)")
    axes[1, 1].set_ylabel("影片數")
    axes[1, 1].set_title("評論活躍窗口分布")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "halflife_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [time] written halflife_analysis.png")


# ── public API ─────────────────────────────────────────────────────────────────

def analyze_comment_timing(
    channel_id: str,
    data_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Run full comment timing analysis for a channel.

    Reads from  data_dir/raw/{comments,videos}/<channel_id>/
    Writes to   output_dir/  (PNGs + video_metrics.csv)
    Returns summary stats dict.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Comment timing analysis: {channel_id}")
    df_merged, df_videos = load_merged_df(channel_id, data_dir)
    print(f"    {len(df_merged)} comments / {len(df_videos)} videos")

    _plot_time_overview(df_merged, output_dir)

    df_metrics = _compute_video_metrics(df_merged, df_videos)
    _plot_cumulative_curves(df_merged, df_metrics, output_dir)
    _plot_halflife_analysis(df_metrics, output_dir)

    df_metrics.drop(columns=["comments_data"]).to_csv(
        output_dir / "video_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  [time] written video_metrics.csv")

    sorted_days = np.sort(df_merged["days_since_publish"])
    cum = np.arange(1, len(sorted_days) + 1) / len(sorted_days) * 100

    return {
        "total_comments": len(df_merged),
        "total_videos": len(df_videos),
        "day_50pct": round(float(sorted_days[np.argmin(np.abs(cum - 50))]), 3),
        "day_90pct": round(float(sorted_days[np.argmin(np.abs(cum - 90))]), 3),
        "median_halflife_days": round(float(df_metrics["half_life"].median()), 3),
        "median_active_window_days": round(float(df_metrics["active_window"].median()), 3),
    }
