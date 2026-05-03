"""
Identity-gap analysis: BERTopic-based topic modeling on titles vs comments.

Stage 1 (macro)  – IdentityGapAnalyzer
    Builds one topic model on video titles and one on all comments,
    then measures semantic distance between the two topic spaces.

Stage 2 (micro)  – VideoLevelAnalyzer
    Builds per-video topic models and compares top-liked vs average comments.

Usage:
    from modules.analysis.identity_gap import IdentityGapAnalyzer, VideoLevelAnalyzer

    # Stage 1
    gap = IdentityGapAnalyzer("UCxxx", data_dir="./data")
    gap.run(output_dir="./reports/UCxxx/topic_analysis")

    # Stage 2
    micro = VideoLevelAnalyzer("UCxxx", data_dir="./data")
    micro.run(["videoId1", "videoId2"],
              output_dir="./reports/UCxxx/topic_analysis/micro")
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ── shared data loader ────────────────────────────────────────────────────────

def _load_channel_data(channel_id: str, data_dir: Path) -> tuple[list[dict], list[dict], dict[str, list]]:
    """
    Load video metadata, flattened comment list, and per-video comment dict
    from data/raw/<channel_id>/.

    Returns (videos, flat_comments, comments_by_video_id).
    """
    videos_path = data_dir / "raw" / "videos" / channel_id / "video_list.json"
    comments_dir = data_dir / "raw" / "comments" / channel_id

    with open(videos_path, encoding="utf-8") as f:
        videos: list[dict] = json.load(f)

    flat_comments: list[dict] = []
    comments_by_id: dict[str, list] = {}
    for p in sorted(comments_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            comments_by_id[p.stem] = data
            flat_comments.extend(data)

    return videos, flat_comments, comments_by_id


# ── Stage 1: macro identity-gap ───────────────────────────────────────────────

class IdentityGapAnalyzer:
    """
    Macro-level identity gap analysis.

    Trains a BERTopic model on video titles and another on comments,
    then computes a cosine-similarity matrix between the two topic spaces.
    """

    def __init__(self, channel_id: str, data_dir: str | Path):
        self.channel_id = channel_id
        self.data_dir = Path(data_dir)

        self.videos, self.flat_comments, _ = _load_channel_data(
            channel_id, self.data_dir
        )
        self.titles: list[str] = [v["title"] for v in self.videos]
        self.comments: list[str] = [
            c["text"] for c in self.flat_comments if c.get("text")
        ]

        print(f"    {len(self.titles)} titles / {len(self.comments)} comments loaded")

        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.title_model = None
        self.comment_model = None
        self.title_topics: list[int] = []
        self.comment_topics: list[int] = []

    def build_topic_models(
        self,
        n_topics_title: int = 10,
        n_topics_comment: int = 15,
    ) -> None:
        from bertopic import BERTopic

        print("\n  Building title topic model...")
        self.title_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=n_topics_title,
            verbose=True,
            calculate_probabilities=True,
        )
        self.title_topics, _ = self.title_model.fit_transform(self.titles)

        print("\n  Building comment topic model...")
        self.comment_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=n_topics_comment,
            verbose=True,
            calculate_probabilities=True,
        )
        self.comment_topics, _ = self.comment_model.fit_transform(self.comments)

    def visualize_topics(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        fig1 = self.title_model.visualize_barchart(top_n_topics=10)
        fig1.write_html(str(output_dir / "title_topics_barchart.html"))
        print(f"  [gap] written title_topics_barchart.html")

        fig2 = self.comment_model.visualize_barchart(top_n_topics=15)
        fig2.write_html(str(output_dir / "comment_topics_barchart.html"))
        print(f"  [gap] written comment_topics_barchart.html")

        self._plot_topic_distribution(output_dir)

    def _plot_topic_distribution(self, output_dir: Path) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        title_counts = (
            pd.Series(self.title_topics)
            .value_counts()
            .sort_index()
            .drop(-1, errors="ignore")
        )
        ax1.bar(range(len(title_counts)), title_counts.values,
                color="steelblue", alpha=0.7)
        ax1.set_xlabel("主題ID")
        ax1.set_ylabel("影片數量")
        ax1.set_title("標題主題分布 (Internal Identity)")
        ax1.grid(axis="y", alpha=0.3)

        comment_counts = (
            pd.Series(self.comment_topics)
            .value_counts()
            .sort_index()
            .drop(-1, errors="ignore")
        )
        ax2.bar(range(len(comment_counts)), comment_counts.values,
                color="coral", alpha=0.7)
        ax2.set_xlabel("主題ID")
        ax2.set_ylabel("評論數量")
        ax2.set_title("評論主題分布 (External Identity)")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "topic_distribution_comparison.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [gap] written topic_distribution_comparison.png")

    def find_gap_topics(self, output_dir: Path) -> tuple[np.ndarray, list, list]:
        """Compute cross-space cosine similarity and save heatmap."""
        from sklearn.metrics.pairwise import cosine_similarity

        def _topic_embeddings(model):
            ids, embs = [], []
            for tid in model.get_topics():
                if tid == -1:
                    continue
                words = [w[0] for w in model.get_topic(tid)[:10]]
                embs.append(self.embedding_model.encode(" ".join(words)))
                ids.append(tid)
            return ids, np.array(embs)

        title_ids, title_embs = _topic_embeddings(self.title_model)
        comment_ids, comment_embs = _topic_embeddings(self.comment_model)
        sim_matrix = cosine_similarity(title_embs, comment_embs)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(sim_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(comment_ids)))
        ax.set_yticks(range(len(title_ids)))
        ax.set_xticklabels([f"C{i}" for i in comment_ids], rotation=45)
        ax.set_yticklabels([f"T{i}" for i in title_ids])
        ax.set_xlabel("評論主題")
        ax.set_ylabel("標題主題")
        ax.set_title("標題主題 vs 評論主題相似度矩陣")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / "topic_similarity_matrix.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [gap] written topic_similarity_matrix.png")

        return sim_matrix, title_ids, comment_ids

    def export_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        title_df = pd.DataFrame({
            "video_index": range(len(self.titles)),
            "video_id": [v["video_id"] for v in self.videos],
            "title": self.titles,
            "topic": self.title_topics,
        })
        title_df.to_csv(output_dir / "title_topic_assignments.csv",
                        index=False, encoding="utf-8-sig")
        print(f"  [gap] written title_topic_assignments.csv")

        sample = min(1000, len(self.comments))
        comment_df = pd.DataFrame({
            "comment": self.comments[:sample],
            "topic": self.comment_topics[:sample],
        })
        comment_df.to_csv(output_dir / "comment_topic_assignments_sample.csv",
                          index=False, encoding="utf-8-sig")
        print(f"  [gap] written comment_topic_assignments_sample.csv")

    def run(
        self,
        output_dir: str | Path,
        n_topics_title: int = 10,
        n_topics_comment: int = 15,
    ) -> None:
        """Run full stage-1 pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Identity gap analysis (macro): {self.channel_id}")

        self.build_topic_models(n_topics_title, n_topics_comment)
        self.visualize_topics(output_dir)
        self.find_gap_topics(output_dir)
        self.export_results(output_dir)

        print(f"\n  Stage 1 complete → {output_dir}")


# ── Stage 2: per-video micro analysis ─────────────────────────────────────────

class VideoLevelAnalyzer:
    """
    Micro-level per-video topic modeling.

    For each target video: builds an independent BERTopic model,
    compares top-liked vs average comments, and exports an interactive chart.
    """

    def __init__(self, channel_id: str, data_dir: str | Path):
        self.channel_id = channel_id
        self.data_dir = Path(data_dir)

        self.videos, _, self.comments_by_id = _load_channel_data(
            channel_id, self.data_dir
        )
        self.video_lookup: dict[str, dict] = {
            v["video_id"]: v for v in self.videos
        }

        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    def run(
        self,
        video_ids: list[str],
        output_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Analyse target videos and write per-video HTML charts.
        Returns a DataFrame with per-video metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Identity gap analysis (micro): {self.channel_id}")

        results = []
        for vid in video_ids:
            r = self._analyze_single(vid, output_dir)
            if r:
                results.append(r)

        if not results:
            print("  No results – check video IDs.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.to_csv(output_dir / "video_comparison.csv",
                  index=False, encoding="utf-8-sig")
        print(f"  [micro] written video_comparison.csv")
        return df

    def _analyze_single(self, video_id: str, output_dir: Path) -> dict | None:
        from bertopic import BERTopic

        info = self.video_lookup.get(video_id)
        if not info:
            print(f"  [micro] video {video_id} not found – skipped")
            return None

        raw = self.comments_by_id.get(video_id, [])
        texts = [c["text"] for c in raw if c.get("text")]
        if len(texts) < 10:
            print(f"  [micro] {video_id}: too few comments ({len(texts)}) – skipped")
            return None

        print(f"\n  Analysing {video_id}: {info['title'][:60]}")

        model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics="auto",
            min_topic_size=10,
            verbose=False,
        )
        topics, _ = model.fit_transform(texts)

        fig = model.visualize_barchart(top_n_topics=8)
        fig.write_html(str(output_dir / f"{video_id}_topics.html"))
        print(f"  [micro] written {video_id}_topics.html")

        # top-liked vs average
        top_raw = sorted(raw, key=lambda c: c.get("like_count", 0), reverse=True)[:50]
        top_texts = [c["text"] for c in top_raw if c.get("text")]
        if len(top_texts) >= 10:
            top_topics = model.transform(top_texts)[0]
            all_dist = pd.Series(topics).value_counts(normalize=True)
            top_dist = pd.Series(top_topics).value_counts(normalize=True)
            comparison = pd.DataFrame({"整體": all_dist, "高讚": top_dist}).fillna(0)
            print("  Top-liked vs all topic distribution:\n", comparison.head())

        topic_info = model.get_topic_info()
        keywords = []
        for tid in range(min(5, len(topic_info) - 1)):
            if tid in model.get_topics():
                words = [w[0] for w in model.get_topic(tid)[:5]]
                keywords.append(", ".join(words))

        return {
            "video_id": video_id,
            "title": info["title"],
            "comment_count": len(texts),
            "num_topics": len(topic_info) - 1,
            "top_topics_keywords": " | ".join(keywords),
            "view_count": info.get("view_count", 0),
            "like_count": info.get("like_count", 0),
        }
