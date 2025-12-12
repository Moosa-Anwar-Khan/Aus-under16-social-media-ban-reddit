import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------- Configuration ----------

SENTIMENT_PATH = "Reddit/results/sentiment_outputs/reddit_with_sentiment.csv"
TOPIC_PATH = "Reddit/results/topic_modeling/lda_topics.csv"

MERGED_OUTPUT_PATH = (
    "Reddit/results/sentiment_topic_overlay/merged_sentiment_and_topics.csv"
)
REP_OUTPUT_TXT = "Reddit/results/sentiment_topic_overlay/lda_topics.txt"
REP_OUTPUT_MD = "Reddit/results/sentiment_topic_overlay/lda_topics.md"
TOPIC_SENTIMENT_PNG = (
    "Reddit/results/sentiment_topic_overlay/topic_sentiment_overlay.png"
)
TOPIC_SENTIMENT_MD = "Reddit/results/sentiment_topic_overlay/topic_sentiment_summary.md"

NUM_POSTS_PER_TOPIC = 5

TOPIC_LABELS = {
    0: "Politics & Governance",
    1: "Parenting & Youth Challenges",
    2: "Influencers & Platforms",
    3: "Digital Identity & Youth Voice",
    4: "Data Privacy & Regulation",
}

SENTIMENT_COLORS = {
    "Negative": "#B22222",
    "Neutral": "#B0B0B0",
    "Positive": "#4682B4",
}


# ---------- Step 1: Merge Sentiment and Topic Data ----------


def merge_datasets():
    sentiment_df = pd.read_csv(SENTIMENT_PATH)
    topics_df = pd.read_csv(TOPIC_PATH)

    merged = pd.merge(
        sentiment_df,
        topics_df[["Full_Text", "Dominant_Topic"]],
        on="Full_Text",
        how="inner",
    )

    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"[✓] Merged sentiment and topics saved to: {MERGED_OUTPUT_PATH}")
    return merged


# ---------- Step 2: Extract Representative Posts per Topic ----------


def export_representative_posts():
    df = pd.read_csv(TOPIC_PATH).dropna(subset=["Full_Text", "Dominant_Topic"])
    df["Dominant_Topic"] = df["Dominant_Topic"].astype(int)

    txt_lines, md_lines = [], []

    for topic, group in df.groupby("Dominant_Topic"):
        label = TOPIC_LABELS.get(topic, f"Topic {topic}")
        txt_lines.append(f"\n--- Topic {topic}: {label} ---\n")
        md_lines.append(f"## Topic {topic}: {label}\n")

        filtered = group[group["Full_Text"].str.len() > 100]
        selected = filtered.sort_values(
            by="Full_Text", key=lambda x: x.str.len(), ascending=False
        ).head(NUM_POSTS_PER_TOPIC)

        for i, post in enumerate(selected["Full_Text"], 1):
            txt_lines.append(f"Post {i}:\n{post}\n")
            md_lines.append(f"**Post {i}:**\n\n> {post.strip()}\n")

    with open(REP_OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.writelines(txt_lines)
    with open(REP_OUTPUT_MD, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print("[✓] Exported representative posts to:")
    print(f" - {REP_OUTPUT_TXT}\n - {REP_OUTPUT_MD}")


# ---------- Step 4: Sentiment Overlay Visualization ----------


def plot_sentiment_overlay():
    df = pd.read_csv(TOPIC_PATH)
    sentiment_df = pd.read_csv(SENTIMENT_PATH)

    df = df.merge(sentiment_df[["Full_Text", "Full_Label"]], on="Full_Text", how="left")
    sentiment_counts = (
        df.groupby(["Dominant_Topic", "Full_Label"]).size().unstack().fillna(0)
    )
    sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)

    sentiment_percent.index = sentiment_percent.index.map(TOPIC_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(sentiment_percent)

    for sentiment in ["Negative", "Neutral", "Positive"]:
        values = sentiment_percent[sentiment].values
        bars = ax.bar(
            sentiment_percent.index,
            values,
            bottom=bottom,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="black",
            label=sentiment,
        )
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + val / 2,
                    sentiment,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if sentiment != "Neutral" else "black",
                    weight="bold",
                )
        bottom = [i + j for i, j in zip(bottom, values)]

    ax.set_title("Sentiment Overlay by Topic", fontsize=16, weight="bold")
    ax.set_ylabel("Proportion of Sentiment", fontsize=13)
    ax.set_xlabel("Topic", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(sentiment_percent.index, rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend().remove()

    plt.tight_layout()
    plt.savefig(TOPIC_SENTIMENT_PNG, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[✓] Sentiment plot saved to {TOPIC_SENTIMENT_PNG}")

    # Markdown Summary
    md_lines = ["# Sentiment Overlay Report", ""]
    for topic in sorted(sentiment_counts.index):
        label = TOPIC_LABELS.get(topic, f"Topic {topic}")
        total = sentiment_counts.loc[topic].sum()
        md_lines.append(f"## Topic {topic}: {label}")
        for sentiment in ["Positive", "Neutral", "Negative"]:
            count = sentiment_counts.loc[topic].get(sentiment, 0)
            percent = (count / total) * 100 if total > 0 else 0
            md_lines.append(f"- **{sentiment}**: {count} posts ({percent:.1f}%)")
        md_lines.append("")

    with open(TOPIC_SENTIMENT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[✓] Markdown report saved to {TOPIC_SENTIMENT_MD}")


# ---------- Main Execution ----------

if __name__ == "__main__":
    merge_datasets()
    export_representative_posts()
    plot_sentiment_overlay()
