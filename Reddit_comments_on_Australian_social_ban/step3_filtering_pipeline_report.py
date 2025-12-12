import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import graphviz
import numpy as np

# === Ensure output folder exists ===
os.makedirs("Reddit/results/filtering/", exist_ok=True)

# === File paths ===
RAW_PATH = "Reddit/results/reddit_social_media_ban_posts.csv"
STAGE1_PATH = "Reddit/results/preprocessing/reddit_cleaned_stage1.csv"
STAGE3_PATH = "Reddit/results/preprocessing/reddit_keywords_stage3.csv"
STATS_PATH = "Reddit/results/preprocessing/filter_stats.json"

# === Output files ===
OUTPUT_CSV = "Reddit/results/filtering/filtering_pipeline_summary.csv"
OUTPUT_MD = "Reddit/results/filtering/filtering_pipeline_summary.md"
OUTPUT_HTML = "Reddit/results/filtering/filtering_pipeline_summary.html"
OUTPUT_FLOWCHART = "Reddit/results/filtering/filtering_pipeline_flowchart.png"
BAR_CHART_FILE = "Reddit/results/filtering/filtering_pipeline_bar_chart.png"
LINE_CHART_FILE = "Reddit/results/filtering/filtering_pipeline_line_chart.png"

# === Load data ===
raw_df = pd.read_csv(RAW_PATH)
stage1_df = pd.read_csv(STAGE1_PATH)
stage3_df = pd.read_csv(STAGE3_PATH)

with open(STATS_PATH) as f:
    stats = json.load(f)

# === Stage labels ===
steps = [
    "Raw scraped data",
    "After Stage 1 (deduplication + length)",
    "After Placeholder Filter",
    "After Date Filter",
    "After Score Filter",
    "After Length Filter",
    "After Lang Filter",
    "After Author Filter",
    "After Empty Body Filter",
    "After Stage 2 filters (Final)",
    "Keyword Matched (Stage 3)",
]

# === Compute counts dynamically ===
initial = stats["initial"]
counts = [
    len(raw_df),
    len(stage1_df),
    initial - stats["placeholder_removed"],
    initial - stats["placeholder_removed"] - stats["date_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"] - stats["length_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"] - stats["length_filtered"] - stats["lang_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"] - stats["length_filtered"] - stats["lang_filtered"] - stats["author_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"] - stats["length_filtered"] - stats["lang_filtered"] - stats["author_filtered"] - stats["empty_body_filtered"],
    initial - stats["placeholder_removed"] - stats["date_filtered"] - stats["score_filtered"] - stats["length_filtered"] - stats["lang_filtered"] - stats["author_filtered"] - stats["empty_body_filtered"],
    len(stage3_df),
]

# === Create Table ===
df = pd.DataFrame(
    {
        "Stage": steps,
        "Posts Remaining": counts,
    }
)
df.to_csv(OUTPUT_CSV, index=False)
df.to_markdown(OUTPUT_MD, index=False)
df.to_html(OUTPUT_HTML, index=False)

print(f"\n✅ Saved table to:\n{OUTPUT_CSV}\n{OUTPUT_MD}\n{OUTPUT_HTML}")

# === Flowchart ===
g = graphviz.Digraph(format="png")
g.attr(rankdir="LR", size="8,5")
g.attr("node", shape="box", style="filled", fillcolor="lightblue")

for i, step in enumerate(steps):
    g.node(f"step{i}", step)
for i in range(len(steps) - 1):
    g.edge(f"step{i}", f"step{i + 1}")

g.render(filename=OUTPUT_FLOWCHART, cleanup=True)
print(f"\n✅ Flowchart saved to: {OUTPUT_FLOWCHART}")

# === Horizontal Bar Chart ===
stages = df["Stage"]
vals = df["Posts Remaining"].fillna(0).astype(int).tolist()

plt.figure(figsize=(12, 7))
bar_height = 0.4
y_positions = list(range(len(stages)))

plt.barh(
    y_positions,
    vals,
    height=bar_height,
    color="orange",  # Fixed color for general use
)

for i, val in enumerate(vals):
    if val > 0:
        plt.text(val + 100, i, str(val), va="center", fontsize=8)

plt.yticks(y_positions, stages)
plt.gca().invert_yaxis()
plt.xlabel("Number of Posts")
plt.title("Reddit Filtering Pipeline")
plt.tight_layout()
plt.savefig(BAR_CHART_FILE, dpi=300)
plt.show()

# === Line Chart ===
plt.figure(figsize=(14, 6))
plt.plot(
    stages,
    vals,
    marker="o",
    color="orange",
    label="Filtered Dataset",
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Posts Remaining")
plt.title("Reddit Filtering Pipeline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LINE_CHART_FILE, dpi=300)
plt.show()
print(f"\n✅ Bar chart saved to: {BAR_CHART_FILE}")
print(f"✅ Line chart saved to: {LINE_CHART_FILE}")
print("\n✅ All filtering pipeline visualizations completed.")