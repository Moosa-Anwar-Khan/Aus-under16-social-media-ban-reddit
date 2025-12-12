import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings

warnings.filterwarnings("ignore")  # hides matplotlib seaborn deprecation msgs

# === CONFIG ===
INPUT_FILE = "Reddit/results/preprocessing/reddit_keywords_stage3.csv"
OUTPUT_DIR = "Reddit/results/sentiment_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# === Load Data ===
df = pd.read_csv(INPUT_FILE)
df["Full_Text"] = df["Title"].fillna("") + " " + df["Selftext"].fillna("")

# === Sentiment on Post Content ===
def get_post_sentiment(row):
    text = str(row.get("Title", "")) + " " + str(row.get("Selftext", ""))
    return sia.polarity_scores(text)

post_sentiments = df.apply(get_post_sentiment, axis=1, result_type="expand")
df = pd.concat([df, post_sentiments.add_prefix("Post_")], axis=1)

# === Sentiment on Top Comment ===
def get_comment_sentiment(row):
    return sia.polarity_scores(str(row.get("Top_Comments", "")))

comment_sentiments = df.apply(get_comment_sentiment, axis=1, result_type="expand")
df = pd.concat([df, comment_sentiments.add_prefix("Comment_")], axis=1)

# === Full Context Sentiment (Post + Comment) ===
def get_full_context_sentiment(row):
    full_text = (
        str(row.get("Title", ""))
        + " "
        + str(row.get("Selftext", ""))
        + " "
        + str(row.get("Top_Comments", ""))
    )
    return sia.polarity_scores(full_text)

full_sentiments = df.apply(get_full_context_sentiment, axis=1, result_type="expand")
df = pd.concat([df, full_sentiments.add_prefix("Full_")], axis=1)

# === Sentiment Labels ===
def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Post_Label"] = df["Post_compound"].apply(label_sentiment)
df["Comment_Label"] = df["Comment_compound"].apply(label_sentiment)
df["Full_Label"] = df["Full_compound"].apply(label_sentiment)

# === Sentiment Delta ===
df["Comment_vs_Post"] = df["Comment_compound"] - df["Post_compound"]
df["Full_vs_Post"] = df["Full_compound"] - df["Post_compound"]

# === Save Extended Dataset ===
df.to_csv(f"{OUTPUT_DIR}/reddit_with_sentiment.csv", index=False)

# === Plot Distribution Helper ===
def plot_dist(column, title, filename, color="royalblue"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True, color=color)
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

# === Plot Histograms & KDEs ===
plot_dist("Post_compound", "Post Sentiment Distribution", "post_sentiment_dist.png")
plot_dist("Comment_compound", "Comment Sentiment Distribution", "comment_sentiment_dist.png")
plot_dist("Full_compound", "Full Context Sentiment", "full_sentiment_dist.png")
plot_dist("Comment_vs_Post", "Comment vs Post Sentiment Delta", "comment_vs_post_delta.png")
plot_dist("Full_vs_Post", "Full vs Post Sentiment Delta", "full_vs_post_delta.png")

# === Subreddit Sentiment Averages ===
subreddit_avg = (
    df.groupby("Subreddit")[["Post_compound", "Comment_compound", "Full_compound"]]
    .mean()
    .reset_index()
)
subreddit_avg.to_csv(f"{OUTPUT_DIR}/subreddit_sentiment_averages.csv", index=False)

# === Bar Charts: Top Positive & Negative Subreddits ===
def bar_chart_subreddits(data, col, title, fname, top=True):
    top_data = data.sort_values(col, ascending=not top).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=col, y="Subreddit", data=top_data, palette="Blues" if top else "Reds")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}", dpi=300)
    plt.close()

bar_chart_subreddits(subreddit_avg, "Post_compound", "Top Positive Subreddits", "top_positive_subreddits.png", top=True)
bar_chart_subreddits(subreddit_avg, "Post_compound", "Top Negative Subreddits", "top_negative_subreddits.png", top=False)

# === Scatter Plot: Post vs Comment Sentiment ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Post_compound"], y=df["Comment_compound"], alpha=0.5)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Post Sentiment")
plt.ylabel("Comment Sentiment")
plt.title("Post vs Comment Sentiment")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scatter_post_vs_comment.png")
plt.close()

# === Sentiment by Search Term ===
if "Search_Term" in df.columns:
    term_avg = (
        df.groupby("Search_Term")[["Post_compound", "Comment_compound"]]
        .mean()
        .reset_index()
    )
    term_avg.to_csv(f"{OUTPUT_DIR}/sentiment_by_search_term.csv", index=False)

    plt.figure(figsize=(12, 6))
    x = term_avg["Search_Term"]
    x_pos = range(len(x))
    plt.plot(x_pos, term_avg["Post_compound"], label="Post", marker="o", color="royalblue")
    plt.plot(x_pos, term_avg["Comment_compound"], label="Comment", marker="o", color="orange")
    plt.xticks(x_pos, x, rotation=45, ha="right")
    plt.title("Sentiment by Search Term")
    plt.ylabel("Average Compound Sentiment")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_by_search_term.png")
    plt.close()

# === Comment Sentiment Pie Chart ===
comment_counts = df["Comment_Label"].value_counts()
colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
plt.figure(figsize=(5, 5))
comment_counts.plot.pie(
    autopct="%1.1f%%",
    colors=[colors.get(label, "blue") for label in comment_counts.index],
)
plt.title("Comment Sentiment Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comment_sentiment_pie.png")
plt.close()

# === Avg Comment Sentiment per Subreddit (Horizontal Bar) ===
comment_avg = df.groupby("Subreddit")["Comment_compound"].mean().sort_values()
plt.figure(figsize=(12, 8))
comment_avg.plot(kind="barh", color="teal")
plt.title("Average Comment Sentiment per Subreddit")
plt.xlabel("Sentiment Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comment_sentiment_by_subreddit.png", dpi=300)
plt.close()

# === Avg Full Context Sentiment per Subreddit (Vertical Bar) ===
full_context_avg = df.groupby("Subreddit")["Full_compound"].mean().reset_index()
sorted_context = full_context_avg.sort_values("Full_compound")
plt.figure(figsize=(12, 8))
sns.barplot(x="Full_compound", y="Subreddit", data=sorted_context, palette="Purples_r")
plt.title("Average Full Context Sentiment per Subreddit")
plt.xlabel("Full Context Sentiment Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/full_context_sentiment_by_subreddit.png", dpi=300)
plt.close()

# === Sample Comments to CSV/Text ===
sample_rows = []
for label in ["Positive", "Negative", "Neutral"]:
    subset = df[df["Comment_Label"] == label]
    samples = subset[["Subreddit", "Top_Comments"]].dropna().head(5).copy()
    samples["Sentiment_Label"] = label
    sample_rows.append(samples)
if sample_rows:
    sample_df = pd.concat(sample_rows, ignore_index=True)
    sample_df.to_csv(os.path.join(OUTPUT_DIR, "sample_comments_all_sentiments.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "sample_comments.txt"), "w", encoding="utf-8") as f:
    for label in ["Positive", "Neutral", "Negative"]:
        subset = df[df["Comment_Label"] == label]
        samples = subset["Top_Comments"].dropna().sample(min(5, len(subset))).tolist()
        f.write(f"\n--- {label} Comments ---\n")
        for comment in samples:
            f.write(f"- {comment}\n")

# === Markdown Summary ===
with open(os.path.join(OUTPUT_DIR, "sentiment_summary.md"), "w", encoding="utf-8") as f:
    f.write(f"# Sentiment Analysis Summary\n\n")
    f.write(f"- Total posts analyzed: {len(df)}\n")
    for label in ["Post_Label", "Comment_Label", "Full_Label"]:
        f.write(f"\n## {label.replace('_', ' ')} Distribution\n")
        f.write(df[label].value_counts().to_markdown())
        f.write("\n")

    f.write("\n## 📂 Output CSV Files\n")
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".csv"):
            f.write(f"- `{file}`\n")

    f.write("\n## 🖼️ Output Charts\n")
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".png"):
            f.write(f"- ![]({file})\n")

    f.write("\n## 📄 Output Text Files\n")
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".txt"):
            f.write(f"- `{file}`\n")

# === Subreddit-Level Post vs Comment Sentiment Comparison ===
post_comment_comp = (
    df.groupby("Subreddit")[["Post_compound", "Comment_compound"]].mean().reset_index()
)
post_comment_comp.to_csv(os.path.join(OUTPUT_DIR, "subreddit_post_vs_comment_sentiment.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=post_comment_comp,
    x="Post_compound",
    y="Comment_compound",
    hue="Subreddit",
    legend=False,
    alpha=0.7,
)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Subreddit-Level: Post vs Comment Sentiment")
plt.xlabel("Avg Post Sentiment")
plt.ylabel("Avg Comment Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "subreddit_post_vs_comment_scatter.png"), dpi=300)
plt.close()

# === Tone Difference Histogram ===
plt.figure(figsize=(10, 6))
sns.histplot(df["Comment_vs_Post"], bins=30, kde=True, color="darkred")
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Comment - Post Sentiment")
plt.title("Audience vs Author Tone Difference")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tone_difference_hist.png"), dpi=300)
plt.close()

# === Full vs Post Difference Histogram ===
plt.figure(figsize=(10, 6))
sns.histplot(df["Full_vs_Post"], bins=30, kde=True, color="darkblue")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Difference Between Full Context and Post Sentiment")
plt.xlabel("Full Context - Post Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_full_vs_post_difference.png"), dpi=300)
plt.close()

print(f"\n✅ Sentiment pipeline complete. Results saved in: {OUTPUT_DIR}")