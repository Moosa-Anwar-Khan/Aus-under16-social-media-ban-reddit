import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os
import re

# === CONFIG ===
INPUT_FILE = "Reddit/results/preprocessing/reddit_keywords_stage3.csv"
OUTPUT_DIR = "Reddit/results/eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Helper Functions ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text


def top_words(df, col="Title", top_n=20):
    words = []
    for text in df[col].dropna():
        words += clean_text(text).split()
    return Counter(words).most_common(top_n)


# === Load Data ===
df = pd.read_csv(INPUT_FILE)

# === Top Subreddits ===
sub_counts = df["Subreddit"].value_counts()
plt.figure(figsize=(10, 6))
sub_counts.head(10).plot(kind="bar", color="orange")
plt.title("Top 10 Subreddits")
plt.ylabel("Post Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_subreddits.png")
plt.close()

# === Word Frequency (Title) ===
word_df = pd.DataFrame(top_words(df, "Title"), columns=["Word", "Count"])
plt.figure(figsize=(10, 6))
sns.barplot(data=word_df, x="Count", y="Word", color="orange")
plt.title("Top Words in Titles")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_words.png")
plt.close()

# Combined Score and Comment Distribution - Subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Score
sns.histplot(df["Score"].dropna(), bins=30, color="orange", kde=True, ax=axes[0])
axes[0].set_title("Score Distribution")
axes[0].set_xlabel("Score")

# Comments
sns.histplot(df["Num_Comments"].dropna(), bins=30, color="blue", kde=True, ax=axes[1])
axes[1].set_title("Comment Count Distribution")
axes[1].set_xlabel("Number of Comments")

fig.suptitle("Distributions of Post Score(Upvotes) and Comment Count", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUTPUT_DIR}/score_comments_combined.png")
plt.close()


# === Top Keywords per Subreddit ===
def top_keywords_by_subreddit(df, top_n=5):
    result = {}
    grouped = df.groupby("Subreddit")
    for subreddit, group in grouped:
        word_counts = Counter()
        for title in group["Title"].dropna():
            words = clean_text(title).split()
            word_counts.update(words)
        result[subreddit] = word_counts.most_common(top_n)
    return result


subreddit_keywords = top_keywords_by_subreddit(df)
with open(f"{OUTPUT_DIR}/top_keywords.txt", "w", encoding="utf-8") as f:
    for sub, words in subreddit_keywords.items():
        f.write(f"{sub}:\n")
        for word, count in words:
            f.write(f"  {word}: {count}\n")
        f.write("\n")

print(f"\n✅ EDA complete. Outputs saved to: {OUTPUT_DIR}")
