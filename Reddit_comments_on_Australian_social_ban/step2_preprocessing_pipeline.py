import pandas as pd
import re
import json
from datetime import datetime
from langdetect import detect, LangDetectException
import os

# === CONFIG ===
RAW_PATH = "Reddit/results/reddit_social_media_ban_posts.csv"
OUTPUT_DIR = "Reddit/results/preprocessing"
STAGE1_OUTPUT = f"{OUTPUT_DIR}/reddit_cleaned_stage1.csv"
STAGE2_OUTPUT = f"{OUTPUT_DIR}/reddit_cleaned_stage2.csv"
STAGE3_OUTPUT = f"{OUTPUT_DIR}/reddit_keywords_stage3.csv"
STATS_OUTPUT = f"{OUTPUT_DIR}/filter_stats.json"

# === FILTER THRESHOLDS (formerly 'relaxed') ===
score_threshold = 3
date_cutoff = pd.to_datetime("2023-10-01")
min_length = 20

# === STAGE 1: Initial Cleanup ===
df_raw = pd.read_csv(RAW_PATH)
print(f"[1] Raw rows: {len(df_raw)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = df_raw.sort_values(by="Score", ascending=False).drop_duplicates(subset="URL", keep="first")
print(f"[1] After deduplication: {len(df)}")

df = df[
    df.apply(
        lambda row: len(str(row.get("Title", "")) + str(row.get("Selftext", ""))) > min_length,
        axis=1,
    )
]
print(f"[1] After length filter: {len(df)}")
df.to_csv(STAGE1_OUTPUT, index=False)

# === STAGE 2: Filtering ===
filter_stats = {
    "initial": len(df),
    "placeholder_removed": 0,
    "date_filtered": 0,
    "score_filtered": 0,
    "length_filtered": 0,
    "lang_filtered": 0,
    "author_filtered": 0,
    "empty_body_filtered": 0,
    "profanity_flagged": 0,
}

# Remove placeholders
before = len(df)
df = df[~df["Title"].str.strip().str.lower().isin(["[deleted]", "[removed]", ""])]
filter_stats["placeholder_removed"] = before - len(df)

# Date filter
df["Created_Date"] = pd.to_datetime(df["Created_UTC"], errors="coerce")
before = len(df)
df = df[df["Created_Date"] >= date_cutoff]
filter_stats["date_filtered"] = before - len(df)

# Score filter
before = len(df)
df = df[df["Score"] >= score_threshold]
filter_stats["score_filtered"] = before - len(df)

# Length filter again
df["combined_text"] = df["Title"].fillna("") + " " + df["Selftext"].fillna("")
before = len(df)
df = df[df["combined_text"].str.len() >= min_length]
filter_stats["length_filtered"] = before - len(df)

# Language filter
def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

before = len(df)
df = df[df["combined_text"].apply(is_english)]
filter_stats["lang_filtered"] = before - len(df)

# Drop anonymous authors
before = len(df)
df = df[df["Author"].notna() & (df["Author"].str.lower() != "none")]
filter_stats["author_filtered"] = before - len(df)

# Drop empty bodies
before = len(df)
df = df[df["Selftext"].fillna("").str.strip() != ""]
filter_stats["empty_body_filtered"] = before - len(df)

# Profanity flagging
profanity_words = ["fuck", "shit", "bitch", "asshole", "dick", "bastard"]
df["Profanity_Flag"] = (
    df["combined_text"]
    .str.lower()
    .apply(lambda text: any(word in text for word in profanity_words))
)
filter_stats["profanity_flagged"] = df["Profanity_Flag"].sum()

df.drop(columns=["combined_text"], inplace=True)
df.to_csv(STAGE2_OUTPUT, index=False)
print(f"[2] Stage 2 saved to: {STAGE2_OUTPUT} ({len(df)} rows)")

with open(STATS_OUTPUT, "w") as f:
    json.dump({k: int(v) for k, v in filter_stats.items()}, f, indent=2)
print(f"[2] Filter stats saved to: {STATS_OUTPUT}")

# === STAGE 3: Keyword Filtering ===
ban_keywords = [
    "social media ban", "under 16", "Online Safety", "age verification", "Albanese",
    "let kids be kids", "Online Safety Commissioner", "digital ID", "age restriction",
    "kids off social media"
]
pattern = "|".join(ban_keywords)

keyword_filtered = df[
    df["Title"].str.contains(pattern, case=False, na=False)
    | df.get("Search_Term", "").astype(str).str.contains(pattern, case=False, na=False)
]

keyword_filtered.to_csv(STAGE3_OUTPUT, index=False)
print(f"[3] Stage 3 complete. Final keyword-filtered file saved: {STAGE3_OUTPUT} ({len(keyword_filtered)} posts)")
