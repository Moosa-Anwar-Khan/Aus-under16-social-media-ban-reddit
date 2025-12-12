import os
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt

import pyLDAvis.gensim_models
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel

# === Download NLTK resources ===
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# === Configurable paths ===
BASE_FOLDER = "Reddit/results/topic_modeling"
INPUT_CSV = "Reddit/results/preprocessing/reddit_keywords_stage3.csv"
PREPROCESSED_PATH = os.path.join(BASE_FOLDER, "lda_preprocessed.pkl")
MODEL_PATH = os.path.join(BASE_FOLDER, "lda_model_.gensim")
TOPICS_TXT_PATH = os.path.join(BASE_FOLDER, "lda_topics.txt")
OUTPUT_CSV_PATH = os.path.join(BASE_FOLDER, "lda_topics.csv")
OUTPUT_IMG_PATH = os.path.join(BASE_FOLDER, "lda_topic_distribution.png")
# OUTPUT_HTML_PATH = os.path.join(BASE_FOLDER, "lda_pyldavis.html")

os.makedirs(BASE_FOLDER, exist_ok=True)


# === Step 1: Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english") and len(t) >= 3]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def run_preprocessing():
    print("[1] Preprocessing...")
    df = pd.read_csv(INPUT_CSV)
    df["Full_Text"] = df["Title"].fillna("") + " " + df["Selftext"].fillna("")
    df["Tokens"] = df["Full_Text"].apply(preprocess_text)
    df.to_pickle(PREPROCESSED_PATH)
    print(f"[✓] Preprocessing complete. Saved to {PREPROCESSED_PATH}")


# === Step 2: Model Training ===
def run_lda_training():
    print("[2] Training LDA model...")
    df = pd.read_pickle(PREPROCESSED_PATH)
    tokenized_docs = df["Tokens"].tolist()
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )
    lda_model.save(MODEL_PATH)
    with open(TOPICS_TXT_PATH, "w", encoding="utf-8") as f:
        for idx, topic in lda_model.print_topics(num_words=10):
            f.write(f"Topic {idx}: {topic}\n")
    print(f"[✓] Model and topics saved to {MODEL_PATH} and {TOPICS_TXT_PATH}")


# === Step 3: Assign Dominant Topics ===
def run_assign_topics():
    print("[3] Assigning dominant topics with probabilities...")
    df = pd.read_pickle(PREPROCESSED_PATH)
    tokenized_docs = df["Tokens"].tolist()
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    lda_model = LdaModel.load(MODEL_PATH)

    dominant_topics = []
    topic_probs = []

    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        if topics:
            dominant_topic, max_prob = max(topics, key=lambda x: x[1])
        else:
            dominant_topic, max_prob = None, None
        dominant_topics.append(dominant_topic)
        topic_probs.append(max_prob)

    df["Dominant_Topic"] = dominant_topics
    df["Topic_Probability"] = topic_probs

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[✓] Topics and probabilities saved to {OUTPUT_CSV_PATH}")


# === Step 4: Plot & Visualize ===
def run_plot_visualization():
    print("[4] Generating plots and HTML visualizations...")
    df = pd.read_csv(OUTPUT_CSV_PATH)

    # Count number of posts per topic
    topic_counts = df["Dominant_Topic"].value_counts().sort_index()

    # Final topic labels in order of Topic ID 0–4
    topic_labels = [
        "Data Privacy & Digital Safety",
        "Platform Features & Tech Industry News",
        "Youth Experiences & Impact on Kids",
        "Confusion & Uncertainty",
        "Political Reactions & Public Opinion",
    ]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(topic_counts)), topic_counts.values)
    plt.xticks(range(len(topic_counts)), topic_labels, rotation=45, ha="right")
    plt.ylabel("Number of Posts")
    plt.title("Topic Distribution")

    # Optional: Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"[✓] Bar chart saved to {OUTPUT_IMG_PATH}")


# === Step 5: Extract Representative Posts ===
def run_extract_representative_posts():
    print("[5] Extracting representative posts for each topic...")
    input_csv = OUTPUT_CSV_PATH  # 'Reddit/results/topic_modeling/lda_topics.csv'
    topic_column = "Dominant_Topic"
    prob_column = "Topic_Probability"
    output_per_topic = 2
    min_length = 200
    max_length = 300
    prob_threshold = 0.5
    output_path = os.path.join(BASE_FOLDER, "reddit_representative_quotes.csv")

    df = pd.read_csv(input_csv)
    df["Full_Text"] = df["Title"].fillna("") + " " + df["Selftext"].fillna("")
    df["Full_Text"] = df["Full_Text"].str.strip()
    df["Length"] = df["Full_Text"].str.len()

    # Filter by length and probability
    df = df[(df["Length"] >= min_length) & (df["Length"] <= max_length)]
    if prob_column in df.columns:
        df = df[df[prob_column] >= prob_threshold]
    else:
        print(
            f"[⚠] Column '{prob_column}' not found in data. Skipping probability filter."
        )

    # Select top N posts per topic
    sampled_posts = []
    for topic in sorted(df[topic_column].dropna().unique()):
        group = df[df[topic_column] == topic].copy()
        group["LengthDiff"] = (group["Length"] - 250).abs()
        top_posts = group.sort_values("LengthDiff").head(output_per_topic)
        for _, row in top_posts.iterrows():
            sampled_posts.append(
                {
                    "Topic": topic,
                    "Topic_Probability": row.get(prob_column, None),
                    "Reddit_Post": row["Full_Text"].strip(),
                }
            )

    out_df = pd.DataFrame(sampled_posts)
    out_df.to_csv(output_path, index=False)
    print(f"[✓] Representative posts saved to {output_path}")


# === Run All ===
if __name__ == "__main__":
    run_preprocessing()
    run_lda_training()
    run_assign_topics()
    run_plot_visualization()
    run_extract_representative_posts()
    print("[✔] All steps completed.")
