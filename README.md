# Australian Under-16 Social Media Ban: A Reddit-Based Sentiment and Topic Analysis

> **Team project by:**  
> **Moosa Anwar Khan (moosaanwarkhan@gmail.com) . Saad Abdullah (abdullah2@uni-potsdam.de) . Abdul Azeem Sikandar (sikandar@uni-potsdam.de)**

This repository contains the code, results, and figures for the research project:

**“Australian Under-16 Social Media Ban: A Reddit-Based Sentiment and Topic Analysis” (SS 2025)**  

We analyse how Australians respond to the **Online Safety Amendment (Social Media Minimum Age) Act 2024**, which raises the minimum legal age for social media accounts from 13 to 16. Using Reddit data, we combine **topic modeling** and **multi-level sentiment analysis** to understand which communities support or oppose the regulation and how reactions cluster around different themes.

The project was conducted as part of the course **“Social Media and Business Analytics – Research Project”** at the **University of Potsdam**.

---

##  Research Question & Objectives

**Research question**

> *How has the Australian public responded to the social media ban for under-16s, and what dominant sentiments and themes emerge in Reddit discussions around this policy?*

**Objectives**

1. Collect Reddit posts and comments discussing the Australian under-16 social media ban.  
2. Identify key themes using LDA topic modeling.  
3. Measure sentiment at post, comment, and full-thread levels using VADER.  
4. Overlay topics and sentiment to see which themes attract support or opposition.  
5. Compare communities (parent-oriented, youth-oriented, and civics/politics subreddits).

---

##  Key Results (Highlighted)

### 1. Overall Sentiment

- **Posts are more negative than positive.**  
  - Post-level sentiment is dominated by **negative views**, especially when users first react to the ban or headline news.
- **Comments are more balanced and slightly more positive.**  
  - Comment-level sentiment shows **more positive** and **more neutral** tones, suggesting that discussion moderates initial reactions.
- **Full-thread sentiment is relatively balanced.**  
  - When we combine post + comments, positive and negative sentiments are **roughly equal**, with a smaller neutral share.

 **Interpretation:** Initial reactions tend to be polarised, but discussion threads often bring in counter-arguments, leading to a more balanced overall tone.

---

### 2. Main Topics (LDA)

LDA topic modeling (5 topics) reveals the following dominant themes:

1. **Parenting & Youth Challenges**  
   - Everyday struggles of parents and teenagers, screen time, school performance, and mental health.
2. **Digital Identity & Youth Voice**  
   - Youth autonomy, fairness, enforceability, and whether teenagers' voices are being heard.
3. **Data Privacy & Regulation**  
   - Concerns about digital ID, data collection, surveillance, and platform obligations.
4. **Politics & Governance**  
   - Government trust, political parties, policy design, and legislative process.
5. **Influencers & Platforms**  
   - Role of TikTok, Meta, YouTube, and creators; advertising and platform incentives.

 **Interpretation:** Public debate is not just “for or against the ban” — it spreads across parenting, privacy, politics, and platform power.

---

### 3. Sentiment by Topic (Sentiment–Topic Overlay)

When we overlay sentiment on the topics:

- **Parenting & Youth Challenges**  
  - Has the **highest share of positive sentiment**.  
  - Many users support stricter rules when framed as *protecting children* and improving mental health.
- **Influencers & Platforms**  
  - Shows the **highest share of negative sentiment**.  
  - Users express frustration with platforms’ business models, targeted ads, and perceived hypocrisy.
- **Digital Identity & Youth Voice**, **Data Privacy & Regulation**, and **Politics & Governance**  
  - Show **mixed sentiment**, combining:
    - Positive views about safety and regulation, and  
    - Strong concerns about surveillance, digital ID, and exclusion of young people from decision-making.

 **Interpretation:** The ban is viewed most positively when framed as *parenting & safety* and most negatively when framed as *platform power & data practices*.

---

### 4. Community Differences (Subreddit-Level Patterns)

- **Parent & civics subreddits** (e.g. `r/Parenting`, `r/SocialMedia`, `r/AustralianPolitics`)  
  - More **supportive of the ban**, emphasising:
    - Protection from harmful content  
    - Reducing screen time  
    - Holding platforms accountable
- **Youth-focused communities** (e.g. `r/teenagers`, `r/YouthRights`)  
  - More **critical**, emphasising:
    - Youth autonomy and rights  
    - Practical enforceability issues  
    - Fears of overreach and surveillance
- **Tech and digital rights communities** (e.g. `r/DigitalRights`, `r/privacy`)  
  - Mixed but often **skeptical**, focusing on:
    - Data privacy  
    - Government overreach  
    - Technical and legal loopholes

 **Interpretation:** Different stakeholders talk about the same law in *very* different frames – safety vs rights vs surveillance vs practicality.

---

##  Data

### Source & Collection

- **Platform:** Reddit  
- **Tooling:** Python Reddit API Wrapper (**PRAW**)  
- **Time window:** 1 October 2023 – May 2025  
- **Unit of analysis:** Reddit posts (plus top-level comments) that discuss the under-16 social media ban.

Data collection is implemented in:

- `Reddit/step1_export.py`  

>  **Important:** This script requires your own Reddit API credentials.  
> Do **not** commit `client_id` or `client_secret` to GitHub. Use environment variables or a local config file in `.gitignore`.

### Search Queries

We used **19 keyword queries** grouped into three perspectives:

**Legal / Policy Framing**

- `Online Safety Amendment Act 2024`  
- `social media age restriction Australia`  
- `age verification social media Australia`  
- `Online Safety Commissioner Australia`  
- `Albanese social media ban`  
- `Australia under 16 social media ban`  
- `Australia digital ID law`  

**Public / Parenting Concerns**

- `let kids be kids campaign`  
- `parental controls on social media`  
- `protecting children online`  
- `social media harm to teens`  
- `is TikTok dangerous for kids`  
- `online safety for teenagers`  

**Youth / Teenage Perspective**

- `kids off social media`  
- `teenagers banned from Instagram`  
- `do teens need social media`  
- `social media addiction teens Australia`  
- `young people and social media ban`  
- `should kids be banned from social media`  

### Targeted Subreddits

We focused on **15 subreddits** to capture a wide range of views:

- `r/australia`  
- `r/AustralianPolitics`  
- `r/technology`  
- `r/news`  
- `r/SocialMedia`  
- `r/Futurology`  
- `r/Parenting`  
- `r/AskParents`  
- `r/teenagers`  
- `r/YouthRights`  
- `r/DigitalRights`  
- `r/privacy`  
- `r/Education`  
- `r/MediaSkeptic`  
- `r/AskAnAustralian`  

### Filtering & Final Sample

Our filtering pipeline:

1. Remove duplicates; require minimum combined length (title + body ≥ 20 characters).  
2. Filter by date (posts after 1 October 2023).  
3. Filter by Reddit score (score ≥ 3).  
4. Remove posts with near-empty bodies.  
5. Keep English posts only.  
6. Ensure posts contain at least one of the predefined search terms.

**Final dataset:**

- **597 posts** (each with top-level comments) about the under-16 social media ban.

Filtering outputs & stats:

- `Reddit/results/preprocessing/`  
- `Reddit/results/filtering/`

---

##  Methods

### Preprocessing

Implemented in:

- `Reddit/step2_preprocessing_pipeline.py`

Steps:

- Merge title + selftext into a single text field.  
- Lowercase, remove URLs and punctuation.  
- Tokenise, remove stopwords, lemmatise.  
- Build a dictionary + bag-of-words corpus.  
- Remove extremely rare and extremely frequent terms.

Outputs (in `Reddit/results/preprocessing/`):

- `reddit_cleaned_stage1.csv`  
- `reddit_cleaned_stage2.csv`  
- `reddit_keywords_stage3.csv`  
- `reddit_dataset_cleaned.csv`  
- `filter_stats.json`  

### Topic Modeling (LDA)

Implemented in:

- `Reddit/step6_lda_master_pipeline.py`

Details:

- Library: **Gensim**  
- Model: **Latent Dirichlet Allocation (LDA)**  
- Final model: **5 topics**, based on topic coherence and interpretability.

Outputs (in `Reddit/results/topic_modeling/`):

- `lda_model_.gensim*` – trained model files  
- `lda_preprocessed.pkl` – preprocessed corpus  
- `lda_topics.csv`, `lda_topics.md`, `lda_topics.txt` – topic keywords & labels  
- `lda_topic_distribution.png` – topic prevalence  
- `lda_pyldavis.html` – interactive visualization  
- `reddit_representative_quotes.csv` – exemplar posts per topic  

### Sentiment Analysis (VADER)

Implemented in:

- `Reddit/step5_sentiment_pipeline.py`

Details:

- Tool: **VADER**  
- Levels:
  - **Post-level** sentiment  
  - **Comment-level** sentiment (top 5 comments)  
  - **Full-thread** sentiment (combined)  

Labeling (VADER compound):

- `compound ≥ 0.05` → **Positive**  
- `compound ≤ -0.05` → **Negative**  
- otherwise → **Neutral**

Outputs (in `Reddit/results/sentiment_outputs/`):

- `reddit_with_sentiment.csv`  
- `sentiment_by_search_term.csv` (+ PNG)  
- `subreddit_post_vs_comment_sentiment.csv`  
- `subreddit_sentiment_averages.csv`  
- Plots:
  - `full_sentiment_dist.png`  
  - `post_sentiment_dist.png`  
  - `comment_sentiment_dist.png`  
  - `comment_sentiment_by_subreddit.png`  
  - `full_context_sentiment_by_subreddit.png`  
  - `comment_vs_post_delta.png`  
  - `full_vs_post_delta.png`  
  - `hist_full_vs_post_difference.png`  
  - `tone_difference_hist.png`  
  - `top_positive_subreddits.png`  
  - `top_negative_subreddits.png`  
- Summary:
  - `sentiment_summary.md`  

### Sentiment–Topic Overlay

Implemented in:

- `Reddit/step7_sentiment_topic_overlay.py`

Steps:

- Assign each post a dominant topic.  
- Combine topic labels with sentiment labels.  
- Aggregate sentiment proportions per topic.  
- Visualise sentiment–topic combinations.

Outputs (in `Reddit/results/sentiment_topic_overlay/`):

- `merged_sentiment_and_topics.csv`  
- `topic_sentiment_overlay.png`  
- `topic_sentiment_summary.md`  
- Topic description references: `lda_topics.md`, `lda_topics.txt`