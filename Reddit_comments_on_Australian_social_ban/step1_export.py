import praw
import pandas as pd
import time
import os
from datetime import datetime
from datetime import UTC
from collections import defaultdict

# ---------- Configuration ----------

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "DataScraper by u/samroof94")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "Reddit API credentials not set. Please define REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as environment variables."
    )

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)


SEARCH_TERMS = [
    # Legal/Policy Framing
    "Online Safety Amendment Act 2024",
    "social media age restriction Australia",
    "age verification social media Australia",
    "Online Safety Commissioner Australia",
    "Albanese social media ban",
    "Australia under 16 social media ban",
    "Australia digital ID law",
    # Public/Parenting Concerns
    "let kids be kids campaign",
    "parental controls on social media",
    "protecting children online",
    "social media harm to teens",
    "is TikTok dangerous for kids",
    "online safety for teenagers",
    # Youth/Teenage Perspective
    "kids off social media",
    "teenagers banned from Instagram",
    "do teens need social media",
    "social media addiction teens Australia",
    "young people and social media ban",
    "should kids be banned from social media",
]

SUBREDDITS = [
    "australia",
    "AustralianPolitics",
    "technology",
    "news",
    "Futurology",
    "Parenting",
    "AskParents",
    "teenagers",
    "YouthRights",
    "DigitalRights",
    "privacy",
    "Education",
    "SocialMedia",
    "MediaSkeptic",
    "AskAnAustralian",
]

output_folder = "Reddit/results"
os.makedirs(output_folder, exist_ok=True)

autosave_path = os.path.join(output_folder, "reddit_autosave_temp.csv")
output_filename = os.path.join(output_folder, "reddit_social_media_ban_posts.csv")
pairwise_counts_path = os.path.join(output_folder, "pairwise_counts.csv")

# ---------- Resume or Start Fresh ----------
if os.path.exists(autosave_path):
    print("Resuming from autosave...")
    df_existing = pd.read_csv(autosave_path)
    posts = df_existing.to_dict(orient="records")
else:
    print("Starting fresh scrape...")
    posts = []

# ---------- Tracking ----------
pairwise_counts = defaultdict(int)

# ---------- Scraping Loop ----------
for subreddit_name in SUBREDDITS:
    subreddit = reddit.subreddit(subreddit_name)
    for term in SEARCH_TERMS:
        print(f"Searching '{term}' in r/{subreddit_name}...")
        try:
            for post in subreddit.search(term, limit=100):
                try:
                    submission = reddit.submission(id=post.id)
                    submission.comments.replace_more(limit=0)
                    top_comments = [comment.body for comment in submission.comments[:5]]
                    combined_comments = "\n---\n".join(top_comments)
                except Exception as e:
                    print(f"Error fetching comments for post {post.id}: {e}")
                    combined_comments = ""

                formatted_time = datetime.fromtimestamp(post.created_utc, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                posts.append(
                    {
                        "Subreddit": subreddit_name,
                        "Search_Term": term,
                        "Title": post.title,
                        "Selftext": post.selftext,
                        "Score": post.score,
                        "Num_Comments": post.num_comments,
                        "Author": str(post.author),
                        "URL": post.url,
                        "Created_UTC": formatted_time,
                        "Top_Comments": combined_comments,
                    }
                )

                pairwise_counts[(subreddit_name, term)] += 1
                time.sleep(0.5)

            pd.DataFrame(posts).to_csv(autosave_path, index=False)
            print(f"Autosaved after: {term} in r/{subreddit_name}")

        except Exception as e:
            print(f"Error in r/{subreddit_name} for term '{term}': {e}")

        time.sleep(1)

# ---------- Save Final Results ----------
df = pd.DataFrame(posts)
print(f"\nTotal posts collected: {len(df)}")
df.to_csv(output_filename, index=False)
print(f"Final save successful: {output_filename}")

# ---------- Save Pairwise Counts ----------
counts_df = pd.DataFrame(
    [
        {"Subreddit": s, "Search_Term": t, "Count": c}
        for (s, t), c in pairwise_counts.items()
    ]
)
counts_df.to_csv(pairwise_counts_path, index=False)
print(f"Search-term/subreddit count saved: {pairwise_counts_path}")
