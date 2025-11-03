
from github import Github, Auth
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ CONFIG ------------------
GITHUB_TOKEN = "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" 
REPO_NAME = "Snailclimb/JavaGuide"
save_dir = "Developer_Analysis/data/developer_analysis_results.csv"
START_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)# SET FOR TESTING, DO since=datetime(1970, 1, 1) FOR FULL HISRTORY
END_DATE   = datetime(2024, 9, 30, tzinfo=timezone.utc)
MAX_WORKERS = 5  # Number of threads for parallel review fetching
# --------------------------------------------

# Authenticate with the new PyGithub method
auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
repo = g.get_repo(REPO_NAME)

data = {}

def ensure_user(user):
    """Ensure a user entry exists in the data dictionary."""
    if user not in data:
        data[user] = {
            "commits": 0,
            "loc_added": 0,
            "loc_deleted": 0,
            "pull_requests": 0,
            "reviews": 0,
            "issues": 0
        }

print(f"Collecting data for {REPO_NAME} up to {END_DATE.date()}...\n")

# ------------------ COMMITS ------------------
print("Fetching commits...")
commits = repo.get_commits(since=START_DATE , until=END_DATE) 

for commit in commits:
    if commit.author:
        user = commit.author.login
        ensure_user(user)
        data[user]["commits"] += 1
        # Only fetch stats if they exist to save API calls
        if hasattr(commit, "stats") and commit.stats:
            data[user]["loc_added"] += commit.stats.additions
            data[user]["loc_deleted"] += commit.stats.deletions

# ------------------ PULL REQUESTS & REVIEWS ------------------
print("Fetching pull requests and reviews...")
prs = repo.get_pulls(state="all", sort="created", direction="asc")

def process_pr(pr):
    """Process a single PR and its reviews."""
    updates = []
    if pr.created_at <= END_DATE and pr.user:
        pr_user = pr.user.login
        ensure_user(pr_user)
        updates.append((pr_user, "pull_requests", 1))

        # Fetch reviews for this PR
        for review in pr.get_reviews():
            if review.submitted_at and review.submitted_at <= END_DATE and review.user:
                reviewer = review.user.login
                ensure_user(reviewer)
                updates.append((reviewer, "reviews", 1))
    return updates

# Parallelize PR review fetching
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_pr = {executor.submit(process_pr, pr): pr for pr in prs}
    for future in as_completed(future_to_pr):
        for user, field, count in future.result():
            data[user][field] += count

# ------------------ ISSUES ------------------
print("Fetching issues...")
issues = repo.get_issues(state="all", since=datetime(1970, 1, 1))

for issue in issues:
    if issue.created_at <= END_DATE and issue.user:
        user = issue.user.login
        ensure_user(user)
        data[user]["issues"] += 1

# ------------------ SAVE RESULTS ------------------
df = pd.DataFrame.from_dict(data, orient="index").reset_index()
df.rename(columns={"index": "developer"}, inplace=True)
df.sort_values(by="commits", ascending=False, inplace=True)

print(df.head())
df.to_csv(save_dir, index=False)
print(f"\nSaved results to {save_dir}")
