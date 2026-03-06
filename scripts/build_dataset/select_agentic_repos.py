from github import Github, Auth
import pandas as pd
import time
from tqdm import tqdm  

all_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
all_repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
all_user_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")

GITHUB_TOKEN = ""  
g = Github(auth=Auth.Token(GITHUB_TOKEN))


all_pr_df['created_at'] = pd.to_datetime(all_pr_df['created_at'])
agent_pr_df = all_pr_df[all_pr_df['agent'].notnull()]  # only agent PRs
pr_counts = agent_pr_df.groupby('repo_id').size().reset_index(name='agent_pr_count')
large_repos = pr_counts[pr_counts['agent_pr_count'] >= 50]
first_agent_pr_dates = agent_pr_df.groupby('repo_id')['created_at'].min().reset_index(name='first_agent_pr_date')
result = large_repos.merge(first_agent_pr_dates, on='repo_id')


result = result.merge(all_pr_df[['repo_id', 'repo_url']], on='repo_id').drop_duplicates(subset=['repo_id'])
result['full_name'] = result['repo_url'].apply(lambda x: '/'.join(x.rstrip('/').split('/')[-2:]))


creation_dates = []
for idx, row in tqdm(result.iterrows(), total=result.shape[0], desc="Fetching repo creation dates"):
    repo_name = row['full_name']  
    try:
        repo = g.get_repo(repo_name)
        creation_dates.append(repo.created_at)
        time.sleep(0.5)  
    except Exception as e:
        print(f"\nFailed to get repo {repo_name}: {e}")
        creation_dates.append(pd.NaT)

result['repo_creation_date'] = creation_dates

# Filter repos with >=3 years before first agent PR 
three_years = pd.Timedelta(days=3*365)
result = result[result['first_agent_pr_date'] - result['repo_creation_date'] >= three_years]


result = result.sort_values(by='agent_pr_count', ascending=False)

top = result  

with open("test.txt", "w", encoding="utf-8") as f:
    for _, row in top.iterrows():
        f.write(f"{row['full_name']} | {row['agent_pr_count']} agent PRs | "
                f"first agent PR: {row['first_agent_pr_date'].date()} | "
                f"repo creation: {row['repo_creation_date'].date()}\n")

print("Saved agent repos with >=50 PRs and >=3 years prior history to test.txt")