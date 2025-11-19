# from github import Github, Auth
# from datetime import datetime, timezone
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# class DeveloperAnalyzer:
#     def __init__(self, token, repo_name, start_date, end_date, max_workers=5):
#         self.token = token
#         self.repo_name = repo_name
#         self.start_date = start_date
#         self.end_date = end_date
#         self.max_workers = max_workers
#         self.records = []  # <-- Store individual activity entries

#         # Authenticate and get repo
#         auth = Auth.Token(token)
#         g = Github(auth=auth)
#         self.repo = g.get_repo(repo_name)

#     # ------------------ HELPERS ------------------
#     def add_record(self, author, activity_type, date, **kwargs):
#         """Add a record for a specific developer activity."""
#         if not author or not date:
#             return
#         record = {"author": author, "activity_type": activity_type, "date": date}
#         record.update(kwargs)
#         self.records.append(record)

#     # ------------------ COMMITS ------------------
#     def fetch_commits(self):
#         print("Fetching commits numeric analysis")
#         commits = self.repo.get_commits(since=self.start_date, until=self.end_date)
#         for commit in commits:
#             if commit.author and commit.commit.author.date <= self.end_date:
#                 stats = getattr(commit, "stats", None)
#                 additions = stats.additions if stats else 0
#                 deletions = stats.deletions if stats else 0
#                 files_changed = stats.total if stats else 0
#                 self.add_record(
#                     author=commit.author.login,
#                     activity_type="commit",
#                     date=commit.commit.author.date,
#                     loc_added=additions,
#                     loc_deleted=deletions,
#                     files_changed=files_changed,
#                 )

#     # ------------------ PULL REQUESTS ------------------
#     def fetch_pull_requests_and_reviews(self):
#         print("Fetching pull requests and reviews numeric analysis")
#         prs = self.repo.get_pulls(state="all", sort="created", direction="asc")

#         def process_pr(pr):
#             local_records = []
#             if pr.created_at <= self.end_date and pr.user:
#                 pr_user = pr.user.login
#                 local_records.append({
#                     "author": pr_user,
#                     "activity_type": "pr_created",
#                     "date": pr.created_at,
#                     "additions": getattr(pr, "additions", 0),
#                     "deletions": getattr(pr, "deletions", 0),
#                 })

#                 if pr.merged and pr.merged_at:
#                     time_to_merge = (pr.merged_at - pr.created_at).days
#                     local_records.append({
#                         "author": pr_user,
#                         "activity_type": "pr_merged",
#                         "date": pr.merged_at,
#                         "time_to_merge_days": time_to_merge,
#                     })

#                 for review in pr.get_reviews():
#                     if review.user and review.submitted_at <= self.end_date:
#                         local_records.append({
#                             "author": review.user.login,
#                             "activity_type": "review",
#                             "date": review.submitted_at,
#                         })
#             return local_records

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = {executor.submit(process_pr, pr): pr for pr in prs}
#             for future in as_completed(futures):
#                 for rec in future.result():
#                     self.add_record(**rec)

#     # ------------------ ISSUES ------------------
#     def fetch_issues(self):
#         print("Fetching issues numeric analysis")
#         issues = self.repo.get_issues(state="all", since=self.start_date)
#         for issue in issues:
#             if issue.user and issue.created_at <= self.end_date:
#                 self.add_record(
#                     author=issue.user.login,
#                     activity_type="issue_opened",
#                     date=issue.created_at,
#                 )
#             if issue.closed_at and issue.closed_at <= self.end_date:
#                 time_to_close = (issue.closed_at - issue.created_at).days
#                 self.add_record(
#                     author=issue.user.login,
#                     activity_type="issue_closed",
#                     date=issue.closed_at,
#                     time_to_close_days=time_to_close,
#                 )

#     # ------------------ MAIN EXECUTION ------------------
#     def run(self, save_path=None):
#             print(f"\nCollecting event-level data for {self.repo_name} "
#                 f"from {self.start_date.date()} to {self.end_date.date()}...\n")

#             self.fetch_commits()
#             self.fetch_pull_requests_and_reviews()
#             self.fetch_issues()

#             # Create DataFrame only if we have data
#             if not self.records:
#                 print("⚠️ No activity data found for this period.")
#                 return pd.DataFrame()

#             df = pd.DataFrame(self.records)

#             # Filter out any malformed rows missing 'date'
#             if "date" not in df.columns:
#                 print("⚠️ No valid 'date' field found in records.")
#                 return df

#             df = df.dropna(subset=["date"])
#             df.sort_values(by="date", inplace=True)

#             if save_path:
#                 df.to_csv(save_path, index=False)
#                 print(f"Saved numeric analysis to {save_path}")

#             return df


# if __name__ == "__main__":
#     analyzer = DeveloperAnalyzer(
#         token="gh_token" ,
#         repo_name="Snailclimb/JavaGuide",
#         start_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
#         end_date=datetime(2024, 9, 30, tzinfo=timezone.utc),
#     )
#     df = analyzer.run(save_path="Developer_Analysis/data/results.csv")

from github import Github, Auth
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DeveloperAnalyzer:
    def __init__(self, token, repo_name, start_date, end_date, max_workers=5):
        self.token = token
        self.repo_name = repo_name
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers
        self.records = []

        auth = Auth.Token(token)
        g = Github(auth=auth, per_page=100)
        self.repo = g.get_repo(repo_name)

    def add_record(self, author, activity_type, date, **kwargs):
        if not author or not date:
            return
        record = {"author": author, "activity_type": activity_type, "date": date}
        record.update(kwargs)
        self.records.append(record)

    # ------------------ COMMITS ------------------
    def fetch_commit_batch(self, commits):
        for commit in commits:
            if commit.author and commit.commit.author.date <= self.end_date:
                stats = getattr(commit, "stats", None)
                self.add_record(
                    author=commit.author.login,
                    activity_type="commit",
                    date=commit.commit.author.date,
                    loc_added=stats.additions if stats else 0,
                    loc_deleted=stats.deletions if stats else 0,
                    files_changed=stats.total if stats else 0,
                )

    from tqdm import tqdm

    def fetch_commits(self):
        print("Fetching commits...")
        commits = list(self.repo.get_commits(since=self.start_date, until=self.end_date))

        for commit in tqdm(commits, desc="Commits processed"):
            if commit.author and commit.commit.author.date <= self.end_date:
                stats = getattr(commit, "stats", None)
                self.add_record(
                    author=commit.author.login,
                    activity_type="commit",
                    date=commit.commit.author.date,
                    loc_added=stats.additions if stats else 0,
                    loc_deleted=stats.deletions if stats else 0,
                    files_changed=stats.total if stats else 0,
            )

    # ------------------ PULL REQUESTS ------------------
    def process_pr(self, pr):
        local_records = []
        if pr.created_at <= self.end_date and pr.user:
            pr_user = pr.user.login
            local_records.append({
                "author": pr_user,
                "activity_type": "pr_created",
                "date": pr.created_at,
                "additions": getattr(pr, "additions", 0),
                "deletions": getattr(pr, "deletions", 0),
            })
            if pr.merged and pr.merged_at:
                time_to_merge = (pr.merged_at - pr.created_at).days
                local_records.append({
                    "author": pr_user,
                    "activity_type": "pr_merged",
                    "date": pr.merged_at,
                    "time_to_merge_days": time_to_merge,
                })
        return local_records

    def fetch_pull_requests(self):
        print("Fetching pull requests...")
        prs = list(self.repo.get_pulls(state="all", sort="created", direction="asc"))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_pr, pr): pr for pr in prs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Pull Requests"):
                for rec in future.result():
                    self.add_record(**rec)

    # ------------------ ISSUES ------------------
    def process_issue(self, issue):
        local_records = []
        if issue.user and issue.created_at <= self.end_date:
            local_records.append({
                "author": issue.user.login,
                "activity_type": "issue_opened",
                "date": issue.created_at,
            })
        if issue.closed_at and issue.closed_at <= self.end_date:
            time_to_close = (issue.closed_at - issue.created_at).days
            local_records.append({
                "author": issue.user.login,
                "activity_type": "issue_closed",
                "date": issue.closed_at,
                "time_to_close_days": time_to_close,
            })
        return local_records

    def fetch_issues(self):
        print("Fetching issues...")
        issues = list(self.repo.get_issues(state="all", since=self.start_date))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_issue, issue): issue for issue in issues}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Issues"):
                for rec in future.result():
                    self.add_record(**rec)

    # ------------------ MAIN EXECUTION ------------------
    def run(self, save_path=None):
        print(f"\nCollecting data for {self.repo_name} "
              f"from {self.start_date.date()} to {self.end_date.date()}...\n")

        self.fetch_commits()
        self.fetch_pull_requests()
        self.fetch_issues()

        if not self.records:
            print("⚠️ No activity data found for this period.")
            return pd.DataFrame()

        df = pd.DataFrame(self.records)
        df = df.dropna(subset=["date"])
        df.sort_values(by="date", inplace=True)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved numeric analysis to {save_path}")

        return df


if __name__ == "__main__":
    analyzer = DeveloperAnalyzerFast(
        token="gh_token",
        repo_name="Snailclimb/JavaGuide",
        start_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 9, 30, tzinfo=timezone.utc),
        max_workers=10
    )
    df = analyzer.run(save_path="Developer_Analysis/data/results_fast_with_progress.csv")
