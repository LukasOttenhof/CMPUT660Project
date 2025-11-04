from github import Github, Auth
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class DeveloperAnalyzer:
    def __init__(self, token, repo_name, start_date, end_date, max_workers=5):
        self.token = token
        self.repo_name = repo_name
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers
        self.data = {}

        # Authenticate and get repo
        auth = Auth.Token(token)
        g = Github(auth=auth)
        self.repo = g.get_repo(repo_name)

    # ------------------ HELPER ------------------
    def ensure_user(self, user):
        """Ensure a user entry exists in the data dictionary."""
        if user not in self.data:
            self.data[user] = {
                # Commit-level
                "commits": 0,
                "loc_added": 0,
                "loc_deleted": 0,
                "files_changed": 0,

                # PR-level
                "pull_requests": 0,
                "merged_prs": 0,
                "reviews": 0,
                "pr_comments": 0,
                "avg_pr_additions": 0,
                "avg_pr_deletions": 0,
                "avg_pr_time_to_merge_days": 0.0,

                # Issue-level
                "issues": 0,
                "closed_issues": 0,
                "issue_comments": 0,
                "avg_issue_time_to_close_days": 0.0,
            }

    # ------------------ COMMITS ------------------
    def fetch_commits(self):
        print("Fetching commits...")
        commits = self.repo.get_commits(since=self.start_date, until=self.end_date)
        for commit in commits:
            if commit.author:
                user = commit.author.login
                self.ensure_user(user)
                self.data[user]["commits"] += 1

                if hasattr(commit, "stats") and commit.stats:
                    self.data[user]["loc_added"] += commit.stats.additions
                    self.data[user]["loc_deleted"] += commit.stats.deletions
                    self.data[user]["files_changed"] += commit.stats.total

    # ------------------ PULL REQUESTS ------------------
    def fetch_pull_requests_and_reviews(self):
        print("Fetching pull requests and reviews...")
        prs = self.repo.get_pulls(state="all", sort="created", direction="asc")

        def process_pr(pr):
            updates = []
            if pr.created_at <= self.end_date and pr.user:
                pr_user = pr.user.login
                self.ensure_user(pr_user)
                updates.append((pr_user, "pull_requests", 1))

                # PR metrics
                if hasattr(pr, "additions"):
                    updates.append((pr_user, "avg_pr_additions", pr.additions))
                if hasattr(pr, "deletions"):
                    updates.append((pr_user, "avg_pr_deletions", pr.deletions))
                if hasattr(pr, "comments"):
                    updates.append((pr_user, "pr_comments", pr.comments))
                if pr.merged:
                    updates.append((pr_user, "merged_prs", 1))
                    if pr.merged_at:
                        time_to_merge = (pr.merged_at - pr.created_at).days
                        updates.append((pr_user, "avg_pr_time_to_merge_days", time_to_merge))

                # Fetch reviews for this PR
                for review in pr.get_reviews():
                    if review.submitted_at and review.submitted_at <= self.end_date and review.user:
                        reviewer = review.user.login
                        self.ensure_user(reviewer)
                        updates.append((reviewer, "reviews", 1))
            return updates

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pr = {executor.submit(process_pr, pr): pr for pr in prs}
            for future in as_completed(future_to_pr):
                for user, field, count in future.result():
                    self.data[user][field] += count

    # ------------------ ISSUES ------------------
    def fetch_issues(self):
        print("Fetching issues...")
        issues = self.repo.get_issues(state="all", since=datetime(1970, 1, 1, tzinfo=timezone.utc))
        for issue in issues:
            if issue.created_at <= self.end_date and issue.user:
                user = issue.user.login
                self.ensure_user(user)
                self.data[user]["issues"] += 1

                if issue.state == "closed" and issue.closed_at:
                    self.data[user]["closed_issues"] += 1
                    time_to_close = (issue.closed_at - issue.created_at).days
                    self.data[user]["avg_issue_time_to_close_days"] += time_to_close

                if hasattr(issue, "comments"):
                    self.data[user]["issue_comments"] += issue.comments

    # ------------------ AGGREGATION ------------------
    def finalize_averages(self):
        """Convert cumulative metrics into averages."""
        for user, stats in self.data.items():
            if stats["pull_requests"] > 0:
                stats["avg_pr_additions"] /= stats["pull_requests"]
                stats["avg_pr_deletions"] /= stats["pull_requests"]
                stats["avg_pr_time_to_merge_days"] /= stats["merged_prs"] or 1
            if stats["closed_issues"] > 0:
                stats["avg_issue_time_to_close_days"] /= stats["closed_issues"]

    # ------------------ MAIN EXECUTION ------------------
    def run(self, save_path=None):
        print(f"\nCollecting data for {self.repo_name} from {self.start_date.date()} to {self.end_date.date()}...\n")

        self.fetch_commits()
        self.fetch_pull_requests_and_reviews()
        self.fetch_issues()
        self.finalize_averages()

        df = pd.DataFrame.from_dict(self.data, orient="index").reset_index()
        df.rename(columns={"index": "developer"}, inplace=True)
        df.sort_values(by="commits", ascending=False, inplace=True)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nSaved results to {save_path}")

        return df
if __name__ == "__main__":
    analyzer = DeveloperAnalyzer(
        token="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",
        repo_name="GeeeekExplorer/nano-vllm",
        start_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 9, 30, tzinfo=timezone.utc),
    )
    df = analyzer.run(save_path="Developer_Analysis/data/results.csv")