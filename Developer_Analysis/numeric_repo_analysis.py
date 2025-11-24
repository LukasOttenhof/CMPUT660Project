from github import Github, Auth
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

class DeveloperAnalyzerTopContributors:
    def __init__(self, token, repo_name, start_date, end_date, top_n=5, max_workers=5):
        self.token = token
        self.repo_name = repo_name
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.max_workers = max_workers

        # separate lists for numeric and text records
        self.numeric_records = []
        self.text_records = []

        auth = Auth.Token(token)
        self.gh = Github(auth=auth, per_page=100)
        self.repo = self.gh.get_repo(repo_name)

    # ------------------ RECORD HELPERS ------------------
    def add_numeric_record(self, author, activity_type, date, **kwargs):
        if author and date:
            record = {"author": author, "activity_type": activity_type, "date": date}
            record.update(kwargs)
            self.numeric_records.append(record)

    def add_text_record(self, author, source, text, date):
        if author and text:
            self.text_records.append({
                "author": author,
                "source": source,
                "text": text.strip(),
                "date": date
            })

    # ------------------ TOP CONTRIBUTORS ------------------
    def get_top_contributors(self):
        print("Fetching contributors...")
        contributors = list(self.repo.get_contributors())
        top_contribs = sorted(contributors, key=lambda c: c.contributions, reverse=True)[:self.top_n]
        top_usernames = [c.login for c in top_contribs]
        print(f"Top {self.top_n} contributors: {top_usernames}")
        return top_usernames

    # ------------------ COMMITS ------------------
    def fetch_commits_for_user(self, username):
        commits = self.repo.get_commits(author=username, since=self.start_date, until=self.end_date)
        for commit in commits:
            if commit.author and commit.commit.author.date <= self.end_date:
                stats = getattr(commit, "stats", None)
                # numeric data
                self.add_numeric_record(
                    author=username,
                    activity_type="commit",
                    date=commit.commit.author.date,
                    loc_added=stats.additions if stats else 0,
                    loc_deleted=stats.deletions if stats else 0,
                    files_changed=stats.total if stats else 0,
                )
                # text data
                self.add_text_record(
                    author=username,
                    source="commit",
                    text=commit.commit.message,
                    date=commit.commit.author.date
                )

    # ------------------ PULL REQUESTS ------------------
    def fetch_prs_for_user(self, username):
        query = (
            f"repo:{self.repo.owner.login}/{self.repo.name} "
            f"type:pr author:{username} "
            f"created:{self.start_date.date()}..{self.end_date.date()}"
        )
        prs = self.gh.search_issues(query)
        for pr in prs:
            if hasattr(pr, "pull_request") and pr.pull_request:
                # numeric data
                self.add_numeric_record(
                    author=username,
                    activity_type="pr_created",
                    date=pr.created_at,
                    additions=getattr(pr, "additions", 0),
                    deletions=getattr(pr, "deletions", 0),
                )
                if pr.pull_request.merged:
                    merged_at = pr.pull_request.merged_at
                    if merged_at and merged_at <= self.end_date:
                        self.add_numeric_record(
                            author=username,
                            activity_type="pr_merged",
                            date=merged_at,
                            time_to_merge_days=(merged_at - pr.created_at).days
                        )
                # text data
                pr_text = pr.title + "\n" + (pr.body or "")
                self.add_text_record(username, "pull_request", pr_text, pr.created_at)

    # ------------------ ISSUES ------------------
    def fetch_issues_for_user(self, username):
        issues = self.repo.get_issues(creator=username, state="all", since=self.start_date)
        for issue in issues:
            if issue.created_at <= self.end_date:
                self.add_numeric_record(
                    author=username,
                    activity_type="issue_opened",
                    date=issue.created_at,
                )
            if issue.closed_at and issue.closed_at <= self.end_date:
                self.add_numeric_record(
                    author=username,
                    activity_type="issue_closed",
                    date=issue.closed_at,
                    time_to_close_days=(issue.closed_at - issue.created_at).days,
                )
            # text data
            self.add_text_record(username, "issue", issue.title + "\n" + (issue.body or ""), issue.created_at)

    # ------------------ MAIN EXECUTION ------------------
    def run(self, numeric_save=None, text_save=None):
        print(f"\nCollecting data for {self.repo_name} "
              f"from {self.start_date.date()} to {self.end_date.date()}...\n")

        top_users = self.get_top_contributors()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for username in top_users:
                futures.append(executor.submit(self.fetch_commits_for_user, username))
                futures.append(executor.submit(self.fetch_prs_for_user, username))
                futures.append(executor.submit(self.fetch_issues_for_user, username))

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Fetching user data"):
                pass

        # save numeric data
        df_numeric = pd.DataFrame(self.numeric_records)
        df_numeric.sort_values(by="date", inplace=True)
        if numeric_save:
            df_numeric.to_csv(numeric_save, index=False)
            print(f"Saved numeric data to {numeric_save}")

        # save text data
        df_text = pd.DataFrame(self.text_records)
        df_text.sort_values(by="date", inplace=True)
        if text_save:
            df_text.to_csv(text_save, index=False)
            print(f"Saved text data to {text_save}")

        return df_numeric, df_text

if __name__ == "__main__":
    repo_file = r"/Users/lukas./Desktop/CMPUT660Project/repos_min1000pr.txt" # path of list of repos
    with open(repo_file, "r", encoding="utf-8") as f:
        # Extract the full_name (first column before '|') from each line
        repo_list = [line.split("|")[0].strip() for line in f.readlines()]
    print("Got repo list")

    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 11, 1, tzinfo=timezone.utc)

    for repo_name in repo_list:
        print(f"\nProcessing repo: {repo_name}")

        # create folder for this repo
        repo_folder = os.path.join("Developer_Analysis", "data", repo_name.replace("/", "_"))
        os.makedirs(repo_folder, exist_ok=True)

        analyzer = DeveloperAnalyzerTopContributors(
            token="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", 
            repo_name=repo_name,
            start_date=start_date,
            end_date=end_date,
            top_n=5,
            max_workers=10
        )

        numeric_save = os.path.join(repo_folder, "results_top5_authors_numeric.csv")
        text_save = os.path.join(repo_folder, "results_top5_authors_text.csv")

        numeric_df, text_df = analyzer.run(
            numeric_save=numeric_save,
            text_save=text_save
        )
       # break