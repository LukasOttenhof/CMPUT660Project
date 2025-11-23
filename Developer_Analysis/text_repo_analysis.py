# from github import Github, Auth, GithubException
# from datetime import datetime, timezone
# import pandas as pd
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import urllib

# class DeveloperTextCollector:
#     def __init__(self, tokens, repo_name, start_date, end_date, max_workers=5):
#         """
#         tokens: list of GitHub access tokens
#         repo_name: str, e.g., "owner/repo"
#         """
#         self.tokens = tokens
#         self.repo_name = repo_name
#         self.start_date = start_date
#         self.end_date = end_date
#         self.max_workers = max_workers

#         self.current_token_index = 0
#         self.github = self._get_github_client()
#         self.repo = self._get_repo()

#         self.records = []  # (author, source, text, date)

#     # ------------------ TOKEN HANDLING ------------------
#     def _get_github_client(self):
#         """Return a GitHub client for the current token."""
#         token = self.tokens[self.current_token_index]
#         return Github(token)


#     def _rotate_token(self):
#         """Switch to the next token in the list."""
#         old_token = self.current_token_index
#         self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
#         print(f"⚠️ Switching token ({old_token} → {self.current_token_index})...")
#         self.github = self._get_github_client()
#         self.repo = self._get_repo()

#     def _get_repo(self):
#         """Fetch the repo safely with retries."""
#         for _ in range(len(self.tokens)):
#             try:
#                 return self.github.get_repo(self.repo_name)
#             except GithubException as e:
#                 if e.status in [401, 403]:
#                     print(f"Rate limit or bad credentials on token {self.current_token_index}. Rotating...")
#                     self._rotate_token()
#                     time.sleep(2)
#                 else:
#                     raise
#         print("⏳ All tokens exhausted. Sleeping for 1 minute...")
#         time.sleep(60)
#         return self._get_repo()

#     # ------------------ HELPERS ------------------
#     def add_record(self, author, source, text, date):
#         if author and text:
#             self.records.append({
#                 "author": author,
#                 "source": source,
#                 "text": text.strip(),
#                 "date": date
#             })

#     def _safe_call(self, func, *args, **kwargs):
#         """Wrap API calls to handle rate limits and rotate tokens."""
#         for _ in range(len(self.tokens)):
#             try:
#                 return func(*args, **kwargs)
#             except GithubException as e:
#                 if e.status in [401, 403]:
#                     print(f"API limit reached or forbidden (token {self.current_token_index}). Rotating...")
#                     self._rotate_token()
#                     time.sleep(2)
#                 else:
#                     raise
#         print("⏳ All tokens temporarily blocked. Waiting 60 seconds...")
#         time.sleep(60)
#         return self._safe_call(func, *args, **kwargs)
#     # ------------------ ISSUES ------------------
#     def fetch_issues_and_comments(self):
#         print("Fetching issues and issue comments...")
#         try:
#             issues = self._safe_call(self.repo.get_issues, state="all", since=self.start_date)
#         except Exception as e:
#             print(f"⚠️ Failed to fetch issues: {e}")
#             return

#         for issue in issues:
#             try:
#                 if issue.created_at <= self.end_date and issue.user:
#                     self.add_record(issue.user.login, "issue", issue.title + "\n" + (issue.body or ""), issue.created_at)

#                 try:
#                     comments = self._safe_call(issue.get_comments)
#                     for comment in comments:
#                         if comment.created_at <= self.end_date and comment.user:
#                             self.add_record(comment.user.login, "issue_comment", comment.body, comment.created_at)
#                 except GithubException as e:
#                     print(f"⚠️ Skipping comments for issue {issue.number} due to API error: {e}")
#                     self._rotate_token()
#             except Exception as e:
#                 print(f"⚠️ Skipping issue {issue.number} due to error: {e}")
#                 continue

#     # ------------------ PULL REQUESTS ------------------
#     def fetch_pull_requests_and_reviews(self):
#         print("Fetching pull requests, reviews, and PR comments...")
#         try:
#             prs = self._safe_call(self.repo.get_pulls, state="all", sort="created", direction="asc")
#         except Exception as e:
#             print(f"⚠️ Failed to fetch pull requests: {e}")
#             return

#         def process_pr(pr):
#             local_records = []
#             try:
#                 if pr.created_at <= self.end_date and pr.user:
#                     pr_text = pr.title + "\n" + (pr.body or "")
#                     local_records.append((pr.user.login, "pull_request", pr_text, pr.created_at))

#                     try:
#                         comments = self._safe_call(pr.get_issue_comments)
#                         for comment in comments:
#                             if comment.created_at <= self.end_date and comment.user:
#                                 local_records.append((comment.user.login, "pr_comment", comment.body, comment.created_at))
#                     except GithubException as e:
#                         print(f"⚠️ Skipping comments for PR {pr.number} due to API error: {e}")
#                         self._rotate_token()

#                     try:
#                         reviews = self._safe_call(pr.get_reviews)
#                         for review in reviews:
#                             if review.submitted_at and review.submitted_at <= self.end_date and review.user:
#                                 local_records.append((review.user.login, "pr_review", review.body, review.submitted_at))
#                     except GithubException as e:
#                         print(f"⚠️ Skipping reviews for PR {pr.number} due to API error: {e}")
#                         self._rotate_token()
#             except Exception as e:
#                 print(f"⚠️ Skipping PR {pr.number} due to error: {e}")
#             return local_records

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = {executor.submit(process_pr, pr): pr for pr in prs}
#             for future in as_completed(futures):
#                 for author, source, text, date in future.result():
#                     self.add_record(author, source, text, date)

#     # ------------------ DISCUSSIONS ------------------
#     def fetch_discussions(self):
#         print("Fetching discussions (if available)...")
#         try:
#             discussions = self._safe_call(self.repo.get_discussions)
#         except Exception as e:
#             print(f"⚠️ Discussions not available for this repository: {e}")
#             return

#         for discussion in discussions:
#             try:
#                 if discussion.created_at <= self.end_date and discussion.user:
#                     self.add_record(discussion.user.login, "discussion", discussion.title + "\n" + (discussion.body or ""), discussion.created_at)

#                 try:
#                     comments = self._safe_call(discussion.get_comments)
#                     for comment in comments:
#                         if comment.created_at <= self.end_date and comment.user:
#                             self.add_record(comment.user.login, "discussion_comment", comment.body, comment.created_at)
#                 except GithubException as e:
#                     print(f"⚠️ Skipping comments for discussion {discussion.id} due to API error: {e}")
#                     self._rotate_token()
#             except Exception as e:
#                 print(f"⚠️ Skipping discussion {discussion.id} due to error: {e}")

#     # ------------------ MAIN EXECUTION ------------------
#     def run(self, save_path=None):
#         save_path = save_path
#         print(f"Using token index {self.current_token_index}/{len(self.tokens)-1}")
#         print(f"\nCollecting text data for {self.repo_name} from {self.start_date.date()} to {self.end_date.date()}...\n")

#         self.fetch_issues_and_comments()
#         self.fetch_pull_requests_and_reviews()
#         self.fetch_discussions()

#         df = pd.DataFrame(self.records, columns=["author", "source", "text", "date"])
#         df.sort_values(by="date", inplace=True)

#         if save_path:
#             df.to_csv(save_path, index=False)
#             print(f"\n Saved text data to {save_path}")

#         return df

# if __name__ == "__main__":
#     tokens = [
#         ""
#     ]
#     print(1)
#     collector = DeveloperTextCollector(
#         tokens=tokens,
#         repo_name="Snailclimb/JavaGuide",
#         start_date=datetime(2024, 9, 28, tzinfo=timezone.utc),
#         end_date=datetime(2024, 9, 30, tzinfo=timezone.utc),
#     )
#     print("Starting text data collection...")
#     df = collector.run(save_path="snailclimb_text.csv")